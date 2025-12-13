# train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import json
import random
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from config import Config
from dataset import JerseySequenceDataset
from model import JerseyTemporalNet


# ======================= MIXUP =======================
def mixup_data(x, d1_target, d2_target, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, d1_target, d1_target[index], d2_target, d2_target[index], lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
# =================================================================


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.verbose = verbose

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"Early stopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.early_stop


class Trainer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        Config.create_dirs()
        self.use_amp = Config.USE_MIXED_PRECISION and self.device.type == 'cuda'
        self.scaler = GradScaler() if self.use_amp else None

        # Initialize TensorBoard
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.writer = SummaryWriter(os.path.join(Config.LOGS_DIR, f'run_{timestamp}'))
        print(f" TensorBoard logging to: {Config.LOGS_DIR}/run_{timestamp}")

        self.setup_data()
        self.setup_model()

        self.early_stopping = EarlyStopping(patience=Config.EARLY_STOPPING_PATIENCE)
        self.history = {k: [] for k in ['train_loss', 'val_loss', 'train_acc', 'val_acc',
                                        'val_d1_acc', 'val_d2_acc', 'val_jersey_acc',
                                        'epoch', 'learning_rate']}
        self.best_val_acc = 0.0

    def setup_data(self):
        print("\n" + "="*60 + "\nSETTING UP DATA\n" + "="*60)

        self.train_transform = transforms.Compose([
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            transforms.RandomAffine(degrees=Config.AUG_ROTATION_DEGREES,
                                    translate=Config.AUG_TRANSLATE, scale=Config.AUG_SCALE),
            transforms.ColorJitter(brightness=Config.AUG_BRIGHTNESS, contrast=Config.AUG_CONTRAST),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.val_transform = transforms.Compose([
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        temp_ds = JerseySequenceDataset(Config.DATA_ROOT, transform=None,
                                        seq_length=Config.SEQ_LENGTH,
                                        allowed_classes=Config.TRAIN_CLASSES, mode='full')
        indices = np.random.RandomState(Config.SEED).permutation(len(temp_ds))
        split = int(Config.VAL_SPLIT * len(indices))
        val_idx, train_idx = indices[:split], indices[split:]

        train_ds = Subset(JerseySequenceDataset(Config.DATA_ROOT, self.train_transform,
                                                seq_length=Config.SEQ_LENGTH,
                                                allowed_classes=Config.TRAIN_CLASSES,
                                                mode='train', sampling_strategy=Config.TEMPORAL_SAMPLING),
                          train_idx)
        val_ds = Subset(JerseySequenceDataset(Config.DATA_ROOT, self.val_transform,
                                              seq_length=Config.SEQ_LENGTH,
                                              allowed_classes=Config.TRAIN_CLASSES,
                                              mode='val', sampling_strategy='uniform'),
                        val_idx)

        full_train_ds = JerseySequenceDataset(Config.DATA_ROOT, self.train_transform,
                                              seq_length=Config.SEQ_LENGTH,
                                              allowed_classes=Config.TRAIN_CLASSES, mode='train')
        self.w_d1, self.w_d2 = full_train_ds.get_class_weights(method='effective_number')

        self.train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True,
                                       num_workers=Config.NUM_WORKERS, pin_memory=True,
                                       persistent_workers=Config.NUM_WORKERS > 0)
        self.val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False,
                                     num_workers=Config.NUM_WORKERS, pin_memory=True,
                                     persistent_workers=Config.NUM_WORKERS > 0)

        print(f"Train batches: {len(self.train_loader)} | Val batches: {len(self.val_loader)}")

    def setup_model(self):
        print("\n" + "="*60 + "\nSETTING UP MODEL\n" + "="*60)
        self.model = JerseyTemporalNet(
            num_classes=Config.NUM_DIGIT_CLASSES,
            hidden_dim=Config.HIDDEN_DIM,
            dropout=Config.DROPOUT,
            backbone=Config.BACKBONE,
            use_spatial_attention=Config.USE_SPATIAL_ATTENTION,
            use_temporal_attention=Config.USE_TEMPORAL_ATTENTION,
            bidirectional=Config.BIDIRECTIONAL_LSTM
        ).to(self.device)

        self.criterion_d1 = nn.CrossEntropyLoss(weight=self.w_d1.to(self.device), label_smoothing=0.1)
        self.criterion_d2 = nn.CrossEntropyLoss(weight=self.w_d2.to(self.device), label_smoothing=0.1)

        self.optimizer = optim.AdamW(self.model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=Config.NUM_EPOCHS)

        # Log model graph to TensorBoard
        dummy_input = torch.randn(1, Config.SEQ_LENGTH, 3, Config.IMG_SIZE, Config.IMG_SIZE).to(self.device)
        try:
            self.writer.add_graph(self.model, dummy_input)
            print(" Model graph saved to TensorBoard")
        except Exception as e:
            print(f" Could not save model graph: {e}")

        print("Model ready — MixUp + effective_number weights with suppressed empty class")

    def forward_with_output(self, x):
        output = self.model(x)
        if len(output) == 3:
            d1_logit, d2_logit, _ = output
        else:
            d1_logit, d2_logit = output
        return d1_logit, d2_logit

    def calculate_metrics(self, d1p, d2p, d1t, d2t):
        both = ((d1p == d1t) & (d2p == d2t)).float().mean().item() * 100
        d1_acc = (d1p == d1t).float().mean().item() * 100
        d2_acc = (d2p == d2t).float().mean().item() * 100
        jersey_pred = torch.where(d1p == 10, d2p, d1p * 10 + d2p)
        jersey_true = torch.where(d1t == 10, d2t, d1t * 10 + d2t)
        jersey_acc = (jersey_pred == jersey_true).float().mean().item() * 100
        return {'both_acc': both, 'd1_acc': d1_acc, 'd2_acc': d2_acc, 'jersey_acc': jersey_acc}

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        all_d1_pred, all_d2_pred, all_d1_true, all_d2_true = [], [], [], []

        for seq, d1_true, d2_true in tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]"):
            seq = seq.to(self.device)
            d1_true = d1_true.to(self.device)
            d2_true = d2_true.to(self.device)

            self.optimizer.zero_grad()

            if random.random() < 0.5:  # MixUp
                seq_mix, d1a, d1b, d2a, d2b, lam = mixup_data(seq, d1_true, d2_true, alpha=1.0)
                with autocast(enabled=self.use_amp):
                    d1_logit, d2_logit = self.forward_with_output(seq_mix)
                    loss = (mixup_criterion(self.criterion_d1, d1_logit, d1a, d1b, lam) +
                            mixup_criterion(self.criterion_d2, d2_logit, d2a, d2b, lam))
            else:
                with autocast(enabled=self.use_amp):
                    d1_logit, d2_logit = self.forward_with_output(seq)
                    loss = self.criterion_d1(d1_logit, d1_true) + self.criterion_d2(d2_logit, d2_true)

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), Config.GRADIENT_CLIP_VALUE)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), Config.GRADIENT_CLIP_VALUE)
                self.optimizer.step()

            total_loss += loss.item()
            all_d1_pred.append(d1_logit.argmax(1).cpu())
            all_d2_pred.append(d2_logit.argmax(1).cpu())
            all_d1_true.append(d1_true.cpu())
            all_d2_true.append(d2_true.cpu())

        metrics = self.calculate_metrics(
            torch.cat(all_d1_pred), torch.cat(all_d2_pred),
            torch.cat(all_d1_true), torch.cat(all_d2_true)
        )
        
        avg_loss = total_loss / len(self.train_loader)
        
        # Log to TensorBoard
        self.writer.add_scalar('Loss/train', avg_loss, epoch)
        self.writer.add_scalar('Accuracy/train_both', metrics['both_acc'], epoch)
        self.writer.add_scalar('Accuracy/train_d1', metrics['d1_acc'], epoch)
        self.writer.add_scalar('Accuracy/train_d2', metrics['d2_acc'], epoch)
        self.writer.add_scalar('Accuracy/train_jersey', metrics['jersey_acc'], epoch)
        
        return avg_loss, metrics

    def validate(self, epoch):
        self.model.eval()
        total_loss = 0.0
        all_d1_pred, all_d2_pred, all_d1_true, all_d2_true = [], [], [], []

        with torch.no_grad():
            for seq, d1_true, d2_true in tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]  "):
                seq, d1_true, d2_true = seq.to(self.device), d1_true.to(self.device), d2_true.to(self.device)
                with autocast(enabled=self.use_amp):
                    d1_logit, d2_logit = self.forward_with_output(seq)
                    loss = self.criterion_d1(d1_logit, d1_true) + self.criterion_d2(d2_logit, d2_true)

                total_loss += loss.item()
                all_d1_pred.append(d1_logit.argmax(1).cpu())
                all_d2_pred.append(d2_logit.argmax(1).cpu())
                all_d1_true.append(d1_true.cpu())
                all_d2_true.append(d2_true.cpu())

        metrics = self.calculate_metrics(
            torch.cat(all_d1_pred), torch.cat(all_d2_pred),
            torch.cat(all_d1_true), torch.cat(all_d2_true)
        )
        
        avg_loss = total_loss / len(self.val_loader)
        
        # Log to TensorBoard
        self.writer.add_scalar('Loss/val', avg_loss, epoch)
        self.writer.add_scalar('Accuracy/val_both', metrics['both_acc'], epoch)
        self.writer.add_scalar('Accuracy/val_d1', metrics['d1_acc'], epoch)
        self.writer.add_scalar('Accuracy/val_d2', metrics['d2_acc'], epoch)
        self.writer.add_scalar('Accuracy/val_jersey', metrics['jersey_acc'], epoch)
        
        return avg_loss, metrics

    def train(self):
        print("\n" + "="*60)
        print("TRAINING WITH MIXUP")
        print("="*60)

        for epoch in range(1, Config.NUM_EPOCHS + 1):
            train_loss, train_metrics = self.train_epoch(epoch)
            val_loss, val_metrics = self.validate(epoch)
            self.scheduler.step()

            self.history['epoch'].append(epoch)
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_metrics['both_acc'])
            self.history['val_acc'].append(val_metrics['both_acc'])
            self.history['val_d1_acc'].append(val_metrics['d1_acc'])
            self.history['val_d2_acc'].append(val_metrics['d2_acc'])
            self.history['val_jersey_acc'].append(val_metrics['jersey_acc'])
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])

            # Log learning rate
            self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Log train vs val comparison
            self.writer.add_scalars('Loss/train_vs_val', {
                'train': train_loss,
                'val': val_loss
            }, epoch)
            
            self.writer.add_scalars('Accuracy/train_vs_val', {
                'train': train_metrics['both_acc'],
                'val': val_metrics['both_acc']
            }, epoch)

            print(f"\nEpoch {epoch:2d} | Val Acc: {val_metrics['both_acc']:5.2f}% | "
                  f"D1: {val_metrics['d1_acc']:5.2f}% | D2: {val_metrics['d2_acc']:5.2f}%")

            if val_metrics['both_acc'] > self.best_val_acc:
                self.best_val_acc = val_metrics['both_acc']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_val_acc': self.best_val_acc,
                    'history': self.history
                }, os.path.join(Config.CHECKPOINT_DIR, 'best_model.pth'))
                print(f"NEW BEST MODEL SAVED: {self.best_val_acc:.2f}%")

            if self.early_stopping(val_loss):
                print("Early stopping triggered")
                break

        # Close TensorBoard writer
        self.writer.close()
        
        print(f"\nTRAINING COMPLETE! Best Val Acc: {self.best_val_acc:.2f}%")
        print(f" View TensorBoard: tensorboard --logdir={Config.LOGS_DIR}")
        print("Now run: python evaluate.py --checkpoint checkpoints/best_model.pth")


if __name__ == "__main__":
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)
    random.seed(Config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(Config.SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    Trainer().train()