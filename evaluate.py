# evaluate.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import json
from collections import defaultdict
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from config import Config
from dataset import JerseySequenceDataset
from model import JerseyTemporalNet


class Evaluator:
    def __init__(self, checkpoint_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.checkpoint_path = checkpoint_path
        self.load_checkpoint()
        self.setup_data()
        self.results = {'overall': {}, 'per_class': {}, 'predictions': [], 'confusion_matrices': {}}

    def load_checkpoint(self):
        print("\n" + "="*60)
        print("LOADING MODEL")
        print("="*60)

        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        
        if 'config' in checkpoint:
            cfg = checkpoint['config']
        else:
            # New format: vars(Config) saved directly
            cfg = {k: v for k, v in checkpoint.items() if k in [
                'num_digit_classes', 'hidden_dim', 'backbone', 'train_classes', 'test_classes'
            ]}

        # Reconstruct model
        self.model = JerseyTemporalNet(
            num_classes=cfg.get('num_digit_classes', Config.NUM_DIGIT_CLASSES),
            hidden_dim=cfg.get('hidden_dim', Config.HIDDEN_DIM),
            dropout=Config.DROPOUT,
            backbone=cfg.get('backbone', Config.BACKBONE),
            use_spatial_attention=Config.USE_SPATIAL_ATTENTION,
            use_temporal_attention=Config.USE_TEMPORAL_ATTENTION,
            bidirectional=Config.BIDIRECTIONAL_LSTM
        ).to(self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        print(f"Model loaded from: {self.checkpoint_path}")
        print(f"Train classes: {cfg.get('train_classes', Config.TRAIN_CLASSES)}")
        print(f"Best val acc: {checkpoint.get('best_val_acc', 'N/A'):.2f}%")
        print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"Parameters: {self.model.get_num_params():,}")

    def setup_data(self):
        print("\n" + "="*60)
        print("SETTING UP TEST DATA")
        print("="*60)

        self.test_transform = transforms.Compose([
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.test_dataset = JerseySequenceDataset(
            root_dir=Config.DATA_ROOT,
            transform=self.test_transform,
            seq_length=Config.SEQ_LENGTH,
            allowed_classes=Config.TEST_CLASSES,
            mode='test',
            sampling_strategy='uniform'
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=Config.PIN_MEMORY and self.device.type == 'cuda'
        )

        print(f"Test samples: {len(self.test_dataset)} | Batches: {len(self.test_loader)}")

    def evaluate(self):
        print("\n" + "="*60)
        print("RUNNING EVALUATION ON HELD-OUT CLASSES:", Config.TEST_CLASSES)
        print("="*60)

        self.model.eval()
        all_d1_true, all_d2_true = [], []
        all_d1_pred, all_d2_pred = [], []
        all_d1_conf, all_d2_conf = [], []

        class_total = defaultdict(int)
        class_correct = defaultdict(int)

        # Anti-empty-bias heuristic parameters
        EMPTY_CLASS = 10
        EMPTY_THRESHOLD = 0.55  # If "empty" has < 55% confidence, use next best
        
        total_modified = 0
        total_samples = 0

        with torch.no_grad():
            for seq, d1_true, d2_true in tqdm(self.test_loader, desc="Evaluating"):
                seq = seq.to(self.device)
                d1_true = d1_true.to(self.device)
                d2_true = d2_true.to(self.device)

                # Safe forward (handles 2 or 3 return values)
                out = self.model(seq)
                if len(out) == 3:
                    d1_logit, d2_logit, _ = out
                else:
                    d1_logit, d2_logit = out

                # D2 prediction 
                d2_pred = d2_logit.argmax(1)

                # D1 prediction with anti-empty-bias heuristic
                d1_probs_softmax = d1_logit.softmax(1)
                d1_pred_list = []
                batch_modified = 0

                for i in range(len(d1_probs_softmax)):
                    probs = d1_probs_softmax[i]
                    
                    if probs[EMPTY_CLASS] < EMPTY_THRESHOLD:
                        
                        masked_probs = probs.clone()
                        masked_probs[EMPTY_CLASS] = 0
                        prediction = masked_probs.argmax().item()
                        d1_pred_list.append(prediction)
                        
                        # Count if we changed from what argmax would give
                        if probs.argmax().item() == EMPTY_CLASS:
                            batch_modified += 1
                    else:
                        
                        d1_pred_list.append(EMPTY_CLASS)

                d1_pred = torch.tensor(d1_pred_list, device=d1_logit.device)
                
                total_modified += batch_modified
                total_samples += len(d1_pred_list)

                # Get confidences for the predictions we made
                d1_prob = d1_probs_softmax
                d2_prob = d2_logit.softmax(1)
                d1_conf = d1_prob.gather(1, d1_pred.unsqueeze(1)).squeeze(1)
                d2_conf = d2_prob.gather(1, d2_pred.unsqueeze(1)).squeeze(1)

                all_d1_true.extend(d1_true.cpu().numpy())
                all_d2_true.extend(d2_true.cpu().numpy())
                all_d1_pred.extend(d1_pred.cpu().numpy())
                all_d2_pred.extend(d2_pred.cpu().numpy())
                all_d1_conf.extend(d1_conf.cpu().numpy())
                all_d2_conf.extend(d2_conf.cpu().numpy())

                # Per-class stats — CONFIDENCE-AWARE COMPOSITION
                for i in range(len(d1_true)):
                    # True jersey number
                    true_cls = int(d2_true[i].item() if d1_true[i] == 10 else d1_true[i].item() * 10 + d2_true[i].item())
                    
                    # Predicted jersey number (adaptive rule)
                    if (d1_pred[i] != 10 and 
                            d1_conf[i] > Config.D1_CONF_THRESHOLD and 
                            d2_conf[i] > Config.D2_CONF_THRESHOLD):
                        pred_cls = int(d1_pred[i].item() * 10 + d2_pred[i].item())  # two-digit
                    else:
                        pred_cls = int(d2_pred[i].item())  # fallback to single-digit

                    class_total[true_cls] += 1
                    if pred_cls == true_cls:
                        class_correct[true_cls] += 1

        print(f"\n Anti-empty-bias heuristic modified {total_modified}/{total_samples} predictions " + 
              f"({100*total_modified/total_samples:.1f}%, threshold={EMPTY_THRESHOLD})")

        # Convert to numpy
        d1_true = np.array(all_d1_true)
        d2_true = np.array(all_d2_true)
        d1_pred = np.array(all_d1_pred)
        d2_pred = np.array(all_d2_pred)
        d1_conf = np.array(all_d1_conf)
        d2_conf = np.array(all_d2_conf)

        # Overall accuracy (confidence-aware)
        overall_correct = 0
        for i in range(len(d1_true)):
            true_num = int(d2_true[i] if d1_true[i] == 10 else d1_true[i] * 10 + d2_true[i])
            if (d1_pred[i] != 10 and 
                    d1_conf[i] > Config.D1_CONF_THRESHOLD and 
                    d2_conf[i] > Config.D2_CONF_THRESHOLD):
                pred_num = int(d1_pred[i] * 10 + d2_pred[i])
            else:
                pred_num = int(d2_pred[i])
            if pred_num == true_num:
                overall_correct += 1

        total = len(d1_true)
        acc = 100.0 * overall_correct / total
        d1_acc = 100.0 * (d1_true == d1_pred).mean()
        d2_acc = 100.0 * (d2_true == d2_pred).mean()
        avg_conf = np.mean((d1_conf + d2_conf) / 2)

        self.results['overall'] = {
            'accuracy': float(acc),
            'd1_accuracy': float(d1_acc),
            'd2_accuracy': float(d2_acc),
            'avg_confidence': float(avg_conf),
            'total_samples': int(total),
            'correct_samples': int(overall_correct),
            'empty_threshold': float(EMPTY_THRESHOLD),
            'predictions_modified': int(total_modified)
        }

        # Per-class
        for cls in sorted(class_total.keys()):
            self.results['per_class'][int(cls)] = {
                'total': class_total[cls],
                'correct': class_correct[cls],
                'accuracy': 100.0 * class_correct[cls] / class_total[cls]
            }

        # Confusion matrices (digit-wise only — reconstruction is post-hoc)
        self.results['confusion_matrices'] = {
            'd1': confusion_matrix(d1_true, d1_pred).tolist(),
            'd2': confusion_matrix(d2_true, d2_pred).tolist()
        }

        self.print_results()
        self.save_results()
        self.generate_visualizations(d1_true, d2_true, d1_pred, d2_pred)

    def print_results(self):
        print("\n" + "="*60)
        print("GENERALIZATION TEST RESULTS (with Anti-Empty-Bias Heuristic)")
        print("="*60)
        o = self.results['overall']
        print(f"Overall Accuracy: {o['accuracy']:.2f}%")
        print(f"Tens Digit Acc:   {o['d1_accuracy']:.2f}%")
        print(f"Units Digit Acc:  {o['d2_accuracy']:.2f}%")
        print(f"Avg Confidence:   {o['avg_confidence']:.4f}")
        print(f"Total Samples:    {o['total_samples']}")
        print(f"Predictions Modified: {o['predictions_modified']}/{o['total_samples']}")

        print("\nPer-Class Accuracy:")
        for cls, stats in sorted(self.results['per_class'].items()):
            print(f"  Class {cls:2d} → {stats['accuracy']:6.2f}% ({stats['correct']}/{stats['total']})")

        if self.results['per_class']:
            best = max(self.results['per_class'].items(), key=lambda x: x[1]['accuracy'])
            worst = min(self.results['per_class'].items(), key=lambda x: x[1]['accuracy'])
            print(f"\nBest:  Class {best[0]} → {best[1]['accuracy']:.2f}%")
            print(f"Worst: Class {worst[0]} → {worst[1]['accuracy']:.2f}%")

    def save_results(self):
        path = os.path.join(Config.RESULTS_DIR, 'evaluation_results.json')
        result_data = {
            'metadata': {
                'checkpoint': self.checkpoint_path,
                'test_classes': Config.TEST_CLASSES,
                'timestamp': datetime.now().isoformat(),
                'empty_threshold': self.results['overall']['empty_threshold'],
                'reconstruction_rule': f"d1_conf > {Config.D1_CONF_THRESHOLD} and d2_conf > {Config.D2_CONF_THRESHOLD}"
            },
            'results': self.results
        }
        with open(path, 'w') as f:
            json.dump(result_data, f, indent=4)
        print(f"\nResults saved to {path}")

    def generate_visualizations(self, d1t, d2t, d1p, d2p):
        print("\nGenerating plots...")
        os.makedirs(os.path.join(Config.RESULTS_DIR, 'confusion_matrices'), exist_ok=True)

        # Confusion matrices
        fig, ax = plt.subplots(1, 2, figsize=(16, 6))
        sns.heatmap(confusion_matrix(d1t, d1p), annot=True, fmt='d', cmap='Blues', ax=ax[0])
        ax[0].set_title('Tens Digit Confusion')
        sns.heatmap(confusion_matrix(d2t, d2p), annot=True, fmt='d', cmap='Greens', ax=ax[1])
        ax[1].set_title('Units Digit Confusion')
        plt.tight_layout()
        plt.savefig(os.path.join(Config.RESULTS_DIR, 'confusion_matrices', 'confusion_matrices.png'), dpi=300)
        plt.close()

        # Per-class bar chart
        if self.results['per_class']:
            classes = sorted(self.results['per_class'].keys())
            accs = [self.results['per_class'][c]['accuracy'] for c in classes]
            plt.figure(figsize=(10, 6))
            bars = plt.bar(classes, accs, color=['green' if a >= 80 else 'orange' if a >= 60 else 'red' for a in accs])
            plt.axhline(y=self.results['overall']['accuracy'], color='red', linestyle='--',
                        label=f"Overall: {self.results['overall']['accuracy']:.2f}%")
            plt.title('Per-Class Accuracy on Test Set (with Anti-Empty-Bias Heuristic)')
            plt.xlabel('Jersey Number')
            plt.ylabel('Accuracy (%)')
            plt.legend()
            plt.grid(axis='y', alpha=0.3)
            plt.savefig(os.path.join(Config.RESULTS_DIR, 'per_class_accuracy.png'), dpi=300, bbox_inches='tight')
            plt.close()

        print("All plots saved!")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth')
    args = parser.parse_args()

    evaluator = Evaluator(args.checkpoint)
    evaluator.evaluate()

    print("\n" + "="*60)
    print("EVALUATION COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()