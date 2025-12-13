# dataset.py

"""
Custom Dataset Loader for Jersey Number Recognition
Handles hierarchical directory structure: Root/{Class}/{Jersey_Seq}/{Subdir}/*.jpg
"""

import os
import glob
import random
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageEnhance
import numpy as np
from config import Config

class JerseySequenceDataset(Dataset):
    def __init__(self, root_dir, transform=None, seq_length=5, 
                 allowed_classes=None, mode='train', sampling_strategy='uniform'):
        """
        Args:
            root_dir (str): Root directory of dataset
            transform: torchvision transforms to apply
            seq_length (int): Number of frames to sample per sequence
            allowed_classes (list): List of class numbers to include (e.g., [4, 6, 46])
                                   If None, uses all available classes
            mode (str): 'train', 'val', or 'test' - for logging purposes
            sampling_strategy (str): 'uniform', 'random', or 'center'
        """
        self.root_dir = root_dir
        self.transform = transform
        self.seq_length = seq_length
        self.allowed_classes = allowed_classes
        self.mode = mode
        self.sampling_strategy = sampling_strategy
        self.samples = []  # List of (sequence_path, label_d1, label_d2, class_name)
        
        self._crawl_dataset()
    
    def _parse_class_label(self, class_name):
        """
        Parse class folder name to digit labels.
        Handles: '4', '04', '48', '08', etc.
        
        Args:
            class_name: Folder name (e.g., '4', '48', '04')
        
        Returns:
            (label_d1, label_d2): Tuple of digit labels
                - label_d1: Tens digit (0-9) or 10 for empty
                - label_d2: Units digit (0-9)
        """
        try:
            class_num = int(class_name)
            
            if class_num < 0 or class_num >= 100:
                raise ValueError(f"Class number out of range [0, 99]: {class_num}")
            
            if class_num < 10:
                # Single digit: 0-9
                label_d1 = 10  # Empty tens place
                label_d2 = class_num
            else:
                # Two digits: 10-99
                label_d1 = class_num // 10
                label_d2 = class_num % 10
            
            return label_d1, label_d2
        
        except ValueError as e:
            print(f"⚠️  Warning: Invalid class folder name '{class_name}': {e}")
            return None, None
    
    def _crawl_dataset(self):
        """
        Crawls the directory structure:
        Root/{Class}/{Jersey_Seq_ID}/{Label_Subdir}/*.jpg
        """
        if not os.path.exists(self.root_dir):
            raise ValueError(f"Dataset path does not exist: {self.root_dir}")
        
        # Get all class folders (e.g., '4', '48', '8')
        class_folders = [d for d in os.listdir(self.root_dir) 
                        if os.path.isdir(os.path.join(self.root_dir, d))]
        
        # Filter by allowed classes if specified
        if self.allowed_classes is not None:
            class_folders = [c for c in class_folders 
                           if c.isdigit() and int(c) in self.allowed_classes]
        
        class_folders = sorted(class_folders, key=lambda x: int(x) if x.isdigit() else 999)
        
        total_sequences = 0
        class_distribution = {}
        
        for class_name in class_folders:
            class_path = os.path.join(self.root_dir, class_name)
            
            # Parse labels using improved function
            label_d1, label_d2 = self._parse_class_label(class_name)
            
            # Skip invalid class names
            if label_d1 is None or label_d2 is None:
                continue
            
            # Iterate over Jersey/Sequence folders
            seq_folders = [f for f in os.listdir(class_path) 
                          if os.path.isdir(os.path.join(class_path, f))]
            
            class_seq_count = 0
            
            for seq_id in seq_folders:
                seq_path = os.path.join(class_path, seq_id)
                
                # Iterate over internal label subdirectories (0, 1, 2...)
                # These are tracklets/sub-sequences
                sub_dirs = [s for s in os.listdir(seq_path) 
                           if os.path.isdir(os.path.join(seq_path, s))]
                
                for sub in sub_dirs:
                    final_path = os.path.join(seq_path, sub)
                    
                    # Check if it contains images
                    images = glob.glob(os.path.join(final_path, "*.jpg")) + \
                             glob.glob(os.path.join(final_path, "*.png"))
                    
                    if len(images) >= 1:  # At least 1 frame
                        self.samples.append((final_path, label_d1, label_d2, class_name))
                        class_seq_count += 1
            
            if class_seq_count > 0:
                class_distribution[class_name] = class_seq_count
                total_sequences += class_seq_count
        
        # Print dataset statistics
        print(f"\n{'='*60}")
        print(f"📊 {self.mode.upper()} Dataset Loaded")
        print(f"{'='*60}")
        print(f"Total sequences: {total_sequences}")
        print(f"Classes found: {len(class_distribution)}")
        print(f"\nClass distribution:")
        for cls, count in sorted(class_distribution.items(), key=lambda x: int(x[0])):
            print(f"  Class {cls:>2s}: {count:>4d} sequences")
        print(f"{'='*60}\n")
    
    def _sample_frames(self, image_files):
        """
        Sample frames using different strategies, prioritizing anchor frames.
        
        Args:
            image_files: List of image file paths
            
        Returns:
            indices: List of frame indices to use
        """
        num_frames = len(image_files)
        indices = []
        
        # Find anchor frames
        anchor_indices = [i for i, f in enumerate(image_files) if '_anchor' in f]
        
        # Always include at least one anchor frame if available
        if anchor_indices:
            # Use the first anchor as frame 0
            indices.append(anchor_indices[0])
            # Remove it from candidate pool to avoid duplication
            remaining_indices = [i for i in range(num_frames) if i not in anchor_indices]
        else:
            # No anchors — sample from all frames
            remaining_indices = list(range(num_frames))
        
        # Sample remaining frames based on strategy
        remaining_needed = self.seq_length - len(indices)
        
        if remaining_needed > 0:
            if self.sampling_strategy == 'uniform':
                if len(remaining_indices) >= remaining_needed:
                    step = len(remaining_indices) / remaining_needed
                    sampled = [int(i * step) for i in range(remaining_needed)]
                    indices.extend([remaining_indices[i] for i in sampled])
                else:
                    indices.extend(remaining_indices)
                    while len(indices) < self.seq_length:
                        indices.append(remaining_indices[-1])  # Repeat last
            
            elif self.sampling_strategy == 'random':
                if len(remaining_indices) >= remaining_needed:
                    sampled = np.random.choice(remaining_indices, remaining_needed, replace=False)
                    indices.extend(sampled.tolist())
                else:
                    indices.extend(remaining_indices)
                    while len(indices) < self.seq_length:
                        indices.append(np.random.choice(remaining_indices))
            
            elif self.sampling_strategy == 'center':
                center = len(remaining_indices) // 2
                half_seq = remaining_needed // 2
                start = max(0, center - half_seq)
                end = min(len(remaining_indices), start + remaining_needed)
                sampled = remaining_indices[start:end]
                indices.extend(sampled)
                while len(indices) < self.seq_length:
                    indices.append(sampled[-1])
            
            else:
                raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}")
        
        # Ensure we have exactly seq_length frames
        indices = indices[:self.seq_length]
        while len(indices) < self.seq_length:
            indices.append(indices[-1])
        
        return indices
    
    def __len__(self):
        return len(self.samples)
    
    def _enhance_image(self, img):
        """
        Enhance image quality: sharpen and optionally upscale.
        
        Args:
            img: PIL.Image
            
        Returns:
            Enhanced PIL.Image
        """
        # Option 1: Sharpen using PIL (lightweight, no extra deps)
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(2.0)  # Increase sharpness by 2x
        
        # Optional: Upscale if image is very small (e.g., < 64px)
        min_size = 64
        if img.width < min_size or img.height < min_size:
            # Use LANCZOS for best quality upsampling
            scale_factor = max(min_size / img.width, min_size / img.height)
            new_size = (int(img.width * scale_factor), int(img.height * scale_factor))
            img = img.resize(new_size, Image.LANCZOS)
        
        return img
    
    def __getitem__(self, idx):
        seq_path, d1, d2, class_name = self.samples[idx]
        
        # Get all image files
        image_files = sorted(
            glob.glob(os.path.join(seq_path, "*.jpg")) + 
            glob.glob(os.path.join(seq_path, "*.png"))
        )
        
        # Sample frames using chosen strategy
        indices = self._sample_frames(image_files)
        
        # Load and Transform Images
        frames = []
        for i in indices:
            img_path = image_files[i]
            try:
                img = Image.open(img_path).convert('RGB')
                
                # 👇 ENHANCE IMAGE BEFORE TRANSFORM 👇
                img = self._enhance_image(img)
                
                if self.transform:
                    img = self.transform(img)
                else:
                    # Always resize + normalize for consistency
                    from torchvision import transforms
                    default_transform = transforms.Compose([
                        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
                    img = default_transform(img)
                frames.append(img)
            except Exception as e:
                print(f"⚠️  Warning: Failed to load {img_path}: {e}")
                if self.transform:
                    black_frame = torch.zeros(3, Config.IMG_SIZE, Config.IMG_SIZE)
                else:
                    black_frame = torch.zeros(3, 224, 224)
                frames.append(black_frame)
        
        # Stack into tensor: [Seq_Len, Channels, Height, Width]
        seq_tensor = torch.stack(frames)
        
        return seq_tensor, torch.tensor(d1, dtype=torch.long), torch.tensor(d2, dtype=torch.long)
    
    def get_class_weights(self, method='balanced', suppress_empty_weight=True):
        """
        Calculate class weights for handling imbalanced data.
        
        Args:
            method: 'balanced' or 'effective_number'
            suppress_empty_weight: If True, heavily suppress the weight for D1=10 (empty)
        
        Returns:
            (weights_d1, weights_d2): Tuple of weight tensors
        """
        from collections import Counter
        
        d1_labels = [s[1] for s in self.samples]
        d2_labels = [s[2] for s in self.samples]
        
        d1_counts = Counter(d1_labels)
        d2_counts = Counter(d2_labels)
        
        print(f"\n📊 Label Distribution:")
        print(f"D1 counts: {dict(sorted(d1_counts.items()))}")
        print(f"D2 counts: {dict(sorted(d2_counts.items()))}")
        
        # Initialize weights
        d1_weights = torch.ones(Config.NUM_DIGIT_CLASSES)
        d2_weights = torch.ones(Config.NUM_DIGIT_CLASSES)
        
        total_samples = len(self.samples)
        
        if method == 'balanced':
            # Inverse frequency weighting
            for digit in range(Config.NUM_DIGIT_CLASSES):
                if digit in d1_counts and d1_counts[digit] > 0:
                    d1_weights[digit] = total_samples / (d1_counts[digit] * Config.NUM_DIGIT_CLASSES)
                else:
                    # Missing class: give very low weight (don't penalize)
                    d1_weights[digit] = 0.001
            
            for digit in range(Config.NUM_DIGIT_CLASSES):
                if digit in d2_counts and d2_counts[digit] > 0:
                    d2_weights[digit] = total_samples / (d2_counts[digit] * Config.NUM_DIGIT_CLASSES)
                else:
                    d2_weights[digit] = 0.001
        
        elif method == 'effective_number':
            # Effective number of samples (better for extreme imbalance)
            beta = 0.9999
            
            for digit in range(Config.NUM_DIGIT_CLASSES):
                if digit in d1_counts and d1_counts[digit] > 0:
                    effective_num = (1.0 - beta ** d1_counts[digit]) / (1.0 - beta)
                    d1_weights[digit] = 1.0 / effective_num
                else:
                    d1_weights[digit] = 0.001
            
            for digit in range(Config.NUM_DIGIT_CLASSES):
                if digit in d2_counts and d2_counts[digit] > 0:
                    effective_num = (1.0 - beta ** d2_counts[digit]) / (1.0 - beta)
                    d2_weights[digit] = 1.0 / effective_num
                else:
                    d2_weights[digit] = 0.001
            
            # ADDITIONAL FIX: Manually suppress the "empty" class (10) for D1
            if suppress_empty_weight and 10 in d1_counts:
                print(f"\n🔧 Suppressing D1 'empty' class weight by 10x")
                d1_weights[10] = d1_weights[10] * 0.01  # Reduce by 10x
        
        else:
            raise ValueError(f"Unknown class weight method: {method}")
        
        # DON'T NORMALIZE! Keep the raw weights for maximum reweighting effect
        # Comment out these lines:
        # d1_weights = d1_weights / d1_weights.sum() * Config.NUM_DIGIT_CLASSES
        # d2_weights = d2_weights / d2_weights.sum() * Config.NUM_DIGIT_CLASSES
        
        # Instead, just normalize to prevent extreme gradients
        d1_weights = torch.sqrt(d1_weights)  # Square root to soften
        d2_weights = torch.sqrt(d2_weights)
        
        # Print weight distribution with better formatting
        print(f"\n⚖️  Class Weights (method={method}):")
        print(f"\nD1 weights (Tens digit):")
        for i in range(Config.NUM_DIGIT_CLASSES):
            label = "Empty" if i == 10 else str(i)
            count = d1_counts.get(i, 0)
            print(f"  {label:>5s} (count: {count:>6d}): weight = {d1_weights[i]:.6f}")
        
        print(f"\nD2 weights (Units digit):")
        for i in range(Config.NUM_DIGIT_CLASSES):
            label = "Empty" if i == 10 else str(i)
            count = d2_counts.get(i, 0)
            if count > 0:  # Only print non-zero
                print(f"  {label:>5s} (count: {count:>6d}): weight = {d2_weights[i]:.6f}")
        
        return d1_weights, d2_weights
    
    def set_transform(self, transform):
        """Change the transform (useful for switching between train/val)"""
        self.transform = transform


# Test the dataset
if __name__ == "__main__":
    from torchvision import transforms
    from config import Config, validate_config
    
    validate_config()
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset
    dataset = JerseySequenceDataset(
        root_dir=Config.DATA_ROOT,
        transform=transform,
        seq_length=Config.SEQ_LENGTH,
        allowed_classes=Config.TRAIN_CLASSES,
        mode='train',
        sampling_strategy='uniform'
    )
    
    print(f"\n✅ Dataset created successfully!")
    print(f"Total samples: {len(dataset)}")
    
    # Test loading one sample
    if len(dataset) > 0:
        seq, d1, d2 = dataset[0]
        print(f"\nSample shape: {seq.shape}")
        print(f"Label D1 (tens): {d1.item()}")
        print(f"Label D2 (units): {d2.item()}")
        
        # Test class weights
        w1, w2 = dataset.get_class_weights(method='balanced')
        print(f"\n✅ Class weights calculated!")