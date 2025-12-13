# config.py

"""
Configuration file for Jersey Number Recognition
Optimized for your actual dataset
"""

import os
import warnings
from dotenv import load_dotenv


load_dotenv()  # Load environment variables from .env file if present

class Config:
    # ============================================
    # DATA CONFIGURATION
    # ============================================
    # Use environment variable with fallback to relative path
    DATA_ROOT = os.environ.get('JERSEY_DATA_ROOT')
    
    # ============================================
    #  TRAINING/TEST SPLIT
    # ============================================
    
    TRAIN_CLASSES = [4, 6, 8, 9, 49, 66, 89]  
    TEST_CLASSES = [48, 64, 88]                       
    
    # Validation split (from training data)
    VAL_SPLIT = 0.2
    
    # ============================================
    # MODEL CONFIGURATION
    # ============================================
    NUM_DIGIT_CLASSES = 11  # 0-9 + empty = 11
    HIDDEN_DIM = 256
    IMG_SIZE = 64
    SEQ_LENGTH = 5
    
    # Model architecture options
    BACKBONE = 'resnet18'  
    USE_SPATIAL_ATTENTION = True
    USE_TEMPORAL_ATTENTION = True
    BIDIRECTIONAL_LSTM = True
    
    # ============================================
    # TRAINING CONFIGURATION
    # ============================================
    BATCH_SIZE = 16
    NUM_EPOCHS = 30
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    DROPOUT = 0.3
    
    # Gradient clipping
    GRADIENT_CLIP_VALUE = 1.0
    
    # Learning rate scheduler
    USE_SCHEDULER = True
    SCHEDULER_STEP_SIZE = 10
    SCHEDULER_GAMMA = 0.5
    
    # Early stopping
    EARLY_STOPPING_PATIENCE = 10
    EARLY_STOPPING_MIN_DELTA = 0.001
    
    # Mixed precision training 
    USE_MIXED_PRECISION = True
    
    # Class weight method
    CLASS_WEIGHT_METHOD = 'effective_number'  # 'balanced' or 'effective_number'
    
    # ============================================
    # DATA AUGMENTATION
    # ============================================
    # Safe augmentations that don't change digit identity
    AUG_ROTATION_DEGREES = 5
    AUG_TRANSLATE = (0.05, 0.05)
    AUG_SCALE = (0.95, 1.05)
    AUG_BRIGHTNESS = 0.2
    AUG_CONTRAST = 0.2
    
    # Temporal sampling strategy
    TEMPORAL_SAMPLING = 'uniform'  # 'uniform', 'random', 'center'
    
    # ============================================
    # SYSTEM CONFIGURATION
    # ============================================
    NUM_WORKERS = 4
    DEVICE = 'cuda'
    SEED = 42
    PIN_MEMORY = True


    # Confidence thresholds for zero-shot jersey number reconstruction
    D1_CONF_THRESHOLD = 0.10   # Tens digit confidence threshold
    D2_CONF_THRESHOLD = 0.5   # Units digit confidence threshold
    
    # ============================================
    # OUTPUT PATHS
    # ============================================
    CHECKPOINT_DIR = 'checkpoints'
    RESULTS_DIR = 'results'
    LOGS_DIR = 'logs'
    
    @staticmethod
    def create_dirs():
        """Create all necessary output directories"""
        os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(Config.RESULTS_DIR, exist_ok=True)
        os.makedirs(Config.LOGS_DIR, exist_ok=True)
        os.makedirs(os.path.join(Config.RESULTS_DIR, 'confusion_matrices'), exist_ok=True)
        os.makedirs(os.path.join(Config.RESULTS_DIR, 'predictions'), exist_ok=True)
        os.makedirs(os.path.join(Config.RESULTS_DIR, 'visualizations'), exist_ok=True)
    
    @staticmethod
    def validate():
        """Validate configuration parameters"""
        errors = []
        warnings_list = []
        
        # Check data root exists
        if not os.path.exists(Config.DATA_ROOT):
            errors.append(f"Data root does not exist: {Config.DATA_ROOT}")
        
        # Check train/test split
        if Config.TRAIN_CLASSES is not None and Config.TEST_CLASSES is not None:
            overlap = set(Config.TRAIN_CLASSES) & set(Config.TEST_CLASSES)
            if overlap:
                errors.append(f"Train and test classes overlap: {overlap}")
        
        # Validate hyperparameters
        if Config.BATCH_SIZE < 1:
            errors.append("Batch size must be >= 1")
        
        if not (32 <= Config.IMG_SIZE <= 512):
            warnings_list.append(f"IMG_SIZE={Config.IMG_SIZE} is unusual (recommended: 128-256)")
        
        if Config.SEQ_LENGTH < 1:
            errors.append("SEQ_LENGTH must be >= 1")
        
        if not (0.0 < Config.VAL_SPLIT < 1.0):
            errors.append("VAL_SPLIT must be between 0 and 1")
        
        if not (0.0 <= Config.DROPOUT < 1.0):
            errors.append("DROPOUT must be between 0 and 1")
        
        # Check GPU availability
        if Config.DEVICE == 'cuda':
            try:
                import torch
                if not torch.cuda.is_available():
                    warnings_list.append("CUDA requested but not available, will fall back to CPU")
                    Config.DEVICE = 'cpu'
            except ImportError:
                errors.append("PyTorch not installed")
        
        # Print errors
        if errors:
            print("\n CONFIGURATION ERRORS:")
            for error in errors:
                print(f"  • {error}")
            raise ValueError("Configuration validation failed")
        
        # Print warnings
        if warnings_list:
            print("\n CONFIGURATION WARNINGS:")
            for warning in warnings_list:
                print(f"  • {warning}")
        
        return True


def validate_config():
    """Check if configuration is valid and analyze digit coverage"""
    print("\n" + "="*70)
    print("VALIDATING CONFIGURATION")
    print("="*70)
    
    # Validate parameters
    Config.validate()
    
    # Validate digit coverage
    print("\n🔍 Analyzing Digit Coverage:")
    
    if Config.TRAIN_CLASSES is None:
        print(" No specific training classes defined")
        return
    
    train_tens = set()
    train_units = set()
    
    for cls in Config.TRAIN_CLASSES:
        if cls < 10:
            train_units.add(cls)
        else:
            train_tens.add(cls // 10)
            train_units.add(cls % 10)
    
    print(f"  Training tens digits: {sorted(train_tens)} + empty")
    print(f"  Training units digits: {sorted(train_units)}")
    
    # Check generalization to test classes
    if Config.TEST_CLASSES:
        print(f"\n  Checking generalization to test classes:")
        can_generalize_all = True
        
        for test_cls in Config.TEST_CLASSES:
            if test_cls < 10:
                if test_cls in train_units:
                    print(f" Class {test_cls}: covered (single digit)")
                else:
                    print(f" Class {test_cls}: NOT covered")
                    can_generalize_all = False
            else:
                tens = test_cls // 10
                units = test_cls % 10
                
                tens_covered = tens in train_tens
                units_covered = units in train_units
                
                if tens_covered and units_covered:
                    print(f" Class {test_cls} = {tens}{units}: covered")
                else:
                    missing = []
                    if not tens_covered:
                        missing.append(f"tens={tens}")
                    if not units_covered:
                        missing.append(f"units={units}")
                    print(f" Class {test_cls}: NOT covered ({', '.join(missing)})")
                    can_generalize_all = False
        
        if can_generalize_all:
            print("\n Model should generalize to all test classes!")
        else:
            print("\n WARNING: Model may struggle with some test classes")
    
    print("\n" + "="*70)
    print("CONFIGURATION SUMMARY")
    print("="*70)
    print(f" Data root: {Config.DATA_ROOT}")
    print(f" Train classes: {Config.TRAIN_CLASSES}")
    print(f" Test classes: {Config.TEST_CLASSES}")
    print(f" Validation split: {Config.VAL_SPLIT*100:.0f}%")
    print(f" Backbone: {Config.BACKBONE}")
    print(f" Image size: {Config.IMG_SIZE}x{Config.IMG_SIZE}")
    print(f" Sequence length: {Config.SEQ_LENGTH} frames")
    print(f" Batch size: {Config.BATCH_SIZE}")
    print(f" Epochs: {Config.NUM_EPOCHS}")
    print(f" Learning rate: {Config.LEARNING_RATE}")
    print(f" Checkpoints: {Config.CHECKPOINT_DIR}/")
    print(f" Results: {Config.RESULTS_DIR}/")
    print("="*70 + "\n")


if __name__ == "__main__":
    validate_config()