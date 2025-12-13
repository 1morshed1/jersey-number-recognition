# analyze_weights.py - Summary of Trained Weights
import torch
import json
import os
from config import Config
from model import JerseyTemporalNet

def analyze_model_weights(checkpoint_path):
    """Generate comprehensive weight summary"""
    
    print("="*70)
    print("TRAINED WEIGHTS SUMMARY")
    print("="*70)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print(f"\n📁 Checkpoint: {checkpoint_path}")
    print(f"🏆 Best Validation Accuracy: {checkpoint.get('best_val_acc', 'N/A'):.2f}%")
    print(f"📅 Training Epoch: {checkpoint.get('epoch', 'N/A')}")
    
    # Create model
    model = JerseyTemporalNet(
        num_classes=Config.NUM_DIGIT_CLASSES,
        hidden_dim=Config.HIDDEN_DIM,
        dropout=Config.DROPOUT,
        backbone=Config.BACKBONE,
        use_spatial_attention=Config.USE_SPATIAL_ATTENTION,
        use_temporal_attention=Config.USE_TEMPORAL_ATTENTION,
        bidirectional=Config.BIDIRECTIONAL_LSTM
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Analyze weights by component
    print("\n" + "="*70)
    print("MODEL ARCHITECTURE SUMMARY")
    print("="*70)
    
    components = {
        'backbone': [],
        'spatial_attention': [],
        'lstm': [],
        'temporal_attention': [],
        'head_digit1': [],
        'head_digit2': [],
        'other': []
    }
    
    # Categorize parameters
    for name, param in model.named_parameters():
        if 'backbone' in name:
            components['backbone'].append((name, param))
        elif 'spatial_attention' in name:
            components['spatial_attention'].append((name, param))
        elif 'lstm' in name:
            components['lstm'].append((name, param))
        elif 'temporal_attention' in name:
            components['temporal_attention'].append((name, param))
        elif 'head_digit1' in name:
            components['head_digit1'].append((name, param))
        elif 'head_digit2' in name:
            components['head_digit2'].append((name, param))
        else:
            components['other'].append((name, param))
    
    # Print summary for each component
    total_params = 0
    trainable_params = 0
    
    for component_name, params in components.items():
        if not params:
            continue
            
        component_total = sum(p.numel() for _, p in params)
        component_trainable = sum(p.numel() for _, p in params if p.requires_grad)
        
        total_params += component_total
        trainable_params += component_trainable
        
        print(f"\n🔹 {component_name.upper().replace('_', ' ')}")
        print(f"   Total Parameters: {component_total:,}")
        print(f"   Trainable: {component_trainable:,} ({100*component_trainable/component_total:.1f}%)")
        print(f"   Layers: {len(params)}")
        
        # Show first 3 layers
        for i, (name, param) in enumerate(params[:3]):
            print(f"      {name}: {list(param.shape)}")
        if len(params) > 3:
            print(f"      ... ({len(params)-3} more layers)")
    
    print("\n" + "="*70)
    print("OVERALL STATISTICS")
    print("="*70)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
    print(f"Frozen Parameters: {total_params - trainable_params:,}")
    print(f"Model Size: {total_params * 4 / (1024**2):.2f} MB (FP32)")
    
    # Analyze digit heads specifically
    print("\n" + "="*70)
    print("DIGIT HEAD ANALYSIS (KEY FOR GENERALIZATION)")
    print("="*70)
    
    for head_name in ['head_digit1', 'head_digit2']:
        print(f"\n🎯 {head_name.upper()} (Tens Digit)" if '1' in head_name else f"\n🎯 {head_name.upper()} (Units Digit)")
        head_params = components[head_name]
        
        for name, param in head_params:
            if 'fc_out.weight' in name:
                print(f"   Output Layer Shape: {list(param.shape)}")
                print(f"   → Maps {param.shape[1]} features → {param.shape[0]} digit classes")
                print(f"   Weight Statistics:")
                print(f"      Mean: {param.data.mean().item():.4f}")
                print(f"      Std:  {param.data.std().item():.4f}")
                print(f"      Min:  {param.data.min().item():.4f}")
                print(f"      Max:  {param.data.max().item():.4f}")
    
    # Save summary to JSON
    summary = {
        'checkpoint_path': checkpoint_path,
        'best_val_acc': float(checkpoint.get('best_val_acc', 0)),
        'epoch': int(checkpoint.get('epoch', 0)),
        'total_parameters': int(total_params),
        'trainable_parameters': int(trainable_params),
        'model_size_mb': float(total_params * 4 / (1024**2)),
        'components': {
            name: {
                'total_params': int(sum(p.numel() for _, p in params)),
                'trainable_params': int(sum(p.numel() for _, p in params if p.requires_grad)),
                'num_layers': len(params)
            }
            for name, params in components.items() if params
        }
    }
    
    output_path = os.path.join(Config.RESULTS_DIR, 'weight_summary.json')
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"\n✅ Summary saved to: {output_path}")
    print("="*70)

if __name__ == "__main__":
    analyze_model_weights('checkpoints/best_model.pth')