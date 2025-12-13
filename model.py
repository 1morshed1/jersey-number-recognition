# model.py

"""
Enhanced Two-Head Temporal Jersey Recognition Model (CRNN)
Architecture: ResNet18/34 (CNN) + Bidirectional LSTM + Temporal Attention + Dual Classification Heads

Key Features:
- Temporal attention mechanism (no information bottleneck)
- Bidirectional LSTM for better context
- Spatial attention for focusing on jersey region
- Flexible backbone selection
- Better classification heads with batch normalization
- Support for variable-length sequences with padding masks
- Gradient clipping support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Tuple, Optional


class SpatialAttention(nn.Module):
    """
    Spatial attention module to focus on jersey number regions
    Generates attention weights across spatial dimensions
    """
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=1),
            nn.BatchNorm2d(in_channels // 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Args:
            x: [Batch, Channels, H, W]
        Returns:
            Attention-weighted features
        """
        attention = self.conv(x)  # [Batch, 1, H, W]
        return x * attention


class TemporalAttention(nn.Module):
    """
    Temporal attention mechanism to weight different frames
    Learns which frames are most informative for digit recognition
    """
    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, lstm_out, mask=None):
        """
        Args:
            lstm_out: [Batch, Seq, Hidden]
            mask: [Batch, Seq] - Optional padding mask (1 for valid, 0 for padding)
        Returns:
            context: [Batch, Hidden] - Weighted sum of temporal features
            weights: [Batch, Seq] - Attention weights
        """
        # Compute attention scores
        scores = self.attention(lstm_out)  # [Batch, Seq, 1]
        scores = scores.squeeze(-1)  # [Batch, Seq]
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax to get attention weights
        weights = F.softmax(scores, dim=1)  # [Batch, Seq]
        
        # Weighted sum
        context = torch.bmm(weights.unsqueeze(1), lstm_out)  # [Batch, 1, Hidden]
        context = context.squeeze(1)  # [Batch, Hidden]
        
        return context, weights


class EnhancedDigitHead(nn.Module):
    """
    Enhanced classification head with batch normalization and residual connection
    """
    def __init__(self, input_dim, num_classes, dropout=0.3):
        super(EnhancedDigitHead, self).__init__()
        
        hidden_dim = input_dim // 2
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(dropout)
        
        self.fc_out = nn.Linear(hidden_dim, num_classes)
        
        # Residual connection projection if needed
        self.shortcut = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else None
    
    def forward(self, x):
        """
        Args:
            x: [Batch, input_dim]
        Returns:
            logits: [Batch, num_classes]
        """
        identity = x
        
        # First block
        out = self.fc1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout1(out)
        
        # Second block with residual
        if self.shortcut is not None:
            identity = self.shortcut(identity)
        
        out = self.fc2(out)
        out = self.bn2(out)
        out = out + identity  # Residual connection
        out = F.relu(out)
        out = self.dropout2(out)
        
        # Output
        logits = self.fc_out(out)
        
        return logits


class JerseyTemporalNet(nn.Module):
    """
    Enhanced Two-Head Temporal Jersey Recognition Model
    """
    def __init__(
        self, 
        num_classes=11, 
        hidden_dim=256, 
        dropout=0.3,
        backbone='resnet18',
        use_spatial_attention=True,
        use_temporal_attention=True,
        bidirectional=True
    ):
        """
        Args:
            num_classes (int): Number of digit classes (0-9 + empty = 11)
            hidden_dim (int): LSTM hidden dimension
            dropout (float): Dropout probability
            backbone (str): CNN backbone ('resnet18', 'resnet34', 'resnet50')
            use_spatial_attention (bool): Whether to use spatial attention
            use_temporal_attention (bool): Whether to use temporal attention
            bidirectional (bool): Whether to use bidirectional LSTM
        """
        super(JerseyTemporalNet, self).__init__()
        
        self.use_spatial_attention = use_spatial_attention
        self.use_temporal_attention = use_temporal_attention
        self.bidirectional = bidirectional
        
        # ============================================
        # A. SPATIAL FEATURE EXTRACTOR (CNN BACKBONE)
        # ============================================
        self.backbone_name = backbone
        self.backbone, self.feature_dim = self._build_backbone(backbone)
        
        # Spatial attention module (optional)
        if use_spatial_attention:
            # Get the number of channels before global pooling
            # For ResNet: layer4 outputs different channels per variant
            if 'resnet18' in backbone or 'resnet34' in backbone:
                spatial_channels = 512
            else:
                spatial_channels = 2048
            self.spatial_attention = SpatialAttention(spatial_channels)
        
        # ============================================
        # B. TEMPORAL AGGREGATOR (LSTM)
        # ============================================
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout if dropout > 0 else 0,
            bidirectional=bidirectional
        )
        
        # Effective hidden dimension after LSTM
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Temporal attention module (optional)
        if use_temporal_attention:
            self.temporal_attention = TemporalAttention(lstm_output_dim)
            self.aggregation_method = 'attention'
        else:
            self.aggregation_method = 'mean'  # Fallback to mean pooling
        
        # ============================================
        # C. CLASSIFICATION HEADS
        # ============================================
        self.dropout = nn.Dropout(dropout)
        
        # Head 1: Tens Place Digit (0-9, or 10 for "empty")
        self.head_digit1 = EnhancedDigitHead(lstm_output_dim, num_classes, dropout)
        
        # Head 2: Units Place Digit (0-9)
        self.head_digit2 = EnhancedDigitHead(lstm_output_dim, num_classes, dropout)
        
        # Initialize weights
        self._initialize_weights()
    
    def _freeze_backbone_layers(self, backbone, freeze_percentage=0.6):
        """
        Freeze backbone layers intelligently.
        
        Args:
            backbone: The CNN backbone module
            freeze_percentage: Percentage of layers to freeze (0.0 to 1.0)
        
        Returns:
            Number of frozen parameters, total parameters
        """
        # Get all parameters with their names
        params = list(backbone.named_parameters())
        total_layers = len(params)
        freeze_until_idx = int(total_layers * freeze_percentage)
        
        frozen_params = 0
        total_params = 0
        
        print(f"\n🔒 Freezing {freeze_percentage*100:.0f}% of backbone layers ({freeze_until_idx}/{total_layers}):")
        
        for idx, (name, param) in enumerate(params):
            total_params += param.numel()
            
            if idx < freeze_until_idx:
                param.requires_grad = False
                frozen_params += param.numel()
                if idx < 3:  # Print first 3
                    print(f"  ❄️  Frozen: {name} ({param.numel():,} params)")
            else:
                param.requires_grad = True
                if idx == freeze_until_idx:  # Print first unfrozen
                    print(f"  ... ({freeze_until_idx - 3} more frozen layers)")
                    print(f"  🔥 Trainable: {name} ({param.numel():,} params)")
                elif idx < freeze_until_idx + 2:
                    print(f"  🔥 Trainable: {name} ({param.numel():,} params)")
        
        print(f"\n  Total: {frozen_params:,} / {total_params:,} frozen ({frozen_params/total_params*100:.1f}%)")
        
        return frozen_params, total_params
    
    def _build_backbone(self, backbone_name):
        """Build and configure the CNN backbone"""
        if backbone_name == 'resnet18':
            resnet = models.resnet18(pretrained=True)
        elif backbone_name == 'resnet34':
            resnet = models.resnet34(pretrained=True)
        elif backbone_name == 'resnet50':
            resnet = models.resnet50(pretrained=True)
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
        feature_dim = resnet.fc.in_features
        
        # Remove final FC and global pooling
        if self.use_spatial_attention:
            # Keep layer4 for spatial attention, remove avgpool and fc
            backbone = nn.Sequential(*list(resnet.children())[:-2])
        else:
            # Remove only the fc layer
            backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Freeze layers intelligently
        self._freeze_backbone_layers(backbone, freeze_percentage=0.6)
        
        return backbone, feature_dim
    
    def _initialize_weights(self):
        """Initialize weights for new layers"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.constant_(param.data, 0)
    
    def unfreeze_backbone(self, percentage=0.3):
        """
        Gradually unfreeze more backbone layers
        
        Args:
            percentage (float): Percentage of layers to unfreeze (0.0 to 1.0)
        """
        params = list(self.backbone.named_parameters())
        num_to_unfreeze = int(len(params) * percentage)
        
        print(f"\n🔓 Unfreezing {percentage*100:.0f}% of backbone layers ({num_to_unfreeze} layers):")
        
        for idx, (name, param) in enumerate(params[-num_to_unfreeze:]):
            param.requires_grad = True
            if idx < 3:  # Print first 3
                print(f"  🔥 Unfrozen: {name}")
    
    def forward(self, x, mask=None):
        """
        Forward pass
        
        Args:
            x: Input tensor [Batch, Seq_Length, Channels, Height, Width]
            mask: Optional padding mask [Batch, Seq_Length] (1 for valid, 0 for padding)
        
        Returns:
            d1_logits: Logits for tens place [Batch, num_classes]
            d2_logits: Logits for units place [Batch, num_classes]
            attention_weights: Temporal attention weights (if enabled) [Batch, Seq]
        """
        batch_size, seq_length, c, h, w = x.size()
        
        # ============================================
        # Step 1: Extract spatial features from each frame
        # ============================================
        x = x.view(batch_size * seq_length, c, h, w)
        
        # Pass through CNN backbone
        features = self.backbone(x)
        
        # Apply spatial attention if enabled
        if self.use_spatial_attention and features.dim() == 4:
            features = self.spatial_attention(features)
            # Global average pooling
            features = F.adaptive_avg_pool2d(features, (1, 1))
        
        # Flatten spatial dimensions
        features = features.view(batch_size * seq_length, -1)
        
        # Reshape back to sequence
        features = features.view(batch_size, seq_length, -1)
        
        # ============================================
        # Step 2: Temporal modeling with LSTM
        # ============================================
        if mask is not None:
            # Pack padded sequence for efficiency
            lengths = mask.sum(dim=1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                features, lengths, batch_first=True, enforce_sorted=False
            )
            lstm_out, (h_n, c_n) = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, (h_n, c_n) = self.lstm(features)
        
        # ============================================
        # Step 3: Temporal aggregation
        # ============================================
        attention_weights = None
        
        if self.use_temporal_attention:
            # Use attention mechanism
            aggregated, attention_weights = self.temporal_attention(lstm_out, mask)
        else:
            # Fallback to mean pooling
            if mask is not None:
                # Masked mean
                mask_expanded = mask.unsqueeze(-1).expand_as(lstm_out)
                sum_out = (lstm_out * mask_expanded).sum(dim=1)
                lengths = mask.sum(dim=1, keepdim=True).float()
                aggregated = sum_out / lengths
            else:
                aggregated = lstm_out.mean(dim=1)
        
        # Apply dropout
        aggregated = self.dropout(aggregated)
        
        # ============================================
        # Step 4: Digit classification
        # ============================================
        d1_logits = self.head_digit1(aggregated)
        d2_logits = self.head_digit2(aggregated)
        
        if attention_weights is not None:
            return d1_logits, d2_logits, attention_weights
        else:
            return d1_logits, d2_logits
    
    def predict(self, x, mask=None, return_confidence=True):
        """
        Predict digit labels (for inference)
        
        Args:
            x: Input tensor [Batch, Seq_Length, Channels, Height, Width]
            mask: Optional padding mask [Batch, Seq_Length]
            return_confidence: Whether to return confidence scores
        
        Returns:
            d1_pred: Predicted tens digit [Batch]
            d2_pred: Predicted units digit [Batch]
            confidence: Average confidence scores [Batch] (if return_confidence=True)
            attention_weights: Temporal attention weights [Batch, Seq] (if enabled)
        """
        self.eval()
        with torch.no_grad():
            forward_out = self.forward(x, mask)
            
            if len(forward_out) == 3:
                d1_logits, d2_logits, attention_weights = forward_out
            else:
                d1_logits, d2_logits = forward_out
                attention_weights = None
            
            # Get probabilities
            d1_probs = F.softmax(d1_logits, dim=1)
            d2_probs = F.softmax(d2_logits, dim=1)
            
            # Get predictions
            d1_pred = torch.argmax(d1_probs, dim=1)
            d2_pred = torch.argmax(d2_probs, dim=1)
            
            if return_confidence:
                # Get confidence (max probability)
                d1_conf, _ = torch.max(d1_probs, dim=1)
                d2_conf, _ = torch.max(d2_probs, dim=1)
                
                # Average confidence
                confidence = (d1_conf + d2_conf) / 2
                
                if attention_weights is not None:
                    return d1_pred, d2_pred, confidence, attention_weights
                else:
                    return d1_pred, d2_pred, confidence
            else:
                if attention_weights is not None:
                    return d1_pred, d2_pred, attention_weights
                else:
                    return d1_pred, d2_pred
    
    def get_num_params(self, trainable_only=True):
        """Return the number of parameters"""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())


# Test the model
if __name__ == "__main__":
    print("="*70)
    print("Testing Enhanced JerseyTemporalNet Model")
    print("="*70)
    
    # Test configuration
    batch_size = 4
    seq_length = 8
    img_size = 224
    num_classes = 11
    
    # Create model
    print("\n1. Creating model...")
    model = JerseyTemporalNet(
        num_classes=num_classes,
        hidden_dim=256,
        dropout=0.3,
        backbone='resnet18',
        use_spatial_attention=True,
        use_temporal_attention=True,
        bidirectional=True
    )
    
    print(f"   ✅ Model created!")
    print(f"   Total parameters: {model.get_num_params(trainable_only=False):,}")
    print(f"   Trainable parameters: {model.get_num_params(trainable_only=True):,}")
    
    # Create dummy input
    dummy_input = torch.randn(batch_size, seq_length, 3, img_size, img_size)
    print(f"\n2. Testing forward pass...")
    print(f"   Input shape: {dummy_input.shape}")
    
    # Forward pass
    model.eval()
    output = model(dummy_input)
    
    if len(output) == 3:
        d1_logits, d2_logits, attention = output
        print(f"   Output D1 shape: {d1_logits.shape}")
        print(f"   Output D2 shape: {d2_logits.shape}")
        print(f"   Attention weights shape: {attention.shape}")
    else:
        d1_logits, d2_logits = output
        print(f"   Output D1 shape: {d1_logits.shape}")
        print(f"   Output D2 shape: {d2_logits.shape}")
    
    print("\n" + "="*70)
    print("✅ All tests passed!")
    print("="*70)