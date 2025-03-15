import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CONFIG

class LightweightLipReader(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        
        # Use MobileNetV2 as the visual feature extractor
        mobilenet = torchvision.models.mobilenet_v2(weights='DEFAULT')
        self.cnn = mobilenet.features
        
        # Freeze early layers to reduce training parameters
        for param in list(self.cnn.parameters())[:-8]:
            param.requires_grad = False
            
        # Reduce feature dimensions
        self.feature_projection = nn.Conv2d(1280, CONFIG['feature_dim'], kernel_size=1)
        
        # Lightweight temporal modeling with GRU
        self.temporal_model = nn.GRU(
            input_size=CONFIG['feature_dim']*4*4,  # Spatial features after pooling
            hidden_size=CONFIG['hidden_size'],
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        # Output projection
        self.classifier = nn.Linear(CONFIG['hidden_size']*2, vocab_size)  # *2 for bidirectional
        
    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()
        
        # Process each frame through CNN
        cnn_output = []
        for t in range(seq_len):
            # Handle batch efficiently to avoid OOM
            frame_features = self.cnn(x[:, t])
            frame_features = self.feature_projection(frame_features)
            frame_features = F.adaptive_avg_pool2d(frame_features, (4, 4))
            cnn_output.append(frame_features.view(batch_size, -1))
        
        # Stack along temporal dimension
        cnn_output = torch.stack(cnn_output, dim=1)
        
        # Process through temporal model
        temporal_output, _ = self.temporal_model(cnn_output)
        
        # Classify each time step
        output = self.classifier(temporal_output)
        
        return output
