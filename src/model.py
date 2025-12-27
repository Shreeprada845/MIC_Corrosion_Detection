# src/model.py

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import resnet18, ResNet18_Weights

class CorrosionCNN(nn.Module):
    def __init__(self, pretrained=True, embedding_dim=512):
        super(CorrosionCNN, self).__init__()
        
        # Load pretrained ResNet18
        if pretrained:
             self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        else:
             self.backbone = resnet18(weights=None)
        # Remove the last fully connected layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # keep features only
        
        # New classifier head
        self.classifier = nn.Sequential(
            nn.Linear(num_features, embedding_dim),  # embedding layer
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(embedding_dim, 1),  # binary output
            nn.Sigmoid()
        )
    
    def forward(self, x, return_embedding=False):
        features = self.backbone(x)
        out = self.classifier(features)
        
        if return_embedding:
            return out, features  # for MIC fusion later
        return out

if __name__ == "__main__":
    # Quick test
    model = CorrosionCNN()
    dummy_input = torch.randn(2, 3, 224, 224)  # batch_size=2, RGB 224x224
    output, embeddings = model(dummy_input, return_embedding=True)
    print("Output shape:", output.shape)
    print("Embedding shape:", embeddings.shape)
