import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class DynamicBPaCoModel(nn.Module):
    """
    Step 4: BPaCo with Dynamic Heatmap Fusion, Grad-CAM Guidance, and 
    Advanced Contrastive Strategies.
    """
    def __init__(self, backbone='resnet18', num_classes=19, proj_dim=128):
        super().__init__()
        # Load backbone
        model_fun = getattr(models, backbone)
        self.encoder = model_fun(weights='DEFAULT')
        
        # Modify first conv to accept 4 channels (RGB + Heatmap)
        # However, the user approach was [Gray, Gray, Heatmap]. 
        # For OCTA, we might want [Image, Image, Heatmap] or [Image, Heatmap, Morph].
        # Let's assume 4 channels [R, G, B, Heatmap] for flexibility.
        original_conv1 = self.encoder.conv1
        self.encoder.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.encoder.conv1.weight[:, :3, :, :] = original_conv1.weight
            # Initialize 4th channel (Heatmap) with mean of RGB weights
            self.encoder.conv1.weight[:, 3, :, :] = original_conv1.weight.mean(dim=1)

        self.feat_dim = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()
        
        # Projection head for Contrastive Learning
        self.proj = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feat_dim, proj_dim)
        )
        
        # Classifier
        self.classifier = nn.Linear(self.feat_dim, num_classes)
        
        # Prototypes for Long-tail (Class-complement)
        self.prototypes = nn.Parameter(torch.randn(num_classes, proj_dim))
        
        # Grad-CAM Heatmap extractor (hook for guidance)
        self.gradients = None
        self.activations = None

    def _save_gradient(self, grad):
        self.gradients = grad

    def forward(self, x, heatmap=None, lambda_val=0.0):
        """
        x: [B, 3, H, W]
        heatmap: [B, 1, H, W]
        lambda_val: Dynamic fusion weight.
        """
        if heatmap is not None:
            # Step 4.1: Dynamic Fusion
            # Scale heatmap by lambda before concatenation or additive fusion
            x_input = torch.cat([x, heatmap * lambda_val], dim=1)
        else:
            x_input = torch.cat([x, torch.zeros_like(x[:,:1,:,:])], dim=1)

        # Grad-CAM related hooks (usually on the last conv layer)
        # For simplicity, we assume we want to guide the features from the encoder
        features = self.encoder(x_input)
        
        # Classifier & Projector
        logits = self.classifier(features)
        z = F.normalize(self.proj(features), dim=1)
        
        return features, logits, z

    def get_gradcam_loss(self, target_heatmap):
        """
        Step 4.2: Grad-CAM Guidance.
        Ensures model's internal attention (CAM) focuses on the same regions as Grounding DINO.
        """
        if self.gradients is None or self.activations is None:
            return torch.tensor(0.0, device=target_heatmap.device)
        
        # Simple Global Average Pooling implementation of CAM
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Scale to match target size
        cam = F.interpolate(cam, size=target_heatmap.shape[2:], mode='bilinear', align_corners=False)
        
        # Normalize
        cam = cam / (cam.max() + 1e-8)
        
        return F.mse_loss(cam, target_heatmap)

def compute_contrastive_loss(z, labels, prototypes, inter_class_penalty=True):
    """
    Step 4.4: Inter-class Contrast with Penalty.
    """
    # Standard Supervised Contrastive
    # ...
    
    # Step 4.4 specific penalty:
    # If classes i and j are confusing AND heatmap features are similar, push them apart more.
    penalty = 0.0
    if inter_class_penalty:
        # Example logic: calculate pairwise similarity between prototypes
        # and increase loss for high-similarity pairs
        proto_sim = torch.matmul(prototypes, prototypes.T)
        penalty = torch.mean(torch.clamp(proto_sim - 0.7, min=0)) # Push prototypes apart
        
    return F.cross_entropy(torch.matmul(z, prototypes.T), labels) + penalty

# Example usage
if __name__ == "__main__":
    model = DynamicBPaCoModel()
    img = torch.randn(1, 3, 224, 224)
    hmap = torch.randn(1, 1, 224, 224)
    f, l, z = model(img, hmap, lambda_val=0.1)
    print(f.shape, l.shape, z.shape)
