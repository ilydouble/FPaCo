
import torch
import torch.nn as nn
from train_gpaco import GPaCoLoss

def test_gpaco_loss():
    # Simulate a batch size of 32
    batch_size = 32
    feat_dim = 128
    num_classes = 10
    
    # Simulate features from encoder (z_q)
    z_q = torch.randn(batch_size, feat_dim)
    
    # Simulate empty queue scenario (features = z_q)
    # This is what happens when queue is empty or not full yet and we don't concatenate
    # But wait, the code says:
    # if q_feats is not None: features = cat([z_q, q_feats])
    # else: features = z_q
    
    features = z_q
    labels = torch.randint(0, num_classes, (batch_size,))
    sup_logits = torch.randn(batch_size, num_classes)
    
    # Initialize Loss
    # K=8192 is default
    criterion = GPaCoLoss(K=8192, num_classes=num_classes)
    
    print("Attempting forward pass with features shape:", features.shape)
    try:
        loss = criterion(features, labels=labels, sup_logits=sup_logits)
        print("Forward pass successful with loss:", loss.item())
    except RuntimeError as e:
        print("Caught expected RuntimeError:", e)
    except Exception as e:
        print("Caught unexpected exception:", e)

if __name__ == "__main__":
    test_gpaco_loss()
