import os
import yaml
import torch
import math
import numpy as np
import clip
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


# =========================
# Metrics & helpers
# =========================

def get_entropy(loss, clip_weights):
    max_entropy = math.log2(clip_weights.size(1))
    return float(loss / max_entropy)


def softmax_entropy(x):
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def avg_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)


def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc


# =========================
# CLIP helpers (DEVICE SAFE)
# =========================

def clip_classifier(classnames, template, clip_model):
    device = next(clip_model.parameters()).device

    texts = [template[0].format(c.replace('_', ' ')) for c in classnames]
    texts = clip.tokenize(texts).to(device)

    with torch.no_grad():
        text_features = clip_model.encode_text(texts)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    return text_features.t()


def get_clip_logits(images, clip_model, clip_weights):
    device = next(clip_model.parameters()).device

    with torch.no_grad():
        if isinstance(images, list):
            images = torch.cat(images, dim=0).to(device)
        else:
            images = images.to(device)

        image_features = clip_model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        clip_logits = 100.0 * image_features @ clip_weights

        if image_features.size(0) > 1:
            batch_entropy = softmax_entropy(clip_logits)
            selected_idx = torch.argsort(batch_entropy)[: max(1, int(batch_entropy.size(0) * 0.1))]
            output = clip_logits[selected_idx]

            image_features = image_features[selected_idx].mean(0, keepdim=True)
            clip_logits = output.mean(0, keepdim=True)

            loss = avg_entropy(output)
            prob_map = output.softmax(1).mean(0, keepdim=True)
            pred = int(clip_logits.topk(1, 1, True, True)[1].item())
        else:
            loss = softmax_entropy(clip_logits)
            prob_map = clip_logits.softmax(1)
            pred = int(clip_logits.topk(1, 1, True, True)[1].item())

        return image_features, clip_logits, loss, prob_map, pred


# =========================
# Config loader
# =========================

def get_config_file(config_path, dataset_name):
    config_file = os.path.join(config_path, f"{dataset_name}.yaml")

    if not os.path.exists(config_file):
        raise FileNotFoundError(
            f"The configuration file was not found:\n{os.path.abspath(config_file)}"
        )

    with open(config_file, "r") as file:
        cfg = yaml.safe_load(file)

    return cfg


# =========================
# Dataset loader (YOUR STRUCTURE)
# =========================

def build_test_data_loader(dataset_name, data_root, preprocess):
    DATASET_MAP = {
        'oral_cancer': 'oral_cancer_classification_dataset',
        'aptos': 'aptos_classification_dataset',
        'finger': 'fingerprint_classification_dataset',
        'mias': 'mias_classification_dataset',
        'octa': 'octa_classification_dataset',
    }

    if dataset_name not in DATASET_MAP:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    dataset_path = os.path.join(data_root, DATASET_MAP[dataset_name])

    test_dir = os.path.join(dataset_path, 'test')
    if not os.path.exists(test_dir):
        test_dir = os.path.join(dataset_path, 'val')

    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"No test/ or val/ folder found in {dataset_path}")

    dataset = datasets.ImageFolder(test_dir, transform=preprocess)

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4
    )

    classnames = dataset.classes
    template = ["a photo of a {}"]

    return loader, classnames, template
