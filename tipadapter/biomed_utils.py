from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn
import open_clip

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import numpy as np

def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc

def evaluate_metrics(logits, targets):
    probs = torch.softmax(logits, dim=1).cpu().numpy()
    preds = logits.argmax(dim=1).cpu().numpy()
    targets = targets.cpu().numpy()
    
    acc = accuracy_score(targets, preds) * 100
    f1 = f1_score(targets, preds, average='macro')
    
    try:
        if probs.shape[1] == 2:
            auc = roc_auc_score(targets, probs[:, 1])
        else:
            auc = roc_auc_score(targets, probs, multi_class='ovr', average='macro')
    except:
        auc = 0.0
        
    return acc, f1, auc

def clip_classifier(classnames, template, clip_model, tokenizer, device):
    with torch.no_grad():
        clip_weights = []

        for classname in classnames:
            # Check if classname is actually a list of prompts (Ensemble)
            if isinstance(classname, list):
                texts = classname
            else:
                # Single prompt string
                classname = classname.replace('_', ' ')
                texts = [t.format(classname) for t in template]
                
            texts = tokenizer(texts).to(device)
            # prompt ensemble
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=1).to(device)
    return clip_weights

def build_cache_model(cfg, clip_model, train_loader_cache, device):
    if cfg['load_cache'] == False:    
        cache_keys = []
        cache_values = []

        with torch.no_grad():
            # Data augmentation for the cache model
            for augment_idx in range(cfg['augment_epoch']):
                train_features = []

                print('Augment Epoch: {:} / {:}'.format(augment_idx, cfg['augment_epoch']))
                for i, (images, target) in enumerate(tqdm(train_loader_cache)):
                    images = images.to(device)
                    # BioMedCLIP encode_image
                    image_features = clip_model.encode_image(images)
                    train_features.append(image_features)
                    if augment_idx == 0:
                        target = target.to(device)
                        cache_values.append(target)
                cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))
            
        cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
        cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
        cache_keys = cache_keys.permute(1, 0)
        cache_values = F.one_hot(torch.cat(cache_values, dim=0), num_classes=cfg['num_classes']).float()
        # Note: Added num_classes to one_hot for safety, though typically inferred from max val
        # Original uses F.one_hot(torch.cat(cache_values, dim=0)).half()
        # BioMedCLIP might be float32, let's stick to float or match model dtype later.

        if cfg.get('save_cache', False):
            os.makedirs(cfg['cache_dir'], exist_ok=True)
            torch.save(cache_keys, cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots.pt")
            torch.save(cache_values, cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots.pt")

    else:
        cache_keys = torch.load(cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots.pt")
        cache_values = torch.load(cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots.pt")

    return cache_keys, cache_values

def pre_load_features(cfg, split, clip_model, loader, device):
    if cfg['load_pre_feat'] == False:
        features, labels = [], []

        with torch.no_grad():
            for i, (images, target) in enumerate(tqdm(loader)):
                images, target = images.to(device), target.to(device)
                image_features = clip_model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                features.append(image_features)
                labels.append(target)

        features, labels = torch.cat(features), torch.cat(labels)

        if cfg.get('save_cache', False):
            os.makedirs(cfg['cache_dir'], exist_ok=True)
            torch.save(features, cfg['cache_dir'] + "/" + split + "_f.pt")
            torch.save(labels, cfg['cache_dir'] + "/" + split + "_l.pt")
   
    else:
        features = torch.load(cfg['cache_dir'] + "/" + split + "_f.pt")
        labels = torch.load(cfg['cache_dir'] + "/" + split + "_l.pt")
    
    return features, labels

def search_hp(cfg, cache_keys, cache_values, features, labels, clip_weights, adapter=None):
    if cfg['search_hp'] == True:
    
        beta_list = [i * (cfg['search_scale'][0] - 0.1) / cfg['search_step'][0] + 0.1 for i in range(cfg['search_step'][0])]
        alpha_list = [i * (cfg['search_scale'][1] - 0.1) / cfg['search_step'][1] + 0.1 for i in range(cfg['search_step'][1])]

        best_acc = 0
        best_beta, best_alpha = 0, 0

        for beta in beta_list:
            for alpha in alpha_list:
                if adapter:
                    affinity = adapter(features)
                else:
                    affinity = features @ cache_keys

                cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
                clip_logits = 100. * features @ clip_weights
                tip_logits = clip_logits + cache_logits * alpha
                acc = cls_acc(tip_logits, labels)
            
                if acc > best_acc:
                    # print("New best setting, beta: {:.2f}, alpha: {:.2f}; accuracy: {:.2f}".format(beta, alpha, acc))
                    best_acc = acc
                    best_beta = beta
                    best_alpha = alpha

        print("\nAfter searching, the best accuracy: {:.2f} (beta={:.2f}, alpha={:.2f}).\n".format(best_acc, best_beta, best_alpha))
    else:
        best_beta, best_alpha = cfg['init_beta'], cfg['init_alpha']

    return best_beta, best_alpha
