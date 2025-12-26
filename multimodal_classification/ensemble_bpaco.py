#!/usr/bin/env python3
"""
集成学习: BPaCo + BPaCo Multimodal + CSV特征
使用多种集成策略提升性能
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import csv
import json

# 导入模型
from train_bpaco_19cls import BpacoEncoder as BpacoEncoderImage
from train_bpaco_multimodal import BpacoEncoder as BpacoEncoderMultimodal
from train_bpaco_multimodal import MultiModalBPaCoDataset


class EnsemblePredictor:
    """集成预测器"""
    
    def __init__(
        self,
        bpaco_model_path,
        multimodal_model_path,
        keypoint_features_file,
        device='cuda'
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.keypoint_features_file = keypoint_features_file
        
        # 加载关键点特征
        self.keypoint_features = self._load_keypoint_features()
        
        # 加载BPaCo模型（纯图像）
        print("加载BPaCo模型（纯图像）...")
        self.bpaco_model, self.bpaco_classifier = self._load_bpaco_model(
            bpaco_model_path, multimodal=False
        )
        
        # 加载BPaCo Multimodal模型
        print("加载BPaCo Multimodal模型...")
        self.multimodal_model, self.multimodal_classifier = self._load_bpaco_model(
            multimodal_model_path, multimodal=True
        )
        
        # 集成学习器（将在训练时初始化）
        self.meta_learner = None
        
    def _load_keypoint_features(self):
        """加载关键点特征"""
        features = {}
        with open(self.keypoint_features_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rel_path = row['image_path']
                features[rel_path] = {
                    'num_kp1': int(row['num_kp1']),
                    'num_kp2': int(row['num_kp2']),
                    'is_left_hand': int(row['is_left_hand']),
                    'is_right_hand': int(row['is_right_hand']),
                    'is_hard_sample': int(row['is_hard_sample']),
                    'kp1_between_kp2': int(row['kp1_between_kp2']),
                    'kp1_left_of_kp2': int(row['kp1_left_of_kp2']),
                    'kp1_right_of_kp2': int(row['kp1_right_of_kp2']),
                }
        return features
    
    def _load_bpaco_model(self, checkpoint_path, multimodal=False):
        """加载BPaCo模型"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if multimodal:
            # 多模态模型
            model = BpacoEncoderMultimodal(
                backbone='resnet34', proj_dim=128, pretrained=False
            ).to(self.device)
            fused_feat_dim = model.fused_feat_dim
        else:
            # 纯图像模型
            model = BpacoEncoderImage(
                backbone='resnet34', proj_dim=128, pretrained=False
            ).to(self.device)
            fused_feat_dim = model.feat_dim
        
        model.load_state_dict(checkpoint['model_q_state_dict'])
        model.eval()
        
        # 分类器
        classifier = nn.Sequential(
            nn.Linear(fused_feat_dim * 2, fused_feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fused_feat_dim, 19),
        ).to(self.device)
        classifier.load_state_dict(checkpoint['classifier_state_dict'])
        classifier.eval()
        
        return model, classifier
    
    @torch.no_grad()
    def extract_features(self, dataloader, include_logits=True, include_preds=True):
        """
        提取所有模型的特征和预测

        Args:
            dataloader: 数据加载器
            include_logits: 是否包含logits
            include_preds: 是否包含预测类别（one-hot编码）

        Returns:
            features: [N, feature_dim] - 拼接的特征
            labels: [N] - 真实标签
        """
        all_features = []
        all_labels = []

        for v1, v2, kp_features, labels in dataloader:
            v1 = v1.to(self.device)
            v2 = v2.to(self.device)
            kp_features = kp_features.to(self.device)

            # BPaCo模型预测（纯图像）
            feat_q1, _ = self.bpaco_model(v1)
            feat_k1, _ = self.bpaco_model(v2)
            feat1 = torch.cat([feat_q1, feat_k1], dim=1)
            logits1 = self.bpaco_classifier(feat1)
            probs1 = torch.softmax(logits1, dim=1)
            preds1 = torch.argmax(logits1, dim=1)

            # BPaCo Multimodal模型预测
            feat_q2, _ = self.multimodal_model(v1, kp_features)
            feat_k2, _ = self.multimodal_model(v2, kp_features)
            feat2 = torch.cat([feat_q2, feat_k2], dim=1)
            logits2 = self.multimodal_classifier(feat2)
            probs2 = torch.softmax(logits2, dim=1)
            preds2 = torch.argmax(logits2, dim=1)

            # 构建特征向量
            feature_list = []

            # 1. BPaCo概率分布 [B, 19]
            feature_list.append(probs1)

            # 2. BPaCo Multimodal概率分布 [B, 19]
            feature_list.append(probs2)

            # 3. BPaCo logits [B, 19] (可选)
            if include_logits:
                feature_list.append(logits1)
                feature_list.append(logits2)

            # 4. BPaCo预测类别 (one-hot) [B, 19] (可选)
            if include_preds:
                preds1_onehot = torch.nn.functional.one_hot(preds1, num_classes=19).float()
                preds2_onehot = torch.nn.functional.one_hot(preds2, num_classes=19).float()
                feature_list.append(preds1_onehot)
                feature_list.append(preds2_onehot)

            # 5. 关键点特征 [B, 8]
            feature_list.append(kp_features)

            # 6. 预测一致性特征 [B, 1]
            # 两个模型预测是否一致
            agreement = (preds1 == preds2).float().unsqueeze(1)
            feature_list.append(agreement)

            # 7. 置信度特征 [B, 2]
            # 两个模型的最大概率（置信度）
            conf1 = torch.max(probs1, dim=1, keepdim=True)[0]
            conf2 = torch.max(probs2, dim=1, keepdim=True)[0]
            feature_list.append(conf1)
            feature_list.append(conf2)

            # 8. 概率差异特征 [B, 19]
            # 两个模型概率分布的差异
            prob_diff = torch.abs(probs1 - probs2)
            feature_list.append(prob_diff)

            # 拼接所有特征
            batch_features = torch.cat(feature_list, dim=1)

            all_features.append(batch_features.cpu().numpy())
            all_labels.append(labels.numpy())

        features = np.vstack(all_features)
        labels = np.concatenate(all_labels)

        return features, labels

    def train_meta_learner(self, train_features, train_labels, method='xgboost'):
        """
        训练元学习器

        Args:
            train_features: [N, 46] - 训练特征
            train_labels: [N] - 训练标签
            method: 'xgboost', 'rf', 'gbdt', 'lr'
        """
        print(f"\n训练元学习器: {method}")

        if method == 'xgboost':
            self.meta_learner = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                use_label_encoder=False,
                eval_metric='mlogloss'
            )
        elif method == 'rf':
            self.meta_learner = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        elif method == 'gbdt':
            self.meta_learner = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        elif method == 'lr':
            self.meta_learner = LogisticRegression(
                max_iter=1000,
                random_state=42,
                n_jobs=-1
            )
        else:
            raise ValueError(f"不支持的方法: {method}")

        self.meta_learner.fit(train_features, train_labels)
        print(f"✅ 元学习器训练完成")

    def predict(self, test_features):
        """使用元学习器预测"""
        if self.meta_learner is None:
            raise ValueError("元学习器未训练！请先调用 train_meta_learner()")

        return self.meta_learner.predict(test_features)

    def predict_proba(self, test_features):
        """预测概率"""
        if self.meta_learner is None:
            raise ValueError("元学习器未训练！请先调用 train_meta_learner()")

        return self.meta_learner.predict_proba(test_features)


def simple_ensemble(features, method='average'):
    """
    简单集成策略（不需要训练）

    Args:
        features: [N, feature_dim] - 特征矩阵
            前19维: BPaCo概率
            19-38维: Multimodal概率
        method: 'average', 'max', 'weighted'

    Returns:
        preds: [N] - 预测标签
    """
    # 提取概率分布
    probs1 = features[:, :19]  # BPaCo概率
    probs2 = features[:, 19:38]  # Multimodal概率

    if method == 'average':
        # 平均概率
        probs = (probs1 + probs2) / 2
    elif method == 'max':
        # 取最大概率
        probs = np.maximum(probs1, probs2)
    elif method == 'weighted':
        # 加权平均（根据历史性能）
        # BPaCo: F1=0.5026, Multimodal: F1=0.5905
        w1 = 0.5026 / (0.5026 + 0.5905)
        w2 = 0.5905 / (0.5026 + 0.5905)
        probs = w1 * probs1 + w2 * probs2
    else:
        raise ValueError(f"不支持的方法: {method}")

    return np.argmax(probs, axis=1)


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='BPaCo集成学习')
    parser.add_argument('--dataset', type=str, required=True,
                       help='数据集路径')
    parser.add_argument('--keypoint-features', type=str, required=True,
                       help='关键点特征CSV文件')
    parser.add_argument('--bpaco-model', type=str,
                       default='results/bpaco_19cls/best_model.pth',
                       help='BPaCo模型路径')
    parser.add_argument('--multimodal-model', type=str,
                       default='results/bpaco_multimodal/best_model.pth',
                       help='BPaCo Multimodal模型路径')
    parser.add_argument('--method', type=str, default='xgboost',
                       choices=['xgboost', 'rf', 'gbdt', 'lr', 'average', 'weighted'],
                       help='集成方法')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--output', type=str, default='results/ensemble',
                       help='输出目录')

    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("BPaCo集成学习")
    print("=" * 60)
    print(f"数据集: {args.dataset}")
    print(f"集成方法: {args.method}")
    print(f"输出目录: {args.output}")

    # 创建数据加载器
    print("\n加载数据集...")
    train_dataset = MultiModalBPaCoDataset(
        args.dataset, args.keypoint_features, split='train', image_size=224
    )
    val_dataset = MultiModalBPaCoDataset(
        args.dataset, args.keypoint_features, split='val', image_size=224
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    # 创建集成预测器
    predictor = EnsemblePredictor(
        bpaco_model_path=args.bpaco_model,
        multimodal_model_path=args.multimodal_model,
        keypoint_features_file=args.keypoint_features,
    )

    # 提取特征
    print("\n提取训练集特征...")
    train_features, train_labels = predictor.extract_features(
        train_loader, include_logits=True, include_preds=True
    )
    print(f"训练特征形状: {train_features.shape}")

    print("\n提取验证集特征...")
    val_features, val_labels = predictor.extract_features(
        val_loader, include_logits=True, include_preds=True
    )
    print(f"验证特征形状: {val_features.shape}")

    # 打印特征组成
    print("\n特征组成:")
    print("  - BPaCo概率分布: 19维")
    print("  - Multimodal概率分布: 19维")
    print("  - BPaCo logits: 19维")
    print("  - Multimodal logits: 19维")
    print("  - BPaCo预测类别(one-hot): 19维")
    print("  - Multimodal预测类别(one-hot): 19维")
    print("  - 关键点特征: 8维")
    print("  - 预测一致性: 1维")
    print("  - BPaCo置信度: 1维")
    print("  - Multimodal置信度: 1维")
    print("  - 概率差异: 19维")
    print(f"  总计: {train_features.shape[1]}维")

    # 集成预测
    if args.method in ['xgboost', 'rf', 'gbdt', 'lr']:
        # 训练元学习器
        predictor.train_meta_learner(train_features, train_labels, method=args.method)
        val_preds = predictor.predict(val_features)
    else:
        # 简单集成
        val_preds = simple_ensemble(val_features, method=args.method)

    # 评估
    acc = accuracy_score(val_labels, val_preds)
    f1 = f1_score(val_labels, val_preds, average='macro')

    print("\n" + "=" * 60)
    print("集成结果")
    print("=" * 60)
    print(f"验证准确率: {acc:.4f}")
    print(f"验证F1:     {f1:.4f}")

    # 详细报告
    report = classification_report(val_labels, val_preds, output_dict=True)

    # 保存结果
    results = {
        'method': args.method,
        'accuracy': float(acc),
        'f1': float(f1),
        'classification_report': report
    }

    with open(output_dir / 'ensemble_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n结果已保存至: {output_dir / 'ensemble_results.json'}")


if __name__ == '__main__':
    main()


