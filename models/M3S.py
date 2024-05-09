import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from src.utils import reset, set_random_seeds, masking
from sklearn.cluster import KMeans
from embedder import embedder
from torch_geometric.utils import to_dense_adj
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict
from torch.nn.utils import clip_grad_norm_

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj
from scipy.spatial.distance import pdist, squareform

def contrastive_loss(centroids, positive_samples, negative_samples, temperature=0.1):
    
    # print("Centroids expanded shape:", centroids.shape)
    # print("Positive samples shape:", positive_samples.shape)
    # print("Negative samples shape:", negative_samples.shape)    
    # 计算余弦相似度
    pos_similarities = F.cosine_similarity(centroids, positive_samples, dim=1) / temperature
    neg_similarities = F.cosine_similarity(centroids, negative_samples, dim=1) / temperature

    exp_pos_similarities = torch.exp(pos_similarities)
    exp_neg_similarities = torch.exp(neg_similarities)

    positive_sum = exp_pos_similarities.sum(dim=0)
    negative_sum = exp_neg_similarities.sum(dim=0)

    # 计算InfoNCE
    loss = -torch.log(positive_sum / (positive_sum + negative_sum + 1e-8))

    return loss.mean()



class M3S_Trainer(embedder):
    def __init__(self, args):
        embedder.__init__(self, args)
        self.args = args

    def _init_model(self):
        self.model = M3S(self.encoder, self.decoder, self.classifier).to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.decay)
        
    def _init_dataset(self):
        
        self.labels = deepcopy(self.data.y)
        self.running_train_mask = deepcopy(self.train_mask)
        
        eta = self.data.num_nodes / (to_dense_adj(self.data.edge_index).sum() / self.data.num_nodes)**len(self.hidden_layers)
        self.t = (self.labels[self.train_mask].unique(return_counts=True)[1]*3*eta/len(self.labels[self.train_mask])).type(torch.int64)
        self.t = self.t / self.args.stage



    def pretrain(self, mask, stage):
        
        n_rescon_1 = 1
        n_rescon_2 = 0.1
        
        for epoch in range(200):
            self.model.train()
            self.optimizer.zero_grad()

            logits, _ = self.model.cls(self.data)
            
  
            classification_loss = F.cross_entropy(logits[self.running_train_mask], self.labels[self.running_train_mask])
            
            # 计算logits
            _, _, reconstructed_logits = self.model(self.data)
            # 将edge_index转为邻近矩阵
            adj = to_dense_adj(self.data.edge_index).squeeze(0).to(self.device)
            # 计算重构损失
            labels = adj
            pos_weight = torch.tensor([(labels.size(0) ** 2 - labels.sum()) / labels.sum()], device=self.device)
            reconstruction_loss = F.binary_cross_entropy_with_logits(reconstructed_logits, labels, pos_weight=pos_weight)
            # 添加正则化项
            norm = labels.size(0) ** 2 / ((labels.size(0) ** 2 - labels.sum()) * 2)
            reconstruction_loss = norm * reconstruction_loss
            
            loss = reconstruction_loss * n_rescon_1 + classification_loss
            
            loss.backward()
            self.optimizer.step()

            st = '[Fold : {}][Stage : {}/{}][Epoch {}/{}] Loss: {:.4f}'.format(mask+1, stage+1, self.args.stage, epoch+1, 200, loss.item())
            print(st)
        
        
        
        # Clustering
        self.model.eval()
        
        print("pretraining test")
        self.evaluate(self.data, st)
        if self.cnt == self.args.patience:
            print("early stopping!")
        

        
        self.model.train()
        # optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01) 
               
        # Training loop
        for epoch in range(200):
            self.model.train()    
            self.optimizer.zero_grad()
            
            
            
            
            
            # Compute embeddings
            rep = self.model.encoder(self.data).detach()  
            rep = F.normalize(rep, dim=1)
            rep_tensor = rep.to(self.device)
            
            rep = rep.to('cpu').numpy()
            # Compute initial centroids based on labeled data
            initial_centroids = np.array([rep[self.labels.cpu() == i].mean(axis=0) for i in range(self.num_classes)])
            
            labeled_indices = torch.where(self.train_mask)[0]
            unlabeled_indices = torch.where(~self.train_mask)[0]
            
            # KNN Search using updated centroids
            n_neighbors = self.args.KNNneighbors
            knn = NearestNeighbors(n_neighbors=n_neighbors)
            knn.fit(rep[unlabeled_indices.cpu()])
            # knn.fit(rep)      
            
            
     
            # 构建正负样本对
            positive_samples = [[] for _ in range(self.num_classes)]
            negative_samples = [[] for _ in range(self.num_classes)]
            
            # 用于记录每个样本出现的次数
            sample_counts = {}
            
            for i in range(self.num_classes):
                # 获取当前类别的已标记数据点索引
                class_labeled_indices = labeled_indices[self.labels[labeled_indices] == i]
                class_positive_samples = []  # 临时存储当前类的正样本索引
            
                for idx in class_labeled_indices:
                    # 对每个已标记数据点进行KNN搜索
                    _, neighbors = knn.kneighbors([rep[idx.cpu().numpy()]], n_neighbors=n_neighbors)
                    neighbors = neighbors.flatten()
                    
                    # # 只保留unlabeled data
                    # neighbors = [neighbor for neighbor in neighbors if neighbor in unlabeled_indices]
                    # 将相对索引转换为全局索引
                    global_neighbors = unlabeled_indices[neighbors].cpu().numpy()
                    
                    # 累积当前类的正样本
                    class_positive_samples.extend(global_neighbors)
            
                # 对当前类的正样本进行去重
                class_positive_samples = list(set(class_positive_samples))
                
                # 更新样本出现的次数
                for sample in class_positive_samples:
                    if sample in sample_counts:
                        sample_counts[sample] += 1
                    else:
                        sample_counts[sample] = 1
            
                # 临时存储去重后的正样本
                positive_samples[i] = class_positive_samples
            
            # 移除在多个类别中出现的样本
            for i in range(self.num_classes):
                positive_samples[i] = [sample for sample in positive_samples[i] if sample_counts[sample] == 1]
            
            # 转换为Tensor
            for i in range(self.num_classes):
                positive_samples[i] = torch.tensor(positive_samples[i], dtype=torch.long, device=self.device)
                
            # print("筛选前")
            # for i in range(self.num_classes):
            #     print(f"Class {i}:")
            #     print(f"  Positive samples count: {len(positive_samples[i])}")
            
            # 阈值筛选
            logits, _ = self.model.cls(self.data)
            probs = torch.softmax(logits, dim=1)
            # 保留具有高置信度的正样本
            for i in range(self.num_classes):
                # 使用正样本索引检索logits
                pos_logits = probs[positive_samples[i]]
            
                # 检查每个正样本的最大logit是否大于阈值
                max_logits_values, max_logits_indices = torch.max(pos_logits, dim=1)
                
                avg_confidence = max_logits_values.mean().item()
                # print(f"Class {i} average max probability: {avg_confidence}")
                
                threshold = 0.8 * avg_confidence
                # print(f"Class {i} threshold: {threshold}")
                mask_logits = (max_logits_values > threshold) & (max_logits_indices == i)
                
                # 更新正样本列表，只保留置信度大于阈值的样本
                positive_samples[i] = positive_samples[i][mask_logits]
                
            
            
            # if epoch == 199:
            #     # 执行t-SNE降维
            #     tsne = TSNE(n_components=2, random_state=42, perplexity=30)
            #     rep_2d = tsne.fit_transform(rep)
                
            #     # 其他未标记数据点的索引是除了正样本之外的未标记数据点
            #     all_positive_indices = np.hstack([positive_samples[i].cpu().numpy() for i in range(len(positive_samples))])
            #     other_unlabeled_indices = np.setdiff1d(unlabeled_indices.cpu().numpy(), all_positive_indices)
                
            #     # 为每个类别生成一个颜色映射
            #     colors = cm.rainbow(np.linspace(0, 1, len(positive_samples)))
                
            #     plt.figure(figsize=(12, 10))
                
            #     # 用于图例的标签和手柄列表
            #     legend_labels = []
            #     legend_handles = []
                
            #     # 绘制其他未标记数据点
            #     plt.scatter(rep_2d[other_unlabeled_indices, 0], rep_2d[other_unlabeled_indices, 1], c='grey', alpha=0.4)
                
            #     # 对每个类别进行循环，画出其已标记数据点和KNN找到的未标记数据点
            #     for i, class_positive_samples in enumerate(positive_samples):
            #         class_labeled_indices = labeled_indices[self.labels[labeled_indices] == i].cpu().numpy()
            #         class_positive_samples = class_positive_samples.cpu().numpy()
                    
            #         # 绘制该类的KNN找到的未标记数据点
            #         scatter_unlabeled = plt.scatter(rep_2d[class_positive_samples, 0], rep_2d[class_positive_samples, 1], 
            #                     c=[colors[i]], marker='o', alpha=0.4)
                    
            #         # 绘制该类的已标记数据点
            #         scatter_labeled = plt.scatter(rep_2d[class_labeled_indices, 0], rep_2d[class_labeled_indices, 1], 
            #                     c=[colors[i]], marker='x', alpha=0.8)
            #         # 添加图例标签和手柄
            #         legend_labels.extend([f'Class {i} Labeled', f'Class {i} KNN Unlabeled'])
            #         legend_handles.extend([scatter_labeled, scatter_unlabeled])
                    
            #     # 设置图例位置到图形的外部
            #     plt.legend(legend_handles, legend_labels, loc='center left', bbox_to_anchor=(1, 0.5), title="Legend", markerscale=2, scatterpoints=1)        
            #     plt.tight_layout()
    
            #     plt.show()
            
               
            
            # 遍历每个类别，构建负样本集
            for i in range(self.num_classes):
                # 对于当前类别i，将其他所有类别的正样本合集视为负样本
                other_classes_samples = [positive_samples[j] for j in range(self.num_classes) if j != i]
                # 合并其他所有类的正样本作为当前类的负样本
                negative_samples[i] = torch.cat(other_classes_samples).unique()

            # # 输出每个类别的正负样本数量
            # print("筛选后")
            # for i in range(self.num_classes):
            #     print(f"Class {i}:")
            #     print(f"  Positive samples count: {len(positive_samples[i])}")
            #     print(f"  Negative samples count: {len(negative_samples[i])}")
            
            
            
            # 纯度计算
            true_labels = self.data.y
            
            # 用于存储纯度分数
            positive_purity_scores = []
            negative_purity_scores = []
            
            for i in range(self.num_classes):
                # 正样本真实标签
                pos_true_labels = true_labels[positive_samples[i]]
                # 负样本真实标签
                neg_true_labels = true_labels[negative_samples[i]]
            
                # 计算正样本的纯度
                pos_purity = (pos_true_labels == i).float().mean().item()
                # 计算负样本的纯度
                neg_purity = (neg_true_labels != i).float().mean().item()
            
                # 保存纯度分数
                positive_purity_scores.append(pos_purity)
                negative_purity_scores.append(neg_purity)
            
                # print(f'Class {i} Positive Purity: {pos_purity * 100:.2f}%, Negative Purity: {neg_purity * 100:.2f}%')
            
            # 计算整体纯度
            overall_pos_purity = sum(positive_purity_scores) / self.num_classes
            overall_neg_purity = sum(negative_purity_scores) / self.num_classes
            # print(f'Overall Positive Purity: {overall_pos_purity * 100:.2f}%')
            # print(f'Overall Negative Purity: {overall_neg_purity * 100:.2f}%')

            
            
            # 获取正负样本的嵌入
            for i in range(self.num_classes):
                # 将当前类的已标记数据点加入正样本集
                class_labeled_indices = labeled_indices[self.labels[labeled_indices] == i]
                combined_positive_indices = torch.cat([positive_samples[i], class_labeled_indices])
                positive_samples[i] = rep_tensor[combined_positive_indices]
            
                # 负样本包括KNN找到的负样本和其他类的已标记数据点
                other_class_labeled_indices = labeled_indices[self.labels[labeled_indices] != i]
                combined_negative_indices = torch.cat([negative_samples[i], other_class_labeled_indices])
                negative_samples[i] = rep_tensor[combined_negative_indices]
        
            # 计算对比损失
            cont_loss = 0
            for i in range(len(positive_samples)):
                centroid = initial_centroids[i]
                pos_embeddings = positive_samples[i]
                neg_embeddings = negative_samples[i]
                centroid_embedding = torch.tensor(centroid, device=self.device).unsqueeze(0)
                class_cont_loss = contrastive_loss(centroid_embedding, pos_embeddings, neg_embeddings)
                cont_loss += class_cont_loss
            
            cont_loss /= self.num_classes
            
            if epoch == 0:
                print("\nBefore cons:")
                # 计算每个类的内部平均距离
                for i in range(self.num_classes):
                    # 选择当前类的标记数据的嵌入
                    class_indices = labeled_indices[self.labels[labeled_indices] == i].cpu().numpy()
                    class_embeddings = rep[class_indices]
                    
                    # 如果当前类别的标记数据点大于1个，计算平均距离
                    if len(class_indices) > 1:
                        scaler = StandardScaler()
                        normalized_embeddings = scaler.fit_transform(class_embeddings)
                        distances = pdist(normalized_embeddings, 'euclidean')
                        avg_distance = np.mean(distances)
                    else:
                        avg_distance = 0  # 如果该类只有一个数据点，平均距离设为0
                        
                    print(f"Class {i} average intra-class distance: {avg_distance}")
        
        
        
        
        
            # 计算logits
            _, _, reconstructed_logits = self.model(self.data)
            # 将edge_index转为邻近矩阵
            adj = to_dense_adj(self.data.edge_index).squeeze(0).to(self.device)
            # 计算重构损失
            labels = adj
            pos_weight = torch.tensor([(labels.size(0) ** 2 - labels.sum()) / labels.sum()], device=self.device)
            reconstruction_loss = F.binary_cross_entropy_with_logits(reconstructed_logits, labels, pos_weight=pos_weight)
            # 添加正则化项
            norm = labels.size(0) ** 2 / ((labels.size(0) ** 2 - labels.sum()) * 2)
            reconstruction_loss = norm * reconstruction_loss
            
            loss = reconstruction_loss * n_rescon_2 + cont_loss

            loss.backward()
            


            self.optimizer.step()
    
            # Logging
            st = '[Fold : {}][Stage : {}/{}][Epoch {}/{}] Loss: {:.4f}'.format(mask+1, stage+1, self.args.stage, epoch+1, 200, loss.item())
            print(st)
        
        
        
        # print("Before reassignment:")
        for i in range(self.num_classes):
            # 从self.labels中获取所有已标记数据点的类别标签
            labeled_class_labels = self.labels[labeled_indices]
            # 统计属于类别i的已标记数据点的数量
            class_count = (labeled_class_labels == i).sum().item()
            # print(f"Class {i} has {class_count} labeled data points.")
        
        # 获取所有数据点的类别预测概率
        logits, _ = self.model.cls(self.data)
        probs = torch.softmax(logits, dim=1)
        
        # 用新的encoder更新embeddings
        new_rep = self.model.encoder(self.data).detach()
        new_rep = F.normalize(new_rep, dim=1)

        new_rep = new_rep.to('cpu').numpy()
        
        unlabeled_indices_cpu = unlabeled_indices.cpu().numpy()
        labeled_indices_cpu = labeled_indices.cpu().numpy()
        
        print("\nAfter cons:")
        # 计算每个类的内部平均距离
        for i in range(self.num_classes):
            # 选择当前类的标记数据的嵌入
            class_indices = labeled_indices[self.labels[labeled_indices] == i].cpu().numpy()
            class_embeddings = new_rep[class_indices]
            
            # 如果当前类别的标记数据点大于1个，计算平均距离
            if len(class_indices) > 1:
                scaler = StandardScaler()
                normalized_embeddings = scaler.fit_transform(class_embeddings)
                distances = pdist(normalized_embeddings, 'euclidean')
                avg_distance = np.mean(distances)
            else:
                avg_distance = 0  # 如果该类只有一个数据点，平均距离设为0
                
            print(f"Class {i} average intra-class distance: {avg_distance}")
        
        # KNN邻近搜索最近n_neighbors个点
        knn = NearestNeighbors(n_neighbors=1)
        knn.fit(new_rep[unlabeled_indices_cpu])
        labeled_rep = new_rep[labeled_indices_cpu]

        _, initial_knn_indices  = knn.kneighbors(labeled_rep)
        
        knn = NearestNeighbors(n_neighbors=30)
        knn.fit(new_rep[unlabeled_indices_cpu])
        
        # 需要一个Tensor来更新train_mask
        unlabeled_indices = unlabeled_indices.to(self.data.y.device)
               
        # 更新标签和掩码
        for i, indices in enumerate(initial_knn_indices ):
            # 确保indices是一维的
            indices = indices.flatten()
            # 更新训练掩码，将找到的未标记数据标记为已标记
            # 转换索引回原始数据集的索引
            true_indices = unlabeled_indices_cpu[indices]
            
            # 计算找到的未标记数据点的周围30个未标记邻居的平均logits
            _, neighbor_indices = knn.kneighbors(new_rep[true_indices])
            neighbor_indices = neighbor_indices.flatten()
            global_neighbor_indices = unlabeled_indices_cpu[neighbor_indices]
            
            target_class = self.labels[labeled_indices[i]].item()
            # 计算平均logits
            neighbor_logits = probs[global_neighbor_indices, target_class]
            
            avg_confidence = neighbor_logits.mean().item()
            # 检查是否满足转变为已标记的条件
            if avg_confidence > 0.8:
                # 更新训练掩码，将找到的未标记数据标记为已标记
                valid_indices_mask = true_indices < len(unlabeled_indices)
                valid_indices = true_indices[valid_indices_mask]
                self.train_mask[unlabeled_indices[valid_indices]] = True
            
            labeled_indices = torch.where(self.train_mask)[0]
            unlabeled_indices = torch.where(~self.train_mask)[0]
        
        # 更新labeled_indices和unlabeled_indices以反映更改
        labeled_indices = torch.where(self.train_mask)[0]
        unlabeled_indices = torch.where(~self.train_mask)[0]
        
        # print("\nAfter reassignment:")
        for i in range(self.num_classes):
            # 重新获取所有已标记数据点的类别标签，因为可能有更新
            labeled_class_labels = self.labels[labeled_indices]
            # 再次统计属于类别i的已标记数据点的数量
            class_count = (labeled_class_labels == i).sum().item()
            # print(f"Class {i} has {class_count} labeled data points.")
        

        
       
        
    def train(self):
        
        for fold in range(self.args.folds):
            set_random_seeds(fold)
            self.train_mask, self.val_mask, self.test_mask = masking(fold, self.data, self.args.label_rate)
            self._init_dataset()

            for stage in range(self.args.stage):
                self._init_model()
                self.pretrain(fold, stage)
                
            for epoch in range(1,self.args.epochs+1):
                self.model.train()
                self.optimizer.zero_grad()

                logits, _ = self.model.cls(self.data)
                loss = F.cross_entropy(logits[self.running_train_mask], self.labels[self.running_train_mask])
                loss.backward()
                self.optimizer.step()

                st = '[Fold : {}][Epoch {}/{}] Loss: {:.4f}'.format(fold+1, epoch, self.args.epochs, loss.item())

                # evaluation
                self.evaluate(self.data, st)
                if self.cnt == self.args.patience:
                    print("early stopping!")
                    break
            self.save_results(fold)
        
        self.summary()
                

class M3S(nn.Module):
    def __init__(self, encoder, decoder, classifier):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.classifier = classifier
        self.reset_parameters()

    def forward(self, data):
        encoded = self.encoder(data)
        decoded = self.decoder(encoded)
        logits, predictions = self.classifier(encoded)
        return logits, predictions, decoded

    def cls(self, data):
        logits, predictions, _ = self.forward(data)
        return logits, predictions

    def reset_parameters(self):
        reset(self.encoder)
        reset(self.decoder)
        reset(self.classifier)


def sample_mask(idx, l):
    """Create mask."""
    mask = torch.zeros(l)
    mask[idx] = 1
    return torch.as_tensor(mask, dtype=torch.bool)
