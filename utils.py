import torch.nn as nn
import torch
import torch.utils.data as data
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 移除了 MLP 类

class ModelConfig:
    def __init__(self):
        self.n_users = 2289
        self.n_items = 956

        self.latent_dim = 64  #pw:256  gowalla:64
        self.n_layers = 3 #pw:5  gowalla:3

        self.lamdba1 = 1e-7 #pw:1e-5  gowalla:1e-7  # str_loss_weight
        self.lamdba3 = 1e-4  # reg_loss_weight

        self.str_loss_item_weight = 0.7
        self.str_loss_user_weight = 0.3

        self.str_loss_temp = 0.1

        self.r = 1.25
        self.alpha = 0.7
        self.epochs = 180
        self.lr = 0.001

        self.batch_size = 1024
        self.test_batch_size = 128

        self.topk = 20


class TrnData(data.Dataset):
    def __init__(self, coomat):
        self.rows = coomat.row
        self.cols = coomat.col
        self.dokmat = coomat.todok()
        # 修改：只保留一个负样本数组
        self.negs = np.zeros(len(self.rows)).astype(np.int32)

    def neg_sampling(self):
        n_items = self.dokmat.shape[1]
        for i in range(len(self.rows)):
            u = self.rows[i]
            # 这里的逻辑是标准的随机负采样
            while True:
                i_neg = np.random.randint(n_items)
                if (u, i_neg) not in self.dokmat:
                    break
            self.negs[i] = i_neg

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        # 修改：只返回三个值 (User, Pos, Neg)
        return self.rows[idx], self.cols[idx], self.negs[idx]


class BPRLoss(nn.Module):
    def __init__(self, k=None, gamma=1e-10):
        super(BPRLoss, self).__init__()
        self.gamma = gamma

    # 修改：forward 只接收 pos_score 和 neg_score
    def forward(self, pos_score, neg_score):
        loss = -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8).mean()
        return loss


class EmbLoss(nn.Module):
    """EmbLoss, regularization on embeddings"""

    def __init__(self, norm=2):
        super(EmbLoss, self).__init__()
        self.norm = norm

    def forward(self, *embeddings, require_pow=False):
        if require_pow:
            emb_loss = torch.zeros(1).to(embeddings[-1].device)
            for embedding in embeddings:
                emb_loss += torch.pow(
                    input=torch.norm(embedding, p=self.norm), exponent=self.norm
                )
            emb_loss /= embeddings[-1].shape[0]
            emb_loss /= self.norm
            return emb_loss
        else:
            emb_loss = torch.zeros(1).to(embeddings[-1].device)
            for embedding in embeddings:
                emb_loss += torch.norm(embedding, p=self.norm)
            emb_loss /= embeddings[-1].shape[0]
            return emb_loss


def dcg_at_k(scores, k):
    scores = np.asarray(scores, dtype=float)[:k]
    if scores.size == 0:
        return 0.0
    return np.sum((2 ** scores - 1) / np.log2(np.arange(2, scores.size + 2)))


def ndcg_at_k(predicted_scores, true_scores, k):
    sorted_true_scores = [true for _, true in sorted(zip(predicted_scores, true_scores), reverse=True)]
    dcg = dcg_at_k(sorted_true_scores, k)
    idcg = dcg_at_k(sorted(true_scores, reverse=True), k)
    return dcg / idcg if idcg > 0 else 0.0


class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = 0
        self.counter = 0

    def __call__(self, val_score):
        if val_score > self.best_score + self.min_delta:
            self.best_score = val_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"Early stopping triggered after {self.patience} epochs of no improvement.")
                return True
        return False