import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np
from utils import BPRLoss, EmbLoss, ndcg_at_k


class DOGCL(nn.Module):
    def __init__(self, n_users, n_items, latent_dim, n_layers, str_loss_temp, lambda1, lambda3, r,
                 str_loss_user_weight, str_loss_item_weight, alpha, interaction_matrix, device,
                 test_mapping, train_mapping, topk, epoch_num):
        super(DOGCL, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.latent_dim = latent_dim
        self.n_layers = n_layers
        self.str_loss_temp = str_loss_temp
        self.lambda1 = lambda1
        self.lambda3 = lambda3
        self.r = r
        self.str_loss_user_weight = str_loss_user_weight
        self.str_loss_item_weight = str_loss_item_weight
        self.device = device
        self.alpha = torch.tensor(alpha).to(self.device)
        self.mf_loss = BPRLoss(epoch_num)
        self.reg_loss = EmbLoss()
        self.user_embedding = nn.Embedding(n_users, latent_dim)
        self.item_embedding = nn.Embedding(n_items, latent_dim)
        self.interaction_matrix = interaction_matrix
        self.acc_norm_adj_mat = self.acc_get_norm_adj_mat().to(device)
        self.nacc_norm_adj_mat = self.nacc_get_norm_adj_mat().to(device)
        self.test_mapping = test_mapping
        self.train_mapping = train_mapping
        self.top_k = topk
        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.user_embedding.weight, 0, 0.01)
        nn.init.normal_(self.item_embedding.weight, 0, 0.01)

    def acc_get_norm_adj_mat(self):
        A = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = inter_M.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        dict.update(A, data_dict)
        sumArr = (A > 0).sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D @ A @ D
        L = sp.coo_matrix(L)
        row, col, data = L.row, L.col, L.data
        return torch.sparse_coo_tensor(torch.LongTensor([row, col]), torch.FloatTensor(data), torch.Size(L.shape))

    def nacc_get_norm_adj_mat(self):
        A = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = inter_M.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        dict.update(A, data_dict)
        sumArr = (A > 0).sum(axis=1)
        diag_left = np.array(sumArr.flatten())[0] + 1e-7
        diag_left = np.power(diag_left, -self.r)
        diag_right = np.array(sumArr.flatten())[0] + 1e-7
        diag_right = np.power(diag_right, -(1 - self.r))
        D_left = sp.diags(diag_left)
        D_right = sp.diags(diag_right)
        L = D_left @ A @ D_right
        L = sp.coo_matrix(L)
        row, col, data = L.row, L.col, L.data
        return torch.sparse_coo_tensor(torch.LongTensor([row, col]), torch.FloatTensor(data), torch.Size(L.shape))

    def update_geometric_nacc(self):
        with torch.no_grad():
            user_emb = self.user_embedding.weight
            item_emb = self.item_embedding.weight
            coo = self.interaction_matrix.tocoo()
            rows = torch.tensor(coo.row, dtype=torch.long, device=self.device)
            cols = torch.tensor(coo.col, dtype=torch.long, device=self.device)
            indices = torch.stack([rows, cols])
            values = torch.ones(len(rows), device=self.device)
            adj_ui = torch.sparse_coo_tensor(indices, values, (self.n_users, self.n_items))
            user_neighbor_sum = torch.sparse.mm(adj_ui, item_emb)
            user_degree = torch.sparse.mm(adj_ui, torch.ones((self.n_items, 1), device=self.device)).squeeze()
            user_degree = user_degree.clamp(min=1e-7).unsqueeze(1)
            user_centroids = user_neighbor_sum / user_degree
            indices_t = torch.stack([cols, rows])
            adj_iu = torch.sparse_coo_tensor(indices_t, values, (self.n_items, self.n_users))
            item_neighbor_sum = torch.sparse.mm(adj_iu, user_emb)
            item_degree = torch.sparse.mm(adj_iu, torch.ones((self.n_users, 1), device=self.device)).squeeze()
            item_degree = item_degree.clamp(min=1e-7).unsqueeze(1)
            item_centroids = item_neighbor_sum / item_degree
            u_centroids_expanded = user_centroids[rows]
            i_embeddings_expanded = item_emb[cols]
            cos_sim_ui = F.cosine_similarity(u_centroids_expanded, i_embeddings_expanded, dim=1)
            weights_ui = 1.0 - cos_sim_ui
            i_centroids_expanded = item_centroids[cols]
            u_embeddings_expanded = user_emb[rows]
            cos_sim_iu = F.cosine_similarity(i_centroids_expanded, u_embeddings_expanded, dim=1)
            weights_iu = 1.0 - cos_sim_iu
            top_right_rows = rows
            top_right_cols = cols + self.n_users
            bottom_left_rows = cols + self.n_users
            bottom_left_cols = rows
            final_rows = torch.cat([top_right_rows, bottom_left_rows])
            final_cols = torch.cat([top_right_cols, bottom_left_cols])
            final_data = torch.cat([weights_ui, weights_iu])
            size = self.n_users + self.n_items
            indices_all = torch.stack([final_rows, final_cols])
            ones_data = torch.ones_like(final_data)
            A_structure = torch.sparse_coo_tensor(indices_all, ones_data, (size, size), device=self.device)
            dense_ones = torch.ones((size, 1), device=self.device)
            degrees = torch.sparse.mm(A_structure, dense_ones).flatten()
            degrees = degrees.clamp(min=1e-7)
            d_inv_left = torch.pow(degrees, -self.r)
            d_inv_right = torch.pow(degrees, -(1 - self.r))
            row_d = d_inv_left[final_rows]
            col_d = d_inv_right[final_cols]
            norm_data = row_d * final_data * col_d
            self.nacc_norm_adj_mat = torch.sparse_coo_tensor(indices_all, norm_data, (size, size), device=self.device)

    def get_ego_embeddings(self):
        return torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)

    def forward(self):
        acc_all_embeddings = self.get_ego_embeddings()
        acc_embeddings_list = [acc_all_embeddings]
        nacc_all_embeddings = self.get_ego_embeddings()
        nacc_embeddings_list = [nacc_all_embeddings]
        for _ in range(self.n_layers):
            acc_all_embeddings = torch.sparse.mm(self.acc_norm_adj_mat, acc_all_embeddings)
            nacc_all_embeddings = torch.sparse.mm(self.nacc_norm_adj_mat, nacc_all_embeddings)
            acc_embeddings_list.append(acc_all_embeddings)
            nacc_embeddings_list.append(nacc_all_embeddings)
        lightgcn_acc_all_embeddings = torch.mean(torch.stack(acc_embeddings_list, dim=1), dim=1)
        lightgcn_nacc_all_embeddings = torch.mean(torch.stack(nacc_embeddings_list, dim=1), dim=1)
        u_acc, i_acc = torch.split(lightgcn_acc_all_embeddings, [self.n_users, self.n_items])
        u_nacc, i_nacc = torch.split(lightgcn_nacc_all_embeddings, [self.n_users, self.n_items])
        user_final = self.alpha * u_acc + (1 - self.alpha) * u_nacc
        item_final = self.alpha * i_acc + (1 - self.alpha) * i_nacc
        return lightgcn_acc_all_embeddings, lightgcn_nacc_all_embeddings, user_final, item_final

    def str_loss(self, acc_embedding, nacc_embedding, user, item):
        acc_u, acc_i = torch.split(acc_embedding, [self.n_users, self.n_items])
        nacc_u_all, nacc_i_all = torch.split(nacc_embedding, [self.n_users, self.n_items])
        u_acc_sub = F.normalize(acc_u[user])
        u_nacc_sub = F.normalize(nacc_u_all[user])
        u_nacc_all = F.normalize(nacc_u_all)
        pos_user = torch.exp(torch.mul(u_acc_sub, u_nacc_sub).sum(dim=1) / self.str_loss_temp)
        ttl_user = torch.exp(torch.matmul(u_acc_sub, u_nacc_all.T) / self.str_loss_temp).sum(dim=1)
        loss_u = -torch.log(pos_user / ttl_user).sum()
        i_acc_sub = F.normalize(acc_i[item])
        i_nacc_sub = F.normalize(nacc_i_all[item])
        i_nacc_all = F.normalize(nacc_i_all)
        pos_item = torch.exp(torch.mul(i_acc_sub, i_nacc_sub).sum(dim=1) / self.str_loss_temp)
        ttl_item = torch.exp(torch.matmul(i_acc_sub, i_nacc_all.T) / self.str_loss_temp).sum(dim=1)
        loss_i = -torch.log(pos_item / ttl_item).sum()
        return self.lambda1 * (self.str_loss_user_weight * loss_u + self.str_loss_item_weight * loss_i)

    def calculate_loss(self, user, pos, neg, epoch):
        acc_emb, nacc_emb, user_final, item_final = self.forward()
        str_loss = self.str_loss(acc_emb, nacc_emb, user, pos)
        u_emb = user_final[user]
        pos_emb = item_final[pos]
        neg_emb = item_final[neg]
        pos_scores = torch.mul(u_emb, pos_emb).sum(dim=1)
        neg_scores = torch.mul(u_emb, neg_emb).sum(dim=1)  # 只算一个
        mf_loss = self.mf_loss(pos_scores, neg_scores)
        u_ego = self.user_embedding(user)
        pos_ego = self.item_embedding(pos)
        neg_ego = self.item_embedding(neg)  # 只正则化这个负样本
        reg_loss = self.reg_loss(u_ego, pos_ego, neg_ego)
        return mf_loss + self.lambda3 * reg_loss, str_loss

    def pred(self, uids, is_tail_api, item_counts):
        _, _, user_final, item_final = self.forward()
        test_user_emb = user_final[uids]
        scores = torch.matmul(test_user_emb, item_final.T)
        if len(item_counts) < self.n_items:
            pad_width = self.n_items - len(item_counts)
            item_counts = np.pad(item_counts, (0, pad_width), 'constant', constant_values=0)
        log_n_users = np.log2(self.n_users)
        safe_counts = np.maximum(item_counts, 1)  # 避免 log(0)
        safe_counts = np.minimum(safe_counts, self.n_users)
        item_self_info = 1.0 - (np.log2(safe_counts) / log_n_users)
        item_self_info = np.clip(item_self_info, 0.0, 1.0)
        all_recall, all_ndcg, all_tail, all_nov = 0, 0, 0, 0
        coverage_api = set()
        for i in range(len(uids)):
            train_items_list = self.train_mapping.get(uids[i], [])
            if len(train_items_list) > 0:
                train_items = torch.tensor(train_items_list, dtype=torch.long, device=self.device)
                scores[i, train_items] = -float("inf")
        topk_values, topk_indices = torch.topk(scores, k=self.top_k, dim=1)
        topk_indices_list = topk_indices.tolist()
        topk_values_list = topk_values.tolist()
        for i in range(len(uids)):
            hit = 0
            hit_tail = 0
            user_nov = 0
            ground_truth = self.test_mapping[uids[i]]
            rec_indices = topk_indices_list[i]
            rec_scores = topk_values_list[i]
            binary_hit_list = []
            for pred_api in rec_indices:
                coverage_api.add(pred_api)
                if pred_api in ground_truth:
                    hit += 1
                    binary_hit_list.append(1)
                else:
                    binary_hit_list.append(0)
                if is_tail_api[pred_api] == 1:
                    hit_tail += 1
                user_nov += item_self_info[pred_api]
            if len(ground_truth) > 0:
                all_recall += hit / len(ground_truth)
            else:
                all_recall += 0
            all_ndcg += ndcg_at_k(rec_scores, binary_hit_list, self.top_k)
            all_tail += hit_tail / self.top_k
            all_nov += user_nov / self.top_k
        mean_nov = all_nov / len(uids)
        return all_recall / len(uids), all_ndcg / len(uids), coverage_api, all_tail / len(uids), mean_nov