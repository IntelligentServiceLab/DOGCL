from model import DOGCL
import random
# 移除了 get_bert_emb
from data_loader import get_lgn_data, get_test_mapping, get_train_mapping
# 移除了 MLP
from utils import ModelConfig, TrnData, EarlyStopping
import torch
import torch.utils.data as data
from tqdm import tqdm
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = ModelConfig()

# 读取配置
n_users = config.n_users
n_items = config.n_items
latent_dim = config.latent_dim
n_layers = config.n_layers
str_loss_temp = config.str_loss_temp
# sem_loss_temp = config.sem_loss_temp # 已移除
lambda1 = config.lamdba1
# lambda2 = config.lamdba2 # 已移除
lambda3 = config.lamdba3
str_loss_user_weight = config.str_loss_user_weight
str_loss_item_weight = config.str_loss_item_weight
alpha = config.alpha
epochs = config.epochs
lr = config.lr
r = config.r
batch_size = config.batch_size
# sem_loss_item_weight = config.sem_loss_item_weight # 已移除
# sem_loss_user_weight = config.sem_loss_user_weight # 已移除
test_batch_size = config.test_batch_size
topk = config.topk
# gama = config.gama # 已移除

if __name__ == '__main__':

    train_mapping = get_train_mapping()
    test_mapping = get_test_mapping()
    interaction_matrix = get_lgn_data()

    max_train_user = interaction_matrix.shape[0]
    max_train_item = interaction_matrix.shape[1]

    max_test_user = 0
    max_test_item = 0
    for u, items in test_mapping.items():
        if u > max_test_user: max_test_user = u
        for i in items:
            if i > max_test_item: max_test_item = i

    n_users = max(max_train_user, max_test_user + 1)
    n_items = max(max_train_item, max_test_item + 1)

    print(f"Global Config: n_users={n_users}, n_items={n_items}")

    item_counts = np.array(interaction_matrix.sum(axis=0)).flatten()
    sorted_indices = np.argsort(item_counts)[::-1]  # 降序排列
    head_cutoff = int(n_items * 0.2)
    head_items = set(sorted_indices[:head_cutoff])
    is_tail_api = np.array([0 if i in head_items else 1 for i in range(n_items)])

    print(f"Tail items count: {sum(is_tail_api)} / {n_items}")

    model = DOGCL(
        n_users=n_users,
        n_items=n_items,
        latent_dim=latent_dim,
        n_layers=n_layers,
        str_loss_temp=str_loss_temp,
        lambda1=lambda1,
        lambda3=lambda3,
        r=r,
        str_loss_user_weight=str_loss_user_weight,
        str_loss_item_weight=str_loss_item_weight,
        alpha=alpha,
        interaction_matrix=interaction_matrix,
        device=device,
        test_mapping=test_mapping,
        train_mapping=train_mapping,
        topk=topk,
        epoch_num=epochs
    )
    model.to(device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_data = TrnData(interaction_matrix)
    train_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    early_stopping = EarlyStopping(patience=5000, min_delta=0.0001)

    for epoch in tqdm(range(1, epochs), total=epochs):

        if epoch < 20 or epoch % 5 == 0:
            model.update_geometric_nacc()

        train_loader.dataset.neg_sampling()

        for i, batch in enumerate(train_loader):
            uids, pos, neg = batch
            uids = uids.long().to(device)
            pos = pos.long().to(device)
            neg = neg.long().to(device)
            optimizer.zero_grad()
            main_loss, str_loss = model.calculate_loss(uids, pos, neg, epoch)
            total_loss = main_loss + str_loss
            total_loss.backward(retain_graph=True)
            optimizer.step()

        if (epoch % 5 == 0):
            model.eval()
            coverage_api = set()
            test_uids = np.array(list(test_mapping.keys()))
            batch_no = int(np.ceil(len(test_uids) / test_batch_size))

            all_recall = 0
            all_ndcg = 0
            all_coverage = 0
            all_tail = 0
            all_nov = 0

            with torch.no_grad():  # 建议加上 no_grad
                for batch in range(batch_no):
                    start = batch * test_batch_size
                    end = min((batch + 1) * test_batch_size, len(test_uids))

                    recall, ndcg, coverage, tail, nov = model.pred(test_uids[start:end], is_tail_api, item_counts)

                    all_recall += recall
                    all_ndcg += ndcg
                    all_tail += tail
                    all_nov += nov
                    coverage_api.update(coverage)

            all_recall = all_recall / batch_no
            all_ndcg = all_ndcg / batch_no
            all_tail = all_tail / batch_no
            all_nov = all_nov / batch_no
            all_coverage = len(coverage_api) / n_items

            print(
                f"epoch: {epoch}   Recall: {all_recall:.4f}   NDCG: {all_ndcg:.4f}   IC: {all_coverage:.4f}   Nov: {all_nov:.4f}   Tail: {all_tail:.4f}")

            if early_stopping(all_recall):
                print("Stopping training early!")
                break

    print("\nTraining finished! Extracting and saving final embeddings...")
    model.eval()  # 确保模型处于评估模式
    with torch.no_grad():
        _, _, _, item_final = model.forward()
        final_item_emb = item_final.cpu().numpy()
        np.save('DOGCL_item_emb.npy', final_item_emb)
        print(f"🎉 恭喜！Item Embeddings 成功保存！形状为: {final_item_emb.shape}")
    # =======================================================