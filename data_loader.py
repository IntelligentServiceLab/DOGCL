import scipy.sparse as sp
import numpy as np
import random
import os

# 保持随机种子一致性
random.seed(8080)


def get_lgn_data():
    mashup_id, api_id, interaction = [], [], []

    # 修改：使用更健壮的路径查找（优先查找当前目录，其次查找 dataset 目录）
    file_path = "train.txt"
    if not os.path.exists(file_path):
        file_path = "./data/gowalla/train.txt"  # 兼容你之前的路径
    if not os.path.exists(file_path):
        raise FileNotFoundError("Could not find train.txt in current directory or ./data/programmable-web/")

    print(f"Loading training data from: {file_path}")
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            items = line.split(' ')
            items = list(map(int, items))
            for item in items[1:]:
                mashup_id.append(items[0])
                api_id.append(item)
                interaction.append(1)

    # 动态计算矩阵大小，不依赖硬编码
    num_users = max(mashup_id) + 1
    num_items = max(api_id) + 1

    interaction_matrix = sp.coo_matrix((interaction, (mashup_id, api_id)), shape=(num_users, num_items))
    return interaction_matrix


def get_test_mapping():
    test_mapping = {}

    file_path = "test.txt"
    if not os.path.exists(file_path):
        file_path = "./data/gowalla/test.txt"
    if not os.path.exists(file_path):
        raise FileNotFoundError("Could not find test.txt")

    print(f"Loading test mapping from: {file_path}")  # 打印路径以确认
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            items = line.split(' ')
            items = list(map(int, items))
            # 第一个是用户ID，后面是测试集交互项
            test_mapping[items[0]] = items[1:]
    return test_mapping


def get_train_mapping():
    train_mapping = {}

    file_path = "train.txt"
    if not os.path.exists(file_path):
        file_path = "./data/gowalla/train.txt"

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            items = line.split(' ')
            items = list(map(int, items))
            train_mapping[items[0]] = items[1:]
    return train_mapping


if __name__ == '__main__':
    # 简单测试路径是否正确
    try:
        get_lgn_data()
        get_train_mapping()
        tm = get_test_mapping()
        print(f"Success! Test mapping loaded {len(tm)} users.")
    except Exception as e:
        print(f"Error: {e}")