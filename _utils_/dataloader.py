import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader


def get_transform(dataset_name):
    """获取对应数据集的预处理转换"""
    if dataset_name == "mnist":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    elif dataset_name == "cifar10":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")


def load_dataset(dataset_name, data_dir="./data"):
    """加载指定数据集的训练集和测试集"""
    # 这里的 transforms 引用是多余的，已经在上面导入
    transform = get_transform(dataset_name)

    if dataset_name == "mnist":
        train_dataset = datasets.MNIST(
            data_dir, train=True, download=True, transform=transform
        )
        test_dataset = datasets.MNIST(
            data_dir, train=False, download=True, transform=transform
        )
    elif dataset_name == "cifar10":
        train_dataset = datasets.CIFAR10(
            data_dir, train=True, download=True, transform=transform
        )
        test_dataset = datasets.CIFAR10(
            data_dir, train=False, download=True, transform=transform
        )
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")

    return train_dataset, test_dataset


def split_iid(dataset, num_clients, batch_size):
    """IID数据划分"""
    total_size = len(dataset)
    indices = np.random.permutation(total_size)
    split_size = total_size // num_clients
    client_indices = [
        indices[i * split_size: (i + 1) * split_size].tolist()
        for i in range(num_clients)
    ]

    # 处理剩余数据
    remaining = indices[num_clients * split_size:]
    for i in range(len(remaining)):
        client_indices[i].append(remaining[i])

    # 创建数据加载器
    client_dataloaders = []
    for indices in client_indices:
        # 确保数据量不小于批次大小
        if len(indices) < batch_size:
            while len(indices) < batch_size:
                indices.extend(indices[:min(len(indices), batch_size - len(indices))])
        subset = Subset(dataset, indices)
        client_dataloaders.append(
            DataLoader(subset, batch_size=batch_size, shuffle=True)
        )

    return client_dataloaders


def split_noniid(dataset, num_clients, batch_size, dataset_name, alpha=0.1):
    """Non-IID数据划分（基于Dirichlet分布）"""
    if dataset_name == "mnist":
        targets = dataset.targets.numpy()
    elif dataset_name == "cifar10":
        targets = np.array(dataset.targets)
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")

    classes = np.unique(targets)
    client_indices = [[] for _ in range(num_clients)]
    class_indices = {c: np.where(targets == c)[0] for c in classes}

    for c in classes:
        # 为每个类别生成Dirichlet分布的客户端分配权重
        dirichlet_weights = np.random.dirichlet(np.ones(num_clients) * alpha)
        class_size = len(class_indices[c])
        # 按权重划分当前类别的数据
        # 防止 split 空数组报错
        if class_size > 0:
            split_points = np.cumsum(dirichlet_weights[:-1] * class_size).astype(int)
            client_split = np.split(class_indices[c], split_points)
            for i, idx in enumerate(client_split):
                client_indices[i].extend(idx.tolist())

    # 创建数据加载器
    client_dataloaders = []
    for indices in client_indices:
        # 即使某个客户端分到的数据很少，也要处理
        if len(indices) == 0:
            # 极端情况：分配少量随机数据防止报错
            indices = np.random.choice(len(dataset), batch_size).tolist()
        
        if len(indices) < batch_size:
            while len(indices) < batch_size:
                indices.extend(indices[:min(len(indices), batch_size - len(indices))])
        
        subset = Subset(dataset, indices)
        client_dataloaders.append(
            DataLoader(subset, batch_size=batch_size, shuffle=True)
        )

    return client_dataloaders


def get_dataloader(dataset_name, num_users, batch_size, non_iid=True, alpha=0.1, data_dir="./data"):
    """
    统一接口：加载并划分数据集
    (已重命名为 get_dataloader 并调整参数名以匹配 main.py)

    参数:
        dataset_name: 数据集名称
        num_users: 客户端数量 (对应 main.py 的 num_users)
        batch_size: 批次大小
        non_iid: 是否使用Non-IID划分 (对应 main.py 的 non_iid)
        alpha: Dirichlet参数
        data_dir: 数据路径
    """
    # 加载原始数据集
    train_dataset, test_dataset = load_dataset(dataset_name, data_dir)

    # 划分训练集
    if non_iid:
        client_dataloaders = split_noniid(
            train_dataset, num_users, batch_size, dataset_name, alpha
        )
    else:
        client_dataloaders = split_iid(
            train_dataset, num_users, batch_size
        )

    # 创建测试集加载器
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    return client_dataloaders, test_loader