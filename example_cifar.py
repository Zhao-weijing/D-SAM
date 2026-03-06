import argparse
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, SVHN, ImageNet
from timm.loss import LabelSmoothingCrossEntropy
from homura.vision.models.cifar_resnet import wrn28_2, wrn28_10, resnet20, resnet56, resnext29_32x4d
from model import *
from sam import SAM
from FriendlySAM import FriendlySAM
from D_FriendlySAM import D_FriendlySAM
from D_ASAM import D_ASAM
from D_SAM import D_SAM
from scipy.io import savemat
import os
import random
import numpy as np
import time
from adai import Adai

from torch.optim import SGD

# 设置随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_checkpoint_dir(directory):
    expanded_dir = os.path.expanduser(directory)
    if not os.path.exists(expanded_dir):
        os.makedirs(expanded_dir)
    
def add_label_noise(labels, noise_rate):
    """
    为数据集的标签添加噪声
    labels: 原始标签
    noise_rate: 标签噪声的比例（0-1之间的浮点数）
    """
    noisy_labels = labels.copy()
    num_samples = len(labels)
    num_noisy = int(noise_rate * num_samples)  # 计算需要添加噪声的标签数量

    # 随机选择要添加噪声的样本索引
    noisy_indices = np.random.choice(num_samples, num_noisy, replace=False)

    for idx in noisy_indices:
        current_label = labels[idx]
        # 生成除去当前正确标签的随机标签
        noisy_label = random.choice([i for i in range(10) if i != current_label])
        noisy_labels[idx] = noisy_label

    return noisy_labels

def load_data(dataset, batch_size=256, num_workers=2, noise_rate=0.0):
    if dataset == 'CIFAR10':
        data_loader = CIFAR10
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif dataset == 'CIFAR100':
        data_loader = CIFAR100
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif dataset == 'SVHN':
        data_loader = SVHN
        mean = (0.4377, 0.4438, 0.4728)
        std = (0.1980, 0.2010, 0.1970)
    elif dataset == 'ImageNet':
        data_loader = ImageNet
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    else:
        raise ValueError("Unsupported dataset. Choose from CIFAR10, CIFAR100, SVHN, ImageNet.")

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224) if dataset == 'ImageNet' else transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    test_transform = [
        transforms.Resize(256) if dataset == 'ImageNet' else transforms.ToTensor()
    ]
    if dataset == 'ImageNet':
        test_transform.append(transforms.CenterCrop(224))
    test_transform.append(transforms.Normalize(mean, std))
    test_transform = transforms.Compose(test_transform)

    if dataset == 'SVHN':
        train_set = data_loader(root='/data/', split='train', download=True, transform=train_transform)
        test_set = data_loader(root='/data/', split='test', download=True, transform=test_transform)
    else:
        train_set = data_loader(root='/data/', train=True, download=True, transform=train_transform)
        test_set = data_loader(root='/data/', train=False, download=True, transform=test_transform)

    # 为 CIFAR10 数据集添加标签噪声
    if dataset == 'CIFAR10' and noise_rate > 0.0:
        train_labels = np.array(train_set.targets)
        noisy_labels = add_label_noise(train_labels, noise_rate)
        train_set.targets = noisy_labels  # 将噪声标签应用到数据集
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader

def get_optimizer(model, opti_name, lr, momentum, weight_decay, rho=0.05, lmbda=0.1, sigma=1, gamma=0.9):
    if opti_name == 'SGD':
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif opti_name == 'AdamW':
        return torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.1)
    elif opti_name == 'SAM':
        base_optimizer = torch.optim.SGD
        return SAM(model.parameters(), base_optimizer, rho=rho, adaptive=None, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif opti_name == 'ASAM':
        base_optimizer = torch.optim.SGD
        return SAM(model.parameters(), base_optimizer, rho=rho, adaptive=True, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif opti_name == 'FriendlySAM':
        base_optimizer = torch.optim.SGD
        return FriendlySAM(model.parameters(), base_optimizer, rho=rho, sigma=sigma, lmbda=lmbda, adaptive=False, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif opti_name == 'D_FriendlySAM':
        base_optimizer = torch.optim.SGD
        return D_FriendlySAM(model.parameters(), base_optimizer, rho=rho, sigma=sigma, lmbda=lmbda, adaptive=True, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif opti_name == 'D_ASAM':
        return D_ASAM(model.parameters(), lr=lr, beta=momentum, rho=rho, weight_decay=weight_decay)
    elif opti_name == 'D_SAM':
        return D_SAM(model.parameters(), lr=lr, beta=momentum, rho=rho, weight_decay=weight_decay)
    # elif opti_name == 'D_SAM_ablation_1':
    #     return D_SAM_ablation_1(model.parameters(), lr=lr, beta=momentum, rho=rho, weight_decay=weight_decay)
    # elif opti_name == 'D_SAM_ablation_2':
    #     return D_SAM_ablation_2(model.parameters(), lr=lr, beta=momentum, rho=rho, weight_decay=weight_decay)
    elif opti_name == 'Adai':
        return Adai(model.parameters(), lr=lr, betas=(0.1, 0.99), eps=1e-03, weight_decay=weight_decay)
    else:
        raise ValueError("Unsupported optimizer.")

def train(args):
    train_loader, test_loader = load_data(args.dataset, batch_size=args.batch_size, noise_rate=args.noise_rate)
    num_classes = 10 if args.dataset in ['CIFAR10', 'SVHN'] else 100 if args.dataset == 'CIFAR100' else 1000
    model = eval(args.model)(num_classes=num_classes).cuda()
    
    optimizer = get_optimizer(model, args.optimizer, args.lr, args.momentum, args.weight_decay, args.rho, args.lmbda, args.sigma)

    # lambda_lr = lambda epoch: 0.5 ** (epoch // 30) 
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing) if args.smoothing else torch.nn.CrossEntropyLoss()

    best_accuracy = -1.0  # 设置初始值为-1.0
    optimizer_name = args.optimizer
    history = {optimizer_name + '_train_loss': [], optimizer_name + '_train_accuracy': [], optimizer_name + '_test_accuracy': []}
    # 在 epoch 循环的开始和结束添加时间记录
    start_time = time.time()
    for epoch in range(args.epochs):
        # 在每个 epoch 开始前记录开始时间
        epoch_start_time = time.time()
        model.train()
        train_loss = 0.
        train_accuracy = 0.
        cnt = 0.
        for inputs, targets in train_loader:
            inputs = inputs.cuda()
            targets = targets.cuda()

            optimizer.zero_grad()

            if args.optimizer == 'SAM' or args.optimizer == 'AS_SAM' or args.optimizer == 'ASAM' or args.optimizer == 'FriendlySAM' or args.optimizer == 'D_FriendlySAM' or args.optimizer == 'D_SAM':
                predictions = model(inputs)
                batch_loss = criterion(predictions, targets)
                batch_loss.mean().backward()
                optimizer.first_step(zero_grad=True)

                predictions = model(inputs)
                batch_loss = criterion(predictions, targets)
                batch_loss.mean().backward()
                optimizer.second_step(zero_grad=True)
            elif args.optimizer == 'D_ASAM':
                def closure():
                    predictions = model(inputs)
                    batch_loss = criterion(predictions, targets)
                    batch_loss.mean().backward()
                    return (batch_loss, predictions)
                batch_loss, predictions = optimizer.step(closure)
            else:
                predictions = model(inputs)
                batch_loss = criterion(predictions, targets)
                batch_loss.mean().backward()
                optimizer.step()

            with torch.no_grad():
                train_loss += batch_loss.sum().item()
                train_accuracy += (torch.argmax(predictions, 1) == targets).sum().item()
            cnt += len(targets)
        train_loss /= cnt
        train_accuracy *= 100. / cnt
        history[optimizer_name + '_train_loss'].append(train_loss)
        history[optimizer_name + '_train_accuracy'].append(train_accuracy)
        print(f"Epoch: {epoch}, Train accuracy: {train_accuracy:6.2f} %, Train loss: {train_loss:8.5f}")
        scheduler.step()

        model.eval()
        test_loss = 0.
        test_accuracy = 0.
        cnt = 0.
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.cuda()
                targets = targets.cuda()
                predictions = model(inputs)
                test_loss += criterion(predictions, targets).sum().item()
                test_accuracy += (torch.argmax(predictions, 1) == targets).sum().item()
                cnt += len(targets)
            test_loss /= cnt
            test_accuracy *= 100. / cnt
        history[optimizer_name + '_test_accuracy'].append(test_accuracy)
        # 计算并打印每个 epoch 的时间和测试准确率
        epoch_duration = time.time() - epoch_start_time
        remaining_epochs = args.epochs - (epoch + 1)
        estimated_time_left = remaining_epochs * epoch_duration
        print(f"Epoch: {epoch}, Test accuracy: {test_accuracy:6.2f} %, Epoch time: {epoch_duration:.2f} seconds, Estimated time left: {estimated_time_left / 60:.2f} minutes.")
        # 文件名中加入模型名、优化器、优化器参数、迭代次数、随机种子数等信息
        model_name = args.model
        
        optimizer_params = f"lr_{args.lr}_wd_{args.weight_decay}_mom_{args.momentum}"
        if args.optimizer in ['SGD', 'SAM', 'ASAM', 'D_SAM', 'D_ASAM', 'FriendlySAM', 'D_FriendlySAM']:
            optimizer_params = f"lr_{args.lr}_wd_{args.weight_decay}_mom_{args.momentum}_rho_{args.rho}"
        elif args.optimizer == 'AdamW':
            optimizer_params = f"lr_{args.lr}_wd_{args.weight_decay}"
        else:
            optimizer_params = f"lr_{args.lr}"
        seed = args.seed
        filename_base = f"{args.dataset}_{args.model}_batch_{args.batch_size}_{args.optimizer}_{optimizer_params}_epochs_{args.epochs}_seed_{args.seed}_noise_{args.noise_rate}"

        # 保存最佳模型
        if best_accuracy < test_accuracy:
            best_accuracy = test_accuracy
            best_model_path = os.path.join(os.path.expanduser(args.checkpoint_dir), f'best_{filename_base}.pth')
            state = {
                'state_dict': model.state_dict(),
            }
            torch.save(state, best_model_path)

    history_path = os.path.join(os.path.expanduser(args.checkpoint_dir), f'training_history_{filename_base}.mat')
    savemat(history_path, history)
    print(f"Best test accuracy: {best_accuracy}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0, type=int, help="Random seed.")
    parser.add_argument("--dataset", default='CIFAR10', type=str, help="CIFAR10, CIFAR100, SVHN or ImageNet.")
    parser.add_argument("--model", default='ResNet18', type=str, help="Name of model architecture")
    parser.add_argument("--optimizer", default='D_SAM', type=str, help="SGD, AdamW, SAM, ASAM, D_SAM or or Adai.")
    parser.add_argument("--lr", default=0.1, type=float, help="Initial learning rate.")
    parser.add_argument("--momentum", default=0.9, type=float, help="Momentum.")
    parser.add_argument("--weight_decay", default=5e-4, type=float, help="Weight decay factor.")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size")
    parser.add_argument("--epochs", default=200, type=int, help="Number of epochs.")
    parser.add_argument("--smoothing", default=0.1, type=float, help="Label smoothing.")
    parser.add_argument("--rho", default=0.5, type=float, help="Rho for SAM Variants.")
    parser.add_argument("--eta", default=0.0, type=float, help="Eta for ASAM.")
    parser.add_argument("--lmbda", default=0.9, type=float, help="Lambda for FriendlySAM.")
    parser.add_argument("--sigma", default=1, type=float, help="Sigma for FriendlySAM.")
    # parser.add_argument("--gamma", default=0.9, type=float, help="Gamma for WeightedSAM.")
    parser.add_argument("--checkpoint_dir", default='enter your checkpoint directory path', type=str, help="Directory to save checkpoints and history.")
    parser.add_argument("--noise_rate", default=0, type=float, help="Label noise rate.")
    args = parser.parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    set_seed(args.seed)  # 设置随机种子
    create_checkpoint_dir(args.checkpoint_dir)
    assert args.dataset in ['CIFAR10', 'CIFAR100', 'SVHN', 'ImageNet'], \
        "Invalid dataset. Please select CIFAR10, CIFAR100, SVHN or ImageNet"
    train(args)
