import argparse
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
import os
from model import ResNet18
import time

# 设置随机种子
def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# CIFAR-10-C数据集类
class CIFAR10C(Dataset):
    def __init__(self, root, corruption_type=None, severity=None, transform=None):
        """
        CIFAR-10-C数据集加载器
        
        Args:
            root: CIFAR-10-C数据集根目录
            corruption_type: 失真类型，如果为None则加载所有类型
            severity: 失真强度 (1-5)，如果为None则加载所有强度
            transform: 数据变换
        """
        self.root = root
        self.transform = transform
        self.data = []
        self.targets = []
        
        # CIFAR-10-C的失真类型列表
        corruption_types = [
            'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
            'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
            'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
        ]
        
        # 加载标签文件（所有失真类型共享相同的标签）
        labels_path = os.path.join(root, 'labels.npy')
        if os.path.exists(labels_path):
            all_labels = np.load(labels_path)
        else:
            raise FileNotFoundError(f"找不到标签文件: {labels_path}")
        
        # 如果指定了corruption_type，只加载该类型
        if corruption_type is not None:
            corruption_types = [corruption_type]
        
        # 如果指定了severity，只加载该强度
        if severity is not None:
            severities = [severity]
        else:
            severities = [1, 2, 3, 4, 5]
        
        # 加载数据
        # CIFAR-10-C数据格式: 每个corruption类型有一个.npy文件
        # 每个文件包含50000张图像 (5个强度级别 × 10000张图像)
        # 数据形状: [50000, 32, 32, 3]
        for corr_type in corruption_types:
            data_path = os.path.join(root, f'{corr_type}.npy')
            if os.path.exists(data_path):
                images = np.load(data_path)
                # 每个强度级别有10000张图像
                for sev in severities:
                    start_idx = (sev - 1) * 10000
                    end_idx = sev * 10000
                    images_sev = images[start_idx:end_idx]
                    labels_sev = all_labels[start_idx:end_idx]
                    
                    self.data.append(images_sev)
                    self.targets.append(labels_sev)
            else:
                print(f"警告: 找不到数据文件 {data_path}")
        
        if len(self.data) > 0:
            self.data = np.concatenate(self.data, axis=0)
            self.targets = np.concatenate(self.targets, axis=0)
        else:
            raise ValueError("未找到任何数据文件")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        # 确保图像数据在正确的范围内 [0, 255]
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
        img = Image.fromarray(img)
        
        if self.transform:
            img = self.transform(img)
        
        return img, int(target)

def load_cifar10c_data(root, batch_size=256, num_workers=2, corruption_type=None, severity=None):
    """
    加载CIFAR-10-C数据集
    
    Args:
        root: CIFAR-10-C数据集根目录
        batch_size: 批次大小
        num_workers: 数据加载器工作进程数
        corruption_type: 失真类型，None表示加载所有类型
        severity: 失真强度，None表示加载所有强度
    """
    # CIFAR-10的均值和标准差
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    
    # 测试时的数据变换（与训练时一致）
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    dataset = CIFAR10C(root=root, corruption_type=corruption_type, severity=severity, transform=test_transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return dataloader

def test_robustness(args):
    """
    在CIFAR-10-C数据集上进行鲁棒性测试
    """
    # 设置随机种子
    set_seed(args.seed)
    
    # 加载模型
    print(f"加载模型: {args.model}")
    num_classes = 10  # CIFAR-10有10个类别
    model = ResNet18(num_classes=num_classes).cuda()
    
    # 加载checkpoint
    print(f"加载checkpoint: {args.checkpoint_path}")
    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"找不到checkpoint文件: {args.checkpoint_path}")
    
    checkpoint = torch.load(args.checkpoint_path, map_location='cuda')
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print("模型加载完成")
    
    # CIFAR-10-C的失真类型列表
    corruption_types = [
        'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
        'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
        'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
    ]
    
    # 存储结果
    results = {}
    
    # 测试每个失真类型和强度
    print("\n开始鲁棒性测试...")
    print("=" * 80)
    
    for corr_type in corruption_types:
        print(f"\n测试失真类型: {corr_type}")
        print("-" * 80)
        corr_results = {}
        
        for severity in range(1, 6):
            try:
                # 加载该失真类型和强度的数据
                test_loader = load_cifar10c_data(
                    root=args.data_root,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    corruption_type=corr_type,
                    severity=severity
                )
                
                # 测试
                correct = 0
                total = 0
                start_time = time.time()
                
                with torch.no_grad():
                    for inputs, targets in test_loader:
                        inputs = inputs.cuda()
                        targets = targets.cuda()
                        outputs = model(inputs)
                        _, predicted = outputs.max(1)
                        total += targets.size(0)
                        correct += predicted.eq(targets).sum().item()
                
                accuracy = 100. * correct / total
                elapsed_time = time.time() - start_time
                corr_results[severity] = accuracy
                
                print(f"  强度 {severity}: 准确率 = {accuracy:.2f}% ({correct}/{total}), 耗时 = {elapsed_time:.2f}秒")
                
            except Exception as e:
                print(f"  强度 {severity}: 错误 - {str(e)}")
                corr_results[severity] = None
        
        # 计算该失真类型的平均准确率
        valid_results = [v for v in corr_results.values() if v is not None]
        if valid_results:
            avg_accuracy = np.mean(valid_results)
            print(f"  平均准确率: {avg_accuracy:.2f}%")
            results[corr_type] = {
                'severities': corr_results,
                'average': avg_accuracy
            }
        else:
            results[corr_type] = {
                'severities': corr_results,
                'average': None
            }
    
    # 计算总体平均准确率
    print("\n" + "=" * 80)
    print("测试结果汇总:")
    print("=" * 80)
    
    all_averages = [r['average'] for r in results.values() if r['average'] is not None]
    if all_averages:
        overall_accuracy = np.mean(all_averages)
        print(f"\n总体平均准确率: {overall_accuracy:.2f}%")
        
        # 按失真类型打印结果
        print("\n各失真类型平均准确率:")
        for corr_type, result in sorted(results.items()):
            if result['average'] is not None:
                print(f"  {corr_type:25s}: {result['average']:6.2f}%")
    
    # 保存结果为CSV文件
    import csv
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 生成CSV文件名（包含checkpoint信息）
    checkpoint_name = os.path.basename(args.checkpoint_path).replace('.pth', '')
    csv_filename = f'robustness_results_{checkpoint_name}.csv'
    results_path = os.path.join(args.output_dir, csv_filename)
    
    # 写入CSV文件
    with open(results_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # 写入表头
        writer.writerow(['失真类型', '强度1', '强度2', '强度3', '强度4', '强度5', '平均准确率'])
        
        # 写入每个失真类型的结果
        for corr_type in sorted(results.keys()):
            result = results[corr_type]
            row = [corr_type]
            
            # 添加每个强度级别的准确率
            for severity in range(1, 6):
                acc = result['severities'].get(severity)
                if acc is not None:
                    row.append(f'{acc:.2f}')
                else:
                    row.append('N/A')
            
            # 添加平均准确率
            if result['average'] is not None:
                row.append(f'{result["average"]:.2f}')
            else:
                row.append('N/A')
            
            writer.writerow(row)
        
        # 写入总体平均准确率
        if all_averages:
            writer.writerow(['总体平均', '', '', '', '', '', f'{overall_accuracy:.2f}'])
    
    print(f"\n结果已保存到CSV文件: {results_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CIFAR-10-C鲁棒性测试')
    parser.add_argument("--seed", default=0, type=int, help="随机种子")
    parser.add_argument("--model", default='ResNet18', type=str, help="模型架构名称")
    parser.add_argument("--checkpoint_path", 
                       default='enter your checkpoint path here',
                       type=str, help="模型checkpoint路径") 
    parser.add_argument("--data_root", 
                       default='enter your CIFAR-10-C dataset root directory path here',
                       type=str, help="CIFAR-10-C数据集根目录")
    parser.add_argument("--batch_size", default=128, type=int, help="批次大小")
    parser.add_argument("--num_workers", default=2, type=int, help="数据加载器工作进程数")
    parser.add_argument("--save_results", action='store_true', help="是否保存测试结果（默认保存为CSV文件）")
    parser.add_argument("--output_dir", default='enter your output directory path here', type=str, help="结果保存目录")
    
    args = parser.parse_args()
    
    # 展开路径
    args.checkpoint_path = os.path.expanduser(args.checkpoint_path)
    args.data_root = os.path.expanduser(args.data_root)
    args.output_dir = os.path.expanduser(args.output_dir)
    
    test_robustness(args)

