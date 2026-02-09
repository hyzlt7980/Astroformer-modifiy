import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import timm
import os
import argparse
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def setup_ddp():
    """初始化分布式训练环境"""
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError("请使用 torchrun 启动脚本以支持 DDP")
    
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    
    return local_rank

def cleanup_ddp():
    dist.destroy_process_group()

def reduce_tensor(tensor):
    """用于多卡 Loss 打印时的平均化"""
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt

def main():
    # 1. DDP 初始化
    local_rank = setup_ddp()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", local_rank)

    if rank == 0:
        print(f"==> DDP Initialized. World Size: {world_size}")

    # 2. 数据准备
    stats = ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))

    transform_train = transforms.Compose([
        transforms.Resize(64),
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.TrivialAugmentWide(), # 指定的数据增强
        transforms.ToTensor(),
        transforms.Normalize(*stats),
    ])

    #cutmix,mixup,dropout,labelsmotthing, SWA
    #/ regularization正则，对抗过拟合overfittng

    transform_test = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize(*stats),
    ])

    # Rank 0 负责下载数据
    if rank == 0:
        if not os.path.exists('./data'):
            os.makedirs('./data')
        torchvision.datasets.CIFAR100(root='./data', train=True, download=True)
        torchvision.datasets.CIFAR100(root='./data', train=False, download=True)
    
    dist.barrier() # 等待数据下载完成

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, 
                                           download=False, transform=transform_train)
    train_sampler = DistributedSampler(trainset, shuffle=True)
    
    # Batch Size: 每张卡 128
    batch_size_per_gpu = 128
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_per_gpu, 
                                            sampler=train_sampler, num_workers=4, pin_memory=True)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False, 
                                          download=False, transform=transform_test)
    test_sampler = DistributedSampler(testset, shuffle=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, 
                                           sampler=test_sampler, num_workers=4, pin_memory=True)

    # 3. 模型定义 (纯净 Swin-Tiny)
    if rank == 0:
        print(f"==> Building Swin-Tiny (img_size=64, window_size=4)...")

    model = timm.create_model(
        'swin_tiny_patch4_window7_224',
        pretrained=False,
        num_classes=100,
        img_size=64,       # 强制修改分辨率
        window_size=4,     # 强制修改窗口大小
        drop_path_rate=0.1 # 保持默认防过拟合
    )
    
    model = model.to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    criterion = nn.CrossEntropyLoss().to(device)

    # 4. 优化器设置 (修复点：固定 LR)
    # 之前自动缩放导致 LR=2e-3 太大，现在固定为 5e-4
    base_lr = 5e-4
    
    if rank == 0:
        print(f"==> Learning Rate Fixed to: {base_lr} (Safe for training from scratch)")

    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.05)
    
    # Cosine Annealing 调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)

    # 5. 训练循环
    best_acc = 0.0

    for epoch in range(300):
        train_sampler.set_epoch(epoch)
        
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # 记录 Loss
            reduced_loss = reduce_tensor(loss.data)
            train_loss += reduced_loss.item()
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if rank == 0 and batch_idx % 50 == 0:
                print(f'[Epoch {epoch+1}] Step {batch_idx}/{len(trainloader)} | Loss: {reduced_loss.item():.4f} | LR: {optimizer.param_groups[0]["lr"]:.2e}')

        # Validation
        model.eval()
        correct_val = torch.tensor(0.0).to(device)
        total_val = torch.tensor(0.0).to(device)
        
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                
                total_val += targets.size(0)
                correct_val += predicted.eq(targets).sum()

        # 汇总所有卡的结果
        dist.all_reduce(correct_val, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_val, op=dist.ReduceOp.SUM)

        acc = 100. * correct_val / total_val

        # 仅 Rank 0 保存和打印结果
        if rank == 0:
            avg_loss = train_loss / len(trainloader)
            print(f"==> Epoch {epoch+1} Finished | Avg Loss: {avg_loss:.4f} | Validation Acc: {acc:.2f}%")
            if acc > best_acc:
                print("==> 🏆 New Best Model Saved!")
                best_acc = acc
                # torch.save(model.module.state_dict(), 'swin_t_pure_baseline.pth')

        scheduler.step()

    cleanup_ddp()

if __name__ == '__main__':
    main()
