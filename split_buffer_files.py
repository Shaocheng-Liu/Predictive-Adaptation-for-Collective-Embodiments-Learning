import os
import shutil
from pathlib import Path
import argparse
import random
import torch

# 保持你原有的逻辑：内部切分单个文件，并根据步数重命名
def split_single_pt_file(pt_path: Path, train_dir: Path, val_dir: Path, ratio: float):
    print(f"Splitting single file: {pt_path}")
    
    # 1. 加载数据
    payload = torch.load(pt_path, map_location='cpu', weights_only=False)
    length = payload[0].shape[0]
    
    # 2. 计算切分点
    split_idx = int(length * ratio)
    split_idx = (split_idx // 400) * 400  # round down to multiple of 400

    # 3. 解析原文件名中的步数范围 (例如 0_19999 -> start=0, end=19999)
    # 这是你代码中非常好的部分，保证了文件名的可追溯性
    try:
        start, end = [int(x) for x in pt_path.stem.split("_")]
    except ValueError:
        # 容错处理：如果文件名不是 standard format (比如是 buffer.pt)
        # 这种情况下只能用默认逻辑，但在你的pipeline里应该都是规范命名的
        print(f"⚠️ Warning: Filename {pt_path.name} format unknown, using generic indices.")
        start = 0
        end = length - 1

    mid = start + split_idx

    # 4. 数据切片
    train_payload = [arr[:split_idx] for arr in payload]
    val_payload = [arr[split_idx:] for arr in payload]

    # 5. 构造新的文件名 (例如 Train: 0_15999.pt, Val: 16000_19999.pt)
    train_out = train_dir / f"{start}_{mid - 1}.pt"
    val_out = val_dir / f"{mid}_{end}.pt"

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # 6. 保存
    # 只有当切分后的数据不为空时才保存
    if len(train_payload[0]) > 0:
        torch.save(train_payload, train_out)
    
    if len(val_payload[0]) > 0:
        torch.save(val_payload, val_out)
        
    print(f"✅ Saved split to:\n  {train_out}\n  {val_out}")

def split_buffer_files(source_dir, train_dir, val_dir, ratio=0.8, seed=42):
    source_path = Path(source_dir)
    # 按顺序处理，保证日志输出好看
    pt_files = sorted(source_path.glob("*.pt"))
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    if not pt_files:
        print(f"❌ No .pt files found in {source_dir}")
        return

    print(f"🚀 Found {len(pt_files)} files. Executing Intra-File Split for ALL files...")

    # ================= 修改的核心区域 =================
    # 原逻辑：if len==1 split, else shuffle & copy
    # 新逻辑：无论多少文件，全部循环执行内部切分
    
    for f in pt_files:
        split_single_pt_file(f, Path(train_dir), Path(val_dir), ratio)
    
    # ===============================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    split_buffer_files(args.source, args.train, args.val, args.ratio, args.seed)