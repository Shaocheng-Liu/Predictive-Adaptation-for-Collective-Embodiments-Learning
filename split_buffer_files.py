import os
import shutil
from pathlib import Path
import argparse
import random
import torch

# Keep original logic: internally split a single file and rename based on step numbers
def split_single_pt_file(pt_path: Path, train_dir: Path, val_dir: Path, ratio: float):
    print(f"Splitting single file: {pt_path}")
    
    # 1. Load data
    payload = torch.load(pt_path, map_location='cpu', weights_only=False)
    length = payload[0].shape[0]
    
    # 2. Calculate split point
    split_idx = int(length * ratio)
    split_idx = (split_idx // 400) * 400  # round down to multiple of 400

    # 3. Parse step range from original filename (e.g. 0_19999 -> start=0, end=19999)
    # This is a great part of the code, ensuring filename traceability
    try:
        start, end = [int(x) for x in pt_path.stem.split("_")]
    except ValueError:
        # Fallback: if filename is not in standard format (e.g. buffer.pt)
        # In this case we can only use default logic, but in the pipeline filenames should all be standardized
        print(f"⚠️ Warning: Filename {pt_path.name} format unknown, using generic indices.")
        start = 0
        end = length - 1

    mid = start + split_idx

    # 4. Slice the data
    train_payload = [arr[:split_idx] for arr in payload]
    val_payload = [arr[split_idx:] for arr in payload]

    # 5. Construct new filenames (e.g. Train: 0_15999.pt, Val: 16000_19999.pt)
    train_out = train_dir / f"{start}_{mid - 1}.pt"
    val_out = val_dir / f"{mid}_{end}.pt"

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # 6. Save
    # Only save if the split data is non-empty
    if len(train_payload[0]) > 0:
        torch.save(train_payload, train_out)
    
    if len(val_payload[0]) > 0:
        torch.save(val_payload, val_out)
        
    print(f"✅ Saved split to:\n  {train_out}\n  {val_out}")

def split_buffer_files(source_dir, train_dir, val_dir, ratio=0.8, seed=42):
    source_path = Path(source_dir)
    # Process in order for cleaner log output
    pt_files = sorted(source_path.glob("*.pt"))
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    if not pt_files:
        print(f"❌ No .pt files found in {source_dir}")
        return

    print(f"🚀 Found {len(pt_files)} files. Executing Intra-File Split for ALL files...")

    # ================= Core modification area =================
    # Original logic: if len==1 split, else shuffle & copy
    # New logic: regardless of file count, loop through all and perform intra-file split
    
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