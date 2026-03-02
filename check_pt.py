import torch
import os
import re
import math
from collections import defaultdict
# ================= Configuration =================
COLLECTIVE_DIR = "/home/sherma/Robot-manipulation-learning-via-cross-embodiments/logs/experiment_test/buffer/collective_buffer"
TRAIN_ROOT = os.path.join(COLLECTIVE_DIR, "train")
VAL_ROOT = os.path.join(COLLECTIVE_DIR, "validation")

EXPECTED_RATIO = 0.8
CHECK_CONTENT_LENGTH = True
# ===========================================

import torch
import mtenv # Check if editable package is connected

print(f"CUDA availability: {torch.cuda.is_available()}")

def get_files_in_dir(dir_path):
    """
    Get info for all .pt files matching the expected format in a directory.
    Returns dict: {start_step: {'filename': ..., 'end': ..., 'path': ...}}
    """
    if not os.path.exists(dir_path):
        return {}
    
    files_map = {}
    for f in os.listdir(dir_path):
        if not f.endswith(".pt"):
            continue
        
        # Parse filename e.g. 16000_19999.pt
        match = re.match(r'^(\d+)_(\d+)\.pt$', f)
        if match:
            start = int(match.group(1))
            end = int(match.group(2))
            files_map[start] = {
                'filename': f,
                'path': os.path.join(dir_path, f),
                'start': start,
                'end': end,
                'len': end - start + 1
            }
    return files_map

def get_real_len(path):
    try:
        payload = torch.load(path, map_location='cpu', weights_only=False)
        return payload[0].shape[0]
    except:
        return -1

def audit_multi_file_buffers():
    print(f"🔍 Starting MULTI-FILE Audit on: {COLLECTIVE_DIR}")
    print(f"   Logic: Pairing Train End+1 -> Val Start")
    print("=" * 120)

    if not os.path.exists(TRAIN_ROOT):
        print("❌ Train root directory not found.")
        return

    # Get all task directories
    task_dirs = sorted([d for d in os.listdir(TRAIN_ROOT) if os.path.isdir(os.path.join(TRAIN_ROOT, d))])
    
    total_issues = 0
    
    for task_dir in task_dirs:
        # Abbreviated name for display
        short_name = (task_dir[:50] + '..') if len(task_dir) > 50 else task_dir
        
        t_dir_path = os.path.join(TRAIN_ROOT, task_dir)
        v_dir_path = os.path.join(VAL_ROOT, task_dir)
        
        # 1. Read all files in this directory
        t_files = get_files_in_dir(t_dir_path)
        v_files = get_files_in_dir(v_dir_path) # key is start_step
        
        if not t_files and not v_files:
            continue # Skip empty directories

        issues_in_task = []
        
        # 2. Attempt pairing (Train -> Val)
        # Iterate over each Train file to find its matching Val counterpart
        # Pairing logic: Val.start == Train.end + 1
        
        # Track which Val files have been matched; the rest are orphans
        matched_val_starts = set()
        
        # Process Train files in chronological order
        for t_start in sorted(t_files.keys()):
            t_info = t_files[t_start]
            expected_val_start = t_info['end'] + 1
            
            # Find matching Val
            v_info = v_files.get(expected_val_start)
            
            status_msgs = []
            is_error = False
            
            # --- Check Train content ---
            if CHECK_CONTENT_LENGTH:
                real_l = get_real_len(t_info['path'])
                if real_l != t_info['len']:
                    status_msgs.append(f"T_SIZE_ERR({real_l})")
                    is_error = True

            if v_info:
                # === Pairing successful ===
                matched_val_starts.add(expected_val_start)
                
                # --- Check ratio ---
                total_len = t_info['len'] + v_info['len']
                ratio = t_info['len'] / total_len
                if not math.isclose(ratio, EXPECTED_RATIO, rel_tol=1e-9):
                    status_msgs.append(f"RATIO_ERR({ratio:.2f})")
                    is_error = True
                
                # --- Check Val content ---
                if CHECK_CONTENT_LENGTH:
                    real_l = get_real_len(v_info['path'])
                    if real_l != v_info['len']:
                        status_msgs.append(f"V_SIZE_ERR({real_l})")
                        is_error = True
                
                if not is_error:
                    # A perfect pair, no need to print unless you want verbose logs
                    # print(f"  OK: {t_info['filename']} -> {v_info['filename']}")
                    pass
                else:
                    # Paired but has errors
                    print(f"🔴 {short_name} | PAIR: {t_info['filename']} + {v_info['filename']} | {', '.join(status_msgs)}")
                    issues_in_task.append("Pair Error")

            else:
                # === Orphan Train (head without tail) ===
                # This could be a serious GAP, or it might just be the last file (with an 80-20 split every file should have a tail).
                # Since the strategy is intra-file split, every Train must have a corresponding Val.
                print(f"🔴 {short_name} | ORPHAN TRAIN: {t_info['filename']} (Expected val start: {expected_val_start})")
                issues_in_task.append(f"Orphan Train {t_info['filename']}")

        # 3. Check orphaned Val files (tail without head)
        for v_start in sorted(v_files.keys()):
            if v_start not in matched_val_starts:
                v_info = v_files[v_start]
                # Check if content is corrupted
                extra_msg = ""
                if CHECK_CONTENT_LENGTH:
                    real = get_real_len(v_info['path'])
                    if real != v_info['len']: extra_msg = f"SIZE_ERR({real})"
                
                print(f"🔴 {short_name} | ORPHAN VAL: {v_info['filename']} {extra_msg}")
                issues_in_task.append(f"Orphan Val {v_info['filename']}")

        # 4. Directory-level check
        if not os.path.exists(v_dir_path) and t_files:
             print(f"🔴 {short_name} | MISSING VAL DIR")
             issues_in_task.append("Missing Val Dir")

        if issues_in_task:
            total_issues += 1
            print("-" * 100)
    
    print("\n" + "="*30 + " SUMMARY " + "="*30)
    if total_issues == 0:
        print("✅ ALL FILES PASSED! Every train chunk has a matching val chunk with correct ratio.")
    else:
        print(f"🚨 Found issues in {total_issues} task folders.")

if __name__ == "__main__":
    audit_multi_file_buffers()