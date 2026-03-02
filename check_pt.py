import torch
import os
import re
import math
from collections import defaultdict
# ================= 配置区域 =================
COLLECTIVE_DIR = "/{PROJECT_ROOT}/logs/experiment_test/buffer/collective_buffer"
TRAIN_ROOT = os.path.join(COLLECTIVE_DIR, "train")
VAL_ROOT = os.path.join(COLLECTIVE_DIR, "validation")

EXPECTED_RATIO = 0.8
CHECK_CONTENT_LENGTH = True
# ===========================================

import torch
import mtenv # 检查可编辑包是否连通

print(f"CUDA 可用性: {torch.cuda.is_available()}")

def get_files_in_dir(dir_path):
    """
    获取目录下所有符合格式的 .pt 文件信息
    返回字典: {start_step: {'filename': ..., 'end': ..., 'path': ...}}
    """
    if not os.path.exists(dir_path):
        return {}
    
    files_map = {}
    for f in os.listdir(dir_path):
        if not f.endswith(".pt"):
            continue
        
        # 解析文件名 16000_19999.pt
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

    # 获取所有任务文件夹
    task_dirs = sorted([d for d in os.listdir(TRAIN_ROOT) if os.path.isdir(os.path.join(TRAIN_ROOT, d))])
    
    total_issues = 0
    
    for task_dir in task_dirs:
        # 缩略名用于显示
        short_name = (task_dir[:50] + '..') if len(task_dir) > 50 else task_dir
        
        t_dir_path = os.path.join(TRAIN_ROOT, task_dir)
        v_dir_path = os.path.join(VAL_ROOT, task_dir)
        
        # 1. 读取该文件夹下所有文件
        t_files = get_files_in_dir(t_dir_path)
        v_files = get_files_in_dir(v_dir_path) # key 是 start_step
        
        if not t_files and not v_files:
            continue # 空文件夹跳过

        issues_in_task = []
        
        # 2. 尝试配对 (Train -> Val)
        # 我们遍历每一个 Train 文件，看能不能找到它的 Val 搭档
        # 配对逻辑: Val.start == Train.end + 1
        
        # 记录哪些 Val 文件被匹配过了，剩下的就是孤儿
        matched_val_starts = set()
        
        # 按时间顺序处理 Train 文件
        for t_start in sorted(t_files.keys()):
            t_info = t_files[t_start]
            expected_val_start = t_info['end'] + 1
            
            # 查找匹配的 Val
            v_info = v_files.get(expected_val_start)
            
            status_msgs = []
            is_error = False
            
            # --- 检查 Train 内容 ---
            if CHECK_CONTENT_LENGTH:
                real_l = get_real_len(t_info['path'])
                if real_l != t_info['len']:
                    status_msgs.append(f"T_SIZE_ERR({real_l})")
                    is_error = True

            if v_info:
                # === 配对成功 ===
                matched_val_starts.add(expected_val_start)
                
                # --- 检查 比例 ---
                total_len = t_info['len'] + v_info['len']
                ratio = t_info['len'] / total_len
                if not math.isclose(ratio, EXPECTED_RATIO, rel_tol=1e-9):
                    status_msgs.append(f"RATIO_ERR({ratio:.2f})")
                    is_error = True
                
                # --- 检查 Val 内容 ---
                if CHECK_CONTENT_LENGTH:
                    real_l = get_real_len(v_info['path'])
                    if real_l != v_info['len']:
                        status_msgs.append(f"V_SIZE_ERR({real_l})")
                        is_error = True
                
                if not is_error:
                    # 完美的一对，不需要打印，除非你想看详细日志
                    # print(f"  OK: {t_info['filename']} -> {v_info['filename']}")
                    pass
                else:
                    # 虽然配对但有错
                    print(f"🔴 {short_name} | PAIR: {t_info['filename']} + {v_info['filename']} | {', '.join(status_msgs)}")
                    issues_in_task.append("Pair Error")

            else:
                # === Train 落单 (有头无尾) ===
                # 这可能是严重的 GAP，也可能是这就是最后一个文件 (如果这是按 80-20 切分的，那么每个文件都应该有尾巴)
                # 既然你的策略是 intra-file split，那么 Train 必须有 Val 对应
                print(f"🔴 {short_name} | ORPHAN TRAIN: {t_info['filename']} (Expected val start: {expected_val_start})")
                issues_in_task.append(f"Orphan Train {t_info['filename']}")

        # 3. 检查落单的 Val 文件 (有尾无头)
        for v_start in sorted(v_files.keys()):
            if v_start not in matched_val_starts:
                v_info = v_files[v_start]
                # 检查内容是否损坏
                extra_msg = ""
                if CHECK_CONTENT_LENGTH:
                    real = get_real_len(v_info['path'])
                    if real != v_info['len']: extra_msg = f"SIZE_ERR({real})"
                
                print(f"🔴 {short_name} | ORPHAN VAL: {v_info['filename']} {extra_msg}")
                issues_in_task.append(f"Orphan Val {v_info['filename']}")

        # 4. 文件夹级检查
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