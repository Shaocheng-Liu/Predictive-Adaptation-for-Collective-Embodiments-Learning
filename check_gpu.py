import torch
import time

def check_cuda_health():
    print("--- GPU 环境自检开始 ---")
    
    # 1. 基础可用性检查
    cuda_available = torch.cuda.is_available()
    print(f"1. CUDA 是否可用: {cuda_available}")
    
    if not cuda_available:
        print("   错误: PyTorch 无法识别到 GPU，请检查显卡驱动或 PyTorch 版本。")
        return

    # 2. 设备信息检查
    device_count = torch.cuda.device_count()
    current_device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(current_device)
    
    print(f"2. 检测到 GPU 数量: {device_count}")
    print(f"3. 当前正在使用的设备: {device_name} (ID: {current_device})")
    
    # 3. 架构兼容性验证
    # 4090 应该是 8.9，5060 应该是更高版本
    capability = torch.cuda.get_device_capability(current_device)
    print(f"4. 计算能力 (Compute Capability): {capability[0]}.{capability[1]}")

    # 4. 压力测试：执行大规模矩阵乘法
    # 模拟 Transformer 训练中的计算负载
    print("5. 正在进行张量计算测试...")
    try:
        start_time = time.time()
        # 创建两个大矩阵并移至 GPU
        a = torch.randn(5000, 5000).to("cuda")
        b = torch.randn(5000, 5000).to("cuda")
        # 执行乘法
        c = torch.matmul(a, b)
        # 强制同步以获取准确时间
        torch.cuda.synchronize()
        end_time = time.time()
        
        print(f"   计算测试成功！5000x5000 矩阵乘法耗时: {end_time - start_time:.4f} 秒")
        print(f"   结果张量位置: {c.device}")
    except Exception as e:
        print(f"   计算测试失败: {e}")

    print("--- 自检结束 ---")

if __name__ == "__main__":
    check_cuda_health()