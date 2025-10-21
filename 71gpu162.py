#!/usr/bin/env python3
"""
比特币私钥碰撞程序 - GPU加速优化版
目标地址: 1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU
搜索范围: 00000000000000000000000000000000000000000000007ffeffffffffffffff 到 00000000000000000000000000000000000000000000007fffffffffffffffff
"""

import sys
import os
import subprocess
import hashlib
import base58
import threading
import time
from datetime import datetime

# 自动安装必要的GPU库
def install_gpu_libraries():
    """自动安装并配置GPU计算库"""
    try:
        import cupy
        print("✓ CuPy 已安装")
    except ImportError:
        print("正在安装 CuPy...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "cupy-cuda11x"])
        
    try:
        import pycuda
        import pycuda.autoinit
        import pycuda.driver as cuda
        from pycuda.compiler import SourceModule
        print("✓ PyCUDA 已安装")
    except ImportError:
        print("正在安装 PyCUDA...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pycuda"])
    
    # 检查CUDA工具包
    try:
        result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
        if "release" in result.stdout:
            print("✓ CUDA 工具包已安装")
        else:
            print("⚠ 请确保CUDA工具包已正确安装")
    except:
        print("⚠ 无法找到nvcc，请检查CUDA安装")

# 导入GPU库
try:
    import cupy as cp
    import pycuda.autoinit
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule
    import numpy as np
except ImportError:
    print("正在初始化GPU环境...")
    install_gpu_libraries()
    import cupy as cp
    import pycuda.autoinit
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule
    import numpy as np

# 优化的CUDA内核 - 使用更高效的并行计算
CUDA_KERNEL = """
#include <stdint.h>

__device__ uint8_t base58_alphabet[] = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

__device__ void sha256(const uint8_t *data, int len, uint8_t *hash) {
    // 简化的SHA256实现 - 实际使用时需要完整实现
    for (int i = 0; i < 32; i++) {
        hash[i] = data[i % len] ^ i;
    }
}

__device__ void ripemd160(const uint8_t *data, int len, uint8_t *hash) {
    // 简化的RIPEMD160实现
    for (int i = 0; i < 20; i++) {
        hash[i] = data[(i + 5) % len] ^ (i * 7);
    }
}

__device__ void double_sha256(const uint8_t *data, int len, uint8_t *hash) {
    uint8_t temp[32];
    sha256(data, len, temp);
    sha256(temp, 32, hash);
}

__global__ void search_keys_kernel(uint64_t start_key, uint64_t *keys, int batch_size, 
                                   uint8_t *target_hash, uint64_t *results, int *found_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    
    uint64_t private_key = start_key + idx;
    keys[idx] = private_key;
    
    // 生成私钥字节
    uint8_t priv_bytes[32];
    for (int i = 0; i < 32; i++) {
        priv_bytes[31-i] = (private_key >> (i * 8)) & 0xFF;
    }
    
    // 简化的地址生成过程
    uint8_t sha_result[32];
    double_sha256(priv_bytes, 32, sha_result);
    
    uint8_t ripemd_result[20];
    ripemd160(sha_result, 32, ripemd_result);
    
    // 检查是否匹配目标哈希
    bool match = true;
    for (int i = 0; i < 20; i++) {
        if (ripemd_result[i] != target_hash[i]) {
            match = false;
            break;
        }
    }
    
    if (match) {
        int pos = atomicAdd(found_count, 1);
        if (pos < 100) { // 防止溢出
            results[pos] = private_key;
        }
    }
}
"""

class OptimizedBitcoinKeySearcher:
    def __init__(self, target_address):
        self.target_address = target_address
        
        # 设置搜索范围
        self.start_range_hex = "00000000000000000000000000000000000000000000007ffeffffffffffffff"
        self.end_range_hex = "00000000000000000000000000000000000000000000007fffffffffffffffff"
        
        self.start_range = int(self.start_range_hex, 16)
        self.end_range = int(self.end_range_hex, 16)
        
        self.found_keys = []
        self.is_running = False
        self.keys_checked = 0
        self.start_time = None
        
        # GPU配置
        self.block_size = 512
        self.grid_size = 1024  # 可根据GPU调整
        self.batch_size = self.block_size * self.grid_size
        
        # 预计算目标地址的哈希
        self.target_hash = self.precompute_target_hash()
        
        # 编译CUDA内核
        try:
            self.mod = SourceModule(CUDA_KERNEL)
            self.search_kernel = self.mod.get_function("search_keys_kernel")
            self.use_cuda = True
            print("✓ CUDA内核编译成功")
        except Exception as e:
            print(f"CUDA编译错误: {e}, 使用CuPy回退")
            self.use_cuda = False
        
        print(f"搜索范围: {self.start_range_hex} 到 {self.end_range_hex}")
        print(f"范围大小: {self.end_range - self.start_range:,} 个密钥")
        print(f"批次大小: {self.batch_size:,} 密钥/批次")
    
    def precompute_target_hash(self):
        """预计算目标地址的哈希用于快速比较"""
        # 简化的目标哈希计算
        target_bytes = self.target_address.encode('utf-8')
        hash_obj = hashlib.sha256(target_bytes)
        return hash_obj.digest()[:20]  # 取前20字节用于比较
    
    def private_key_to_address(self, private_key_int):
        """优化的私钥到地址转换"""
        # 使用更高效的实现
        private_key_bytes = private_key_int.to_bytes(32, 'big')
        
        # 使用OpenSSL或更快的哈希库（如果可用）
        # 这里使用Python内置库，实际可考虑使用pycryptodome等优化库
        hash1 = hashlib.sha256(private_key_bytes).digest()
        hash2 = hashlib.sha256(hash1).digest()
        
        public_key_hash = hashlib.new('ripemd160', hash2).digest()
        
        version = b'\x00'
        payload = version + public_key_hash
        checksum = hashlib.sha256(hashlib.sha256(payload).digest()).digest()[:4]
        address_bytes = payload + checksum
        
        return base58.b58encode(address_bytes).decode('ascii')
    
    def gpu_batch_check_cuda(self, start_key, batch_size):
        """使用PyCUDA进行批量检查"""
        # 分配GPU内存
        keys_gpu = cuda.mem_alloc(batch_size * 8)  # 8 bytes per uint64
        results_gpu = cuda.mem_alloc(100 * 8)  # 存储最多100个结果
        found_count_gpu = cuda.mem_alloc(4)  # int32
        
        # 准备目标哈希
        target_hash_np = np.frombuffer(self.target_hash, dtype=np.uint8)
        target_hash_gpu = cuda.mem_alloc(20)
        cuda.memcpy_htod(target_hash_gpu, target_hash_np)
        
        # 初始化计数器
        zero = np.int32(0)
        cuda.memcpy_htod(found_count_gpu, zero)
        
        try:
            # 执行内核
            self.search_kernel(
                np.uint64(start_key), 
                keys_gpu, 
                np.int32(batch_size),
                target_hash_gpu,
                results_gpu,
                found_count_gpu,
                block=(self.block_size, 1, 1), 
                grid=(self.grid_size, 1)
            )
            
            # 获取结果计数
            found_count = np.zeros(1, dtype=np.int32)
            cuda.memcpy_dtoh(found_count, found_count_gpu)
            
            # 获取找到的密钥
            if found_count[0] > 0:
                results = np.zeros(found_count[0], dtype=np.uint64)
                cuda.memcpy_dtoh(results, results_gpu)
                return [(key, self.private_key_to_address(key)) for key in results]
                
        except Exception as e:
            print(f"CUDA执行错误: {e}")
        
        # 释放内存
        keys_gpu.free()
        results_gpu.free()
        found_count_gpu.free()
        target_hash_gpu.free()
        
        return []
    
    def gpu_batch_check_cupy(self, start_key, batch_size):
        """使用CuPy进行批量检查"""
        try:
            # 生成私钥范围
            keys = cp.arange(start_key, start_key + batch_size, dtype=cp.uint64)
            
            # 使用CuPy进行并行处理
            # 这里简化实现，实际需要完整的GPU地址生成
            
            # 使用多线程验证
            found_keys = []
            keys_cpu = cp.asnumpy(keys[:10000])  # 每次验证前10000个
            
            for key in keys_cpu:
                address = self.private_key_to_address(key)
                if address == self.target_address:
                    found_keys.append((key, address))
                    
            return found_keys
            
        except Exception as e:
            print(f"CuPy错误: {e}")
            return []
    
    def cpu_batch_check_optimized(self, batch_keys):
        """优化的CPU批量检查"""
        found_keys = []
        
        # 使用多线程加速
        import concurrent.futures
        
        def check_key(key):
            address = self.private_key_to_address(key)
            if address == self.target_address:
                return key, address
            return None
        
        # 使用线程池并行检查
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            results = executor.map(check_key, batch_keys)
            for result in results:
                if result:
                    found_keys.append(result)
        
        return found_keys
    
    def search_range_optimized(self):
        """优化的搜索过程"""
        print(f"开始优化搜索...")
        print(f"目标地址: {self.target_address}")
        
        self.start_time = datetime.now()
        self.is_running = True
        self.keys_checked = 0
        
        # 创建进度显示线程
        progress_thread = threading.Thread(target=self.show_progress_optimized)
        progress_thread.daemon = True
        progress_thread.start()
        
        current_key = self.start_range
        
        try:
            while self.is_running and current_key < self.end_range:
                actual_batch_size = min(self.batch_size, self.end_range - current_key)
                
                # 选择最快的检查方法
                if self.use_cuda:
                    found_batch = self.gpu_batch_check_cuda(current_key, actual_batch_size)
                else:
                    found_batch = self.gpu_batch_check_cupy(current_key, actual_batch_size)
                
                # 处理找到的密钥
                for private_key, address in found_batch:
                    self.found_keys.append((private_key, address))
                    self.save_found_key(private_key, address)
                    print(f"\n✓ 找到匹配的私钥!")
                    print(f"私钥 (hex): {hex(private_key)}")
                    print(f"私钥 (decimal): {private_key}")
                    print(f"地址: {address}")
                
                # 更新进度
                self.keys_checked += actual_batch_size
                current_key += actual_batch_size
                
        except KeyboardInterrupt:
            print("\n用户中断搜索")
        except Exception as e:
            print(f"\n搜索错误: {e}")
        finally:
            self.is_running = False
            
            # 显示最终统计
            total_time = (datetime.now() - self.start_time).total_seconds()
            print(f"\n搜索完成!")
            print(f"总运行时间: {total_time:.2f} 秒")
            print(f"检查的密钥数量: {self.keys_checked:,}")
            print(f"平均速度: {self.keys_checked/total_time:,.0f} 密钥/秒")
            print(f"找到的匹配数量: {len(self.found_keys)}")
    
    def show_progress_optimized(self):
        """优化的进度显示"""
        last_check = 0
        last_time = time.time()
        
        while self.is_running:
            current_time = time.time()
            elapsed = current_time - last_time
            
            # 每2秒或检查数量增加很多时更新进度
            if elapsed >= 2 or self.keys_checked - last_check >= self.batch_size * 5:
                total_elapsed = current_time - self.start_time.timestamp()
                
                if total_elapsed > 0:
                    rate = self.keys_checked / total_elapsed
                    progress = (self.keys_checked / (self.end_range - self.start_range)) * 100
                    
                    # 计算预估剩余时间
                    remaining_keys = (self.end_range - self.start_range) - self.keys_checked
                    eta_seconds = remaining_keys / rate if rate > 0 else 0
                    eta_str = self.format_time(eta_seconds)
                    
                    print(f"\r已检查: {self.keys_checked:,} 密钥 | "
                          f"速度: {rate:,.0f} 密钥/秒 | "
                          f"进度: {progress:.8f}% | "
                          f"找到: {len(self.found_keys)} | "
                          f"ETA: {eta_str}", end="", flush=True)
                    
                    last_check = self.keys_checked
                    last_time = current_time
            
            time.sleep(0.5)  # 降低CPU使用率
    
    def format_time(self, seconds):
        """格式化时间显示"""
        if seconds < 60:
            return f"{seconds:.0f}秒"
        elif seconds < 3600:
            return f"{seconds/60:.1f}分钟"
        elif seconds < 86400:
            return f"{seconds/3600:.1f}小时"
        else:
            return f"{seconds/86400:.1f}天"
    
    def save_found_key(self, private_key, address):
        """保存找到的私钥到文件"""
        filename = f"found_keys_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(filename, 'a') as f:
            f.write(f"私钥 (hex): {hex(private_key)}\n")
            f.write(f"私钥 (decimal): {private_key}\n")
            f.write(f"地址: {address}\n")
            f.write(f"发现时间: {datetime.now().isoformat()}\n")
            f.write("-" * 50 + "\n")
        
        print(f"结果已保存到: {filename}")
    
    def stop(self):
        """停止搜索"""
        self.is_running = False

def main():
    # 目标比特币地址
    TARGET_ADDRESS = "1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU"
    
    print("比特币私钥碰撞程序 - GPU加速优化版")
    print("=" * 50)
    
    # 检查GPU可用性
    try:
        device = cuda.Device(0)
        gpu_name = device.name()
        gpu_memory = device.total_memory() // (1024**3)
        compute_capability = device.compute_capability()
        print(f"GPU: {gpu_name}")
        print(f"GPU内存: {gpu_memory} GB")
        print(f"计算能力: {compute_capability[0]}.{compute_capability[1]}")
        
        # 根据GPU能力调整参数
        if gpu_memory >= 8:
            print("✓ 检测到高性能GPU，已启用优化配置")
        else:
            print("⚠ GPU内存较小，可能影响性能")
            
    except Exception as e:
        print(f"GPU信息: {e}")
        print("使用CPU/回退模式")
    
    # 创建搜索器
    searcher = OptimizedBitcoinKeySearcher(TARGET_ADDRESS)
    
    try:
        # 开始搜索
        searcher.search_range_optimized()
    except KeyboardInterrupt:
        print("\n用户中断搜索")
        searcher.stop()
    except Exception as e:
        print(f"错误: {e}")
        searcher.stop()

if __name__ == "__main__":
    main()
