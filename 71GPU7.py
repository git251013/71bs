import sys
import subprocess
import os

def install_required_packages():
    """自动安装必要的包"""
    required_packages = {
        'cupy-cuda11x': 'cupy',
        'pycuda': 'pycuda', 
        'base58': 'base58',
        'ecdsa': 'ecdsa'
    }
    
    for package, import_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"✓ {package} 已安装")
        except ImportError:
            print(f"正在安装 {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--quiet"])

install_required_packages()

import cupy as cp
import base58
import hashlib
import ecdsa
from ecdsa import SECP256k1
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

class GPUPrivateKeyHunter:
    def __init__(self, target_address):
        self.target_address = target_address
        self.found = False
        self.result_private_key = None
        self.result_address = None
        
    def private_key_to_address_cpu(self, private_key_hex):
        """CPU版本的私钥到地址转换（用于验证）"""
        try:
            # 转换为字节
            private_key_bytes = bytes.fromhex(private_key_hex)
            
            # 生成公钥
            sk = ecdsa.SigningKey.from_string(private_key_bytes, curve=SECP256k1)
            vk = sk.get_verifying_key()
            public_key_bytes = b'\x04' + vk.to_string()
            
            # SHA256
            sha256_b = hashlib.sha256(public_key_bytes).digest()
            
            # RIPEMD160
            ripemd160_b = hashlib.new('ripemd160', sha256_b).digest()
            
            # 添加版本字节
            versioned = b'\x00' + ripemd160_b
            
            # 计算校验和
            checksum = hashlib.sha256(hashlib.sha256(versioned).digest()).digest()[:4]
            
            # Base58编码
            binary_addr = versioned + checksum
            bitcoin_address = base58.b58encode(binary_addr)
            
            return bitcoin_address.decode('utf-8')
        except Exception as e:
            return None

    def setup_gpu_kernel(self):
        """设置GPU内核函数"""
        kernel_code = """
        #include <cupy/complex.cuh>
        
        extern "C" {
        
        __global__ void generate_private_keys(uint64_t* private_keys, uint64_t start, int batch_size) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < batch_size) {
                private_keys[idx] = start + idx;
            }
        }
        
        }
        """
        return SourceModule(kernel_code)

    def search_range_gpu_batch(self, start_hex, end_hex, batch_size=1000000):
        """使用GPU批量搜索私钥"""
        start_int = int(start_hex, 16)
        end_int = int(end_hex, 16)
        
        print(f"开始搜索范围: {start_hex} 到 {end_hex}")
        print(f"批次大小: {batch_size}")
        
        # 设置GPU内核
        mod = self.setup_gpu_kernel()
        generate_keys_kernel = mod.get_function("generate_private_keys")
        
        # 计算网格和块大小
        threads_per_block = 256
        blocks_per_grid = (batch_size + threads_per_block - 1) // threads_per_block
        
        current_position = start_int
        
        while current_position <= end_int and not self.found:
            # 计算当前批次的实际大小
            current_batch_size = min(batch_size, end_int - current_position + 1)
            
            if current_batch_size <= 0:
                break
            
            # 在GPU上分配内存
            private_keys_gpu = cp.zeros(current_batch_size, dtype=cp.uint64)
            
            # 执行GPU内核生成私钥
            generate_keys_kernel(
                private_keys_gpu, 
                cp.uint64(current_position),
                np.int32(current_batch_size),
                block=(threads_per_block, 1, 1),
                grid=(blocks_per_grid, 1)
            )
            
            # 将私钥传输到CPU进行处理
            private_keys_cpu = cp.asnumpy(private_keys_gpu)
            
            # 批量处理私钥
            for i in range(0, len(private_keys_cpu), 10000):  # 子批次处理
                if self.found:
                    break
                    
                end_idx = min(i + 10000, len(private_keys_cpu))
                batch_keys = private_keys_cpu[i:end_idx]
                
                for private_key_int in batch_keys:
                    if self.found:
                        break
                        
                    private_key_hex = format(private_key_int, '064x')
                    
                    # 生成地址并检查
                    address = self.private_key_to_address_cpu(private_key_hex)
                    
                    if address == self.target_address:
                        self.found = True
                        self.result_private_key = private_key_hex
                        self.result_address = address
                        
                        print(f"\n🎉 找到匹配的私钥!")
                        print(f"私钥: {private_key_hex}")
                        print(f"地址: {address}")
                        
                        # 保存结果
                        with open("found_private_key.txt", "w") as f:
                            f.write(f"私钥: {private_key_hex}\\n")
                            f.write(f"地址: {address}\\n")
                            f.write(f"WIF格式可能需要额外转换\\n")
                        
                        return True
            
            # 更新位置
            current_position += current_batch_size
            
            # 清理GPU内存
            del private_keys_gpu
            cp.get_default_memory_pool().free_all_blocks()
        
        return False

def main():
    # 目标地址
    target_address = "1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU"
    
    # 搜索范围
    start_range = "0000000000000000000000000000000000000000000000400000000000000000"
    end_range = "00000000000000000000000000000000000000000000007fffffffffffffffff"
    
    print("=" * 60)
    print("GPU比特币私钥搜索工具")
    print("=" * 60)
    print(f"目标地址: {target_address}")
    print(f"搜索范围: {start_range}")
    print(f"        到 {end_range}")
    print("=" * 60)
    
    # 检查GPU
    try:
        gpu_memory = cp.cuda.Device(0).mem_info
        print(f"GPU内存: {gpu_memory[0] // (1024**3)} GB 可用")
    except:
        print("警告: 无法获取GPU信息，可能影响性能")
    
    # 创建搜索器实例
    hunter = GPUPrivateKeyHunter(target_address)
    
    # 开始搜索
    import time
    start_time = time.time()
    
    try:
        found = hunter.search_range_gpu_batch(start_range, end_range)
        
        end_time = time.time()
        
        if found:
            print(f"\n✅ 搜索成功完成!")
            print(f"总耗时: {end_time - start_time:.2f} 秒")
        else:
            print(f"\n❌ 在指定范围内未找到匹配的私钥")
            print(f"搜索耗时: {end_time - start_time:.2f} 秒")
            
    except Exception as e:
        print(f"\n❌ 搜索过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
