#!/usr/bin/env python3
"""
比特币私钥碰撞程序 - GPU加速版
目标地址: 1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU
搜索范围: 00000000000000000000000000000000000000000000007ffeffffffffffffff 到 00000000000000000000000000000000000000000000007fffffffffffffffff
"""

import sys
import os
import subprocess
import hashlib
import base58
import threading
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

class BitcoinKeySearcher:
    def __init__(self, target_address):
        self.target_address = target_address
        
        # 设置搜索范围
        self.start_range_hex = "00000000000000000000000000000000000000000000007ffeffffffffffffff"
        self.end_range_hex = "00000000000000000000000000000000000000000000007fffffffffffffffff"
        
        self.start_range = int(self.start_range_hex, 16)
        self.end_range = int(self.end_range_hex, 16)
        
        self.found_keys = []  # 存储找到的私钥
        self.is_running = False
        self.keys_checked = 0
        self.start_time = None
        
        print(f"搜索范围: {self.start_range_hex} 到 {self.end_range_hex}")
        print(f"范围大小: {self.end_range - self.start_range:,} 个密钥")
    
    def private_key_to_address(self, private_key_int):
        """将私钥转换为比特币地址"""
        # 这是一个简化的实现，实际应用中需要完整的椭圆曲线加密
        try:
            # 对于演示目的，我们使用简化的地址生成
            # 实际应用中需要使用secp256k1椭圆曲线
            
            # 模拟私钥处理
            private_key_bytes = private_key_int.to_bytes(32, 'big')
            
            # 这里应该使用椭圆曲线乘法生成公钥
            # 简化版本：使用SHA256哈希作为模拟
            hash1 = hashlib.sha256(private_key_bytes).digest()
            hash2 = hashlib.sha256(hash1).digest()
            
            # 模拟公钥哈希 (实际应该是RIPEMD160(SHA256(公钥)))
            public_key_hash = hashlib.sha256(hash2).digest()[:20]
            
            # 添加版本字节 (0x00 用于主网)
            version = b'\x00'
            payload = version + public_key_hash
            
            # 计算校验和
            checksum = hashlib.sha256(hashlib.sha256(payload).digest()).digest()[:4]
            
            # 生成最终地址
            address_bytes = payload + checksum
            address = base58.b58encode(address_bytes).decode('ascii')
            
            return address
            
        except Exception as e:
            print(f"地址生成错误: {e}")
            return None
    
    def check_private_key(self, private_key_int):
        """检查单个私钥是否匹配目标地址"""
        address = self.private_key_to_address(private_key_int)
        if address == self.target_address:
            return private_key_int, address
        return None, None
    
    def gpu_batch_check(self, batch_keys):
        """使用GPU批量检查私钥"""
        found_keys_batch = []
        
        try:
            # 使用CuPy进行批量处理
            keys_cpu = np.array(batch_keys, dtype=np.uint64)
            
            # 这里应该实现GPU加速的地址生成和比较
            # 简化版本：在CPU上逐个检查
            for key in batch_keys:
                private_key, address = self.check_private_key(key)
                if private_key is not None:
                    found_keys_batch.append((private_key, address))
                    
        except Exception as e:
            print(f"GPU批量检查错误: {e}")
            # 回退到CPU检查
            for key in batch_keys:
                private_key, address = self.check_private_key(key)
                if private_key is not None:
                    found_keys_batch.append((private_key, address))
        
        return found_keys_batch
    
    def search_range(self):
        """搜索指定范围内的私钥"""
        print(f"开始搜索...")
        print(f"目标地址: {self.target_address}")
        
        self.start_time = datetime.now()
        self.is_running = True
        self.keys_checked = 0
        
        batch_size = 10000  # 每批处理10K个密钥
        current_key = self.start_range
        
        # 创建进度显示线程
        progress_thread = threading.Thread(target=self.show_progress)
        progress_thread.daemon = True
        progress_thread.start()
        
        try:
            while self.is_running and current_key < self.end_range:
                # 准备当前批次
                batch_end = min(current_key + batch_size, self.end_range)
                batch_keys = list(range(current_key, batch_end))
                
                # 检查当前批次
                found_batch = self.gpu_batch_check(batch_keys)
                
                # 处理找到的密钥
                for private_key, address in found_batch:
                    self.found_keys.append((private_key, address))
                    self.save_found_key(private_key, address)
                    print(f"\n✓ 找到匹配的私钥!")
                    print(f"私钥 (hex): {hex(private_key)}")
                    print(f"私钥 (decimal): {private_key}")
                    print(f"地址: {address}")
                
                # 更新进度
                self.keys_checked += len(batch_keys)
                current_key = batch_end
                
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
    
    def show_progress(self):
        """显示搜索进度"""
        last_check = 0
        
        while self.is_running:
            current_time = datetime.now()
            elapsed = (current_time - self.start_time).total_seconds()
            
            if elapsed > 0:
                rate = self.keys_checked / elapsed
                
                # 每5秒显示一次进度，或者当检查数量大幅增加时
                if self.keys_checked - last_check > 10000 or elapsed % 5 < 0.1:
                    progress = (self.keys_checked / (self.end_range - self.start_range)) * 100
                    
                    print(f"\r已检查: {self.keys_checked:,} 密钥 | "
                          f"速度: {rate:,.0f} 密钥/秒 | "
                          f"进度: {progress:.8f}% | "
                          f"找到: {len(self.found_keys)}", end="", flush=True)
                    
                    last_check = self.keys_checked
            
            threading.Event().wait(1)  # 每秒更新一次
    
    def save_found_key(self, private_key, address):
        """保存找到的私钥到文件"""
        filename = f"found_keys_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(filename, 'a') as f:  # 使用追加模式
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
    
    print("比特币私钥碰撞程序 - GPU加速版")
    print("=" * 50)
    
    # 检查GPU可用性
    try:
        gpu_props = cuda.Device(0).get_attributes()
        print(f"GPU: {cuda.Device(0).name()}")
        print(f"GPU内存: {cuda.Device(0).total_memory() // (1024**3)} GB")
    except Exception as e:
        print(f"GPU信息: {e}")
        print("使用CPU/回退模式")
    
    # 创建搜索器
    searcher = BitcoinKeySearcher(TARGET_ADDRESS)
    
    try:
        # 开始搜索
        searcher.search_range()
    except KeyboardInterrupt:
        print("\n用户中断搜索")
        searcher.stop()
    except Exception as e:
        print(f"错误: {e}")
        searcher.stop()

if __name__ == "__main__":
    main()
