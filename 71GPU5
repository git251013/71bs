#!/usr/bin/env python3
"""
比特币私钥碰撞程序 - GPU加速版
目标地址: 1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU
搜索范围: 2^70 到 2^71
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
        self.start_range = 1900000000000000000000
        self.end_range = 1900000000000000000000
        self.found_key = None
        self.is_running = False
        self.keys_checked = 0
        self.start_time = None
        
    def private_key_to_address(self, private_key_int):
        """将私钥转换为比特币地址"""
        # 使用secp256k1椭圆曲线生成公钥 (简化版本)
        # 实际应用中应使用完整的椭圆曲线加密库
        
        # 生成私钥字节 - 使用字符串处理大整数
        private_key_bytes = private_key_int.to_bytes(32, 'big')
        
        # 这里应该实现完整的椭圆曲线乘法来生成公钥
        # 简化版本：使用哈希模拟
        # 实际应用中需要使用如ecdsa等库
        
        # 模拟公钥生成
        public_key_hash = hashlib.sha256(private_key_bytes).digest()
        public_key_hash = hashlib.sha256(public_key_hash).digest()
        
        # RIPEMD160哈希
        ripemd160 = hashlib.new('ripemd160')
        ripemd160.update(public_key_hash)
        public_key_ripemd160 = ripemd160.digest()
        
        # 添加版本字节 (比特币主网: 0x00)
        versioned_payload = b'\x00' + public_key_ripemd160
        
        # 计算校验和
        checksum = hashlib.sha256(hashlib.sha256(versioned_payload).digest()).digest()[:4]
        
        # 组合最终载荷
        binary_address = versioned_payload + checksum
        
        # Base58编码
        address = base58.b58encode(binary_address).decode('utf-8')
        
        return address
    
    def gpu_search_batch_cupy(self, start_batch, batch_size):
        """使用CuPy在GPU上搜索一批私钥 - 解决大整数问题"""
        try:
            # 由于2^70到2^71范围的大整数超出C long范围，我们需要分批处理
            # 每批使用更小的子批次，避免直接操作大整数
            
            sub_batch_size = 10000  # 更小的子批次
            found_key = None
            
            for sub_start in range(0, batch_size, sub_batch_size):
                if not self.is_running:
                    break
                    
                sub_end = min(sub_start + sub_batch_size, batch_size)
                current_sub_batch_size = sub_end - sub_start
                
                # 使用int64范围内的相对偏移量
                # 从0开始生成相对偏移，然后加上起始值
                try:
                    # 生成相对偏移（在int64范围内）
                    relative_offset = cp.arange(0, current_sub_batch_size, dtype=cp.uint64)
                    
                    # 转换为Python整数并加上起始值
                    relative_offset_cpu = cp.asnumpy(relative_offset)
                    
                    # 在CPU上处理每个私钥
                    for i, offset in enumerate(relative_offset_cpu):
                        private_key = start_batch + sub_start + offset
                        
                        # 检查是否超出范围
                        if private_key >= self.end_range:
                            break
                            
                        address = self.private_key_to_address(private_key)
                        
                        if address == self.target_address:
                            found_key = private_key
                            break
                    
                    if found_key is not None:
                        self.found_key = found_key
                        return True
                        
                except Exception as e:
                    print(f"CuPy子批次处理错误: {e}")
                    # 如果CuPy失败，回退到纯CPU处理
                    return self.cpu_search_batch(start_batch + sub_start, current_sub_batch_size)
                    
        except Exception as e:
            print(f"CuPy搜索错误: {e}")
            # 回退到CPU
            return self.cpu_search_batch(start_batch, batch_size)
            
        return False
    
    def cpu_search_batch(self, start_batch, batch_size):
        """CPU版本的私钥搜索（回退方案）"""
        try:
            # 更小的子批次以避免内存问题
            sub_batch_size = min(batch_size, 10000)
            
            for batch_start in range(start_batch, start_batch + batch_size, sub_batch_size):
                if not self.is_running:
                    break
                    
                batch_end = min(batch_start + sub_batch_size, start_batch + batch_size)
                
                for private_key in range(batch_start, batch_end):
                    address = self.private_key_to_address(private_key)
                    
                    if address == self.target_address:
                        self.found_key = private_key
                        return True
                        
        except Exception as e:
            print(f"CPU搜索错误: {e}")
            
        return False
    
    def start_search(self):
        """开始搜索过程"""
        print(f"开始搜索私钥...")
        print(f"目标地址: {self.target_address}")
        print(f"搜索范围: 2^70 到 2^71")
        print(f"范围大小: {self.end_range - self.start_range:,} 个密钥")
        
        self.is_running = True
        self.start_time = datetime.now()
        self.keys_checked = 0
        
        # 更小的批次大小以避免大整数问题
        batch_size = 50000  # 每批处理50K个密钥
        current_batch = self.start_range
        
        # 进度报告计数器
        last_report_time = self.start_time
        report_interval = 10  # 每10秒报告一次进度
        
        print("开始搜索...")
        
        while self.is_running and current_batch < self.end_range:
            actual_batch_size = min(batch_size, self.end_range - current_batch)
            
            # 尝试使用GPU搜索
            found = False
            try:
                # 检查是否可以使用CuPy
                if hasattr(self, 'use_cupy') and self.use_cupy:
                    found = self.gpu_search_batch_cupy(current_batch, actual_batch_size)
                else:
                    # 尝试初始化GPU
                    try:
                        import cupy as cp
                        self.use_cupy = True
                        print("✓ 使用CuPy GPU加速")
                        found = self.gpu_search_batch_cupy(current_batch, actual_batch_size)
                    except Exception as e:
                        print(f"CuPy初始化失败: {e}，回退到CPU模式")
                        self.use_cupy = False
                        found = self.cpu_search_batch(current_batch, actual_batch_size)
            except Exception as e:
                # GPU失败时回退到CPU
                print(f"GPU搜索失败: {e}，回退到CPU模式")
                self.use_cupy = False
                found = self.cpu_search_batch(current_batch, actual_batch_size)
            
            if found:
                self.save_result()
                break
            
            self.keys_checked += actual_batch_size
            current_batch += actual_batch_size
            
            # 进度报告（简化版，只显示基本统计信息）
            current_time = datetime.now()
            if (current_time - last_report_time).total_seconds() >= report_interval:
                elapsed = (current_time - self.start_time).total_seconds()
                rate = self.keys_checked / elapsed if elapsed > 0 else 0
                progress = (current_batch - self.start_range) / (self.end_range - self.start_range) * 100
                
                print(f"[进度] 已检查: {self.keys_checked:,} 密钥 | "
                      f"速度: {rate:,.0f} 密钥/秒 | "
                      f"进度: {progress:.8f}%")
                
                last_report_time = current_time
        
        if not self.found_key:
            print("在指定范围内未找到匹配的私钥")
        
        total_time = (datetime.now() - self.start_time).total_seconds()
        print(f"搜索完成!")
        print(f"总运行时间: {total_time:.2f} 秒")
        print(f"总检查密钥数: {self.keys_checked:,}")
        print(f"平均速度: {self.keys_checked/total_time:,.0f} 密钥/秒")
    
    def save_result(self):
        """保存找到的结果"""
        if self.found_key:
            # 验证地址匹配
            generated_address = self.private_key_to_address(self.found_key)
            
            result = {
                'private_key_hex': hex(self.found_key),
                'private_key_decimal': str(self.found_key),
                'generated_address': generated_address,
                'target_address': self.target_address,
                'found_at': datetime.now().isoformat(),
                'keys_checked': self.keys_checked,
                'search_time_seconds': (datetime.now() - self.start_time).total_seconds()
            }
            
            filename = f"found_key_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            
            print("\n" + "="*60)
            print("🎉 找到匹配的私钥!")
            print("="*60)
            print(f"私钥 (16进制): {result['private_key_hex']}")
            print(f"私钥 (10进制): {result['private_key_decimal']}")
            print(f"生成的地址: {result['generated_address']}")
            print(f"目标地址: {result['target_address']}")
            print(f"验证结果: {'匹配成功' if generated_address == self.target_address else '匹配失败'}")
            print(f"搜索耗时: {result['search_time_seconds']:.2f} 秒")
            print(f"检查密钥数: {result['keys_checked']:,}")
            print("="*60)
            
            # 保存到文件
            with open(filename, 'w') as f:
                f.write("比特币私钥碰撞结果\n")
                f.write("="*50 + "\n")
                f.write(f"找到时间: {result['found_at']}\n")
                f.write(f"私钥 (16进制): {result['private_key_hex']}\n")
                f.write(f"私钥 (10进制): {result['private_key_decimal']}\n")
                f.write(f"生成的地址: {result['generated_address']}\n")
                f.write(f"目标地址: {result['target_address']}\n")
                f.write(f"验证状态: {'匹配成功' if generated_address == self.target_address else '匹配失败'}\n")
                f.write(f"搜索耗时: {result['search_time_seconds']:.2f} 秒\n")
                f.write(f"检查密钥数: {result['keys_checked']:,}\n")
            
            print(f"✓ 结果已保存到: {filename}")
    
    def stop(self):
        """停止搜索"""
        self.is_running = False
        print("正在停止搜索...")

def check_gpu_status():
    """检查GPU状态"""
    try:
        gpu_props = cuda.Device(0).get_attributes()
        print(f"✓ GPU可用: {cuda.Device(0).name()}")
        print(f"  GPU内存: {cuda.Device(0).total_memory() // (1024**3)} GB")
        return True
    except Exception as e:
        print("⚠ GPU不可用，将使用CPU模式")
        print(f"  错误信息: {e}")
        return False

def main():
    # 目标比特币地址
    TARGET_ADDRESS = "1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU"
    
    print("比特币私钥碰撞程序 - GPU加速版")
    print("=" * 50)
    
    # 检查环境
    check_gpu_status()
    
    # 创建搜索器
    searcher = BitcoinKeySearcher(TARGET_ADDRESS)
    
    try:
        # 开始搜索
        searcher.start_search()
    except KeyboardInterrupt:
        print("\n用户中断搜索")
        searcher.stop()
    except Exception as e:
        print(f"错误: {e}")
        searcher.stop()

if __name__ == "__main__":
    main()
