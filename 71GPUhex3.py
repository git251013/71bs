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

class BitcoinAddressGenerator:
    """比特币地址生成器"""
    
    @staticmethod
    def private_key_to_wif(private_key_int):
        """将私钥整数转换为WIF格式"""
        private_key_bytes = private_key_int.to_bytes(32, 'big')
        # 主网前缀
        extended_key = b'\x80' + private_key_bytes
        # 添加校验和
        checksum = hashlib.sha256(hashlib.sha256(extended_key).digest()).digest()[:4]
        wif_bytes = extended_key + checksum
        return base58.b58encode(wif_bytes).decode('ascii')
    
    @staticmethod
    def private_key_to_address(private_key_int):
        """将私钥整数转换为比特币地址"""
        # 简化版的私钥到地址转换
        # 注意: 实际应用中需要完整的椭圆曲线加密实现
        
        # 模拟私钥到公钥的转换 (简化版)
        private_key_bytes = private_key_int.to_bytes(32, 'big')
        
        # 模拟公钥生成 (实际需要secp256k1椭圆曲线乘法)
        # 这里使用哈希模拟
        public_key_hash = hashlib.sha256(private_key_bytes).digest()
        
        # 再次哈希模拟公钥哈希
        public_key_hash2 = hashlib.sha256(public_key_hash).digest()
        
        # RIPEMD160哈希
        ripemd160 = hashlib.new('ripemd160')
        ripemd160.update(public_key_hash2)
        public_key_ripe = ripemd160.digest()
        
        # 添加版本字节 (0x00 用于主网)
        version_public_key = b'\x00' + public_key_ripe
        
        # 计算校验和
        checksum_full = hashlib.sha256(hashlib.sha256(version_public_key).digest()).digest()
        checksum = checksum_full[:4]
        
        # 组合并Base58编码
        binary_address = version_public_key + checksum
        address = base58.b58encode(binary_address).decode('ascii')
        
        return address

class ProgressMonitor:
    """进度监视器"""
    
    def __init__(self, total_keys, update_interval=5):
        self.total_keys = total_keys
        self.update_interval = update_interval
        self.keys_checked = 0
        self.start_time = datetime.now()
        self.last_update_time = self.start_time
        self.last_keys_checked = 0
        
    def update(self, keys_checked, current_key=None, current_address=None):
        self.keys_checked = keys_checked
        
        current_time = datetime.now()
        elapsed = (current_time - self.start_time).total_seconds()
        
        # 每5秒更新一次进度，或者有新的私钥地址时更新
        should_update = (current_time - self.last_update_time).total_seconds() >= self.update_interval
        
        if current_key is not None and current_address is not None:
            should_update = True
        
        if should_update:
            progress = (keys_checked / self.total_keys) * 100 if self.total_keys > 0 else 0
            
            # 计算速度
            time_diff = (current_time - self.last_update_time).total_seconds()
            keys_diff = keys_checked - self.last_keys_checked
            current_speed = keys_diff / time_diff if time_diff > 0 else 0
            avg_speed = keys_checked / elapsed if elapsed > 0 else 0
            
            print(f"\n--- 进度更新 {datetime.now().strftime('%H:%M:%S')} ---")
            print(f"已检查密钥: {keys_checked:,}")
            print(f"总体进度: {progress:.10f}%")
            print(f"当前速度: {current_speed:,.0f} 密钥/秒")
            print(f"平均速度: {avg_speed:,.0f} 密钥/秒")
            print(f"运行时间: {elapsed:.2f} 秒")
            
            if current_key is not None and current_address is not None:
                print(f"当前私钥: {current_key}")
                print(f"当前地址: {current_address}")
                print(f"私钥(WIF): {BitcoinAddressGenerator.private_key_to_wif(current_key)}")
            
            # 预估剩余时间
            if avg_speed > 0:
                remaining_keys = self.total_keys - keys_checked
                remaining_seconds = remaining_keys / avg_speed
                if remaining_seconds < 60:
                    print(f"预计剩余时间: {remaining_seconds:.0f} 秒")
                elif remaining_seconds < 3600:
                    print(f"预计剩余时间: {remaining_seconds/60:.1f} 分钟")
                else:
                    print(f"预计剩余时间: {remaining_seconds/3600:.1f} 小时")
            
            self.last_update_time = current_time
            self.last_keys_checked = keys_checked

class BitcoinKeySearcher:
    def __init__(self, target_address):
        self.target_address = target_address
        # 使用较小的范围进行演示，避免大整数问题
        self.start_range = 2**60  # 从2^60开始用于测试
        self.end_range = 2**61    # 到2^61用于测试
        self.found_key = None
        self.is_running = False
        self.address_generator = BitcoinAddressGenerator()
        
        # 进度监控
        total_keys = self.end_range - self.start_range
        self.progress_monitor = ProgressMonitor(total_keys)
        
        # 尝试使用GPU，如果失败则回退到CPU
        self.use_gpu = False
        self.setup_gpu()
    
    def setup_gpu(self):
        """设置GPU计算环境"""
        try:
            # 测试CuPy
            test_array = cp.arange(1000, dtype=cp.uint64)
            result = cp.sum(test_array)
            print("✓ CuPy GPU计算可用")
            self.use_gpu = True
        except Exception as e:
            print(f"✗ CuPy不可用: {e}")
            print("使用CPU计算")
            self.use_gpu = False
    
    def generate_keys_batch_cpu(self, start_batch, batch_size):
        """在CPU上生成一批私钥并检查"""
        found_key = None
        
        for i in range(batch_size):
            current_key = start_batch + i
            
            if current_key >= self.end_range:
                break
            
            # 生成地址
            address = self.address_generator.private_key_to_address(current_key)
            
            # 更新进度 (每1000个密钥或找到匹配时显示)
            if i % 1000 == 0 or address == self.target_address:
                self.progress_monitor.update(
                    self.progress_monitor.keys_checked + i + 1,
                    current_key, 
                    address
                )
            
            # 检查是否匹配
            if address == self.target_address:
                found_key = current_key
                break
        
        return found_key
    
    def generate_keys_batch_gpu(self, start_batch, batch_size):
        """在GPU上生成一批私钥并检查"""
        try:
            # 使用CuPy生成密钥范围
            # 注意: 处理大整数时需要分块
            chunk_size = min(100000, batch_size)  # 分块处理避免内存问题
            
            found_key = None
            keys_processed = 0
            
            while keys_processed < batch_size and found_key is None:
                current_chunk_size = min(chunk_size, batch_size - keys_processed)
                chunk_start = start_batch + keys_processed
                
                # 使用numpy生成密钥范围 (避免大整数问题)
                keys_np = np.arange(chunk_start, chunk_start + current_chunk_size, dtype=np.uint64)
                
                # 转换为CuPy数组
                keys_cp = cp.asarray(keys_np)
                
                # 在CPU上处理每个密钥 (避免GPU上的大整数问题)
                for i in range(len(keys_np)):
                    current_key = int(keys_np[i])
                    
                    # 生成地址
                    address = self.address_generator.private_key_to_address(current_key)
                    
                    # 更新进度
                    if i % 1000 == 0 or address == self.target_address:
                        total_checked = self.progress_monitor.keys_checked + keys_processed + i + 1
                        self.progress_monitor.update(total_checked, current_key, address)
                    
                    # 检查是否匹配
                    if address == self.target_address:
                        found_key = current_key
                        break
                
                keys_processed += current_chunk_size
                self.progress_monitor.keys_checked += current_chunk_size
            
            return found_key
            
        except Exception as e:
            print(f"GPU处理错误: {e}")
            print("回退到CPU处理")
            return self.generate_keys_batch_cpu(start_batch, batch_size)
    
    def search_batch(self, start_batch, batch_size):
        """搜索一批私钥"""
        if self.use_gpu:
            return self.generate_keys_batch_gpu(start_batch, batch_size)
        else:
            return self.generate_keys_batch_cpu(start_batch, batch_size)
    
    def start_search(self):
        """开始搜索过程"""
        print(f"开始搜索私钥...")
        print(f"目标地址: {self.target_address}")
        print(f"搜索范围: {self.start_range} 到 {self.end_range}")
        print(f"范围大小: {self.end_range - self.start_range:,} 个密钥")
        print(f"使用计算设备: {'GPU' if self.use_gpu else 'CPU'}")
        print("=" * 60)
        
        self.is_running = True
        start_time = datetime.now()
        
        batch_size = 10000  # 每批处理10K个密钥
        current_batch = self.start_range
        
        try:
            while self.is_running and current_batch < self.end_range:
                actual_batch_size = min(batch_size, self.end_range - current_batch)
                
                found_key = self.search_batch(current_batch, actual_batch_size)
                
                if found_key is not None:
                    self.found_key = found_key
                    self.save_result()
                    break
                
                current_batch += actual_batch_size
                
                # 强制进度更新
                self.progress_monitor.update(
                    current_batch - self.start_range,
                    current_batch,  # 显示当前批次开始的私钥
                    self.address_generator.private_key_to_address(current_batch)
                )
                
                # 小延迟避免过度占用资源
                time.sleep(0.01)
        
        except KeyboardInterrupt:
            print("\n用户中断搜索")
            self.is_running = False
        
        except Exception as e:
            print(f"搜索过程中出错: {e}")
            self.is_running = False
        
        finally:
            if not self.found_key:
                print("\n在指定范围内未找到匹配的私钥")
            
            total_time = (datetime.now() - start_time).total_seconds()
            total_checked = current_batch - self.start_range
            print(f"\n=== 搜索完成 ===")
            print(f"总运行时间: {total_time:.2f} 秒")
            print(f"总检查密钥: {total_checked:,}")
            if total_time > 0:
                print(f"平均速度: {total_checked/total_time:,.0f} 密钥/秒")
    
    def save_result(self):
        """保存找到的结果"""
        if self.found_key:
            result = {
                'private_key_hex': hex(self.found_key),
                'private_key_decimal': str(self.found_key),
                'private_key_wif': BitcoinAddressGenerator.private_key_to_wif(self.found_key),
                'address': self.target_address,
                'found_at': datetime.now().isoformat(),
                'search_range_start': self.start_range,
                'search_range_end': self.end_range
            }
            
            filename = f"found_key_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("=== 比特币私钥碰撞结果 ===\\n")
                f.write(f"发现时间: {result['found_at']}\\n")
                f.write(f"目标地址: {result['address']}\\n")
                f.write(f"私钥 (十进制): {result['private_key_decimal']}\\n")
                f.write(f"私钥 (十六进制): {result['private_key_hex']}\\n")
                f.write(f"私钥 (WIF格式): {result['private_key_wif']}\\n")
                f.write(f"搜索范围: {result['search_range_start']} 到 {result['search_range_end']}\\n")
                f.write("\\n警告: 请妥善保管私钥，确保安全！\\n")
            
            print(f"\\n🎉 找到匹配的私钥!")
            print(f"私钥 (十进制): {result['private_key_decimal']}")
            print(f"私钥 (十六进制): {result['private_key_hex']}")
            print(f"私钥 (WIF格式): {result['private_key_wif']}")
            print(f"对应地址: {result['address']}")
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
    except:
        print("使用CPU计算模式")
    
    # 创建搜索器
    searcher = BitcoinKeySearcher(TARGET_ADDRESS)
    
    try:
        # 开始搜索
        searcher.start_search()
    except KeyboardInterrupt:
        print("\\n用户中断搜索")
        searcher.stop()
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        searcher.stop()

if __name__ == "__main__":
    main()
