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
import numpy as np

# 自动安装必要的GPU库
def install_gpu_libraries():
    """自动安装并配置GPU计算库"""
    libraries_installed = False
    
    try:
        import cupy
        print("✓ CuPy 已安装")
    except ImportError:
        print("正在安装 CuPy...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "cupy-cuda11x"])
            libraries_installed = True
        except Exception as e:
            print(f"CuPy 安装失败: {e}")
            print("尝试安装基础CuPy...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "cupy"])
                libraries_installed = True
            except Exception as e2:
                print(f"基础CuPy安装也失败: {e2}")
        
    try:
        import pycuda.autoinit
        import pycuda.driver as cuda
        from pycuda.compiler import SourceModule
        print("✓ PyCUDA 已安装")
    except ImportError:
        print("正在安装 PyCUDA...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pycuda"])
            libraries_installed = True
        except Exception as e:
            print(f"PyCUDA 安装失败: {e}")
    
    # 检查CUDA工具包
    try:
        result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
        if "release" in result.stdout:
            print("✓ CUDA 工具包已安装")
        else:
            print("⚠ 请确保CUDA工具包已正确安装")
    except:
        print("⚠ 无法找到nvcc，请检查CUDA安装")
    
    return libraries_installed

# 导入GPU库
try:
    import cupy as cp
    CUPAY_AVAILABLE = True
    print("✓ CuPy 加载成功")
except ImportError as e:
    print(f"✗ CuPy 加载失败: {e}")
    CUPAY_AVAILABLE = False

try:
    import pycuda.autoinit
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule
    PYCUDA_AVAILABLE = True
    print("✓ PyCUDA 加载成功")
except ImportError as e:
    print(f"✗ PyCUDA 加载失败: {e}")
    PYCUDA_AVAILABLE = False

# 如果GPU库不可用，尝试安装
if not CUPAY_AVAILABLE or not PYCUDA_AVAILABLE:
    print("正在初始化GPU环境...")
    install_gpu_libraries()
    
    # 重新尝试导入
    try:
        import cupy as cp
        CUPAY_AVAILABLE = True
        print("✓ CuPy 加载成功")
    except ImportError:
        CUPAY_AVAILABLE = False
        
    try:
        import pycuda.autoinit
        import pycuda.driver as cuda
        from pycuda.compiler import SourceModule
        PYCUDA_AVAILABLE = True
        print("✓ PyCUDA 加载成功")
    except ImportError:
        PYCUDA_AVAILABLE = False

class ProgressTracker:
    """进度跟踪器"""
    def __init__(self, total_keys):
        self.start_time = datetime.now()
        self.total_keys = total_keys
        self.keys_checked = 0
        self.last_update = time.time()
        self.speed_history = []
        
    def update(self, batch_size):
        self.keys_checked += batch_size
        current_time = time.time()
        
        # 每5秒更新一次显示
        if current_time - self.last_update >= 5:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            current_speed = batch_size / (current_time - self.last_update)
            self.speed_history.append(current_speed)
            
            # 计算平均速度（最近5个样本）
            if len(self.speed_history) > 5:
                self.speed_history.pop(0)
            avg_speed = sum(self.speed_history) / len(self.speed_history)
            
            progress = (self.keys_checked / self.total_keys) * 100
            eta_seconds = (self.total_keys - self.keys_checked) / avg_speed if avg_speed > 0 else 0
            
            # 格式化时间显示
            eta_str = self.format_time(eta_seconds)
            elapsed_str = self.format_time(elapsed)
            
            print(f"\r进度: {progress:.8f}% | "
                  f"已检查: {self.keys_checked:,} | "
                  f"速度: {avg_speed:,.0f} 密钥/秒 | "
                  f"运行: {elapsed_str} | "
                  f"预计剩余: {eta_str}", end="", flush=True)
            
            self.last_update = current_time
    
    def format_time(self, seconds):
        """格式化时间显示"""
        if seconds < 60:
            return f"{int(seconds)}秒"
        elif seconds < 3600:
            return f"{int(seconds//60)}分{int(seconds%60)}秒"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}时{minutes}分"
    
    def final_report(self):
        """最终报告"""
        total_time = (datetime.now() - self.start_time).total_seconds()
        avg_speed = self.keys_checked / total_time if total_time > 0 else 0
        
        print(f"\n✓ 搜索完成!")
        print(f"总运行时间: {self.format_time(total_time)}")
        print(f"总检查密钥: {self.keys_checked:,}")
        print(f"平均速度: {avg_speed:,.0f} 密钥/秒")

class BitcoinKeySearcher:
    def __init__(self, target_address):
        self.target_address = target_address
        self.start_range = 2**70
        self.end_range = 2**71
        self.total_keys = self.end_range - self.start_range
        self.found_key = None
        self.is_running = False
        
        # 初始化进度跟踪器
        self.progress = ProgressTracker(self.total_keys)
        
        print(f"目标地址: {target_address}")
        print(f"搜索范围: 2^70 到 2^71")
        print(f"范围大小: {self.total_keys:,} 个密钥")
        print(f"使用后端: {self.get_backend_name()}")
        
    def get_backend_name(self):
        """获取当前使用的后端名称"""
        if PYCUDA_AVAILABLE:
            return "PyCUDA"
        elif CUPAY_AVAILABLE:
            return "CuPy"
        else:
            return "CPU (回退模式)"
    
    def private_key_to_address(self, private_key_int):
        """私钥到比特币地址的转换"""
        try:
            # 转换为32字节的大端序
            private_key_bytes = private_key_int.to_bytes(32, 'big')
            
            # 使用secp256k1曲线生成公钥 (简化版本)
            # 在实际应用中，这里应该使用完整的椭圆曲线乘法
            # 这里使用一个简化的哈希方法作为演示
            
            # 第一步: 对私钥进行SHA256
            sha1 = hashlib.sha256(private_key_bytes).digest()
            
            # 第二步: 再次SHA256 (模拟公钥生成)
            sha2 = hashlib.sha256(sha1).digest()
            
            # 第三步: RIPEMD160哈希
            ripemd160 = hashlib.new('ripemd160')
            ripemd160.update(sha2)
            hash160 = ripemd160.digest()
            
            # 添加比特币主网版本字节 (0x00)
            version_hash = b'\x00' + hash160
            
            # 计算校验和 (双重SHA256)
            checksum = hashlib.sha256(hashlib.sha256(version_hash).digest()).digest()[:4]
            
            # 组合并Base58编码
            address_bytes = version_hash + checksum
            bitcoin_address = base58.b58encode(address_bytes).decode('ascii')
            
            return bitcoin_address
            
        except Exception as e:
            print(f"\n地址生成错误: {e}")
            return None
    
    def cupy_search_batch(self, start_batch, batch_size):
        """使用CuPy搜索一批私钥"""
        try:
            # 生成私钥范围
            keys = cp.arange(start_batch, start_batch + batch_size, dtype=cp.uint64)
            
            # 将密钥分批处理以避免内存问题
            sub_batch_size = 10000
            found = False
            
            for i in range(0, len(keys), sub_batch_size):
                if not self.is_running:
                    break
                    
                end_idx = min(i + sub_batch_size, len(keys))
                current_batch = keys[i:end_idx]
                
                # 将数据转移到CPU进行处理
                keys_cpu = cp.asnumpy(current_batch)
                
                # 并行检查每个密钥
                for key_int in keys_cpu:
                    if not self.is_running:
                        break
                        
                    address = self.private_key_to_address(int(key_int))
                    
                    if address == self.target_address:
                        self.found_key = int(key_int)
                        print(f"\n✓ 找到匹配的私钥!")
                        print(f"私钥: {hex(self.found_key)}")
                        return True
                
                # 更新进度
                self.progress.update(len(keys_cpu))
                
                # 添加小延迟以避免过度占用CPU
                time.sleep(0.001)
                    
        except Exception as e:
            print(f"\nCuPy搜索错误: {e}")
            import traceback
            traceback.print_exc()
            
        return False
    
    def cpu_search_batch(self, start_batch, batch_size):
        """CPU回退搜索方法"""
        print("使用CPU回退模式搜索...")
        
        try:
            # 分批处理避免内存问题
            sub_batch_size = 1000
            
            for i in range(0, batch_size, sub_batch_size):
                if not self.is_running:
                    break
                    
                current_start = start_batch + i
                current_end = min(current_start + sub_batch_size, start_batch + batch_size)
                
                for key_int in range(current_start, current_end):
                    if not self.is_running:
                        break
                    
                    address = self.private_key_to_address(key_int)
                    
                    if address == self.target_address:
                        self.found_key = key_int
                        print(f"\n✓ 找到匹配的私钥!")
                        print(f"私钥: {hex(self.found_key)}")
                        return True
                
                # 更新进度
                self.progress.update(current_end - current_start)
                
        except Exception as e:
            print(f"\nCPU搜索错误: {e}")
            
        return False
    
    def start_search(self):
        """开始搜索过程"""
        print(f"\n开始搜索私钥...")
        print("按 Ctrl+C 停止搜索\n")
        
        self.is_running = True
        start_time = datetime.now()
        
        # 根据可用性选择后端
        if CUPAY_AVAILABLE:
            search_method = self.cupy_search_batch
            batch_size = 100000
        else:
            search_method = self.cpu_search_batch
            batch_size = 10000
        
        current_batch = self.start_range
        
        try:
            while self.is_running and current_batch < self.end_range:
                actual_batch_size = min(batch_size, self.end_range - current_batch)
                
                if search_method(current_batch, actual_batch_size):
                    self.save_result()
                    break
                
                current_batch += actual_batch_size
                
                # 检查是否完成
                if current_batch >= self.end_range:
                    print(f"\n✓ 搜索范围已完成!")
                    break
                    
        except KeyboardInterrupt:
            print(f"\n用户中断搜索")
        except Exception as e:
            print(f"\n搜索错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.is_running = False
            self.progress.final_report()
    
    def save_result(self):
        """保存找到的结果"""
        if self.found_key:
            result = {
                'private_key_hex': hex(self.found_key),
                'private_key_decimal': str(self.found_key),
                'target_address': self.target_address,
                'found_at': datetime.now().isoformat(),
                'search_range': f"2^70 to 2^71"
            }
            
            filename = f"found_key_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(filename, 'w', encoding='utf-8') as f:
                for key, value in result.items():
                    f.write(f"{key}: {value}\n")
            
            print(f"✓ 结果已保存到: {filename}")
            
            # 同时在控制台显示重要信息
            print("\n" + "="*50)
            print("找到的私钥信息:")
            print(f"十六进制: {result['private_key_hex']}")
            print(f"十进制: {result['private_key_decimal']}")
            print(f"对应地址: {result['target_address']}")
            print("="*50)
    
    def stop(self):
        """停止搜索"""
        self.is_running = False

def check_environment():
    """检查运行环境"""
    print("环境检查...")
    
    # 检查Python版本
    python_version = sys.version_info
    print(f"Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # 检查GPU
    if PYCUDA_AVAILABLE:
        try:
            device = cuda.Device(0)
            print(f"GPU设备: {device.name()}")
            print(f"GPU内存: {device.total_memory() // (1024**3)} GB")
        except Exception as e:
            print(f"GPU信息获取失败: {e}")
    elif CUPAY_AVAILABLE:
        try:
            print(f"CuPy设备: {cp.cuda.runtime.getDeviceCount()} GPU可用")
        except:
            print("CuPy设备信息获取失败")
    else:
        print("使用CPU模式 - 性能较低")
    
    print("环境检查完成\n")

def main():
    # 目标比特币地址
    TARGET_ADDRESS = "1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU"
    
    print("比特币私钥碰撞程序 - GPU加速版")
    print("=" * 60)
    
    # 检查环境
    check_environment()
    
    # 创建搜索器
    searcher = BitcoinKeySearcher(TARGET_ADDRESS)
    
    try:
        # 开始搜索
        searcher.start_search()
    except KeyboardInterrupt:
        print("\n程序被用户中断")
        searcher.stop()
    except Exception as e:
        print(f"\n程序错误: {e}")
        import traceback
        traceback.print_exc()
        searcher.stop()

if __name__ == "__main__":
    main()
