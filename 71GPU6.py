import sys
import subprocess
import os

# 自动安装必要的库
def install_packages():
    packages = ['cupy-cuda11x', 'pycuda', 'base58', 'ecdsa']
    for package in packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            print(f"正在安装 {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install_packages()

import cupy as cp
import base58
import hashlib
import ecdsa
from ecdsa import SECP256k1
import numpy as np
import time

class GPUBitcoinKeyGenerator:
    def __init__(self):
        self.target_address = "1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU"
        
    def private_key_to_address(self, private_key_hex):
        """将私钥转换为比特币地址"""
        # 将十六进制私钥转换为字节
        private_key_bytes = bytes.fromhex(private_key_hex)
        
        # 生成公钥
        sk = ecdsa.SigningKey.from_string(private_key_bytes, curve=SECP256k1)
        vk = sk.get_verifying_key()
        public_key_bytes = b'\x04' + vk.to_string()
        
        # SHA256哈希
        sha256_hash = hashlib.sha256(public_key_bytes).digest()
        
        # RIPEMD160哈希
        ripemd160_hash = hashlib.new('ripemd160', sha256_hash).digest()
        
        # 添加主网版本字节
        versioned_hash = b'\x00' + ripemd160_hash
        
        # 计算校验和
        checksum = hashlib.sha256(hashlib.sha256(versioned_hash).digest()).digest()[:4]
        
        # 组合并Base58编码
        binary_address = versioned_hash + checksum
        bitcoin_address = base58.b58encode(binary_address).decode('utf-8')
        
        return bitcoin_address

    def generate_and_check_keys_gpu(self, start_hex, end_hex, batch_size=100000):
        """使用GPU批量生成和检查私钥"""
        start_int = int(start_hex, 16)
        end_int = int(end_hex, 16)
        
        current_batch_start = start_int
        
        while current_batch_start <= end_int:
            # 计算当前批次结束位置
            current_batch_end = min(current_batch_start + batch_size, end_int)
            
            # 在GPU上生成私钥批次
            private_keys_int = cp.arange(current_batch_start, current_batch_end + 1, dtype=cp.uint64)
            
            # 将私钥转换为CPU numpy数组进行处理
            private_keys_cpu = cp.asnumpy(private_keys_int)
            
            for private_key_int in private_keys_cpu:
                # 转换为64字符的十六进制格式（前导零）
                private_key_hex = format(private_key_int, '064x')
                
                try:
                    # 生成比特币地址
                    address = self.private_key_to_address(private_key_hex)
                    
                    # 检查是否匹配目标地址
                    if address == self.target_address:
                        print(f"找到匹配的私钥!")
                        print(f"私钥: {private_key_hex}")
                        print(f"地址: {address}")
                        
                        # 保存结果到文件
                        with open("found_key.txt", "w") as f:
                            f.write(f"私钥: {private_key_hex}\n")
                            f.write(f"地址: {address}\n")
                        
                        return True
                        
                except Exception as e:
                    continue
            
            # 更新下一批次的起始位置
            current_batch_start = current_batch_end + 1
        
        return False

def main():
    # 初始化GPU密钥生成器
    generator = GPUBitcoinKeyGenerator()
    
    # 定义搜索范围
    start_range = "0000000000000000000000000000000000000000000000400000000000000000"
    end_range = "00000000000000000000000000000000000000000000007fffffffffffffffff"
    
    print("开始GPU加速的私钥搜索...")
    print(f"目标地址: {generator.target_address}")
    print(f"搜索范围: {start_range} 到 {end_range}")
    print("正在搜索中，请等待...")
    
    start_time = time.time()
    
    # 开始搜索
    found = generator.generate_and_check_keys_gpu(start_range, end_range)
    
    end_time = time.time()
    
    if not found:
        print("在指定范围内未找到匹配的私钥")
    
    print(f"搜索耗时: {end_time - start_time:.2f} 秒")

if __name__ == "__main__":
    main()
