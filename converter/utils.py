"""
实用工具函数，优化性能的关键点
"""

import os
import json
import hashlib
import time
import base64
import numpy as np
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import subprocess
from pathlib import Path
from numba import njit, prange, vectorize
import logging

logger = logging.getLogger(__name__)

# 使用numba进行JIT编译，极大提升性能
@njit(fastmath=True, parallel=True)
def expand_pixels_9x1(data, width, height):
    """
    将一个逻辑像素扩展为3x3的物理像素块（9合1）
    使用numba的并行处理，显著提升性能
    
    Args:
        data: 原始数据数组，形状为(height, width, 3)
        width: 逻辑宽度
        height: 逻辑高度
        
    Returns:
        扩展后的数组，形状为(height*3, width*3, 3)
    """
    expanded = np.empty((height * 3, width * 3, 3), dtype=np.uint8)
    
    for y in prange(height):
        for x in prange(width):
            pixel = data[y, x]
            for i in range(3):
                for j in range(3):
                    expanded[y*3+i, x*3+j] = pixel
                    
    return expanded

# 进一步优化的数据块映射到颜色索引
@njit(fastmath=True)
def bytes_to_color_indices(data_bytes, max_colors):
    """
    将字节数据映射到颜色索引
    
    Args:
        data_bytes: 字节数组
        max_colors: 最大颜色数（16或256）
        
    Returns:
        颜色索引数组
    """
    if max_colors == 16:
        # 每字节存2个4位索引
        result = np.empty(len(data_bytes) * 2, dtype=np.uint8)
        for i in range(len(data_bytes)):
            byte = data_bytes[i]
            result[i*2] = byte >> 4  # 高4位
            result[i*2+1] = byte & 0x0F  # 低4位
        return result
    else:  # 256色
        # 直接一对一映射
        return np.frombuffer(data_bytes, dtype=np.uint8)

# 文件缓存管理器
class CacheManager:
    """高性能缓存管理器，使用分块存储和内存映射"""
    
    def __init__(self, cache_dir, chunk_size=1024*1024):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.chunk_size = chunk_size
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.metadata = self._load_metadata()
        
    def _load_metadata(self):
        """加载缓存元数据"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"缓存元数据加载失败: {e}")
                return {}
        return {}
    
    def _save_metadata(self):
        """保存缓存元数据"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f)
    
    def cache_file(self, filepath):
        """
        将文件缓存到分块存储
        
        Args:
            filepath: 源文件路径
            
        Returns:
            缓存ID
        """
        filepath = Path(filepath)
        file_hash = self._calculate_file_hash(filepath)
        
        # 检查是否已缓存
        if file_hash in self.metadata:
            logger.info(f"文件已缓存: {filepath.name}")
            return file_hash
        
        # 创建新缓存条目
        file_size = filepath.stat().st_size
        chunks = []
        
        with open(filepath, 'rb') as src_file:
            chunk_index = 0
            while True:
                data = src_file.read(self.chunk_size)
                if not data:
                    break
                
                chunk_hash = hashlib.md5(data).hexdigest()
                chunk_path = self.cache_dir / f"{file_hash}_{chunk_index}.bin"
                
                with open(chunk_path, 'wb') as chunk_file:
                    chunk_file.write(data)
                
                chunks.append({
                    "index": chunk_index,
                    "path": str(chunk_path.relative_to(self.cache_dir)),
                    "size": len(data),
                    "hash": chunk_hash
                })
                
                chunk_index += 1
        
        # 更新元数据
        self.metadata[file_hash] = {
            "original_filename": filepath.name,
            "file_size": file_size,
            "date_cached": time.time(),
            "chunks": chunks
        }
        
        self._save_metadata()
        logger.info(f"文件已缓存: {filepath.name}, ID: {file_hash}")
        return file_hash
    
    def read_cached_file(self, cache_id):
        """
        读取缓存文件
        
        Args:
            cache_id: 缓存ID
            
        Returns:
            生成器，逐块返回数据
        """
        if cache_id not in self.metadata:
            raise ValueError(f"缓存ID不存在: {cache_id}")
        
        file_info = self.metadata[cache_id]
        for chunk in file_info["chunks"]:
            chunk_path = self.cache_dir / chunk["path"]
            with open(chunk_path, 'rb') as f:
                yield f.read()
    
    def get_file_info(self, cache_id):
        """获取缓存文件信息"""
        if cache_id not in self.metadata:
            return None
        return self.metadata[cache_id]
    
    def _calculate_file_hash(self, filepath):
        """计算文件哈希值"""
        hasher = hashlib.md5()
        with open(filepath, 'rb') as f:
            # 读取文件头和尾部来计算哈希，对大文件更高效
            head = f.read(8192)
            f.seek(-8192, 2)
            tail = f.read(8192)
            
            # 组合文件大小、头部和尾部计算哈希
            file_size = filepath.stat().st_size
            hasher.update(f"{file_size}".encode())
            hasher.update(head)
            hasher.update(tail)
        
        return hasher.hexdigest()

# 检测NVIDIA硬件编码器是否可用
def is_nvenc_available():
    """检测系统是否支持NVENC硬件加速"""
    try:
        # 通过ffmpeg查询支持的编码器
        cmd = ["ffmpeg", "-encoders"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return "h264_nvenc" in result.stdout
    except Exception as e:
        logger.warning(f"检测NVENC时出错: {e}")
        return False

# 检测Intel QuickSync是否可用
def is_qsv_available():
    """检测系统是否支持Intel QuickSync硬件加速"""
    try:
        cmd = ["ffmpeg", "-encoders"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return "h264_qsv" in result.stdout
    except Exception as e:
        logger.warning(f"检测QSV时出错: {e}")
        return False

# CPU核心数感知的并行任务执行器
class ParallelExecutor:
    """
    高级并行任务执行器，自动选择最优并行级别
    """
    
    def __init__(self, max_workers=None):
        cpu_count = mp.cpu_count()
        self.max_workers = max_workers or max(1, cpu_count - 1)
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
    def map(self, func, iterable):
        """并行映射函数到可迭代对象"""
        return self.executor.map(func, iterable)
    
    def submit(self, func, *args, **kwargs):
        """提交任务到执行器"""
        return self.executor.submit(func, *args, **kwargs)
    
    def shutdown(self):
        """关闭执行器"""
        self.executor.shutdown()

# 高效内存数据流
class MemoryStream:
    """高效内存数据流，避免频繁IO操作"""
    
    def __init__(self, initial_capacity=1024*1024):
        self.buffer = bytearray(initial_capacity)
        self.size = 0
        self.capacity = initial_capacity
        
    def write(self, data):
        """写入数据到缓冲区"""
        data_len = len(data)
        required_size = self.size + data_len
        
        # 如果需要扩容
        if required_size > self.capacity:
            new_capacity = max(self.capacity * 2, required_size)
            new_buffer = bytearray(new_capacity)
            new_buffer[:self.size] = self.buffer[:self.size]
            self.buffer = new_buffer
            self.capacity = new_capacity
        
        # 写入数据
        self.buffer[self.size:self.size+data_len] = data
        self.size += data_len
        
    def get_bytes(self):
        """获取缓冲区数据"""
        return bytes(self.buffer[:self.size])
    
    def clear(self):
        """清空缓冲区"""
        self.size = 0
        
    def __len__(self):
        return self.size

# 快速计算视频相关参数
def calculate_video_params(file_size, resolution="4K", fps=30, nine_to_one=True, 
                          color_count=16, error_correction_ratio=0.1):
    """
    计算视频相关参数
    
    Args:
        file_size: 文件大小(字节)
        resolution: 分辨率("4K", "1080p", "720p")
        fps: 帧率
        nine_to_one: 是否使用9合1模式
        color_count: 颜色数量(16或256)
        error_correction_ratio: 纠错码比例
        
    Returns:
        dict: 包含视频参数的字典
    """
    from . import VIDEO_PRESETS
    
    # 获取分辨率
    preset = VIDEO_PRESETS.get(resolution, VIDEO_PRESETS["4K"])
    width, height = preset["width"], preset["height"]
    
    # 计算每个像素可存储的比特数
    bits_per_pixel = 4 if color_count == 16 else 8
    
    # 计算逻辑分辨率
    if nine_to_one:
        logical_width = width // 3
        logical_height = height // 3
    else:
        logical_width = width
        logical_height = height
    
    # 计算每帧可存储的字节数
    bytes_per_frame = logical_width * logical_height * bits_per_pixel // 8
    
    # 考虑纠错码后的有效载荷
    effective_bytes_per_frame = int(bytes_per_frame * (1 - error_correction_ratio))
    
    # 计算所需帧数
    total_frames = (file_size + effective_bytes_per_frame - 1) // effective_bytes_per_frame
    
    # 计算视频时长
    duration_seconds = total_frames / fps
    
    # 估计视频文件大小 (考虑压缩)
    estimated_bitrate = width * height * fps * 0.07  # 经验值
    estimated_video_size = estimated_bitrate * duration_seconds / 8
    
    return {
        "total_frames": total_frames,
        "duration_seconds": duration_seconds,
        "duration_formatted": f"{int(duration_seconds//60):02d}:{int(duration_seconds%60):02d}",
        "estimated_video_size": estimated_video_size,
        "estimated_video_size_mb": estimated_video_size / (1024 * 1024),
        "logical_width": logical_width,
        "logical_height": logical_height,
        "physical_width": width,
        "physical_height": height,
        "bytes_per_frame": effective_bytes_per_frame
    }
