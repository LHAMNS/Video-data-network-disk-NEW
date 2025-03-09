"""
高性能帧生成器，使用NumPy和Numba进行优化
实现二进制数据到视频帧的映射
"""

import numpy as np
from numba import njit, prange, vectorize
import logging
import time
from io import BytesIO
import base64
from PIL import Image
import cv2
from . import COLOR_PALETTE_16
from .utils import expand_pixels_9x1, bytes_to_color_indices

logger = logging.getLogger(__name__)

# 将16色调色板转换为NumPy数组，便于快速访问
COLOR_PALETTE_16_ARRAY = np.array(COLOR_PALETTE_16, dtype=np.uint8)

# 生成从索引到RGB的查找表
@njit(fastmath=True)
def generate_color_lut(palette, color_count):
    """
    生成颜色查找表(LUT)
    
    Args:
        palette: 颜色调色板
        color_count: 颜色数量
        
    Returns:
        查找表数组，索引为颜色索引，值为RGB值
    """
    lut = np.zeros((color_count, 3), dtype=np.uint8)
    for i in range(min(color_count, len(palette))):
        lut[i] = palette[i]
    
    # 如果需要更多颜色，自动生成
    if color_count > len(palette):
        step = 256 // int(np.cbrt(color_count - len(palette)))
        idx = len(palette)
        for r in range(0, 256, step):
            for g in range(0, 256, step):
                for b in range(0, 256, step):
                    if idx < color_count:
                        lut[idx] = np.array([r, g, b], dtype=np.uint8)
                        idx += 1
    
    return lut

# 使用numba加速的核心帧生成函数
@njit(fastmath=True, parallel=True)
def generate_frame_array(data_indices, width, height, lut):
    """
    根据数据索引生成帧数组
    
    Args:
        data_indices: 颜色索引数组
        width: 帧宽度
        height: 帧高度
        lut: 颜色查找表
        
    Returns:
        RGB帧数组，形状为(height, width, 3)
    """
    # 创建输出帧数组
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 计算需要填充的像素数
    pixels_to_fill = min(width * height, len(data_indices))
    
    # 逐像素填充颜色
    for i in prange(pixels_to_fill):
        y = i // width
        x = i % width
        color_idx = data_indices[i]
        frame[y, x] = lut[color_idx]
    
    return frame

class FrameGenerator:
    """
    高性能帧生成器
    将二进制数据转换为视频帧
    """
    
    def __init__(self, resolution="4K", fps=30, color_count=16, nine_to_one=True):
        """
        初始化帧生成器
        
        Args:
            resolution: 分辨率 ("4K", "1080p", "720p")
            fps: 帧率
            color_count: 颜色数量 (16 或 256)
            nine_to_one: 是否使用9合1像素合并
        """
        from . import VIDEO_PRESETS
        
        # 存储参数
        self.resolution_name = resolution
        self.resolution = VIDEO_PRESETS.get(resolution, VIDEO_PRESETS["4K"])
        self.fps = fps
        self.color_count = color_count
        self.nine_to_one = nine_to_one
        
        # 计算物理和逻辑分辨率
        self.physical_width = self.resolution["width"]
        self.physical_height = self.resolution["height"]
        
        if nine_to_one:
            self.logical_width = self.physical_width // 3
            self.logical_height = self.physical_height // 3
        else:
            self.logical_width = self.physical_width
            self.logical_height = self.physical_height
        
        # 生成颜色查找表
        self.color_lut = generate_color_lut(COLOR_PALETTE_16_ARRAY, color_count)
        
        # 计算每帧可存储的字节数
        bits_per_pixel = 4 if color_count == 16 else 8
        self.bytes_per_frame = self.logical_width * self.logical_height * bits_per_pixel // 8
        
        logger.info(f"帧生成器初始化: {resolution}, {fps}fps, {color_count}色, " +
                 f"9合1={nine_to_one}, 每帧容量={self.bytes_per_frame}字节")
    
    def estimate_frame_count(self, data_size):
        """
        估计给定数据大小需要的帧数
        
        Args:
            data_size: 数据大小(字节)
            
        Returns:
            预计需要的帧数
        """
        return (data_size + self.bytes_per_frame - 1) // self.bytes_per_frame
    
    def calculate_logical_pixel_count(self):
        """计算每帧的逻辑像素数量"""
        return self.logical_width * self.logical_height
    
    def generate_frame(self, data_chunk, frame_index=0):
        """
        生成单个视频帧
        
        Args:
            data_chunk: 二进制数据块
            frame_index: 帧索引（用于调试和记录）
            
        Returns:
            RGB帧数组，形状为(height, width, 3)
        """
        start_time = time.time()
        
        # 将数据转换为颜色索引
        color_indices = bytes_to_color_indices(data_chunk, self.color_count)
        
        # 如果颜色索引不足以填满一帧，进行填充
        indices_needed = self.logical_width * self.logical_height
        if len(color_indices) < indices_needed:
            # 创建填充数组
            padded_indices = np.zeros(indices_needed, dtype=np.uint8)
            padded_indices[:len(color_indices)] = color_indices
            color_indices = padded_indices
        
        # 生成逻辑帧
        logical_frame = generate_frame_array(
            color_indices[:indices_needed], 
            self.logical_width, 
            self.logical_height, 
            self.color_lut
        )
        
        # 如果启用9合1，扩展逻辑帧
        if self.nine_to_one:
            physical_frame = expand_pixels_9x1(
                logical_frame, 
                self.logical_width, 
                self.logical_height
            )
        else:
            physical_frame = logical_frame
        
        elapsed = time.time() - start_time
        logger.debug(f"帧 {frame_index} 生成完成，用时 {elapsed:.4f}s")
        
        return physical_frame
    
    def generate_preview_image(self, frame, max_size=300):
        """
        生成用于UI预览的小图像
        
        Args:
            frame: 完整帧数组
            max_size: 预览图最大尺寸
            
        Returns:
            Base64编码的JPEG图像数据
        """
        # 计算缩放比例
        scale = min(max_size / frame.shape[1], max_size / frame.shape[0])
        if scale < 1:
            new_width = int(frame.shape[1] * scale)
            new_height = int(frame.shape[0] * scale)
            preview = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
        else:
            preview = frame
        
        # 转换为BGR (OpenCV格式)
        preview_bgr = cv2.cvtColor(preview, cv2.COLOR_RGB2BGR)
        
        # 编码为JPEG
        _, jpeg_data = cv2.imencode('.jpg', preview_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        
        # 转换为Base64
        return base64.b64encode(jpeg_data).decode('utf-8')
    
    def generate_frames_from_data(self, data, callback=None):
        """
        从数据生成帧的生成器
        
        Args:
            data: 字节数据或可迭代的数据块
            callback: 回调函数，用于报告进度，参数为(帧索引，总帧数，当前帧)
            
        Yields:
            生成的视频帧
        """
        # 如果data是单个字节对象，将其分割成适合每帧的块
        if isinstance(data, (bytes, bytearray)):
            # 计算总帧数
            total_bytes = len(data)
            total_frames = self.estimate_frame_count(total_bytes)
            
            for frame_idx in range(total_frames):
                # 计算当前块的起始和结束位置
                start_pos = frame_idx * self.bytes_per_frame
                end_pos = min(start_pos + self.bytes_per_frame, total_bytes)
                
                # 获取当前数据块
                current_chunk = data[start_pos:end_pos]
                
                # 生成帧
                frame = self.generate_frame(current_chunk, frame_idx)
                
                # 调用回调函数
                if callback:
                    callback(frame_idx, total_frames, frame)
                
                yield frame
        else:
            # 假设data是可迭代的，每次返回一块数据
            for frame_idx, chunk in enumerate(data):
                frame = self.generate_frame(chunk, frame_idx)
                
                # 调用回调函数 (这种情况下无法知道总帧数)
                if callback:
                    callback(frame_idx, None, frame)
                
                yield frame


class OptimizedFrameGenerator(FrameGenerator):
    """
    更进一步优化的帧生成器
    使用预分配缓冲区和更高级的优化技术
    适用于超高性能场景
    """
    
    def __init__(self, *args, **kwargs):
        """初始化优化帧生成器"""
        super().__init__(*args, **kwargs)
        
        # 预分配帧缓冲区
        self.logical_frame_buffer = np.zeros((self.logical_height, self.logical_width, 3), dtype=np.uint8)
        
        if self.nine_to_one:
            self.physical_frame_buffer = np.zeros((self.physical_height, self.physical_width, 3), dtype=np.uint8)
    
    def generate_frame_optimized(self, data_chunk, frame_index=0):
        """
        优化版本的帧生成函数，使用预分配缓冲区
        
        Args:
            data_chunk: 二进制数据块
            frame_index: 帧索引
            
        Returns:
            生成的帧数组的视图（不复制数据）
        """
        # 将数据转换为颜色索引
        color_indices = bytes_to_color_indices(data_chunk, self.color_count)
        
        # 填充逻辑帧缓冲区
        indices_needed = self.logical_width * self.logical_height
        if len(color_indices) < indices_needed:
            # 创建填充数组
            color_indices = np.pad(color_indices, (0, indices_needed - len(color_indices)))
        
        # 使用向量化操作填充逻辑帧
        pixel_count = min(indices_needed, len(color_indices))
        y_coords = np.arange(pixel_count) // self.logical_width
        x_coords = np.arange(pixel_count) % self.logical_width
        
        # 直接从LUT获取颜色并填充
        self.logical_frame_buffer[y_coords, x_coords] = self.color_lut[color_indices[:pixel_count]]
        
        # 如果启用9合1，使用优化的扩展方法
        if self.nine_to_one:
            # 使用向量化操作进行9合1扩展
            for i in range(3):
                for j in range(3):
                    y_slice = slice(i, self.physical_height, 3)
                    x_slice = slice(j, self.physical_width, 3)
                    self.physical_frame_buffer[y_slice, x_slice] = self.logical_frame_buffer
            
            return self.physical_frame_buffer
        else:
            return self.logical_frame_buffer
    
    def process_large_file(self, data_generator, progress_callback=None):
        """
        处理大文件的特殊优化方法
        
        Args:
            data_generator: 数据块生成器
            progress_callback: 进度回调函数
            
        Yields:
            生成的视频帧
        """
        frame_idx = 0
        buffer = bytearray()
        
        for chunk in data_generator:
            # 添加到缓冲区
            buffer.extend(chunk)
            
            # 当缓冲区达到帧所需大小时处理
            while len(buffer) >= self.bytes_per_frame:
                # 取出一帧所需数据
                frame_data = buffer[:self.bytes_per_frame]
                buffer = buffer[self.bytes_per_frame:]
                
                # 生成帧
                frame = self.generate_frame_optimized(frame_data, frame_idx)
                
                # 调用进度回调
                if progress_callback:
                    progress_callback(frame_idx, None, frame.copy())
                
                frame_idx += 1
                yield frame
        
        # 处理剩余数据
        if buffer:
            frame = self.generate_frame_optimized(buffer, frame_idx)
            
            if progress_callback:
                progress_callback(frame_idx, frame_idx + 1, frame.copy())
            
            yield frame
