"""
高性能解码器，用于从视频还原原始文件
这是配套的解码模块，将视频文件转回原始数据
"""

import os
import cv2
import numpy as np
import logging
import time
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from numba import njit, prange
from .error_correction import ReedSolomonEncoder
from . import COLOR_PALETTE_16

logger = logging.getLogger(__name__)

@njit(fastmath=True)
def color_to_index(pixel, color_lut, color_count):
    """
    将像素颜色映射回索引值
    
    Args:
        pixel: RGB像素值 (numpy数组)
        color_lut: 颜色查找表
        color_count: 颜色总数
        
    Returns:
        最接近的颜色索引
    """
    min_dist = float('inf')
    closest_idx = 0
    
    for i in range(color_count):
        # 计算欧氏距离
        dist = np.sum((pixel - color_lut[i]) ** 2)
        if dist < min_dist:
            min_dist = dist
            closest_idx = i
    
    return closest_idx


@njit(fastmath=True, parallel=True)
def extract_frame_data(frame, logical_width, logical_height, nine_to_one, color_lut, color_count):
    """
    从视频帧中提取数据
    
    Args:
        frame: 帧数据 (numpy数组)
        logical_width: 逻辑宽度
        logical_height: 逻辑高度
        nine_to_one: 是否使用9合1模式
        color_lut: 颜色查找表
        color_count: 颜色总数
        
    Returns:
        索引数组
    """
    if nine_to_one:
        # 从9合1模式中提取像素
        indices = np.zeros(logical_width * logical_height, dtype=np.uint8)
        
        for y in prange(logical_height):
            for x in prange(logical_width):
                # 取9合1块的中心点
                center_y = y * 3 + 1
                center_x = x * 3 + 1
                
                # 获取中心像素
                pixel = frame[center_y, center_x]
                
                # 映射到最接近的颜色索引
                idx = color_to_index(pixel, color_lut, color_count)
                
                # 存储索引
                indices[y * logical_width + x] = idx
    else:
        # 直接映射每个像素
        indices = np.zeros(logical_width * logical_height, dtype=np.uint8)
        
        for y in prange(logical_height):
            for x in prange(logical_width):
                pixel = frame[y, x]
                idx = color_to_index(pixel, color_lut, color_count)
                indices[y * logical_width + x] = idx
    
    return indices


@njit(fastmath=True)
def indices_to_bytes(indices, color_count):
    """
    将颜色索引转换回字节数据
    
    Args:
        indices: 颜色索引数组
        color_count: 颜色总数
        
    Returns:
        字节数组
    """
    if color_count == 16:
        # 每两个4位索引合并为一个字节
        num_bytes = len(indices) // 2
        result = np.zeros(num_bytes, dtype=np.uint8)
        
        for i in range(num_bytes):
            high = indices[i*2] & 0x0F
            low = indices[i*2+1] & 0x0F if i*2+1 < len(indices) else 0
            result[i] = (high << 4) | low
        
        return result
    else:  # 256色
        # 每个索引就是一个字节
        return indices.astype(np.uint8)


class VideoDecoder:
    """
    视频解码器，将特制视频转回原始文件
    """
    
    def __init__(self, video_path, output_path=None, 
                nine_to_one=True, color_count=16, use_error_correction=True):
        """
        初始化视频解码器
        
        Args:
            video_path: 视频文件路径
            output_path: 输出文件路径
            nine_to_one: 是否使用9合1模式
            color_count: 颜色数量
            use_error_correction: 是否使用纠错
        """
        self.video_path = Path(video_path)
        self.output_path = Path(output_path) if output_path else self.video_path.with_suffix('')
        self.nine_to_one = nine_to_one
        self.color_count = color_count
        self.use_error_correction = use_error_correction
        
        # 创建颜色查找表
        self.color_lut = np.array(COLOR_PALETTE_16, dtype=np.uint8)
        
        # 初始化视频捕获
        self.cap = None
        self.frame_count = 0
        self.fps = 0
        self.width = 0
        self.height = 0
        self.logical_width = 0
        self.logical_height = 0
        
        # 提取进度
        self.processed_frames = 0
        self.total_frames = 0
        self.running = False
        self.start_time = 0
        
        # 初始化纠错编码器
        self.error_correction = ReedSolomonEncoder() if use_error_correction else None
        
        # 检查视频路径
        if not self.video_path.exists():
            raise FileNotFoundError(f"视频文件不存在: {self.video_path}")
        
        # 尝试打开视频
        self._open_video()
        
        logger.info(f"视频解码器初始化: {self.video_path.name}, " +
                  f"分辨率: {self.width}x{self.height}, {self.fps}fps, " +
                  f"9合1={self.nine_to_one}, 纠错={self.use_error_correction}")
    
    def _open_video(self):
        """打开视频文件并读取基本信息"""
        self.cap = cv2.VideoCapture(str(self.video_path))
        
        if not self.cap.isOpened():
            raise ValueError(f"无法打开视频文件: {self.video_path}")
        
        # 获取视频信息
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 计算逻辑尺寸
        if self.nine_to_one:
            self.logical_width = self.width // 3
            self.logical_height = self.height // 3
        else:
            self.logical_width = self.width
            self.logical_height = self.height
        
        self.total_frames = self.frame_count
    
    def extract_data(self, callback=None):
        """
        开始数据提取过程
        
        Args:
            callback: 回调函数，用于报告进度
            
        Returns:
            提取的数据
        """
        if self.running:
            logger.warning("解码器已在运行")
            return
        
        self.running = True
        self.start_time = time.time()
        
        try:
            # 确保视频已打开
            if self.cap is None or not self.cap.isOpened():
                self._open_video()
            
            # 创建内存缓冲区存储所有提取的数据
            all_data = bytearray()
            
            # 逐帧处理视频
            self.processed_frames = 0
            
            while self.running and self.processed_frames < self.total_frames:
                # 读取一帧
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # 转换为RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 提取帧数据
                indices = extract_frame_data(
                    frame_rgb, self.logical_width, self.logical_height, 
                    self.nine_to_one, self.color_lut, self.color_count
                )
                
                # 转换回字节
                frame_bytes = indices_to_bytes(indices, self.color_count)
                
                # 追加到数据缓冲区
                all_data.extend(frame_bytes)
                
                # 更新进度
                self.processed_frames += 1
                
                # 调用回调函数
                if callback and self.processed_frames % 10 == 0:
                    elapsed = time.time() - self.start_time
                    fps = self.processed_frames / elapsed if elapsed > 0 else 0
                    callback(self.processed_frames, self.total_frames, fps)
            
            # 应用纠错解码（如果启用）
            if self.use_error_correction and self.error_correction:
                logger.info("应用纠错解码...")
                all_data = self.error_correction.decode_data(bytes(all_data))
            
            # 写入输出文件
            with open(self.output_path, 'wb') as f:
                f.write(all_data)
            
            logger.info(f"数据提取完成，大小: {len(all_data)} 字节，已保存到: {self.output_path}")
            
            return all_data
        
        finally:
            self.running = False
            
            # 释放资源
            if self.cap:
                self.cap.release()
                self.cap = None
    
    def extract_in_background(self, callback=None, complete_callback=None):
        """
        在后台线程中提取数据
        
        Args:
            callback: 进度回调函数
            complete_callback: 完成回调函数
        """
        def worker():
            try:
                data = self.extract_data(callback)
                if complete_callback:
                    complete_callback(True, data, None)
            except Exception as e:
                logger.error(f"数据提取错误: {e}", exc_info=True)
                if complete_callback:
                    complete_callback(False, None, str(e))
        
        # 创建并启动工作线程
        thread = threading.Thread(target=worker)
        thread.daemon = True
        thread.start()
        
        return thread
    
    def stop(self):
        """停止解码过程"""
        self.running = False


class ParallelVideoDecoder(VideoDecoder):
    """
    并行视频解码器，使用多线程加速解码过程
    """
    
    def __init__(self, *args, max_workers=None, **kwargs):
        """
        初始化并行视频解码器
        
        Args:
            max_workers: 最大工作线程数
            其他参数与VideoDecoder相同
        """
        super().__init__(*args, **kwargs)
        
        # 设置工作线程数
        cpu_count = os.cpu_count()
        self.max_workers = max_workers or max(1, cpu_count - 1)
        
        # 每个线程处理的帧数
        self.frames_per_worker = 100
    
    def extract_data(self, callback=None):
        """
        并行提取数据
        
        Args:
            callback: 回调函数
            
        Returns:
            提取的数据
        """
        if self.running:
            logger.warning("解码器已在运行")
            return
        
        self.running = True
        self.start_time = time.time()
        
        try:
            # 确保视频已打开
            if self.cap is None or not self.cap.isOpened():
                self._open_video()
            
            # 创建全局进度计数器
            self.processed_frames = 0
            
            # 创建内存缓冲区
            all_data = bytearray()
            
            # 创建线程池
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # 分块处理帧
                for start_frame in range(0, self.total_frames, self.frames_per_worker):
                    end_frame = min(start_frame + self.frames_per_worker, self.total_frames)
                    
                    # 提交任务
                    future = executor.submit(
                        self._process_frame_batch, 
                        start_frame, 
                        end_frame, 
                        callback
                    )
                    
                    # 获取结果并追加到数据缓冲区
                    batch_data = future.result()
                    all_data.extend(batch_data)
                    
                    # 检查是否被中断
                    if not self.running:
                        break
            
            # 应用纠错解码（如果启用）
            if self.use_error_correction and self.error_correction:
                logger.info("应用纠错解码...")
                all_data = self.error_correction.decode_data(bytes(all_data))
            
            # 写入输出文件
            with open(self.output_path, 'wb') as f:
                f.write(all_data)
            
            logger.info(f"数据提取完成，大小: {len(all_data)} 字节，已保存到: {self.output_path}")
            
            return all_data
        
        finally:
            self.running = False
            
            # 释放资源
            if self.cap:
                self.cap.release()
                self.cap = None
    
    def _process_frame_batch(self, start_frame, end_frame, callback):
        """
        处理一批帧
        
        Args:
            start_frame: 起始帧索引
            end_frame: 结束帧索引
            callback: 回调函数
            
        Returns:
            这批帧提取的数据
        """
        # 创建一个新的视频捕获对象
        cap = cv2.VideoCapture(str(self.video_path))
        
        # 跳到起始帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # 处理这一批帧
        batch_data = bytearray()
        frame_count = 0
        
        for frame_idx in range(start_frame, end_frame):
            # 读取一帧
            ret, frame = cap.read()
            if not ret:
                break
            
            # 转换为RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 提取帧数据
            indices = extract_frame_data(
                frame_rgb, self.logical_width, self.logical_height, 
                self.nine_to_one, self.color_lut, self.color_count
            )
            
            # 转换回字节
            frame_bytes = indices_to_bytes(indices, self.color_count)
            
            # 追加到批处理数据
            batch_data.extend(frame_bytes)
            
            # 更新计数
            frame_count += 1
            
            # 更新全局进度
            with threading.Lock():
                self.processed_frames += 1
                
                # 调用回调函数
                if callback and self.processed_frames % 10 == 0:
                    elapsed = time.time() - self.start_time
                    fps = self.processed_frames / elapsed if elapsed > 0 else 0
                    callback(self.processed_frames, self.total_frames, fps)
            
            # 检查是否被中断
            if not self.running:
                break
        
        # 释放资源
        cap.release()
        
        return batch_data
