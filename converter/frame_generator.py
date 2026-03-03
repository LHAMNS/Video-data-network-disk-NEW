"""
高性能帧生成器，使用NumPy和Numba进行优化
实现二进制数据到视频帧的映射

High-performance frame generator using NumPy and Numba optimization
Implements binary data to video frame mapping
"""

import numpy as np
import logging
import base64
import cv2
from . import COLOR_PALETTE_16, COLOR_PALETTE_16_BGR
from .utils import expand_pixels_9x1, bytes_to_color_indices

logger = logging.getLogger(__name__)

# RGB palette array for standard frame generation and preview
COLOR_PALETTE_16_ARRAY = np.array(COLOR_PALETTE_16, dtype=np.uint8)

# BGR palette array for direct AVI writing (avoids per-frame RGB→BGR conversion)
COLOR_PALETTE_16_BGR_ARRAY = np.array(COLOR_PALETTE_16_BGR, dtype=np.uint8)

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
    
    # 从调色板复制已有颜色
    for i in range(min(color_count, len(palette))):
        lut[i] = palette[i]
    
    # 如果需要更多颜色，自动生成
    if color_count > len(palette):
        # 计算颜色空间细分度
        step = 256 // int(np.cbrt(color_count - len(palette)) + 1)
        step = max(step, 1)  # 确保step不为0
        
        idx = len(palette)
        # 在RGB空间均匀分布生成附加颜色
        for r in range(0, 256, step):
            for g in range(0, 256, step):
                for b in range(0, 256, step):
                    if idx < color_count:
                        lut[idx] = np.array([r, g, b], dtype=np.uint8)
                        idx += 1
                    else:
                        break
    
    return lut

def generate_frame_array(data_indices, width, height, lut):
    """
    根据数据索引生成帧数组 - 纯numpy向量化实现

    Uses numpy fancy indexing (lut[indices]) which is implemented in C
    and processes all pixels in a single vectorized operation.
    ~10-100x faster than per-pixel numba loop with bounds checking.

    Args:
        data_indices: 颜色索引数组
        width: 帧宽度
        height: 帧高度
        lut: 颜色查找表

    Returns:
        RGB帧数组，形状为(height, width, 3)
    """
    total_pixels = width * height
    n = min(total_pixels, len(data_indices))

    # Clip indices to valid range (vectorized, replaces per-pixel if-check)
    indices = np.empty(total_pixels, dtype=np.uint8)
    indices[:n] = data_indices[:n]
    if n < total_pixels:
        indices[n:] = 0

    np.clip(indices, 0, len(lut) - 1, out=indices)

    # Single vectorized LUT lookup + reshape: all pixels at once
    return lut[indices].reshape(height, width, 3)

def create_border_pattern(width, height, border_width=10, border_color=1, background_color=0):
    """
    创建带边框的填充模式
    
    Args:
        width: 宽度
        height: 高度
        border_width: 边框宽度
        border_color: 边框颜色索引
        background_color: 背景颜色索引
        
    Returns:
        带边框的索引数组
    """
    pattern = np.full(width * height, background_color, dtype=np.uint8)
    
    # 创建边框
    for i in range(height):
        for j in range(width):
            is_border = (
                i < border_width or 
                i >= height - border_width or
                j < border_width or 
                j >= width - border_width
            )
            if is_border:
                pattern[i * width + j] = border_color
    
    return pattern

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

        # 输入参数验证
        if color_count not in (16, 256):
            logger.warning(f"不支持的颜色数量 {color_count}，将使用默认值 16")
            color_count = 16

        if resolution not in VIDEO_PRESETS:
            logger.warning(f"不支持的分辨率 {resolution}，将使用默认值 4K")
            resolution = "4K"

        # 存储参数
        self.resolution_name = resolution
        self.resolution = VIDEO_PRESETS.get(resolution, VIDEO_PRESETS["4K"])
        self.fps = max(1, min(fps, 120))  # 限制在合理范围内
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

        # 生成颜色查找表 (RGB and BGR)
        self.color_lut = generate_color_lut(COLOR_PALETTE_16_ARRAY, color_count)
        self.color_lut_bgr = generate_color_lut(COLOR_PALETTE_16_BGR_ARRAY, color_count)

        # 计算每帧可存储的字节数
        bits_per_pixel = 4 if color_count == 16 else 8
        self.bytes_per_frame = self.logical_width * self.logical_height * bits_per_pixel // 8

        # Pre-compute for hot path
        self._indices_needed = self.logical_width * self.logical_height

        # 创建边框模式，用于小数据块的展示
        self.border_pattern = create_border_pattern(
            self.logical_width,
            self.logical_height,
            border_width=max(5, min(20, self.logical_width // 30))
        )

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
        if data_size <= 0:
            return 0
        return (data_size + self.bytes_per_frame - 1) // self.bytes_per_frame
    
    def calculate_logical_pixel_count(self):
        """计算每帧的逻辑像素数量"""
        return self.logical_width * self.logical_height
    
    def _generate_frame_impl(self, data_chunk, frame_index, lut):
        """
        Internal frame generation - single implementation for both RGB and BGR.
        Uses vectorized numpy operations throughout for maximum speed.
        """
        # Convert bytes to color indices (vectorized)
        color_indices = bytes_to_color_indices(data_chunk, self.color_count)
        indices_needed = self._indices_needed

        if len(color_indices) < indices_needed:
            if len(color_indices) < indices_needed // 10:
                # Border mode for very small data
                padded_indices = self.border_pattern.copy()
                if len(color_indices) > 0:
                    data_side = max(1, int(np.sqrt(len(color_indices))))
                    start_x = (self.logical_width - data_side) // 2
                    start_y = (self.logical_height - data_side) // 2
                    n = min(len(color_indices), data_side * data_side)
                    # Vectorized center placement
                    ii = np.arange(n)
                    ys = np.clip(start_y + ii // data_side, 0, self.logical_height - 1)
                    xs = np.clip(start_x + ii % data_side, 0, self.logical_width - 1)
                    padded_indices[ys * self.logical_width + xs] = color_indices[:n]
            else:
                # Zero-pad to fill frame
                padded_indices = np.zeros(indices_needed, dtype=np.uint8)
                padded_indices[:len(color_indices)] = color_indices
            color_indices = padded_indices

        # Vectorized LUT lookup + reshape (single numpy operation)
        logical_frame = generate_frame_array(
            color_indices[:indices_needed],
            self.logical_width,
            self.logical_height,
            lut
        )

        # 9-to-1 expansion using np.repeat (vectorized C implementation)
        if self.nine_to_one:
            physical_frame = expand_pixels_9x1(
                logical_frame, self.logical_width, self.logical_height
            )
            if (physical_frame.shape[1] != self.physical_width or
                    physical_frame.shape[0] != self.physical_height):
                corrected = np.zeros((self.physical_height, self.physical_width, 3), dtype=np.uint8)
                h = min(self.physical_height, physical_frame.shape[0])
                w = min(self.physical_width, physical_frame.shape[1])
                corrected[:h, :w] = physical_frame[:h, :w]
                physical_frame = corrected
        else:
            physical_frame = logical_frame

        return physical_frame

    def generate_frame(self, data_chunk, frame_index=0):
        """生成单个RGB视频帧"""
        try:
            return self._generate_frame_impl(data_chunk, frame_index, self.color_lut)
        except Exception as e:
            logger.error(f"生成帧 {frame_index} 时出错: {e}", exc_info=True)
            error_frame = np.zeros((self.physical_height, self.physical_width, 3), dtype=np.uint8)
            error_frame[:, :, 0] = 255
            return error_frame

    def generate_frame_bgr(self, data_chunk, frame_index=0):
        """生成单个BGR视频帧 (直接AVI写入，避免RGB→BGR转换)"""
        try:
            return self._generate_frame_impl(data_chunk, frame_index, self.color_lut_bgr)
        except Exception as e:
            logger.error(f"生成BGR帧 {frame_index} 时出错: {e}", exc_info=True)
            error_frame = np.zeros((self.physical_height, self.physical_width, 3), dtype=np.uint8)
            error_frame[:, :, 2] = 255
            return error_frame

    def generate_preview_image(self, frame, max_size=300):
        """
        生成用于UI预览的小图像
        
        Args:
            frame: 完整帧数组
            max_size: 预览图最大尺寸
            
        Returns:
            Base64编码的JPEG图像数据
        """
        try:
            if frame is None or frame.size == 0:
                logger.warning("无法生成预览：无效的帧数据")
                # 返回小的灰色图像
                empty_frame = np.ones((max_size, max_size, 3), dtype=np.uint8) * 128
                _, jpeg_data = cv2.imencode('.jpg', empty_frame)
                return base64.b64encode(jpeg_data).decode('utf-8')
            
            # 计算缩放比例
            scale = min(max_size / frame.shape[1], max_size / frame.shape[0])
            if scale < 1:
                new_width = max(1, int(frame.shape[1] * scale))
                new_height = max(1, int(frame.shape[0] * scale))
                preview = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
            else:
                preview = frame.copy()
            
            # 转换为BGR (OpenCV格式)
            preview_bgr = cv2.cvtColor(preview, cv2.COLOR_RGB2BGR)
            
            # 编码为JPEG (使用质量参数)
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 85]
            _, jpeg_data = cv2.imencode('.jpg', preview_bgr, encode_params)
            
            # 转换为Base64
            return base64.b64encode(jpeg_data).decode('utf-8')
            
        except Exception as e:
            logger.error(f"生成预览图像时出错: {e}", exc_info=True)
            # 返回小的灰色图像
            empty_frame = np.ones((max_size, max_size, 3), dtype=np.uint8) * 128
            _, jpeg_data = cv2.imencode('.jpg', empty_frame)
            return base64.b64encode(jpeg_data).decode('utf-8')
    
    def generate_frames_from_data(self, data, callback=None, bgr=False):
        """
        从数据生成帧的生成器

        Args:
            data: 字节数据或可迭代的数据块
            callback: 回调函数，用于报告进度，参数为(帧索引，总帧数，当前帧)
            bgr: 如果为True，直接生成BGR帧（用于AVI写入，避免RGB→BGR转换）

        Yields:
            生成的视频帧
        """
        frame_count = 0
        gen_func = self.generate_frame_bgr if bgr else self.generate_frame

        try:
            if isinstance(data, (bytes, bytearray)):
                total_bytes = len(data)
                total_frames = self.estimate_frame_count(total_bytes)

                logger.info(f"开始生成帧，数据大小: {total_bytes} 字节，预计 {total_frames} 帧, BGR={bgr}")

                for frame_idx in range(total_frames):
                    start_pos = frame_idx * self.bytes_per_frame
                    end_pos = min(start_pos + self.bytes_per_frame, total_bytes)
                    current_chunk = data[start_pos:end_pos]

                    frame = gen_func(current_chunk, frame_idx)
                    frame_count += 1

                    if callback:
                        try:
                            callback(frame_idx, total_frames, frame)
                        except Exception as e:
                            logger.error(f"帧回调函数出错: {e}")

                    yield frame

                logger.info(f"帧生成完成，共 {frame_count} 帧")

            else:
                logger.info(f"开始从迭代器生成帧, BGR={bgr}")

                for frame_idx, chunk in enumerate(data):
                    frame = gen_func(chunk, frame_idx)
                    frame_count += 1

                    if callback:
                        try:
                            callback(frame_idx, None, frame)
                        except Exception as e:
                            logger.error(f"帧回调函数出错: {e}")

                    yield frame

                    if frame_idx % 100 == 0:
                        logger.info(f"已生成 {frame_idx + 1} 帧")

                logger.info(f"从迭代器生成帧完成，共 {frame_count} 帧")

        except Exception as e:
            logger.error(f"生成帧时出错: {e}", exc_info=True)
            error_frame = np.zeros((self.physical_height, self.physical_width, 3), dtype=np.uint8)
            error_frame[:, :, 0] = 255

            if frame_count > 0:
                yield error_frame
            else:
                if callback:
                    try:
                        callback(0, 1, error_frame)
                    except:
                        pass
                yield error_frame


class OptimizedFrameGenerator(FrameGenerator):
    """
    更进一步优化的帧生成器
    使用预分配缓冲区和更高级的优化技术
    适用于超高性能场景
    """
    
    def __init__(self, *args, **kwargs):
        """初始化优化帧生成器"""
        super().__init__(*args, **kwargs)

        # Double-buffer: alternate between two buffers so the caller can
        # consume the previous frame while we generate the next one,
        # eliminating the need for .copy() on every yield.
        self._buffers = [
            np.zeros((self.physical_height, self.physical_width, 3), dtype=np.uint8),
            np.zeros((self.physical_height, self.physical_width, 3), dtype=np.uint8),
        ]
        self._buf_idx = 0

        # Pre-allocated index buffer
        self._index_buffer = np.zeros(self._indices_needed, dtype=np.uint8)

        # 用于边框效果的缓存
        self._border_indices = None

        logger.info("创建优化帧生成器，使用双缓冲区")
    
    def _prepare_border_indices(self):
        """准备带边框的索引数组"""
        if self._border_indices is None:
            self._border_indices = create_border_pattern(
                self.logical_width, 
                self.logical_height,
                border_width=max(5, min(20, self.logical_width // 30))
            )
        return self._border_indices
    
    def generate_frame_optimized(self, data_chunk, frame_index=0, bgr=False):
        """
        Optimized frame generation using double-buffered output.
        Returns a buffer that is safe to use until the next call.
        """
        try:
            lut = self.color_lut_bgr if bgr else self.color_lut

            # Get current output buffer and advance for next call
            buf = self._buffers[self._buf_idx]
            self._buf_idx ^= 1  # toggle 0/1

            # Use the unified implementation from parent
            frame = self._generate_frame_impl(data_chunk, frame_index, lut)

            # Copy into our owned buffer so caller has stable memory
            np.copyto(buf, frame)
            return buf

        except Exception as e:
            logger.error(f"优化帧生成时出错: {e}", exc_info=True)
            buf = self._buffers[self._buf_idx]
            self._buf_idx ^= 1
            buf.fill(0)
            buf[:, :, 2 if bgr else 0] = 255
            return buf
    
    def generate_frames_from_data(self, data, callback=None, bgr=False):
        """
        Optimized frame generator using double-buffered output.
        No .copy() needed - double buffer ensures each yielded frame
        is stable until the next frame is yielded.
        """
        frame_count = 0

        try:
            if isinstance(data, (bytes, bytearray)):
                total_bytes = len(data)
                total_frames = self.estimate_frame_count(total_bytes)

                for frame_idx in range(total_frames):
                    start_pos = frame_idx * self.bytes_per_frame
                    end_pos = min(start_pos + self.bytes_per_frame, total_bytes)

                    frame = self.generate_frame_optimized(data[start_pos:end_pos], frame_idx, bgr=bgr)
                    frame_count += 1

                    if callback:
                        try:
                            callback(frame_idx, total_frames, frame)
                        except Exception as e:
                            logger.error(f"帧回调函数出错: {e}")

                    yield frame

            else:
                for frame_idx, chunk in enumerate(data):
                    frame = self.generate_frame_optimized(chunk, frame_idx, bgr=bgr)
                    frame_count += 1

                    if callback:
                        try:
                            callback(frame_idx, None, frame)
                        except Exception as e:
                            logger.error(f"帧回调函数出错: {e}")

                    yield frame

        except Exception as e:
            logger.error(f"[优化] 生成帧时出错: {e}", exc_info=True)
            error_frame = np.zeros((self.physical_height, self.physical_width, 3), dtype=np.uint8)
            error_frame[:, :, 2 if bgr else 0] = 255
            yield error_frame

    def process_large_file(self, data_generator, progress_callback=None, bgr=False):
        """
        处理大文件的特殊优化方法

        Args:
            data_generator: 数据块生成器
            progress_callback: 进度回调函数
            bgr: 如果为True，直接生成BGR帧（用于AVI写入，避免RGB→BGR转换）

        Yields:
            生成的视频帧
        """
        frame_idx = 0
        buffer = bytearray()
        bytes_processed = 0
        start_time = time.time()
        
        try:
            logger.info(f"开始优化处理大文件... BGR={bgr}")
            
            for chunk in data_generator:
                # 添加到缓冲区
                chunk_size = len(chunk)
                buffer.extend(chunk)
                bytes_processed += chunk_size
                
                # 当缓冲区达到帧所需大小时处理
                while len(buffer) >= self.bytes_per_frame:
                    # 取出一帧所需数据
                    frame_data = buffer[:self.bytes_per_frame]
                    buffer = buffer[self.bytes_per_frame:]
                    
                    # 生成帧
                    frame = self.generate_frame_optimized(frame_data, frame_idx, bgr=bgr)
                    
                    # 提供进度信息
                    if progress_callback:
                        try:
                            # 创建副本避免缓冲区修改
                            progress_callback(frame_idx, None, frame.copy())
                        except Exception as e:
                            logger.error(f"进度回调错误: {e}")
                    
                    # 定期记录进度
                    if frame_idx % 50 == 0:
                        elapsed = time.time() - start_time
                        fps = frame_idx / elapsed if elapsed > 0 else 0
                        logger.info(f"已处理 {frame_idx} 帧, {bytes_processed/1024/1024:.2f} MB, {fps:.2f} fps")
                    
                    frame_idx += 1
                    yield frame
            
            # 处理剩余数据
            if buffer:
                frame = self.generate_frame_optimized(buffer, frame_idx, bgr=bgr)
                
                if progress_callback:
                    try:
                        progress_callback(frame_idx, frame_idx + 1, frame.copy())
                    except Exception as e:
                        logger.error(f"最终帧进度回调错误: {e}")
                
                yield frame
                frame_idx += 1
            
            elapsed = time.time() - start_time
            fps = frame_idx / elapsed if elapsed > 0 else 0
            logger.info(f"大文件处理完成: 共 {frame_idx} 帧, {bytes_processed/1024/1024:.2f} MB, 平均 {fps:.2f} fps")
            
        except Exception as e:
            logger.error(f"处理大文件时出错: {e}", exc_info=True)
            # 返回错误指示帧
            error_frame = np.zeros((self.physical_height, self.physical_width, 3), dtype=np.uint8)
            if bgr:
                error_frame[:, :, 2] = 255  # Red in BGR = channel 2
            else:
                error_frame[:, :, 0] = 255  # Red in RGB = channel 0
            
            # 如果已经生成了一些帧，只需返回错误帧
            if frame_idx > 0:
                if progress_callback:
                    try:
                        progress_callback(frame_idx, None, error_frame)
                    except:
                        pass
                yield error_frame
            else:
                # 至少生成一帧
                if progress_callback:
                    try:
                        progress_callback(0, 1, error_frame)
                    except:
                        pass
                yield error_frame