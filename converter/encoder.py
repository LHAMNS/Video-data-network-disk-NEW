"""
视频编码模块，使用FFmpeg进行高性能编码
支持多线程并行处理和硬件加速
"""

import os
import re
import sys
import time
import queue
import signal
import shutil
import logging
import threading
import subprocess
import json
import cv2
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# 从converter模块导入常量
from converter import (
    ROOT_DIR, OUTPUT_DIR, VIDEO_PRESETS,
    MIN_FPS, MAX_FPS, DEFAULT_FPS
)

from .utils import is_nvenc_available, is_qsv_available

logger = logging.getLogger(__name__)

class VideoEncoder:
    """
    视频编码器基类，提供通用接口和功能
    """
    
    def __init__(self, width, height, fps=30, output_path=None, quality="high", 
                hardware_acceleration=True, compression_method="h264"):
        """
        初始化视频编码器
        
        Args:
            width: 视频宽度
            height: 视频高度
            fps: 帧率
            output_path: 输出文件路径
            quality: 视频质量 ("high", "medium", "low")
            hardware_acceleration: 是否使用硬件加速
            compression_method: 压缩方式 ("h264", "h265", "av1")
        """
        self.width = width
        self.height = height
        self.fps = max(MIN_FPS, min(fps, MAX_FPS))
        
        # 默认输出路径
        if output_path is None:
            output_path = OUTPUT_DIR / f"output_{int(time.time())}.mp4"
        self.output_path = Path(output_path)
        
        # 质量和压缩设置
        self.quality = quality
        self.hardware_acceleration = hardware_acceleration
        self.compression_method = compression_method
        
        # 检测可用的硬件加速
        self.nvenc_available = is_nvenc_available()
        self.qsv_available = is_qsv_available()
        
        # 运行时状态
        self.running = False
        self.frames_processed = 0
    
    def start(self):
        """启动视频编码器"""
        raise NotImplementedError("子类必须实现此方法")
    
    def add_frame(self, frame):
        """
        添加帧到编码队列
        
        Args:
            frame: 要编码的帧 (numpy数组)
            
        Returns:
            bool: 是否成功添加
        """
        raise NotImplementedError("子类必须实现此方法")
    
    def stop(self):
        """
        停止编码器并完成视频
        
        Returns:
            dict: 编码统计信息
        """
        raise NotImplementedError("子类必须实现此方法")
    
    def _build_ffmpeg_command(self):
        """
        构建FFmpeg命令行
        
        Returns:
            list: FFmpeg命令行参数列表
        """
        # 基本命令
        cmd = [
            "ffmpeg",
            "-y",  # 覆盖输出文件
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{self.width}x{self.height}",
            "-pix_fmt", "bgr24",
            "-r", str(self.fps),
            "-i", "-",  # 从stdin读取
        ]
        
        # 视频质量设置
        quality_presets = {
            "high": {"h264": "18", "h265": "22", "av1": "30"},
            "medium": {"h264": "23", "h265": "27", "av1": "45"},
            "low": {"h264": "28", "h265": "32", "av1": "60"}
        }
        
        # 获取对应的CRF值
        crf = quality_presets.get(self.quality, quality_presets["medium"]).get(
            self.compression_method, quality_presets["medium"]["h264"]
        )
        
        # 编码器和硬件加速设置
        if self.hardware_acceleration:
            if self.compression_method == "h264":
                if self.nvenc_available:
                    cmd.extend(["-c:v", "h264_nvenc", "-preset", "p7", "-rc:v", "vbr_hq", "-qmin", "0", "-qmax", "50", "-b:v", "0"])
                elif self.qsv_available:
                    cmd.extend(["-c:v", "h264_qsv", "-global_quality", crf, "-preset", "veryslow"])
                else:
                    cmd.extend(["-c:v", "libx264", "-crf", crf, "-preset", "medium"])
            elif self.compression_method == "h265":
                if self.nvenc_available:
                    cmd.extend(["-c:v", "hevc_nvenc", "-preset", "p7", "-rc:v", "vbr_hq", "-qmin", "0", "-qmax", "50", "-b:v", "0"])
                elif self.qsv_available:
                    cmd.extend(["-c:v", "hevc_qsv", "-global_quality", crf, "-preset", "veryslow"])
                else:
                    cmd.extend(["-c:v", "libx265", "-crf", crf, "-preset", "medium"])
            elif self.compression_method == "av1":
                # AV1编码通常无硬件加速
                cmd.extend(["-c:v", "libaom-av1", "-crf", crf, "-strict", "experimental", "-cpu-used", "5"])
            else:
                # 未知压缩方式，使用h264
                cmd.extend(["-c:v", "libx264", "-crf", crf, "-preset", "medium"])
        else:
            # 软件编码
            if self.compression_method == "h264":
                cmd.extend(["-c:v", "libx264", "-crf", crf, "-preset", "medium"])
            elif self.compression_method == "h265":
                cmd.extend(["-c:v", "libx265", "-crf", crf, "-preset", "medium"])
            elif self.compression_method == "av1":
                cmd.extend(["-c:v", "libaom-av1", "-crf", crf, "-strict", "experimental", "-cpu-used", "5"])
            else:
                cmd.extend(["-c:v", "libx264", "-crf", crf, "-preset", "medium"])
        
        # 添加输出文件和其他设置
        cmd.extend([
            "-pix_fmt", "yuv420p",  # 兼容性设置
            "-movflags", "faststart",  # 优化Web观看
            str(self.output_path)
        ])
        
        return cmd


class StreamingVideoEncoder(VideoEncoder):
    """
    流式视频编码器，使用FFmpeg接收连续帧流
    适用于实时生成帧的情况
    """
    
    def __init__(self, width, height, fps=30, output_path=None, quality="high", 
                hardware_acceleration=True, compression_method="h264",
                max_queue_size=100, acceleration_mode=True):
        """
        初始化流式视频编码器
        
        Args:
            width: 视频宽度
            height: 视频高度
            fps: 帧率
            output_path: 输出文件路径
            quality: 视频质量 ("high", "medium", "low")
            hardware_acceleration: 是否使用硬件加速
            compression_method: 压缩方式 ("h264", "h265", "av1")
            max_queue_size: 最大队列大小
            acceleration_mode: 是否启用加速模式
        """
        super().__init__(width, height, fps, output_path, quality, 
                        hardware_acceleration, compression_method)
        
        # 启用加速模式，优化性能
        self.acceleration_mode = acceleration_mode
        
        # 帧队列和线程
        self.max_queue_size = max_queue_size
        self.frame_queue = queue.Queue(maxsize=max_queue_size)
        self.process = None
        self.encoding_thread = None
        self.error_thread = None
        self.input_ended = threading.Event()  # 输入结束信号
        
        # 统计信息
        self.encode_start_time = 0
        self.last_frame_time = 0
        
        # 并发控制
        max_workers = os.cpu_count() * 2 if acceleration_mode else max(1, os.cpu_count() // 2)
        self.worker_threads = max_workers
        
        # 初始化统计
        logger.info(f"并行视频编码器初始化: {width}x{height}, {fps}fps, 最大线程数: {max_workers}, "
                 f"硬件加速: NVENC={self.nvenc_available}, QSV={self.qsv_available}, "
                 f"压缩方式: {compression_method}, 加速模式: {acceleration_mode}")
    
    def start(self):
        """启动视频编码器"""
        if self.running:
            logger.warning("编码器已在运行")
            return False
        
        # 重置标志
        self.running = True
        self.input_ended.clear()  # 重置结束标志
        self.frames_processed = 0
        
        try:
            # 确保输出目录存在
            output_dir = os.path.dirname(self.output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            # 构建FFmpeg命令
            ffmpeg_cmd = self._build_ffmpeg_command()
            
            # 移除可能存在的旧文件
            if os.path.exists(self.output_path):
                try:
                    os.unlink(self.output_path)
                    logger.info(f"已删除现有文件: {self.output_path}")
                except:
                    logger.warning(f"无法删除现有文件: {self.output_path}")
            
            logger.info(f"启动FFmpeg进程: {' '.join(ffmpeg_cmd)}")
            
            # 使用管道模式启动FFmpeg
            self.process = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE,  # 捕获输出以防止阻塞
                bufsize=10*1024*1024  # 10MB缓冲区
            )
            
            # 启动错误监控线程
            self.error_thread = threading.Thread(
                target=self._monitor_ffmpeg_errors,
                daemon=True
            )
            self.error_thread.start()
            
            # 启动帧编码线程
            self.encoding_thread = threading.Thread(
                target=self._encode_frames,
                daemon=True
            )
            self.encoding_thread.start()
            
            # 等待一小段时间确保进程启动
            time.sleep(0.5)
            
            # 检查进程是否仍在运行
            if self.process.poll() is not None:
                error = self.process.stderr.read().decode('utf-8', errors='ignore')
                logger.error(f"FFmpeg进程未能启动: {error}")
                self.running = False
                return False
            
            logger.info(f"并行视频编码器已启动，输出: {self.output_path}")
            return True
            
        except Exception as e:
            logger.error(f"启动视频编码器时出错: {e}", exc_info=True)
            self.running = False
            return False
    
    def add_frame(self, frame):
        """
        添加帧到编码队列
        
        Args:
            frame: 要编码的帧 (numpy数组)
            
        Returns:
            bool: 是否成功添加
        """
        if not self.running:
            logger.warning("编码器未运行")
            return False
        
        if self.process is None or self.process.poll() is not None:
            logger.error("FFmpeg进程未运行")
            return False
        
        try:
            # 不阻塞，如果队列已满则等待短暂时间
            timeout = 0.1 if self.acceleration_mode else 5.0
            self.frame_queue.put(frame, timeout=timeout)
            self.last_frame_time = time.time()
            return True
        except queue.Full:
            logger.warning("编码队列已满，跳过帧")
            return False
        except Exception as e:
            logger.error(f"添加帧到队列时出错: {e}")
            return False
    
    def _encode_frames(self):
        """编码帧的工作线程"""
        try:
            logger.info(f"开始编码线程，输出到 {self.output_path}")
            
            while self.running or not self.frame_queue.empty():
                try:
                    # 使用更短的超时时间，使其能更快响应关闭命令
                    frame = self.frame_queue.get(timeout=0.1)
                    
                    # 编码帧
                    if frame is not None and self.process and self.process.poll() is None:
                        try:
                            if isinstance(frame, np.ndarray):
                                # 确保帧是正确的形状和类型
                                h, w = frame.shape[:2]
                                if h != self.height or w != self.width:
                                    logger.warning(f"帧尺寸不匹配: 期望 {self.width}x{self.height}, 实际 {w}x{h}")
                                    # 调整大小
                                    frame = cv2.resize(frame, (self.width, self.height))
                                    
                                # 确保是BGR格式 (for OpenCV)
                                if frame.shape[2] == 3:  # 检查是否为3通道
                                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                                else:
                                    frame_bgr = frame
                                    
                                # 编码帧
                                ret, encoded_frame = cv2.imencode('.bmp', frame_bgr)
                                if ret:
                                    # 写入FFmpeg进程
                                    self.process.stdin.write(encoded_frame.tobytes())
                                    self.process.stdin.flush()  # 确保数据被写入
                                    self.frames_processed += 1
                                else:
                                    logger.error("帧编码失败")
                            else:
                                logger.warning(f"不支持的帧类型: {type(frame)}")
                        except BrokenPipeError:
                            logger.error("管道已关闭，无法写入更多帧")
                            self.running = False
                            break
                        except Exception as e:
                            logger.error(f"写入帧时出错: {e}")
                            
                except queue.Empty:
                    # 不要因为队列为空就退出
                    # 只有在明确被告知停止且队列为空时才退出
                    if not self.running and self.frame_queue.empty():
                        logger.info("编码器已停止，队列为空，编码完成")
                        break
                    continue
                except Exception as e:
                    logger.error(f"获取帧时出错: {e}")
                    continue
            
            logger.info("编码线程结束")
            
        except Exception as e:
            logger.error(f"编码线程异常: {e}", exc_info=True)
        finally:
            # 只记录我们正在退出，但不更改运行状态
            # 这将由stop()方法管理
            logger.info("编码线程退出")
    
    def _monitor_ffmpeg_errors(self):
        """监控FFmpeg的错误输出"""
        try:
            if not self.process or not self.process.stderr:
                return
                
            # 持续读取错误输出
            for line in iter(self.process.stderr.readline, b''):
                if not line:
                    break
                    
                error_line = line.decode('utf-8', errors='ignore').strip()
                if error_line:
                    # 只记录警告和错误
                    if 'error' in error_line.lower():
                        logger.error(f"FFmpeg错误: {error_line}")
                    elif 'warning' in error_line.lower():
                        logger.warning(f"FFmpeg警告: {error_line}")
                    else:
                        logger.debug(f"FFmpeg: {error_line}")
                        
            logger.info("FFmpeg错误监控结束")
        except Exception as e:
            logger.error(f"监控FFmpeg错误时出错: {e}")
    
    def _process_all_remaining_frames(self):
        """处理队列中所有剩余的帧"""
        if not self.process:
            return
            
        # 处理所有剩余的帧，直到队列为空
        frames_processed = 0
        while not self.frame_queue.empty():
            try:
                frame = self.frame_queue.get_nowait()
                if frame is not None:
                    # 从RGB转换为BGR (OpenCV格式)
                    if isinstance(frame, np.ndarray) and frame.shape[2] == 3:
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    else:
                        frame_bgr = frame
                        
                    # 编码和写入
                    ret, encoded_frame = cv2.imencode('.bmp', frame_bgr)
                    if ret and self.process.poll() is None:
                        self.process.stdin.write(encoded_frame.tobytes())
                        frames_processed += 1
            except queue.Empty:
                break
            except Exception as e:
                logger.error(f"处理剩余帧时出错: {e}")
                break
                
        logger.info(f"处理了 {frames_processed} 个剩余帧")
    
    def _repair_output_file(self):
        """尝试修复输出视频文件"""
        try:
            # 创建一个临时修复文件
            temp_file = self.output_path.with_suffix(".fixed.mp4")
            
            # 使用FFmpeg的faststart选项来修复文件
            cmd = [
                "ffmpeg",
                "-v", "warning",
                "-i", str(self.output_path),
                "-c", "copy",
                "-movflags", "faststart",  # 这是修复MP4的关键标志
                str(temp_file)
            ]
            
            logger.info(f"运行修复命令: {' '.join(cmd)}")
            process = subprocess.run(
                cmd, 
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE,
                timeout=60
            )
            
            if process.returncode == 0 and temp_file.exists() and temp_file.stat().st_size > 0:
                # 修复成功，替换原文件
                logger.info(f"视频文件修复成功，用新文件替换旧文件")
                shutil.move(str(temp_file), str(self.output_path))
                return True
            else:
                logger.warning(f"视频文件修复失败，返回码: {process.returncode}")
                if temp_file.exists():
                    temp_file.unlink()
                return False
        except Exception as e:
            logger.error(f"修复输出文件时出错: {e}")
            if 'temp_file' in locals() and temp_file.exists():
                temp_file.unlink()
            return False
    
    def stop(self):
        """停止流式编码器并返回编码统计信息"""
        if not self.running:
            logger.warning("编码器未运行")
            # 即使它没有运行，也尝试安全地关闭进程（如果存在）
            if self.process:
                try:
                    if hasattr(self.process, 'stdin') and self.process.stdin:
                        logger.info("FFmpeg进程未运行但存在，尝试正常关闭...")
                        self.process.stdin.close()
                        self.process.wait(timeout=10)
                except Exception as e:
                    logger.warning(f"关闭非运行FFmpeg进程时出错: {e}")
            return None
        
        # 发送终止信号
        logger.info("停止流式编码器...")
        self.running = False
        self.input_ended.set()
        
        # 处理任何剩余帧
        remaining_frames = 0
        try:
            remaining_frames = self.frame_queue.qsize()
        except:
            pass
            
        if remaining_frames > 0:
            logger.info(f"处理剩余 {remaining_frames} 个待处理帧...")
            try:
                self._process_all_remaining_frames()
            except Exception as e:
                logger.error(f"处理剩余帧时出错: {e}")
        
        # 等待编码线程结束
        if self.encoding_thread and self.encoding_thread.is_alive():
            logger.info("等待编码线程结束...")
            self.encoding_thread.join(timeout=15)
            if self.encoding_thread.is_alive():
                logger.warning("编码线程未在超时时间内结束")
        
        # 确保在关闭之前刷新所有数据
        if self.process and hasattr(self.process, 'stdin') and self.process.stdin:
            try:
                logger.info("刷新最后的数据到FFmpeg...")
                self.process.stdin.flush()  # 确保所有数据都被写入
            except:
                pass
        
        # 关闭FFmpeg进程
        if self.process:
            try:
                # 正常关闭stdin会让FFmpeg完成处理
                if hasattr(self.process, 'stdin') and self.process.stdin:
                    logger.info("关闭FFmpeg输入流...")
                    self.process.stdin.close()
                
                # 给FFmpeg更多时间来完成视频
                logger.info("等待FFmpeg完成视频文件...")
                self.process.wait(timeout=60)  # 增加超时时间到60秒
                logger.info(f"FFmpeg进程已完成，返回码: {self.process.returncode}")
            except subprocess.TimeoutExpired:
                logger.warning("FFmpeg进程未在超时时间内结束，强制终止")
                try:
                    self.process.terminate()
                    self.process.wait(timeout=5)
                except:
                    logger.error("无法终止FFmpeg进程")
                    if self.process:
                        self.process.kill()
            except Exception as e:
                logger.error(f"关闭FFmpeg进程时出错: {e}")
                if self.process:
                    self.process.kill()
        
        # 尝试修复和验证输出文件
        try:
            if self.output_path.exists() and self.output_path.stat().st_size > 0:
                logger.info(f"尝试修复输出文件: {self.output_path}")
                self._repair_output_file()
        except Exception as e:
            logger.error(f"修复输出文件时出错: {e}")
        
        # 生成并返回编码统计信息
        stats = {
            "frames_processed": self.frames_processed,
            "duration_seconds": self.frames_processed / self.fps if self.fps > 0 else 0,
            "output_path": str(self.output_path),
            "output_size": self.output_path.stat().st_size if self.output_path.exists() else 0
        }
        
        # 写入元数据文件
        try:
            metadata_path = self.output_path.with_suffix(self.output_path.suffix + "_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(stats, f, indent=2)
            logger.info(f"元数据已写入: {metadata_path}")
        except Exception as e:
            logger.error(f"写入元数据时出错: {e}")
        
        return stats


class BatchVideoEncoder(VideoEncoder):
    """
    批量视频编码器，针对一次性处理大量帧优化
    使用并行处理提高性能
    """
    
    def __init__(self, width, height, fps=30, output_path=None, quality="high", 
                hardware_acceleration=True, compression_method="h264",
                max_workers=None, chunk_size=100, acceleration_mode=True):
        """
        初始化批量视频编码器
        
        Args:
            width: 视频宽度
            height: 视频高度
            fps: 帧率
            output_path: 输出文件路径
            quality: 视频质量 ("high", "medium", "low")
            hardware_acceleration: 是否使用硬件加速
            compression_method: 压缩方式 ("h264", "h265", "av1")
            max_workers: 最大工作线程数
            chunk_size: 每个工作块的帧数
            acceleration_mode: 是否启用加速模式
        """
        super().__init__(width, height, fps, output_path, quality, 
                        hardware_acceleration, compression_method)
        
        # 设置工作线程数
        cpu_count = os.cpu_count() or 4
        self.max_workers = max_workers or (cpu_count * 2 if acceleration_mode else max(1, cpu_count // 2))
        self.chunk_size = chunk_size
        self.acceleration_mode = acceleration_mode
        
        # 帧存储和状态
        self.frames = []
        self.temp_dir = None
        self.frame_index = 0
        
        logger.info(f"批量视频编码器初始化: {width}x{height}, {fps}fps, 线程: {self.max_workers}, "
                 f"块大小: {chunk_size}, 硬件加速: {hardware_acceleration}, 加速模式: {acceleration_mode}")
    
    def start(self):
        """准备批量编码"""
        if self.running:
            logger.warning("编码器已在运行")
            return False
        
        # 重置状态
        self.running = True
        self.frames = []
        self.frame_index = 0
        self.frames_processed = 0
        
        # 创建临时目录
        try:
            self.temp_dir = Path(f"{self.output_path}.temp")
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir, ignore_errors=True)
            self.temp_dir.mkdir(exist_ok=True)
            
            logger.info(f"批量编码器已启动，临时目录: {self.temp_dir}")
            return True
        except Exception as e:
            logger.error(f"启动批量编码器时出错: {e}", exc_info=True)
            self.running = False
            return False
    
    def add_frame(self, frame):
        """添加帧到批量处理队列"""
        if not self.running:
            logger.warning("编码器未运行")
            return False
        
        if frame is None:
            logger.warning("尝试添加空帧")
            return False
        
        try:
            # 验证帧
            if not isinstance(frame, np.ndarray):
                logger.warning(f"不支持的帧类型: {type(frame)}")
                return False
            
            # 调整尺寸（如果需要）
            h, w = frame.shape[:2]
            if h != self.height or w != self.width:
                logger.debug(f"调整帧尺寸: {w}x{h} -> {self.width}x{self.height}")
                frame = cv2.resize(frame, (self.width, self.height))
            
            # 保存帧到内存或临时文件
            if self.acceleration_mode:
                # 加速模式：直接保存到临时文件
                frame_path = self.temp_dir / f"frame_{self.frame_index:08d}.png"
                
                # 确保是BGR格式
                if frame.shape[2] == 3:  # 检查是否为3通道
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:
                    frame_bgr = frame
                    
                # 保存为PNG文件
                cv2.imwrite(str(frame_path), frame_bgr)
            else:
                # 常规模式：保存到内存
                self.frames.append(frame)
            
            self.frame_index += 1
            return True
        except Exception as e:
            logger.error(f"添加帧时出错: {e}", exc_info=True)
            return False
    
    def _write_frames_to_temp(self):
        """将内存中的帧写入临时文件"""
        logger.info(f"将 {len(self.frames)} 帧写入临时文件...")
        
        # 确保临时目录存在
        if not self.temp_dir or not self.temp_dir.exists():
            self.temp_dir = Path(f"{self.output_path}.temp")
            self.temp_dir.mkdir(exist_ok=True)
        
        # 并行写入帧
        def write_frame_chunk(chunk_frames, start_idx):
            for i, frame in enumerate(chunk_frames):
                frame_idx = start_idx + i
                frame_path = self.temp_dir / f"frame_{frame_idx:08d}.png"
                
                # 确保是BGR格式
                if frame.shape[2] == 3:  # 检查是否为3通道
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:
                    frame_bgr = frame
                    
                # 保存为PNG文件
                cv2.imwrite(str(frame_path), frame_bgr)
        
        # 分块并行处理
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            chunk_size = max(1, min(100, len(self.frames) // (self.max_workers * 2)))
            futures = []
            
            for i in range(0, len(self.frames), chunk_size):
                chunk = self.frames[i:i+chunk_size]
                future = executor.submit(write_frame_chunk, chunk, i)
                futures.append(future)
            
            # 等待所有任务完成
            for future in futures:
                future.result()
        
        logger.info("所有帧已写入临时文件")
    
    def _generate_video_from_frames(self):
        """从临时帧生成视频"""
        logger.info("从临时帧生成视频...")
        
        # 构建FFmpeg命令
        cmd = [
            "ffmpeg",
            "-y",
            "-framerate", str(self.fps),
            "-i", str(self.temp_dir / "frame_%08d.png")
        ]
        
        # 添加编码设置
        if self.hardware_acceleration:
            if self.compression_method == "h264":
                if self.nvenc_available:
                    cmd.extend(["-c:v", "h264_nvenc", "-preset", "p7", "-rc:v", "vbr_hq"])
                elif self.qsv_available:
                    cmd.extend(["-c:v", "h264_qsv", "-preset", "veryslow"])
                else:
                    cmd.extend(["-c:v", "libx264", "-crf", "23", "-preset", "medium"])
            elif self.compression_method == "h265":
                if self.nvenc_available:
                    cmd.extend(["-c:v", "hevc_nvenc", "-preset", "p7", "-rc:v", "vbr_hq"])
                elif self.qsv_available:
                    cmd.extend(["-c:v", "hevc_qsv", "-preset", "veryslow"])
                else:
                    cmd.extend(["-c:v", "libx265", "-crf", "28", "-preset", "medium"])
            else:
                # 默认使用H.264
                cmd.extend(["-c:v", "libx264", "-crf", "23", "-preset", "medium"])
        else:
            # 软件编码
            if self.compression_method == "h264":
                cmd.extend(["-c:v", "libx264", "-crf", "23", "-preset", "medium"])
            elif self.compression_method == "h265":
                cmd.extend(["-c:v", "libx265", "-crf", "28", "-preset", "medium"])
            else:
                cmd.extend(["-c:v", "libx264", "-crf", "23", "-preset", "medium"])
        
        # 添加输出文件
        cmd.extend([
            "-pix_fmt", "yuv420p",
            "-movflags", "faststart",
            str(self.output_path)
        ])
        
        logger.info(f"执行FFmpeg命令: {' '.join(cmd)}")
        
        try:
            # 运行FFmpeg命令
            process = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
            
            # 检查输出
            if self.output_path.exists() and self.output_path.stat().st_size > 0:
                logger.info(f"视频生成成功: {self.output_path}")
                return True
            else:
                logger.error("FFmpeg运行成功但输出文件无效")
                return False
        except subprocess.CalledProcessError as e:
            logger.error(f"运行FFmpeg命令时出错: {e}")
            logger.error(f"FFmpeg错误输出: {e.stderr.decode('utf-8', errors='ignore')}")
            return False
        except Exception as e:
            logger.error(f"生成视频时出错: {e}", exc_info=True)
            return False
    
    def stop(self):
        """完成批量编码并生成视频"""
        if not self.running:
            logger.warning("编码器未运行")
            return None
        
        try:
            logger.info(f"停止批量编码器，处理 {self.frame_index} 帧...")
            
            # 标记为不运行
            self.running = False
            
            # 将内存中的帧写入临时文件
            if self.frames:
                self._write_frames_to_temp()
                self.frames = []  # 释放内存
            
            # 检查是否有帧需要处理
            if self.frame_index == 0:
                logger.warning("没有帧需要处理")
                return None
            
            # 从临时帧生成视频
            success = self._generate_video_from_frames()
            
            # 设置处理帧数
            self.frames_processed = self.frame_index
            
            # 生成统计信息
            stats = {
                "frames_processed": self.frames_processed,
                "duration_seconds": self.frames_processed / self.fps if self.fps > 0 else 0,
                "output_path": str(self.output_path),
                "success": success
            }
            
            # 写入元数据文件
            try:
                metadata_path = self.output_path.with_suffix(self.output_path.suffix + "_metadata.json")
                with open(metadata_path, 'w') as f:
                    json.dump(stats, f, indent=2)
            except Exception as e:
                logger.error(f"写入元数据时出错: {e}")
            
            # 清理临时文件
            self._cleanup_temp_files()
            
            return stats
        except Exception as e:
            logger.error(f"停止批量编码器时出错: {e}", exc_info=True)
            return None
        finally:
            # 确保清理临时文件
            self._cleanup_temp_files()
    
    def _cleanup_temp_files(self):
        """清理临时文件"""
        try:
            if self.temp_dir and self.temp_dir.exists():
                logger.info(f"清理临时目录: {self.temp_dir}")
                shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception as e:
            logger.warning(f"清理临时文件时出错: {e}")