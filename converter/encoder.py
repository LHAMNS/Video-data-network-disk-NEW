"""
高性能视频编码器，使用FFmpeg和硬件加速
支持NVENC、QSV等硬件编码器
"""

import subprocess
import os
import time
import logging
import threading
import queue
import numpy as np
from pathlib import Path
import shutil
import json
import psutil
from .utils import is_nvenc_available, is_qsv_available

logger = logging.getLogger(__name__)

class VideoEncoder:
    """
    高性能视频编码器，使用FFmpeg流式处理帧数据
    """
    
    def __init__(self, width, height, fps=30, output_path=None, quality="high"):
        """
        初始化视频编码器
        
        Args:
            width: 视频宽度
            height: 视频高度
            fps: 帧率
            output_path: 输出文件路径
            quality: 质量预设 ("high", "medium", "low")
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.output_path = output_path or Path("output/output.mp4")
        self.quality = quality
        self.process = None
        self.start_time = None
        self.frames_written = 0
        self.running = False
        self.frame_queue = queue.Queue(maxsize=100)  # 帧缓冲队列
        self.writer_thread = None
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        
        # 检测可用的硬件编码器
        self.nvenc_available = is_nvenc_available()
        self.qsv_available = is_qsv_available()
        
        # 根据系统状态选择最佳编码参数
        self._configure_encoding_params()
        
        logger.info(f"视频编码器初始化完成: {width}x{height}, {fps}fps, "
                 f"硬件加速: NVENC={self.nvenc_available}, QSV={self.qsv_available}")
    
    def _configure_encoding_params(self):
        """根据系统状态和质量设置配置编码参数"""
        # 基础参数
        self.encoding_params = {
            "high": {
                "crf": 17,
                "preset": "slow" if not self.nvenc_available else "p7",
                "profile": "high",
                "pix_fmt": "yuv420p",
                "g": 15,  # GOP大小
                "tune": "film" if not self.nvenc_available else None,
            },
            "medium": {
                "crf": 22,
                "preset": "medium" if not self.nvenc_available else "p4",
                "profile": "high",
                "pix_fmt": "yuv420p",
                "g": 25,
                "tune": None,
            },
            "low": {
                "crf": 28,
                "preset": "faster" if not self.nvenc_available else "p1",
                "profile": "main",
                "pix_fmt": "yuv420p",
                "g": 30,
                "tune": "fastdecode" if not self.nvenc_available else None,
            }
        }
        
        # NVENC特定参数
        self.nvenc_params = {
            "high": {
                "rc": "constqp",
                "qp": 19,
                "spatial-aq": 1,
                "temporal-aq": 1,
                "b_ref_mode": 1,
            },
            "medium": {
                "rc": "vbr",
                "qmin": 19,
                "qmax": 25,
                "spatial-aq": 1,
            },
            "low": {
                "rc": "vbr",
                "qmin": 25,
                "qmax": 32,
            }
        }
        
        # 根据系统状态自适应调整
        # 检测可用内存和CPU
        mem = psutil.virtual_memory()
        cpu_count = psutil.cpu_count(logical=False)
        
        # 如果系统资源充足，提升某些参数
        if mem.available > 8 * 1024 * 1024 * 1024 and cpu_count >= 4:  # 8GB+ RAM和4+核
            if not self.nvenc_available:
                self.encoding_params["high"]["preset"] = "veryslow"
                self.encoding_params["medium"]["preset"] = "slow"
        
        # 如果资源有限，降低某些参数
        if mem.available < 2 * 1024 * 1024 * 1024 or cpu_count <= 2:  # 2GB- RAM或2-核
            if not self.nvenc_available:
                self.encoding_params["high"]["preset"] = "medium"
                self.encoding_params["medium"]["preset"] = "faster"
                self.encoding_params["low"]["preset"] = "veryfast"
    
    def _build_ffmpeg_command(self):
        """构建ffmpeg命令行"""
        base_cmd = [
            "ffmpeg",
            "-y",  # 覆盖输出文件
            "-f", "rawvideo",
            "-pixel_format", "rgb24",
            "-video_size", f"{self.width}x{self.height}",
            "-framerate", str(self.fps),
            "-i", "pipe:0",  # 从stdin读取输入
        ]
        
        # 获取当前质量的编码参数
        params = self.encoding_params[self.quality]
        
        # 添加编码器选项
        if self.nvenc_available:
            encoder = "h264_nvenc"
            codec_options = ["-c:v", encoder]
            
            # 添加NVENC特定参数
            nvenc_opts = self.nvenc_params[self.quality]
            for k, v in nvenc_opts.items():
                if v is not None:
                    codec_options.extend([f"-{k}", str(v)])
            
            # 添加通用参数
            for k, v in params.items():
                if v is not None and k not in ["crf", "tune"]:
                    codec_options.extend([f"-{k}", str(v)])
        
        elif self.qsv_available:
            encoder = "h264_qsv"
            codec_options = [
                "-c:v", encoder,
                "-global_quality", str(params["crf"]),
                "-preset", "veryslow" if params["preset"] == "veryslow" else "slower",
                "-profile:v", params["profile"],
            ]
        
        else:
            # 软件编码
            encoder = "libx264"
            codec_options = [
                "-c:v", encoder,
                "-crf", str(params["crf"]),
                "-preset", params["preset"],
                "-profile:v", params["profile"],
            ]
            
            if params["tune"]:
                codec_options.extend(["-tune", params["tune"]])
        
        # 添加其他通用参数
        codec_options.extend([
            "-pix_fmt", params["pix_fmt"],
            "-g", str(params["g"])
        ])
        
        # 添加输出文件
        codec_options.append(str(self.output_path))
        
        # 合并完整命令
        full_cmd = base_cmd + codec_options
        logger.debug(f"FFmpeg命令: {' '.join(full_cmd)}")
        
        return full_cmd
    
    def start(self):
        """启动编码器进程"""
        if self.process is not None:
            logger.warning("编码器已在运行")
            return
        
        # 构建ffmpeg命令
        cmd = self._build_ffmpeg_command()
        
        # 启动ffmpeg进程
        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=10**8  # 大缓冲区，提高性能
        )
        
        self.start_time = time.time()
        self.frames_written = 0
        self.running = True
        
        # 启动帧写入线程
        self.writer_thread = threading.Thread(target=self._frame_writer)
        self.writer_thread.daemon = True
        self.writer_thread.start()
        
        logger.info(f"视频编码器已启动，输出: {self.output_path}")
    
    def _frame_writer(self):
        """帧写入线程，从队列读取帧并写入ffmpeg"""
        try:
            while self.running or not self.frame_queue.empty():
                try:
                    # 从队列获取帧，如果队列为空且编码器已停止，则退出
                    frame_data = self.frame_queue.get(timeout=0.5)
                    
                    # 写入帧数据到ffmpeg
                    self.process.stdin.write(frame_data)
                    self.process.stdin.flush()
                    
                    # 更新计数器
                    self.frames_written += 1
                    
                    # 标记任务完成
                    self.frame_queue.task_done()
                    
                except queue.Empty:
                    # 队列为空，继续循环
                    continue
        except Exception as e:
            logger.error(f"帧写入线程错误: {e}")
        finally:
            if self.process and self.process.stdin:
                # 确保stdin已关闭
                self.process.stdin.close()
    
    def add_frame(self, frame):
        """
        添加帧到编码队列
        
        Args:
            frame: NumPy数组，形状为(height, width, 3)
        
        Returns:
            bool: 是否成功添加
        """
        if not self.running:
            logger.error("编码器未运行")
            return False
        
        if not isinstance(frame, np.ndarray):
            logger.error(f"帧必须是NumPy数组，而不是 {type(frame)}")
            return False
        
        if frame.shape != (self.height, self.width, 3):
            logger.error(f"帧尺寸不匹配: {frame.shape} != {(self.height, self.width, 3)}")
            return False
        
        # 将帧数据转换为字节流
        try:
            frame_bytes = frame.astype(np.uint8).tobytes()
            
            # 添加到队列，如果队列已满则等待
            self.frame_queue.put(frame_bytes, timeout=5)
            return True
        
        except Exception as e:
            logger.error(f"添加帧错误: {e}")
            return False
    
    def get_stats(self):
        """获取编码统计信息"""
        if self.start_time is None:
            return {
                "frames_written": 0,
                "elapsed_time": 0,
                "fps": 0,
                "queue_size": 0
            }
        
        elapsed = time.time() - self.start_time
        fps = self.frames_written / elapsed if elapsed > 0 else 0
        
        return {
            "frames_written": self.frames_written,
            "elapsed_time": elapsed,
            "fps": fps,
            "queue_size": self.frame_queue.qsize()
        }
    
    def stop(self):
        """停止编码器"""
        if not self.running:
            logger.warning("编码器未运行")
            return
        
        logger.info("停止编码器...")
        self.running = False
        
        # 等待队列清空
        self.frame_queue.join()
        
        # 关闭ffmpeg的stdin
        if self.process and self.process.stdin:
            self.process.stdin.close()
        
        # 等待ffmpeg进程结束
        if self.process:
            try:
                stdout, stderr = self.process.communicate(timeout=10)
                logger.debug(f"FFmpeg输出: {stderr.decode('utf-8', errors='ignore') if stderr else ''}")
            except subprocess.TimeoutExpired:
                logger.warning("FFmpeg进程超时，强制终止")
                self.process.kill()
            
            self.process = None
        
        # 等待写入线程结束
        if self.writer_thread and self.writer_thread.is_alive():
            self.writer_thread.join(timeout=5)
        
        stats = self.get_stats()
        logger.info(f"编码完成: {stats['frames_written']}帧, "
                 f"{stats['elapsed_time']:.2f}秒, 平均{stats['fps']:.2f}fps")
        
        return stats


class StreamingVideoEncoder(VideoEncoder):
    """
    流式视频编码器，支持更高级的流水线和并行处理
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 添加帧处理队列，用于前处理
        self.processing_queue = queue.Queue(maxsize=100)
        
        # 添加监控线程
        self.monitor_thread = None
        self.monitor_interval = 1.0  # 秒
        
        # 最大允许队列积压帧数
        self.max_backlog = 50
        
        # 添加元数据
        self.metadata = {
            "encoder": "StreamingVideoEncoder",
            "resolution": f"{self.width}x{self.height}",
            "fps": self.fps,
            "created": time.time(),
            "hardware_acceleration": "nvenc" if self.nvenc_available else "qsv" if self.qsv_available else "none"
        }
    
    def start(self):
        """启动增强型编码器"""
        super().start()
        
        # 启动监控线程
        self.monitor_thread = threading.Thread(target=self._monitor_process)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def _monitor_process(self):
        """监控编码进程和队列状态"""
        while self.running:
            try:
                # 检查ffmpeg进程是否仍在运行
                if self.process and self.process.poll() is not None:
                    logger.error(f"FFmpeg进程意外退出，返回码: {self.process.returncode}")
                    self.running = False
                    break
                
                # 检查队列状态
                queue_size = self.frame_queue.qsize()
                if queue_size > self.max_backlog:
                    logger.warning(f"帧队列积压: {queue_size} > {self.max_backlog}")
                
                # 收集和记录性能指标
                stats = self.get_stats()
                if stats["elapsed_time"] > 10:  # 仅记录超过10秒的会话
                    logger.debug(f"编码状态: {stats['frames_written']}帧, "
                             f"{stats['fps']:.2f}fps, 队列: {stats['queue_size']}")
                
                # 等待下一个监控周期
                time.sleep(self.monitor_interval)
                
            except Exception as e:
                logger.error(f"监控线程错误: {e}")
                time.sleep(self.monitor_interval)
    
    def add_frame_with_processing(self, frame, preprocessing_function=None):
        """
        添加需要预处理的帧
        
        Args:
            frame: 原始帧数据
            preprocessing_function: 预处理函数，接收帧并返回处理后的帧
        
        Returns:
            bool: 是否成功加入队列
        """
        if preprocessing_function is None:
            # 如果没有预处理函数，直接添加
            return self.add_frame(frame)
        
        try:
            # 将帧和处理函数放入处理队列
            self.processing_queue.put((frame, preprocessing_function), timeout=5)
            
            # 如果这是第一个处理任务，启动处理线程
            if self.processing_queue.qsize() == 1:
                threading.Thread(target=self._process_frames).start()
            
            return True
        
        except Exception as e:
            logger.error(f"添加预处理帧错误: {e}")
            return False
    
    def _process_frames(self):
        """处理队列中的帧"""
        while not self.processing_queue.empty() and self.running:
            try:
                # 获取下一个任务
                frame, process_func = self.processing_queue.get(timeout=1)
                
                # 应用处理函数
                processed_frame = process_func(frame)
                
                # 将处理后的帧添加到编码队列
                self.add_frame(processed_frame)
                
                # 标记任务完成
                self.processing_queue.task_done()
                
            except queue.Empty:
                break
            except Exception as e:
                logger.error(f"帧处理错误: {e}")
                self.processing_queue.task_done()
    
    def stop(self):
        """增强型停止方法"""
        if not self.running:
            return
        
        logger.info("停止流式编码器...")
        self.running = False
        
        # 处理所有待处理的帧
        self._drain_processing_queue()
        
        # 调用基类停止方法
        stats = super().stop()
        
        # 写入元数据
        try:
            self._write_metadata()
        except Exception as e:
            logger.error(f"写入元数据错误: {e}")
        
        return stats
    
    def _drain_processing_queue(self):
        """处理所有待处理的帧"""
        logger.info(f"处理剩余 {self.processing_queue.qsize()} 个待处理帧...")
        
        while not self.processing_queue.empty():
            try:
                frame, process_func = self.processing_queue.get(timeout=1)
                processed_frame = process_func(frame)
                self.add_frame(processed_frame)
                self.processing_queue.task_done()
            except queue.Empty:
                break
            except Exception as e:
                logger.error(f"清空处理队列错误: {e}")
                self.processing_queue.task_done()
    
    def _write_metadata(self):
        """将元数据写入到输出文件旁边"""
        # 更新元数据
        stats = self.get_stats()
        self.metadata.update({
            "completed": time.time(),
            "duration": stats["elapsed_time"],
            "frames": stats["frames_written"],
            "average_fps": stats["fps"]
        })
        
        # 写入元数据文件
        metadata_path = Path(str(self.output_path).rsplit('.', 1)[0] + '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        logger.info(f"元数据已写入: {metadata_path}")


# 批处理视频编码器，用于超大文件
class BatchVideoEncoder:
    """
    批处理视频编码器，将大文件拆分为多个部分并行编码
    适用于超大文件或多GPU系统
    """
    
    def __init__(self, width, height, fps=30, output_dir=None, 
                batch_duration=300, max_parallel=2):
        """
        初始化批处理编码器
        
        Args:
            width: 视频宽度
            height: 视频高度
            fps: 帧率
            output_dir: 输出目录
            batch_duration: 每批处理的秒数
            max_parallel: 最大并行编码数
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.output_dir = Path(output_dir or "output")
        self.output_dir.mkdir(exist_ok=True)
        
        self.batch_duration = batch_duration
        self.frames_per_batch = int(fps * batch_duration)
        self.max_parallel = max_parallel
        
        # 存储当前批次和编码器
        self.current_batch = 0
        self.encoders = {}
        self.completed_batches = []
        
        # 批次元数据
        self.batch_metadata = {}
        
        logger.info(f"批处理编码器初始化: {width}x{height}, {fps}fps, "
                 f"每批{batch_duration}秒({self.frames_per_batch}帧), "
                 f"最大并行数{max_parallel}")
    
    def add_frame(self, frame):
        """
        添加帧到当前批次
        
        Args:
            frame: NumPy数组，形状为(height, width, 3)
            
        Returns:
            bool: 是否成功添加
        """
        # 计算当前帧应该属于哪个批次
        batch_id = len(self.completed_batches) + self.current_batch
        frame_within_batch = sum(len(self.encoders[b].get_stats()["frames_written"]) 
                                for b in self.encoders if b < batch_id)
        
        # 如果当前批次已满，创建新批次
        if frame_within_batch >= self.frames_per_batch:
            self._start_new_batch()
        
        # 确保当前批次有一个活跃的编码器
        if self.current_batch not in self.encoders:
            self._start_encoder(self.current_batch)
        
        # 添加帧到当前编码器
        encoder = self.encoders[self.current_batch]
        return encoder.add_frame(frame)
    
    def _start_encoder(self, batch_id):
        """启动指定批次的编码器"""
        # 如果已经有max_parallel个编码器在运行，等待一个完成
        while len(self.encoders) >= self.max_parallel:
            self._check_completed_encoders()
            time.sleep(0.5)
        
        # 创建输出路径
        output_path = self.output_dir / f"batch_{batch_id:04d}.mp4"
        
        # 创建编码器
        encoder = StreamingVideoEncoder(
            self.width, self.height, self.fps, 
            output_path=output_path, 
            quality="high"
        )
        
        # 启动编码器
        encoder.start()
        self.encoders[batch_id] = encoder
        
        logger.info(f"批次 {batch_id} 编码器已启动: {output_path}")
    
    def _start_new_batch(self):
        """启动新的批次"""
        self.current_batch += 1
        self._start_encoder(self.current_batch)
    
    def _check_completed_encoders(self):
        """检查并处理已完成的编码器"""
        completed = []
        
        for batch_id, encoder in self.encoders.items():
            # 检查编码器是否还在运行
            if not encoder.running:
                # 收集统计信息
                stats = encoder.get_stats()
                self.batch_metadata[batch_id] = {
                    "output_file": str(encoder.output_path),
                    "frames": stats["frames_written"],
                    "duration": stats["elapsed_time"]
                }
                
                completed.append(batch_id)
                self.completed_batches.append(batch_id)
        
        # 从活跃编码器列表中移除已完成的
        for batch_id in completed:
            del self.encoders[batch_id]
    
    def stop(self):
        """停止所有编码器并合并输出"""
        logger.info("停止批处理编码器...")
        
        # 停止所有活跃编码器
        for batch_id, encoder in self.encoders.items():
            stats = encoder.stop()
            
            # 收集统计信息
            self.batch_metadata[batch_id] = {
                "output_file": str(encoder.output_path),
                "frames": stats["frames_written"],
                "duration": stats["elapsed_time"]
            }
            
            self.completed_batches.append(batch_id)
        
        # 清空编码器列表
        self.encoders = {}
        
        # 合并所有批次
        self._merge_outputs()
    
    def _merge_outputs(self):
        """合并所有批次的输出文件"""
        if not self.completed_batches:
            logger.warning("没有完成的批次，跳过合并")
            return
        
        # 对批次排序
        self.completed_batches.sort()
        
        # 创建文件列表
        batch_files = [str(self.batch_metadata[b]["output_file"]) for b in self.completed_batches]
        
        # 创建合并文件
        merged_output = self.output_dir / "merged_output.mp4"
        
        if len(batch_files) == 1:
            # 只有一个文件，直接复制
            shutil.copy(batch_files[0], merged_output)
            logger.info(f"单个批次，已复制到: {merged_output}")
            return
        
        # 创建ffmpeg文件列表
        list_file = self.output_dir / "batch_list.txt"
        with open(list_file, 'w') as f:
            for file_path in batch_files:
                f.write(f"file '{file_path}'\n")
        
        # 调用ffmpeg合并文件
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(list_file),
            "-c", "copy",
            str(merged_output)
        ]
        
        logger.info(f"合并 {len(batch_files)} 个批次...")
        try:
            subprocess.run(cmd, check=True)
            logger.info(f"合并完成: {merged_output}")
            
            # 写入合并元数据
            self._write_merged_metadata(merged_output)
            
        except subprocess.CalledProcessError as e:
            logger.error(f"合并失败: {e}")
    
    def _write_merged_metadata(self, output_file):
        """写入合并元数据"""
        # 计算总帧数和持续时间
        total_frames = sum(self.batch_metadata[b]["frames"] for b in self.completed_batches)
        total_duration = sum(self.batch_metadata[b]["duration"] for b in self.completed_batches)
        
        # 创建元数据
        metadata = {
            "encoder": "BatchVideoEncoder",
            "resolution": f"{self.width}x{self.height}",
            "fps": self.fps,
            "created": time.time(),
            "completed": time.time(),
            "total_frames": total_frames,
            "total_duration": total_duration,
            "batch_count": len(self.completed_batches),
            "batches": self.batch_metadata
        }
        
        # 写入元数据文件
        metadata_path = Path(str(output_file).rsplit('.', 1)[0] + '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"合并元数据已写入: {metadata_path}")
