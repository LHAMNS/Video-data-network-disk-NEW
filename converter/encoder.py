"""
视频编码模块 - 直接AVI写入实现
Direct AVI writing implementation for maximum I/O throughput
"""

import os
import time
import queue
import logging
import threading
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from converter import (
    ROOT_DIR, OUTPUT_DIR, VIDEO_PRESETS,
    MIN_FPS, MAX_FPS, DEFAULT_FPS
)
from .avi_writer import SimpleAVIWriter
from .frame_generator import FrameGenerator, OptimizedFrameGenerator

logger = logging.getLogger(__name__)


class DirectAVIEncoder:
    """
    Direct AVI encoder that writes uncompressed RGB frames
    Performance is limited only by disk I/O speed
    """
    
    def __init__(self, width, height, fps=30, output_path=None):
        """
        Initialize direct AVI encoder
        
        Args:
            width: Video width
            height: Video height  
            fps: Frame rate
            output_path: Output file path
        """
        self.width = width
        self.height = height
        self.fps = max(MIN_FPS, min(fps, MAX_FPS))
        
        if output_path is None:
            output_path = OUTPUT_DIR / f"output_{int(time.time())}.avi"
        self.output_path = Path(output_path)
        
        # AVI writer instance
        self.avi_writer = None
        
        # Runtime state
        self.running = False
        self.frames_written = 0
        self.bytes_written = 0
        self.start_time = 0
        
        # Performance metrics
        self.write_times = []
        self.max_write_time = 0
        self.min_write_time = float('inf')
        
        logger.info(f"Direct AVI encoder initialized: {width}x{height} @ {fps}fps")
    
    def start(self):
        """Start the AVI encoder"""
        if self.running:
            logger.warning("Encoder already running")
            return False
            
        self.running = True
        self.frames_written = 0
        self.bytes_written = 0
        self.start_time = time.time()
        self.write_times = []
        
        try:
            # Initialize AVI writer
            self.avi_writer = SimpleAVIWriter(
                width=self.width,
                height=self.height,
                fps=self.fps,
                output_path=str(self.output_path)
            )
            self.avi_writer.open()
            
            logger.info(f"AVI writer started, output: {self.output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start AVI writer: {e}", exc_info=True)
            self.running = False
            return False
    
    def add_frame(self, frame):
        """
        Add RGB frame directly to AVI
        
        Args:
            frame: RGB frame as numpy array (height, width, 3)
            
        Returns:
            bool: Success status
        """
        if not self.running or self.avi_writer is None:
            logger.warning("Encoder not running")
            return False
            
        try:
            # Measure write time
            write_start = time.time()
            
            # Write frame directly to AVI
            self.avi_writer.add_rgb_frame(frame)
            
            write_time = time.time() - write_start
            self.write_times.append(write_time)
            self.max_write_time = max(self.max_write_time, write_time)
            self.min_write_time = min(self.min_write_time, write_time)
            
            self.frames_written += 1
            self.bytes_written += frame.nbytes
            
            # Log performance every 100 frames
            if self.frames_written % 100 == 0:
                avg_write_time = sum(self.write_times[-100:]) / 100
                throughput_mbps = (frame.nbytes * 100 / avg_write_time) / (1024 * 1024) / 100
                logger.info(f"Frame {self.frames_written}: avg write time {avg_write_time*1000:.2f}ms, "
                          f"throughput {throughput_mbps:.1f} MB/s")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to write frame: {e}")
            return False
    
    def stop(self):
        """Stop encoder and finalize AVI file"""
        if not self.running:
            logger.warning("Encoder not running")
            return None
            
        self.running = False
        
        try:
            if self.avi_writer:
                self.avi_writer.close()
                
            elapsed = time.time() - self.start_time
            
            # Calculate statistics
            stats = {
                "frames_written": self.frames_written,
                "bytes_written": self.bytes_written,
                "duration_seconds": self.frames_written / self.fps if self.fps > 0 else 0,
                "elapsed_seconds": elapsed,
                "average_fps": self.frames_written / elapsed if elapsed > 0 else 0,
                "average_throughput_mbps": (self.bytes_written / elapsed) / (1024 * 1024) if elapsed > 0 else 0,
                "output_path": str(self.output_path),
                "output_size": self.output_path.stat().st_size if self.output_path.exists() else 0,
                "performance": {
                    "avg_write_time_ms": (sum(self.write_times) / len(self.write_times) * 1000) if self.write_times else 0,
                    "max_write_time_ms": self.max_write_time * 1000,
                    "min_write_time_ms": self.min_write_time * 1000 if self.min_write_time != float('inf') else 0
                }
            }
            
            logger.info(f"AVI encoding complete: {self.frames_written} frames in {elapsed:.2f}s "
                       f"({stats['average_fps']:.1f} fps, {stats['average_throughput_mbps']:.1f} MB/s)")
            
            return stats
            
        except Exception as e:
            logger.error(f"Error stopping encoder: {e}", exc_info=True)
            return None


class StreamingDirectAVIEncoder(DirectAVIEncoder):
    """
    Streaming version with frame queue for smooth I/O flow
    """
    
    def __init__(self, width, height, fps=30, output_path=None, queue_size=30):
        super().__init__(width, height, fps, output_path)
        
        self.queue_size = queue_size
        self.frame_queue = queue.Queue(maxsize=queue_size)
        self.writer_thread = None
        self.stop_event = threading.Event()
        
    def start(self):
        """Start encoder with background writer thread"""
        if not super().start():
            return False
            
        # Start writer thread
        self.stop_event.clear()
        self.writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.writer_thread.start()
        
        logger.info("Streaming AVI encoder started with background writer")
        return True
    
    def add_frame(self, frame):
        """Add frame to queue for background writing"""
        if not self.running:
            return False
            
        try:
            # Non-blocking put with timeout
            self.frame_queue.put(frame, timeout=0.1)
            return True
        except queue.Full:
            logger.warning("Frame queue full, dropping frame")
            return False
    
    def _writer_loop(self):
        """Background thread for writing frames"""
        logger.info("Writer thread started")
        
        while not self.stop_event.is_set() or not self.frame_queue.empty():
            try:
                # Get frame with timeout
                frame = self.frame_queue.get(timeout=0.1)
                super().add_frame(frame)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Writer thread error: {e}")
                
        logger.info("Writer thread finished")
    
    def stop(self):
        """Stop encoder and wait for queue to empty"""
        if not self.running:
            return None
            
        logger.info("Stopping streaming encoder...")
        self.running = False
        self.stop_event.set()
        
        # Wait for writer thread
        if self.writer_thread and self.writer_thread.is_alive():
            self.writer_thread.join(timeout=30)
            
        return super().stop()


class ParallelDirectAVIEncoder:
    """
    Parallel AVI encoder using multiple writer instances for extreme throughput
    Splits output into multiple AVI files that can be concatenated later
    """
    
    def __init__(self, width, height, fps=30, output_path=None, num_workers=4):
        self.width = width
        self.height = height
        self.fps = fps
        self.num_workers = num_workers
        
        if output_path is None:
            output_path = OUTPUT_DIR / f"output_{int(time.time())}.avi"
        self.base_output_path = Path(output_path)
        
        # Create worker encoders
        self.workers = []
        self.current_worker = 0
        self.running = False
        
        # Statistics
        self.total_frames = 0
        self.start_time = 0
        
    def start(self):
        """Start all worker encoders"""
        if self.running:
            return False
            
        self.running = True
        self.start_time = time.time()
        self.total_frames = 0
        self.current_worker = 0
        
        # Initialize worker encoders
        for i in range(self.num_workers):
            output_path = self.base_output_path.with_stem(f"{self.base_output_path.stem}_part{i}")
            worker = StreamingDirectAVIEncoder(
                width=self.width,
                height=self.height,
                fps=self.fps,
                output_path=output_path
            )
            if not worker.start():
                logger.error(f"Failed to start worker {i}")
                self.stop()
                return False
            self.workers.append(worker)
            
        logger.info(f"Started {self.num_workers} parallel AVI encoders")
        return True
    
    def add_frame(self, frame):
        """Distribute frame to next worker in round-robin"""
        if not self.running or not self.workers:
            return False
            
        # Round-robin distribution
        worker = self.workers[self.current_worker]
        success = worker.add_frame(frame)
        
        if success:
            self.total_frames += 1
            self.current_worker = (self.current_worker + 1) % self.num_workers
            
        return success
    
    def stop(self):
        """Stop all workers and aggregate statistics"""
        if not self.running:
            return None
            
        self.running = False
        elapsed = time.time() - self.start_time
        
        # Stop all workers
        all_stats = []
        total_bytes = 0
        
        for i, worker in enumerate(self.workers):
            stats = worker.stop()
            if stats:
                all_stats.append(stats)
                total_bytes += stats['bytes_written']
                
        # Aggregate statistics
        aggregated_stats = {
            "total_frames": self.total_frames,
            "total_bytes": total_bytes,
            "duration_seconds": self.total_frames / self.fps if self.fps > 0 else 0,
            "elapsed_seconds": elapsed,
            "average_fps": self.total_frames / elapsed if elapsed > 0 else 0,
            "average_throughput_mbps": (total_bytes / elapsed) / (1024 * 1024) if elapsed > 0 else 0,
            "num_workers": self.num_workers,
            "output_files": [worker.output_path for worker in self.workers],
            "worker_stats": all_stats
        }
        
        logger.info(f"Parallel encoding complete: {self.total_frames} frames across {self.num_workers} files "
                   f"in {elapsed:.2f}s ({aggregated_stats['average_fps']:.1f} fps, "
                   f"{aggregated_stats['average_throughput_mbps']:.1f} MB/s total)")
        
        self.workers.clear()
        return aggregated_stats


# Compatibility aliases for existing code
VideoEncoder = DirectAVIEncoder
StreamingVideoEncoder = StreamingDirectAVIEncoder
BatchVideoEncoder = DirectAVIEncoder  # Same as direct encoder for uncompressed


def get_optimal_encoder(width, height, fps, output_path=None, **kwargs):
    """
    Get optimal encoder based on system capabilities
    
    Returns most appropriate encoder class
    """
    # Check available disk I/O bandwidth
    # For now, use streaming encoder as default
    
    # Check if system has high-speed storage (NVMe)
    # This is a simplified check - in production would use actual benchmarks
    cpu_count = os.cpu_count() or 4
    
    if cpu_count >= 8:
        # High-end system, use parallel encoder
        logger.info("Using parallel AVI encoder for maximum throughput")
        return ParallelDirectAVIEncoder(width, height, fps, output_path, 
                                       num_workers=min(cpu_count // 2, 8))
    else:
        # Standard system, use streaming encoder
        logger.info("Using streaming AVI encoder")
        return StreamingDirectAVIEncoder(width, height, fps, output_path)
