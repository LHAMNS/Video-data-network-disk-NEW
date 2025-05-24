"""GPU-based frame generation pipeline optimized for high throughput."""


import numpy as np
import cupy as cp
import logging
from typing import Iterator, Optional, Callable
import os

logger = logging.getLogger(__name__)

class GPUFrameGenerator:
    """
    CUDA-accelerated frame generation
    Processes data entirely on GPU, minimizing PCIe transfers
    """
    
    def __init__(self, resolution: str = "4K", fps: int = 30, 
                 color_count: int = 16, nine_to_one: bool = True):
        """
        Initialize GPU frame generator
        
        Args:
            resolution: Output resolution
            fps: Frame rate  
            color_count: Number of colors (16 or 256)
            nine_to_one: Enable 9x upsampling
        """
        from . import VIDEO_PRESETS, COLOR_PALETTE_16
        
        # Video parameters
        self.resolution_name = resolution
        preset = VIDEO_PRESETS.get(resolution, VIDEO_PRESETS["4K"])
        self.physical_width = preset["width"]
        self.physical_height = preset["height"]
        self.fps = fps
        self.color_count = color_count
        self.nine_to_one = nine_to_one
        
        # Calculate logical dimensions
        if nine_to_one:
            self.logical_width = self.physical_width // 3
            self.logical_height = self.physical_height // 3
        else:
            self.logical_width = self.physical_width
            self.logical_height = self.physical_height
        
        # Bytes per frame calculation
        bits_per_pixel = 4 if color_count == 16 else 8
        self.bytes_per_frame = self.logical_width * self.logical_height * bits_per_pixel // 8
        
        # Check CUDA availability
        if not self._check_cuda():
            raise RuntimeError("CUDA not available for GPU frame generation")
        
        # Transfer color palette to GPU
        self.d_color_lut = cp.asarray(COLOR_PALETTE_16[:color_count], dtype=cp.uint8)
        
        # Pre-allocate GPU buffers
        self._init_gpu_buffers()
        
        # Compile CUDA kernels
        self._init_kernels()
        
        logger.info(f"GPU frame generator initialized: {resolution} @ {fps}fps, "
                   f"logical: {self.logical_width}x{self.logical_height}, "
                   f"bytes/frame: {self.bytes_per_frame}")
    
    def _check_cuda(self) -> bool:
        """Verify CUDA availability"""
        try:
            device = cp.cuda.Device()
            self.gpu_name = device.name.decode()
            self.gpu_memory = device.mem_info[1] / 1024**3
            logger.info(f"GPU: {self.gpu_name}, Memory: {self.gpu_memory:.1f} GB")
            return True
        except Exception as e:
            logger.error(f"CUDA not available: {e}")
            return False
    
    def _init_gpu_buffers(self):
        """Pre-allocate GPU memory buffers"""
        # Calculate buffer sizes
        self.max_batch_frames = 30  # Process 30 frames at once
        
        # Input buffer for raw bytes
        self.d_input_buffer = cp.zeros(
            self.max_batch_frames * self.bytes_per_frame, 
            dtype=cp.uint8
        )
        
        # Intermediate buffers
        self.d_indices = cp.zeros(
            self.max_batch_frames * self.logical_width * self.logical_height,
            dtype=cp.uint8
        )
        
        # Output RGB buffer
        self.d_rgb_frames = cp.zeros(
            (self.max_batch_frames, self.physical_height, self.physical_width, 3),
            dtype=cp.uint8
        )
        
        # Pinned memory for fast transfers
        cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
        
        logger.info(f"Allocated GPU buffers: {self.max_batch_frames} frames, "
                   f"{(self.d_rgb_frames.nbytes / 1024**2):.1f} MB")
    
    def _init_kernels(self):
        """Compile optimized CUDA kernels"""
        
        # Kernel for 4-bit to index conversion
        unpack_4bit_kernel = '''
        extern "C" __global__
        void unpack_4bit_to_indices(
            const unsigned char* __restrict__ packed_data,
            unsigned char* __restrict__ indices,
            int total_pixels
        ) {
            int tid = blockIdx.x * blockDim.x + threadIdx.x;
            int byte_idx = tid / 2;
            
            if (tid < total_pixels) {
                unsigned char byte = packed_data[byte_idx];
                if (tid % 2 == 0) {
                    indices[tid] = (byte >> 4) & 0x0F;
                } else {
                    indices[tid] = byte & 0x0F;
                }
            }
        }
        '''
        
        # Kernel for direct index to RGB mapping
        indices_to_rgb_kernel = '''
        extern "C" __global__
        void indices_to_rgb_direct(
            const unsigned char* __restrict__ indices,
            unsigned char* __restrict__ rgb_output,
            const unsigned char* __restrict__ color_lut,
            int width,
            int height,
            int logical_width,
            int logical_height,
            bool nine_to_one
        ) {
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;
            int frame = blockIdx.z;
            
            if (x >= width || y >= height) return;
            
            int src_x, src_y;
            if (nine_to_one) {
                src_x = x / 3;
                src_y = y / 3;
            } else {
                src_x = x;
                src_y = y;
            }
            
            int index_offset = frame * logical_width * logical_height;
            int color_idx = indices[index_offset + src_y * logical_width + src_x];
            
            // Direct RGB assignment
            int pixel_offset = ((frame * height + y) * width + x) * 3;
            rgb_output[pixel_offset + 0] = color_lut[color_idx * 3 + 0];
            rgb_output[pixel_offset + 1] = color_lut[color_idx * 3 + 1];
            rgb_output[pixel_offset + 2] = color_lut[color_idx * 3 + 2];
        }
        '''
        
        # Optimized 9x1 upsampling kernel
        nine_to_one_kernel = '''
        extern "C" __global__
        void upsample_9x1_optimized(
            const unsigned char* __restrict__ logical_rgb,
            unsigned char* __restrict__ physical_rgb,
            int logical_width,
            int logical_height,
            int frame_idx
        ) {
            int logical_x = blockIdx.x * blockDim.x + threadIdx.x;
            int logical_y = blockIdx.y * blockDim.y + threadIdx.y;
            
            if (logical_x >= logical_width || logical_y >= logical_height) return;
            
            // Source pixel
            int src_offset = ((frame_idx * logical_height + logical_y) * logical_width + logical_x) * 3;
            unsigned char r = logical_rgb[src_offset + 0];
            unsigned char g = logical_rgb[src_offset + 1];
            unsigned char b = logical_rgb[src_offset + 2];
            
            // Write 9 pixels
            int phys_base_x = logical_x * 3;
            int phys_base_y = logical_y * 3;
            int phys_width = logical_width * 3;
            int phys_height = logical_height * 3;
            
            for (int dy = 0; dy < 3; dy++) {
                for (int dx = 0; dx < 3; dx++) {
                    int phys_x = phys_base_x + dx;
                    int phys_y = phys_base_y + dy;
                    int dst_offset = ((frame_idx * phys_height + phys_y) * phys_width + phys_x) * 3;
                    
                    physical_rgb[dst_offset + 0] = r;
                    physical_rgb[dst_offset + 1] = g;
                    physical_rgb[dst_offset + 2] = b;
                }
            }
        }
        '''
        
        # Compile kernels
        self.unpack_kernel = cp.RawKernel(unpack_4bit_kernel, 'unpack_4bit_to_indices')
        self.rgb_kernel = cp.RawKernel(indices_to_rgb_kernel, 'indices_to_rgb_direct')
        self.upsample_kernel = cp.RawKernel(nine_to_one_kernel, 'upsample_9x1_optimized')
        
        logger.info("CUDA kernels compiled successfully")
    
    def process_frames_batch(self, data: bytes, num_frames: int) -> np.ndarray:
        """
        Process multiple frames in a single GPU batch
        
        Args:
            data: Raw byte data for frames
            num_frames: Number of frames to process
            
        Returns:
            RGB frames as numpy array (num_frames, height, width, 3)
        """
        # Transfer data to GPU
        data_array = np.frombuffer(data, dtype=np.uint8)
        self.d_input_buffer[:len(data_array)] = cp.asarray(data_array)
        
        # Unpack 4-bit data to indices
        if self.color_count == 16:
            total_pixels = num_frames * self.logical_width * self.logical_height
            threads = 256
            blocks = (total_pixels + threads - 1) // threads
            
            self.unpack_kernel(
                (blocks,), (threads,),
                (self.d_input_buffer, self.d_indices, total_pixels)
            )
        else:
            # Direct copy for 8-bit
            self.d_indices[:len(data_array)] = self.d_input_buffer[:len(data_array)]
        
        # Convert indices to RGB
        threads_2d = (16, 16)
        blocks_x = (self.physical_width + threads_2d[0] - 1) // threads_2d[0]
        blocks_y = (self.physical_height + threads_2d[1] - 1) // threads_2d[1]
        
        self.rgb_kernel(
            (blocks_x, blocks_y, num_frames), threads_2d,
            (self.d_indices, self.d_rgb_frames, self.d_color_lut,
             self.physical_width, self.physical_height,
             self.logical_width, self.logical_height,
             self.nine_to_one)
        )
        
        # Synchronize and transfer back
        cp.cuda.Stream.null.synchronize()
        return self.d_rgb_frames[:num_frames].get()
    
    def generate_frames_from_data(self, data: bytes, 
                                 callback: Optional[Callable] = None) -> Iterator[np.ndarray]:
        """
        Generate frames from data with GPU acceleration
        
        Args:
            data: Input byte data
            callback: Progress callback function
            
        Yields:
            RGB frames as numpy arrays
        """
        total_bytes = len(data)
        total_frames = (total_bytes + self.bytes_per_frame - 1) // self.bytes_per_frame
        frames_processed = 0
        
        logger.info(f"Starting GPU frame generation: {total_bytes} bytes, {total_frames} frames")
        
        # Process in batches
        for batch_start in range(0, total_frames, self.max_batch_frames):
            batch_frames = min(self.max_batch_frames, total_frames - batch_start)
            
            # Extract batch data
            start_byte = batch_start * self.bytes_per_frame
            end_byte = min(start_byte + batch_frames * self.bytes_per_frame, total_bytes)
            batch_data = data[start_byte:end_byte]
            
            # Pad if necessary
            expected_size = batch_frames * self.bytes_per_frame
            if len(batch_data) < expected_size:
                batch_data = batch_data + bytes(expected_size - len(batch_data))
            
            # Process batch on GPU
            rgb_frames = self.process_frames_batch(batch_data, batch_frames)
            
            # Yield individual frames
            for i in range(batch_frames):
                frame = rgb_frames[i]
                frames_processed += 1
                
                if callback:
                    callback(frames_processed - 1, total_frames, frame)
                
                yield frame
        
        logger.info(f"GPU frame generation complete: {frames_processed} frames")


class HybridGPUPipeline:
    """
    Complete GPU-accelerated pipeline: Error Correction → Frame Generation → AVI Writing
    """
    
    def __init__(self, resolution: str = "4K", fps: int = 30, 
                 redundancy_ratio: float = 0.1, output_path: str = None):
        self.resolution = resolution
        self.fps = fps
        self.redundancy_ratio = redundancy_ratio
        self.output_path = output_path
        
        # Initialize components
        from .gpu_error_correction import get_optimal_error_corrector
        from .encoder import StreamingDirectAVIEncoder
        
        self.error_corrector = get_optimal_error_corrector(redundancy_ratio)
        self.frame_generator = GPUFrameGenerator(resolution=resolution, fps=fps)
        self.avi_encoder = StreamingDirectAVIEncoder(
            width=self.frame_generator.physical_width,
            height=self.frame_generator.physical_height,
            fps=fps,
            output_path=output_path
        )
    
    def process_file(self, file_path: str, progress_callback: Optional[Callable] = None):
        """
        Process entire file through GPU pipeline
        
        Args:
            file_path: Input file path
            progress_callback: Progress reporting function
        """
        import time
        start_time = time.time()
        
        # Read file
        with open(file_path, 'rb') as f:
            file_data = f.read()
        
        logger.info(f"Processing file: {file_path}, size: {len(file_data)} bytes")
        
        # GPU error correction
        encoded_data, ec_stats = self.error_corrector.process_file_data(file_data)
        logger.info(f"Error correction: {ec_stats['throughput_mbps']:.1f} MB/s")
        
        # Start AVI encoder
        self.avi_encoder.start()
        
        # GPU frame generation → Direct AVI writing
        for frame in self.frame_generator.generate_frames_from_data(encoded_data, progress_callback):
            self.avi_encoder.add_frame(frame)
        
        # Finalize
        stats = self.avi_encoder.stop()
        
        total_time = time.time() - start_time
        overall_throughput = len(file_data) / total_time / (1024 * 1024)
        
        logger.info(f"Pipeline complete: {overall_throughput:.1f} MB/s overall throughput")
        
        return {
            'file_size': len(file_data),
            'encoded_size': len(encoded_data),
            'total_time': total_time,
            'throughput_mbps': overall_throughput,
            'error_correction_stats': ec_stats,
            'encoding_stats': stats
        }
