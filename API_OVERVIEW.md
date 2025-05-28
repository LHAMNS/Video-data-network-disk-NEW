# API Overview

This document summarizes the key modules, classes, and functions in the project.  
It complements `README.md` and explains how major pieces interact.

## Converter package

### `FrameGenerator` (`converter/frame_generator.py`)
- Converts byte data into RGB video frames.
- Supports 16 or 256 colors and optional *9‑to‑1* pixel expansion for better compression resistance.
- `generate_frames_from_data(data, callback=None)` yields frames from bytes or iterables, invoking `callback(frame_index, total_frames, frame)`.
- `generate_preview_image(frame, max_size=300)` returns a base64 JPEG for UI previews.

### `OptimizedFrameGenerator`
- Subclass of `FrameGenerator` with preallocated buffers and optimized Numba routines.
- Adds `generate_frame_optimized` and `process_large_file` for high‑throughput scenarios.

### `GPUFrameGenerator` (`converter/gpu_frame_generator.py`)
- CUDA implementation for generating frames on the GPU.
- Uses custom kernels to unpack bit data and map indices to RGB.
- Provides `generate_frames_from_data` similar to the CPU version.

### Video Encoding (`converter/encoder.py`)
- `DirectAVIEncoder` writes raw RGB frames directly to an AVI using `SimpleAVIWriter`.
- `StreamingDirectAVIEncoder` extends it with a frame queue and background writer thread.
- `ParallelDirectAVIEncoder` distributes frames across multiple `StreamingDirectAVIEncoder` workers for maximum throughput.
- `get_optimal_encoder(width, height, fps, output_path=None, **kwargs)` chooses an encoder based on CPU count.
- `VideoEncoder`, `StreamingVideoEncoder`, and `BatchVideoEncoder` are aliases to the above for compatibility.

### Error Correction (`converter/error_correction.py`)
- `ReedSolomonEncoder` provides multi‑threaded Reed–Solomon coding via `encode_data`/`decode_data`.
- `XORInterleaver` offers a lightweight XOR based scheme.
- `HybridErrorCorrection` combines both.

### Additional utilities
- `avi_writer.py` contains `SimpleAVIWriter` for writing RGB24 AVI files and `convert_bytes_to_avi` helper.
- `avi_validator.py` validates generated AVI files and checks extracted data.
- `decoder.py` restores original bytes from a video; `ParallelVideoDecoder` runs extraction in threads.
- `utils.py` offers helper functions such as `expand_pixels_9x1`, `bytes_to_color_indices`, caching (`CacheManager`), hardware detection (`is_nvenc_available`, `is_qsv_available`), and more.
- `pipeline.py` defines a pluggable `ConversionPipeline` with stages for error correction, frame generation and encoding.

## `ConversionTask` (`web_ui/server.py`)
Manages a single file‑to‑video conversion job. Important attributes and methods include:
- Initialization parses parameters (resolution, fps, nine‑to‑one mode, color count, error‑correction ratio). Video parameters are computed via `calculate_video_params`.
- `start()` launches a background thread calling `_conversion_worker`.
- `_frame_generated_callback` updates progress and emits Socket.IO events.
- `_verify_output_video` attempts to repair and validate the resulting AVI with FFmpeg.
- `stop()` stops the thread and encoder, updating task status.

## Web API (`web_ui/server.py`)
The Flask application exposes several routes:
- `/` – render the main HTML page.
- `/api/hardware-info` – report NVENC/QSV availability and system info.
- `/api/upload` – accept a file upload, cache it using `CacheManager`, and return default video parameters.
- `/api/start-conversion` – create a `ConversionTask` for the uploaded file and start processing.
- `/api/stop-conversion` – stop a running task.
- `/api/task/<task_id>` – fetch progress for a single task.
- `/api/tasks` – list all tasks.
- `/api/download/<task_id>` and `/api/download/file/<path>` – download finished videos.
- `/api/clear-cache` and `/api/clean-tasks` – maintenance endpoints to remove temporary files or old tasks.

Socket.IO events (`connect`, `disconnect`, `get_tasks`, `get_task`) provide real‑time status updates to the frontend.

## Relationships and Workflow
1. Files uploaded via `/api/upload` are stored in `cache/` by `CacheManager`.
2. When `/api/start-conversion` is called, `ConversionTask` creates a `FrameGenerator` (CPU or GPU) and a `StreamingDirectAVIEncoder`. Optional error correction (`get_optimal_error_corrector`) is inserted via the pipeline.
3. Frames are generated from cached data and passed to the encoder; `_frame_generated_callback` relays progress back to the client.
4. Upon completion, `_verify_output_video` checks the AVI file. Results are available through `/api/task/<task_id>` and downloadable via `/api/download/<task_id>`.

For a broader introduction, see `README.md` which outlines the full workflow from upload to video creation and optional decoding.
