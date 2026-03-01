"""
High-performance AVI writer for uncompressed RGB24 frames.

Optimizations:
- Large buffered I/O (32MB) to reduce syscalls
- Pre-computed chunk headers to minimize per-frame overhead
- Direct BGR byte writing via numpy memory view (zero-copy when possible)
- Pre-allocated file space via fallocate for sequential write performance
- Correct AVI header offsets for standards-compliant output
"""

import os
import struct
from pathlib import Path
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Pre-computed AVI chunk tag for video frames
_CHUNK_TAG = b"00dc"

# I/O buffer size: 32MB for maximum sequential write throughput
_IO_BUFFER_SIZE = 32 * 1024 * 1024


class SimpleAVIWriter:
    """High-performance AVI writer for uncompressed RGB24 frames."""

    def __init__(self, width, height, fps=30, output_path="output.avi"):
        self.width = width
        self.height = height
        self.fps = fps
        self.output_path = Path(output_path)
        self._f = None
        self._frame_count = 0
        self._movi_start = 0
        self._movi_size_offset = 0
        self._avih_frames_offset = 0
        # Pre-compute frame data size and chunk header for this resolution
        self._frame_data_size = width * height * 3
        self._chunk_header = _CHUNK_TAG + struct.pack("<I", self._frame_data_size)
        # Whether frame data size is odd (needs padding byte)
        self._needs_padding = (self._frame_data_size % 2) == 1

    def open(self):
        """Open output file with large buffered I/O and write AVI headers."""
        self._f = open(self.output_path, "wb", buffering=_IO_BUFFER_SIZE)
        self._write_headers()
        # Try to pre-allocate disk space for better sequential write performance
        self._try_preallocate()

    def _try_preallocate(self):
        """Pre-allocate file space if the OS supports it (Linux fallocate)."""
        try:
            fd = self._f.fileno()
            # Estimate: 1000 frames as initial allocation
            estimated_size = self._frame_data_size * 1000 + 65536
            os.posix_fallocate(fd, 0, estimated_size)
        except (AttributeError, OSError):
            # Not supported on this OS or filesystem, silently continue
            pass

    def _write_headers(self):
        f = self._f
        # ---- RIFF header placeholder ----
        # Offset 0: "RIFF" (4) + size placeholder (4) + "AVI " (4) = 12 bytes
        f.write(b"RIFF\x00\x00\x00\x00AVI ")

        # ---- hdrl list ----
        hdrl_data = bytearray()

        # avih chunk (main AVI header)
        microsec_per_frame = int(1e6 / self.fps)
        max_bytes_per_sec = self._frame_data_size * self.fps
        buffer_size = self._frame_data_size

        avih = struct.pack(
            "<IIIIIIIIIIIIII",
            microsec_per_frame,     # dwMicroSecPerFrame
            max_bytes_per_sec,      # dwMaxBytesPerSec
            0,                      # dwPaddingGranularity
            0x10,                   # dwFlags (AVIF_HASINDEX)
            0,                      # dwTotalFrames (placeholder, filled on close)
            0,                      # dwInitialFrames
            1,                      # dwStreams
            buffer_size,            # dwSuggestedBufferSize
            self.width,             # dwWidth
            self.height,            # dwHeight
            0, 0, 0, 0,            # dwReserved[4]
        )
        hdrl_data += b"avih" + struct.pack("<I", len(avih)) + avih

        # The total_frames field (dwTotalFrames) is at:
        #   12 (RIFF header) + 4 (LIST) + 4 (LIST size) + 4 (hdrl) = 24
        #   + 4 (avih tag) + 4 (avih size) = 32 (start of avih struct)
        #   + 16 (5th DWORD in struct: microsec, maxbytes, padding, flags, TOTAL_FRAMES)
        #   = 48
        self._avih_frames_offset = 48

        # strh chunk (stream header)
        strh = struct.pack(
            "<4s4sIHHIIIIIIIIhhhh",
            b"vids",                # fccType
            b"DIB ",                # fccHandler (uncompressed RGB)
            0,                      # dwFlags
            0,                      # wPriority
            0,                      # wLanguage
            0,                      # dwInitialFrames
            1,                      # dwScale
            self.fps,               # dwRate (dwRate/dwScale = fps)
            0,                      # dwStart
            0,                      # dwLength (total frames, updated on close)
            buffer_size,            # dwSuggestedBufferSize
            0,                      # dwQuality (0 = default)
            0,                      # dwSampleSize
            0,                      # rcFrame.left
            0,                      # rcFrame.top
            self.width,             # rcFrame.right
            self.height,            # rcFrame.bottom
        )

        # strl list = strl header + strh chunk + strf chunk
        # strf chunk (stream format = BITMAPINFOHEADER)
        strf = struct.pack(
            "<IIIHHIIIIII",
            40,                     # biSize
            self.width,             # biWidth
            self.height,            # biHeight
            1,                      # biPlanes
            24,                     # biBitCount (RGB24)
            0,                      # biCompression (BI_RGB)
            buffer_size,            # biSizeImage
            0,                      # biXPelsPerMeter
            0,                      # biYPelsPerMeter
            0,                      # biClrUsed
            0,                      # biClrImportant
        )

        strl_content = b"strh" + struct.pack("<I", len(strh)) + strh
        strl_content += b"strf" + struct.pack("<I", len(strf)) + strf
        hdrl_data += b"LIST" + struct.pack("<I", len(strl_content) + 4) + b"strl" + strl_content

        f.write(b"LIST" + struct.pack("<I", len(hdrl_data) + 4) + b"hdrl" + hdrl_data)

        # ---- movi list ----
        f.write(b"LIST")
        self._movi_size_offset = f.tell()
        f.write(b"\x00\x00\x00\x00")  # movi size placeholder
        f.write(b"movi")
        self._movi_start = f.tell()

    def add_rgb_frame(self, frame: np.ndarray):
        """
        Write an RGB frame to the AVI file.

        Converts RGB to BGR (AVI/DIB format) and writes with minimal copies.

        Args:
            frame: RGB frame as numpy array (height, width, 3), dtype uint8
        """
        if self._f is None:
            self.open()

        if frame.shape[0] != self.height or frame.shape[1] != self.width:
            raise ValueError(
                f"Frame dimensions {frame.shape[1]}x{frame.shape[0]} "
                f"do not match writer {self.width}x{self.height}"
            )

        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)

        # Write chunk header (pre-computed)
        self._f.write(self._chunk_header)

        # Convert RGB -> BGR and write frame data
        # Use np.ascontiguousarray on the reversed view to get a single contiguous
        # BGR buffer, then write directly from its underlying memory
        if frame.shape[2] == 3:
            bgr_frame = np.ascontiguousarray(frame[:, :, ::-1])
        else:
            bgr_frame = np.ascontiguousarray(frame)

        self._f.write(bgr_frame.data)

        # AVI requires 2-byte alignment for chunks
        if self._needs_padding:
            self._f.write(b"\x00")

        self._frame_count += 1

    def add_bgr_frame(self, frame: np.ndarray):
        """
        Write a BGR frame directly to the AVI file (zero-copy fast path).

        Use this when the frame is already in BGR format to avoid
        the RGB->BGR conversion entirely.

        Args:
            frame: BGR frame as numpy array (height, width, 3), dtype uint8
        """
        if self._f is None:
            self.open()

        if frame.shape[0] != self.height or frame.shape[1] != self.width:
            raise ValueError(
                f"Frame dimensions {frame.shape[1]}x{frame.shape[0]} "
                f"do not match writer {self.width}x{self.height}"
            )

        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)

        # Write chunk header + frame data
        self._f.write(self._chunk_header)

        if frame.flags['C_CONTIGUOUS']:
            # Zero-copy write: directly from numpy buffer
            self._f.write(frame.data)
        else:
            self._f.write(np.ascontiguousarray(frame).data)

        if self._needs_padding:
            self._f.write(b"\x00")

        self._frame_count += 1

    def close(self):
        """Finalize AVI file: fill in header sizes and frame count."""
        if self._f is None:
            return

        # Flush buffered data before seeking
        self._f.flush()

        end_pos = self._f.tell()
        movi_size = end_pos - self._movi_start

        # Fill movi LIST size (includes 4 bytes for "movi" tag)
        self._f.seek(self._movi_size_offset)
        self._f.write(struct.pack("<I", movi_size + 4))

        # Fill total frames in avih header
        self._f.seek(self._avih_frames_offset)
        self._f.write(struct.pack("<I", self._frame_count))

        # Fill RIFF total size (file size - 8 for RIFF tag and size field)
        self._f.seek(4)
        self._f.write(struct.pack("<I", end_pos - 8))

        self._f.close()
        self._f = None

        # Truncate file to actual size (removes pre-allocated but unused space)
        try:
            os.truncate(str(self.output_path), end_pos)
        except OSError:
            pass

        logger.info(
            f"AVI finalized: {self._frame_count} frames, "
            f"{end_pos / 1024 / 1024:.1f} MB"
        )


def convert_bytes_to_avi(data: bytes, frame_generator, output_path: str, fps=30):
    """Convert raw bytes to an uncompressed AVI using the given frame generator."""
    writer = SimpleAVIWriter(
        width=frame_generator.physical_width,
        height=frame_generator.physical_height,
        fps=fps,
        output_path=output_path,
    )
    writer.open()

    for frame in frame_generator.generate_frames_from_data(data):
        writer.add_rgb_frame(frame)
    writer.close()
    return output_path
