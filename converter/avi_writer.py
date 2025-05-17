import struct
from pathlib import Path
import numpy as np

class SimpleAVIWriter:
    """Minimal AVI writer for uncompressed RGB24 frames."""

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

    def open(self):
        self._f = open(self.output_path, "wb")
        self._write_headers()

    def _write_headers(self):
        f = self._f
        # RIFF header placeholder
        f.write(b"RIFF\x00\x00\x00\x00AVI ")

        # === hdrl list ===
        hdrl_data = bytearray()

        # avih chunk
        microsec_per_frame = int(1e6 / self.fps)
        max_bytes_per_sec = self.width * self.height * 3 * self.fps
        padding = 0
        flags = 0x10  # HAS_INDEX
        total_frames = 0  # placeholder
        initial_frames = 0
        streams = 1
        buffer_size = self.width * self.height * 3
        width = self.width
        height = self.height
        reserved = (0, 0, 0, 0)
        avih = struct.pack(
            "<IIIIIIIIIIIIIIII",
            microsec_per_frame,
            max_bytes_per_sec,
            padding,
            flags,
            total_frames,
            initial_frames,
            streams,
            buffer_size,
            width,
            height,
            *reserved,
        )
        hdrl_data += b"avih" + struct.pack("<I", len(avih)) + avih
        self._avih_frames_offset = 20 + 8  # RIFF header 12 bytes + LIST hdrl 4 + size 4 + 'hdrl' 4 -> 20; then 'avih' + size

        # strh chunk
        strh = struct.pack(
            "<4s4sIHHIIIIIIIIhhhh",
            b"vids",
            b"DIB ",
            0,
            0,
            0,
            0,
            1,
            int(1e6 / self.fps),
            0,
            0,
            buffer_size,
            -1,
            0,
            0,
            width,
            height,
        )
        hdrl_data += b"LIST" + struct.pack("<I", 4 + 8 + len(strh) + 8 + 40)
        hdrl_data += b"strl"
        hdrl_data += b"strh" + struct.pack("<I", len(strh)) + strh
        # strf chunk
        strf = struct.pack(
            "<IIIHHIIIIII",
            40,
            width,
            height,
            1,
            24,
            0,
            buffer_size,
            0,
            0,
            0,
            0,
        )
        hdrl_data += b"strf" + struct.pack("<I", len(strf)) + strf

        f.write(b"LIST" + struct.pack("<I", len(hdrl_data) + 4) + b"hdrl" + hdrl_data)

        # === movi list ===
        f.write(b"LIST")
        self._movi_size_offset = f.tell()
        f.write(b"\x00\x00\x00\x00")
        f.write(b"movi")
        self._movi_start = f.tell()

    def add_rgb_frame(self, frame: np.ndarray):
        if self._f is None:
            self.open()
        if frame.shape[0] != self.height or frame.shape[1] != self.width:
            raise ValueError("Frame dimensions do not match")
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        # Convert RGB to BGR
        if frame.shape[2] == 3:
            frame_bgr = frame[:, :, ::-1]
        else:
            frame_bgr = frame
        data = frame_bgr.tobytes()
        self._f.write(b"00dc" + struct.pack("<I", len(data)))
        self._f.write(data)
        if len(data) % 2:
            self._f.write(b"\x00")
        self._frame_count += 1

    def close(self):
        if self._f is None:
            return
        end_pos = self._f.tell()
        movi_size = end_pos - self._movi_start
        # fill movi size
        self._f.seek(self._movi_size_offset)
        self._f.write(struct.pack("<I", movi_size + 4))
        # fill total frames
        self._f.seek(self._avih_frames_offset)
        self._f.write(struct.pack("<I", self._frame_count))
        # fill RIFF size
        self._f.seek(4)
        self._f.write(struct.pack("<I", end_pos - 8))
        self._f.close()
        self._f = None


def convert_bytes_to_avi(data: bytes, frame_generator, output_path: str, fps=30):
    """Convert raw bytes to an uncompressed AVI using the given frame generator."""
    writer = SimpleAVIWriter(
        width=frame_generator.physical_width,
        height=frame_generator.physical_height,
        fps=fps,
        output_path=output_path,
    )

    for frame in frame_generator.generate_frames_from_data(data):
        writer.add_rgb_frame(frame)
    writer.close()
    return output_path
