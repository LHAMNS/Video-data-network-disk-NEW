"""
Comprehensive roundtrip tests for the Video-data-network-disk encode/decode pipeline.

Tests cover:
  1. Full encode -> decode roundtrip with data integrity verification
  2. BGR frame generation correctness
  3. AVI file RIFF header structure
  4. AVI frame count in headers
  5. Hardware detection module
  6. Color palette RGB/BGR consistency
  7. Frame data capacity (bytes_per_frame) calculation
  8. Video parameter calculation utility
"""

import os
import sys
import struct
import tempfile
import unittest

import numpy as np

# Ensure the project root is on sys.path so ``converter`` can be imported
# regardless of where pytest / unittest is invoked from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from converter import (
    COLOR_PALETTE_16,
    COLOR_PALETTE_16_BGR,
    VIDEO_PRESETS,
)
from converter.frame_generator import (
    FrameGenerator,
    OptimizedFrameGenerator,
    generate_color_lut,
    COLOR_PALETTE_16_ARRAY,
    COLOR_PALETTE_16_BGR_ARRAY,
)
from converter.avi_writer import SimpleAVIWriter
from converter.utils import (
    bytes_to_color_indices,
    calculate_video_params,
)
from converter.decoder import (
    indices_to_bytes,
    extract_frame_data,
    color_to_index,
)
from converter.hardware import (
    detect_hardware,
    HardwareProfile,
    CPUInfo,
    GPUInfo,
    get_optimal_thread_count,
    get_hardware_summary,
    _detect_platform,
    _detect_cpu,
)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _ffmpeg_available() -> bool:
    """Return True if ffmpeg is on PATH."""
    import subprocess
    try:
        r = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            timeout=5,
        )
        return r.returncode == 0
    except Exception:
        return False


def _read_raw_avi_frames(avi_path, width, height):
    """
    Read raw uncompressed BGR24 frames directly from an AVI file
    written by SimpleAVIWriter.  This avoids using cv2.VideoCapture,
    which can crash in some container environments.

    Each frame chunk in the movi list is: b"00dc" + <4-byte LE size> + <raw data>.
    The raw data is BGR24 (height * width * 3 bytes).

    Returns a list of numpy arrays, each shaped (height, width, 3) in BGR order.
    """
    frame_data_size = width * height * 3
    frames = []

    with open(avi_path, "rb") as f:
        data = f.read()

    # Find the movi list.  Its content starts right after the "movi" tag.
    movi_idx = data.find(b"movi")
    if movi_idx == -1:
        raise ValueError("No movi chunk found in the AVI file")

    pos = movi_idx + 4  # skip "movi"
    end = len(data)

    while pos + 8 <= end:
        chunk_tag = data[pos : pos + 4]
        chunk_size = struct.unpack("<I", data[pos + 4 : pos + 8])[0]

        if chunk_tag == b"00dc":
            raw = data[pos + 8 : pos + 8 + frame_data_size]
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
            frames.append(frame.copy())

        # Advance past this chunk (tag + size + payload, padded to 2-byte boundary)
        payload_padded = chunk_size + (chunk_size % 2)
        pos += 8 + payload_padded

    return frames


class TestSmallDataRoundtrip(unittest.TestCase):
    """
    1. test_small_data_roundtrip
    Encode a small byte string -> generate frames -> write AVI ->
    read AVI -> decode frames -> extract bytes -> verify match.

    The test uses 720p with nine_to_one=True and 16 colours so that
    it runs quickly while still exercising the full pipeline.  Because the
    AVI is uncompressed there is *no* lossy compression step: the pixel
    colours written should be *exactly* those read back.

    Frames are read back by parsing the raw AVI binary (no cv2.VideoCapture
    dependency), which avoids potential glibc crashes in some environments.
    """

    RESOLUTION = "720p"
    FPS = 30
    COLOR_COUNT = 16
    NINE_TO_ONE = True

    def _encode_and_write(self, data: bytes, avi_path: str):
        """Encode *data* into frames and write an uncompressed AVI."""
        gen = FrameGenerator(
            resolution=self.RESOLUTION,
            fps=self.FPS,
            color_count=self.COLOR_COUNT,
            nine_to_one=self.NINE_TO_ONE,
        )
        writer = SimpleAVIWriter(
            width=gen.physical_width,
            height=gen.physical_height,
            fps=self.FPS,
            output_path=avi_path,
        )
        writer.open()

        frame_count = 0
        for frame in gen.generate_frames_from_data(data, bgr=False):
            writer.add_rgb_frame(frame)
            frame_count += 1

        writer.close()
        return gen, frame_count

    def _decode_frames_from_avi(self, avi_path, gen, expected_frames):
        """
        Read raw BGR frames from the AVI, convert to RGB, and decode
        colour indices back to bytes using the decoder module.
        """
        frames_bgr = _read_raw_avi_frames(
            avi_path, gen.physical_width, gen.physical_height,
        )
        self.assertEqual(
            len(frames_bgr), expected_frames,
            f"Expected {expected_frames} frames in AVI, found {len(frames_bgr)}",
        )

        color_lut = np.array(COLOR_PALETTE_16, dtype=np.uint8)
        all_bytes = bytearray()

        for frame_bgr in frames_bgr:
            # AVI stores BGR; swap to RGB for palette matching
            frame_rgb = frame_bgr[:, :, ::-1].copy()

            indices = extract_frame_data(
                frame_rgb,
                gen.logical_width,
                gen.logical_height,
                self.NINE_TO_ONE,
                color_lut,
                self.COLOR_COUNT,
            )
            frame_bytes = indices_to_bytes(indices, self.COLOR_COUNT)
            all_bytes.extend(frame_bytes)

        return bytes(all_bytes)

    def test_small_data_roundtrip(self):
        """
        Data filling exactly one frame survives encode -> AVI -> decode.

        We use exactly bytes_per_frame bytes so the data fills the entire
        logical pixel grid.  This avoids the border-pattern code path that
        rearranges data spatially for very small payloads.
        """
        gen_tmp = FrameGenerator(
            resolution=self.RESOLUTION,
            fps=self.FPS,
            color_count=self.COLOR_COUNT,
            nine_to_one=self.NINE_TO_ONE,
        )
        original_data = os.urandom(gen_tmp.bytes_per_frame)

        with tempfile.TemporaryDirectory() as tmpdir:
            avi_path = os.path.join(tmpdir, "roundtrip.avi")

            gen, frame_count = self._encode_and_write(original_data, avi_path)

            self.assertEqual(frame_count, 1, "Expected exactly 1 frame")
            self.assertTrue(
                os.path.isfile(avi_path),
                "AVI file was not created on disk",
            )

            # Decode
            decoded_all = self._decode_frames_from_avi(avi_path, gen, frame_count)

            # Trim to original length (no padding needed for full frames)
            decoded_trimmed = decoded_all[: len(original_data)]

            self.assertEqual(
                decoded_trimmed,
                original_data,
                "Decoded data does not match the original input",
            )

    def test_roundtrip_multiple_frames(self):
        """Data spanning exactly 3 full frames survives the roundtrip."""
        gen_tmp = FrameGenerator(
            resolution=self.RESOLUTION,
            fps=self.FPS,
            color_count=self.COLOR_COUNT,
            nine_to_one=self.NINE_TO_ONE,
        )
        # Use exactly 3 frames of data so every frame is fully filled
        data_size = gen_tmp.bytes_per_frame * 3
        original_data = os.urandom(data_size)

        with tempfile.TemporaryDirectory() as tmpdir:
            avi_path = os.path.join(tmpdir, "multi_frame.avi")
            gen, frame_count = self._encode_and_write(original_data, avi_path)
            self.assertEqual(frame_count, 3)

            decoded_all = self._decode_frames_from_avi(avi_path, gen, frame_count)
            decoded_trimmed = decoded_all[: len(original_data)]

            self.assertEqual(decoded_trimmed, original_data)

    def test_roundtrip_partial_last_frame(self):
        """Data that partially fills the last frame still roundtrips correctly."""
        gen_tmp = FrameGenerator(
            resolution=self.RESOLUTION,
            fps=self.FPS,
            color_count=self.COLOR_COUNT,
            nine_to_one=self.NINE_TO_ONE,
        )
        # 2 full frames + a large partial frame (>10% so no border pattern)
        data_size = gen_tmp.bytes_per_frame * 2 + gen_tmp.bytes_per_frame // 2
        original_data = os.urandom(data_size)

        with tempfile.TemporaryDirectory() as tmpdir:
            avi_path = os.path.join(tmpdir, "partial.avi")
            gen, frame_count = self._encode_and_write(original_data, avi_path)
            self.assertEqual(frame_count, 3)

            decoded_all = self._decode_frames_from_avi(avi_path, gen, frame_count)
            decoded_trimmed = decoded_all[: len(original_data)]

            self.assertEqual(decoded_trimmed, original_data)

    def test_roundtrip_bgr_pipeline(self):
        """Roundtrip using the BGR frame generation path (zero-copy AVI write)."""
        gen = FrameGenerator(
            resolution=self.RESOLUTION,
            fps=self.FPS,
            color_count=self.COLOR_COUNT,
            nine_to_one=self.NINE_TO_ONE,
        )
        # Use exactly one full frame of data
        original_data = os.urandom(gen.bytes_per_frame)

        with tempfile.TemporaryDirectory() as tmpdir:
            avi_path = os.path.join(tmpdir, "bgr_roundtrip.avi")
            writer = SimpleAVIWriter(
                width=gen.physical_width,
                height=gen.physical_height,
                fps=self.FPS,
                output_path=avi_path,
            )
            writer.open()

            frame_count = 0
            for frame_bgr in gen.generate_frames_from_data(original_data, bgr=True):
                writer.add_bgr_frame(frame_bgr)
                frame_count += 1
            writer.close()

            self.assertEqual(frame_count, 1)

            # Decode -- raw AVI reader returns BGR frames
            decoded_all = self._decode_frames_from_avi(avi_path, gen, frame_count)
            decoded_trimmed = decoded_all[: len(original_data)]
            self.assertEqual(decoded_trimmed, original_data)


class TestFrameGeneratorBGR(unittest.TestCase):
    """
    2. test_frame_generator_bgr
    Verify that ``generate_frame_bgr`` produces correct BGR frames
    whose channel ordering matches the BGR palette.
    """

    def setUp(self):
        self.gen = FrameGenerator(
            resolution="720p", fps=30, color_count=16, nine_to_one=False,
        )
        # Small data chunk: 64 bytes
        self.data_chunk = bytes(range(256)) * 1  # 256 bytes, plenty for a chunk

    def test_bgr_frame_shape(self):
        """BGR frame has the correct (H, W, 3) shape."""
        frame = self.gen.generate_frame_bgr(self.data_chunk)
        self.assertEqual(frame.shape, (720, 1280, 3))
        self.assertEqual(frame.dtype, np.uint8)

    def test_bgr_vs_rgb_channel_swap(self):
        """
        For a given data chunk the BGR frame must be the channel-swapped
        version of the RGB frame (when nine_to_one is False so there is no
        upscaling noise).
        """
        # Use enough data to fill the frame so no border pattern is applied
        data = os.urandom(self.gen.bytes_per_frame)
        rgb_frame = self.gen.generate_frame(data)
        bgr_frame = self.gen.generate_frame_bgr(data)

        # Swap BGR -> RGB and compare
        bgr_as_rgb = bgr_frame[:, :, ::-1]
        np.testing.assert_array_equal(
            rgb_frame, bgr_as_rgb,
            err_msg="BGR frame is not the channel-swap of the RGB frame",
        )

    def test_bgr_palette_colours_present(self):
        """At least some BGR palette colours appear in the output frame."""
        data = os.urandom(self.gen.bytes_per_frame)
        frame = self.gen.generate_frame_bgr(data)
        frame_flat = frame.reshape(-1, 3)

        bgr_palette_set = set(tuple(c) for c in COLOR_PALETTE_16_BGR)

        found = set()
        for pixel in frame_flat:
            t = tuple(pixel)
            if t in bgr_palette_set:
                found.add(t)

        # We expect at least a few palette colours to appear in a random frame
        self.assertGreater(
            len(found), 0,
            "No BGR palette colours were found in the generated frame",
        )

    def test_bgr_nine_to_one(self):
        """BGR frame with nine_to_one=True has the physical resolution."""
        gen_9 = FrameGenerator(
            resolution="720p", fps=30, color_count=16, nine_to_one=True,
        )
        data = os.urandom(gen_9.bytes_per_frame)
        frame = gen_9.generate_frame_bgr(data)
        self.assertEqual(frame.shape, (720, 1280, 3))


class TestAVIWriterHeaders(unittest.TestCase):
    """
    3. test_avi_writer_headers
    Verify that the AVI file starts with the correct RIFF / AVI header bytes.
    """

    def _write_dummy_avi(self, n_frames=1, width=320, height=240, fps=30):
        """Write *n_frames* black frames and return the file path."""
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "test.avi")

        writer = SimpleAVIWriter(width=width, height=height, fps=fps, output_path=path)
        writer.open()

        black_frame = np.zeros((height, width, 3), dtype=np.uint8)
        for _ in range(n_frames):
            writer.add_rgb_frame(black_frame)
        writer.close()

        return path

    def test_riff_avi_signature(self):
        """First 12 bytes must be RIFF<size>AVI ."""
        path = self._write_dummy_avi(n_frames=1)
        try:
            with open(path, "rb") as f:
                header = f.read(12)

            self.assertEqual(len(header), 12)
            self.assertEqual(header[:4], b"RIFF", "Missing RIFF tag")
            # Bytes 4..7 are the file-size minus 8 (little-endian uint32).
            riff_size = struct.unpack("<I", header[4:8])[0]
            file_size = os.path.getsize(path)
            self.assertEqual(
                riff_size,
                file_size - 8,
                "RIFF size field does not match actual file size minus 8",
            )
            self.assertEqual(header[8:12], b"AVI ", "Missing AVI  form type")
        finally:
            os.unlink(path)
            os.rmdir(os.path.dirname(path))

    def test_hdrl_list_present(self):
        """The header should contain a LIST/hdrl chunk."""
        path = self._write_dummy_avi(n_frames=1)
        try:
            with open(path, "rb") as f:
                data = f.read(256)  # headers fit in the first 256 bytes

            # Find LIST tag after the initial 12 bytes
            pos = data.find(b"LIST", 12)
            self.assertNotEqual(pos, -1, "No LIST chunk found after RIFF header")

            # The list type should be 'hdrl'
            list_type = data[pos + 8 : pos + 12]
            self.assertEqual(list_type, b"hdrl", "First LIST is not hdrl")
        finally:
            os.unlink(path)
            os.rmdir(os.path.dirname(path))

    def test_movi_list_present(self):
        """The file should contain a LIST/movi chunk."""
        path = self._write_dummy_avi(n_frames=1)
        try:
            with open(path, "rb") as f:
                data = f.read()

            # movi may appear after hdrl
            idx = data.find(b"movi")
            self.assertNotEqual(idx, -1, "No movi chunk found in the AVI file")
        finally:
            os.unlink(path)
            os.rmdir(os.path.dirname(path))

    def test_avih_chunk_present(self):
        """An 'avih' chunk (main AVI header) must be present."""
        path = self._write_dummy_avi(n_frames=1)
        try:
            with open(path, "rb") as f:
                data = f.read(512)

            idx = data.find(b"avih")
            self.assertNotEqual(idx, -1, "No avih chunk in the AVI file")
        finally:
            os.unlink(path)
            os.rmdir(os.path.dirname(path))

    def test_strf_bitmapinfo(self):
        """The stream format chunk (strf) should specify 24-bit RGB."""
        path = self._write_dummy_avi(n_frames=1, width=640, height=480)
        try:
            with open(path, "rb") as f:
                data = f.read(512)

            idx = data.find(b"strf")
            self.assertNotEqual(idx, -1, "No strf chunk")

            # strf chunk: 4 bytes tag + 4 bytes size + BITMAPINFOHEADER
            bih_start = idx + 8  # skip tag + size
            # BITMAPINFOHEADER fields (all little-endian):
            #   biSize (4), biWidth (4), biHeight (4), biPlanes (2), biBitCount (2)
            bi_size, bi_width, bi_height = struct.unpack_from("<III", data, bih_start)
            bi_planes, bi_bitcount = struct.unpack_from("<HH", data, bih_start + 12)

            self.assertEqual(bi_size, 40, "BITMAPINFOHEADER.biSize should be 40")
            self.assertEqual(bi_width, 640)
            self.assertEqual(bi_height, 480)
            self.assertEqual(bi_planes, 1)
            self.assertEqual(bi_bitcount, 24, "Expected 24-bit RGB")
        finally:
            os.unlink(path)
            os.rmdir(os.path.dirname(path))


class TestAVIWriterFrameCount(unittest.TestCase):
    """
    4. test_avi_writer_frame_count
    Write N frames, verify the avih header's dwTotalFrames matches.
    """

    def _write_and_check_frame_count(self, n_frames, width=320, height=240):
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "count.avi")

        writer = SimpleAVIWriter(
            width=width, height=height, fps=30, output_path=path,
        )
        writer.open()

        frame = np.zeros((height, width, 3), dtype=np.uint8)
        for _ in range(n_frames):
            writer.add_rgb_frame(frame)
        writer.close()

        # Read the dwTotalFrames field from the avih header.
        # According to SimpleAVIWriter, this is at offset 48.
        with open(path, "rb") as f:
            f.seek(48)
            stored_count = struct.unpack("<I", f.read(4))[0]

        os.unlink(path)
        os.rmdir(tmpdir)
        return stored_count

    def test_single_frame(self):
        self.assertEqual(self._write_and_check_frame_count(1), 1)

    def test_ten_frames(self):
        self.assertEqual(self._write_and_check_frame_count(10), 10)

    def test_fifty_frames(self):
        self.assertEqual(self._write_and_check_frame_count(50), 50)

    def test_zero_frames(self):
        """An AVI with zero data frames should report 0."""
        self.assertEqual(self._write_and_check_frame_count(0), 0)

    def test_bgr_frame_count(self):
        """Frame count should be correct when using add_bgr_frame."""
        n_frames = 7
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "bgr_count.avi")

        width, height = 320, 240
        writer = SimpleAVIWriter(width=width, height=height, fps=30, output_path=path)
        writer.open()

        frame = np.zeros((height, width, 3), dtype=np.uint8)
        for _ in range(n_frames):
            writer.add_bgr_frame(frame)
        writer.close()

        with open(path, "rb") as f:
            f.seek(48)
            stored_count = struct.unpack("<I", f.read(4))[0]

        os.unlink(path)
        os.rmdir(tmpdir)
        self.assertEqual(stored_count, n_frames)


class TestHardwareDetection(unittest.TestCase):
    """
    5. test_hardware_detection
    Ensure the hardware detection module returns valid, well-structured data
    without crashing -- even in environments with no GPU or FFmpeg.
    """

    def test_detect_hardware_returns_profile(self):
        """detect_hardware() must return a HardwareProfile instance."""
        profile = detect_hardware(force_refresh=True)
        self.assertIsInstance(profile, HardwareProfile)

    def test_profile_has_platform(self):
        profile = detect_hardware()
        self.assertIn(
            profile.platform,
            ("linux", "darwin", "windows", ""),
            "Platform should be one of linux/darwin/windows or empty",
        )

    def test_profile_cpu_info(self):
        """CPU info must have valid core counts."""
        profile = detect_hardware()
        cpu = profile.cpu
        self.assertIsInstance(cpu, CPUInfo)
        self.assertGreaterEqual(cpu.cores_physical, 1)
        self.assertGreaterEqual(cpu.cores_logical, 1)
        self.assertIsInstance(cpu.architecture, str)
        self.assertIsInstance(cpu.features, list)

    def test_profile_gpus_is_list(self):
        profile = detect_hardware()
        self.assertIsInstance(profile.gpus, list)
        for gpu in profile.gpus:
            self.assertIsInstance(gpu, GPUInfo)
            self.assertIsInstance(gpu.name, str)
            self.assertIn(
                gpu.vendor,
                ("nvidia", "amd", "intel", "apple", "unknown"),
            )

    def test_recommended_backend_valid(self):
        profile = detect_hardware()
        valid_backends = {"cuda", "rocm", "metal", "opencl", "cpu", "none"}
        self.assertIn(profile.recommended_backend, valid_backends)

    def test_optimal_thread_count(self):
        count = get_optimal_thread_count()
        self.assertGreaterEqual(count, 1)

    def test_hardware_summary_keys(self):
        summary = get_hardware_summary()
        self.assertIsInstance(summary, dict)
        for key in ("platform", "cpu", "gpus", "recommended_backend",
                     "ffmpeg_encoders", "ffmpeg_decoders", "optimal_threads"):
            self.assertIn(key, summary, f"Missing key '{key}' in hardware summary")

    def test_detect_platform(self):
        plat = _detect_platform()
        self.assertIsInstance(plat, str)
        self.assertGreater(len(plat), 0)

    def test_detect_cpu(self):
        cpu = _detect_cpu()
        self.assertIsInstance(cpu, CPUInfo)
        self.assertGreaterEqual(cpu.cores_logical, 1)


class TestColorPaletteConsistency(unittest.TestCase):
    """
    6. test_color_palette_consistency
    Verify that the RGB and BGR palettes are consistent (one is the
    channel-reverse of the other) and that the NumPy array forms match.
    """

    def test_palette_length(self):
        self.assertEqual(len(COLOR_PALETTE_16), 16)
        self.assertEqual(len(COLOR_PALETTE_16_BGR), 16)

    def test_bgr_is_reverse_of_rgb(self):
        """Each BGR entry must be (B, G, R) where (R, G, B) is the RGB entry."""
        for i, (rgb, bgr) in enumerate(zip(COLOR_PALETTE_16, COLOR_PALETTE_16_BGR)):
            r, g, b = rgb
            expected_bgr = (b, g, r)
            self.assertEqual(
                bgr, expected_bgr,
                f"Palette index {i}: RGB={rgb} but BGR={bgr}, expected {expected_bgr}",
            )

    def test_numpy_arrays_match_lists(self):
        """The NumPy palette arrays must match the Python lists."""
        for i in range(16):
            np.testing.assert_array_equal(
                COLOR_PALETTE_16_ARRAY[i],
                np.array(COLOR_PALETTE_16[i], dtype=np.uint8),
            )
            np.testing.assert_array_equal(
                COLOR_PALETTE_16_BGR_ARRAY[i],
                np.array(COLOR_PALETTE_16_BGR[i], dtype=np.uint8),
            )

    def test_numpy_arrays_dtype(self):
        self.assertEqual(COLOR_PALETTE_16_ARRAY.dtype, np.uint8)
        self.assertEqual(COLOR_PALETTE_16_BGR_ARRAY.dtype, np.uint8)

    def test_numpy_arrays_shape(self):
        self.assertEqual(COLOR_PALETTE_16_ARRAY.shape, (16, 3))
        self.assertEqual(COLOR_PALETTE_16_BGR_ARRAY.shape, (16, 3))

    def test_palette_values_in_range(self):
        """All palette values must be in [0, 255]."""
        for rgb in COLOR_PALETTE_16:
            for ch in rgb:
                self.assertGreaterEqual(ch, 0)
                self.assertLessEqual(ch, 255)

    def test_color_lut_generation(self):
        """generate_color_lut should produce a (16, 3) uint8 array for 16 colours."""
        lut = generate_color_lut(COLOR_PALETTE_16_ARRAY, 16)
        self.assertEqual(lut.shape, (16, 3))
        self.assertEqual(lut.dtype, np.uint8)
        # First 16 entries should match the palette
        np.testing.assert_array_equal(lut, COLOR_PALETTE_16_ARRAY)

    def test_color_lut_extended(self):
        """generate_color_lut for >16 colours should still start with the base palette."""
        lut = generate_color_lut(COLOR_PALETTE_16_ARRAY, 256)
        self.assertEqual(lut.shape, (256, 3))
        # First 16 entries must match
        np.testing.assert_array_equal(lut[:16], COLOR_PALETTE_16_ARRAY)


class TestFrameDataCapacity(unittest.TestCase):
    """
    7. test_frame_data_capacity
    Verify that ``bytes_per_frame`` is calculated correctly for various
    resolution / colour-count / nine-to-one combinations.
    """

    def _expected_bytes_per_frame(self, resolution, color_count, nine_to_one):
        preset = VIDEO_PRESETS[resolution]
        w, h = preset["width"], preset["height"]
        if nine_to_one:
            lw, lh = w // 3, h // 3
        else:
            lw, lh = w, h
        bits_per_pixel = 4 if color_count == 16 else 8
        return lw * lh * bits_per_pixel // 8

    def test_4k_16c_9to1(self):
        gen = FrameGenerator(resolution="4K", color_count=16, nine_to_one=True)
        expected = self._expected_bytes_per_frame("4K", 16, True)
        self.assertEqual(gen.bytes_per_frame, expected)

    def test_1080p_16c_9to1(self):
        gen = FrameGenerator(resolution="1080p", color_count=16, nine_to_one=True)
        expected = self._expected_bytes_per_frame("1080p", 16, True)
        self.assertEqual(gen.bytes_per_frame, expected)

    def test_720p_16c_9to1(self):
        gen = FrameGenerator(resolution="720p", color_count=16, nine_to_one=True)
        expected = self._expected_bytes_per_frame("720p", 16, True)
        self.assertEqual(gen.bytes_per_frame, expected)

    def test_720p_16c_no_9to1(self):
        gen = FrameGenerator(resolution="720p", color_count=16, nine_to_one=False)
        expected = self._expected_bytes_per_frame("720p", 16, False)
        self.assertEqual(gen.bytes_per_frame, expected)

    def test_720p_256c_9to1(self):
        gen = FrameGenerator(resolution="720p", color_count=256, nine_to_one=True)
        expected = self._expected_bytes_per_frame("720p", 256, True)
        self.assertEqual(gen.bytes_per_frame, expected)

    def test_720p_256c_no_9to1(self):
        gen = FrameGenerator(resolution="720p", color_count=256, nine_to_one=False)
        expected = self._expected_bytes_per_frame("720p", 256, False)
        self.assertEqual(gen.bytes_per_frame, expected)

    def test_4k_256c_no_9to1(self):
        gen = FrameGenerator(resolution="4K", color_count=256, nine_to_one=False)
        expected = self._expected_bytes_per_frame("4K", 256, False)
        self.assertEqual(gen.bytes_per_frame, expected)

    def test_estimate_frame_count_single(self):
        """Data exactly fitting one frame should yield frame_count == 1."""
        gen = FrameGenerator(resolution="720p", color_count=16, nine_to_one=True)
        self.assertEqual(gen.estimate_frame_count(gen.bytes_per_frame), 1)

    def test_estimate_frame_count_multiple(self):
        gen = FrameGenerator(resolution="720p", color_count=16, nine_to_one=True)
        # bytes_per_frame + 1 should need 2 frames
        self.assertEqual(gen.estimate_frame_count(gen.bytes_per_frame + 1), 2)

    def test_estimate_frame_count_zero(self):
        gen = FrameGenerator(resolution="720p", color_count=16, nine_to_one=True)
        self.assertEqual(gen.estimate_frame_count(0), 0)

    def test_logical_pixel_count(self):
        gen = FrameGenerator(resolution="720p", color_count=16, nine_to_one=True)
        expected = gen.logical_width * gen.logical_height
        self.assertEqual(gen.calculate_logical_pixel_count(), expected)


class TestCalculateVideoParams(unittest.TestCase):
    """
    8. test_calculate_video_params
    Test the ``calculate_video_params`` utility from ``converter.utils``.
    """

    def test_returns_dict(self):
        result = calculate_video_params(file_size=1024 * 1024)
        self.assertIsInstance(result, dict)

    def test_required_keys(self):
        result = calculate_video_params(file_size=1024 * 1024)
        expected_keys = {
            "total_frames",
            "duration_seconds",
            "duration_formatted",
            "estimated_video_size",
            "estimated_video_size_mb",
            "logical_width",
            "logical_height",
            "physical_width",
            "physical_height",
            "bytes_per_frame",
        }
        for key in expected_keys:
            self.assertIn(key, result, f"Missing key: {key}")

    def test_4k_dimensions(self):
        result = calculate_video_params(file_size=1024, resolution="4K")
        self.assertEqual(result["physical_width"], 3840)
        self.assertEqual(result["physical_height"], 2160)

    def test_1080p_dimensions(self):
        result = calculate_video_params(file_size=1024, resolution="1080p")
        self.assertEqual(result["physical_width"], 1920)
        self.assertEqual(result["physical_height"], 1080)

    def test_720p_dimensions(self):
        result = calculate_video_params(file_size=1024, resolution="720p")
        self.assertEqual(result["physical_width"], 1280)
        self.assertEqual(result["physical_height"], 720)

    def test_logical_dimensions_nine_to_one(self):
        result = calculate_video_params(
            file_size=1024, resolution="720p", nine_to_one=True,
        )
        self.assertEqual(result["logical_width"], 1280 // 3)
        self.assertEqual(result["logical_height"], 720 // 3)

    def test_logical_dimensions_no_nine_to_one(self):
        result = calculate_video_params(
            file_size=1024, resolution="720p", nine_to_one=False,
        )
        self.assertEqual(result["logical_width"], 1280)
        self.assertEqual(result["logical_height"], 720)

    def test_total_frames_positive(self):
        result = calculate_video_params(file_size=10 * 1024 * 1024)
        self.assertGreater(result["total_frames"], 0)

    def test_duration_positive(self):
        result = calculate_video_params(file_size=10 * 1024 * 1024)
        self.assertGreater(result["duration_seconds"], 0)

    def test_duration_formatted(self):
        """Duration format should be MM:SS."""
        result = calculate_video_params(file_size=10 * 1024 * 1024)
        fmt = result["duration_formatted"]
        self.assertRegex(fmt, r"^\d{2}:\d{2}$")

    def test_error_correction_reduces_capacity(self):
        """Non-zero error_correction_ratio should reduce bytes_per_frame."""
        r0 = calculate_video_params(file_size=1024, error_correction_ratio=0.0)
        r10 = calculate_video_params(file_size=1024, error_correction_ratio=0.1)
        self.assertGreater(r0["bytes_per_frame"], r10["bytes_per_frame"])

    def test_256_color_more_capacity(self):
        """256 colours should double the per-frame capacity vs 16 colours."""
        r16 = calculate_video_params(
            file_size=1024, color_count=16, error_correction_ratio=0,
        )
        r256 = calculate_video_params(
            file_size=1024, color_count=256, error_correction_ratio=0,
        )
        self.assertEqual(r256["bytes_per_frame"], r16["bytes_per_frame"] * 2)

    def test_higher_fps_shorter_duration(self):
        """Higher FPS means shorter video for the same data."""
        r30 = calculate_video_params(file_size=10 * 1024 * 1024, fps=30)
        r60 = calculate_video_params(file_size=10 * 1024 * 1024, fps=60)
        # Same number of frames, but 60fps plays twice as fast
        self.assertLess(r60["duration_seconds"], r30["duration_seconds"])


# ---------------------------------------------------------------------------
# Additional low-level tests
# ---------------------------------------------------------------------------

class TestBytesToColorIndices(unittest.TestCase):
    """Verify the bytes <-> colour-index conversion is lossless."""

    def test_16_color_roundtrip(self):
        """bytes -> 4-bit indices -> bytes must be identity."""
        original = os.urandom(128)
        indices = bytes_to_color_indices(original, 16)
        recovered = indices_to_bytes(indices, 16)
        self.assertEqual(bytes(recovered), original)

    def test_256_color_roundtrip(self):
        """bytes -> 8-bit indices -> bytes must be identity."""
        original = os.urandom(128)
        indices = bytes_to_color_indices(original, 256)
        recovered = indices_to_bytes(indices, 256)
        self.assertEqual(bytes(recovered), original)

    def test_16_color_index_range(self):
        """All 4-bit indices must be in [0, 15]."""
        data = os.urandom(256)
        indices = bytes_to_color_indices(data, 16)
        self.assertTrue(np.all(indices <= 15))
        self.assertTrue(np.all(indices >= 0))

    def test_16_color_index_count(self):
        """16-colour mode produces 2 indices per byte."""
        data = os.urandom(100)
        indices = bytes_to_color_indices(data, 16)
        self.assertEqual(len(indices), 200)

    def test_256_color_identity(self):
        """256-colour indices are just the raw bytes."""
        data = os.urandom(100)
        indices = bytes_to_color_indices(data, 256)
        np.testing.assert_array_equal(indices, np.frombuffer(data, dtype=np.uint8))


class TestColorToIndex(unittest.TestCase):
    """Verify that exact palette pixels map back to the correct index."""

    def test_exact_palette_match(self):
        """Feeding an exact palette colour to color_to_index returns the right index."""
        lut = np.array(COLOR_PALETTE_16, dtype=np.uint8)
        for expected_idx in range(16):
            pixel = lut[expected_idx]
            recovered = color_to_index(pixel, lut, 16)
            self.assertEqual(
                recovered, expected_idx,
                f"Palette pixel {tuple(pixel)} mapped to index {recovered}, "
                f"expected {expected_idx}",
            )


class TestAVIWriterFileSize(unittest.TestCase):
    """Sanity-check that the AVI file size grows linearly with frame count."""

    def test_file_size_proportional(self):
        width, height = 64, 48
        frame_data_size = width * height * 3

        sizes = {}
        for n in (1, 5, 10):
            tmpdir = tempfile.mkdtemp()
            path = os.path.join(tmpdir, "sz.avi")
            writer = SimpleAVIWriter(width=width, height=height, fps=30, output_path=path)
            writer.open()
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            for _ in range(n):
                writer.add_rgb_frame(frame)
            writer.close()
            sizes[n] = os.path.getsize(path)
            os.unlink(path)
            os.rmdir(tmpdir)

        # Each additional frame adds exactly (8 + frame_data_size) bytes
        # (4 bytes chunk tag + 4 bytes chunk size + raw data).
        chunk_size = 8 + frame_data_size
        delta_5_1 = sizes[5] - sizes[1]
        delta_10_5 = sizes[10] - sizes[5]

        self.assertEqual(delta_5_1, 4 * chunk_size)
        self.assertEqual(delta_10_5, 5 * chunk_size)


if __name__ == "__main__":
    unittest.main()
