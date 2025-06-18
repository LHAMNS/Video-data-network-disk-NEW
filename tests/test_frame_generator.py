import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from converter.frame_generator import FrameGenerator

class TestFrameGenerator(unittest.TestCase):
    def setUp(self):
        self.test_data = os.urandom(1024 * 1024)  # 1MB random data
        self.generator = FrameGenerator(resolution="720p", fps=30, color_count=16, nine_to_one=True)

    def test_frame_generation_size(self):
        frame = self.generator.generate_frame(self.test_data[:1000], 0)
        self.assertEqual(frame.shape[0], 720)
        self.assertEqual(frame.shape[1], 1280)
        self.assertEqual(frame.shape[2], 3)

    def test_full_pipeline(self):
        frame_count = 0
        for frame in self.generator.generate_frames_from_data(self.test_data):
            frame_count += 1
            self.assertIsNotNone(frame)
            self.assertEqual(frame.shape, (720, 1280, 3))
        expected_frames = self.generator.estimate_frame_count(len(self.test_data))
        self.assertEqual(frame_count, expected_frames)

if __name__ == '__main__':
    unittest.main()
