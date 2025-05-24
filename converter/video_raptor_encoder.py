import numpy as np

class VideoRaptorEncoder:
    """Simple metadata and sync frame generator"""

    def __init__(self, width: int, height: int, fps: int = 30):
        self.width = width
        self.height = height
        self.fps = fps

    def create_metadata_frame(self, file_info: dict) -> np.ndarray:
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        return frame

    def create_calibration_frame(self) -> np.ndarray:
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        return frame

    def _add_sync_pattern(self, frame: np.ndarray) -> None:
        step_y = max(1, self.height // 20)
        step_x = max(1, self.width // 20)
        frame[::step_y, :] = 255
        frame[:, ::step_x] = 255

