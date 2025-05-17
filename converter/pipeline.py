import threading
import logging

class ResourceAllocation:
    def __init__(self, manager, size):
        self.manager = manager
        self.size = size

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.manager.release(self)


class AtomicCounter:
    def __init__(self, value=0):
        self.value = value
        self.lock = threading.Lock()

    def add(self, num):
        with self.lock:
            self.value += num

    def subtract(self, num):
        with self.lock:
            self.value -= num

    def get(self):
        with self.lock:
            return self.value


class ResourceManager:
    """Simple resource manager for memory and concurrency limits."""

    def __init__(self, limits):
        self.memory_limit = limits.get("memory_mb", 1024) * 1024 * 1024
        self.concurrent_tasks = limits.get("max_concurrent_tasks", 2)
        self.current_memory = AtomicCounter(0)
        self.semaphore = threading.BoundedSemaphore(self.concurrent_tasks)

    def allocate(self, required_memory):
        self.semaphore.acquire()
        if self.current_memory.get() + required_memory > self.memory_limit:
            self.semaphore.release()
            raise MemoryError("Memory limit exceeded")
        self.current_memory.add(required_memory)
        return ResourceAllocation(self, required_memory)

    def release(self, allocation):
        self.current_memory.subtract(allocation.size)
        self.semaphore.release()


class Stage:
    """Pipeline processing stage."""

    def process(self, data, context):
        raise NotImplementedError


class ErrorHandler:
    def __init__(self, strategies=None):
        self.logger = logging.getLogger(__name__)
        self.strategies = strategies or {}

    def handle(self, error, stage, data, context):
        self.logger.error(f"Stage {stage.__class__.__name__} failed: {error}", exc_info=True)
        raise error


class ConversionPipeline:
    """Execution pipeline with pluggable stages."""

    def __init__(self, config=None):
        config = config or {}
        self.stages = []
        self.resource_manager = ResourceManager(config.get("limits", {}))
        self.error_handler = ErrorHandler(config.get("recovery_strategies", {}))

    def register_stage(self, stage):
        self.stages.append(stage)

    def execute(self, input_data, context=None):
        context = context or {}
        result = input_data
        for stage in self.stages:
            try:
                result = stage.process(result, context)
            except Exception as e:
                return self.error_handler.handle(e, stage, result, context)
        return result


# Example stages used by the web server
from .error_correction import ReedSolomonEncoder
from .frame_generator import FrameGenerator, OptimizedFrameGenerator
from .encoder import StreamingVideoEncoder


class ErrorCorrectionStage(Stage):
    def __init__(self, encoder: ReedSolomonEncoder):
        self.encoder = encoder

    def process(self, data, context):
        # Expect bytes-like object
        return self.encoder.encode_data(data)


class FrameGenerationStage(Stage):
    def __init__(self, generator: FrameGenerator, callback=None):
        self.generator = generator
        self.callback = callback

    def process(self, data, context):
        # data can be bytes or an iterable of bytes
        return self.generator.generate_frames_from_data(data, callback=self.callback)


class VideoEncodingStage(Stage):
    def __init__(self, encoder: StreamingVideoEncoder):
        self.encoder = encoder

    def process(self, frame_iter, context):
        self.encoder.start()
        for frame in frame_iter:
            if not self.encoder.add_frame(frame):
                raise RuntimeError("Failed to add frame to encoder")
        stats = self.encoder.stop()
        context["output_path"] = str(self.encoder.output_path)
        context["encoder_stats"] = stats
        return self.encoder.output_path
