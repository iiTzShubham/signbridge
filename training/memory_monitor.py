"""Background thread that samples memory use every 100 ms during a run and keeps the peak value.
"""
import threading
import time


class PeakMemoryMonitor:
    """Poll process RSS and, when available, CUDA/MPS allocations during a run."""
    def __init__(self, device, interval=0.1):
        self.device = device
        self.interval = interval
        self.stop_event = threading.Event()
        self.thread = None
        self.peak_rss_bytes = None
        self.peak_accelerator_bytes = None

    def _sample(self):
        try:
            import psutil
            process = psutil.Process()
            while not self.stop_event.wait(self.interval):
                rss = process.memory_info().rss
                self.peak_rss_bytes = max(self.peak_rss_bytes or 0, rss)
                try:
                    import torch
                    if self.device.type == "cuda":
                        allocated = torch.cuda.memory_allocated(self.device)
                    elif self.device.type == "mps":
                        allocated = torch.mps.driver_allocated_memory()
                    else:
                        allocated = 0
                    self.peak_accelerator_bytes = max(self.peak_accelerator_bytes or 0, allocated)
                except (ImportError, AttributeError, RuntimeError):
                    pass
        except ImportError:
            return

    def start(self):
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()
        return self

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=2)
        time.sleep(0)
