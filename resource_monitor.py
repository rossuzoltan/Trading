import os
import psutil
import subprocess
import logging
import threading
import time
from dataclasses import dataclass

@dataclass
class SystemMetrics:
    cpu_pct: float
    ram_pct: float
    gpu_pct: float | None = None
    vram_pct: float | None = None

class SystemMonitor:
    """
    Background thread to monitor system resources without blocking the training loop.
    """
    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.metrics = SystemMetrics(cpu_pct=0.0, ram_pct=0.0)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._logger = logging.getLogger("resource_monitor")

    def start(self):
        if not self._thread.is_alive():
            self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _monitor_loop(self):
        while not self._stop_event.is_set():
            try:
                # CPU and RAM are easy with psutil
                cpu = psutil.cpu_percent(interval=None)
                ram = psutil.virtual_memory().percent
                
                # GPU requires nvidia-smi (robust fallback)
                gpu, vram = self._get_gpu_metrics()
                
                self.metrics = SystemMetrics(
                    cpu_pct=cpu,
                    ram_pct=ram,
                    gpu_pct=gpu,
                    vram_pct=vram
                )
            except Exception as e:
                self._logger.error(f"Error monitoring resources: {e}")
            
            time.sleep(self.interval)

    def _get_gpu_metrics(self) -> tuple[float | None, float | None]:
        """Parse nvidia-smi output for GPU utilization and memory usage."""
        try:
            # We target the first GPU (index 0) which is usually the training GPU
            cmd = "nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv,noheader,nounits"
            output = subprocess.check_output(cmd, shell=True).decode("utf-8").strip()
            if "," in output:
                gpu_util, vram_util = output.split(",")
                return float(gpu_util), float(vram_util)
        except Exception:
            pass
        return None, None

    def get_latest(self) -> SystemMetrics:
        return self.metrics

# Global singleton for easy access in callbacks
monitor = SystemMonitor()
