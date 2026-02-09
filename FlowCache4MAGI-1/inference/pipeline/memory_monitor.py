# Copyright (c) 2025 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import time
from typing import List, Optional, Dict, Any
import logging

class MemoryMonitor:
    """
    Enhanced memory monitor for tracking tensor memory usage
    """

    def __init__(self, log_file: Optional[str] = None, enable_logging: bool = True):
        self.log_file = log_file
        self.enable_logging = enable_logging
        self.peak_memory_usage = 0.0
        self.memory_history = []

        # Setup logging
        if self.enable_logging:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(log_file) if log_file else logging.StreamHandler(),
                    logging.StreamHandler()
                ]
            )
            self.logger = logging.getLogger(__name__)

    def get_tensor_memory_info(self, tensor_list: List[Any], unit: str = 'MB') -> Dict[str, Any]:
        """
        Get detailed memory information for a list of tensors
        """
        total_bytes = 0
        non_null_tensors = 0
        tensor_shapes = []
        tensor_devices = []

        for i, t in enumerate(tensor_list):
            if isinstance(t, torch.Tensor):
                non_null_tensors += 1
                tensor_shapes.append(t.shape)
                tensor_devices.append(str(t.device))

                if t.is_cuda:
                    total_bytes += t.element_size() * t.numel()
            else:
                tensor_shapes.append(None)
                tensor_devices.append(None)

        # Convert to requested unit
        unit = unit.upper()
        scale_dict = {
            'B': 1,
            'KB': 1024,
            'MB': 1024 ** 2,
            'GB': 1024 ** 3,
        }
        scale = scale_dict[unit]

        return {
            'total_memory_mb': total_bytes / scale,
            'total_memory_gb': total_bytes / (scale_dict['GB']),
            'total_elements': sum(t.numel() for t in tensor_list if isinstance(t, torch.Tensor) and t.is_cuda),
            'non_null_count': non_null_tensors,
            'total_count': len(tensor_list),
            'tensor_shapes': tensor_shapes,
            'tensor_devices': tensor_devices,
            'utilization_rate': non_null_tensors / len(tensor_list) if tensor_list else 0
        }

    def monitor_residual_memory(self, previous_residual: List[Any],
                             step: int, chunk_info: Optional[Dict] = None,
                             log_immediately: bool = True) -> Dict[str, Any]:
        """
        Monitor memory usage of previous_residual specifically
        """
        memory_info = self.get_tensor_memory_info(previous_residual, unit='MB')
        memory_info['step'] = step
        memory_info['timestamp'] = time.time()

        if chunk_info:
            memory_info.update(chunk_info)

        # Track peak memory
        if memory_info['total_memory_mb'] > self.peak_memory_usage:
            self.peak_memory_usage = memory_info['total_memory_mb']
            memory_info['is_peak'] = True
        else:
            memory_info['is_peak'] = False

        # Store history
        self.memory_history.append(memory_info)

        # Log if requested
        if log_immediately and self.enable_logging:
            self._log_memory_info(memory_info)

        return memory_info

    def _log_memory_info(self, memory_info: Dict[str, Any]):
        """
        Log memory information in a formatted way
        """
        msg = (
            f"Step {memory_info['step']:3d} | "
            f"Residual Memory: {memory_info['total_memory_mb']:6.2f} MB | "
            f"Tensors: {memory_info['non_null_count']:2d}/{memory_info['total_count']:2d} | "
            f"Utilization: {memory_info['utilization_rate']*100:5.1f}%"
        )

        if memory_info['is_peak']:
            msg += " | [NEW PEAK]"

        self.logger.info(msg)

        # Log detailed tensor shapes for debugging
        if memory_info['non_null_count'] > 0:
            shapes_str = ", ".join([str(s) for s in memory_info['tensor_shapes'] if s is not None])
            self.logger.debug(f"  Tensor shapes: {shapes_str}")

    def get_memory_summary(self) -> Dict[str, Any]:
        """
        Get summary of memory usage over time
        """
        if not self.memory_history:
            return {'error': 'No memory history available'}

        memory_values = [h['total_memory_mb'] for h in self.memory_history]

        return {
            'peak_memory_mb': max(memory_values),
            'average_memory_mb': sum(memory_values) / len(memory_values),
            'min_memory_mb': min(memory_values),
            'total_steps': len(self.memory_history),
            'peak_step': max(self.memory_history, key=lambda x: x['total_memory_mb'])['step'],
            'final_memory_mb': memory_values[-1] if memory_values else 0,
            'memory_growth': memory_values[-1] - memory_values[0] if len(memory_values) > 1 else 0
        }

    def save_memory_report(self, filename: str):
        """
        Save detailed memory report to file
        """
        import json

        summary = self.get_memory_summary()
        report = {
            'summary': summary,
            'detailed_history': self.memory_history,
            'peak_memory_gb': self.peak_memory_usage / 1024
        }

        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print(f"Memory report saved to: {filename}")

    def reset(self):
        """
        Reset monitor state
        """
        self.peak_memory_usage = 0.0
        self.memory_history = []

# Global monitor instance
_global_monitor = None

def get_memory_monitor() -> MemoryMonitor:
    """Get or create global memory monitor"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = MemoryMonitor()
    return _global_monitor

def monitor_residual_memory_step(previous_residual: List[Any], step: int,
                               chunk_info: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Convenience function to monitor a single step
    """
    monitor = get_memory_monitor()
    return monitor.monitor_residual_memory(previous_residual, step, chunk_info)