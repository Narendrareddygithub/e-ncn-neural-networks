"""Energy Profiling and Measurement Tools for E-NCN.

This module provides comprehensive tools for measuring and profiling
energy consumption of E-NCN networks on various hardware platforms.

Key Features:
- GPU power monitoring via NVIDIA-ML
- CPU energy measurement via RAPL counters
- Memory bandwidth profiling
- FLOP counting and operation analysis
- Real-time energy tracking
- Comparative benchmarking
"""

import time
import torch
import psutil
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager
import threading
import json
from pathlib import Path
import warnings

try:
    import pynvml
    import nvidia_ml_py3 as nml
    NVIDIA_AVAILABLE = True
except ImportError:
    NVIDIA_AVAILABLE = False
    warnings.warn("NVIDIA monitoring not available. Install nvidia-ml-py3 for GPU profiling.")

try:
    import py_cpuinfo
    CPU_INFO_AVAILABLE = True
except ImportError:
    CPU_INFO_AVAILABLE = False
    warnings.warn("CPU info not available. Install py-cpuinfo for detailed CPU monitoring.")


@dataclass
class EnergyMeasurement:
    """Container for energy measurement data."""
    timestamp: float
    gpu_power_watts: float = 0.0
    cpu_power_watts: float = 0.0
    memory_power_watts: float = 0.0
    total_power_watts: float = 0.0
    gpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    temperature_c: float = 0.0
    flops_performed: int = 0
    memory_accesses: int = 0
    
    def __post_init__(self):
        self.total_power_watts = (
            self.gpu_power_watts + 
            self.cpu_power_watts + 
            self.memory_power_watts
        )


@dataclass
class BenchmarkResult:
    """Results from energy benchmarking."""
    model_name: str
    dataset: str
    batch_size: int
    input_shape: Tuple[int, ...]
    
    # Performance metrics
    accuracy: float = 0.0
    inference_time_ms: float = 0.0
    throughput_samples_per_sec: float = 0.0
    
    # Energy metrics
    total_energy_joules: float = 0.0
    energy_per_sample_mj: float = 0.0
    avg_power_watts: float = 0.0
    
    # Computational metrics
    total_flops: int = 0
    sparse_flops: int = 0
    sparsity_ratio: float = 0.0
    flops_per_joule: float = 0.0
    
    # Memory metrics
    peak_memory_mb: float = 0.0
    memory_bandwidth_gb_per_s: float = 0.0
    
    # Comparison metrics (vs baseline)
    energy_reduction_ratio: float = 1.0
    speedup_ratio: float = 1.0
    efficiency_improvement: float = 1.0
    
    measurements: List[EnergyMeasurement] = field(default_factory=list)


class HardwareProfiler:
    """Hardware-specific energy profiling."""
    
    def __init__(self, enable_gpu: bool = True, enable_cpu: bool = True):
        self.enable_gpu = enable_gpu and NVIDIA_AVAILABLE
        self.enable_cpu = enable_cpu
        self.is_initialized = False
        
        if self.enable_gpu:
            self._init_gpu_monitoring()
        if self.enable_cpu:
            self._init_cpu_monitoring()
            
    def _init_gpu_monitoring(self):
        """Initialize GPU monitoring via NVIDIA-ML."""
        try:
            pynvml.nvmlInit()
            self.gpu_count = pynvml.nvmlDeviceGetCount()
            self.gpu_handles = []
            
            for i in range(self.gpu_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                self.gpu_handles.append(handle)
                
            self.is_initialized = True
            print(f"Initialized GPU monitoring for {self.gpu_count} GPUs")
            
        except Exception as e:
            warnings.warn(f"Failed to initialize GPU monitoring: {e}")
            self.enable_gpu = False
            
    def _init_cpu_monitoring(self):
        """Initialize CPU monitoring."""
        try:
            # Check for RAPL support (Linux)
            self.rapl_available = Path("/sys/class/powercap/intel-rapl").exists()
            if not self.rapl_available:
                warnings.warn("RAPL not available. CPU energy estimates will be approximate.")
                
        except Exception as e:
            warnings.warn(f"CPU monitoring initialization failed: {e}")
            
    def measure_gpu_power(self) -> Dict[str, float]:
        """Measure current GPU power consumption."""
        if not self.enable_gpu:
            return {'power_watts': 0.0, 'utilization': 0.0, 'temperature': 0.0}
            
        try:
            total_power = 0.0
            total_util = 0.0
            total_temp = 0.0
            
            for handle in self.gpu_handles:
                # Power consumption in milliwatts
                power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                power_w = power_mw / 1000.0
                total_power += power_w
                
                # GPU utilization
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                total_util += util.gpu
                
                # Temperature
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                total_temp += temp
                
            return {
                'power_watts': total_power,
                'utilization': total_util / self.gpu_count,
                'temperature': total_temp / self.gpu_count
            }
            
        except Exception as e:
            warnings.warn(f"GPU power measurement failed: {e}")
            return {'power_watts': 0.0, 'utilization': 0.0, 'temperature': 0.0}
            
    def measure_cpu_power(self) -> float:
        """Estimate CPU power consumption."""
        if not self.enable_cpu:
            return 0.0
            
        try:
            # Use CPU utilization as proxy for power consumption
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Rough estimate: Modern CPUs use 15-150W depending on load
            # This is a simplified model - actual measurement would use RAPL
            base_power = 25.0  # Idle power consumption
            max_additional_power = 100.0  # Additional power under full load
            
            estimated_power = base_power + (cpu_percent / 100.0) * max_additional_power
            return estimated_power
            
        except Exception as e:
            warnings.warn(f"CPU power estimation failed: {e}")
            return 0.0
            
    def get_memory_stats(self) -> Dict[str, float]:
        """Get memory usage statistics."""
        try:
            memory = psutil.virtual_memory()
            
            # Estimate memory power (rough approximation)
            # DDR4 uses approximately 3-5W per 8GB module
            memory_gb = memory.total / (1024**3)
            estimated_memory_power = (memory_gb / 8.0) * 4.0 * (memory.percent / 100.0)
            
            return {
                'power_watts': estimated_memory_power,
                'utilization_percent': memory.percent,
                'available_gb': memory.available / (1024**3),
                'total_gb': memory_gb
            }
            
        except Exception as e:
            warnings.warn(f"Memory stats collection failed: {e}")
            return {'power_watts': 0.0, 'utilization_percent': 0.0, 'available_gb': 0.0, 'total_gb': 0.0}
            
    def take_measurement(self) -> EnergyMeasurement:
        """Take a single energy measurement."""
        timestamp = time.time()
        
        gpu_stats = self.measure_gpu_power()
        cpu_power = self.measure_cpu_power()
        memory_stats = self.get_memory_stats()
        
        return EnergyMeasurement(
            timestamp=timestamp,
            gpu_power_watts=gpu_stats['power_watts'],
            cpu_power_watts=cpu_power,
            memory_power_watts=memory_stats['power_watts'],
            gpu_utilization=gpu_stats['utilization'],
            memory_utilization=memory_stats['utilization_percent'],
            temperature_c=gpu_stats['temperature']
        )


class EnergyProfiler:
    """High-level energy profiling interface."""
    
    def __init__(self, sampling_rate_hz: float = 10.0):
        self.sampling_rate_hz = sampling_rate_hz
        self.sampling_interval = 1.0 / sampling_rate_hz
        
        self.hardware_profiler = HardwareProfiler()
        self.measurements: List[EnergyMeasurement] = []
        
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        
    def start_monitoring(self):
        """Start continuous energy monitoring in background thread."""
        if self._monitoring:
            warnings.warn("Energy monitoring already active")
            return
            
        self._monitoring = True
        self.measurements.clear()
        
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop energy monitoring."""
        if not self._monitoring:
            return
            
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
            
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self._monitoring:
            try:
                measurement = self.hardware_profiler.take_measurement()
                self.measurements.append(measurement)
                time.sleep(self.sampling_interval)
                
            except Exception as e:
                warnings.warn(f"Monitoring error: {e}")
                time.sleep(self.sampling_interval)
                
    @contextmanager
    def profile_execution(self, model: torch.nn.Module, description: str = ""):
        """Context manager for profiling model execution.
        
        Usage:
            profiler = EnergyProfiler()
            with profiler.profile_execution(model, "MNIST inference"):
                outputs = model(inputs)
        """
        print(f"Starting energy profiling: {description}")
        
        # Clear previous measurements
        self.measurements.clear()
        
        # Start monitoring
        self.start_monitoring()
        
        # Record start time
        start_time = time.time()
        
        try:
            yield self
        finally:
            # Stop monitoring
            end_time = time.time()
            self.stop_monitoring()
            
            duration = end_time - start_time
            print(f"Profiling completed. Duration: {duration:.2f}s, Samples: {len(self.measurements)}")
            
    def get_energy_summary(self) -> Dict[str, Any]:
        """Calculate energy consumption summary."""
        if not self.measurements:
            return {'error': 'No measurements available'}
            
        total_energy_joules = 0.0
        avg_power_watts = 0.0
        peak_power_watts = 0.0
        
        for i, measurement in enumerate(self.measurements):
            if i > 0:
                # Calculate energy as power × time
                time_delta = measurement.timestamp - self.measurements[i-1].timestamp
                energy_delta = measurement.total_power_watts * time_delta
                total_energy_joules += energy_delta
                
            avg_power_watts += measurement.total_power_watts
            peak_power_watts = max(peak_power_watts, measurement.total_power_watts)
            
        avg_power_watts /= len(self.measurements)
        
        duration = self.measurements[-1].timestamp - self.measurements[0].timestamp
        
        return {
            'duration_seconds': duration,
            'total_energy_joules': total_energy_joules,
            'average_power_watts': avg_power_watts,
            'peak_power_watts': peak_power_watts,
            'num_samples': len(self.measurements),
            'sampling_rate_hz': len(self.measurements) / duration if duration > 0 else 0
        }


class FLOPCounter:
    """FLOP counting for neural network operations."""
    
    def __init__(self):
        self.total_flops = 0
        self.sparse_flops = 0
        
    def count_linear_layer(self, input_features: int, output_features: int, 
                          batch_size: int, sparsity: float = 0.0) -> int:
        """Count FLOPs for linear layer.
        
        Args:
            input_features: Number of input features
            output_features: Number of output features  
            batch_size: Batch size
            sparsity: Sparsity ratio (0.0 = dense, 1.0 = fully sparse)
            
        Returns:
            Number of FLOPs
        """
        # Dense computation: batch_size * input_features * output_features
        dense_flops = batch_size * input_features * output_features
        
        # Sparse computation: reduced by sparsity factor
        sparse_flops = int(dense_flops * (1.0 - sparsity))
        
        self.total_flops += dense_flops
        self.sparse_flops += sparse_flops
        
        return sparse_flops
        
    def reset(self):
        """Reset FLOP counters."""
        self.total_flops = 0
        self.sparse_flops = 0
        
    def get_reduction_ratio(self) -> float:
        """Get FLOP reduction ratio."""
        if self.sparse_flops == 0:
            return 1.0
        return self.total_flops / self.sparse_flops


class NetworkBenchmarker:
    """Comprehensive benchmarking for E-NCN networks."""
    
    def __init__(self, device: torch.device = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.profiler = EnergyProfiler()
        self.flop_counter = FLOPCounter()
        
    def benchmark_model(self, 
                       model: torch.nn.Module,
                       data_loader: torch.utils.data.DataLoader,
                       model_name: str,
                       num_batches: int = 10) -> BenchmarkResult:
        """Comprehensive model benchmarking.
        
        Args:
            model: PyTorch model to benchmark
            data_loader: Data loader for evaluation
            model_name: Name for the model
            num_batches: Number of batches to benchmark
            
        Returns:
            BenchmarkResult with comprehensive metrics
        """
        model.to(self.device)
        model.eval()
        
        # Initialize result container
        first_batch = next(iter(data_loader))
        if isinstance(first_batch, (list, tuple)):
            input_shape = first_batch[0].shape[1:]  # Exclude batch dimension
            batch_size = first_batch[0].shape[0]
        else:
            input_shape = first_batch.shape[1:]
            batch_size = first_batch.shape[0]
            
        result = BenchmarkResult(
            model_name=model_name,
            dataset="custom",
            batch_size=batch_size,
            input_shape=input_shape
        )
        
        total_samples = 0
        correct_predictions = 0
        inference_times = []
        
        print(f"Benchmarking {model_name} on {self.device}...")
        
        with torch.no_grad():
            with self.profiler.profile_execution(model, f"{model_name} benchmark"):
                for batch_idx, batch in enumerate(data_loader):
                    if batch_idx >= num_batches:
                        break
                        
                    if isinstance(batch, (list, tuple)):
                        inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
                    else:
                        inputs = batch.to(self.device)
                        targets = None
                        
                    # Measure inference time
                    start_time = time.time()
                    
                    # Forward pass
                    outputs = model(inputs)
                    
                    # Synchronize GPU if available
                    if self.device.type == 'cuda':
                        torch.cuda.synchronize()
                        
                    inference_time = (time.time() - start_time) * 1000  # Convert to ms
                    inference_times.append(inference_time)
                    
                    # Calculate accuracy if targets available
                    if targets is not None:
                        predictions = outputs.argmax(dim=1)
                        correct_predictions += (predictions == targets).sum().item()
                        
                    total_samples += inputs.size(0)
                    
                    # Count FLOPs for E-NCN layers
                    if hasattr(model, 'modules'):
                        for module in model.modules():
                            if hasattr(module, 'get_energy_stats'):
                                stats = module.get_energy_stats()
                                if 'sparse_operations' in stats:
                                    self.flop_counter.sparse_flops += stats['sparse_operations']
                                if 'total_operations' in stats:
                                    self.flop_counter.total_flops += stats['total_operations']
        
        # Calculate final metrics
        energy_summary = self.profiler.get_energy_summary()
        
        result.accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
        result.inference_time_ms = sum(inference_times) / len(inference_times)
        result.throughput_samples_per_sec = total_samples / energy_summary.get('duration_seconds', 1.0)
        
        result.total_energy_joules = energy_summary.get('total_energy_joules', 0.0)
        result.energy_per_sample_mj = (result.total_energy_joules * 1000) / total_samples
        result.avg_power_watts = energy_summary.get('average_power_watts', 0.0)
        
        result.total_flops = self.flop_counter.total_flops
        result.sparse_flops = self.flop_counter.sparse_flops
        result.sparsity_ratio = 1.0 - (self.flop_counter.sparse_flops / max(self.flop_counter.total_flops, 1))
        result.flops_per_joule = self.flop_counter.sparse_flops / max(result.total_energy_joules, 1e-9)
        
        # Memory statistics
        if self.device.type == 'cuda':
            result.peak_memory_mb = torch.cuda.max_memory_allocated(self.device) / (1024**2)
            
        result.measurements = self.profiler.measurements.copy()
        
        # Reset counters for next benchmark
        self.flop_counter.reset()
        
        return result
        
    def compare_models(self, results: List[BenchmarkResult], 
                      baseline_name: str = None) -> Dict[str, Any]:
        """Compare multiple benchmark results.
        
        Args:
            results: List of benchmark results to compare
            baseline_name: Name of baseline model for comparison
            
        Returns:
            Comparison dictionary with relative metrics
        """
        if not results:
            return {'error': 'No results provided'}
            
        # Find baseline
        baseline = None
        if baseline_name:
            for result in results:
                if result.model_name == baseline_name:
                    baseline = result
                    break
        else:
            # Use first result as baseline
            baseline = results[0]
            
        if not baseline:
            return {'error': 'Baseline not found'}
            
        comparison = {
            'baseline': baseline.model_name,
            'comparisons': []
        }
        
        for result in results:
            comp = {
                'model_name': result.model_name,
                'accuracy': result.accuracy,
                'energy_reduction': baseline.total_energy_joules / max(result.total_energy_joules, 1e-9),
                'speedup': result.throughput_samples_per_sec / max(baseline.throughput_samples_per_sec, 1e-9),
                'sparsity_ratio': result.sparsity_ratio,
                'efficiency_score': (result.accuracy * result.flops_per_joule) / max((baseline.accuracy * baseline.flops_per_joule), 1e-9)
            }
            comparison['comparisons'].append(comp)
            
        return comparison
        
    def save_results(self, results: List[BenchmarkResult], filename: str):
        """Save benchmark results to JSON file."""
        data = []
        for result in results:
            # Convert to serializable format
            result_dict = {
                'model_name': result.model_name,
                'dataset': result.dataset,
                'batch_size': result.batch_size,
                'input_shape': list(result.input_shape),
                'accuracy': result.accuracy,
                'inference_time_ms': result.inference_time_ms,
                'total_energy_joules': result.total_energy_joules,
                'energy_per_sample_mj': result.energy_per_sample_mj,
                'sparsity_ratio': result.sparsity_ratio,
                'flops_per_joule': result.flops_per_joule,
                'peak_memory_mb': result.peak_memory_mb
            }
            data.append(result_dict)
            
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
            
        print(f"Results saved to {filename}")


# Example usage
if __name__ == "__main__":
    # Test hardware profiler
    print("Testing Hardware Profiler...")
    profiler = HardwareProfiler()
    measurement = profiler.take_measurement()
    print(f"GPU Power: {measurement.gpu_power_watts:.2f}W")
    print(f"CPU Power: {measurement.cpu_power_watts:.2f}W")
    print(f"Total Power: {measurement.total_power_watts:.2f}W")
    
    # Test FLOP counter
    print("\nTesting FLOP Counter...")
    counter = FLOPCounter()
    flops = counter.count_linear_layer(784, 128, 32, sparsity=0.99)
    print(f"Sparse FLOPs: {flops:,}")
    print(f"Reduction ratio: {counter.get_reduction_ratio():.1f}x")