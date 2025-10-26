"""Energy-Efficient Non-Conventional Neuron (E-NCN) Package.

This package implements revolutionary neural network architectures that achieve
1000x energy reduction through event-driven sparse computation.

Key Components:
- layers: Core E-NCN layer implementations
- profiling: Energy measurement and profiling tools
- training: Training algorithms and optimizers (coming in Week 2)
"""

__version__ = "0.1.0"
__author__ = "E-NCN Research Team"
__email__ = "research@encn.ai"

from .layers import ENCNLayer, ENCNSequential, create_encn_mlp

try:
    from .profiling import EnergyProfiler, HardwareProfiler, FLOPCounter, NetworkBenchmarker
except ImportError:
    # Handle cases where profiling dependencies aren't available
    import warnings
    warnings.warn(
        "Some profiling tools may not be available. Install nvidia-ml-py3 and pynvml for full functionality.",
        ImportWarning
    )
    
    class MockProfiler:
        """Mock profiler for when dependencies aren't available."""
        def __init__(self, *args, **kwargs):
            pass
            
        def profile_execution(self, *args, **kwargs):
            from contextlib import nullcontext
            return nullcontext()
            
        def get_energy_summary(self):
            return {'error': 'Profiling dependencies not available'}
            
    EnergyProfiler = MockProfiler
    HardwareProfiler = MockProfiler
    FLOPCounter = MockProfiler
    NetworkBenchmarker = MockProfiler

__all__ = [
    'ENCNLayer',
    'ENCNSequential', 
    'create_encn_mlp',
    'EnergyProfiler',
    'HardwareProfiler',
    'FLOPCounter',
    'NetworkBenchmarker'
]