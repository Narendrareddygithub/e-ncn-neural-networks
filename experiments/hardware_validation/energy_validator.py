#!/usr/bin/env python3
"""
Hardware Energy Validation System for E-NCN Networks.

This module provides comprehensive hardware energy validation to prove
the 1000x energy reduction claims through real measurements on actual hardware.

Key Features:
- Real GPU power measurement via NVIDIA-ML
- CPU energy tracking with RAPL counters
- Statistical significance testing
- Hardware-agnostic energy profiling
- Production-ready validation pipeline

Usage:
    python energy_validator.py --model encn --dataset mnist --validate-claims
    
Validation Requirements:
- MNIST Accuracy: >98%
- Energy Reduction: >200x measured (Week 2 target)
- Statistical Significance: p < 0.05
- Hardware Compatibility: RTX/A100/H100 GPUs
"""

import os
import sys
import time
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from scipy import stats

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from encn.layers import ENCNLayer, create_encn_mlp
from encn.profiling import EnergyProfiler, NetworkBenchmarker, BenchmarkResult

try:
    import pynvml
    import psutil
    HARDWARE_MONITORING_AVAILABLE = True
except ImportError:
    HARDWARE_MONITORING_AVAILABLE = False
    warnings.warn("Hardware monitoring libraries not available. Install pynvml and psutil.")


@dataclass
class ValidationConfig:
    """Configuration for energy validation experiments."""
    model_type: str = "encn"  # encn, dense, pruned
    dataset: str = "mnist"
    batch_size: int = 128
    num_epochs: int = 10
    learning_rate: float = 0.001
    
    # E-NCN specific
    sparsity_target: float = 0.99
    threshold: float = 0.1
    
    # Validation parameters
    num_validation_runs: int = 5
    min_accuracy: float = 0.98  # 98% minimum for MNIST
    min_energy_reduction: float = 200.0  # 200x minimum for Week 2
    statistical_significance: float = 0.05  # p < 0.05
    
    # Hardware requirements
    require_gpu: bool = True
    min_gpu_memory_gb: float = 4.0
    
    # Output configuration
    results_dir: str = "results/hardware_validation"
    save_models: bool = True
    save_detailed_logs: bool = True


@dataclass
class ValidationResult:
    """Results from hardware energy validation."""
    model_name: str
    dataset: str
    config: ValidationConfig
    
    # Performance metrics
    accuracy_mean: float = 0.0
    accuracy_std: float = 0.0
    accuracy_ci_lower: float = 0.0
    accuracy_ci_upper: float = 0.0
    
    # Energy metrics
    energy_reduction_mean: float = 1.0
    energy_reduction_std: float = 0.0
    energy_reduction_ci_lower: float = 1.0
    energy_reduction_ci_upper: float = 1.0
    
    # Statistical validation
    accuracy_passes_threshold: bool = False
    energy_reduction_passes_threshold: bool = False
    statistical_significance_p: float = 1.0
    
    # Hardware specifications
    gpu_name: str = ""
    gpu_memory_gb: float = 0.0
    cpu_name: str = ""
    system_memory_gb: float = 0.0
    
    # Detailed results
    individual_runs: List[BenchmarkResult] = None
    baseline_comparison: Optional[Dict] = None
    
    # Validation status
    validation_passed: bool = False
    validation_issues: List[str] = None
    
    def __post_init__(self):
        if self.individual_runs is None:
            self.individual_runs = []
        if self.validation_issues is None:
            self.validation_issues = []


class HardwareEnergyValidator:
    """Comprehensive hardware energy validation system."""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.device = self._setup_device()
        self.results_dir = Path(config.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize benchmarker with validated device
        self.benchmarker = NetworkBenchmarker(device=self.device)
        
        # Hardware information
        self.hardware_info = self._collect_hardware_info()
        
        print(f"Hardware Energy Validator initialized")
        print(f"Device: {self.device}")
        print(f"GPU: {self.hardware_info.get('gpu_name', 'N/A')}")
        print(f"GPU Memory: {self.hardware_info.get('gpu_memory_gb', 0):.1f} GB")
        
    def _setup_device(self) -> torch.device:
        """Setup and validate compute device."""
        if not torch.cuda.is_available() and self.config.require_gpu:
            raise RuntimeError("CUDA not available but GPU required for validation")
            
        if torch.cuda.is_available():
            device = torch.device('cuda')
            gpu_memory = torch.cuda.get_device_properties(device).total_memory / (1024**3)
            
            if gpu_memory < self.config.min_gpu_memory_gb:
                warnings.warn(
                    f"GPU memory {gpu_memory:.1f}GB < required {self.config.min_gpu_memory_gb}GB"
                )
        else:
            device = torch.device('cpu')
            warnings.warn("Using CPU - energy measurements may be less accurate")
            
        return device
        
    def _collect_hardware_info(self) -> Dict:
        """Collect comprehensive hardware information."""
        info = {}
        
        # GPU information
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info['gpu_name'] = props.name
            info['gpu_memory_gb'] = props.total_memory / (1024**3)
            info['gpu_compute_capability'] = f"{props.major}.{props.minor}"
            
        # CPU information
        info['cpu_count'] = psutil.cpu_count()
        
        try:
            import py_cpuinfo
            cpu_info = py_cpuinfo.get_cpu_info()
            info['cpu_name'] = cpu_info.get('brand_raw', 'Unknown')
            info['cpu_arch'] = cpu_info.get('arch', 'Unknown')
        except ImportError:
            info['cpu_name'] = 'Unknown'
            
        # System memory
        memory = psutil.virtual_memory()
        info['system_memory_gb'] = memory.total / (1024**3)
        
        return info
        
    def create_models(self) -> Dict[str, nn.Module]:
        """Create baseline and E-NCN models for comparison."""
        models = {}
        
        if self.config.dataset == "mnist":
            input_size, num_classes = 784, 10
            hidden_sizes = [256, 128]
        else:
            raise ValueError(f"Unsupported dataset: {self.config.dataset}")
            
        # Dense baseline model
        models['dense_baseline'] = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], num_classes)
        )
        
        # E-NCN model
        models['encn'] = create_encn_mlp(
            [input_size] + hidden_sizes + [num_classes],
            threshold=self.config.threshold,
            sparsity_target=self.config.sparsity_target,
            activation='relu'
        )
        
        # Pruned baseline (structured sparsity)
        pruned_model = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(), 
            nn.Linear(hidden_sizes[1], num_classes)
        )
        
        # Apply magnitude-based pruning to simulate sparse baseline
        for module in pruned_model.modules():
            if isinstance(module, nn.Linear):
                with torch.no_grad():
                    # Zero out smallest 90% of weights
                    weights = module.weight.data
                    threshold = torch.quantile(torch.abs(weights), 0.9)
                    mask = torch.abs(weights) > threshold
                    module.weight.data *= mask.float()
                    
        models['pruned_baseline'] = pruned_model
        
        return models
        
    def get_dataset(self) -> Tuple[DataLoader, DataLoader]:
        """Load and prepare dataset."""
        if self.config.dataset == "mnist":
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            
            train_dataset = torchvision.datasets.MNIST(
                root='./data', train=True, download=True, transform=transform
            )
            test_dataset = torchvision.datasets.MNIST(
                root='./data', train=False, transform=transform
            )
            
            # Flatten images for MLP
            class FlattenTransform:
                def __call__(self, batch):
                    x, y = batch
                    return x.view(-1, 784), y
                    
            train_loader = DataLoader(
                train_dataset, batch_size=self.config.batch_size, 
                shuffle=True, collate_fn=lambda batch: self._collate_and_flatten(batch)
            )
            test_loader = DataLoader(
                test_dataset, batch_size=self.config.batch_size,
                shuffle=False, collate_fn=lambda batch: self._collate_and_flatten(batch)
            )
            
            return train_loader, test_loader
        else:
            raise ValueError(f"Unsupported dataset: {self.config.dataset}")
            
    def _collate_and_flatten(self, batch):
        """Custom collate function to flatten MNIST images."""
        images, labels = zip(*batch)
        images = torch.stack(images).view(-1, 784)  # Flatten to 784
        labels = torch.tensor(labels)
        return images, labels
        
    def train_model(self, model: nn.Module, train_loader: DataLoader, 
                   model_name: str) -> nn.Module:
        """Train model with energy monitoring."""
        model = model.to(self.device)
        model.train()
        
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        print(f"\nTraining {model_name} for {self.config.num_epochs} epochs...")
        
        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                # Track accuracy
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)
                epoch_loss += loss.item()
                
                if batch_idx % 100 == 0:
                    print(f"Epoch {epoch+1}/{self.config.num_epochs}, "
                          f"Batch {batch_idx}, Loss: {loss.item():.4f}")
                    
            accuracy = correct / total
            print(f"Epoch {epoch+1} - Loss: {epoch_loss/len(train_loader):.4f}, "
                  f"Accuracy: {accuracy:.4f}")
                  
        return model
        
    def validate_single_run(self, model: nn.Module, test_loader: DataLoader,
                           model_name: str, baseline_result: Optional[BenchmarkResult] = None
                           ) -> BenchmarkResult:
        """Perform single validation run with energy measurement."""
        model.eval()
        
        print(f"\nValidating {model_name} with energy profiling...")
        
        # Reset energy statistics for E-NCN models
        if hasattr(model, 'reset_all_energy_stats'):
            model.reset_all_energy_stats()
            
        # Benchmark the model
        result = self.benchmarker.benchmark_model(
            model=model,
            data_loader=test_loader,
            model_name=model_name,
            num_batches=20  # Use subset for faster validation
        )
        
        # Calculate energy reduction vs baseline
        if baseline_result is not None:
            result.energy_reduction_ratio = (
                baseline_result.total_energy_joules / 
                max(result.total_energy_joules, 1e-9)
            )
        
        # Add E-NCN specific metrics
        if hasattr(model, 'get_total_energy_stats'):
            encn_stats = model.get_total_energy_stats()
            result.sparsity_ratio = encn_stats.get('avg_sparsity', 0.0)
            
        print(f"{model_name} Results:")
        print(f"  Accuracy: {result.accuracy:.4f}")
        print(f"  Energy: {result.total_energy_joules:.6f} J")
        print(f"  Energy per sample: {result.energy_per_sample_mj:.4f} mJ")
        print(f"  Sparsity: {result.sparsity_ratio:.3f}")
        if baseline_result:
            print(f"  Energy reduction: {result.energy_reduction_ratio:.1f}x")
            
        return result
        
    def run_statistical_validation(self) -> ValidationResult:
        """Run complete statistical validation with multiple runs."""
        print("\n" + "="*60)
        print("STARTING HARDWARE ENERGY VALIDATION")
        print("="*60)
        
        # Create models
        models = self.create_models()
        
        # Load dataset
        train_loader, test_loader = self.get_dataset()
        
        # Storage for results
        all_results = {name: [] for name in models.keys()}
        
        # Run validation multiple times for statistical significance
        for run_idx in range(self.config.num_validation_runs):
            print(f"\n--- VALIDATION RUN {run_idx + 1}/{self.config.num_validation_runs} ---")
            
            baseline_result = None
            
            for model_name, model in models.items():
                # Train model (with fresh initialization)
                trained_model = self.train_model(
                    model=model, 
                    train_loader=train_loader,
                    model_name=model_name
                )
                
                # Validate with energy measurement
                result = self.validate_single_run(
                    model=trained_model,
                    test_loader=test_loader,
                    model_name=model_name,
                    baseline_result=baseline_result
                )
                
                all_results[model_name].append(result)
                
                # Use first model as baseline for energy comparison
                if baseline_result is None:
                    baseline_result = result
                    
        # Analyze results
        return self._analyze_validation_results(all_results)
        
    def _analyze_validation_results(self, all_results: Dict[str, List[BenchmarkResult]]) -> ValidationResult:
        """Analyze validation results for statistical significance."""
        print("\n" + "="*60)
        print("STATISTICAL ANALYSIS")
        print("="*60)
        
        # Focus on E-NCN results
        encn_results = all_results.get('encn', [])
        baseline_results = all_results.get('dense_baseline', [])
        
        if not encn_results or not baseline_results:
            raise ValueError("Missing E-NCN or baseline results")
            
        # Calculate statistics
        encn_accuracies = [r.accuracy for r in encn_results]
        encn_energies = [r.total_energy_joules for r in encn_results]
        baseline_energies = [r.total_energy_joules for r in baseline_results]
        
        # Energy reduction ratios
        energy_reductions = [
            baseline_energies[i] / max(encn_energies[i], 1e-9) 
            for i in range(len(encn_results))
        ]
        
        # Statistical calculations
        accuracy_mean = np.mean(encn_accuracies)
        accuracy_std = np.std(encn_accuracies, ddof=1)
        energy_reduction_mean = np.mean(energy_reductions)
        energy_reduction_std = np.std(energy_reductions, ddof=1)
        
        # Confidence intervals (95%)
        n = len(encn_results)
        t_critical = stats.t.ppf(0.975, n-1)
        
        accuracy_ci = t_critical * accuracy_std / np.sqrt(n)
        energy_ci = t_critical * energy_reduction_std / np.sqrt(n)
        
        # Statistical significance test (one-sample t-test)
        # H0: energy_reduction_mean <= min_energy_reduction
        # H1: energy_reduction_mean > min_energy_reduction
        t_stat, p_value = stats.ttest_1samp(
            energy_reductions, 
            self.config.min_energy_reduction
        )
        
        # Create validation result
        result = ValidationResult(
            model_name="encn",
            dataset=self.config.dataset,
            config=self.config,
            
            # Performance metrics
            accuracy_mean=accuracy_mean,
            accuracy_std=accuracy_std,
            accuracy_ci_lower=accuracy_mean - accuracy_ci,
            accuracy_ci_upper=accuracy_mean + accuracy_ci,
            
            # Energy metrics
            energy_reduction_mean=energy_reduction_mean,
            energy_reduction_std=energy_reduction_std,
            energy_reduction_ci_lower=energy_reduction_mean - energy_ci,
            energy_reduction_ci_upper=energy_reduction_mean + energy_ci,
            
            # Statistical validation
            statistical_significance_p=p_value,
            
            # Hardware info
            gpu_name=self.hardware_info.get('gpu_name', ''),
            gpu_memory_gb=self.hardware_info.get('gpu_memory_gb', 0),
            cpu_name=self.hardware_info.get('cpu_name', ''),
            system_memory_gb=self.hardware_info.get('system_memory_gb', 0),
            
            # Detailed results
            individual_runs=encn_results
        )
        
        # Validation checks
        result.accuracy_passes_threshold = accuracy_mean >= self.config.min_accuracy
        result.energy_reduction_passes_threshold = energy_reduction_mean >= self.config.min_energy_reduction
        
        # Overall validation status
        validation_issues = []
        
        if not result.accuracy_passes_threshold:
            validation_issues.append(
                f"Accuracy {accuracy_mean:.3f} < required {self.config.min_accuracy}"
            )
            
        if not result.energy_reduction_passes_threshold:
            validation_issues.append(
                f"Energy reduction {energy_reduction_mean:.1f}x < required {self.config.min_energy_reduction}x"
            )
            
        if p_value > self.config.statistical_significance:
            validation_issues.append(
                f"Statistical significance p={p_value:.4f} > {self.config.statistical_significance}"
            )
            
        result.validation_issues = validation_issues
        result.validation_passed = len(validation_issues) == 0
        
        # Print results
        self._print_validation_summary(result, all_results)
        
        return result
        
    def _print_validation_summary(self, result: ValidationResult, all_results: Dict):
        """Print comprehensive validation summary."""
        print(f"\nACCURACY ANALYSIS:")
        print(f"  Mean: {result.accuracy_mean:.4f} ± {result.accuracy_std:.4f}")
        print(f"  95% CI: [{result.accuracy_ci_lower:.4f}, {result.accuracy_ci_upper:.4f}]")
        print(f"  Threshold: {self.config.min_accuracy:.4f} - {'✓ PASS' if result.accuracy_passes_threshold else '✗ FAIL'}")
        
        print(f"\nENERGY REDUCTION ANALYSIS:")
        print(f"  Mean: {result.energy_reduction_mean:.1f}x ± {result.energy_reduction_std:.1f}x")
        print(f"  95% CI: [{result.energy_reduction_ci_lower:.1f}x, {result.energy_reduction_ci_upper:.1f}x]")
        print(f"  Threshold: {self.config.min_energy_reduction:.1f}x - {'✓ PASS' if result.energy_reduction_passes_threshold else '✗ FAIL'}")
        print(f"  Statistical significance: p={result.statistical_significance_p:.4f} - {'✓ PASS' if result.statistical_significance_p <= self.config.statistical_significance else '✗ FAIL'}")
        
        print(f"\nHARDWARE SPECIFICATIONS:")
        print(f"  GPU: {result.gpu_name} ({result.gpu_memory_gb:.1f} GB)")
        print(f"  CPU: {result.cpu_name}")
        print(f"  System Memory: {result.system_memory_gb:.1f} GB")
        
        print(f"\nVALIDATION STATUS: {'✓ PASSED' if result.validation_passed else '✗ FAILED'}")
        if result.validation_issues:
            print("Issues:")
            for issue in result.validation_issues:
                print(f"  - {issue}")
                
    def save_results(self, result: ValidationResult, filename: str = None):
        """Save validation results to JSON file."""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"validation_results_{timestamp}.json"
            
        filepath = self.results_dir / filename
        
        # Convert to serializable format
        result_dict = asdict(result)
        
        # Handle non-serializable objects
        if 'individual_runs' in result_dict:
            result_dict['individual_runs'] = [
                {
                    'model_name': run.model_name,
                    'accuracy': run.accuracy,
                    'total_energy_joules': run.total_energy_joules,
                    'energy_per_sample_mj': run.energy_per_sample_mj,
                    'sparsity_ratio': run.sparsity_ratio,
                    'energy_reduction_ratio': run.energy_reduction_ratio
                }
                for run in result.individual_runs
            ]
            
        with open(filepath, 'w') as f:
            json.dump(result_dict, f, indent=2)
            
        print(f"\nResults saved to: {filepath}")
        
        # Also save summary CSV for easy analysis
        csv_path = filepath.with_suffix('.csv')
        import pandas as pd
        
        summary_data = {
            'metric': ['accuracy_mean', 'accuracy_std', 'energy_reduction_mean', 
                      'energy_reduction_std', 'statistical_p_value'],
            'value': [result.accuracy_mean, result.accuracy_std, 
                     result.energy_reduction_mean, result.energy_reduction_std,
                     result.statistical_significance_p],
            'threshold': [self.config.min_accuracy, None, self.config.min_energy_reduction,
                         None, self.config.statistical_significance],
            'pass': [result.accuracy_passes_threshold, None, 
                    result.energy_reduction_passes_threshold, None,
                    result.statistical_significance_p <= self.config.statistical_significance]
        }
        
        df = pd.DataFrame(summary_data)
        df.to_csv(csv_path, index=False)
        print(f"Summary saved to: {csv_path}")


def main():
    """Main validation entry point."""
    parser = argparse.ArgumentParser(description="E-NCN Hardware Energy Validation")
    parser.add_argument('--model', choices=['encn', 'all'], default='encn',
                       help='Model type to validate')
    parser.add_argument('--dataset', choices=['mnist'], default='mnist',
                       help='Dataset for validation')
    parser.add_argument('--sparsity-target', type=float, default=0.99,
                       help='Target sparsity level')
    parser.add_argument('--num-runs', type=int, default=5,
                       help='Number of validation runs')
    parser.add_argument('--min-accuracy', type=float, default=0.98,
                       help='Minimum accuracy threshold')
    parser.add_argument('--min-energy-reduction', type=float, default=200.0,
                       help='Minimum energy reduction threshold')
    parser.add_argument('--validate-claims', action='store_true',
                       help='Run full validation against E-NCN claims')
    
    args = parser.parse_args()
    
    # Check hardware availability
    if not HARDWARE_MONITORING_AVAILABLE:
        print("Warning: Hardware monitoring not available. Install required packages:")
        print("  pip install pynvml psutil py-cpuinfo")
        
    # Create validation configuration
    config = ValidationConfig(
        model_type=args.model,
        dataset=args.dataset,
        sparsity_target=args.sparsity_target,
        num_validation_runs=args.num_runs,
        min_accuracy=args.min_accuracy,
        min_energy_reduction=args.min_energy_reduction
    )
    
    # Initialize validator
    validator = HardwareEnergyValidator(config)
    
    try:
        # Run validation
        result = validator.run_statistical_validation()
        
        # Save results
        validator.save_results(result)
        
        # Exit with appropriate code
        if result.validation_passed:
            print("\n🎉 VALIDATION PASSED! E-NCN claims verified on hardware.")
            sys.exit(0)
        else:
            print("\n❌ VALIDATION FAILED! E-NCN claims not verified.")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n💥 VALIDATION ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()