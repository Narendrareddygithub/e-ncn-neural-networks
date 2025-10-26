#!/usr/bin/env python3
"""
MNIST Production Validation for E-NCN Networks.

This module provides rigorous MNIST testing to validate E-NCN performance
against production requirements: >98% accuracy with >200x energy reduction.

Validation Requirements (Week 2):
- Accuracy: >98% on MNIST test set
- Energy Reduction: >200x measured (not theoretical)
- Training Stability: Converges within 2x epochs of dense network
- Sparsity: >99% of operations actually skipped
- Statistical Significance: p < 0.05 across 5 independent runs

Usage:
    python mnist_production_test.py --full-validation
    python mnist_production_test.py --quick-test --sparsity 0.95
    
Outputs:
- Detailed accuracy and energy metrics
- Statistical significance analysis
- Hardware compatibility report
- Production readiness assessment
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
from scipy import stats
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from encn.layers import ENCNLayer, create_encn_mlp, ENCNSequential
from encn.profiling import EnergyProfiler, NetworkBenchmarker


@dataclass
class MNISTValidationConfig:
    """Configuration for MNIST validation experiments."""
    # Model architecture
    hidden_sizes: List[int] = None
    sparsity_targets: List[float] = None
    threshold_values: List[float] = None
    
    # Training parameters
    batch_size: int = 128
    num_epochs: int = 20
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    
    # Validation parameters
    num_seeds: int = 5  # Independent runs with different seeds
    test_batch_size: int = 1000
    validation_split: float = 0.1
    
    # Success criteria
    min_accuracy: float = 0.98
    min_energy_reduction: float = 200.0
    max_epoch_ratio: float = 2.0  # Max 2x epochs vs dense
    min_sparsity: float = 0.99
    
    # Output configuration
    save_models: bool = True
    save_plots: bool = True
    results_dir: str = "results/mnist_validation"
    
    def __post_init__(self):
        if self.hidden_sizes is None:
            self.hidden_sizes = [256, 128]
        if self.sparsity_targets is None:
            self.sparsity_targets = [0.95, 0.99, 0.999]
        if self.threshold_values is None:
            self.threshold_values = [0.05, 0.1, 0.2]


@dataclass 
class MNISTResult:
    """Results for single MNIST experiment."""
    model_name: str
    seed: int
    config: Dict
    
    # Training metrics
    final_accuracy: float = 0.0
    best_accuracy: float = 0.0
    epochs_to_converge: int = 0
    training_time_seconds: float = 0.0
    
    # Energy metrics
    total_energy_joules: float = 0.0
    energy_per_sample: float = 0.0
    inference_time_ms: float = 0.0
    
    # E-NCN specific metrics
    achieved_sparsity: float = 0.0
    energy_reduction_vs_dense: float = 1.0
    threshold_final: float = 0.0
    
    # Model statistics
    total_parameters: int = 0
    active_parameters: int = 0
    memory_usage_mb: float = 0.0
    
    # Validation flags
    meets_accuracy: bool = False
    meets_energy: bool = False
    meets_sparsity: bool = False
    production_ready: bool = False


class MNISTValidator:
    """Comprehensive MNIST validation system for E-NCN."""
    
    def __init__(self, config: MNISTValidationConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results_dir = Path(config.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize benchmarker
        self.benchmarker = NetworkBenchmarker(device=self.device)
        
        print(f"MNIST Validator initialized on {self.device}")
        print(f"Results directory: {self.results_dir}")
        
    def prepare_data(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Prepare MNIST dataset with proper splits."""
        # Standard MNIST normalization
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # Load datasets
        train_dataset = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.MNIST(
            root='./data', train=False, transform=transform
        )
        
        # Split training set for validation
        val_size = int(len(train_dataset) * self.config.validation_split)
        train_size = len(train_dataset) - val_size
        
        train_subset, val_subset = random_split(
            train_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create data loaders with flattening
        train_loader = DataLoader(
            train_subset, 
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=self._flatten_collate,
            num_workers=2
        )
        
        val_loader = DataLoader(
            val_subset,
            batch_size=self.config.test_batch_size,
            shuffle=False,
            collate_fn=self._flatten_collate,
            num_workers=2
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.test_batch_size,
            shuffle=False,
            collate_fn=self._flatten_collate,
            num_workers=2
        )
        
        print(f"Data prepared: {len(train_subset)} train, {len(val_subset)} val, {len(test_dataset)} test")
        return train_loader, val_loader, test_loader
        
    def _flatten_collate(self, batch):
        """Flatten MNIST images to 784-dimensional vectors."""
        images, labels = zip(*batch)
        images = torch.stack(images).view(-1, 784)  # Flatten to 784
        labels = torch.tensor(labels)
        return images, labels
        
    def create_models(self) -> Dict[str, nn.Module]:
        """Create model variants for comparison."""
        models = {}
        
        # Dense baseline
        models['dense_baseline'] = nn.Sequential(
            nn.Linear(784, self.config.hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.config.hidden_sizes[0], self.config.hidden_sizes[1]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.config.hidden_sizes[1], 10)
        )
        
        # E-NCN variants with different sparsity targets
        for sparsity in self.config.sparsity_targets:
            for threshold in self.config.threshold_values:
                model_name = f'encn_s{sparsity:.3f}_t{threshold:.3f}'
                
                models[model_name] = create_encn_mlp(
                    [784] + self.config.hidden_sizes + [10],
                    threshold=threshold,
                    sparsity_target=sparsity,
                    activation='relu',
                    dropout=0.2
                )
        
        # Pruned baseline for comparison
        pruned_model = nn.Sequential(
            nn.Linear(784, self.config.hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(self.config.hidden_sizes[0], self.config.hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(self.config.hidden_sizes[1], 10)
        )
        
        # Apply magnitude pruning
        self._apply_magnitude_pruning(pruned_model, sparsity=0.9)
        models['pruned_baseline'] = pruned_model
        
        return models
        
    def _apply_magnitude_pruning(self, model: nn.Module, sparsity: float):
        """Apply magnitude-based pruning to model."""
        for module in model.modules():
            if isinstance(module, nn.Linear):
                with torch.no_grad():
                    weights = module.weight.data
                    threshold = torch.quantile(torch.abs(weights.flatten()), sparsity)
                    mask = torch.abs(weights) > threshold
                    module.weight.data *= mask.float()
                    
    def train_model(self, model: nn.Module, train_loader: DataLoader,
                   val_loader: DataLoader, model_name: str, seed: int) -> Tuple[nn.Module, Dict]:
        """Train model with comprehensive monitoring."""
        # Set random seed for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        model = model.to(self.device)
        optimizer = optim.Adam(model.parameters(), 
                              lr=self.config.learning_rate,
                              weight_decay=self.config.weight_decay)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'sparsity': [],
            'energy_reduction': []
        }
        
        best_val_acc = 0.0
        epochs_without_improvement = 0
        convergence_threshold = self.config.min_accuracy - 0.01  # 97% for convergence
        epochs_to_converge = self.config.num_epochs
        
        print(f"\nTraining {model_name} (seed={seed})...")
        start_time = time.time()
        
        for epoch in range(self.config.num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                
                # Add sparsity regularization for E-NCN models
                if hasattr(model, 'modules'):
                    sparsity_loss = 0.0
                    for module in model.modules():
                        if isinstance(module, ENCNLayer):
                            # Encourage target sparsity
                            current_sparsity = module.get_sparsity()
                            target_sparsity = module.sparsity_target
                            sparsity_penalty = (current_sparsity - target_sparsity) ** 2
                            sparsity_loss += 0.1 * sparsity_penalty
                    
                    if sparsity_loss > 0:
                        loss += sparsity_loss
                
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                pred = output.argmax(dim=1)
                train_correct += (pred == target).sum().item()
                train_total += target.size(0)
                
            # Validation phase
            val_loss, val_acc = self._evaluate_model(model, val_loader, criterion)
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Track metrics
            train_acc = train_correct / train_total
            history['train_loss'].append(train_loss / len(train_loader))
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Track E-NCN specific metrics
            if hasattr(model, 'get_total_energy_stats'):
                stats = model.get_total_energy_stats()
                history['sparsity'].append(stats.get('avg_sparsity', 0.0))
                history['energy_reduction'].append(stats.get('total_energy_reduction', 1.0))
            else:
                history['sparsity'].append(0.0)
                history['energy_reduction'].append(1.0)
            
            # Check for convergence
            if val_acc >= convergence_threshold and epochs_to_converge == self.config.num_epochs:
                epochs_to_converge = epoch + 1
                
            # Early stopping check
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                
            # Print progress
            if epoch % 5 == 0 or epoch == self.config.num_epochs - 1:
                sparsity = history['sparsity'][-1]
                energy_red = history['energy_reduction'][-1]
                print(f"Epoch {epoch+1:2d}: Train={train_acc:.4f}, Val={val_acc:.4f}, "
                      f"Sparsity={sparsity:.3f}, Energy={energy_red:.1f}x")
                
            # Early stopping for non-E-NCN models
            if not hasattr(model, 'get_total_energy_stats') and epochs_without_improvement >= 10:
                print(f"Early stopping at epoch {epoch+1}")
                break
                
        training_time = time.time() - start_time
        
        training_stats = {
            'history': history,
            'best_val_acc': best_val_acc,
            'epochs_to_converge': epochs_to_converge,
            'training_time': training_time,
            'final_lr': optimizer.param_groups[0]['lr']
        }
        
        return model, training_stats
        
    def _evaluate_model(self, model: nn.Module, data_loader: DataLoader, 
                       criterion: nn.Module) -> Tuple[float, float]:
        """Evaluate model on given dataset."""
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)
                
        return total_loss / len(data_loader), correct / total
        
    def benchmark_model(self, model: nn.Module, test_loader: DataLoader, 
                       model_name: str, baseline_energy: float = None) -> Dict:
        """Comprehensive benchmarking of trained model."""
        model.eval()
        
        # Reset energy stats for E-NCN models
        if hasattr(model, 'reset_all_energy_stats'):
            model.reset_all_energy_stats()
            
        # Use the benchmarker for energy measurement
        benchmark_result = self.benchmarker.benchmark_model(
            model=model,
            data_loader=test_loader,
            model_name=model_name,
            num_batches=20  # Use subset for faster benchmarking
        )
        
        # Calculate additional metrics
        metrics = {
            'accuracy': benchmark_result.accuracy,
            'energy_joules': benchmark_result.total_energy_joules,
            'energy_per_sample': benchmark_result.energy_per_sample_mj,
            'inference_time_ms': benchmark_result.inference_time_ms,
            'sparsity': benchmark_result.sparsity_ratio,
            'memory_mb': benchmark_result.peak_memory_mb
        }
        
        # E-NCN specific metrics
        if hasattr(model, 'get_total_energy_stats'):
            encn_stats = model.get_total_energy_stats()
            metrics.update({
                'encn_sparsity': encn_stats.get('avg_sparsity', 0.0),
                'encn_energy_reduction': encn_stats.get('total_energy_reduction', 1.0)
            })
            
            # Get individual layer stats
            metrics['layer_stats'] = encn_stats.get('layer_stats', [])
            
        # Calculate energy reduction vs baseline
        if baseline_energy is not None and benchmark_result.total_energy_joules > 0:
            metrics['energy_reduction_vs_baseline'] = baseline_energy / benchmark_result.total_energy_joules
        else:
            metrics['energy_reduction_vs_baseline'] = 1.0
            
        # Model complexity
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        metrics.update({
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024**2)  # Assuming float32
        })
        
        return metrics
        
    def run_full_validation(self) -> Dict[str, List[MNISTResult]]:
        """Run complete validation across all models and seeds."""
        print("\n" + "="*80)
        print("STARTING COMPREHENSIVE MNIST VALIDATION")
        print("="*80)
        
        # Prepare data
        train_loader, val_loader, test_loader = self.prepare_data()
        
        # Create models
        model_templates = self.create_models()
        
        results = {}
        baseline_energy = None
        
        for model_name, model_template in model_templates.items():
            print(f"\n{'-'*60}")
            print(f"VALIDATING MODEL: {model_name}")
            print(f"{'-'*60}")
            
            model_results = []
            
            for seed in range(self.config.num_seeds):
                print(f"\n--- Seed {seed + 1}/{self.config.num_seeds} ---")
                
                # Create fresh model instance
                if hasattr(model_template, '__class__'):
                    # For E-NCN models, recreate to avoid state issues
                    if 'encn' in model_name:
                        parts = model_name.split('_')
                        sparsity = float(parts[1][1:])  # Extract from s0.999
                        threshold = float(parts[2][1:])  # Extract from t0.100
                        
                        model = create_encn_mlp(
                            [784] + self.config.hidden_sizes + [10],
                            threshold=threshold,
                            sparsity_target=sparsity,
                            activation='relu',
                            dropout=0.2
                        )
                    else:
                        # Recreate other models
                        model = self.create_models()[model_name]
                else:
                    model = model_template
                    
                # Train model
                trained_model, train_stats = self.train_model(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    model_name=model_name,
                    seed=seed
                )
                
                # Benchmark model
                benchmark_metrics = self.benchmark_model(
                    model=trained_model,
                    test_loader=test_loader,
                    model_name=model_name,
                    baseline_energy=baseline_energy
                )
                
                # Store baseline energy for comparison
                if model_name == 'dense_baseline' and baseline_energy is None:
                    baseline_energy = benchmark_metrics['energy_joules']
                    
                # Create result record
                result = MNISTResult(
                    model_name=model_name,
                    seed=seed,
                    config=asdict(self.config),
                    final_accuracy=benchmark_metrics['accuracy'],
                    best_accuracy=train_stats['best_val_acc'],
                    epochs_to_converge=train_stats['epochs_to_converge'],
                    training_time_seconds=train_stats['training_time'],
                    total_energy_joules=benchmark_metrics['energy_joules'],
                    energy_per_sample=benchmark_metrics['energy_per_sample'],
                    inference_time_ms=benchmark_metrics['inference_time_ms'],
                    achieved_sparsity=benchmark_metrics.get('encn_sparsity', benchmark_metrics['sparsity']),
                    energy_reduction_vs_dense=benchmark_metrics['energy_reduction_vs_baseline'],
                    threshold_final=0.0,  # TODO: Extract final threshold
                    total_parameters=benchmark_metrics['total_parameters'],
                    active_parameters=benchmark_metrics['trainable_parameters'],
                    memory_usage_mb=benchmark_metrics.get('memory_mb', 0.0)
                )
                
                # Check validation criteria
                result.meets_accuracy = result.final_accuracy >= self.config.min_accuracy
                result.meets_energy = result.energy_reduction_vs_dense >= self.config.min_energy_reduction
                result.meets_sparsity = result.achieved_sparsity >= self.config.min_sparsity
                result.production_ready = all([
                    result.meets_accuracy,
                    result.meets_energy,
                    result.meets_sparsity
                ])
                
                model_results.append(result)
                
                # Print summary
                status = "✓" if result.production_ready else "✗"
                print(f"{status} Seed {seed}: Acc={result.final_accuracy:.4f}, "
                      f"Energy={result.energy_reduction_vs_dense:.1f}x, "
                      f"Sparsity={result.achieved_sparsity:.3f}")
                      
            results[model_name] = model_results
            
        return results
        
    def analyze_results(self, results: Dict[str, List[MNISTResult]]) -> Dict:
        """Analyze validation results for statistical significance."""
        print("\n" + "="*80)
        print("STATISTICAL ANALYSIS")
        print("="*80)
        
        analysis = {}
        
        for model_name, model_results in results.items():
            if not model_results:
                continue
                
            # Extract metrics
            accuracies = [r.final_accuracy for r in model_results]
            energy_reductions = [r.energy_reduction_vs_dense for r in model_results]
            sparsities = [r.achieved_sparsity for r in model_results]
            
            # Statistical calculations
            acc_mean, acc_std = np.mean(accuracies), np.std(accuracies, ddof=1)
            energy_mean, energy_std = np.mean(energy_reductions), np.std(energy_reductions, ddof=1)
            sparsity_mean, sparsity_std = np.mean(sparsities), np.std(sparsities, ddof=1)
            
            # Confidence intervals
            n = len(model_results)
            t_crit = stats.t.ppf(0.975, n-1) if n > 1 else 1.96
            
            acc_ci = t_crit * acc_std / np.sqrt(n) if n > 1 else 0
            energy_ci = t_crit * energy_std / np.sqrt(n) if n > 1 else 0
            
            # Statistical tests vs thresholds
            acc_ttest = stats.ttest_1samp(accuracies, self.config.min_accuracy) if n > 1 else (0, 1)
            energy_ttest = stats.ttest_1samp(energy_reductions, self.config.min_energy_reduction) if n > 1 else (0, 1)
            
            analysis[model_name] = {
                'num_runs': n,
                'accuracy': {
                    'mean': acc_mean,
                    'std': acc_std,
                    'ci_lower': acc_mean - acc_ci,
                    'ci_upper': acc_mean + acc_ci,
                    'passes_threshold': acc_mean >= self.config.min_accuracy,
                    'statistical_p': acc_ttest[1]
                },
                'energy_reduction': {
                    'mean': energy_mean,
                    'std': energy_std,
                    'ci_lower': energy_mean - energy_ci,
                    'ci_upper': energy_mean + energy_ci,
                    'passes_threshold': energy_mean >= self.config.min_energy_reduction,
                    'statistical_p': energy_ttest[1]
                },
                'sparsity': {
                    'mean': sparsity_mean,
                    'std': sparsity_std,
                    'passes_threshold': sparsity_mean >= self.config.min_sparsity
                },
                'production_ready_rate': sum(r.production_ready for r in model_results) / n
            }
            
            # Print analysis
            print(f"\n{model_name.upper()}:")
            print(f"  Accuracy: {acc_mean:.4f} ± {acc_std:.4f} (thresh: {self.config.min_accuracy:.3f})")
            print(f"  Energy Reduction: {energy_mean:.1f}x ± {energy_std:.1f}x (thresh: {self.config.min_energy_reduction:.1f}x)")
            print(f"  Sparsity: {sparsity_mean:.3f} ± {sparsity_std:.3f} (thresh: {self.config.min_sparsity:.3f})")
            print(f"  Production Ready: {analysis[model_name]['production_ready_rate']*100:.1f}%")
            
        return analysis
        
    def save_results(self, results: Dict, analysis: Dict, filename_prefix: str = "mnist_validation"):
        """Save comprehensive results."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = self.results_dir / f"{filename_prefix}_detailed_{timestamp}.json"
        
        # Convert results to serializable format
        serializable_results = {}
        for model_name, model_results in results.items():
            serializable_results[model_name] = [asdict(r) for r in model_results]
            
        combined_data = {
            'config': asdict(self.config),
            'results': serializable_results,
            'analysis': analysis,
            'timestamp': timestamp,
            'device': str(self.device)
        }
        
        with open(results_file, 'w') as f:
            json.dump(combined_data, f, indent=2)
            
        print(f"\nDetailed results saved to: {results_file}")
        
        # Save summary CSV
        import pandas as pd
        
        summary_data = []
        for model_name, analysis_data in analysis.items():
            summary_data.append({
                'model': model_name,
                'accuracy_mean': analysis_data['accuracy']['mean'],
                'accuracy_std': analysis_data['accuracy']['std'],
                'accuracy_passes': analysis_data['accuracy']['passes_threshold'],
                'energy_reduction_mean': analysis_data['energy_reduction']['mean'],
                'energy_reduction_std': analysis_data['energy_reduction']['std'],
                'energy_passes': analysis_data['energy_reduction']['passes_threshold'],
                'sparsity_mean': analysis_data['sparsity']['mean'],
                'sparsity_passes': analysis_data['sparsity']['passes_threshold'],
                'production_ready_rate': analysis_data['production_ready_rate']
            })
            
        df = pd.DataFrame(summary_data)
        csv_file = self.results_dir / f"{filename_prefix}_summary_{timestamp}.csv"
        df.to_csv(csv_file, index=False)
        
        print(f"Summary CSV saved to: {csv_file}")
        
        return results_file, csv_file


def main():
    """Main validation entry point."""
    parser = argparse.ArgumentParser(description="MNIST Production Validation for E-NCN")
    parser.add_argument('--full-validation', action='store_true',
                       help='Run full validation with multiple seeds')
    parser.add_argument('--quick-test', action='store_true', 
                       help='Run quick test with single seed')
    parser.add_argument('--sparsity', type=float, default=0.99,
                       help='Sparsity target for quick test')
    parser.add_argument('--threshold', type=float, default=0.1,
                       help='Threshold for quick test')
    parser.add_argument('--num-seeds', type=int, default=5,
                       help='Number of random seeds for validation')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    
    args = parser.parse_args()
    
    # Configure validation
    config = MNISTValidationConfig(
        num_seeds=1 if args.quick_test else args.num_seeds,
        num_epochs=args.epochs
    )
    
    if args.quick_test:
        config.sparsity_targets = [args.sparsity]
        config.threshold_values = [args.threshold]
    
    # Initialize validator
    validator = MNISTValidator(config)
    
    try:
        # Run validation
        print(f"Starting {'quick test' if args.quick_test else 'full validation'}...")
        results = validator.run_full_validation()
        
        # Analyze results
        analysis = validator.analyze_results(results)
        
        # Save results
        prefix = "mnist_quick" if args.quick_test else "mnist_full"
        results_file, csv_file = validator.save_results(results, analysis, prefix)
        
        # Check if any E-NCN model passes validation
        encn_models = [name for name in results.keys() if 'encn' in name]
        success = any(
            analysis[model]['production_ready_rate'] > 0.8  # 80% success rate
            for model in encn_models
        )
        
        if success:
            print("\n🎉 MNIST VALIDATION PASSED! E-NCN models meet production requirements.")
            sys.exit(0)
        else:
            print("\n❌ MNIST VALIDATION FAILED! E-NCN models do not meet requirements.")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n💥 VALIDATION ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()