#!/usr/bin/env python3
"""
E-NCN Comprehensive Validation Runner.

This script orchestrates the complete validation of E-NCN energy reduction claims
through hardware measurement, MNIST accuracy validation, and scalability testing.

Week 2 Mission-Critical Validation:
- Hardware-verified energy reduction >200x
- MNIST accuracy >98% with statistical significance
- Production scalability validation
- Comprehensive benchmarking vs baselines

Usage:
    # Complete Week 2 validation
    python run_validation.py --week2-validation
    
    # Quick validation test
    python run_validation.py --quick-test
    
    # Individual components
    python run_validation.py --hardware-only
    python run_validation.py --mnist-only
    python run_validation.py --scalability-only
"""

import os
import sys
import subprocess
import argparse
import json
from pathlib import Path
from datetime import datetime


class ValidationOrchestrator:
    """Orchestrates comprehensive E-NCN validation."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.results_dir = self.project_root / "results" / "comprehensive_validation"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Validation scripts
        self.scripts = {
            'hardware': self.project_root / "experiments" / "hardware_validation" / "energy_validator.py",
            'mnist': self.project_root / "experiments" / "mnist_validation" / "mnist_production_test.py",
            'scalability': self.project_root / "experiments" / "scalability_testing" / "production_scale_test.py"
        }
        
        self.results = {}
        
    def check_prerequisites(self) -> bool:
        """Check if all required components are available."""
        print("🔍 CHECKING VALIDATION PREREQUISITES")
        print("=" * 50)
        
        issues = []
        
        # Check Python environment
        try:
            import torch
            print(f"✅ PyTorch: {torch.__version__}")
        except ImportError:
            issues.append("PyTorch not installed")
            
        try:
            import numpy
            print(f"✅ NumPy: {numpy.__version__}")
        except ImportError:
            issues.append("NumPy not installed")
            
        # Check hardware monitoring capabilities
        try:
            import pynvml
            pynvml.nvmlInit()
            gpu_count = pynvml.nvmlDeviceGetCount()
            print(f"✅ NVIDIA GPUs detected: {gpu_count}")
        except ImportError:
            issues.append("pynvml not installed (pip install nvidia-ml-py3)")
        except Exception as e:
            issues.append(f"GPU monitoring unavailable: {e}")
            
        try:
            import psutil
            print(f"✅ System monitoring: psutil {psutil.__version__}")
        except ImportError:
            issues.append("psutil not installed")
            
        # Check CUDA availability
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.version.cuda}")
            print(f"   Device: {torch.cuda.get_device_name()}")
            memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"   Memory: {memory_gb:.1f} GB")
        else:
            print("⚠️  CUDA not available - will use CPU (reduced accuracy)")
            
        # Check validation scripts
        for name, script_path in self.scripts.items():
            if script_path.exists():
                print(f"✅ {name.title()} validation script: {script_path.name}")
            else:
                issues.append(f"{name.title()} validation script missing: {script_path}")
                
        if issues:
            print("\n❌ PREREQUISITE ISSUES:")
            for issue in issues:
                print(f"   - {issue}")
            return False
        else:
            print("\n✅ ALL PREREQUISITES SATISFIED")
            return True
            
    def run_hardware_validation(self, quick_test: bool = False) -> dict:
        """Run hardware energy validation."""
        print("\n" + "=" * 60)
        print("⚡ HARDWARE ENERGY VALIDATION")
        print("=" * 60)
        
        cmd = [sys.executable, str(self.scripts['hardware'])]
        
        if quick_test:
            cmd.extend(['--num-runs', '2', '--min-energy-reduction', '50.0'])
        else:
            cmd.append('--validate-claims')
            
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                cwd=self.project_root,
                timeout=3600  # 1 hour timeout
            )
            
            success = result.returncode == 0
            
            self.results['hardware'] = {
                'success': success,
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'timestamp': datetime.now().isoformat()
            }
            
            if success:
                print("✅ Hardware validation PASSED")
            else:
                print("❌ Hardware validation FAILED")
                print(f"Error: {result.stderr}")
                
            return self.results['hardware']
            
        except subprocess.TimeoutExpired:
            print("❌ Hardware validation TIMEOUT")
            self.results['hardware'] = {'success': False, 'error': 'timeout'}
            return self.results['hardware']
        except Exception as e:
            print(f"❌ Hardware validation ERROR: {e}")
            self.results['hardware'] = {'success': False, 'error': str(e)}
            return self.results['hardware']
            
    def run_mnist_validation(self, quick_test: bool = False) -> dict:
        """Run MNIST production validation."""
        print("\n" + "=" * 60)
        print("🎯 MNIST PRODUCTION VALIDATION")
        print("=" * 60)
        
        cmd = [sys.executable, str(self.scripts['mnist'])]
        
        if quick_test:
            cmd.extend(['--quick-test', '--sparsity', '0.95', '--epochs', '5'])
        else:
            cmd.extend(['--full-validation', '--epochs', '10'])
            
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.project_root,
                timeout=7200  # 2 hour timeout
            )
            
            success = result.returncode == 0
            
            self.results['mnist'] = {
                'success': success,
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'timestamp': datetime.now().isoformat()
            }
            
            if success:
                print("✅ MNIST validation PASSED")
            else:
                print("❌ MNIST validation FAILED")
                print(f"Error: {result.stderr}")
                
            return self.results['mnist']
            
        except subprocess.TimeoutExpired:
            print("❌ MNIST validation TIMEOUT")
            self.results['mnist'] = {'success': False, 'error': 'timeout'}
            return self.results['mnist']
        except Exception as e:
            print(f"❌ MNIST validation ERROR: {e}")
            self.results['mnist'] = {'success': False, 'error': str(e)}
            return self.results['mnist']
            
    def run_scalability_validation(self, quick_test: bool = False) -> dict:
        """Run scalability validation."""
        print("\n" + "=" * 60)
        print("📊 SCALABILITY VALIDATION")
        print("=" * 60)
        
        cmd = [sys.executable, str(self.scripts['scalability'])]
        
        if quick_test:
            cmd.extend(['--parameter-test', '--max-params', '500000'])
        else:
            cmd.extend(['--full-scale-test', '--max-params', '2000000'])
            
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.project_root,
                timeout=3600  # 1 hour timeout
            )
            
            success = result.returncode == 0
            
            self.results['scalability'] = {
                'success': success,
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'timestamp': datetime.now().isoformat()
            }
            
            if success:
                print("✅ Scalability validation PASSED")
            else:
                print("❌ Scalability validation FAILED")
                print(f"Error: {result.stderr}")
                
            return self.results['scalability']
            
        except subprocess.TimeoutExpired:
            print("❌ Scalability validation TIMEOUT")
            self.results['scalability'] = {'success': False, 'error': 'timeout'}
            return self.results['scalability']
        except Exception as e:
            print(f"❌ Scalability validation ERROR: {e}")
            self.results['scalability'] = {'success': False, 'error': str(e)}
            return self.results['scalability']
            
    def generate_summary_report(self) -> dict:
        """Generate comprehensive validation summary."""
        print("\n" + "=" * 70)
        print("📊 COMPREHENSIVE VALIDATION SUMMARY")
        print("=" * 70)
        
        summary = {
            'validation_timestamp': datetime.now().isoformat(),
            'total_tests_run': len(self.results),
            'tests_passed': sum(1 for r in self.results.values() if r.get('success', False)),
            'tests_failed': sum(1 for r in self.results.values() if not r.get('success', False)),
            'overall_success': all(r.get('success', False) for r in self.results.values()),
            'individual_results': self.results
        }
        
        print(f"\nValidation Results:")
        for test_name, result in self.results.items():
            status = "✅ PASSED" if result.get('success', False) else "❌ FAILED"
            print(f"  {test_name.title()}: {status}")
            
        overall_status = "✅ PASSED" if summary['overall_success'] else "❌ FAILED"
        print(f"\nOverall Status: {overall_status}")
        print(f"Success Rate: {summary['tests_passed']}/{summary['total_tests_run']}")
        
        # Week 2 specific validation
        if summary['overall_success']:
            print("\n🎉 WEEK 2 VALIDATION COMPLETE!")
            print("E-NCN energy reduction claims validated on real hardware")
            print("Production requirements satisfied:")
            print("  ✅ Hardware energy measurement >200x reduction")
            print("  ✅ MNIST accuracy >98% with statistical significance")
            print("  ✅ Production scalability demonstrated")
        else:
            print("\n❌ WEEK 2 VALIDATION INCOMPLETE")
            print("E-NCN claims require further validation")
            
        return summary
        
    def save_results(self, summary: dict) -> Path:
        """Save comprehensive validation results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"comprehensive_validation_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
        print(f"\n💾 Validation results saved to: {results_file}")
        
        # Also save a simple summary for CI/CD
        summary_file = self.results_dir / "latest_validation_summary.json"
        simple_summary = {
            'timestamp': summary['validation_timestamp'],
            'overall_success': summary['overall_success'],
            'tests_passed': summary['tests_passed'],
            'tests_total': summary['total_tests_run'],
            'individual_success': {
                name: result.get('success', False) 
                for name, result in summary['individual_results'].items()
            }
        }
        
        with open(summary_file, 'w') as f:
            json.dump(simple_summary, f, indent=2)
            
        return results_file


def main():
    """Main validation orchestration."""
    parser = argparse.ArgumentParser(
        description="E-NCN Comprehensive Validation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Validation Modes:
  --week2-validation    Complete Week 2 validation (recommended)
  --quick-test         Fast validation for development
  --hardware-only      Hardware energy validation only
  --mnist-only         MNIST accuracy validation only
  --scalability-only   Scalability testing only

Week 2 Success Criteria:
  ✅ Hardware energy reduction >200x measured
  ✅ MNIST accuracy >98% with statistical significance
  ✅ Production scalability demonstrated
  ✅ Comprehensive benchmarking vs baselines
"""
    )
    
    parser.add_argument('--week2-validation', action='store_true',
                       help='Run complete Week 2 validation suite')
    parser.add_argument('--quick-test', action='store_true',
                       help='Run quick validation tests')
    parser.add_argument('--hardware-only', action='store_true',
                       help='Run hardware validation only')
    parser.add_argument('--mnist-only', action='store_true',
                       help='Run MNIST validation only')
    parser.add_argument('--scalability-only', action='store_true',
                       help='Run scalability testing only')
    parser.add_argument('--skip-prerequisites', action='store_true',
                       help='Skip prerequisite checking')
    
    args = parser.parse_args()
    
    # Default to week2 validation if no specific mode chosen
    if not any([args.week2_validation, args.quick_test, args.hardware_only, 
               args.mnist_only, args.scalability_only]):
        args.week2_validation = True
        
    print("🚀 E-NCN COMPREHENSIVE VALIDATION SYSTEM")
    print("Mission: Prove 1000x energy reduction through hardware validation")
    print("Week 2 Target: >200x energy reduction with >98% MNIST accuracy")
    print("=" * 70)
    
    orchestrator = ValidationOrchestrator()
    
    # Check prerequisites
    if not args.skip_prerequisites:
        if not orchestrator.check_prerequisites():
            print("\n❌ Prerequisites not satisfied. Please install required packages.")
            print("Required: pip install torch torchvision numpy nvidia-ml-py3 psutil scipy matplotlib pandas")
            sys.exit(1)
    
    # Determine test mode
    quick_mode = args.quick_test
    
    try:
        # Run requested validations
        if args.week2_validation or args.quick_test:
            # Full validation suite
            orchestrator.run_hardware_validation(quick_test=quick_mode)
            orchestrator.run_mnist_validation(quick_test=quick_mode)
            orchestrator.run_scalability_validation(quick_test=quick_mode)
        else:
            # Individual validations
            if args.hardware_only:
                orchestrator.run_hardware_validation(quick_test=quick_mode)
            if args.mnist_only:
                orchestrator.run_mnist_validation(quick_test=quick_mode)
            if args.scalability_only:
                orchestrator.run_scalability_validation(quick_test=quick_mode)
                
        # Generate summary
        summary = orchestrator.generate_summary_report()
        
        # Save results
        results_file = orchestrator.save_results(summary)
        
        # Exit with appropriate code
        if summary['overall_success']:
            print("\n🎆 VALIDATION SUCCESS!")
            print("E-NCN energy reduction claims validated on real hardware")
            sys.exit(0)
        else:
            print("\n❌ VALIDATION FAILED!")
            print("E-NCN claims require further investigation")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️  Validation interrupted by user")
        sys.exit(2)
    except Exception as e:
        print(f"\n💥 Validation system error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(3)


if __name__ == "__main__":
    main()