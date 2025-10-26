"""Unit Tests for E-NCN Layers.

Comprehensive test suite for Energy-Efficient Non-Conventional Neuron layers,
validating functionality, performance, and energy efficiency.

"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# Add src directory to path for imports
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from encn.layers import ENCNLayer, ENCNSequential, create_encn_mlp


class TestENCNLayer:
    """Test cases for ENCNLayer class."""
    
    @pytest.fixture
    def basic_layer(self):
        """Create a basic E-NCN layer for testing."""
        return ENCNLayer(in_features=10, out_features=5, threshold=0.1)
    
    @pytest.fixture
    def sample_input(self):
        """Create sample input tensor."""
        torch.manual_seed(42)
        return torch.randn(3, 10)  # Batch size 3, 10 features
    
    def test_layer_initialization(self, basic_layer):
        """Test proper layer initialization."""
        assert basic_layer.in_features == 10
        assert basic_layer.out_features == 5
        assert basic_layer.threshold.item() == pytest.approx(0.1, abs=1e-6)
        assert basic_layer.sparsity_target == 0.99
        assert basic_layer.energy_tracking is True
        
        # Check parameter initialization
        assert basic_layer.weight.shape == (5, 10)
        assert basic_layer.bias.shape == (5,)
        
    def test_forward_pass_shape(self, basic_layer, sample_input):
        """Test forward pass produces correct output shape."""
        output = basic_layer(sample_input)
        assert output.shape == (3, 5)  # batch_size=3, out_features=5
        
    def test_forward_pass_deterministic(self, basic_layer, sample_input):
        """Test forward pass is deterministic with same input."""
        basic_layer.eval()
        with torch.no_grad():
            output1 = basic_layer(sample_input)
            output2 = basic_layer(sample_input)
            torch.testing.assert_close(output1, output2)
            
    def test_threshold_learning(self):
        """Test threshold parameter learning."""
        layer = ENCNLayer(10, 5, threshold=0.1, learn_threshold=True)
        assert layer.threshold.requires_grad is True
        
        layer_no_learning = ENCNLayer(10, 5, threshold=0.1, learn_threshold=False)
        assert layer_no_learning.threshold.requires_grad is False
        
    def test_sparsity_computation(self, sample_input):
        """Test sparsity calculation."""
        # High threshold should produce high sparsity
        high_threshold_layer = ENCNLayer(10, 5, threshold=10.0)
        high_threshold_layer.eval()
        
        with torch.no_grad():
            _ = high_threshold_layer(sample_input)
            high_sparsity = high_threshold_layer.get_sparsity()
            
        # Low threshold should produce low sparsity
        low_threshold_layer = ENCNLayer(10, 5, threshold=0.001)
        low_threshold_layer.eval()
        
        with torch.no_grad():
            _ = low_threshold_layer(sample_input)
            low_sparsity = low_threshold_layer.get_sparsity()
            
        assert high_sparsity > low_sparsity
        assert 0.0 <= high_sparsity <= 1.0
        assert 0.0 <= low_sparsity <= 1.0
        
    def test_energy_tracking(self, basic_layer, sample_input):
        """Test energy consumption tracking."""
        basic_layer.reset_energy_stats()
        
        # Initial state
        stats_before = basic_layer.get_energy_stats()
        assert stats_before['total_operations'] == 0
        assert stats_before['sparse_operations'] == 0
        
        # After forward pass
        _ = basic_layer(sample_input)
        stats_after = basic_layer.get_energy_stats()
        
        assert stats_after['total_operations'] > 0
        assert stats_after['sparse_operations'] >= 0
        assert stats_after['sparse_operations'] <= stats_after['total_operations']
        assert stats_after['energy_reduction'] >= 1.0
        
    def test_gradient_flow(self, basic_layer, sample_input):
        """Test gradient flow through sparse computation."""
        basic_layer.train()
        sample_input.requires_grad_(True)
        
        output = basic_layer(sample_input)
        loss = output.sum()
        loss.backward()
        
        # Check gradients exist
        assert basic_layer.weight.grad is not None
        assert basic_layer.bias.grad is not None
        assert basic_layer.threshold.grad is not None
        assert sample_input.grad is not None
        
        # Check gradient shapes
        assert basic_layer.weight.grad.shape == basic_layer.weight.shape
        assert basic_layer.bias.grad.shape == basic_layer.bias.shape
        assert basic_layer.threshold.grad.shape == basic_layer.threshold.shape
        
    def test_sparsity_target_setting(self, basic_layer):
        """Test sparsity target adjustment."""
        original_threshold = basic_layer.threshold.item()
        
        # Set high sparsity target
        basic_layer.set_sparsity_target(0.995)
        high_sparsity_threshold = basic_layer.threshold.item()
        
        # Set low sparsity target
        basic_layer.set_sparsity_target(0.5)
        low_sparsity_threshold = basic_layer.threshold.item()
        
        # Higher sparsity should generally require higher threshold
        assert high_sparsity_threshold != original_threshold
        assert low_sparsity_threshold != high_sparsity_threshold
        
    def test_invalid_sparsity_target(self, basic_layer):
        """Test invalid sparsity target raises error."""
        with pytest.raises(ValueError):
            basic_layer.set_sparsity_target(-0.1)
            
        with pytest.raises(ValueError):
            basic_layer.set_sparsity_target(1.1)
            
    def test_no_bias_option(self):
        """Test layer without bias term."""
        layer = ENCNLayer(10, 5, bias=False)
        assert layer.bias is None
        
        sample_input = torch.randn(2, 10)
        output = layer(sample_input)
        assert output.shape == (2, 5)
        
    def test_energy_stats_reset(self, basic_layer, sample_input):
        """Test energy statistics reset functionality."""
        # Generate some statistics
        _ = basic_layer(sample_input)
        stats_before = basic_layer.get_energy_stats()
        assert stats_before['total_operations'] > 0
        
        # Reset and check
        basic_layer.reset_energy_stats()
        stats_after = basic_layer.get_energy_stats()
        assert stats_after['total_operations'] == 0
        assert stats_after['sparse_operations'] == 0
        
    def test_layer_representation(self, basic_layer):
        """Test layer string representation."""
        repr_str = repr(basic_layer)
        assert "ENCNLayer" in repr_str
        assert "in_features=10" in repr_str
        assert "out_features=5" in repr_str
        assert "threshold=" in repr_str
        
    def test_different_device_compatibility(self, basic_layer):
        """Test layer works on different devices."""
        device = torch.device('cpu')
        basic_layer.to(device)
        
        sample_input = torch.randn(2, 10, device=device)
        output = basic_layer(sample_input)
        
        assert output.device == device
        assert output.shape == (2, 5)
        
        # Test CUDA if available
        if torch.cuda.is_available():
            cuda_device = torch.device('cuda:0')
            basic_layer.to(cuda_device)
            
            cuda_input = torch.randn(2, 10, device=cuda_device)
            cuda_output = basic_layer(cuda_input)
            
            assert cuda_output.device == cuda_device
            assert cuda_output.shape == (2, 5)


class TestENCNSequential:
    """Test cases for ENCNSequential container."""
    
    @pytest.fixture
    def sequential_model(self):
        """Create sequential E-NCN model."""
        return ENCNSequential(
            ENCNLayer(20, 15, threshold=0.1),
            nn.ReLU(),
            ENCNLayer(15, 10, threshold=0.1),
            nn.ReLU(),
            ENCNLayer(10, 5, threshold=0.1)
        )
    
    @pytest.fixture
    def sample_batch(self):
        """Create sample batch for testing."""
        torch.manual_seed(42)
        return torch.randn(4, 20)
    
    def test_sequential_forward(self, sequential_model, sample_batch):
        """Test forward pass through sequential model."""
        output = sequential_model(sample_batch)
        assert output.shape == (4, 5)
        
    def test_total_energy_stats(self, sequential_model, sample_batch):
        """Test aggregated energy statistics."""
        # Reset all statistics
        sequential_model.reset_all_energy_stats()
        
        # Forward pass
        _ = sequential_model(sample_batch)
        
        # Get total statistics
        total_stats = sequential_model.get_total_energy_stats()
        
        assert total_stats['total_layers'] == 3  # 3 E-NCN layers
        assert 'avg_sparsity' in total_stats
        assert 'total_energy_reduction' in total_stats
        assert len(total_stats['layer_stats']) == 3
        
    def test_set_all_sparsity_targets(self, sequential_model):
        """Test setting sparsity targets for all layers."""
        target_sparsity = 0.95
        sequential_model.set_all_sparsity_targets(target_sparsity)
        
        for module in sequential_model.modules():
            if isinstance(module, ENCNLayer):
                assert module.sparsity_target == target_sparsity


class TestCreateENCNMLP:
    """Test cases for MLP creation utility."""
    
    def test_basic_mlp_creation(self):
        """Test basic MLP creation."""
        layer_sizes = [784, 256, 128, 10]
        mlp = create_encn_mlp(layer_sizes)
        
        sample_input = torch.randn(5, 784)
        output = mlp(sample_input)
        
        assert output.shape == (5, 10)
        
    def test_mlp_with_different_activations(self):
        """Test MLP with different activation functions."""
        layer_sizes = [10, 8, 5]
        
        activations = ['relu', 'leaky_relu', 'gelu', 'tanh']
        
        for activation in activations:
            mlp = create_encn_mlp(layer_sizes, activation=activation)
            sample_input = torch.randn(3, 10)
            output = mlp(sample_input)
            assert output.shape == (3, 5)
            
    def test_mlp_with_dropout(self):
        """Test MLP with dropout."""
        layer_sizes = [10, 8, 5]
        mlp = create_encn_mlp(layer_sizes, dropout=0.2)
        
        sample_input = torch.randn(3, 10)
        output = mlp(sample_input)
        assert output.shape == (3, 5)
        
    def test_mlp_invalid_layer_sizes(self):
        """Test MLP creation with invalid layer sizes."""
        with pytest.raises(ValueError):
            create_encn_mlp([10])  # Need at least 2 layer sizes
            
    def test_mlp_unsupported_activation(self):
        """Test MLP with unsupported activation."""
        with pytest.raises(ValueError):
            create_encn_mlp([10, 5], activation='invalid')


class TestENCNIntegration:
    """Integration tests for E-NCN components."""
    
    def test_mnist_like_network(self):
        """Test E-NCN network on MNIST-like data."""
        # Create network similar to MNIST classifier
        network = create_encn_mlp(
            [784, 256, 128, 10],
            threshold=0.1,
            sparsity_target=0.95,
            activation='relu'
        )
        
        # Simulate MNIST batch
        batch_size = 32
        mnist_batch = torch.randn(batch_size, 784)
        
        # Forward pass
        logits = network(mnist_batch)
        assert logits.shape == (batch_size, 10)
        
        # Check sparsity achieved
        total_stats = network.get_total_energy_stats()
        assert total_stats['avg_sparsity'] > 0.5  # Should achieve some sparsity
        
    def test_training_mode_vs_eval_mode(self):
        """Test behavior difference between training and evaluation modes."""
        layer = ENCNLayer(10, 5, threshold=0.1)
        sample_input = torch.randn(3, 10)
        
        # Training mode
        layer.train()
        layer.reset_energy_stats()
        train_output = layer(sample_input)
        train_stats = layer.get_energy_stats()
        
        # Evaluation mode
        layer.eval()
        layer.reset_energy_stats()
        with torch.no_grad():
            eval_output = layer(sample_input)
            eval_stats = layer.get_energy_stats()
            
        # Outputs should be the same
        torch.testing.assert_close(train_output, eval_output, rtol=1e-5, atol=1e-7)
        
    def test_large_batch_handling(self):
        """Test handling of large batches."""
        layer = ENCNLayer(100, 50, threshold=0.1)
        large_batch = torch.randn(1000, 100)  # Large batch
        
        output = layer(large_batch)
        assert output.shape == (1000, 50)
        
        stats = layer.get_energy_stats()
        assert stats['total_operations'] > 0
        assert stats['energy_reduction'] >= 1.0
        
    def test_extreme_sparsity(self):
        """Test behavior with extreme sparsity settings."""
        # Very high threshold - should produce very high sparsity
        sparse_layer = ENCNLayer(10, 5, threshold=100.0)
        
        sample_input = torch.randn(5, 10)
        output = sparse_layer(sample_input)
        
        assert output.shape == (5, 5)
        sparsity = sparse_layer.get_sparsity()
        assert sparsity > 0.95  # Should be very sparse
        
    def test_memory_efficiency(self):
        """Test memory efficiency of E-NCN layers."""
        import gc
        
        def get_memory_usage():
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                return torch.cuda.memory_allocated()
            else:
                gc.collect()
                return 0  # Approximate for CPU
                
        initial_memory = get_memory_usage()
        
        # Create and use large network
        large_network = create_encn_mlp([1000, 500, 250, 10], sparsity_target=0.99)
        large_input = torch.randn(100, 1000)
        
        if torch.cuda.is_available():
            large_network = large_network.cuda()
            large_input = large_input.cuda()
            
        _ = large_network(large_input)
        peak_memory = get_memory_usage()
        
        # Clean up
        del large_network, large_input
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        final_memory = get_memory_usage()
        
        # Memory should be released (approximately)
        if torch.cuda.is_available():
            assert peak_memory > initial_memory  # Memory was used
            assert final_memory <= peak_memory   # Memory was released


class TestENCNNumericalStability:
    """Test numerical stability of E-NCN computations."""
    
    def test_gradient_magnitude_stability(self):
        """Test gradient magnitudes remain stable."""
        layer = ENCNLayer(50, 25, threshold=0.1)
        
        # Test with different input magnitudes
        input_scales = [0.001, 0.1, 1.0, 10.0, 100.0]
        gradient_norms = []
        
        for scale in input_scales:
            layer.zero_grad()
            sample_input = torch.randn(10, 50) * scale
            sample_input.requires_grad_(True)
            
            output = layer(sample_input)
            loss = output.sum()
            loss.backward()
            
            grad_norm = layer.weight.grad.norm().item()
            gradient_norms.append(grad_norm)
            
        # Gradients shouldn't explode or vanish dramatically
        max_grad = max(gradient_norms)
        min_grad = min([g for g in gradient_norms if g > 0])  # Exclude zeros
        
        if min_grad > 0:
            ratio = max_grad / min_grad
            assert ratio < 1e6, f"Gradient ratio too large: {ratio}"
            
    def test_threshold_optimization_stability(self):
        """Test threshold optimization doesn't cause instability."""
        layer = ENCNLayer(20, 10, threshold=0.1, learn_threshold=True)
        optimizer = torch.optim.Adam(layer.parameters(), lr=0.01)
        
        initial_threshold = layer.threshold.item()
        threshold_history = [initial_threshold]
        
        # Simulate training steps
        for _ in range(100):
            optimizer.zero_grad()
            
            sample_input = torch.randn(8, 20)
            output = layer(sample_input)
            
            # Simple loss: encourage sparsity while maintaining output magnitude
            sparsity = layer.get_sparsity()
            loss = output.pow(2).mean() + 0.1 * (0.95 - sparsity)**2
            
            loss.backward()
            optimizer.step()
            
            threshold_history.append(layer.threshold.item())
            
        # Threshold should not diverge
        final_threshold = threshold_history[-1]
        assert 0.001 < final_threshold < 10.0, f"Threshold diverged: {final_threshold}"
        
        # Should show some adaptation
        threshold_std = np.std(threshold_history[-50:])  # Last 50 steps
        assert threshold_std < 1.0, "Threshold too unstable"


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])