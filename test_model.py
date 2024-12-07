import torch
from torchvision import datasets, transforms
from model import SimpleCNN
import pytest
import glob
import numpy as np
import warnings

warnings.filterwarnings('ignore')

def get_latest_model():
    model_files = glob.glob(f'models/weights_*.pth')
    return max(model_files) if model_files else None

def run_model_tests():
    def test_model_architecture():
        model = SimpleCNN()
        
        # Test input shape
        test_input = torch.randn(1, 1, 28, 28)
        output = model(test_input)
        assert output.shape == (1, 10), f"model Output shape should be (1, 10)"
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params < 25000, f"model Has {total_params} parameters, should be < 25000"

    def test_batch_processing():
        model = SimpleCNN()
        batch_sizes = [1, 16, 32, 64, 128]
        for batch_size in batch_sizes:
            test_input = torch.randn(batch_size, 1, 28, 28)
            output = model(test_input)
            assert output.shape == (batch_size, 10), f"model Failed for batch size {batch_size}"

    def test_output_range():
        model = SimpleCNN()
        test_input = torch.randn(10, 1, 28, 28)
        output = model(test_input)
        
        assert torch.any(output > 1) or torch.any(output < 0), f"model Should output logits"
        assert torch.all(torch.isfinite(output)), f"model Outputs contain NaN or Inf values"

    def test_model_accuracy():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = SimpleCNN().to(device)
        
        model_path = get_latest_model()
        assert model_path is not None, f"No model weights found for model"
        model.load_state_dict(torch.load(model_path))
        
        transform = transforms.Compose([transforms.ToTensor()])
        test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)
        
        model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        accuracy = 100 * correct / total
        print(f"\nmodel Accuracy: {accuracy:.2f}%")
        assert accuracy > 95, f"model Accuracy is {accuracy}%, should be > 95%"
        
        unique_preds = np.unique(all_preds)
        assert len(unique_preds) == 10, f"model Should predict all 10 digits"
        
        pred_counts = np.bincount(all_preds)
        max_pred_ratio = np.max(pred_counts) / len(all_preds)
        assert max_pred_ratio < 0.3, f"model Predictions too biased: {max_pred_ratio:.2f}"

    def test_model_robustness():
        model = SimpleCNN()
        model.eval()
        
        test_input = torch.randn(1, 1, 28, 28)
        noisy_input = test_input + 0.1 * torch.randn_like(test_input)
        
        with torch.no_grad():
            clean_output = model(test_input)
            noisy_output = model(noisy_input)
        
        output_diff = torch.abs(clean_output - noisy_output).mean()
        assert output_diff < 1.0, f"model Too sensitive to noise: {output_diff:.2f}"

    return [
        test_model_architecture,
        test_batch_processing,
        test_output_range,
        test_model_accuracy,
        test_model_robustness
    ]

# Generate test functions for both models
for test_fn in run_model_tests():
    globals()[f"test_simple_{test_fn.__name__}"] = test_fn

if __name__ == "__main__":
    pytest.main([__file__]) 