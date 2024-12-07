import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from model import SimpleCNN
from datetime import datetime
import os
import numpy as np
import random
from torchinfo import summary
from augment import AugmentedMNIST
from torchviz import make_dot
from visualize_augment import visualize_augmentations

def set_seed():
    """Set all seeds for reproducibility"""
    seed=42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    return seed

def print_model_summary(model):
    """Print model summary using torchinfo"""
    print("\nModel Summary:")
    model_stats = summary(model, 
                         input_size=(1, 1, 28, 28),  # (batch_size, channels, height, width)
                         col_names=["input_size", "output_size", "num_params", "kernel_size"],
                         col_width=20,
                         row_settings=["var_names"])
    
    print(f'\nTotal Parameters: {sum(p.numel() for p in model.parameters()):,}')
    print(f'Parameter Budget: 25,000')
    print(f'Budget Remaining: {25000 - sum(p.numel() for p in model.parameters()):,}')
    print('-' * 80)
    
    return model_stats

def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nTraining model on {device}")
    
    # Initialize model
    model = SimpleCNN().to(device)
    print_model_summary(model)
    
    # Create model-specific directory
    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)
    
    # Create visualization directory
    viz_dir = 'visualizations'
    os.makedirs(viz_dir, exist_ok=True)
    
    # Visualize architecture
    sample_input = torch.randn(1, 1, 28, 28).to(device)
    output = model(sample_input)
    # dot = make_dot(output, params=dict(model.named_parameters()))
    # dot.render(os.path.join(viz_dir, "architecture"), format="png", cleanup=True)
    
    # Dataset and loader
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = AugmentedMNIST('data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=128,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(42)
    )
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters())
    
    # Train for one epoch
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0 or batch_idx == len(train_loader) - 1:
            _, predicted = torch.max(output.data, 1)
            correct = (predicted == target).sum().item()
            accuracy = 100 * correct / len(target)
            print(f'Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%')
    
    # Save model with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join(model_dir, f'weights_{timestamp}.pth')
    torch.save(model.state_dict(), save_path)
    print(f'Model saved as {save_path}')

def train():
    # Set seed at the very beginning
    set_seed()
    
    # Visualize augmentations
    print("\nGenerating augmentation visualizations...")
    visualize_augmentations()
    print("Augmentation visualization saved as 'visualizations/augmentation_examples.png'")
    
    # Set seed for reproducibility
    set_seed()

    # Train both models
    train_model()

if __name__ == '__main__':
    train() 