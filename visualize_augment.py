import matplotlib.pyplot as plt
import torch
import torchvision
from augment import AugmentedMNIST
import os
import numpy as np
import random

def set_seed(seed=42):
    """Set all seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def visualize_augmentations(num_samples=5, num_augmentations=3):
    # Set seed before visualization
    set_seed()
    
    # Create visualizations directory if it doesn't exist
    viz_dir = 'visualizations'
    os.makedirs(viz_dir, exist_ok=True)
    
    # Load dataset
    dataset = AugmentedMNIST('data', train=True, download=True, transform=torchvision.transforms.ToTensor())
    
    # Create figure
    fig, axes = plt.subplots(num_samples, num_augmentations + 1, 
                            figsize=(2*(num_augmentations + 1), 2*num_samples))
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    
    # Use seeded random choice
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        # Get original image
        img, label = dataset[idx]
        axes[i, 0].imshow(img.squeeze(), cmap='gray')
        axes[i, 0].set_title(f'Original\n(Digit: {label})')
        axes[i, 0].axis('off')
        
        # Reset seed for each augmentation to ensure reproducibility
        for j in range(num_augmentations):
            set_seed(42 + j)  # Different seed for each augmentation
            aug_img, _ = dataset[idx]
            axes[i, j+1].imshow(aug_img.squeeze(), cmap='gray')
            axes[i, j+1].set_title(f'Aug {j+1}')
            axes[i, j+1].axis('off')
    
    # Save figure
    plt.savefig(os.path.join(viz_dir, 'augmentation_examples.png'), 
                bbox_inches='tight', dpi=150)
    plt.close()

if __name__ == "__main__":
    visualize_augmentations() 