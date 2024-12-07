import torch
import torchvision.datasets as datasets
import imgaug.augmenters as iaa
import numpy as np

class AugmentedMNIST(datasets.MNIST):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # MNIST-specific augmentations
        self.augmenter = iaa.Sequential([
            # Small rotations (digits should still be readable)
            iaa.Affine(
                rotate=(-7, 7),  # Rotation range
                scale=(0.9, 1.1),  # Slight scaling
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},  # Small translations
                mode='constant',
                cval=0,  # Fill with black
                random_state=42  # Add random state
            ),
            # Elastic deformations (common in OCR tasks)
            iaa.Sometimes(0.3, iaa.ElasticTransformation(alpha=(0, 0.5), sigma=0.15, random_state=42)),
            
            
            # Contrast normalization
            iaa.Sometimes(0.3, iaa.contrast.LinearContrast((0.8, 1.2), random_state=42)),
            
            # Slight noise (helps with robustness)
            iaa.Sometimes(0.2, iaa.AdditiveGaussianNoise(scale=(0, 0.1 * 255), random_state=42))
        ], random_order=False, random_state=42)  # Keep order for consistent augmentations
        
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        
        if self.train:
            # Convert to numpy for imgaug
            img_np = np.array(img.squeeze()) * 255
            img_np = img_np.astype(np.uint8)
            
            # Apply augmentation
            img_aug = self.augmenter.augment_image(img_np)
            
            # Ensure the image stays in valid range
            img_aug = np.clip(img_aug, 0, 255)
            
            # Convert back to tensor
            img_aug = img_aug.astype(np.float32) / 255.0
            img = torch.from_numpy(img_aug).unsqueeze(0)
            
            # No need to adjust targets for these augmentations
            # as they preserve digit identity
        
        return img, target