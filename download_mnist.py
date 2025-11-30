#!/usr/bin/env python3
"""
Download MNIST dataset and extract images to PNG files organized by label.
All images (training and test) are stored in a single directory structure.
"""

import os
from pathlib import Path
import numpy as np
from PIL import Image
from torchvision import datasets
from tqdm import tqdm


def download_and_extract_mnist(output_dir="mnist_images"):
    """
    Download MNIST dataset and save all images as PNGs organized by label.
    
    Args:
        output_dir: Directory where images will be saved
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Create subdirectories for each digit (0-9)
    for digit in range(10):
        (output_path / str(digit)).mkdir(exist_ok=True)
    
    print("Downloading MNIST dataset...")
    
    # Download training and test sets
    train_dataset = datasets.MNIST(root='./data', train=True, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True)
    
    # Combine both datasets
    all_images = []
    all_labels = []
    
    print("Processing training set...")
    for img, label in train_dataset:
        all_images.append(np.array(img))
        all_labels.append(label)
    
    print("Processing test set...")
    for img, label in test_dataset:
        all_images.append(np.array(img))
        all_labels.append(label)
    
    print(f"Total images: {len(all_images)}")
    
    # Count images per label for naming
    label_counts = {i: 0 for i in range(10)}
    
    # Save all images as PNGs
    print("Saving images as PNGs...")
    for img_array, label in tqdm(zip(all_images, all_labels), total=len(all_images)):
        # Create filename
        filename = f"{label_counts[label]:05d}.png"
        filepath = output_path / str(label) / filename
        
        # Convert numpy array to PIL Image and save
        img = Image.fromarray(img_array, mode='L')
        img.save(filepath)
        
        label_counts[label] += 1
    
    print(f"\nDone! Images saved to '{output_dir}/' directory")
    print("\nImages per label:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count} images")


if __name__ == "__main__":
    download_and_extract_mnist()
