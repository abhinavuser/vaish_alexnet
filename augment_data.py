#!/usr/bin/env python3
"""
Data Augmentation Script - Generate 1000+ augmented images
Balances classes by augmenting more images from minority class
"""

import os
import numpy as np
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import random
from tqdm import tqdm
import shutil

random.seed(42)
np.random.seed(42)

class DataAugmentor:
    def __init__(self, base_path='Tomato_Leaves'):
        self.base_path = Path(base_path)
        self.images_path = self.base_path / 'Images'
        self.augmented_path = self.base_path / 'Augmented_Images'
        
    def augment_image(self, img, augmentation_type):
        """Apply specific augmentation to image"""
        
        if augmentation_type == 'rotate_90':
            return img.rotate(90, expand=True)
        
        elif augmentation_type == 'rotate_180':
            return img.rotate(180, expand=True)
        
        elif augmentation_type == 'rotate_270':
            return img.rotate(270, expand=True)
        
        elif augmentation_type == 'rotate_45':
            return img.rotate(45, expand=True)
        
        elif augmentation_type == 'flip_horizontal':
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        
        elif augmentation_type == 'flip_vertical':
            return img.transpose(Image.FLIP_TOP_BOTTOM)
        
        elif augmentation_type == 'brightness_up':
            enhancer = ImageEnhance.Brightness(img)
            return enhancer.enhance(1.3)
        
        elif augmentation_type == 'brightness_down':
            enhancer = ImageEnhance.Brightness(img)
            return enhancer.enhance(0.7)
        
        elif augmentation_type == 'contrast_up':
            enhancer = ImageEnhance.Contrast(img)
            return enhancer.enhance(1.3)
        
        elif augmentation_type == 'contrast_down':
            enhancer = ImageEnhance.Contrast(img)
            return enhancer.enhance(0.7)
        
        elif augmentation_type == 'saturation_up':
            enhancer = ImageEnhance.Color(img)
            return enhancer.enhance(1.3)
        
        elif augmentation_type == 'saturation_down':
            enhancer = ImageEnhance.Color(img)
            return enhancer.enhance(0.7)
        
        elif augmentation_type == 'blur':
            return img.filter(ImageFilter.BLUR)
        
        elif augmentation_type == 'sharpen':
            return img.filter(ImageFilter.SHARPEN)
        
        elif augmentation_type == 'combo1':
            # Rotate + flip
            img = img.rotate(15, expand=True)
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        
        elif augmentation_type == 'combo2':
            # Brightness + contrast
            img = ImageEnhance.Brightness(img).enhance(1.2)
            return ImageEnhance.Contrast(img).enhance(1.2)
        
        elif augmentation_type == 'combo3':
            # Rotate + brightness
            img = img.rotate(25, expand=True)
            return ImageEnhance.Brightness(img).enhance(1.1)
        
        return img
    
    def generate_augmented_dataset(self, target_per_class=1000):
        """Generate augmented images to balance classes"""
        
        print("="*80)
        print("DATA AUGMENTATION PIPELINE")
        print("="*80)
        
        augmentation_types = [
            'rotate_90', 'rotate_180', 'rotate_270', 'rotate_45',
            'flip_horizontal', 'flip_vertical',
            'brightness_up', 'brightness_down',
            'contrast_up', 'contrast_down',
            'saturation_up', 'saturation_down',
            'blur', 'sharpen',
            'combo1', 'combo2', 'combo3'
        ]
        
        train_src = self.images_path / 'train'
        train_dst = self.augmented_path / 'train'
        train_dst.mkdir(parents=True, exist_ok=True)
        
        healthy_images = sorted(list(train_src.glob('H*.jpg')))
        diseased_images = sorted(list(train_src.glob('D*.jpg')))
        
        print(f"\nOriginal Training Set:")
        print(f"  Healthy:  {len(healthy_images)} images")
        print(f"  Diseased: {len(diseased_images)} images")
        print(f"  Total:    {len(healthy_images) + len(diseased_images)} images")
        
        print(f"\nTarget per class: {target_per_class} images")
        
        print("\n1. Copying original images...")
        for img_path in tqdm(healthy_images + diseased_images, desc="Copying"):
            shutil.copy(img_path, train_dst / img_path.name)
        
        healthy_needed = target_per_class - len(healthy_images)
        print(f"\n2. Augmenting Healthy images ({healthy_needed} needed)...")
        
        aug_count = 0
        while aug_count < healthy_needed:
            for img_path in tqdm(healthy_images, desc=f"Healthy Aug Round {aug_count // len(healthy_images) + 1}"):
                if aug_count >= healthy_needed:
                    break
                
                img = Image.open(img_path).convert('RGB')
                aug_type = random.choice(augmentation_types)
                aug_img = self.augment_image(img, aug_type)
                
                base_name = img_path.stem
                aug_name = f"{base_name}_aug{aug_count}_{aug_type}.jpg"
                aug_img.save(train_dst / aug_name, quality=95)
                aug_count += 1
        
        diseased_needed = target_per_class - len(diseased_images)
        print(f"\n3. Augmenting Diseased images ({diseased_needed} needed)...")
        
        aug_count = 0
        while aug_count < diseased_needed:
            for img_path in tqdm(diseased_images, desc=f"Diseased Aug Round {aug_count // len(diseased_images) + 1}"):
                if aug_count >= diseased_needed:
                    break
                
                img = Image.open(img_path).convert('RGB')
                aug_type = random.choice(augmentation_types)
                aug_img = self.augment_image(img, aug_type)
                
                base_name = img_path.stem
                aug_name = f"{base_name}_aug{aug_count}_{aug_type}.jpg"
                aug_img.save(train_dst / aug_name, quality=95)
                aug_count += 1
        
        print("\n4. Copying validation set...")
        val_src = self.images_path / 'val'
        val_dst = self.augmented_path / 'val'
        val_dst.mkdir(parents=True, exist_ok=True)
        
        for img_path in tqdm(list(val_src.glob('*.jpg')), desc="Validation"):
            shutil.copy(img_path, val_dst / img_path.name)
        
        # Final counts
        final_healthy = len(list(train_dst.glob('H*.jpg')))
        final_diseased = len(list(train_dst.glob('D*.jpg')))
        final_val = len(list(val_dst.glob('*.jpg')))
        
        print("\n" + "="*80)
        print("AUGMENTATION COMPLETE!")
        print("="*80)
        print(f"\nAugmented Training Set:")
        print(f"  Healthy:  {final_healthy} images (original: {len(healthy_images)})")
        print(f"  Diseased: {final_diseased} images (original: {len(diseased_images)})")
        print(f"  Total:    {final_healthy + final_diseased} images")
        print(f"\nValidation Set: {final_val} images (unchanged)")
        print(f"\nOverall Dataset: {final_healthy + final_diseased + final_val} images")
        print(f"\nClass Balance: {final_healthy}/{final_diseased} = {final_healthy/final_diseased:.2f}")
        
        print(f"\n✓ Augmented dataset saved to: {self.augmented_path}")

if __name__ == '__main__':
    augmentor = DataAugmentor()
    augmentor.generate_augmented_dataset(target_per_class=1000)
