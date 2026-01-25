"""
Data augmentation utilities for image classification
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator


def get_train_augmentation():
    """
    Create data augmentation pipeline for training data
    
    Returns:
        ImageDataGenerator with augmentation parameters
    """
    return ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.15,
        fill_mode='nearest',
        brightness_range=[0.8, 1.2],
        vertical_flip=False
    )


def get_val_augmentation():
    """
    Create data augmentation pipeline for validation data (minimal augmentation)
    
    Returns:
        ImageDataGenerator with minimal augmentation
    """
    return ImageDataGenerator(rescale=1./255)


def get_test_augmentation():
    """
    Create data augmentation pipeline for test data (only normalization)
    
    Returns:
        ImageDataGenerator with only rescaling
    """
    return ImageDataGenerator(rescale=1./255)


def create_custom_augmentation(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=False,
    shear_range=0.15,
    brightness_range=None,
    fill_mode='nearest'
):
    """
    Create custom data augmentation pipeline
    
    Args:
        rotation_range: Random rotation range in degrees
        width_shift_range: Fraction of total width for horizontal shifts
        height_shift_range: Fraction of total height for vertical shifts
        zoom_range: Zoom range
        horizontal_flip: Whether to apply random horizontal flip
        vertical_flip: Whether to apply random vertical flip
        shear_range: Shear range
        brightness_range: Brightness adjustment range [min, max]
        fill_mode: Fill mode for newly created pixels
        
    Returns:
        ImageDataGenerator with custom parameters
    """
    return ImageDataGenerator(
        rescale=1./255,
        rotation_range=rotation_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        horizontal_flip=horizontal_flip,
        vertical_flip=vertical_flip,
        zoom_range=zoom_range,
        shear_range=shear_range,
        brightness_range=brightness_range,
        fill_mode=fill_mode
    )
