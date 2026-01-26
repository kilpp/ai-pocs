"""
CNN Model definitions for image classification
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50, VGG16, MobileNetV2, EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Sequential, Model


def create_simple_cnn(input_shape=(224, 224, 3), num_classes=10):
    """
    Create a simple CNN model from scratch
    
    Args:
        input_shape: Input image shape (height, width, channels)
        num_classes: Number of classification categories
        
    Returns:
        Compiled Keras model
    """
    model = Sequential([
        layers.Conv2D(32, 3, activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),
        
        layers.Conv2D(64, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),
        
        layers.Conv2D(128, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),
        
        layers.Conv2D(256, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),
        
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def create_transfer_learning_model(model_name='resnet50', num_classes=10, 
                                    input_shape=(224, 224, 3), freeze_base=True):
    """
    Create a transfer learning model using pre-trained backbone
    
    Args:
        model_name: Name of the pre-trained model ('resnet50', 'vgg16', 'mobilenetv2', 'efficientnetb0')
        num_classes: Number of classification categories
        input_shape: Input image shape
        freeze_base: Whether to freeze base model weights
        
    Returns:
        Compiled Keras model
    """
    # Validate num_classes to avoid invalid Dense units
    if not isinstance(num_classes, int) or num_classes <= 0:
        raise ValueError(
            f"num_classes must be a positive integer, got {num_classes}. "
            "Ensure your dataset directory contains at least one class subfolder with images."
        )
    
    model_map = {
        'resnet50': ResNet50,
        'vgg16': VGG16,
        'mobilenetv2': MobileNetV2,
        'efficientnetb0': EfficientNetB0,
    }
    
    if model_name.lower() not in model_map:
        raise ValueError(f"Model {model_name} not supported. Choose from {list(model_map.keys())}")
    
    base_model = model_map[model_name.lower()](
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    if freeze_base:
        base_model.trainable = False
    
    inputs = keras.Input(shape=input_shape)
    x = keras.applications.resnet50.preprocess_input(inputs) if model_name.lower() == 'resnet50' else inputs
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def get_callbacks(model_name='model'):
    """
    Get training callbacks for better model performance
    
    Returns:
        List of Keras callbacks
    """
    return [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        ),
        keras.callbacks.ModelCheckpoint(
            f'models/{model_name}_best.h5',
            monitor='val_accuracy',
            save_best_only=True
        ),
        keras.callbacks.TensorBoard(
            log_dir='./logs',
            histogram_freq=1
        )
    ]
