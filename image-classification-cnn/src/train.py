"""
Training script for image classification CNN
"""

import os
import argparse
import numpy as np
from model import create_simple_cnn, create_transfer_learning_model, get_callbacks
from data_loader import create_data_generators, get_class_names
import matplotlib.pyplot as plt


def train_model(train_dir, val_dir=None, model_type='transfer', transfer_model='resnet50',
                epochs=50, batch_size=32, learning_rate=0.001):
    """
    Train image classification model
    
    Args:
        train_dir: Path to training data
        val_dir: Path to validation data
        model_type: 'simple' or 'transfer'
        transfer_model: Name of transfer learning model if model_type='transfer'
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
    """
    
    print(f"Loading data from {train_dir}...")
    train_generator, val_generator = create_data_generators(
        train_dir,
        val_dir=val_dir,
        batch_size=batch_size
    )
    
    num_classes = len(get_class_names(train_dir))
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {get_class_names(train_dir)}")
    
    if model_type.lower() == 'simple':
        print("Creating simple CNN model...")
        model = create_simple_cnn(num_classes=num_classes)
    elif model_type.lower() == 'transfer':
        print(f"Creating transfer learning model with {transfer_model}...")
        model = create_transfer_learning_model(
            model_name=transfer_model,
            num_classes=num_classes
        )
    else:
        raise ValueError("model_type must be 'simple' or 'transfer'")
    
    print("\nModel Summary:")
    model.summary()
    
    callbacks = get_callbacks(model_name=f"{model_type}_{transfer_model if model_type == 'transfer' else 'cnn'}")
    
    print(f"\nTraining for {epochs} epochs...")
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=callbacks
    )
    
    save_path = f"models/{model_type}_{transfer_model if model_type == 'transfer' else 'cnn'}_final.h5"
    model.save(save_path)
    print(f"\nModel saved to {save_path}")
    
    plot_training_history(history, model_type, transfer_model if model_type == 'transfer' else 'cnn')
    
    return model, history


def plot_training_history(history, model_type, model_name):
    """
    Plot training history
    
    Args:
        history: Training history object
        model_type: Type of model trained
        model_name: Name of the model
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    axes[0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(history.history['loss'], label='Training Loss')
    axes[1].plot(history.history['val_loss'], label='Validation Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    save_path = f"models/{model_type}_{model_name}_training_history.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training history plot saved to {save_path}")
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train image classification model')
    parser.add_argument('--train-dir', type=str, default='data/train',
                        help='Path to training data directory')
    parser.add_argument('--val-dir', type=str, default='data/val',
                        help='Path to validation data directory')
    parser.add_argument('--model-type', type=str, default='transfer',
                        choices=['simple', 'transfer'],
                        help='Type of model to train')
    parser.add_argument('--transfer-model', type=str, default='resnet50',
                        choices=['resnet50', 'vgg16', 'mobilenetv2', 'efficientnetb0'],
                        help='Pre-trained model for transfer learning')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Learning rate')
    
    args = parser.parse_args()
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    train_model(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        model_type=args.model_type,
        transfer_model=args.transfer_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
