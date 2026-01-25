"""
Prediction script for image classification
"""

import argparse
import numpy as np
from tensorflow import keras
from data_loader import prepare_single_image, get_class_names
import matplotlib.pyplot as plt


def predict_image(image_path, model_path, class_names=None):
    """
    Make prediction on a single image
    
    Args:
        image_path: Path to image file
        model_path: Path to trained model
        class_names: List of class names
        
    Returns:
        Predicted class and confidence
    """
    
    # Load model
    model = keras.models.load_model(model_path)
    print(f"Model loaded from {model_path}")
    
    # Prepare image
    img_array = prepare_single_image(image_path)
    
    # Make prediction
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx]
    
    if class_names:
        predicted_class = class_names[predicted_class_idx]
    else:
        predicted_class = f"Class {predicted_class_idx}"
    
    print(f"\nPrediction Results:")
    print(f"Predicted Class: {predicted_class}")
    print(f"Confidence: {confidence:.2%}")
    
    # Print top 5 predictions
    print("\nTop 5 Predictions:")
    top_5_idx = np.argsort(predictions[0])[-5:][::-1]
    for idx, class_idx in enumerate(top_5_idx, 1):
        class_name = class_names[class_idx] if class_names else f"Class {class_idx}"
        conf = predictions[0][class_idx]
        print(f"  {idx}. {class_name}: {conf:.2%}")
    
    return predicted_class, confidence, predictions[0]


def visualize_prediction(image_path, model_path, class_names=None):
    """
    Visualize prediction on image
    
    Args:
        image_path: Path to image file
        model_path: Path to trained model
        class_names: List of class names
    """
    from PIL import Image
    
    predicted_class, confidence, all_predictions = predict_image(
        image_path, model_path, class_names
    )
    
    img = Image.open(image_path)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Display image
    axes[0].imshow(img)
    axes[0].set_title(f"Predicted: {predicted_class}\nConfidence: {confidence:.2%}")
    axes[0].axis('off')
    
    # Display predictions bar chart
    if class_names:
        classes = class_names
    else:
        classes = [f"Class {i}" for i in range(len(all_predictions))]
    
    axes[1].barh(classes, all_predictions)
    axes[1].set_xlabel('Confidence')
    axes[1].set_title('Prediction Probabilities')
    axes[1].set_xlim([0, 1])
    
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make predictions using trained model')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to image file')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model file')
    parser.add_argument('--classes-dir', type=str, default=None,
                        help='Path to directory with class folders for class names')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize prediction')
    
    args = parser.parse_args()
    
    class_names = None
    if args.classes_dir:
        class_names = get_class_names(args.classes_dir)
    
    if args.visualize:
        visualize_prediction(args.image, args.model, class_names)
    else:
        predict_image(args.image, args.model, class_names)
