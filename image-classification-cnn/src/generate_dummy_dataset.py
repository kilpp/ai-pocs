import os
import numpy as np
from PIL import Image

root = os.path.join(os.path.dirname(__file__), '..', 'data')
root = os.path.abspath(root)
train_dir = os.path.join(root, 'train')
val_dir = os.path.join(root, 'val')
classes = ['classA', 'classB']

# Create directories
for split in [train_dir, val_dir]:
    for cls in classes:
        os.makedirs(os.path.join(split, cls), exist_ok=True)

# Helper to create a random image with a base color
def make_image(base_color):
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    noise = np.random.randint(0, 50, (224, 224, 3), dtype=np.uint8)
    img[:] = base_color
    img = np.clip(img + noise, 0, 255)
    return Image.fromarray(img)

# Generate images
counts = {"train": {}, "val": {}}
base_colors = {
    "classA": (200, 50, 50),   # reddish
    "classB": (50, 50, 200),   # bluish
}

for cls in classes:
    # 10 training images per class
    cls_train = os.path.join(train_dir, cls)
    for i in range(10):
        img = make_image(base_colors[cls])
        img.save(os.path.join(cls_train, f"{cls}_train_{i}.png"))
    counts["train"][cls] = len([f for f in os.listdir(cls_train) if f.lower().endswith(('.png','.jpg','.jpeg'))])

    # 4 validation images per class
    cls_val = os.path.join(val_dir, cls)
    for i in range(4):
        img = make_image(base_colors[cls])
        img.save(os.path.join(cls_val, f"{cls}_val_{i}.png"))
    counts["val"][cls] = len([f for f in os.listdir(cls_val) if f.lower().endswith(('.png','.jpg','.jpeg'))])

print("Dataset created at:", root)
print("Train:", counts["train"]) 
print("Val:", counts["val"]) 
