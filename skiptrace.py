import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from sklearn.neighbors import NearestNeighbors

# Device setup (GPU/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} ...")

# Load ResNet model
resnet_model = models.resnet50(pretrained=True)
resnet_model.fc = nn.Identity()  # Remove classification layer
resnet_model = resnet_model.to(device)
resnet_model.eval()

# Image transformation pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Paths to dataset and query images
dataset_folder = "data\Task5\database"
query_images = ["data\Task5\query\query_1.jpg", "data\Task5\query\query_2.jpg", "data\Task5\query\query_3.jpg"]

# Preprocessing images for ResNet in batches
def preprocess_images_batch(image_paths):
    image_tensors = []
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        image_array = np.array(image)

        # Mask white pixels (replace with black)
        mask = np.all(image_array == [255, 255, 255], axis=-1)
        image_array[mask] = [0, 0, 0]  # Replace white with black
        masked_image = Image.fromarray(image_array)

        # Apply transformations
        tensor = transform(masked_image)
        image_tensors.append(tensor)
    
    return torch.stack(image_tensors).to(device)  # Batch of tensors

# Preprocessing images for ResNet in batches
def preprocess_images_batch(image_paths):
    image_tensors = []
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        image_array = np.array(image)

        # Mask white pixels (replace with black)
        mask = np.all(image_array == [255, 255, 255], axis=-1)
        image_array[mask] = [0, 0, 0]  # Replace white with black
        masked_image = Image.fromarray(image_array)

        # Apply transformations
        tensor = transform(masked_image)
        image_tensors.append(tensor)
    
    return torch.stack(image_tensors).to(device)  # Batch of tensors

# Feature extraction in batches
def extract_features_batch(image_paths, model, batch_size=16):
    features = []
    for i in range(0, len(image_paths), batch_size):
        print(f"\rProcessing {i} / {len(image_paths)} images ...", end="")
        batch_paths = image_paths[i:i + batch_size]
        image_tensors = preprocess_images_batch(batch_paths)
        
        with torch.no_grad():
            batch_features = model(image_tensors).cpu().numpy()  # Move features to CPU
        features.extend(batch_features)
    
    return features

# Step 1: Feature extraction for dataset and query images
dataset_files = [os.path.join(dataset_folder, f) for f in os.listdir(dataset_folder) if f.endswith('.jpg')]
dataset_features = extract_features_batch(dataset_files, resnet_model, batch_size=32)
query_features = extract_features_batch(query_images, resnet_model, batch_size=32)

# Step 2: Location matching using k-NN
k = 1  # Number of closest matches to retrieve
knn = NearestNeighbors(n_neighbors=k, metric='cosine')
knn.fit(dataset_features)

# Find k-nearest neighbors for each query
all_matched_images = {}
for i, query in enumerate(query_features):
    distances, indices = knn.kneighbors([query])
    matched_images = [dataset_files[idx] for idx in indices.flatten()]
    all_matched_images[query_images[i]] = matched_images

# Print matched images
for query_img, matches in all_matched_images.items():
    print(f"Query image: {query_img}")
    print("Matched images:", matches)
