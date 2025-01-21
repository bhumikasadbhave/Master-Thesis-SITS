import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import vgg16, VGG16_Weights
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score 
from torchgeo.models import resnet18
from torchgeo.models import ResNet18_Weights


def get_pretrained_resnet50(in_channels):
    """Load ResNet-50 and freeze weights for feature extraction."""

    resnet50_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    resnet50_model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    feature_extractor = torch.nn.Sequential(*list(resnet50_model.children())[:-1])  # Remove the FC layer

    # Freeze the model weights
    for param in feature_extractor.parameters():
        param.requires_grad = False

    return feature_extractor


def get_pretrained_vgg16(in_channels):
    """Load VGG16, modify it to accept `in_channels` input channels, and freeze weights for feature extraction."""

    vgg16_model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    vgg16_model.features[0] = nn.Conv2d(in_channels, 64, kernel_size=(3, 3), padding=(1, 1))
    feature_extractor = torch.nn.Sequential(*list(vgg16_model.features.children()))  # Use only the convolutional layers
    
    # Freeze the model weights
    for param in feature_extractor.parameters():
        param.requires_grad = False

    return feature_extractor



def extract_features(feature_extractor, dataloader, model_name):
    """Extract features using the given feature extractor and dataloader."""
    features = []
    field_numbers = []  
    with torch.no_grad():
        for inputs, fields in dataloader:
            
            inputs = F.interpolate(inputs, size=(224, 224), mode="bilinear", align_corners=False)   # Resize inputs to 224x224 to match ResNet50 input size
            outputs = feature_extractor(inputs)

            if model_name == 'resnet50':
                # ResNet50 output: [batch_size, 2048, 1, 1] after the convolutional layers
                outputs = outputs.squeeze(-1).squeeze(-1)     # Flatten to [batch_size, 2048]
            elif model_name == 'vgg16':
                # VGG16 output: [batch_size, 512, 7, 7] after the convolutional layers
                outputs = outputs.view(outputs.size(0), -1)  # Flatten to [batch_size, 512*7*7] or [batch_size, 4096]
            else:
                raise ValueError(f"Unsupported model_name: {model_name}. Choose either 'vgg16' or 'resnet50'.")

            features.append(outputs)
            field_numbers.extend(fields)
    features = torch.cat(features, dim=0).numpy()  
    return features, field_numbers


def perform_clustering(train_features, n_clusters=2, random_state=42):
    """Train a KMeans clustering model on the given features."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(train_features)
    return kmeans


# def evaluate_clustering(kmeans, test_features, test_fields, config):
#     """Evaluate clustering performance on the test set."""
#     test_predictions = kmeans.predict(test_features)
#     test_gt_aligned, test_pred_aligned = get_gt_and_pred_aligned(test_fields, test_predictions, config.labels_path)
#     test_metrics = evaluate_clustering_metrics(test_gt_aligned, test_pred_aligned)
#     return test_metrics


####### Feature Extraction for torch.geo models ########
def extract_features_resnet(dataset, num_channels, pretrained_weights=ResNet18_Weights.SENTINEL2_ALL_MOCO):

    model = resnet18(weights=pretrained_weights)
    original_conv1 = model.conv1
    new_conv1 = nn.Conv2d(
        in_channels=num_channels,
        out_channels=original_conv1.out_channels,
        kernel_size=original_conv1.kernel_size,
        stride=original_conv1.stride,
        padding=original_conv1.padding,
        bias=original_conv1.bias is not None,
    )
    with torch.no_grad():
        new_conv1.weight[:, :3] = original_conv1.weight[:, :3]
        if num_channels > 3:
            new_conv1.weight[:, 3:] = torch.randn_like(new_conv1.weight[:, 3:]) * 0.01
    model.conv1 = new_conv1
    feature_extractor = nn.Sequential(*list(model.children())[:-1])
    model.eval()
    with torch.no_grad():
        features = feature_extractor(dataset)
    # return features.view(features.size(0), -1).cpu().numpy()
    return features


def extract_features_resnet_rgb(dataset, num_channels, pretrained_weights=ResNet18_Weights.SENTINEL2_RGB_MOCO):
    model = resnet18(weights=pretrained_weights)
    dataset_rgb = dataset[:, :3, :, :]  #(batch_size, 10, height, width)
    feature_extractor = nn.Sequential(*list(model.children())[:-1])
    model.eval()
    with torch.no_grad():
        features = feature_extractor(dataset_rgb)  
    # return features.view(features.size(0), -1).cpu().numpy()
    return features
