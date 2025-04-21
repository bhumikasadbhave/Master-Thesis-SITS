import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader
from einops import rearrange
from transformers import ViTModel, ViTConfig
from torchgeo.models import resnet18, ResNet18_Weights
from torchvision.models.video import r3d_18, R3D_18_Weights
from transformers import TimesformerModel
from torchvision.models import resnet50
from transformers import AutoModel
import torch.nn.functional as F
from sklearn.decomposition import PCA
import timm
import cv2
import numpy as np
from skimage.feature import hog
from skimage import color


###### -------------------------- Classical techniques for feature extraction -------------------------- ######

# 1. Channel-wise histogram features
def extract_channel_histograms(data, bins=32):
    """ Returns: numpy array of shape (N, T, C * bins): Flattened histograms per time step and channel"""

    N, T, C, H, W = data.shape
    all_features = []
    for i in range(N):
        sample_feats = []
        for t in range(T):
            for c in range(C):
                channel_data = data[i, t, c] 
                valid_pixels = channel_data[channel_data != 0]
                hist, _ = np.histogram(valid_pixels, bins=bins, range=(1e-6, 1)) 
                hist = hist.astype(np.float32)  
                hist /= hist.sum()
                sample_feats.extend(hist.tolist()) 
        all_features.append(sample_feats)

    return np.array(all_features)


# 2. Feature reduction using PCA on the channel dimension of Sentinel-2 data
def pca_feature_extraction(data, n_components=3):
   
    N, T, C, H, W = data.shape
    reshaped = data.permute(0, 1, 3, 4, 2).reshape(-1, C) # Shape: (N*T*H*W, C)
    valid_mask = ~(reshaped == 0).all(axis=1)        # Remove zero pixels
    valid_pixels = reshaped[valid_mask]
    valid_pixels_np = valid_pixels.cpu().numpy()

    # Apply PCA across channels
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(valid_pixels_np)
    top_channel_indices = np.argsort(np.abs(pca.components_), axis=1)[:, ::-1]

    return pca, transformed, top_channel_indices



###### -------------------------- Pre-trained models for feature extraction -------------------------- #######

# 1. ResNet3D for Spatiotemporal Feature Extraction
class ResNet3DFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        weights = R3D_18_Weights.DEFAULT 
        self.resnet3d = r3d_18(weights=weights)
        self.resnet3d.fc = nn.Identity()

    def forward(self, x):
        x = x[:, :, :3, :, :]
        x = x.permute(0, 2, 1, 3, 4)
        return self.resnet3d(x)


# 2. Spectral Sentinel-2 Resnet-18
class SpectralSentinel2FeatureExtractor(nn.Module):
    def __init__(self, num_channels, pretrained_weights=ResNet18_Weights.SENTINEL2_ALL_MOCO):
        super().__init__()
        
        self.model = resnet18(weights=pretrained_weights)
        
        # Modify the first convolution layer to accept the specified number of channels
        original_conv1 = self.model.conv1
        new_conv1 = nn.Conv2d(
            in_channels=num_channels,
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias is not None,
        )
        
        with torch.no_grad():                                       #Reinitialize weights
            new_conv1.weight[:, :3] = original_conv1.weight[:, :3]
            if num_channels > 3:
                new_conv1.weight[:, 3:] = torch.randn_like(new_conv1.weight[:, 3:]) * 0.01
        self.model.conv1 = new_conv1
        
        # Remove the final fully connected layer to get features
        self.feature_extractor = nn.Sequential(*list(self.model.children())[:-1])
    
    def forward(self, x):
        return self.feature_extractor(x)


# 3. Vision Transformer Feature Extractor
class VisionTransformerExtractor(nn.Module):
    def __init__(self, pretrained_model_name="google/vit-base-patch16-224-in21k"):
        super().__init__()
        self.transformer = ViTModel.from_pretrained(pretrained_model_name)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = x[:, :3, :, :, :]
        b, c, t, h, w = x.shape
        x = x.reshape(b * t, c, h, w)                       # Flatten temporal axis
        x = self.transformer(x).last_hidden_state[:, 0, :]  # Extract CLS token
        x = x.view(b, t, -1)                                # Reshape back (batch, timesteps, features)
        x = self.pool(x.permute(0, 2, 1)).squeeze(-1)       # Pooling over time
        return x


## Utility functions --------------------------------------------------------------------------------------

def extract_features(model, dataloader, device):
    model.to(device)
    model.eval()
    features_list = []
    field_numbers_all = []
    with torch.no_grad():
        for batch_inputs, field_numbers in dataloader:
            batch_inputs = batch_inputs.to(device)
            features = model(batch_inputs)
            features_list.append(features.cpu())
            field_numbers_all.extend(field_numbers)
    return torch.cat(features_list, dim=0), field_numbers_all


def resize_images_transfer_learning(images, size=(224, 224)):
    N, T, C, H, W = images.shape
    resize_transform = transforms.Compose([
        transforms.Resize(size)
    ])
    resized_images = []
    for i in range(N):
        resized_images_t = []
        for t in range(T):
            image = images[i, t]
            # print(image.shape)
            image_pil = transforms.ToPILImage()(image)
            image_resized = resize_transform(image_pil)
            image_resized_tensor = transforms.ToTensor()(image_resized).unsqueeze(0)  # Convert back to tensor (C, H', W')
            resized_images_t.append(image_resized_tensor)
        resized_images.append(torch.cat(resized_images_t, dim=0))  # Concatenate across the time dimension (T)
    return torch.stack(resized_images, dim=0)


def resize_images_multichannel(images, size=(224, 224)):
    N, T, C, H, W = images.shape
    images = images.view(N * T, C, H, W)
    images_resized = F.interpolate(images, size=size, mode='bilinear', align_corners=False)
    return images_resized.view(N, T, C, size[0], size[1])


