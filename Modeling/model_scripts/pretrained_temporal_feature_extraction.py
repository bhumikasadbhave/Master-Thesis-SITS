import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader
from einops import rearrange
from transformers import ViTModel, ViTConfig
from transformers import TimesformerModel
from transformers import AutoModel
from sklearn.decomposition import PCA
import timm
import cv2
import numpy as np
from skimage.feature import hog
from skimage import color


###### Classical CV techniques for feature extraction ######
# 1. SIFT features
def extract_sift_features(data):

    sift = cv2.SIFT_create()
    sift_features = []
    for i in range(data.shape[0]):  
        sample_features = []
        for t in range(data.shape[1]):  
            for c in range(data.shape[2]):  
                img = data[i, t, c]  
                img = np.uint8(img)     #uint8 for SIFT
                keypoints, descriptors = sift.detectAndCompute(img, None)
                if descriptors is not None:
                    sample_features.append(descriptors)
        
        # Flatten the list of features from all channels and time steps
        sample_features = np.vstack(sample_features) if sample_features else np.array([])
        sift_features.append(sample_features)
    return sift_features


# 2. HOG features
def extract_hog_features(data, pixels_per_cell=(2, 2), cells_per_block=(2, 2), visualize=False):

    hog_features = []
    # hog_images = []
    for i in range(data.shape[0]):  
        sample_features = []
        for t in range(data.shape[1]):  
            for c in range(data.shape[2]):  
                img = data[i, t, c] 
                img = np.uint8(img)  #uint8 for HOG
                feature = hog(img, pixels_per_cell=pixels_per_cell, 
                                         cells_per_block=cells_per_block, 
                                         visualize=visualize, 
                                         channel_axis=None)           
                sample_features.append(feature)
                # hog_images.append(hog_image)
        
        sample_features = np.concatenate(sample_features) if sample_features else np.array([])
        hog_features.append(sample_features)
    return hog_features


# 3. Channel-wise histogram features
def extract_channel_histograms(data, bins=256):
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


# 4. Feature reduction using PCA on the channel dimension of Sentinel-2 data
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



###### Pre-trained models for feature extraction #######
# 1. ResNet3D for Spatiotemporal Feature Extraction
class ResNet3DFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet3d = models.video.r3d_18(pretrained=True)
        self.resnet3d.fc = nn.Identity()

    def forward(self, x):
        x = x[:, :, :3, :, :]
        x = x.permute(0, 2, 1, 3, 4)
        return self.resnet3d(x)


# 2. Vision Transformer Feature Extractor
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


# 3. Spectral Sentinel-2 ViT with time as channels
class Sentinel2ViTFeatureExtractor(nn.Module):
    def __init__(self, time_steps, in_channels, model_name="duygu/sentinel2-vit"):
        super().__init__()
        self.time_steps = time_steps
        self.in_channels = in_channels
        self.total_channels = time_steps * in_channels

        # Load pretrained ViT config & model
        self.vit = ViTModel.from_pretrained(model_name)
        config = self.vit.config

        # Modify input embedding layer to accept more channels
        old_conv = self.vit.embeddings.patch_embeddings.projection      # Conv2d layer
        self.vit.embeddings.patch_embeddings.projection = nn.Conv2d(
            in_channels=self.total_channels,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding
        )

        # Reinitialize the new conv layer weights
        nn.init.kaiming_normal_(self.vit.embeddings.patch_embeddings.projection.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        b, t, c, h, w = x.shape
        x = x.view(b, t * c, h, w)          # Flatten time and channel dims -> (B, T×C, H, W)
        outputs = self.vit(pixel_values=x)
        return outputs.last_hidden_state    # (batch, seq_len, hidden_dim)



## remove? --------------------------------------------------------------------------------------

# Pretrained Timesformer
class PretrainedTimeSformerFeatureExtractor1(nn.Module):
    def __init__(self):
        super().__init__()
        self.timesformer = TimesformerModel.from_pretrained("facebook/timesformer-base-finetuned-k400")
        self.timesformer.config.use_cache = False  

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)   # (batch_size, seq_length, channels, height, width)
        inputs = {'pixel_values': x}
        # print(f"Inputs for TimeSformer: {x.shape}")  
        outputs = self.timesformer(**inputs)
        return outputs.last_hidden_state 


# earthformer
class PretrainedEarthformerFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.earthformer = AutoModel.from_pretrained("OpenClimateFix/earthformer-base")

    def forward(self, x):
        # input: (batch, time, channels, height, width)
        x = x.permute(0, 2, 1, 3, 4) 
        inputs = {'pixel_values': x}
        outputs = self.earthformer(**inputs)
        return outputs.last_hidden_state


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

