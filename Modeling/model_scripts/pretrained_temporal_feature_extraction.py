import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader
from einops import rearrange
from transformers import ViTModel
from transformers import TimesformerModel
import timm

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


# 2. Earthformer Transformer Feature Extractor
class EarthformerFeatureExtractor(nn.Module):
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


# 3. Pretrained Timesformer
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



# x. ConvLSTM for Spatiotemporal Feature Extraction -> Remove
class ConvLSTMFeatureExtractor(nn.Module):
    def __init__(self, input_channels=10, hidden_dim=64, kernel_size=3, num_layers=2):
        super().__init__()
        self.convlstm = nn.LSTM(input_size=input_channels * 5 * 5, 
                                hidden_size=hidden_dim, 
                                num_layers=num_layers, 
                                batch_first=True)
    
    def forward(self, x):
        batch, time_steps, channels, height, width = x.shape
        x = x.view(batch, time_steps, -1)
        
        output, (hn, cn) = self.convlstm(x)
        return hn[-1]  # Return last hidden state as feature representation
    


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

