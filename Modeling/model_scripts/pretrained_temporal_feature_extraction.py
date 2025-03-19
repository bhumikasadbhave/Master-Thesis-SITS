import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader

# ResNet3D for Spatiotemporal Feature Extraction
class ResNet3DFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet3d = models.video.r3d_18(pretrained=True)
        self.resnet3d.fc = nn.Identity()

    def forward(self, x):
        x = x[:, :, :3, :, :]
        x = x.permute(0, 2, 1, 3, 4)
        return self.resnet3d(x)


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


# 2. ConvLSTM for Spatiotemporal Feature Extraction -> Remove
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