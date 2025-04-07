import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datetime import datetime
import torch.optim as optim
import config

# ViT Block Definition
class ViTBlock(nn.Module):
    def __init__(self, dim, num_heads, ff_dim, dropout=0.1):
        super(ViTBlock, self).__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # x: (num_patches, batch_size, patch_dim)
        # mask: (batch_size, num_patches) - binary mask for valid patches

        # Multihead Attention
        attn_out, _ = self.attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feedforward Network
        ff_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x


class ViTAutoencoder(nn.Module):
    def __init__(self, patch_size, dim, num_heads, ff_dim, num_layers, num_patches, num_channels, max_val=106, dropout=0.1, max_len=512):
        super().__init__()
        # ------- Initialisations ---------------------------------------------------------------
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.num_channels = num_channels
        self.num_heads = num_heads
        self.max_len = max_len  # Maximum number of patches for positional encoding
        self.dim=dim

        # Patch Embedding Layer - using Linear Layer
        self.patch_size_flat = patch_size * patch_size * num_channels  # Flattened patch size (height * width * channels)
        self.patch_embedding = nn.Linear(self.patch_size_flat, dim)  # Linear layer to embed each patch
 
        # Create a series of ViT Blocks (Encoder)
        self.encoder_blocks = nn.ModuleList([ViTBlock(dim, num_heads, ff_dim, dropout) for _ in range(num_layers)])

        # Decoder (reverse of encoder - you can modify this to a proper decoder, e.g., transpose convs)
        self.decoder = nn.ConvTranspose2d(dim, num_channels, kernel_size=patch_size, stride=patch_size)

        # Positional Embedding Layer (Fixed Sinusoidal)
        self.positional_embedding = self.create_positional_encoding(dim, max_len)

        # Temporal Embedding (For Acquisition Dates)
        self.temporal_embedding = nn.Embedding(max_len, dim)  # Learned Temporal Embedding (can use sinusoidal too)


     # ------- Encodings ---------------------------------------------------------------
    def create_positional_encoding(self, dim, max_len):
        """Generate fixed sinusoidal positional encodings."""
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * -(np.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add batch dimension for broadcasting
        return pe

    def get_temporal_encoding(self, date_str, max_val=106):
        """ Compute a 1D temporal encoding for an acquisition date as a normalized scalar """
        date = datetime.strptime(date_str, "%Y%m%d.%f")  # Parse the date string into a datetime object
        ref = datetime.strptime('20190601.0', "%Y%m%d.%f")  # Parse the reference date string

        date_diff = (date - ref).days  # Get the difference in days
        return (date_diff / max_val)  # Normalize the date difference


     # ------- Forward ---------------------------------------------------------------
    def forward(self, x, acquisition_dates, mask=None):
        batch_size, T, C, H, W = x.shape
        
        # 1. Flatten the image into patches
        x = x.view(batch_size * T, C, H, W)
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(batch_size * T, -1, self.patch_size_flat)  # (batch_size * T, num_patches, patch_size_flat)
        patches = self.patch_embedding(patches)  # (batch_size * T, num_patches, dim)
        print('patches shape:', patches.shape)

        # 2. Temporal Embeddings (per sample, per timestep)
        temporal_embeddings = []
        for b in range(batch_size):
            sample_embeddings = []
            for t in range(T):
                acq_date = acquisition_dates[b][t]
                encoding =self.get_temporal_encoding(acq_date, max_val=106)  # (dim,)
                temporal_encoding = torch.tensor([encoding] * self.dim).float()
                sample_embeddings.append(temporal_encoding)
            sample_embeddings = torch.stack(sample_embeddings)
            temporal_embeddings.append(sample_embeddings)

        temporal_embeddings = torch.stack(temporal_embeddings)  # (batch_size, T, dim)
        temporal_embeddings = temporal_embeddings.unsqueeze(2).expand(-1, -1, int(self.num_patches), -1)  # (batch_size, T, num_patches, dim)
        temporal_embeddings = temporal_embeddings = temporal_embeddings.view(batch_size * T, int(self.num_patches), self.dim)
        print('temp embedding size:',temporal_embeddings.shape)
        
        # 3. Add temporal embeddings
        temporal_embeddings = temporal_embeddings.to('cuda')
        patches = patches + temporal_embeddings

        # 4. Add positional embeddings (shared across time)
        pos_embedding = self.positional_embedding[:, :self.num_patches].expand(batch_size, -1, -1)  # (batch_size, num_patches, dim)
        pos_embedding = pos_embedding.repeat_interleave(T, dim=0)  # (batch_size * T, num_patches, dim)
        
        pos_embedding = pos_embedding.to('cuda')
        patches = patches + pos_embedding

        # 5. Encoder blocks
        latent = patches
        mask = mask.view(batch_size, self.num_patches)
        mask = mask.unsqueeze(1)  # Add temporal dimension: (batch_size, T, num_patches)
        mask = mask.expand(-1, T, -1)
        # expanded_mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)  # Shape: (batch_size, num_heads, T, num_patches, num_patches)
        # expanded_mask = expanded_mask.contiguous().view(batch_size * self.num_heads, T, self.num_patches, self.num_patches)
        # print(expanded_mask.shape)
        # print(mask.shape)
        for block in self.encoder_blocks:
            latent = block(latent, mask)

        # 6. Decoder
        reconstructed = self.decoder(latent.view(batch_size * T, -1, H, W))
        return latent, reconstructed



# # Create a binary mask for valid and invalid patches based on the last timestep image and last channel.
# def create_mask_using_threshold(dataloader):
#     """ Create a binary mask for valid and invalid patches based on the last timestep image and the last channel."""

#     batch_masks = []
#     for batch_idx, x in enumerate(dataloader):
        
#         T, C, H, W = x.shape

#         last_timestep_image = x[-1, :, :, :]  # Shape: (C, H, W)
#         last_channel = last_timestep_image[-1, :, :]  # Shape: (H, W)
        
#         mask = (last_channel > 0).float()
#         flattened_mask = mask.view(-1) 
#         print('mask shape', flattened_mask.shape)
#         batch_masks.append(mask)#.view(-1))  # Flatten to match num_patches (batch_size, num_patches)

#     masks_tensor = torch.stack(batch_masks)   # Shape: (total_samples, num_patches)
#     return masks_tensor


def create_mask_using_threshold(dataloader, patch_size):
    """Create a binary mask for valid and invalid patches based on the sum of pixel values in each patch."""
    batch_masks = []

    for batch_idx, x in enumerate(dataloader):
        T, C, H, W = x.shape
        
        image = x[-1, -1, :, :]  #Taking the last timestep, last channel image (H, W)

        num_patches_h = H // patch_size
        num_patches_w = W // patch_size
        num_patches = num_patches_h * num_patches_w  

        mask = torch.zeros(num_patches)
        patch_idx = 0
        for h in range(0, H, patch_size):
            for w in range(0, W, patch_size):
                patch = image[h:h+patch_size, w:w+patch_size]  
                patch_sum = patch.sum() 

                # If the sum is greater than a threshold, mark the patch as valid (1), else invalid (0)
                if patch_sum > 0:
                    mask[patch_idx] = 1  # Mark as valid patch
                patch_idx += 1
        # print(np.unique(mask))
        batch_masks.append(mask)

    masks_tensor = torch.cat(batch_masks, dim=0)      #final shape: total_samples, num_patches)
    return masks_tensor




# Train model function
def train_model_vitae(model, train_dataloader, test_dataloader, patch_size=config.subpatch_size, epochs=10, optimizer='Adam', lr=0.001, momentum=0.9, weight_decay=0.01, device='mps'):
    """ Vanilla function to train the Autoencoder. This function includes the creation of valid/invalid masks for patches."""
    # Loss and optimizer
    criterion = nn.MSELoss(reduction='none')  # We'll calculate MSE loss per pixel and later apply the mask
    if optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    
    epoch_train_losses = []
    epoch_test_losses = []

    for epoch in range(epochs):
        model.train()  
        train_loss = 0.0
        for inputs_cpu, field_numbers, timestamps in train_dataloader:

            inputs = inputs_cpu.to(device)

            masks = create_mask_using_threshold(inputs, patch_size)  # Shape: (batch_size, num_patches)
            # if epoch == 0:
            #     print(masks[0])
            print(inputs[0].shape)
            print(timestamps[0])
            reconstructed = model(inputs, timestamps, mask=masks)

            reconstructed_flat = reconstructed.view(-1)
            inputs_flat = inputs.view(-1)
            masked_loss = (criterion(reconstructed_flat, inputs_flat) * masks.view(-1)).sum()  # Sum of the losses for valid patches

            # Backpropagation
            optimizer.zero_grad()
            masked_loss.backward()
            optimizer.step()

            train_loss += masked_loss.item()

        epoch_train_losses.append(train_loss / len(train_dataloader))
        
        # Evaluate on the test set
        model.eval()  
        test_loss = 0.0
        with torch.no_grad():  
            for inputs_cpu, field_numbers, timestamps in test_dataloader:
                inputs, timestamps = inputs_cpu.to(device), timestamps.to(device)

                masks = create_mask_using_threshold(inputs)  # Shape: (batch_size, num_patches)
                reconstructed = model(inputs, timestamps, mask=masks)

                reconstructed_flat = reconstructed.view(-1)
                inputs_flat = inputs.view(-1)
                masked_loss = (criterion(reconstructed_flat, inputs_flat) * masks.view(-1)).sum()  # Sum of the losses for valid patches

                test_loss += masked_loss.item()

        epoch_test_losses.append(test_loss / len(test_dataloader))
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss / len(train_dataloader):.6f}, Test Loss: {test_loss / len(test_dataloader):.6f}")
    
    return model, epoch_train_losses, epoch_test_losses
