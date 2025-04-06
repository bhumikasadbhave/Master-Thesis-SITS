from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Block
from model_scripts.model_visualiser import normalize_for_display
from  model_scripts.pos_embed import *
from torchvision.utils import save_image
import config
from torchvision import transforms

class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=64, patch_size=4, in_chans=3,
                 embed_dim=32, depth=4, num_heads=16,
                 decoder_embed_dim=64, decoder_depth=2, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, same_mask=False):
        super().__init__()

        # MAE encoder initialisations --------------------------------------------------------------------------
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.dec_count = 0

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim - 384), requires_grad=False)  # fixed sin-cos embedding

        # -- Comment transformer blocks just for now -- #
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)
        

        # MAE decoder initialisations --------------------------------------------------------------------------
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        # self.mask_token = nn.Parameter(torch.randn(1, 1, decoder_embed_dim) * 0.02)  # Small random initialization

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim - 192), requires_grad=False)  # fixed sin-cos embedding

        # -- Comment transformer blocks just for now -- #
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        
        # General Variables -----------------------------------------------------------------------------------
        self.norm_pix_loss = norm_pix_loss
        self.same_mask = same_mask
        self.initialize_weights()
        self.counter = 0


     # Initialize Weights -----------------------------------------------------------------------------------
    def initialize_weights(self):

        # Initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    # Patchify -----------------------------------------------------------------------------------
    def patchify(self, images):
        """ images: (N,C,H,W)
            patches(x): (N, L, patch_size ** 2 * C) 
            valid_patch_mask: (N, L)
        """

        N, C, H, W = images.shape
        # p = patch_size
        # print("inside patchify, patchsize=",self.patch_embed.patch_size)
        p = self.patch_embed.patch_size[0]

        assert H % p == 0 and W % p == 0, "Image dimensions must be divisible by patch_size"

        h = w = H // p
        x = images.reshape(shape = (N, C, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(N, h * w, p**2 * C))
        # print("After patchify=",x.shape)

        # Create valid patch mask (1 for non-zero patches, 0 for black patches)
        # A black patch is a patch with all zeros, so check if the sum of the patch is zero
        # valid_patch_mask = (x.sum(dim=-1) != 0).float()  # Shape: (B, num_patches)
        
        return x#, valid_patch_mask
    

    # Unpatchify -----------------------------------------------------------------------------------
    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 * C)
        imgs: (N, C, H, W)
        """
        N, L, O = x.shape
        # p = patch_size
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]

        C = O // p**2
        x = x.reshape(N, h, w, p, p, C)
        x = torch.einsum('nhwpqc->nchpwq', x)

        images = x.reshape(N, C, h*p, h*p)
        return images
    

    # Random Masking -----------------------------------------------------------------------------------
    def random_masking(self, x, mask_ratio, mask=None):
        """Perform random masking based on the mask ratio and track original indices."""
        
        N, L, D = x.shape 
        len_keep = int(L * (1 - mask_ratio)) 
        perm = torch.rand(N, L, device=x.device).argsort(dim=1)  # Random permutation of patch indices
        ids_keep = perm[:, :len_keep]  
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))  # Gather kept patches

        # Binary mask (1 = masked/removed, 0 = kept)
        mask = torch.ones([N, L], device=x.device)
        # print('inside random masking: mask=',mask.shape)
        mask[:, :len_keep] = 0 
        mask = torch.gather(mask, dim=1, index=perm)  # Restore the original order of the mask
        ids_restore = torch.argsort(perm, dim=1)  # The original indices to restore the shuffled order
        # print(ids_restore)

        return x_masked, mask, ids_restore
    

    def random_masking_with_valid_patch_exclusion(self, x, mask_ratio, valid_mask, mask=None):
        """Perform random masking but exclude invalid patches from masking based on valid_mask."""
        
        N, L, D = x.shape  # N = batch size, L = number of patches, D = patch size * channels
        len_keep = int(L * (1 - mask_ratio))  

        valid_indices = valid_mask.bool()  
        valid_patch_indices = torch.where(valid_indices)[1]  

        full_indices = torch.arange(L, device=x.device).repeat(N, 1)

        perm = torch.rand(N, len(valid_patch_indices), device=x.device).argsort(dim=1)  # Random permutation of valid patches
        ids_keep = valid_patch_indices[perm[:, :len_keep]]  # Select kept patches from valid patches
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))  # Gather kept patches
        
        # Generate the binary mask (1 = masked, 0 = kept)
        mask = torch.ones([N, L], device=x.device)  
        mask[:, :len_keep] = 0  
        mask = torch.gather(mask, dim=1, index=perm) 
        ids_restore = torch.argsort(torch.cat([ids_keep, valid_patch_indices[perm[:, len_keep:]]], dim=1), dim=1)
        
        return x_masked, mask, ids_restore

    
    # Encoder Forward -----------------------------------------------------------------------------------
    # def forward_encoder(self, x, timestamps, mask_ratio, valid_patch_mask, mask=None):
    def forward_encoder(self, x, timestamps, mask_ratio, mask=None):
        
        # Patch Embeddings for all 3 temporal images
        # print('encoder forward', x.shape)
        # print('encoder forward', x[:, 0].shape)
        print("INPUTS: before encoding:", x.min(), x.max())
        print(f"Input Tensor dtype: {x.dtype}")

        x1 = self.patch_embed(x[:, 0])
        x2 = self.patch_embed(x[:, 1])
        x3 = self.patch_embed(x[:, 2])
        x = torch.cat([x1, x2, x3], dim=1)
        
        # Temporal Embeddings
        # print('after patch embed 1st temporal image:', x1.shape)
        ts_embed = torch.cat([
            get_1d_sincos_pos_embed_from_grid_torch(128, timestamps.reshape(-1, 3)[:, 0].float()),
            get_1d_sincos_pos_embed_from_grid_torch(128, timestamps.reshape(-1, 3)[:, 1].float()),
            get_1d_sincos_pos_embed_from_grid_torch(128, timestamps.reshape(-1, 3)[:, 2].float())], dim=1).float()
        
        ts_embed = ts_embed.reshape(-1, 3, ts_embed.shape[-1]).unsqueeze(2)
        ts_embed = ts_embed.expand(-1, -1, x.shape[1] // 3, -1).reshape(x.shape[0], -1, ts_embed.shape[-1])

        # print('x before concat',x.shape)
        # print('ts before concat',x.shape)

        # Add positional embedding without cls token
        x = x + torch.cat([self.pos_embed[:, 1:, :].repeat(ts_embed.shape[0], 3, 1), ts_embed], dim=-1)
        # print('x after pos encoding',x.shape)

        # Masking: length -> length * mask_ratio
        x_masked, mask, ids_restore = self.random_masking(x, mask_ratio, mask=mask)
        # print('After random masking',x_masked.shape)
        # print('ids restore', ids_restore.shape)

        # if self.counter % 100 == 0:  # Visualize every 100 steps (optional, to avoid clutter)
        #     self.visualize_masked_images(x[0], x_masked[0], mask[0])
        # self.counter += 1

        x = x_masked

        # Append cls token
        cls_token = self.cls_token #+ self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        # print('x after adding cls token',x.shape)

        # Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x, mask, ids_restore


    # Decoder Forward -----------------------------------------------------------------------------------
    def forward_decoder(self, x, timestamps, ids_restore):

        print("LATENT: Latents after encoding:", x.min(), x.max())
        print(f"Latent Tensor dtype: {x.dtype}")
        # mean = x.mean()
        # std = x.std()
        # x = (x - mean) / std
        # x = (x - x.min()) / (x.max() - x.min())    #Normalise latent!?
        # print("LATENT: Latents after normalisation:", x.min(), x.max())
        
        # print("before decoder embed",x.shape)
        # Decoder Embeddings for tokens
        x = self.decoder_embed(x)
        # print("before adding mask tokens",x.shape)

        # Append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)                                       # No cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # Unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)                                                 # Append cls token
        # print("after adding mask tokens ",x.shape)

        assert timestamps.shape[-1] == 3, f"Expected timestamps with 3 channels, got {timestamps.shape}"

        ts_embed = torch.cat([get_1d_sincos_pos_embed_from_grid_torch(64, timestamps.reshape(-1, 3)[:, 0].float()),
                   get_1d_sincos_pos_embed_from_grid_torch(64, timestamps.reshape(-1, 3)[:, 1].float()),
                   get_1d_sincos_pos_embed_from_grid_torch(64, timestamps.reshape(-1, 3)[:, 2].float())], dim=1).float()
        
        # print("ts embedd shape before reshape",ts_embed.shape)
        # print("x before reshape",x.shape)
        
        ts_embed = ts_embed.reshape(-1, 3, ts_embed.shape[-1]).unsqueeze(2)
        ts_embed = ts_embed.expand(-1, -1, x.shape[1] // 3, -1).reshape(x.shape[0], -1, ts_embed.shape[-1])
        ts_embed = torch.cat([torch.zeros((ts_embed.shape[0], 1, ts_embed.shape[2]), device=ts_embed.device), ts_embed], dim=1)

        # print("ts embedd shape before concat", ts_embed.shape)
        # print("x before concat", x.shape)

        # Add positional embedding
        x = x + torch.cat(
            [torch.cat([self.decoder_pos_embed[:, :1, :], self.decoder_pos_embed[:, 1:, :].repeat(1, 3, 1)], dim=1).expand(ts_embed.shape[0], -1, -1),
             ts_embed], dim=-1)
        
        # print('x after pos embedding')

        # Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # Predictor projection
        x = self.decoder_pred(x)

        # print('x after decoder transformer blocks')

        # Remove cls token
        x = x[:, 1:, :]
        x = x.float()
        # print('x before decoder return',x.shape)
        print("RECONSTRUCTIONS: after decoding:", x.min(), x.max())
        print(f"Reconstruction Tensor dtype: {x.dtype}") 
        x = torch.sigmoid(x)
        print("RECONSTRUCTIONS: after sigmoid:", x.min(), x.max())

        self.dec_count+=x.max()>0.5
        return x
    

    # Loss Forward -----------------------------------------------------------------------------------
    def forward_loss(self, images, pred, mask):
        """
        images: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove
        """
        # target1, valid_mask1 = self.patchify(images[:, 0])
        # target2, valid_mask2 = self.patchify(images[:, 1])
        # target3, valid_mask3 = self.patchify(images[:, 2])
        target1 = self.patchify(images[:, 0])
        target2 = self.patchify(images[:, 1])
        target3 = self.patchify(images[:, 2])

        # print("x after patchify: target1",target1.shape)

        target = torch.cat([target1, target2, target3], dim=1)
        # valid_mask = torch.cat([valid_mask1, valid_mask2, valid_mask3], dim=1)

        previous_target = target
        
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
        

        self.visualize_images(pred, mask, previous_target, save_dir=config.mae_save_dir)

        # final_mask = valid_mask * mask
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)

        loss = (loss * mask).sum() / (mask.sum())  # mean loss on removed patches
        # loss = loss.mean(dim=1)
        # print("Loss requires grad:", loss.requires_grad)  
        # print("Pred requires grad:", pred.requires_grad)
        # print("Loss shape before returning:", loss.shape) 
        # print("Mask Shape:", mask.shape, "Valid Mask Shape:", valid_mask.shape)

        # Apply valid_patch_mask to ensure black patches don't contribute to loss
        # loss = (loss * final_mask).sum() / (final_mask).sum()
        # print("LOSS:",loss.mean())
        return loss


    # Forward function for main class -----------------------------------------------------------------------------------
    def forward(self, imgs, timestamps, mask_ratio=0.75, mask=None):


        # x1, valid_mask1 = self.patchify(imgs[:, 0])
        # x2, valid_mask2 = self.patchify(imgs[:, 1])
        # x3, valid_mask3 = self.patchify(imgs[:, 2])

        # print('forward', imgs.shape)

        # valid_patch_mask = torch.cat([valid_mask1, valid_mask2, valid_mask3], dim=1)
        # latent, mask, ids_restore = self.forward_encoder(imgs, timestamps, mask_ratio, valid_patch_mask, mask=mask)

        latent, mask, ids_restore = self.forward_encoder(imgs, timestamps, mask_ratio, mask=mask)
        pred = self.forward_decoder(latent, timestamps, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)
        print('DECODER COUNT',self.dec_count)
        # print("Loss requires grad in main forward:", loss.requires_grad)  
        # loss = self.forward_loss(imgs, pred, mask, valid_patch_mask)

        # print('forward', latent.shape)
        # print('forward', mask.shape)
        # print('forward', ids_restore.shape)

        return loss, pred, mask, latent



    def visualize_images(self, pred, mask, previous_target, mean=None, std=None, save_dir='viz1'):

        image=pred
        bs = image.shape[0]
        image = image.reshape(bs, 3, -1, image.shape[-1])[0]
        image = self.unpatchify(image).detach().cpu()
        print("Reconstructed min:", image.min().item())
        print("Reconstructed max:", image.max().item())

        save_image((image), save_dir + f'viz1/viz_{self.counter}.png')
        print('in viz, image',image.shape)
        masked_image = self.patchify(image)
        print('in viz, masked image',masked_image.shape)
        print('in viz, mask',mask.shape)
        masked_image.reshape(-1, 768)[mask[0].bool()] = 0.5
        masked_image = self.unpatchify(masked_image.reshape(3, -1 ,768))
        save_image((masked_image), save_dir + f'viz1/viz_mask_{self.counter}.png')

        previous_target = previous_target.reshape(bs, 3, -1, previous_target.shape[-1])[0]
        previous_target = self.unpatchify(previous_target).detach().cpu()
        previous_target = previous_target 
        save_image(normalize_for_display(previous_target), save_dir + f'viz1/target_{self.counter}.png')

        masked_image_target = self.patchify(previous_target)
        masked_image_target.reshape(-1, 768)[mask[0].bool()] = 0.5
        masked_image_target = self.unpatchify(masked_image_target.reshape(3, -1 ,768))
        save_image(normalize_for_display(masked_image_target), save_dir + f'viz1/viz_target_mask_{self.counter}.png')

        # if self.counter % 100 == 0:
        #     fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        #     for t in range(3):
        #         axes[t].imshow(normalize_for_display(image[t]).permute(1, 2, 0))  # Convert to HWC for plotting
        #         axes[t].set_title(f'Temporal Frame {t+1}')
        #         axes[t].axis('off')
        #     # plt.tight_layout()
        #     # plt.show()

        self.counter += 1









