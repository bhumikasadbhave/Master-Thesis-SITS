import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Block
from  model_scripts.pos_embed import *

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

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim - 384), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # MAE decoder initialisations --------------------------------------------------------------------------
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim - 192), requires_grad=False)  # fixed sin-cos embedding

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
        p = self.patch_embed.patch_size[0]

        assert H % p == 0 and W % p == 0, "Image dimensions must be divisible by patch_size"

        h = w = H // p
        x = images.reshape(shape = (N, C, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(N, h * w, p**2 * C))

        # Create valid patch mask (1 for non-zero patches, 0 for black patches)
        # A black patch is a patch with all zeros, so check if the sum of the patch is zero
        valid_patch_mask = (x.sum(dim=-1) != 0).float()  # Shape: (B, num_patches)
        return x , valid_patch_mask
    
    # Unpatchify -----------------------------------------------------------------------------------
    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 * C)
        imgs: (N, C, H, W)
        Dont need to handle valid_patch_mask because we wont be sending invalid patches to the encoder and decoder.. 
        So at the end we only have valid patches to unpatchify..
        """
        print(x.shape)
        N, L, O = x.shape
        # p = patch_size
        p = self.patch_embed.patch_size[0]
        print(p)
        h = w = int(x.shape[1]**.5)
        print(h,w)
        assert h * w == x.shape[1]

        C = O // p**2
        x = x.reshape(N, h, w, p, p, C)
        x = torch.einsum('nhwpqc->nchpwq', x)

        images = x.reshape(N, C, h*p, h*p)
        return images
    
    # Random Masking -----------------------------------------------------------------------------------
    def random_masking(self, x, mask_ratio, mask=None):
    # def random_masking(self, x, mask_ratio, valid_patch_mask, mask=None):
        """ Perform per-sample random masking by per-sample shuffling.
            Per-sample shuffiling is done by argsort random noise.
            x: [N,L,D], sequence
            Here, we completely discard the black patches, and use only valid patches as input for- masking and shuffing ..
        """

        N, L, D = x.shape  # Batch, length, p^2 * C
        len_keep = int(L * (1 - mask_ratio))

        # valid_patch_mask = valid_patch_mask.bool()
        # print(x.shape, valid_patch_mask.shape)          # [2425, 256, 48], [2425, 256]
        # Remove samples with no valid patches
        # valid_samples = valid_patch_mask.sum(dim=1) > 0  
        # x = x[valid_samples]
        # valid_patch_mask = valid_patch_mask[valid_samples]
        # print(x.shape, valid_patch_mask.shape)          # [2345, 256, 48], [2345, 256]

        # Update variables
        N = x.shape[0]  
        noise = torch.randn(N, L, device=x.device)  # Noise [0,1]

        if self.same_mask:
        # if True:

            while L % 4 != 0:
                L += 1

            L2 = L // 4             # 4 components (Original code had 3 components, but we keep 4)
            assert 4 * L2 == L
            noise = torch.randn(N, L2, device=x.device)  # Noise [0,1] for L2 tokens
            ids_shuffle = torch.argsort(noise, dim=1)    # Shuffling
            ids_shuffle = [ids_shuffle + i * L2 for i in range(3)]

            ids_shuffle_keep = [z[: ,:int(L2 * (1 - mask_ratio))] for z in ids_shuffle]     # To Keep
            ids_shuffle_disc = [z[: ,int(L2 * (1 - mask_ratio)):] for z in ids_shuffle]     # To Mask
            ids_shuffle = []
            for z in ids_shuffle_keep:
                ids_shuffle.append(z)
            for z in ids_shuffle_disc:
                ids_shuffle.append(z)
            ids_shuffle = torch.cat(ids_shuffle, dim=1)

        else:
            if mask is None:
                ids_shuffle = torch.argsort(noise, dim=1)   # if no mask, noise small=keep, noise large=remove
            else:
                ids_shuffle = mask
        
        ids_restore = torch.argsort(ids_shuffle, dim=1) # For restoring the unshuffled version

        ids_keep = ids_shuffle[:,:len_keep]             # Keep 1st subset
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1,1,D))

        # Binary mask: 0 = keep, 1 = remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)      # Unshuffle to get the binary mask

        # print('Inside random masking', x_masked.shape)
        return x_masked, mask, ids_restore
    
    # Encoder Forward -----------------------------------------------------------------------------------
    # def forward_encoder(self, x, timestamps, mask_ratio, valid_patch_mask, mask=None):
    def forward_encoder(self, x, timestamps, mask_ratio, mask=None):
        
        # Patch Embeddings for all 3 temporal images
        # print('encoder forward', x.shape)
        # print('encoder forward', x[:, 0].shape)
        # x1 = self.patch_embed(x[:, 0])
        # x2 = self.patch_embed(x[:, 1])
        # x3 = self.patch_embed(x[:, 2])
        x1 = self.patch_embed(x[:, :, 0])
        x2 = self.patch_embed(x[:, :, 1])
        x3 = self.patch_embed(x[:, :, 2])
        x = torch.cat([x1, x2, x3], dim=1)
        
        # Temporal Embeddings
        # print(timestamps.shape, x.shape)
        ts_embed = torch.cat([
            get_1d_sincos_pos_embed_from_grid_torch(128, timestamps.reshape(-1, 3)[:, 0].float()),
            get_1d_sincos_pos_embed_from_grid_torch(128, timestamps.reshape(-1, 3)[:, 1].float()),
            get_1d_sincos_pos_embed_from_grid_torch(128, timestamps.reshape(-1, 3)[:, 2].float())], dim=1).float()
        
        ts_embed = ts_embed.reshape(-1, 3, ts_embed.shape[-1]).unsqueeze(2)
        ts_embed = ts_embed.expand(-1, -1, x.shape[1] // 3, -1).reshape(x.shape[0], -1, ts_embed.shape[-1])

        # Add positional embedding without cls token
        x = x + torch.cat([self.pos_embed[:, 1:, :].repeat(ts_embed.shape[0], 3, 1), ts_embed], dim=-1)

        # Masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio, mask=mask)
        # print('After random masking',x.shape)

        # Append cls token
        cls_token = self.cls_token #+ self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        # print('Adding cls token',x.shape)

        # Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x, mask, ids_restore


    # Decoder Forward -----------------------------------------------------------------------------------
    def forward_decoder(self, x, timestamps, ids_restore):
        
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
        
        # print('x before decoder transformer blocks')

        # Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # Predictor projection
        x = self.decoder_pred(x)

        # print('x after decoder transformer blocks')

        # Remove cls token
        x = x[:, 1:, :]

        return x
    

    # Loss Forward -----------------------------------------------------------------------------------
    def forward_loss(self, images, pred, mask):
        """
        images: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove
        """
        target1, valid_mask1 = self.patchify(images[:, 0])
        target2, valid_mask2 = self.patchify(images[:, 1])
        target3, valid_mask3 = self.patchify(images[:, 2])
        print("target1",target1.shape)
        target = torch.cat([target1, target2, target3], dim=1)
        previous_target = target
        
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
        
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)                          # [N, L], mean loss per patch
        loss = (loss * mask).sum() / (mask.sum())  # mean loss on removed patches

        # print("Loss requires grad:", loss.requires_grad)  
        # print("Pred requires grad:", pred.requires_grad)
        # print("Loss shape before returning:", loss.shape) 

        # Apply valid_patch_mask to ensure black patches don't contribute to loss
        # loss = (loss * mask * valid_patch_mask).sum() / (mask * valid_patch_mask).sum()
        return loss


    # Forward function for main class -----------------------------------------------------------------------------------
    def forward(self, imgs, timestamps, mask_ratio=0.75, mask=None):

        # x1, valid_mask1 = self.patchify(imgs[:, 0])
        # x2, valid_mask2 = self.patchify(imgs[:, 1])
        # x3, valid_mask3 = self.patchify(imgs[:, 2])
        # x = torch.cat([x1, x2, x3], dim=1)

        # print('forward', imgs.shape)

        # valid_patch_mask = torch.cat([valid_mask1, valid_mask2, valid_mask3], dim=1)
        # latent, mask, ids_restore = self.forward_encoder(imgs, timestamps, mask_ratio, valid_patch_mask, mask=mask)

        latent, mask, ids_restore = self.forward_encoder(imgs, timestamps, mask_ratio, mask=mask)
        pred = self.forward_decoder(latent, timestamps, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)
        # print("Loss requires grad in main forward:", loss.requires_grad)  
        # loss = self.forward_loss(imgs, pred, mask, valid_patch_mask)

        # print('forward', latent.shape)
        # print('forward', mask.shape)
        # print('forward', ids_restore.shape)

        return loss, pred, mask, latent

    
