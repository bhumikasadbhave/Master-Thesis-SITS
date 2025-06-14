from Pipeline.temporal_preprocessing_pipeline import *
from evaluation_scripts.evaluation_helper import get_string_fielddata
from model_scripts.executions import compile_results_table_with_metrics, train_model_multiple_runs_with_metrics
from model_scripts.subpatch_extraction import non_overlapping_sliding_window_with_date_emb

# Autoencoder class
class Conv3DAutoencoder_Time_Addition(nn.Module):
    def __init__(self, in_channels, time_steps, latent_size, patch_size):
        super(Conv3DAutoencoder_Time_Addition, self).__init__()

        self.time_steps = time_steps
        self.in_channels = in_channels
        self.patch_size = patch_size

        # --- Encoder (3D Convolutions) ---
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1)

        # --- Fully Connected Latent Space ---
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * patch_size * patch_size * time_steps, 512)   
        self.fc2 = nn.Linear(512, latent_size)

        # --- Decoder (Fully Connected) ---
        self.fc3 = nn.Linear(latent_size, 512)
        self.fc4 = nn.Linear(512, 256 * patch_size * patch_size * time_steps)

        # --- 3D Deconvolutions (Transpose convolutions) ---
        self.unflatten = nn.Unflatten(1, (256, time_steps, patch_size, patch_size))
        self.deconv1 = nn.ConvTranspose3d(256, 128, kernel_size=3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose3d(128, 64, kernel_size=3, stride=1, padding=1)
        self.deconv3 = nn.ConvTranspose3d(64, in_channels, kernel_size=3, stride=1, padding=1)

        # --- Temporal embedding projection to match channels (needed for alignment) ---
        self.temb_proj = nn.Conv3d(2, in_channels, kernel_size=1)


    def forward(self, x, date_embeddings):

        # --- Date embedding processing ---
        # Convert the date embeddings to the shape (B, 2, 7, 4, 4)
        if not isinstance(date_embeddings, torch.Tensor):
            date_embeddings = torch.tensor(date_embeddings, dtype=torch.float32).to(x.device)    # Shape: (B, 7, 2)
        date_embeddings_tensor = date_embeddings.permute(0, 2, 1)                                # Shape: (B, 2, 7)
        date_embeddings_tensor = date_embeddings_tensor.unsqueeze(-1).unsqueeze(-1)                     # Shape: (B, 2, 7, 1, 1)
        date_embeddings_tensor = date_embeddings_tensor.expand(-1, -1, -1, x.shape[3], x.shape[4])      # Shape: (B, 2, 7, 4, 4)

        # Project the date embeddings to match the channels
        date_embeddings_tensor = self.temb_proj(date_embeddings_tensor)                                 # Shape: (B, 10, 7, 4, 4)
        # print('x shape before time embedding:',x.shape)
        # print('time embeddings:',date_embeddings_tensor.shape)
        
        # --- Add date embeddings to the input tensor ---
        x = x + date_embeddings_tensor                                                                  # Shape: (B, 10, 7, 4, 4)
        # print('x shape after time embedding',x.shape)
        
        # --- Encoder ---
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # --- Flatten and Fully Connected ---
        b, c, t, h, w = x.shape                 # (B, C, T, H, W)
        x = self.flatten(x)  
        x = F.relu(self.fc1(x))
        z = self.fc2(x)                         # Bottleneck    

        # --- Decoder ---
        x = F.relu(self.fc3(z))
        x = F.relu(self.fc4(x))

        # --- Reshape and 3D Deconvolutions ---
        x = self.unflatten(x)                   # (B, C, H, W, T)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x_reconstructed = self.deconv3(x)       # Reconstruction

        return z, x_reconstructed

# Preprocess the dara
def pre_process():
    # Pre-pocessing the data
    preprocessing_pipeline = PreProcessingPipelineTemporal()

    # create patches and pre-process them
    preprocessing_pipeline.run_temporal_patch_save_pipeline(type='train')
    preprocessing_pipeline.run_temporal_patch_save_pipeline(type='eval')
    field_numbers_train, acquisition_dates_train, date_emb_train, patch_tensor_train, images_visualisation_train = preprocessing_pipeline.get_processed_temporal_cubes('train', 'b10_add', method='sin-cos')
    field_numbers_eval, acquisition_dates_eval, date_emb_eval, patch_tensor_eval, images_visualisation_eval = preprocessing_pipeline.get_processed_temporal_cubes('eval', 'b10_add', method='sin-cos')

    # create subpatches
    train_subpatches, train_subpatch_coords, train_subpatch_date_emb = non_overlapping_sliding_window_with_date_emb(patch_tensor_train, field_numbers_train, date_emb_train, patch_size=config.subpatch_size)
    eval_subpatches, eval_subpatch_coords, eval_subpatch_date_emb = non_overlapping_sliding_window_with_date_emb(patch_tensor_eval, field_numbers_eval, date_emb_eval, patch_size=config.subpatch_size)
    train_coord_dataloader = get_string_fielddata(train_subpatch_coords)
    eval_coord_dataloader = get_string_fielddata(eval_subpatch_coords)

    # split the data and create data loaders
    train_subpatches_dl, test_subpatches, train_field_numbers, test_field_numbers, train_date_embeddings, test_date_embeddings = train_test_split(
        train_subpatches, train_coord_dataloader, train_subpatch_date_emb, test_size=1-config.ae_train_test_ratio, random_state=42
    )
    dataloader_train_add = create_data_loader_mae(train_subpatches_dl, train_field_numbers, train_date_embeddings, mae=False, batch_size=config.ae_batch_size, shuffle=True)
    dataloader_test_add = create_data_loader_mae(test_subpatches, test_field_numbers, test_date_embeddings, mae=False, batch_size=config.ae_batch_size, shuffle=False)
    dataloader_eval_add = create_data_loader_mae(eval_subpatches, eval_coord_dataloader, eval_subpatch_date_emb, mae=False, batch_size=config.ae_batch_size, shuffle=False)

    return dataloader_train_add, dataloader_test_add, dataloader_eval_add

# Train and evaluate
def train_eval(dataloader_train_add, dataloader_test_add, dataloader_eval_add):
    # Training and saving the results
    device = 'cuda'
    epochs = 50
    momentum=0.9
    lr = 0.001
    vae_lr=0.001
    latent_dim = 32
    channels = 10
    time_steps = 7
    optimizer = 'Adam'
    vae_optimizer = 'Adam'
    patch_size = config.subpatch_size

    model_names = ["3D_AE_script_run"]
    model_objs = [Conv3DAutoencoder_Time_Addition]  
    train_loss = {}
    test_loss = {}
    metrics = {}

    for name, obj in zip(model_names, model_objs):
        avg_train_loss, avg_test_loss, avg_metrics = train_model_multiple_runs_with_metrics(
            model_name=name,
            model_class=obj,
            dataloader_train=dataloader_train_add,
            dataloader_test=dataloader_test_add,
            dataloader_eval=dataloader_eval_add,
            channels=channels,
            timestamps=time_steps,
            epochs=epochs,
            optimizer=optimizer,
            lr=lr,
            vae_lr=vae_lr,
            vae_optimizer=vae_optimizer,
            momentum=momentum,
            device=device,
            config=config,
            output_dir=config.results_json_path
        )
        print("Model ",name," trained")

    # return results
    model_names = ["3D_AE_script_run"]
    df_loss, df_accuracy, df_recall, df_f1 = compile_results_table_with_metrics(model_names, output_dir=config.results_json_path)

    return df_accuracy, df_f1


def __main__():
    dataloader_train_add, dataloader_test_add, dataloader_eval_add = pre_process()
    df_accuracy, df_f1 = train_eval(dataloader_train_add, dataloader_test_add, dataloader_eval_add)
    return df_accuracy, df_f1

