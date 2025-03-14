from datetime import datetime
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
from matplotlib import lines


def draw_diseased_patches(temporal_images, x_y_coords, save_path="output/", patch_size=5):
    """ Funcion to draw map the diseased sub-patches onto the sugar-beet fields (patches)
    """
    os.makedirs(save_path, exist_ok=True)

    for img_idx in range(len(temporal_images)):
        img = temporal_images[img_idx][-1][:, :, :3]  # Extract first 3 channels (BGR)

        if isinstance(img, torch.Tensor):
            img_np = img.cpu().numpy()
        else: img_np = img

        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        else: img_np = img_np.astype(np.uint8)

        rgb_image = np.stack([img_np[..., 2], img_np[..., 1], img_np[..., 0]], axis=-1)  # Convert from BGR to RGB
        rgb_image = np.clip(rgb_image / np.max(rgb_image), 0, 1)                         # Normalize the image

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(rgb_image)
        ax.axis("off")  

        field_id_channel = temporal_images[img_idx][-1][:, :, -2]   # Field ID
        unique_field_ids = np.unique(field_id_channel)
        unique_field_ids = unique_field_ids[unique_field_ids != 0]

        if len(unique_field_ids) == 0:
            continue  

        field_id = str(int(unique_field_ids[0]))

        # Draw Rectangles with the help of spatial co-ordinates (input dictionary) 
        for coord_key, is_diseased in x_y_coords.items():
            coord_field_num, (x, y) = coord_key  
            if is_diseased == 1 and field_id in coord_field_num:
                rect_size = patch_size
                x_min = x - 1
                y_min = y - 1
                x_max = x_min + rect_size
                y_max = y_min + rect_size
                
                # Draw thin lines on each side of the rectangle
                ax.add_line(lines.Line2D([y_min, y_min], [x_min, x_max], color='red', linewidth=1))
                ax.add_line(lines.Line2D([y_min, y_max], [x_max, x_max], color='red', linewidth=1))
                ax.add_line(lines.Line2D([y_max, y_max], [x_max, x_min], color='red', linewidth=1))
                ax.add_line(lines.Line2D([y_max, y_min], [x_min, x_min], color='red', linewidth=1))

        # Save images
        save_filename = os.path.join(save_path, f"img_{field_id}.png")
        plt.savefig(save_filename, bbox_inches='tight', pad_inches=0, transparent=True)
        plt.close(fig)  

        print(f"Saved: {save_filename}")


def visualize_single_patch_temporal_rgb(patch, patch_coordinates, acquisition_dates, patch_size=5, num_of_timestamps=7):

    field_number, i, j = patch_coordinates  
    dates = acquisition_dates[field_number]
    fig, axes = plt.subplots(1, num_of_timestamps, figsize=(20, 5))
    fig.suptitle(f"Subpatch-level Temporal Stack Visualisation (RGB) (Field {int(float(field_number))} - Subpatch ({i},{j}))", fontsize=16)
    for t in range(num_of_timestamps):
        rgb_image = np.stack([patch[t, 0], patch[t, 1], patch[t, 2]], axis=-1)  
        min_val = np.min(rgb_image)
        max_val = np.max(rgb_image)
        if max_val > min_val:
            rgb_image = (rgb_image - min_val) / (max_val - min_val)  
        rgb_image = np.clip(rgb_image, 0, 1)  

        date = dates[t]
        if len(date) > 0:
            int_date = int(float(date)) # yyyymmdd.0
            year = int_date // 10000
            month = (int_date // 100) % 100
            day = int_date % 100
            acquisition_date = datetime(year, month, day).strftime("%Y-%m-%d")
        else:
            acquisition_date = "No Date"

        axes[t].imshow(rgb_image, cmap='viridis') 
        axes[t].set_title(f"{acquisition_date}", fontsize=10)
        # axes[t].axis("off")
    plt.tight_layout(rect=[0, 0, 1, 0.95])  
    plt.show()


def visualize_subpatches(image_data, field_numbers_test, patch_coordinates, field_index, patch_size=5):

    num_of_img_train, t, channels, height, width = image_data.shape
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    img = image_data[field_index, -1, 0].cpu().numpy()
    img_fn = field_numbers_test[field_index]
    ax.imshow(img, cmap='gray')
    ax.set_title(f"Subpatch-Patch Mapping")
    ax.axis("off")
    for field_number, i, j in patch_coordinates:
        if field_number == img_fn:
            rect = patches.Rectangle((j, i), patch_size, patch_size, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
    plt.show()
