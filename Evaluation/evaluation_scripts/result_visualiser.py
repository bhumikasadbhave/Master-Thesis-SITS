from datetime import datetime
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
from matplotlib import lines


def draw_diseased_patches(temporal_images, x_y_coords, save_path="output/", patch_size=4):
    """ Funcion to draw map the diseased sub-patches onto the sugar-beet fields (patches) on last timestamp image
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
        if np.max(rgb_image) != 0:
            rgb_image = np.clip(rgb_image / np.max(rgb_image), 0, 1)                         # Normalize the image

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(rgb_image)
        ax.axis("off")  

        field_id_channel = temporal_images[img_idx][-1][:, :, -2]   # Field ID
        unique_field_ids = np.unique(field_id_channel)
        unique_field_ids = unique_field_ids[unique_field_ids != 0]

        if len(unique_field_ids) == 0:
            continue  

        field_id = str(int(float(unique_field_ids[0])))

        # Draw Rectangles with the help of spatial co-ordinates (input dictionary) 
        for coord_key, is_diseased in x_y_coords.items():
            split_key = coord_key.split('_')
            x = int(split_key[-2])  
            y = int(split_key[-1]) 
            coord_field_nums = split_key[:-2]
            coord_field_nums = [str(int(float(field))) for field in split_key[:-2]]

            if is_diseased == 1 and field_id in coord_field_nums:
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



def draw_diseased_patches_temporal(temporal_images, x_y_coords, T=7, save_path="output/", patch_size=4):
    """Final Deliverable Images with diseased subpatches alongwith their entire temporal stack"""

    os.makedirs(save_path, exist_ok=True)
    for field_idx in range(len(temporal_images)):
        fig, axs = plt.subplots(1, T, figsize=(T*3, 3))

        for img_idx in range(len(temporal_images[0])):
            img = temporal_images[field_idx][img_idx][:, :, :3]

            if isinstance(img, torch.Tensor):
                img_np = img.cpu().numpy()
            else:
                img_np = img

            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
            else:
                img_np = img_np.astype(np.uint8)

            rgb_image = np.stack([img_np[..., 2], img_np[..., 1], img_np[..., 0]], axis=-1)

            if np.max(rgb_image) != 0:
                rgb_image = np.clip(rgb_image / np.max(rgb_image), 0, 1)

            axs[img_idx].imshow(rgb_image)
            axs[img_idx].axis("off")

            field_id_channel = temporal_images[field_idx][img_idx][:, :, -2]
            unique_field_ids = np.unique(field_id_channel)
            unique_field_ids = unique_field_ids[unique_field_ids != 0]

            if len(unique_field_ids) == 0:
                continue  

            field_id_image = str(int(float(unique_field_ids[0])))

            for coord_key, is_diseased in x_y_coords.items():
                split_key = coord_key.split('_')
                x = int(split_key[-2])
                y = int(split_key[-1])
                coord_field_nums = [str(int(float(field))) for field in split_key[:-2]]

                if is_diseased == 1 and field_id_image in coord_field_nums and img_idx==T-1:
                    # print('true....')
                    rect_size = patch_size
                    x_min = x - 1
                    y_min = y - 1
                    x_max = x_min + rect_size
                    y_max = y_min + rect_size

                    axs[img_idx].add_line(lines.Line2D([y_min, y_min], [x_min, x_max], color='red', linewidth=1))
                    axs[img_idx].add_line(lines.Line2D([y_min, y_max], [x_max, x_max], color='red', linewidth=1))
                    axs[img_idx].add_line(lines.Line2D([y_max, y_max], [x_max, x_min], color='red', linewidth=1))
                    axs[img_idx].add_line(lines.Line2D([y_max, y_min], [x_min, x_min], color='red', linewidth=1))

        save_filename = os.path.join(save_path, f"{field_id_image}.png")
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.savefig(save_filename, bbox_inches='tight', pad_inches=0, transparent=True)
        plt.close(fig)

        print(f"Saved: {save_filename}")




def visualize_single_patch_temporal_rgb(patch, patch_coordinates, acquisition_dates, patch_size=5, num_of_timestamps=7):
    """Visualise all temporal images for a single patch"""

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


def visualize_subpatches(image_data, field_numbers_test, patch_coordinates, field_index, patch_size=4):
    """Visualise patch with rectangles according to subpatch size"""

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
