from ast import List, Tuple
from ctypes import Union
import math
from typing import Optional
import warnings
import matplotlib.pyplot as plt
import torch


def plot_loss(train_loss, test_loss, title="Training vs Test Loss"):
    """
    Plots training and test loss over epochs.
    """
    epochs = list(range(1, len(train_loss) + 1))  

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label='Train Loss', linestyle='-')
    plt.plot(epochs, test_loss, label='Test Loss', linestyle='-')

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_loss_log_scale(train_loss, test_loss, title="Training vs Test Loss (Log Scale)"):
    """
    Plots training and test loss over epochs with a logarithmic scale on y-axis.
    """
    epochs = list(range(1, len(train_loss) + 1))  

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label='Train Loss', linestyle='-', color='blue')
    plt.plot(epochs, test_loss, label='Test Loss', linestyle='-', color='red')

    plt.yscale('log')  # Set log scale for y-axis
    plt.xlabel("Epochs")
    plt.ylabel("Loss (Log Scale)")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5)  # Grid for log scale
    plt.show()



def visualize_reconstructions(model, dataloader, device, num_images=5):
    model.eval()
    imgs_list, recon_list = [], []
    
    with torch.no_grad():
        for imgs, fn, timestamps in dataloader:
            imgs, timestamps = imgs.to(device), timestamps.to(device)
            _, pred, _, _ = model(imgs, timestamps)

            imgs_list.append(imgs.cpu())
            recon_list.append(model.unpatchify(pred).cpu())

            if len(imgs_list) * imgs.shape[0] >= num_images:
                break

    imgs = torch.cat(imgs_list, dim=0)[:num_images]
    recons = torch.cat(recon_list, dim=0)[:num_images]

    fig, axes = plt.subplots(num_images, 2, figsize=(8, 2 * num_images))
    for i in range(num_images):
        axes[i, 0].imshow(imgs[i].permute(1, 2, 0))  # Original
        axes[i, 1].imshow(recons[i].permute(1, 2, 0))  # Reconstruction
        axes[i, 0].axis('off')
        axes[i, 1].axis('off')
        axes[i, 0].set_title("Original")
        axes[i, 1].set_title("Reconstruction")
    plt.show()


# @torch.no_grad()
# def make_grid(
#     tensor: Union[torch.Tensor, List[torch.Tensor]],
#     nrow: int = 8,
#     padding: int = 2,
#     normalize: bool = False,
#     value_range: Optional[Tuple[int, int]] = None,
#     scale_each: bool = False,
#     pad_value: float = 0.0,
#     **kwargs,
# ) -> torch.Tensor:
#     """
#     Make a grid of images.

#     Args:
#         tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
#             or a list of images all of the same size.
#         nrow (int, optional): Number of images displayed in each row of the grid.
#             The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
#         padding (int, optional): amount of padding. Default: ``2``.
#         normalize (bool, optional): If True, shift the image to the range (0, 1),
#             by the min and max values specified by ``value_range``. Default: ``False``.
#         value_range (tuple, optional): tuple (min, max) where min and max are numbers,
#             then these numbers are used to normalize the image. By default, min and max
#             are computed from the tensor.
#         range (tuple. optional):
#             .. warning::
#                 This parameter was deprecated in ``0.12`` and will be removed in ``0.14``. Please use ``value_range``
#                 instead.
#         scale_each (bool, optional): If ``True``, scale each image in the batch of
#             images separately rather than the (min, max) over all images. Default: ``False``.
#         pad_value (float, optional): Value for the padded pixels. Default: ``0``.

#     Returns:
#         grid (Tensor): the tensor containing grid of images.
#     # """
#     # if not torch.jit.is_scripting() and not torch.jit.is_tracing():
#     #     _log_api_usage_once(make_grid)
#     if not torch.is_tensor(tensor):
#         if isinstance(tensor, list):
#             for t in tensor:
#                 if not torch.is_tensor(t):
#                     raise TypeError(f"tensor or list of tensors expected, got a list containing {type(t)}")
#         else:
#             raise TypeError(f"tensor or list of tensors expected, got {type(tensor)}")

#     if "range" in kwargs.keys():
#         warnings.warn(
#             "The parameter 'range' is deprecated since 0.12 and will be removed in 0.14. "
#             "Please use 'value_range' instead."
#         )
#         value_range = kwargs["range"]

#     # if list of tensors, convert to a 4D mini-batch Tensor
#     if isinstance(tensor, list):
#         tensor = torch.stack(tensor, dim=0)

#     if tensor.dim() == 2:  # single image H x W
#         tensor = tensor.unsqueeze(0)
#     if tensor.dim() == 3:  # single image
#         if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
#             tensor = torch.cat((tensor, tensor, tensor), 0)
#         tensor = tensor.unsqueeze(0)

#     if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
#         tensor = torch.cat((tensor, tensor, tensor), 1)

#     if normalize is True:
#         tensor = tensor.clone()  # avoid modifying tensor in-place
#         if value_range is not None and not isinstance(value_range, tuple):
#             raise TypeError("value_range has to be a tuple (min, max) if specified. min and max are numbers")

#         def norm_ip(img, low, high):
#             img.clamp_(min=low, max=high)
#             img.sub_(low).div_(max(high - low, 1e-5))

#         def norm_range(t, value_range):
#             if value_range is not None:
#                 norm_ip(t, value_range[0], value_range[1])
#             else:
#                 norm_ip(t, float(t.min()), float(t.max()))

#         if scale_each is True:
#             for t in tensor:  # loop over mini-batch dimension
#                 norm_range(t, value_range)
#         else:
#             norm_range(tensor, value_range)

#     if not isinstance(tensor, torch.Tensor):
#         raise TypeError("tensor should be of type torch.Tensor")
#     if tensor.size(0) == 1:
#         return tensor.squeeze(0)

#     # make the mini-batch of images into a grid
#     nmaps = tensor.size(0)
#     xmaps = min(nrow, nmaps)
#     ymaps = int(math.ceil(float(nmaps) / xmaps))
#     height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
#     num_channels = tensor.size(1)
#     grid = tensor.new_full((num_channels, height * ymaps + padding, width * xmaps + padding), pad_value)
#     k = 0
#     for y in range(ymaps):
#         for x in range(xmaps):
#             if k >= nmaps:
#                 break
#             # Tensor.copy_() is a valid method but seems to be missing from the stubs
#             # https://pytorch.org/docs/stable/tensors.html#torch.Tensor.copy_
#             grid.narrow(1, y * height + padding, height - padding).narrow(  # type: ignore[attr-defined]
#                 2, x * width + padding, width - padding
#             ).copy_(tensor[k])
#             k = k + 1
#     return grid