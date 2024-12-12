"""
explain_model.py

Description: Creates images for model explainability, where GradCAM is layered.
"""

# Standard libraries
import logging
import os
from collections import defaultdict

# Non-standard libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.io
from pytorch_grad_cam import AblationCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision.utils import make_grid
from torchvision.transforms import v2 as T
from tqdm import tqdm

# Custom libraries
from config import constants
from src.utils.data import viz_data
from src.utils.data.dataset import load_dataset_from_paths
from src.utils.model import load_model


################################################################################
#                                  Constants                                   #
################################################################################
# Configure logging
LOGGER = logging.getLogger(__name__)


################################################################################
#                                Main Functions                                #
################################################################################
def explain_binary_model_on_images(exp_name, img_paths, save_path, **kwargs):
    """
    Extract GradCAMs from a binary classification model for a list of images.

    Parameters
    ----------
    exp_name : str
        Experiment name
    img_paths : list of str
        List of paths to images
    save_path : str
        Location to save grid plot of GradCAMs
    """
    # Load hyperparameters
    exp_hparams = load_model.get_hyperparameters(exp_name=exp_name)

    # Load model
    model = load_model.load_pretrained_model(exp_name=exp_name)
    # Get last convolutional layer for model
    target_layers = [load_model.get_last_conv_layer(model)]
    # Set model to train
    model.train()

    # Create GradCAM object
    cam = AblationCAM(
        model=model,
        target_layers=target_layers
    )

    # Load images and create index to images
    idx_to_imgs = {
        1: load_images(img_paths, exp_hparams)
    }

    # Create GradCAMs (for positive class)
    overlayed_imgs = extract_gradcams(cam, idx_to_imgs)[1][1]
    # Convert images to torch tensors
    overlayed_imgs = convert_numpy_to_torch_image(np.array(overlayed_imgs))

    # Determine the number of rows/columns
    n = len(overlayed_imgs)
    nrows = int(np.ceil(np.sqrt(n)))

    # Create a grid of images
    grid = make_grid(overlayed_imgs, nrow=nrows, padding=1)

    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Plot the images
    plt.figure(figsize=(10, 10))
    plt.imshow(np.transpose(grid, (1, 2, 0)))
    plt.axis("off")
    plt.savefig(save_path, bbox_inches="tight")


def extract_gradcams(cam, idx_to_imgs):
    """
    Extract class activation maps (CAM) and overlay over images provided.

    Parameters
    ----------
    cam : pytorch_grad_cam.GradCAM
        GradCAM object loaded with a model
    idx_to_imgs : dict of (int, dict(str, images))
        Mapping of prediction index to example images (in array/tensor format)

    Returns
    -------
    tuple of (dict of (int, np.ndarray))
        First tuple: Mapping of prediction index to original images
        Second tuple: Mapping of prediction index to GradCAM overlayed image
    """
    # Accumulate original and overlayed images
    idx_to_orig_img = defaultdict(list)
    idx_to_overlayed_img = defaultdict(list)

    # For each label, create overlayed images
    for label_idx, type_to_imgs in tqdm(idx_to_imgs.items()):
        # Specify target (label) of interest
        targets = [ClassifierOutputTarget(label_idx)]

        imgs_array = type_to_imgs["array"]
        imgs_tensor = type_to_imgs["tensor"]

        # POST-PROCESSING: For each image, create GradCAM-overlayed image
        for i in range(len(imgs_array)):
            # Extract GradCAM
            cam_mask = cam(
                input_tensor=imgs_tensor[i].unsqueeze(0),
                targets=targets,
                aug_smooth=True,
                eigen_smooth=True,
            )

            # Store original image
            img_arr = imgs_array[i]
            idx_to_orig_img[label_idx].append(np.uint8(img_arr * 255))

            # Create overlayed image
            overlayed_img = show_cam_on_image(img_arr, cam_mask[0],
                                              use_rgb=True)
            idx_to_overlayed_img[label_idx].append(overlayed_img)

    return idx_to_orig_img, idx_to_overlayed_img


################################################################################
#                               Helper Functions                               #
################################################################################
def load_images(img_paths, exp_hparams):
    """
    Load a list of images with optional resizing according to hyperparameters.

    Parameters
    ----------
    img_paths : list of str
        List of paths to images
    exp_hparams : dict
        Experiment hyperparameters

    Returns
    -------
    dict with keys "array" and "tensor"
        Mapping of images loaded as numpy arrays and PyTorch tensors
    """
    dataset = load_dataset_from_paths(img_paths=img_paths, hparams=exp_hparams)

    # Load images as array and as a tensor
    accum_imgs = {
        "array": [],
        "tensor": []
    }
    for row_idx in range(len(dataset)):
        img_path = img_paths[row_idx]
        img = torchvision.io.read_image(img_path, mode="RGB").to(float)
        transform = T.Compose([
            T.Resize(exp_hparams.get("img_size", constants.IMG_SIZE)),
            T.ToDtype(torch.float32, scale=True),
        ])
        img = transform(img)
        img_arr = convert_torch_to_numpy_image(img)
        accum_imgs["array"].append(img_arr)
        accum_imgs["tensor"].append(dataset[row_idx][0])

    # Stack images
    accum_imgs["array"] = np.stack(accum_imgs["array"])
    accum_imgs["tensor"] = torch.stack(accum_imgs["tensor"])

    return accum_imgs


def convert_torch_to_numpy_image(img_tensor, swap_channels=True):
    """
    Convert a PyTorch tensor image to a numpy array with optional channel reordering.

    Parameters
    ----------
    img_tensor : torch.Tensor
        Image tensor to convert, with pixel values expected to be in the range [0, 255].
    swap_channels : bool, optional
        If True, swap the channel order from (C, H, W) to (H, W, C) for single images or
        from (N, C, H, W) to (N, H, W, C) for batches, where C is the number of channels,
        H is height, W is width, and N is the batch size.

    Returns
    -------
    np.ndarray
        Converted image as a numpy array, either with original or reordered channels.
    """
    # Scale between 0 and 1
    if img_tensor.max() > 1:
        assert img_tensor.max() <= 255, f"Image has values greater than 255! Max: {img.max()}"
        img_tensor = img_tensor.to(float) / 255.

    # Convert to RGB numpy array
    img_arr = img_tensor.numpy()

    # Early return, if not reordering channels
    if not swap_channels:
        return img_arr

    # Reorder channel dimension
    # CASE 1: Single RGB image
    if len(img_arr.shape) == 3 and img_arr.shape[0] == 3:
        img_arr = np.moveaxis(img_arr, 0, 2)
    # CASE 2: Batch size in first dimension
    elif len(img_arr.shape) == 4 and img_arr.shape[1] == 3:
        img_arr = np.moveaxis(img_arr, 1, 3)

    return img_arr


def convert_numpy_to_torch_image(img_arr):
    """
    Convert a numpy array image to a PyTorch tensor image.

    Parameters
    ----------
    img_arr : np.ndarray
        Image array to convert, with pixel values expected to be in the range [0, 255].

    Returns
    -------
    torch.Tensor
        Converted image as a PyTorch tensor, either with original or reordered channels, and
        with pixel values in the range [0, 1].
    """
    # Reorder channel dimension
    # CASE 1: Single RGB image
    if len(img_arr.shape) == 3 and img_arr.shape[2] == 3:
        img_arr = np.moveaxis(img_arr, 2, 0)
    # CASE 2: Batch size in first dimension
    elif len(img_arr.shape) == 4 and img_arr.shape[3] == 3:
        img_arr = np.moveaxis(img_arr, 3, 1)

    # Convert to PyTorch Tensor
    img = torch.from_numpy(img_arr)

    # Scale between 0 and 1
    if img.max() > 1:
        assert img.max() <= 255, f"Image has values greater than 255! Max: {img.max()}"
        img = img.float() / 255.

    return img


# TODO: Remove this
def gradcam_cardiomegaly_on_vindr_pcxr(exp_names, img_paths):
    exp_names = exp_names or [
        "exp_cardiomegaly-vindr_cxr-mixup-imb_sampler",
        "exp_cardiomegaly-nih_cxr18-mixup-imb_sampler",
        "exp_cardiomegaly-padchest-mixup-imb_sampler",
        "exp_cardiomegaly-chexbert-mixup-imb_sampler",
    ]
    img_paths = img_paths or ['data/cxr_datasets/vindr-pcxr/1.0.0/train_test_processed/079df7b5cc2619db2da3285aa2732fa0.png', 'data/cxr_datasets/vindr-pcxr/1.0.0/train_test_processed/730f67d531e4b6100aaf20515048ef41.png', 'data/cxr_datasets/vindr-pcxr/1.0.0/train_test_processed/e0031223022f8c9c44967fc5cea53a7e.png', 'data/cxr_datasets/vindr-pcxr/1.0.0/train_test_processed/8eca95331d0c860d4f58cfa5d2faa2a9.png', 'data/cxr_datasets/vindr-pcxr/1.0.0/train_test_processed/f1012c1aa14154662e3bc1f6820d7545.png', 'data/cxr_datasets/vindr-pcxr/1.0.0/train_test_processed/50540bfd11a301dea8e792d9251195e6.png', 'data/cxr_datasets/vindr-pcxr/1.0.0/train_test_processed/f68207cedf2233b8a0ee72dba6a1dbb7.png', 'data/cxr_datasets/vindr-pcxr/1.0.0/train_test_processed/2429aec16375094341e3b8ea4cccbd70.png', 'data/cxr_datasets/vindr-pcxr/1.0.0/train_test_processed/2d1da6825bc47ce57be0d50892fc7a02.png', 'data/cxr_datasets/vindr-pcxr/1.0.0/train_test_processed/14e052307736edee159d6a9f00dc79cd.png', 'data/cxr_datasets/vindr-pcxr/1.0.0/train_test_processed/1a3997f73e838b9f70d05cbd32e72d82.png', 'data/cxr_datasets/vindr-pcxr/1.0.0/train_test_processed/cdc3156616f87eff3c5caa446166bb46.png', 'data/cxr_datasets/vindr-pcxr/1.0.0/train_test_processed/e5ad906fafbb08b4a58ce81e59a5ddf6.png', 'data/cxr_datasets/vindr-pcxr/1.0.0/train_test_processed/7e0071896416fe849769454e9959230c.png', 'data/cxr_datasets/vindr-pcxr/1.0.0/train_test_processed/7272b3aa8487121c4e04f043c40d1258.png', 'data/cxr_datasets/vindr-pcxr/1.0.0/train_test_processed/98f09fd63ff4bf77116da6acc7955e83.png']

    for exp_name in tqdm(exp_names):
        train_dset = exp_name.split("-")[1]
        explain_binary_model_on_images(
            exp_name=exp_name,
            img_paths=img_paths,
            save_path=os.path.join(constants.DIR_FIGURES_CAM, f"vindr_pcxr (cardiomegaly, {train_dset}).png")
        )


if __name__ == "__main__":
    from fire import Fire
    Fire()
