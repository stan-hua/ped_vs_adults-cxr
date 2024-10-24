"""
viz_data.py

Description: Used to visualize data
"""

# Standard libraries
import logging
import os

# Non-standard libraries
import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from fire import Fire
from skimage import exposure
from tqdm import tqdm

# Custom libraries
from config import constants


################################################################################
#                                  Constants                                   #
################################################################################
LOGGER = logging.getLogger(__name__)


################################################################################
#                                  Functions                                   #
################################################################################
def compute_avg_healthy_image_by_age_group(dset, n=100):
    """
    Computes average image from a healthy patient.

    Notes
    -----
    All images in directory are assumed to have the same size.

    Parameters
    ----------
    dir_path : str
        Path to directory containing images
    n : int
        Number of images to sample
    """
    # Load metadata to identify image paths of healthy patients
    df_metadata = pd.read_csv(constants.DIR_METADATA_MAP[dset]["png"])
    # NOTE: Keep only healthy patients WITH age annotations
    df_metadata = df_metadata[df_metadata["No finding"].astype(bool)]

    # Create directory to save figures
    save_dir = os.path.join(constants.DIR_FIGURES_EDA, dset)
    os.makedirs(save_dir, exist_ok=True)

    # 0. Plot avg. image of healthy patient across ages
    # Sample n patients
    df_sample = df_metadata.sample(n=min(n, len(df_metadata)))
    # Compute average image for sample
    mean_image = compute_avg_image((df_sample["dirname"] + "/" + df_sample["filename"]).tolist())
    # Save image
    print(f"Computing average healthy image in `{dset}` over `{len(df_sample)}` images")
    save_path = os.path.join(save_dir, f"avg_healthy_patient.png")
    cv2.imwrite(save_path, mean_image)

    # Filter for healthy patients with age annotations (for next images)
    df_metadata = df_metadata.dropna(subset=["age"])

    # CASE 1: If Adult dataset, convert age to years
    age_bins = []
    if dset == "vindr_cxr":
        df_metadata["age_years"] = df_metadata["age"].map(lambda x: x.replace("Y", ""))
        df_metadata["age_years"] = df_metadata["age_years"].map(lambda x: int(x) if x.isdigit() and int(x) >= 18 else None)
        df_metadata = df_metadata.dropna(subset=["age_years"])
        age_bins = [18, 25, 30, 40, 60, 80]
    # CASE 2: If Pediatric dataset, parse age which is in months / years
    # NOTE: VinDr-PCXR should only have patients <= 10 years old
    elif dset == "vindr_pcxr":
        df_metadata["age_years"] = df_metadata["age"].map(lambda x: "000Y" if "M" in x else x)
        df_metadata["age_years"] = df_metadata["age_years"].map(lambda x: x.replace("Y", ""))
        df_metadata["age_years"] = df_metadata["age_years"].map(lambda x: int(x) if x.isdigit() and int(x) <= 10 else None)
        age_bins = [0, 2, 4, 6, 8, 10]
    else:
        raise ValueError(f"Invalid dataset: `{dset}`")

    # 1. Plot histogram of ages
    sns.histplot(data=df_metadata, x="age_years", kde=True, bins=30)
    plt.title("Age Distribution of Healthy Patients w/ Age Annotated")
    plt.xlabel("Age (in Years)")
    save_path = os.path.join(save_dir, "hist_age.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.clf()

    # 2. Plot the average image by age group
    saved_paths = []
    texts = []
    for idx in range(1, len(age_bins)):
        age_lower, age_upper = age_bins[idx - 1], age_bins[idx]
        mask = (df_metadata["age_years"] >= age_lower) & (df_metadata["age_years"] < age_upper)
        # Skip, if no patients in this age group
        if mask.sum() == 0:
            print(f"No patients in age group [{age_lower}, {age_upper})")
            continue

        # Sample n patients
        df_age_group = df_metadata[mask]
        df_age_group = df_age_group.sample(n=min(n, len(df_age_group)))

        # Compute average image for patients in age group
        img_paths = (df_age_group["dirname"] + "/" + df_age_group["filename"]).tolist()
        mean_image = compute_avg_image(img_paths)

        # Log
        print(f"Computing average healthy image in `{dset}` - Ages "
              f"[{age_lower}, {age_upper}) over `{len(img_paths)}` images")

        # Save image
        save_path = os.path.join(save_dir, f"avg_healthy_patient-[{age_lower}, {age_upper}).png")
        cv2.imwrite(save_path, mean_image)

        # Store for later GIF
        saved_paths.append(save_path)
        texts.append(f"[{age_lower}, {age_upper})")

    # Create a GIF with the ages
    gif_save_path = os.path.join(save_dir, "avg_healthy_patient-by_age.gif")
    create_gif(saved_paths, gif_save_path, texts, "Age (in Years)")


def create_gif(img_paths, save_path, texts=None, text_name="Label"):
    """
    Creates a GIF from a list of images, with a text label superimposed on
    each image. The label is drawn on a black background box at the bottom
    of each image.

    Parameters
    ----------
    img_paths : list of str
        List of paths to images to be included in the GIF.
    save_path : str
        Save GIF to path
    texts : list of str, optional
        List of text labels to superimpose on each image.
    text_name : str, optional
        Text label to use for the text labels, by default "Label".

    Returns
    -------
    list of np.ndarray
        List of images with labels added.
    """
    # Put images into a GIF in sequence
    images = []
    for idx, img_path in enumerate(img_paths):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_CUBIC)
        height, width = img.shape[0], img.shape[1]

        # Add text to image with a background box, if provided
        if texts:
            box_height = 45
            cv2.rectangle(img, (0, height-box_height), (width,height), (0,0,0), -1)
            # Calculate text size and position
            text = f"{text_name}: {texts[idx]}"
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 1
            thickness = 2
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            text_x = (width - text_size[0]) // 2
            text_y = height - (box_height - text_size[1]) // 2
            
            cv2.putText(img, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
        images.append(img)

    # Save GIF
    imageio.mimsave(save_path, images, fps=2, loop=0)


def compute_avg_image(img_paths):
    """
    Computes the average image from a list of image paths.

    Parameters
    ----------
    img_paths : list of str
        List of file paths to the images to be averaged.

    Returns
    -------
    np.ndarray
        The computed average image, with histogram equalization applied, as an 8-bit unsigned integer array.
    """
    # Sum up pixels across all images
    sum_image = None
    for img_path in tqdm(img_paths):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if sum_image is None:
            sum_image = np.zeros_like(img, np.float64)
        sum_image += img

    # Compute average image
    mean_image = sum_image / len(img_paths)
    mean_image = np.clip(mean_image, 0, 255)

    # Apply histogram equalization
    mean_image = (255 * exposure.equalize_hist(mean_image)).astype(np.uint8)

    return mean_image


################################################################################
#                                User Interface                                #
################################################################################
if __name__ == "__main__":
    Fire({
        "avg_healthy_image": compute_avg_healthy_image_by_age_group
    })
