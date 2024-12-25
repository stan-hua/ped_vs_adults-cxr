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
import math
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from fire import Fire
from matplotlib import rc
from matplotlib.container import ErrorbarContainer
from skimage import exposure
from torchvision.utils import make_grid
from tqdm import tqdm

# Custom libraries
from config import constants
from . import utils


################################################################################
#                                  Constants                                   #
################################################################################
LOGGER = logging.getLogger(__name__)

# Random seed
SEED = 42


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
    df_metadata = pd.read_csv(constants.DIR_METADATA_MAP[dset]["image"])
    # NOTE: Keep only healthy patients WITH age annotations
    df_metadata = df_metadata[~df_metadata["Has Finding"].astype(bool)]

    # Create directory to save figures
    save_dir = os.path.join(constants.DIR_FIGURES_EDA, dset, "avg_images")
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
    df_metadata = utils.extract_age(dset, df_metadata, age_col="age")
    df_metadata = df_metadata.dropna(subset=["age_years"])

    # Specify age bins for later age sub-grouping
    # CASE 1: If Adult dataset, convert age to years
    age_bins = []
    if dset == "vindr_cxr":
        age_bins = [18, 25, 30, 40, 60, 80]
    # CASE 2: If Pediatric dataset
    elif dset == "vindr_pcxr":
        age_bins = list(range(12))

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


def sample_healthy_image_by_age_group(dset):
    """
    Samples 1 image from a healthy patient.

    Notes
    -----
    All images in directory are assumed to have the same size.

    Parameters
    ----------
    dir_path : str
        Path to directory containing images
    """
    # Load metadata to identify image paths of healthy patients
    df_metadata = pd.read_csv(constants.DIR_METADATA_MAP[dset]["image"])
    # NOTE: Keep only healthy patients WITH age annotations
    df_metadata = df_metadata[~df_metadata["Has Finding"].astype(bool)]

    # Create directory to save figures
    save_dir = os.path.join(constants.DIR_FIGURES_EDA, dset, "sampled_healthy_imgs")
    os.makedirs(save_dir, exist_ok=True)

    # Filter for healthy patients with age annotations (for next images)
    if "age_years" not in df_metadata.columns:
        df_metadata = utils.extract_age(dset, df_metadata)
    df_metadata = df_metadata.dropna(subset=["age_years"])

    # Specify age bins for later age sub-grouping
    age_bins = []
    # CASE 1: If Pediatric dataset
    if dset == "vindr_pcxr":
        age_bins = list(range(12))
    # CASE 2: If Adult dataset, convert age to years
    else:
        age_bins = [18, 25, 30, 40, 60, 80]

    # 1. Plot sampled image by age group
    for idx in range(1, len(age_bins)):
        age_lower, age_upper = age_bins[idx - 1], age_bins[idx]
        mask = (df_metadata["age_years"] >= age_lower) & (df_metadata["age_years"] < age_upper)
        # Skip, if no patients in this age group
        if mask.sum() == 0:
            print(f"No patients in age group [{age_lower}, {age_upper})")
            continue

        # Sample 1 patient
        df_age_group = df_metadata[mask]
        df_age_group = df_age_group.sample(n=1, random_state=SEED)
        img_paths = (df_age_group["dirname"] + "/" + df_age_group["filename"]).tolist()
        curr_age = df_age_group["age_years"].iloc[0]

        # Load image
        sampled_img = cv2.imread(img_paths[0], cv2.IMREAD_COLOR)
        # Apply histogram equalization
        sampled_img = (255 * exposure.equalize_hist(sampled_img)).astype(np.uint8)

        # Save image
        save_path = os.path.join(save_dir, f"sampled_healthy_patient-({curr_age})-[{age_lower}, {age_upper}).png")
        cv2.imwrite(save_path, sampled_img)


def plot_age_histograms(*dsets, peds=False):
    """
    Plots age histograms for chosen datasets.

    Parameters
    ----------
    dset : *args
        List of dataset name/s to plot age distribution.
    peds : bool, optional
        If True, filter adult datasets for peds data
    """
    # Sort datasets by reverse alphabetical order
    dsets = sorted(dsets, reverse=True)

    # Load metadata to identify image paths of healthy patients
    accum_metadata = []
    hue_order = []
    for dset in dsets:
        df_curr_metadata = pd.read_csv(constants.DIR_METADATA_MAP[dset]["image"])
        df_curr_metadata["Dataset"] = utils.stringify_dataset_split(dset)
        hue_order.append(utils.stringify_dataset_split(dset))
        if "age_years" not in df_curr_metadata.columns:
            df_curr_metadata = utils.extract_age(dset, df_curr_metadata)

        # Drop all patients without age
        df_curr_metadata = df_curr_metadata.dropna(subset=["age_years"])

        # If peds, filter for all pediatric data
        split = None
        if peds:
            df_curr_metadata = df_curr_metadata[df_curr_metadata["age_years"] < 18]
            # NOTE: Mark split as "peds" to get appropriate peds age bins
            split = "test_peds"

        # Determine age bins
        df_curr_metadata["age_bin"] = utils.get_age_bins(
            df_curr_metadata, dset, split,
            include_peds=True
        )

        accum_metadata.append(df_curr_metadata)
    df_metadata = pd.concat(accum_metadata, ignore_index=True)

    # Determine if all age bins are scalar
    is_age_scalar = df_metadata["age_bin"].str.isnumeric().all()
    xlabel = "Age (in Years)" if is_age_scalar else "Age Bin (in Years)"
    # Drop any NaNs at this point
    df_metadata = df_metadata[df_metadata["age_bin"] != "nan"]
    # Convert to integer if it's an age, and string if age bin
    if is_age_scalar:
        df_metadata["age_bin"] = df_metadata["age_bin"].astype(int)

    # Sort by age bin
    df_metadata = df_metadata.sort_values("age_bin")

    # Convert age bin to string
    df_metadata["age_bin"] = df_metadata["age_bin"].astype(str)

    # Create title
    title = "Age Distribution Across Datasets"
    if len(dsets) == 1:
        title = f"Age Distribution for {utils.stringify_dataset_split(dsets[0])}"
    # Get color for each dataset
    palette = get_color_for_dsets(*dsets)

    # Filename prefix
    fname_prefix = "peds-" if peds or "vindr_pcxr" in dsets else "adult-"

    # Choose y-axis upper limit
    # NOTE: Special case when showing peds data in NIH and PadChest
    y_lim_upper = 20 if peds and set(dsets) == set(["nih_cxr18", "padchest"]) else 50

    # Create histogram plot
    figsize = (12, 8)
    set_theme(figsize=figsize, tick_scale=2)
    catplot(
        df_metadata, x="age_bin", hue="Dataset",
        plot_type="hist", exclude_bar_labels=True,
        stat="percent", multiple="dodge", common_norm=False, discrete=True,
        shrink=0.85 if len(dsets) > 1 else 1,
        xlabel=xlabel, ylabel="Percentage of Patients",
        title="Age Distribution",
        hue_order=hue_order,
        palette=palette,
        # y_lim=(0, y_lim_upper),
        legend=False,
        save_dir=constants.DIR_FIGURES_EDA,
        save_fname=f"{fname_prefix}age_histogram ({','.join(dsets)}).svg",
    )


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
        Text label to use for the text labels, By default "Cardiomegaly".

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


def plot_dataset_samples(
        dataset, dset, num_samples=25,
        shuffle=True,
        save_dir=constants.DIR_FIGURES_EDA,
        ext="svg",
    ):
    """
    Plots a grid of images from a given dataset.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        The dataset to sample images from
    dset : str
        The name of the dataset, for file naming purposes
    num_samples : int, optional
        The number of samples to draw from the dataset, by default 25
    shuffle : bool, optional
        If True, shuffle dataloaders
    save_dir : str, optional
        The directory to save the figure in, by default constants.DIR_FIGURES_EDA
    ext : str, optional
        File extension to save image as
    """
    # Create a DataLoader to sample images
    loader = torch.utils.data.DataLoader(dataset, batch_size=num_samples, shuffle=shuffle)
    # Get a batch of images
    images, _ = next(iter(loader))

    # Create a grid of images
    grid = make_grid(images, nrow=int(np.sqrt(len(images))), padding=2)
    # Convert the grid to a numpy array and transpose the dimensions
    np_grid = grid.numpy().transpose((1, 2, 0))

    # Plot the images
    plt.figure(figsize=(10, 10))
    plt.imshow(np_grid)
    plt.axis("off")

    # Save figure, if path provided
    if save_dir:
        curr_save_dir = os.path.join(save_dir, dset)
        os.makedirs(curr_save_dir, exist_ok=True)
        plt.savefig(
            os.path.join(curr_save_dir, f"{dset}-sampled_imgs.{ext}"),
            bbox_inches="tight"
        )


def plot_pixel_histograms(
        df_metadata,
        path_cols=("dirname", "filename"),
        dset_col="dset",
        num_samples=25,
        save_dir=constants.DIR_FIGURES_EDA,
        ext="svg",
    ):
    """
    Plot pixel histogram across datasets

    Parameters
    ----------
    df_metadata : pd.DataFrame
        Metadata table containing image paths for different datasets
    path_cols : list of str, optional
        Name of columns to combine to create full image paths
    dset_col : str, optional
        Name of column containing dataset name, by default "dset"
    num_samples : int, optional
        The number of samples to draw from each dataset
    save_dir : str, optional
        The directory to save the figure in, by default constants.DIR_FIGURES_EDA
    ext : str, optional
        File extension to save image as
    """
    # Ensure path columns are a list
    path_cols = list(path_cols)

    # Choose images to compute histogram with
    df_samples = df_metadata.groupby(dset_col).sample(num_samples, random_state=SEED)

    # Assign each training dataset a color
    dsets = df_metadata[dset_col].unique()
    dsets = sorted(dsets, reverse=True)
    dset_colors = get_color_for_dsets(*dsets)

    # For each dataset, plot bar plot of performance on its healthy adults
    figsize = (12, 8)
    set_theme(figsize=figsize, tick_scale=2)
    fig, axs = plt.subplots(
        ncols=min(2, len(dsets)), nrows=math.ceil(len(dsets)/2),
        sharex=True, sharey=False,
        figsize=figsize,
        dpi=300,
        constrained_layout=True
    )
    # NOTE: Flatten axes for easier indexing
    axs = [curr_ax for group_ax in axs for curr_ax in group_ax] if len(dsets) > 1 else [axs]
    for idx, dset in enumerate(dsets):
        ax = axs[idx]
        dset_color = dset_colors[idx]

        # Create image paths
        img_paths = df_samples[df_samples[dset_col] == dset].apply(
            lambda row: os.path.join(*row[path_cols].tolist()),
            axis=1,
        ).tolist()

        # Compute pixel histogram incrementally
        load_img_kwargs = {"img_mode": 1, "as_numpy": True}
        pixel_histogram = utils.compute_histogram_incrementally(img_paths, **load_img_kwargs)
        df_pixel = pd.DataFrame({
            "Pixel Intensity": np.arange(len(pixel_histogram)),
            "Count": pixel_histogram,
        })

        # Get dataset color
        color = get_color_for_dsets(dset)[0]

        # Create pixel histogram plot
        catplot(
            df=df_pixel, x="Pixel Intensity", y="Count",
            plot_type="bar",
            linewidth=0, width=1.05,
            color=dset_color, saturation=1,
            xlabel="", ylabel="",
            x_lim=(0, 255),
            legend=False,
            ax=ax
        )
        # Draw a line on top to smoothen the histogram
        numplot(
            df=df_pixel, x="Pixel Intensity", y="Count",
            plot_type="line",
            linewidth=2,
            color=dset_color,
            xlabel="", ylabel="",
            legend=False,
            ax=ax
        )

        # Set x-axis ticks to specific values
        plt.xticks([0, 50, 100, 150, 200, 255])

        # Remove the y axis
        ax.spines['left'].set_visible(False)
        ax.yaxis.set_ticks([])

    # Add shared x and y labels
    fig.supxlabel("Intensity")

    # Add title
    fig.suptitle(f"Pixel Histogram")

    # Save figure
    if save_dir:
        save_fname = f"pixel_histogram ({','.join(dsets)}).svg"
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, save_fname), bbox_inches="tight")

        # Clear figure, after saving
        plt.clf()
        plt.close()


def create_all_dset_legend():
    """
    Create legend, containing all datasets.
    """
    dsets = ["vindr_pcxr", "vindr_cxr", "padchest", "nih_cxr18", "chexbert"]
    dset_colors = dict(zip(dsets, get_color_for_dsets(*dsets)))

    # Create figure
    set_theme(tick_scale=1.8, figsize=(10, 5))
    fig = plt.figure()
    plt.axis("off")

    # Create custom legend at the bottom
    legend_handles = [
        mpatches.Patch(color=curr_color, label=utils.stringify_dataset_split(dset))
        for dset, curr_color in dset_colors.items()
    ]
    fig.legend(
        handles=legend_handles,
        ncol=len(legend_handles),
        title="Datasets",
    )

    # Save figure
    fig.savefig(
        os.path.join(constants.DIR_FIGURES_EDA, "all-dset-legend.svg"),
        bbox_inches="tight"
    )
    plt.close()


################################################################################
#                              Plotting Functions                              #
################################################################################
def set_theme(tick_scale=1.3, figsize=(10, 6)):
    """
    Create scientific theme for plot
    """
    custom_params = {
        "axes.spines.right": False, "axes.spines.top": False,
        "figure.figsize": figsize,
    }
    sns.set_theme(style="ticks", font_scale=tick_scale, rc=custom_params)


def catplot(
        df, x=None, y=None, hue=None,
        bar_labels=None, exclude_bar_labels=False,
        palette="colorblind",
        plot_type="bar",
        figsize=None,
        title=None, title_size=None,
        xlabel=None,
        ylabel=None,
        x_lim=None,
        y_lim=None,
        tick_params=None,
        legend=False,
        save_dir=None,
        save_fname=None,
        **extra_plot_kwargs,
    ):
    """
    Creates a categorical plot. One of bar/count/pie

    Note
    ----
    For bar plots, has option to add label on bar plot

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data for the bar plot.
    x : str
        Column name for the x-axis variable.
    y : str
        Column name for the y-axis variable.
    hue : str, optional
        Column name for grouping variable that will produce bars with different colors, by default None.
    bar_labels : list, optional
        List of text to place on top of each bar, by default None.
    exclude_bar_labels : bool, optional
        If True, exclude bar labels, by default None.
    palette : list, optional
        List of colors to use for the bars, by default None.
    plot_type : str, optional
        Type of plot to create, by default "bar".
    figsize : tuple, optional
        Tuple specifying the figure size, by default None.
    title : str, optional
        Title for the plot, by default None.
    title_size : int, optional
        Font size for title, by default 17
    xlabel : str, optional
        Label for the x-axis, by default None.
    ylabel : str, optional
        Label for the y-axis, by default None.
    x_lim : tuple, optional
        Tuple specifying the x-axis limits, by default None.
    y_lim : tuple, optional
        Tuple specifying the y-axis limits, by default None.
    tick_params : dict, optional
        Dictionary specifying the tick parameters, by default None
    legend : bool, optional
        Whether to include a legend, by default False.
    save_dir : str, optional
        Directory to save the plot, by default None.
    save_fname : str, optional
        Filename to save the plot, by default None.
    extra_plot_kwargs : dict, optional
        Additional keyword arguments to pass to the plot function, by default {}

    Returns
    -------
    matplotlib.axes.Axes
        Returns the Axes object with the plot for further tweaking.
    """
    # NOTE: If palette and colors provided, always chose color
    if palette and (extra_plot_kwargs.get("color") or extra_plot_kwargs.get("colors")):
        palette = None

    # Create plot keyword arguments
    plot_kwargs = {
        "data": df, "x": x, "y": y, "hue": hue,
        "legend": legend,
        "palette": palette,
        **extra_plot_kwargs,
    }
    # Add colors, if not seaborn function
    if plot_type in ["bar_with_ci", "grouped_bar_with_ci", "pie"]:
        color_key = "colors" if plot_type == "pie" else "color"
        if color_key not in plot_kwargs:
            plot_kwargs[color_key] = sns.color_palette()
            if palette:
                plot_kwargs[color_key] = sns.color_palette(palette)
        plot_kwargs.pop("palette", None)

    # Raise error, if plot type is invalid
    supported_types = ["bar", "bar_with_ci", "grouped_bar_with_ci", "count", "pie", "hist", "kde"]
    if plot_type not in supported_types:
        raise ValueError(f"Invalid plot type: `{plot_type}`! See supported types: {supported_types}")

    # Create plot based on requested function
    plot_func = None
    if plot_type == "bar":
        plot_func = sns.barplot
        add_default_dict_vals(plot_kwargs, width=0.95)
    elif plot_type == "bar_with_ci":
        plot_func = barplot_with_ci
        add_default_dict_vals(plot_kwargs, width=0.95)
        remove_dict_keys(plot_kwargs, ["hue", "legend"])
    elif plot_type == "grouped_bar_with_ci":
        plot_func = grouped_barplot_with_ci
        remove_dict_keys(plot_kwargs, ["legend"])
    elif plot_type == "count":
        plot_func = sns.countplot
        add_default_dict_vals(plot_kwargs, width=0.95)
    elif plot_type == "hist":
        plot_func = sns.histplot
    elif plot_type == "kde":
        plot_func = sns.kdeplot
    elif plot_type == "pie":
        assert x, "Must specify x-axis variable for pie plot"

        # Remove incompatible keys
        remove_dict_keys(plot_kwargs, ["data", "y", "hue", "legend", "palette"])
        # Add defaults
        add_default_dict_vals(
            plot_kwargs, autopct="%1.1f%%", startangle=140,
            radius=0.5, shadow=True
        )

        # Count the occurrences of each category
        counts = df[x].value_counts()
        plot_kwargs["x"] = counts
        plot_kwargs["labels"] = counts.index

        # Create the pie chart
        plot_func = plt.pie

    # Create figure
    if figsize is not None:
        plt.figure(figsize=figsize)

    # Create plot
    ax = plot_func(**plot_kwargs)

    # Add bar labels
    if not exclude_bar_labels and (bar_labels or plot_type == "count"):
        bar_kwargs = {"labels": bar_labels}
        if plot_type == "count":
            bar_kwargs.pop("labels")

        # Add bar labels
        for container in ax.containers:
            if container is None:
                print("Bar encountered that is empty! Can't place label...")
                continue
            if isinstance(container, ErrorbarContainer):
                print("Skipping labeling of error bar container...")
                continue
            ax.bar_label(container, size=12, weight="semibold", **bar_kwargs)

    # Perform post-plot logic
    ax = post_plot_logic(
        ax=ax,
        title=title, title_size=title_size,
        xlabel=xlabel, ylabel=ylabel,
        x_lim=x_lim, y_lim=y_lim,
        tick_params=tick_params,
        legend=legend and plot_type not in ["hist", "kde"],
        save_dir=save_dir, save_fname=save_fname,
    )


def numplot(
        df, x=None, y=None, hue=None,
        palette=None,
        plot_type="box",
        vertical_lines=None,
        figsize=None,
        title=None, title_size=None,
        xlabel=None,
        ylabel=None,
        x_lim=None,
        y_lim=None,
        tick_params=None,
        legend=False,
        save_dir=None,
        save_fname=None,
        **extra_plot_kwargs,
    ):
    """
    Creates a numeric plot.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data for the bar plot.
    x : str
        Column name for the x-axis variable.
    y : str
        Column name for the y-axis variable.
    hue : str, optional
        Column name for grouping variable that will produce bars with different colors, by default None.
    palette : list, optional
        List of colors to use for the bars, by default None.
    plot_type : str, optional
        Type of plot to create, by default "box".
    vertical_lines : list, optional
        List of x-axis values to draw vertical lines at, by default None.
    figsize : tuple, optional
        Tuple specifying the figure size, by default None.
    title : str, optional
        Title for the plot, by default None.
    title_size : int, optional
        Font size for title, by default 17
    xlabel : str, optional
        Label for the x-axis, by default None.
    ylabel : str, optional
        Label for the y-axis, by default None.
    x_lim : tuple, optional
        Tuple specifying the x-axis limits, by default None.
    y_lim : tuple, optional
        Tuple specifying the y-axis limits, by default None.
    tick_params : dict, optional
        Keyword arguments to pass into `matplotlib.pyplot.tick_params`, by default None
    legend : bool, optional
        Whether to include a legend, by default False.
    save_dir : str, optional
        Directory to save the plot, by default None.
    save_fname : str, optional
        Filename to save the plot, by default None.
    extra_plot_kwargs : dict, optional
        Additional keyword arguments to pass to the plot function, by default {}

    Returns
    -------
    matplotlib.axes.Axes
        Returns the Axes object with the plot for further tweaking.
    """
    # Add default arguments
    palette = palette or "colorblind"

    # Create plot keyword arguments
    plot_kwargs = {
        "data": df, "x": x, "y": y, "hue": hue,
        "legend": legend,
        "palette": palette,
        **extra_plot_kwargs,
    }

    # Raise error, if plot type is invalid
    supported_types = ["box", "strip", "line"]
    if plot_type not in supported_types:
        raise ValueError(f"Invalid plot type: `{plot_type}`! See supported types: {supported_types}")

    # Create plot based on requested function
    plot_func = None
    if plot_type == "box":
        plot_func = sns.boxplot
        default_kwargs = {"width": 0.95}
        plot_kwargs.update({k: v for k, v in default_kwargs.items() if k not in plot_kwargs})
    elif plot_type == "strip":
        plot_func = sns.stripplot
    elif plot_type == "line":
        plot_func = sns.lineplot

    # Create figure
    if figsize is not None:
        plt.figure(figsize=figsize)

    # Create plot
    ax = plot_func(**plot_kwargs)

    # If specified, add vertical lines
    if vertical_lines:
        for curr_x in vertical_lines:
            ax.axvline(x=curr_x, color="black", linestyle="dashed", alpha=0.5)

    # Perform post-plot logic
    ax = post_plot_logic(
        ax=ax,
        title=title, title_size=title_size,
        xlabel=xlabel, ylabel=ylabel,
        x_lim=x_lim, y_lim=y_lim,
        tick_params=tick_params,
        legend=legend,
        save_dir=save_dir, save_fname=save_fname,
    )


def barplot_with_ci(data, x, y, yerr_low, yerr_high, ax=None,
                    **plot_kwargs):
    """
    Create bar plot with custom confidence intervals.

    Parameters
    ----------
    data : pd.DataFrame
        Data
    x : str
        Name of primary column to group by
    y : str
        Name of column with bar values
    yerr_low : str
        Name of column with lower bound on confidence interval
    yerr_high : str
        Name of column with upper bound on confidence interval
    ax : matplotlib.pyplot.Axis, optional
        If provided, draw plot into this Axis instead of creating a new Axis, by
        default None.
    **plot_kwargs : keyword arguments to pass into `matplotlib.pyplot.bar`

    Returns
    -------
    matplotlib.pyplot.Axis.axis
        Grouped bar plot with custom confidence intervals
    """
    # Add default capsize if not specified
    if "capsize" not in plot_kwargs:
        plot_kwargs["capsize"] = 5

    # Create figure
    if ax is None:
        _, ax = plt.subplots()

    # Calculate error values
    yerr = [data[y] - data[yerr_low], data[yerr_high] - data[y]]

    # Create bar plot
    ax.bar(
        x=data[x].values,
        height=data[y].values,
        yerr=yerr,
        **plot_kwargs
    )

    # Remove whitespace on the left and right
    ax.set_xlim(left=-0.5, right=len(data[x].unique()) - 0.5)

    return ax


def grouped_barplot_with_ci(
        data, x, y, hue, yerr_low, yerr_high,
        hue_order=None, color=None, xlabel=None, ylabel=None, ax=None, legend=False,
        **plot_kwargs):
    """
    Create grouped bar plot with custom confidence intervals.

    Parameters
    ----------
    data : pd.DataFrame
        Data
    x : str
        Name of primary column to group by
    y : str
        Name of column with bar values
    hue : str
        Name of secondary column to group by
    yerr_low : str
        Name of column with explicit lower bound on confidence interval for `y`
    yerr_high : str
        Name of column with explicity upper bound on confidence interval for `y`
        interval
    hue_order : list, optional
        Explicit order to use for hue groups, by default None
    color : str, optional
        Color to use for bars, by default None
    xlabel : str, optional
        Label for x-axis, by default None
    ylabel : str, optional
        Label for y-axis, by default None
    ax : matplotlib.pyplot.Axis, optional
        If provided, draw plot into this Axis instead of creating a new Axis, by
        default None.
    legend : bool, optional
        If True, add legend to figure, by default False.
    **plot_kwargs : keyword arguments to pass into `matplotlib.pyplot.bar`

    Returns
    -------
    matplotlib.pyplot.Axis.axis
        Grouped bar plot with custom confidence intervals
    """
    # Get unique values for x and hue
    x_unique = data[x].unique()
    xticks = np.arange(len(x_unique))
    hue_unique = data[hue].unique()

    # If specified, fix hue order
    if hue_order:
        # Check that hue order is valid
        if len(hue_order) != len(hue_unique):
            raise RuntimeError(
                f"`hue_order` ({len(hue_order)}) does not match the number of hue groups! ({len(hue_unique)})"
            )
        hue_unique = hue_order

    # Bar-specific constants
    offsets = np.arange(len(hue_unique)) - np.arange(len(hue_unique)).mean()
    offsets /= len(hue_unique) + 1.
    width = np.diff(offsets).mean()

    # Create figure
    if ax is None:
        _, ax = plt.subplots()

    # Create bar plot per hue group
    for i, hue_group in enumerate(hue_unique):
        # Get color for hue group
        if color is not None:
            # CASE 1: One color for all bars
            if isinstance(color, str):
                plot_kwargs["color"] = color
            # CASE 2: One color per bar
            elif isinstance(color, list):
                plot_kwargs["color"] = color[i]

        # Filter for data from hue group and compute differences
        df_group = data[data[hue] == hue_group]
        # Calculate error values
        yerr = [df_group[y] - df_group[yerr_low], df_group[yerr_high] - df_group[y]]

        # Create bar plot
        ax.bar(
            x=xticks+offsets[i],
            height=df_group[y].values,
            width=width,
            label="{} {}".format(hue, hue_group),
            yerr=yerr,
            **plot_kwargs)

    # Axis labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Set x-axis ticks
    ax.set_xticks(xticks, x_unique)

    if legend:
        ax.legend()

    return ax


def post_plot_logic(
        ax,
        title=None, title_size=None,
        xlabel=None, ylabel=None,
        x_lim=None, y_lim=None,
        tick_params=None,
        legend=False,
        save_dir=None, save_fname=None,
        dpi=600,
    ):
    """
    Perform post plot operations like adding title, labels, and saving

    Parameters
    ----------
    ax : plt.Axis
        Axis to modify
    title : str, optional
        Title for the plot, by default None.
    title_size : int, optional
        Font size for title, by default 17
    xlabel : str, optional
        Label for the x-axis, by default None.
    ylabel : str, optional
        Label for the y-axis, by default None.
    x_lim : tuple, optional
        Tuple specifying the x-axis limits, by default None.
    y_lim : tuple, optional
        Tuple specifying the y-axis limits, by default None.
    tick_params : dict, optional
        Dictionary specifying the tick parameters, by default None
    legend : bool, optional
        Whether to include a legend, by default False.
    save_dir : str, optional
        Directory to save the plot, by default None.
    save_fname : str, optional
        Filename to save the plot, by default None.
    dpi : int, optional
        DPI to save the plot at, by default 600
    """
    # Add x-axis and y-axis labels
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    # Update tick parameters
    if tick_params:
        ax.tick_params(**tick_params)

    # Limit x/y-axis
    if x_lim is not None:
        ax.set_xlim(*x_lim)
    if y_lim is not None:
        ax.set_ylim(*y_lim)

    # Add title
    if title is not None:
        title_kwargs = {}
        if title_size is not None:
            title_kwargs["size"] = title_size
        ax.set_title(title, **title_kwargs)

    # If legend specified, add it outside the figure
    if legend:
        ax.legend(
            loc='center left',
            bbox_to_anchor=(1, 0.5),
            ncol=1,
        )

    # Return plot, if not saving
    if not save_dir or not save_fname:
        return ax

    # Save if specified
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, save_fname), bbox_inches="tight", dpi=dpi)

    # Clear figure, after saving
    plt.clf()
    plt.close()


################################################################################
#                               Helper Functions                               #
################################################################################
def remove_dict_keys(dictionary, keys):
    """
    Remove multiple keys from a dictionary.

    Parameters
    ----------
    dictionary : dict
        The dictionary from which to remove keys.
    keys : list
        List of keys to remove.
    """
    for key in keys:
        dictionary.pop(key, None)


def add_default_dict_vals(dictionary, **kwargs):
    """
    Add default values to a dictionary for missing keys.

    Parameters
    ----------
    dictionary : dict
        The dictionary to which default values will be added.
    **kwargs : dict
        Keyword arguments representing key-value pairs to add to the dictionary
        if the key is not already present.
    """
    for key, val in kwargs.items():
        if key not in dictionary:
            dictionary[key] = val


def extract_colors(palette, n_colors):
    """
    Extract the first n_colors colors from a seaborn color palette.

    Parameters
    ----------
    palette : str
        Name of seaborn color palette to extract colors from.
    n_colors : int
        Number of colors to extract from the palette.

    Returns
    -------
    list
        List of n_colors hex color codes.
    """
    palette = sns.color_palette("colorblind", n_colors)
    return list(map(convert_rgb_to_hex, palette))


def convert_rgb_to_hex(rgb):
    """
    Convert RGB to hex color code.

    Parameters
    ----------
    rgb : tuple of floats
        RGB values in range [0, 1]

    Returns
    -------
    str
        Hex color code
    """
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))


def bolden(text):
    """
    Make a string bold in a matplotlib plot.

    Parameters
    ----------
    text : str
        String to be made bold

    Returns
    -------
    str
        String that will be rendered as bold in a matplotlib plot
    """
    # Ensure latex rendering is enabled
    rc("text", usetex=True)
    return r"\textbf{" + text + r"}"


def get_color_for_dsets(*dsets):
    """
    Retrieve colors for specified datasets.

    Parameters
    ----------
    dsets : tuple of str
        Names of datasets for which to retrieve colors

    Returns
    -------
    list
        List of colors corresponding to the specified datasets
    """
    all_dsets = ["vindr_cxr", "padchest", "nih_cxr18", "chexbert", "vindr_pcxr"]
    colors = sns.color_palette("colorblind")[:5]
    dset_to_color = dict(zip(all_dsets, colors))

    # Add alternative names
    for dset, color in list(dset_to_color.items()):
        dset_to_color[utils.stringify_dataset_split(dset=dset)] = color

    # If no dsets provided, use all
    if not dsets:
        dsets = all_dsets

    return [dset_to_color[dset] for dset in dsets]


################################################################################
#                                User Interface                                #
################################################################################
if __name__ == "__main__":
    # Example command
    # `python -m src.utils.data.viz_data avg_healthy_image vindr_pcxr`
    Fire({
        "avg_healthy_image": compute_avg_healthy_image_by_age_group,
        "sample_healthy_image": sample_healthy_image_by_age_group,
        "age_histogram": plot_age_histograms,
        "create_legend": create_all_dset_legend,
    })
