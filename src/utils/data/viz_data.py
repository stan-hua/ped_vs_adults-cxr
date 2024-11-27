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
from matplotlib.container import ErrorbarContainer
from skimage import exposure
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
    df_metadata = pd.read_csv(constants.DIR_METADATA_MAP[dset]["png"])
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


def sample_avg_healthy_image_by_age_group(dset):
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
    df_metadata = pd.read_csv(constants.DIR_METADATA_MAP[dset]["png"])
    # NOTE: Keep only healthy patients WITH age annotations
    df_metadata = df_metadata[~df_metadata["Has Finding"].astype(bool)]

    # Create directory to save figures
    save_dir = os.path.join(constants.DIR_FIGURES_EDA, dset, "sampled_imgs")
    os.makedirs(save_dir, exist_ok=True)

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

        # Load histogram-equalized image
        sampled_img = load_image(img_paths[0], equalize=True)

        # Save image
        save_path = os.path.join(save_dir, f"sampled_healthy_patient-({curr_age})-[{age_lower}, {age_upper}).png")
        cv2.imwrite(save_path, sampled_img)


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


def load_image(img_path, equalize=False):
    """
    Loads an image from disk and applies histogram equalization, if specified.

    Parameters
    ----------
    img_path : str
        Path to the image file
    equalize : bool
        Whether to apply histogram equalization to the image, by default False

    Returns
    -------
    np.ndarray
        The loaded image, with histogram equalization applied if specified, as an 8-bit unsigned integer array
    """
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    # Apply histogram equalization
    if equalize:
        img = (255 * exposure.equalize_hist(img)).astype(np.uint8)
    return img


################################################################################
#                              Plotting Functions                              #
################################################################################
def set_theme():
    """
    Create scientific theme for plot
    """
    custom_params = {
        "axes.spines.right": False, "axes.spines.top": False,
        "figure.figsize": (10, 6)
    }
    sns.set_theme(style="ticks", font_scale=1.3, rc=custom_params)


def catplot(
        df, x=None, y=None, hue=None,
        bar_labels=None,
        palette="colorblind",
        plot_type="bar",
        figsize=None,
        title=None,
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
    palette : list, optional
        List of colors to use for the bars, by default None.
    plot_type : str, optional
        Type of plot to create, by default "bar".
    figsize : tuple, optional
        Tuple specifying the figure size, by default None.
    title : str, optional
        Title for the plot, by default None.
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
        add_default_dict_vals(plot_kwargs, width=0.95)
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
    if bar_labels or plot_type == "count":
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

    # Add x-axis and y-axis labels
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)

    # Update tick parameters
    if tick_params:
        plt.tick_params(**tick_params)

    # Limit x/y-axis
    if x_lim is not None:
        plt.xlim(*x_lim)
    if y_lim is not None:
        plt.ylim(*y_lim)

    # Add title
    if title is not None:
        plt.title(title, size=17)

    # If legend specified, add it outside the figure
    if legend and plot_type not in ["hist", "kde"]:
        plt.legend(
            loc='center left',
            bbox_to_anchor=(1, 0.5),
            ncol=1,
        )

    # Return plot, if not saving
    if not save_dir or not save_fname:
        return ax

    # Save if specified
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, save_fname), bbox_inches="tight", dpi=300)

    # Clear figure, after saving
    plt.clf()


def numplot(
        df, x=None, y=None, hue=None,
        palette=None,
        plot_type="box",
        vertical_lines=None,
        figsize=None,
        title=None,
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
    supported_types = ["box", "strip"]
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

    # Create figure
    if figsize is not None:
        plt.figure(figsize=figsize)

    # Create plot
    ax = plot_func(**plot_kwargs)

    # If specified, add vertical lines
    if vertical_lines:
        for curr_x in vertical_lines:
            ax.axvline(x=curr_x, color="black", linestyle="dashed", alpha=0.5)

    # Add x-axis and y-axis labels
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)

    # Update tick parameters
    if tick_params:
        plt.tick_params(**tick_params)

    # Limit x/y-axis
    if x_lim is not None:
        plt.xlim(*x_lim)
    if y_lim is not None:
        plt.ylim(*y_lim)

    # Add title
    if title is not None:
        plt.title(title, size=17)

    # If legend specified, add it outside the figure
    if legend:
        plt.legend(
            loc='center left',
            bbox_to_anchor=(1, 0.5),
            ncol=1,
        )

    # Return plot, if not saving
    if not save_dir or not save_fname:
        return ax

    # Save if specified
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, save_fname), bbox_inches="tight", dpi=300)

    # Clear figure, after saving
    plt.clf()
    plt.close()


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


def grouped_barplot_with_ci(data, x, y, hue, yerr_low, yerr_high, legend=False,
                    xlabel=None, ylabel=None, ax=None,
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
        Name of column to subtract y from to create LOWER bound on confidence
        interval
    yerr_high : str
        Name of column to subtract y from to create UPPER bound on confidence
        interval
    legend : bool, optional
        If True, add legend to figure, by default False.
    ax : matplotlib.pyplot.Axis, optional
        If provided, draw plot into this Axis instead of creating a new Axis, by
        default None.
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

    # Bar-specific constants
    offsets = np.arange(len(hue_unique)) - np.arange(len(hue_unique)).mean()
    offsets /= len(hue_unique) + 1.
    width = np.diff(offsets).mean()

    # Create figure
    if ax is None:
        _, ax = plt.subplots()

    # Create bar plot per hue group
    for i, hue_group in enumerate(hue_unique):
        df_group = data[data[hue] == hue_group]
        ax.bar(
            x=xticks+offsets[i],
            height=df_group[y].values,
            width=width,
            label="{} {}".format(hue, hue_group),
            yerr=abs(df_group[[yerr_low, yerr_high]].T.to_numpy()),
            **plot_kwargs)

    # Axis labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Set x-axis ticks
    ax.set_xticks(xticks, x_unique)

    if legend:
        ax.legend()

    return ax


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


################################################################################
#                                User Interface                                #
################################################################################
if __name__ == "__main__":
    # Example command
    # `python -m src.utils.data.viz_data avg_healthy_image vindr_pcxr`
    Fire({
        "avg_healthy_image": compute_avg_healthy_image_by_age_group,
        "sample_healthy_image": sample_avg_healthy_image_by_age_group,
    })
