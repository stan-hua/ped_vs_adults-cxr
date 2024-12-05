"""
eval_model.py

Description: Used to evaluate a trained model's performance on other view
             labeling datasets.
"""

# Standard libraries
import math
import json
import logging
import os
import sys
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field

# Non-standard libraries
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from fire import Fire
from sklearn import metrics as skmetrics
from torchvision.io import read_image
from tqdm import tqdm

# Custom libraries
from config import constants
from src.utils.data import load_data, dataset, utils as data_utils, viz_data
from src.utils.model import load_model
from src.utils.misc.logging import load_comet_logger


################################################################################
#                                Configuration                                 #
################################################################################
# Configure seaborn color palette
sns.set_palette("Paired")

# Add progress_apply to pandas
tqdm.pandas()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)


################################################################################
#                                  Constants                                   #
################################################################################
LOGGER = logging.getLogger(__name__)

# Flag to use GPU or not
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Plot theme (light/dark)
THEME = "dark"

# Flag to overwrite existing results
OVERWRITE_EXISTING = False

# Flag to use CometML logger, by default
# NOTE: Set to False, if you don't want to use Comet ML by default
DEFAULT_USE_COMET_ML = True

# Random seed
SEED = 42


################################################################################
#                                   Classes                                    #
################################################################################
@dataclass(unsafe_hash=True)
class EvalHparams:
    """
    Class for evaluation hyperparameters, and storing intermediate data
    """
    task : str = None                # Task to perform (infer/analyze)

    # Experiment specific
    exp_name : str = ""                 # Name of training run
    ckpt_option : str = "best"          # Checkpoint to use (last or best val)
    model_dir : str = ""

    ############################################################################
    #                              Inference                                   #
    ############################################################################
    exp_hparams : dict = field(default_factory=lambda: {})      # Experiment hyperparameters
    data_hparams : dict = field(default_factory=lambda: {})     # Evaluation hyperparameters
    dset : str = "vindr_pcxr"                           # Datasets to evaluate
    split : str = "test"                                # Splits for datasets to evaluate
    use_comet_logger : bool = DEFAULT_USE_COMET_ML      # Log to CometML if True

    # Flags
    overwrite_existing : bool = OVERWRITE_EXISTING  # If True, overwrite existing predictions
    ############################################################################
    #                                  Evaluation                              #
    ############################################################################
    # If inferring/evaluating over multiple datasets/splits
    dsets : list = field(default_factory=lambda: [])
    splits : list = field(default_factory=lambda: [])
    curr_idx : int = 0
    # Number of datasets-splits to infer/evaluate over
    num_iters : int = -1

    # Saved predictions ((dset, split) to preds)
    df_pred : pd.DataFrame = None
    dset_split_to_preds : dict = field(default_factory=lambda: {})


    def incr_dset_and_split(self):
        """
        If multiple datasets and splits provided, initialize or increment
        dataset or split chosen.

        Returns
        -------
        bool
            True if a new dset/split was set, and False, if there are no more
            to iterate over
        """
        # Assert that dsets/dset is specified
        assert ((self.dsets and self.splits) or (self.dset and self.split)), \
            "(`dsets` and `splits`) or (`dset` and `split`) must be specified!"

        # If only dset and split specified, set dsets/splits for next logic
        # NOTE: Only occurs on first call
        if (self.dset and self.split) and not (self.dsets and self.splits):
            self.dsets = [self.dset]
            self.splits = [self.split]
            self.curr_idx += 1
            return True

        # Early return, if no more iterations possible
        self.num_iters = max(len(self.dsets), len(self.splits))
        if self.curr_idx >= self.num_iters:
            return False

        # Broadcast when one of dsets/splits is 1 and the other isn't
        if self.num_iters > len(self.splits) and len(self.splits) == 1:
            self.splits = self.num_iters * self.splits
        elif self.num_iters > len(self.dsets) and len(self.dsets) == 1:
            self.dsets = self.num_iters * self.dsets
        assert len(self.dsets) == len(self.splits), \
            f"Mismatched dsets/splits! dsets: {self.dsets}, splits: {self.splits}"

        # Set new dataset and split
        self.dset = self.dsets[self.curr_idx]
        self.split = self.splits[self.curr_idx]

        # Update hyperparameters
        self.load_hyperparameters()

        # Increment index
        self.curr_idx += 1


    def __post_init__(self):
        """
        Perform task specified in hyperparameters
        """
        # Early return, if task is false-y
        if not self.task:
            return

        # Ensure that task is valid
        task_to_func = {
            "infer": infer_dset,
            "check_adult_vs_child": eval_are_children_over_predicted,
        }
        assert self.task in task_to_func, f"Invalid task: {self.task}! Valid tasks: {list(task_to_func.keys())}"

        # Iterate over `dset` and `split` pairs
        while self.incr_dset_and_split():
            # Validate hyperparameters for task
            task_to_validate = {
                "infer": self.validate_for_inference,
                "analyze": self.validate_for_evaluation,
                "check_adult_vs_child": self.validate_for_evaluation,
            }
            LOGGER.info(f"[Task `{self.task}`] Validating input...")
            task_to_validate[self.task]()
            LOGGER.info(f"[Task `{self.task}`] Validating input...DONE")

            # Perform task
            LOGGER.info(f"[Task `{self.task}`] Performing task...")
            task_to_func[self.task](self)
            LOGGER.info(f"[Task `{self.task}`] Performing task...DONE")


    ############################################################################
    #                       Dictionary-Like Methods                            #
    ############################################################################
    def __getitem__(self, key):
        return getattr(self, key)


    def __setitem__(self, key, value):
        return setattr(self, key, value)


    ############################################################################
    #                          Validation Methods                              #
    ############################################################################
    def validate_for_inference(self):
        """
        Validate hyperparameters are valid to perform inference
        """
        attr_list = ["exp_name", "ckpt_option", "dset", "split"]
        invalid_attrs = [attr for attr in attr_list if not getattr(self, attr)]
        if invalid_attrs:
            invalid_attr_vals = {attr: getattr(self, attr) for attr in invalid_attrs}
            raise ValueError(f"Insufficient parameters for inference! See: {invalid_attr_vals}")


    def validate_for_evaluation(self):
        """
        Validate hyperparameters are valid to perform evaluation
        """
        attr_list = ["exp_name", "ckpt_option", "dset", "split"]
        invalid_attrs = [attr for attr in attr_list if not getattr(self, attr)]
        if invalid_attrs:
            invalid_attr_vals = {attr: getattr(self, attr) for attr in invalid_attrs}
            raise ValueError(f"Insufficient parameters for evaluation! See: {invalid_attr_vals}")

        # Load hyperparameters
        self.load_hyperparameters()

        # Load predictions
        self.preload_inference()


    ############################################################################
    #                           Helper Functions                               #
    ############################################################################
    def preload_inference(self, dset=None, split=None):
        """
        Load predictions from inference.

        Parameters
        ----------
        dset : str
            Dataset to load predictions for. If not provided, uses `self.dset`
        split : str
            Split to load predictions for. If not provided, uses `self.split`
        """
        dset = dset or self.dset
        split = split or self.split
        assert dset and split, "`dset` and `split` must be specified in the class or provided to the function!"

        # Early return, if predictions are already in cache
        key = (dset, split)
        if key in self.dset_split_to_preds:
            self.df_pred = self.dset_split_to_preds[key]

        # Temporarily set dset/split
        orig_dset, orig_split = self.dset, self.split
        self.dset, self.split = dset, split

        # Load predictions for dset/split
        save_path = create_save_path(self)
        assert os.path.exists(save_path), f"Inference doesn't exist for dset ({dset}) and split ({split}). Expected path: {save_path}"
        self.df_pred = pd.read_csv(save_path)
        self.df_pred = self.process_inference(dset, split, self.df_pred)
        self.dset_split_to_preds[key] = self.df_pred

        # Reset dset/split
        self.dset, self.split = orig_dset, orig_split


    def process_inference(self, dset, split, df_pred):
        """
        Processes the predictions for a given dataset and split.

        Parameters
        ----------
        dset : str
            Dataset to process predictions for
        split : str
            Split to process predictions for
        df_pred : pd.DataFrame
            DataFrame of predictions

        Returns
        -------
        pd.DataFrame
            Processed DataFrame of predictions
        """
        # Extract age
        if "age_years" not in self.df_pred.columns:
            df_pred = data_utils.extract_age(dset, df_pred.copy())

        # CASE 1: VinDr-PCXR
        if dset == "vindr_pcxr" and split == "test":
            # Filter for healthy children with a valid age (<=10)
            df_pred = df_pred.dropna(subset=["age_years"])
            mask = (~df_pred["Has Finding"].astype(bool)) & (df_pred["age_years"] <= 10)
            return df_pred[mask]

        # CASE 2: All pediatric splits
        if split == "test_peds":
            # Filter for healthy children with a valid age annotation
            df_pred = df_pred.dropna(subset=["age_years"])
            mask = (~df_pred["Has Finding"].astype(bool))
            return df_pred[mask]

        # CASE 3: All healthy adults
        if split == "test_healthy_adult":
            # Filter for healthy adults with a valid age annotation
            # NOTE: It should already be filtered for having no finding
            df_pred = df_pred.dropna(subset=["age_years"])
            return df_pred

        return df_pred


    def load_hyperparameters(self):
        """
        Load hyperparameters from training run, and update hyperparameters
        for specific dataset and split.
        """
        # Get experiment hyperparameters
        self.model_dir = load_model.get_exp_dir(self.exp_name)
        self.exp_hparams = load_model.get_hyperparameters(self.model_dir)

        # Overwrite hyperparameters (changes `dset` used in training)
        # NOTE: Used only so that DataModule loads new dset/split instead of
        #       one used during training
        overwrite_hparams = load_data.create_eval_hparams(self.dset)
        self.data_hparams = deepcopy(self.exp_hparams)
        self.data_hparams.update(overwrite_hparams)


################################################################################
#                             Inference - Related                              #
################################################################################
@torch.no_grad()
def predict_on_images(model, img_paths, hparams=None, labels=None, idx_to_label=None):
    """
    Perform inference on a list of images using a pre-trained model.

    Parameters
    ----------
    model : torch.nn.Module
        Pre-trained model to use for inference
    img_paths : list
        List of paths to images to perform inference on
    hparams : dict, optional
        Hyperparameters used to train model, by default {}
    labels : list, optional
        List of labels corresponding to images in `img_paths`, by default None
    idx_to_label : dict, optional
        Optional dictionary to map label indices to label names, by default {}

    Returns
    -------
    pd.DataFrame
        Table containing predictions, probabilities, and losses for each image
    """
    hparams = hparams or {}
    idx_to_label = idx_to_label or {}

    # Set to evaluation mode
    model.eval()
    model = model.to(DEVICE)

    # Prepare dictionary to accumulate results
    accum_ret = {
        "pred": [],
        "prob": [],
        "out": [],
        "loss": [],
        "class_probs": []
    }

    # Predict on each images one-by-one
    for idx, img_path in tqdm(enumerate(img_paths)):
        # Load image
        img = load_image(img_path, hparams.get("img_mode", 3)).to(DEVICE)
        img = img.unsqueeze(0)

        # Perform inference
        out = model(img)

        # Compute loss, if label provided
        # NOTE: Encoded label must be >= 0
        loss = None
        if labels and labels[idx] >= 0:
            label = torch.tensor([labels[idx]], dtype=int, device=out.device)
            loss = round(float(F.cross_entropy(out, label).detach().cpu().item()), 4)
        accum_ret["loss"].append(loss)

        # Store logits / probabilities / prediction
        for key, val in extract_predictions_from_logits(out, idx_to_label).items():
            accum_ret[key].append(val)

    # Pack into dataframe
    df_preds = pd.DataFrame(accum_ret)

    # Remove columns with all NA
    df_preds = df_preds.dropna(axis=1, how="all")

    return df_preds


@torch.no_grad()
def predict_with_cxr_dataset(model, dataset, idx_to_label=None):
    """
    Perform inference on a list of images using a pre-trained model.

    Parameters
    ----------
    model : torch.nn.Module
        Pre-trained model to use for inference
    dataset : dataset.CXRDatasetDataFrame
        Chest x-ray dataset used to load data
    idx_to_label : dict, optional
        Dictionary mapping label indices to label names, by default {}

    Returns
    -------
    pd.DataFrame
        Table containing predictions, probabilities, and losses for each image
    """
    idx_to_label = idx_to_label or {}

    # Set to evaluation mode
    model.eval()
    model = model.to(DEVICE)

    # Prepare dictionary to accumulate results
    accum_ret = {
        "pred": [],
        "prob": [],
        "out": [],
        "loss": [],
        "class_probs": []
    }

    # Predict on each images one-by-one
    num_images = len(dataset)
    for idx in tqdm(range(num_images)):
        img, metadata = dataset[idx]
        label = metadata["label"]

        # Perform inference
        img = img.unsqueeze(0).to(DEVICE)
        out = model(img)

        # Compute loss
        # NOTE: A label of <0 indicates missing label
        loss = None
        if label >= 0:
            label = torch.tensor([label], dtype=int, device=out.device)
            loss = round(float(F.cross_entropy(out, label).detach().cpu().item()), 4)
        accum_ret["loss"].append(loss)

        # Store logits / probabilities / prediction
        for key, val in extract_predictions_from_logits(out, idx_to_label).items():
            accum_ret[key].append(val)

    # Pack into dataframe
    df_preds = pd.DataFrame(accum_ret)

    # Remove columns with all NA
    df_preds = df_preds.dropna(axis=1, how="all")

    return df_preds


################################################################################
#                              Specific Questions                              #
################################################################################
def eval_are_children_over_predicted(eval_hparams: EvalHparams):
    """
    Evaluates if children are over-predicted on model calibrated on test set.

    Parameters
    ----------
    eval_hparams : EvalHparams
        Stores evaluation hyperparameters and predictions
    """
    # Ensure all datasets and splits are loaded
    dsets = ["vindr_cxr", "nih_cxr18", "padchest", "chexbert"]
    dset_to_fpr = defaultdict(dict)
    for curr_dset in dsets:
        curr_split = "test_healthy_adult"
        fpr, df_calib_counts = compute_fpr_after_calibration(eval_hparams, curr_dset, curr_split)
        dset_to_fpr[curr_dset][curr_split] = fpr
        # Plot FPR by age bins
        plot_fpr_after_calibration(eval_hparams, df_calib_counts, curr_dset, curr_split)
        # Plot age histogram
        plot_age_histograms(eval_hparams, curr_dset, curr_split)

    # Plot FPR for the following datasets
    peds_dset_splits = [("vindr_pcxr", "test"), ("nih_cxr18", "test_peds"), ("padchest", "test_peds")]
    for curr_dset, curr_split in peds_dset_splits:
        # Compute FPR
        fpr, df_calib_counts = compute_fpr_after_calibration(eval_hparams, curr_dset, curr_split)
        dset_to_fpr[curr_dset][curr_split] = fpr
        # Plot FPR by age bins
        plot_fpr_after_calibration(eval_hparams, df_calib_counts, curr_dset, curr_split)
        # Plot age histogram
        plot_age_histograms(eval_hparams, curr_dset, curr_split)

    # Store FPRs for all datasets and splits
    save_fpr = os.path.join(constants.DIR_FINDINGS, eval_hparams["exp_name"], "fpr.json")
    with open(save_fpr, "w") as handler:
        json.dump(dict(dset_to_fpr), handler, indent=4)

    return dset_to_fpr


def eval_are_adults_over_predicted_same_source(
        *exp_names, label_col="cardiomegaly"):
    """
    Given models from a same source dataset, check if they overpredict on
    the healthy adult datasets.

    Parameters
    ----------
    *exp_names : *args
        List of experiment names for each model
    label_col : str, optional
        Name of label column
    """
    # NOTE: Only need lower-case version of label columns
    label_col = label_col.lower()

    # For each model and dataset, get the calibration counts for all datasets
    accum_data = []
    train_dsets = []
    for exp_name in exp_names:
        # Get experiment's training set
        hparams = load_model.get_hyperparameters(exp_name=exp_name)
        train_dset = hparams["dset"]

        # Load calibration counts
        df_curr = pd.read_csv(os.path.join(
            constants.DIR_FINDINGS, exp_name, train_dset,
            "test_healthy_adult", f"{label_col}_calib_counts.csv"
        ))
        df_curr["trained on"] = train_dset

        # Convert to false positive rate
        df_curr["fpr"] = df_curr["Pred. Percentage"] / 100
        df_curr["fpr_lower"] = df_curr["fpr_lower"] / 100
        df_curr["fpr_upper"] = df_curr["fpr_upper"] / 100

        # Store temporarily
        accum_data.append(df_curr)
        train_dsets.append(train_dset)

    # Assign each training dataset a color
    train_dsets = sorted(train_dsets, reverse=True)
    train_dset_colors = dict(zip(train_dsets, viz_data.extract_colors("colorblind", len(train_dsets))))

    # For each dataset, plot bar plot of performance on its healthy adults
    viz_data.set_theme(tick_scale=1.1)
    fig, axs = plt.subplots(
        ncols=min(2, len(train_dsets)), nrows=math.ceil(len(train_dsets)/2),
        figsize=(12, 6),
        dpi=300,
        constrained_layout=True
    )
    # NOTE: Flatten axes for easier indexing
    axs = [curr_ax for group_ax in axs for curr_ax in group_ax]
    for idx, (train_dset, df_curr) in enumerate(list(zip(train_dsets, accum_data))):
        ax = axs[idx]
        df_curr = None

        # Plot grouped bar plot
        viz_data.catplot(
            df_curr, x="age_bin", y="fpr", hue="trained on",
            yerr_low="fpr_lower", yerr_high="fpr_upper",
            plot_type="bar_with_ci",
            capsize=7,
            error_kw={"elinewidth": 1},
            color=train_dset_colors[train_dset],
            ylabel="False Positive Rate",
            xlabel="Age (In Years)",
            y_lim=(0, 1),
            legend=False,
            ax=ax,
            title=f"Evaluated on {stringify_dataset_split(train_dset)}",
            title_size=13,
        )

    # Add title
    fig.suptitle(
        f"{label_col.capitalize()} Classification on Healthy Adults",
        size=17,
    )

    # Create custom legend at the bottom
    legend_handles = [
        mpatches.Patch(color=curr_color, label=stringify_dataset_split(train_dset))
        for train_dset, curr_color in train_dset_colors.items()
    ]
    fig.legend(
        handles=legend_handles, loc='lower center', bbox_to_anchor=(0.5, -0.15),
        ncol=len(legend_handles), title="Trained On",
    )

    # Save figure
    save_dir=os.path.join(constants.DIR_FINDINGS, "PedsVsAdult_CXR")
    save_fname=f"same_source-healthy_adults-{label_col}_fpr_by_age ({','.join(train_dsets)}).svg"
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, save_fname), bbox_inches="tight")

    # Clear figure, after saving
    plt.clf()
    plt.close()


def eval_are_adults_over_predicted_different_source(
        *exp_names, adult_dsets="all", label_col="cardiomegaly"):
    """
    Given models from a different source dataset, check if they overpredict on
    the healthy adult datasets.

    Parameters
    ----------
    *exp_names : *args
        List of experiment names for each model
    adult_dsets : str, optional
        If "all", evaluate on all adult datasets. Otherwise, must be a specific
        adult dataset
    label_col : str, optional
        Name of label column
    """
    adult_dsets = ["vindr_cxr", "nih_cxr18", "padchest", "chexbert"] if adult_dsets == "all" else adult_dsets
    # NOTE: Only need lower-case version of label columns
    label_col = label_col.lower()

    # For each model and dataset, get the calibration counts for all datasets
    accum_data = []
    for exp_name in exp_names:
        for eval_dset in adult_dsets:
            # Skip when training dset = eval dset
            train_dset = exp_name.split("-")[1]
            if train_dset == eval_dset:
                continue
            # SPECIAL CASE: NIH isn't named the same
            if train_dset == "nih_cxr" and eval_dset == "nih_cxr18":
                continue
            # Load calibration counts
            df_curr = pd.read_csv(
                os.path.join(constants.DIR_FINDINGS, exp_name, eval_dset,
                             "test_healthy_adult", f"{label_col}_calib_counts.csv"
            ))
            df_curr["trained on"] = exp_name.split("-")[1]
            df_curr["eval_dset"] = eval_dset
            accum_data.append(df_curr)

    # Combine tables
    df_accum = pd.concat(accum_data, axis=0, ignore_index=True)

    # Convert to false positive rate
    df_accum["fpr"] = df_accum["Pred. Percentage"] / 100
    df_accum["fpr_lower"] = df_accum["fpr_lower"] / 100
    df_accum["fpr_upper"] = df_accum["fpr_upper"] / 100

    # Assign each training dataset a color
    train_dsets = sorted(df_accum["trained on"].unique().tolist(), reverse=True)
    train_dset_colors = dict(zip(train_dsets, viz_data.extract_colors("colorblind", len(train_dsets))))

    # For each eval dataset, plot bar plot grouped by evaluation set
    viz_data.set_theme(tick_scale=1.1)
    fig, axs = plt.subplots(
        ncols=min(2, len(adult_dsets)), nrows=math.ceil(len(adult_dsets)/2),
        figsize=(12, 6),
        dpi=300,
        constrained_layout=True
    )
    # NOTE: Flatten axes for easier indexing
    axs = [curr_ax for group_ax in axs for curr_ax in group_ax]
    for idx, eval_dset in enumerate(adult_dsets):
        ax = axs[idx]
        df_accum_dset = df_accum[df_accum["eval_dset"] == eval_dset]

        # Create plot parameters for trained dataset
        curr_train_dsets = sorted(df_accum_dset["trained on"].unique().tolist(), reverse=True)
        curr_colors = [train_dset_colors[train_dset] for train_dset in curr_train_dsets]

        # Plot grouped bar plot
        viz_data.catplot(
            df_accum_dset, x="age_bin", y="fpr", hue="trained on",
            yerr_low="fpr_lower", yerr_high="fpr_upper",
            plot_type="grouped_bar_with_ci",
            capsize=7,
            hue_order=curr_train_dsets,
            color=curr_colors,
            error_kw={"elinewidth": 1},
            ylabel="False Positive Rate",
            xlabel="Age (In Years)",
            y_lim=(0, 1),
            legend=False,
            ax=ax,
            title=f"Evaluated on {stringify_dataset_split(eval_dset)}",
            title_size=13,
        )

    # Add title
    fig.suptitle(
        f"{label_col.capitalize()} Classification on Healthy Adults",
        size=17,
    )

    # Create custom legend at the bottom
    legend_handles = [
        mpatches.Patch(color=curr_color, label=stringify_dataset_split(train_dset))
        for train_dset, curr_color in train_dset_colors.items()
    ]
    fig.legend(
        handles=legend_handles, loc='lower center', bbox_to_anchor=(0.5, -0.15),
        ncol=len(legend_handles), title="Trained On", 
    )

    # Save figure
    save_dir=os.path.join(constants.DIR_FINDINGS, "PedsVsAdult_CXR")
    save_fname=f"diff_source-healthy_adults-{label_col}_fpr_by_age ({','.join(adult_dsets)}).svg"
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, save_fname), bbox_inches="tight")

    # Clear figure, after saving
    plt.clf()
    plt.close()


def compute_fpr_after_calibration(eval_hparams: EvalHparams, dset, split):
    """
    Computes the false positive rate (FPR) after calibration for a given experiment/dataset/split.

    Parameters
    ----------
    eval_hparams : EvalHparams
        Experiment hyperparameters
    dset : str
        Dataset to evaluate
    split : str
        Split to evaluate

    Returns
    -------
    prop_positive : float
        False positive rate across all samples
    df_calib_counts : pd.DataFrame
        Dataframe containing the following columns:
            age_bin : Age bin
            Pred. Count : Number of predictions in each age bin
            Pred. Percentage : Percentage of predictions in each age bin

    Notes
    -----
    Calibration is done using Platt scaling, which is a method of calibrating
    the output of a classifier to produce well-calibrated probabilities.
    The FPR is computed by determining the proportion of predicted positives
    in each age bin.
    """
    # Load experiment hyperparameters
    eval_hparams.load_hyperparameters()
    exp_name = eval_hparams["exp_name"]
    label_col = eval_hparams["exp_hparams"]["label_col"]

    # Stringify label column
    label_col_str = label_col.lower().replace(" ", "_")

    # Get predictions
    eval_hparams.preload_inference(dset=dset, split=split)
    df_pred = eval_hparams.dset_split_to_preds[(dset, split)]

    # Parse class probabilities
    class_probs = np.stack(df_pred["class_probs"].map(lambda x: np.array(json.loads(x))))

    # Get calibrated thresholds
    thresholds = compute_calib_thresholds(eval_hparams)

    # Compute fpr after calibration
    calibrated_preds = (class_probs[:, 1] >= thresholds).astype(int)
    prop_positive = round(100*(calibrated_preds == 1).mean(), 2)
    LOGGER.info(f"[{dset} ({split})] Calibrated % Predicted Positive: {prop_positive}")

    # Stratify by age bins
    df_pred["age_years"] = df_pred["age_years"].astype(int)
    df_pred["age_bin"] = get_age_bins(df_pred, dset, split, age_col="age_years")
    df_pred["pred_calibrated"] = calibrated_preds
    counts = df_pred.groupby("age_bin", observed=True).apply(
        lambda df: df["pred_calibrated"].sum())
    counts.name = "Pred. Count"
    percs = df_pred.groupby("age_bin", observed=True).apply(
        lambda df: round(100 * (df["pred_calibrated"] == 1).mean(), 2)
    )
    percs.name = "Pred. Percentage"

    # Bootstrap confidence intervals
    conf_intervals = df_pred.groupby("age_bin", observed=True).apply(
        lambda df: pd.Series(dict(zip(
            ["fpr_lower", "fpr_upper"],
            bootstrap_metric(
                df,
                metric_func=(lambda x, _: round(100*(x==1), 2).mean()),
                label_col="pred_calibrated",
                pred_col="pred_calibrated")[1]))))

    df_calib_counts = pd.concat([counts, percs], axis=1)
    df_calib_counts = pd.concat([df_calib_counts, conf_intervals], axis=1)
    df_calib_counts = df_calib_counts.reset_index()

    # Save table
    save_dir = os.path.join(constants.DIR_FINDINGS, exp_name, dset, split)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{label_col_str}_calib_counts.csv")
    df_calib_counts.to_csv(save_path, index=False)

    return prop_positive, df_calib_counts


def plot_fpr_after_calibration(eval_hparams: EvalHparams, df_calib_counts, dset, split):
    """
    Plots the false positive rate after calibration for a given experiment/dataset/split.

    Parameters
    ----------
    eval_hparams : EvalHparams
        Experiment hyperparameters
    df_calib_counts : pd.DataFrame
        Dataframe containing the following columns:
            age_bin : Age bin
            Pred. Count : Number of predictions in each age bin
            Pred. Percentage : Percentage of predictions in each age bin
    dset : str
        Dataset to evaluate
    split : str
        Split to evaluate
    """
    viz_data.set_theme()

    # Load experiment hyperparameters
    eval_hparams.load_hyperparameters()
    exp_name = eval_hparams["exp_name"]
    label_col = eval_hparams["exp_hparams"]["label_col"]
    label_col_str = label_col.lower().replace(" ", "_")

    # Only rotate tick labels if age bins are scalar and not an interval
    # TODO: Reconsider bringing back
    is_age_scalar = df_calib_counts["age_bin"].iloc[0].isnumeric()
    # tick_params = {"axis": "x", "rotation": 45} if not is_age_scalar else None
    tick_params = None

    # Specify age bin ordering
    df_calib_counts = df_calib_counts.sort_values("age_bin")
    if is_age_scalar:
        df_calib_counts["temp_age"] = df_calib_counts["age_bin"].astype(int)
        df_calib_counts = df_calib_counts.sort_values("temp_age")

    # Plot false positive rate
    df_calib_counts["fpr"] = df_calib_counts["Pred. Percentage"] / 100
    df_calib_counts["fpr_lower"] = df_calib_counts["fpr_lower"] / 100
    df_calib_counts["fpr_upper"] = df_calib_counts["fpr_upper"] / 100
    viz_data.set_theme()
    viz_data.catplot(
        df_calib_counts, x="age_bin", y="fpr",
        yerr_low="fpr_lower", yerr_high="fpr_upper",
        plot_type="bar_with_ci", capsize=10,
        color="#f55151",
        bar_labels=df_calib_counts["Pred. Count"].tolist(),
        error_kw={"elinewidth": 2},
        ylabel="False Positive Rate",
        xlabel="Age (In Years)",
        tick_params=tick_params,
        y_lim=(0, 1),
        legend=False,
        title=f"Adult {label_col} Classifier Predictions \non " + stringify_dataset_split(dset, split),
        save_dir=os.path.join(constants.DIR_FINDINGS, exp_name, dset, split),
        save_fname=f"{label_col_str}_fpr_by_age.png",
    )


def compute_calib_thresholds(eval_hparams: EvalHparams):
    """
    Computes the calibrated threshold from the adult calibration set.

    Parameters
    ----------
    eval_hparams : EvalHparams
        Evaluation hyperparameters

    Returns
    -------
    thresholds_cxr : float
        Calibrated probability threshold for the current model
    """
    eval_hparams.load_hyperparameters()
    label_col = eval_hparams["exp_hparams"]["label_col"]
    training_dset = eval_hparams["exp_hparams"]["dset"]

    # Get calibration set predictions
    eval_hparams.preload_inference(dset=training_dset, split="test_adult_calib")
    df_pred_calib = eval_hparams.dset_split_to_preds[(training_dset, "test_adult_calib")]

    # Compute thresholds on CXR calibration set
    class_probs = np.stack(df_pred_calib["class_probs"].map(lambda x: np.array(json.loads(x))))
    encoded_labels = df_pred_calib[label_col].to_numpy()
    thresholds_cxr = compute_thresholds(class_probs, encoded_labels, metric="f1")[1]
    LOGGER.info(f"[{training_dset} (Calibration)] Calibrated Threshold: {thresholds_cxr}")

    return thresholds_cxr


def plot_age_histograms(eval_hparams: EvalHparams, dset, split):
    """
    Plots age histograms for the predictions of a given dataset and split.

    Parameters
    ----------
    eval_hparams : EvalHparams
        Evaluation hyperparameters containing predictions and settings.
    dset : str
        Dataset name for which the predictions are plotted.
    split : str
        Split name for which the predictions are plotted.

    The function loads predictions for the specified dataset and split,
    determines the age bins, and creates a count plot displaying the
    distribution of ages for the predictions. The plot is saved to a
    specified directory.
    """
    # Get predictions
    eval_hparams.preload_inference(dset=dset, split=split)
    df_pred = eval_hparams.dset_split_to_preds[(dset, split)]

    # Determine age bins
    df_pred["age_bin"] = get_age_bins(df_pred, dset, split)
    is_age_scalar = df_pred["age_bin"].iloc[0].isnumeric()
    xlabel = "Age (in Years)" if is_age_scalar else "Age Bin (in Years)"

    # Specify age bin ordering
    order = df_pred["age_bin"].value_counts().sort_index().index.tolist()
    if is_age_scalar:
        order = sorted(df_pred["age_years"].astype(int).unique())

    # Create count plot
    viz_data.set_theme()
    viz_data.catplot(
        df_pred, x="age_bin",
        plot_type="count",
        color="#94a4d6",
        width=0.95,
        xlabel=xlabel, ylabel="Number of Patients",
        title="Age Distribution of " + stringify_dataset_split(dset, split),
        order=order,
        hue_order=order,
        legend=False,
        save_dir=os.path.join(constants.DIR_FIGURES_EDA, dset),
        save_fname=f"{dset}-{split}-age_histogram.png",
    )


################################################################################
#                               Helper Functions                               #
################################################################################
def load_image(img_path, img_mode=3):
    """
    Load image as tensor or numpy

    Parameters
    ----------
    img_path : str
        Path to image
    img_mode : int, optional
        Image mode, by default 3

    Returns
    -------
    torch.Tensor
        Image
    """
    # Load image
    img_mode = "RGB" if img_mode == 3 else "GRAY"
    img = read_image(img_path, mode=img_mode)
    img = img.to(torch.float32)
    return img


def create_save_dir(eval_hparams: EvalHparams):
    """
    Create directory to save dset predictions, based on experiment name and
    keyword arguments

    Parameters
    ----------
    eval_hparams : EvalHparams
        Experiment evaluation hyperparameters

    Returns
    -------
    str
        Expected directory to save dset predictions
    """
    for param in ["exp_name", "dset", "split"]:
        assert eval_hparams[param], f"`{param}` must be specified in the EvalHparams class! Found: {eval_hparams[param]}"

    # Create inference directory path
    base_save_dir = os.path.join(
        constants.DIR_INFERENCE,
        eval_hparams["exp_name"],
        eval_hparams["dset"],
        eval_hparams["split"],
    )

    # Add subdirectory based on additional params
    additional_params = ["ckpt_option"]
    subdir_name = "_".join([
        eval_hparams[param] for param in additional_params if eval_hparams[param]
    ])

    # Create save directory
    save_dir = os.path.join(base_save_dir, subdir_name)
    os.makedirs(save_dir, exist_ok=True)

    return save_dir


def create_save_path(eval_hparams: EvalHparams):
    """
    Create file path to dset predictions, based on experiment name and keyword
    arguments

    Parameters
    ----------
    eval_hparams : EvalHparams
        Experiment evaluation hyperparameters

    Returns
    -------
    str
        Expected path to dset predictions
    """
    # Create inference directory path
    save_dir = create_save_dir(eval_hparams)

    # Create path to predictions
    save_path = os.path.join(save_dir, "predictions.csv")
    return save_path


def scale_and_round(x, factor=100, num_places=2):
    """
    Scale and round if value is an integer or float.

    Parameters
    ----------
    x : Any
        Any object
    factor : int
        Factor to multiply by
    num_places : int
        Number of decimals to round
    """
    if not isinstance(x, (int, float)):
        return x
    x = round(factor * x, num_places)
    return x


def bootstrap_metric(df_pred,
                     metric_func=skmetrics.accuracy_score,
                     label_col="label", pred_col="pred",
                     alpha=0.05,
                     n_bootstrap=12000,
                     seed=SEED):
    """
    Perform BCa bootstrap on table of predictions to calculate metric with a
    bootstrapped confidence interval.

    Parameters
    ----------
    df_pred : pandas.DataFrame
        Model predictions. Each row contains a label,
        prediction, and other patient and sequence-related metadata.
    metric_func : function, optional
        Reference to function that can be used to calculate a metric given the
        (label, predictions), by default sklearn.metrics.accuracy_score
    label_col : str, optional
        Name of label column, by default "label"
    pred_col : str, optional
        Name of label column, by default "pred"
    alpha : float, optional
        Desired significance level, by default 0.05
    n_bootstrap : int, optional
        Sample size for each bootstrap iteration
    seed : int, optional
        Random seed

    Returns
    -------
    tuple of (exact, (lower_bound, upper_bound))
        Output of `func` with 95% confidence interval ends
    """
    # Late import to prevent unnecessary
    try:
        from arch.bootstrap import IIDBootstrap
    except ImportError:
        raise ImportError("Missing `arch` dependency! Please install via `pip install arch`")

    # Calculate exact point metric
    # NOTE: Assumes function takes in (label, pred)
    exact_metric = round(metric_func(df_pred[label_col], df_pred[pred_col]), 4)

    # Initialize bootstrap
    bootstrap = IIDBootstrap(
        df_pred[label_col], df_pred[pred_col],
        seed=seed)

    try:
        # Calculate empirical CI bounds
        ci_bounds = bootstrap.conf_int(
            func=metric_func,
            reps=n_bootstrap,
            method="bca",
            size=1-alpha,
            tail="two").flatten()
        # Round to 4 decimal places
        ci_bounds = np.round(ci_bounds, 4)
    except RuntimeError: # NOTE: May occur if all labels are predicted correctly
        print("Failed to compute confidence interval! Returning exact metric.")
        ci_bounds = (exact_metric, exact_metric)

    return exact_metric, tuple(ci_bounds)


def compute_thresholds(class_probs, encoded_labels, metric="f1"):
    """
    Compute probability threshold that maximizes the F1 score

    Note
    ----
    This is typically run on a validation set 

    Parameters
    ----------
    class_probs : np.array
        Array of (num_samples, num_classes) with predicted probabilities
    encoded_labels : np.array
        Array of integer encoded label (num_samples,)
    metric : str, optional
        Metric to maximize, by default "f1"

    Returns
    -------
    list
        List of probability thresholds for each class
    """
    # Assert on supported metrics
    # NOTE: In the future, support more metrics
    supported_metrics = ["f1", "recall", "precision"]
    if metric not in supported_metrics:
        raise RuntimeError(f"Metric `{metric}` not supported! Valid: {supported_metrics}")

    # Ensure class probabilities sum up to 1
    class_probs = class_probs / class_probs.sum(axis=1).reshape(-1, 1)
    num_classes = class_probs.shape[1]

    # Loop through each class
    best_thresholds = {}
    for i in range(num_classes):
        # Find the best threshold (example: maximizing F1 score)
        precision, recall, pr_thresholds = skmetrics.precision_recall_curve(encoded_labels == i, class_probs[:, i])
        f1_scores = 2 * (precision * recall) / (precision + recall)
        metric_to_score = {"f1": f1_scores, "recall": recall, "precision": precision}
        chosen_metric = metric_to_score[metric]

        # Get best probability threshold
        best_threshold = pr_thresholds[chosen_metric.argmax()]
        best_thresholds[i] = round(best_threshold, 4)

    return [best_thresholds[i] for i in range(num_classes)]


def get_age_bins(df_metadata, dset, split, age_col="age_years"):
    """
    Determine age bins based on the dataset and split.

    Parameters
    ----------
    df_metadata : pd.DataFrame
        Metadata table containing age column
    dset : str
        Name of the dataset.
    split : str
        Name of the data split.
    age_col : str, optional
        Name of age column, by default "age_years"

    Returns
    -------
    pd.Series
        List of string of age bins ages
    """
    age_splits = [18, 25, 40, 60, 80, 100]
    if dset in ["vindr_pcxr"] or "peds" in split:
        age_splits = list(range(19))

    # Assign each row to an age bin
    age_bins = pd.cut(df_metadata[age_col], bins=age_splits, right=False)

    # Map them back to integers, if it's only a single age per bin
    age_bins = age_bins.map(
        lambda x: str(x.left) if isinstance(x, pd.Interval) and (x.right - x.left == 1)
                  else str(x))
    age_bins = age_bins.astype(str)

    return age_bins


def stringify_dataset_split(dset, split=None):
    """
    Return a human-readable string for a given dataset-split combination.

    Parameters
    ----------
    dset : str
        Name of dataset
    split : str
        Name of data split

    Returns
    -------
    str
        Human-readable string
    """
    if split == "test":
        assert dset == "vindr_pcxr", "Only VinDr-PCXR has a valid used test split!"

    # Create mappings
    map_dset = {
        "vindr_pcxr": "VinDr-PCXR",
        "vindr_cxr": "VinDr-CXR",
        "padchest": "PadChest",
        "nih_cxr": "NIH",
        "nih_cxr18": "NIH",
        "chexbert": "CheXBERT",
    }
    map_split = {
        "test": "Healthy Children",
        "test_healthy_adult": "Healthy Adults",
        "test_peds": "Healthy Children",
        "test_adult_calib": "Healthy/Unhealthy Adults"
    }

    # CASE 1: Split provided
    if split is not None:
        return f"{map_split[split]} in {map_dset[dset]}"
    # CASE 2: Split not provided
    return map_dset[dset]


def extract_predictions_from_logits(out, idx_to_label=None):
    """
    Extract predictions and probabilities from model logits.

    Parameters
    ----------
    out : torch.Tensor
        Logits output from a model for a batch of samples.
    idx_to_label : dict, optional
        Optional dictionary mapping label indices to label names

    Returns
    -------
    dict
        Dictionary containing:
        - "class_probs": JSON string of per-class probabilities for each sample.
        - "prob": Probability of the predicted class for each sample.
        - "out": Maximum activation value from the logits for each sample.
        - "pred": Name of the predicted class for each sample.
    """
    idx_to_label = idx_to_label or {}
    accum_ret = {}

    # Get index of predicted label
    pred = torch.argmax(out, dim=1)
    pred = int(pred.detach().cpu())

    # Convert to probabilities
    prob = F.softmax(out, dim=1)
    prob_numpy = prob.detach().cpu().numpy().flatten()
    # Store per-class probabilities
    accum_ret["class_probs"] = json.dumps([round(i, 4) for i in prob_numpy.tolist()])
    # Get probability of largest class
    accum_ret["prob"] = round(prob_numpy.max(), 4)

    # Get maximum activation
    out = round(float(out.max().detach().cpu()), 4)
    accum_ret["out"] = out

    # Convert from encoded label to label name
    pred_label = idx_to_label.get(pred, pred)
    accum_ret["pred"] = pred_label

    return accum_ret


################################################################################
#                                  Main Flows                                  #
################################################################################
def infer_dset(eval_hparams: EvalHparams):
    """
    Perform inference on dset, and saves results

    Parameters
    ----------
    eval_hparams : EvalHparams
        Experiment evaluation hyperparameters
    """
    # Validate hyperparameters
    eval_hparams.validate_for_inference()

    # Load hyperparameters
    eval_hparams.load_hyperparameters()

    # 0. Create path to save predictions
    pred_save_path = create_save_path(eval_hparams)
    # Early return, if prediction already made
    if os.path.isfile(pred_save_path) and not eval_hparams["overwrite_existing"]:
        return

    ############################################################################
    #                              Load Model                                  #
    ############################################################################
    # Get experiment hyperparameters
    exp_hparams = eval_hparams["exp_hparams"]

    # Load existing model
    model = load_model.load_pretrained_model(
        model_dir=eval_hparams["model_dir"],
        hparams=exp_hparams,
        ckpt_option=eval_hparams["ckpt_option"],
    )

    ############################################################################
    #                              Load Data                                   #
    ############################################################################
    # Load metadata
    # CASE 1: Dataset is the same as in training
    if exp_hparams["dset"] == eval_hparams["dset"]:
        # NOTE: Need to perform split in data module
        dm = dataset.CXRDataModule(eval_hparams["data_hparams"])
        dm.setup()
        df_metadata = dm.filter_metadata(dset=eval_hparams["dset"], split=eval_hparams["split"])

        if len(df_metadata) == 0:
            assert False, "Unexpected error! Filtered metadata is empty..."
    # CASE 2: Dataset is different from the one in training
    else:
        # Load metadata
        # NOTE: When doing inference, do not filter negatives by default
        df_metadata = dataset.load_metadata(
            dset=eval_hparams["dset"],
            label_col=exp_hparams["label_col"],
            filter_negative=False,
        )
        # Filter on split
        df_metadata = df_metadata[df_metadata["split"] == eval_hparams["split"]]

    # Reset index
    df_metadata = df_metadata.reset_index(drop=True)
    # Add temporary label column
    label_col = exp_hparams["label_col"]
    if label_col not in df_metadata.columns:
        LOGGER.info(
            f"[Infer] Dset `{eval_hparams['dset']}` is missing label columns "
            f"`{label_col}`! Inserting placeholder..."
        )
        df_metadata[label_col] = -1.

    # Define CXR dataset to load data
    cxr_dataset = dataset.CXRDatasetDataFrame(df_metadata, exp_hparams)

    # Store model predictions
    df_preds = predict_with_cxr_dataset(model, cxr_dataset)
    df_metadata = pd.concat([df_metadata, df_preds], axis=1)

    # Save predictions
    df_metadata.to_csv(pred_save_path, index=False)


################################################################################
#                                User Interface                                #
################################################################################
if __name__ == "__main__":
    Fire({
        "main": EvalHparams,
        "check_adult_fpr_same": eval_are_adults_over_predicted_same_source,
        "check_adult_fpr_diff": eval_are_adults_over_predicted_different_source,
})
