"""
eval_model.py

Description: Used to evaluate a trained model's performance on other view
             labeling datasets.
"""

# Standard libraries
import json
import logging
import os
import sys
from copy import deepcopy
from dataclasses import dataclass, field

# Non-standard libraries
import pandas as pd
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
from src.utils.data import load_data, dataset, utils as data_utils
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
OVERWRITE_EXISTING = True

# Flag to use CometML logger, by default
# NOTE: Set to False, if you don't want to use Comet ML by default
DEFAULT_USE_COMET_ML = True

# Random seed
SEED = 42


################################################################################
#                                   Classes                                    #
################################################################################
@dataclass
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
        self.dset_split_to_preds[key] = self.df_pred

        # Reset dset/split
        self.dset, self.split = orig_dset, orig_split


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
        overwrite_hparams = load_data.create_eval_hparams(self.dset, self.split)
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
        Dictionary mapping label indices to label names, by default {}

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
        loss = None
        if labels:
            label = torch.tensor([labels[idx]], dtype=int, device=out.device)
            loss = round(float(F.cross_entropy(out, label).detach().cpu().item()), 4)
        accum_ret["loss"].append(loss)

        # Get index of predicted label
        pred = torch.argmax(out, dim=1)
        pred = int(pred.detach().cpu())

        # Convert to probabilities
        prob = F.softmax(out, dim=1)
        prob_numpy = prob.detach().cpu().numpy().flatten()
        # Store per-class probabilities
        accum_ret["class_probs"].append(json.dumps([round(i, 4) for i in prob_numpy.tolist()]))
        # Get probability of largest class
        accum_ret["prob"].append(round(prob_numpy.max(), 4))

        # Get maximum activation
        out = round(float(out.max().detach().cpu()), 4)
        accum_ret["out"].append(out)

        # Convert from encoded label to label name
        pred_label = idx_to_label.get(pred, pred)
        accum_ret["pred"].append(pred_label)

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
    # Ensure VinDr-CXR (test) and VinDr-PCXR (test) predictions are loaded
    for dset in ["vindr_cxr", "vindr_pcxr"]:
        eval_hparams.preload_inference(dset=dset, split="test")

    # Load experiment hyperparameters
    eval_hparams.load_hyperparameters()
    label_col = eval_hparams["exp_hparams"]["label_col"]

    # Ensure findings directory exists
    os.makedirs(constants.DIR_FINDINGS, exist_ok=True)

    # Get predictions
    df_pred_cxr = eval_hparams.dset_split_to_preds[("vindr_cxr", "test")]
    df_pred_pcxr = eval_hparams.dset_split_to_preds[("vindr_pcxr", "test")]

    # Compute thresholds from VinDr-CXR
    preds = df_pred_cxr["pred"].to_numpy()
    class_probs = np.stack(df_pred_cxr["class_probs"].map(lambda x: np.array(json.loads(x))))
    encoded_labels = df_pred_cxr[label_col].to_numpy()
    thresholds_cxr = compute_thresholds(class_probs, encoded_labels, metric="f1")

    # Only get threshold for the positive class
    thresholds_cxr = thresholds_cxr[1]
    LOGGER.info(f"[VinDr-CXR (Test)] Calibrated Threshold: {thresholds_cxr}")

    # Re-compute accuracy on VinDr-CXR with new probability
    recalibrated_preds = (class_probs[:, 1] >= thresholds_cxr).astype(int)
    metric_funcs = [
        ("Accuracy", skmetrics.accuracy_score),
        ("F1", skmetrics.f1_score),
        ("Recall", skmetrics.recall_score),
        ("Precision", skmetrics.precision_score)
    ]
    metrics_before_vs_after = {}
    for metric_name, metric_func in metric_funcs:
        metric_before = round(100*metric_func(encoded_labels, preds), 2)
        metric_after = round(100*metric_func(encoded_labels, recalibrated_preds), 2)
        metrics_before_vs_after[f"{metric_name.lower()}-p=0.5"] = metric_before
        metrics_before_vs_after[f"{metric_name.lower()}-p={thresholds_cxr}"] = metric_after

        LOGGER.info(f"[VinDr-CXR (Test)] Pre-Calibration {metric_name}: {metric_before}")
        LOGGER.info(f"[VinDr-CXR (Test)] Post-Calibration {metric_name}: {metric_after}")
    metrics_before_vs_after = {"vindr_cxr-test": metrics_before_vs_after}

    # Filter VinDr-PCXR dataset for those that are healthy
    LOGGER.info(f"[VinDr-PCXR (Test)] Filtering for healthy patients...")
    df_pred_pcxr = df_pred_pcxr[~df_pred_pcxr["Has Finding"].astype(bool)]

    # Filter VinDr-PCXR dataset for those with valid age annotations <= 10
    df_pred_pcxr = data_utils.extract_age("vindr_pcxr", df_pred_pcxr.copy())
    df_pred_pcxr = df_pred_pcxr[df_pred_pcxr["age_years"] <= 10].dropna(subset=["age_years"]).copy()

    # Use calibrated threshold on VinDr-PCXR
    pcxr_class_probs = np.stack(df_pred_pcxr["class_probs"].map(lambda x: np.array(json.loads(x))))
    calibrated_preds = (pcxr_class_probs[:, 1] >= thresholds_cxr).astype(int)
    prop_positive = round(100*(calibrated_preds == 1).mean(), 2)
    LOGGER.info(f"[VinDr-PCXR (Test)] Calibrated % Predicted Positive: {prop_positive}")

    # Stratify positive predictions by age
    df_pred_pcxr["age_years"] = df_pred_pcxr["age_years"].astype(int)
    df_pred_pcxr[label_col] = df_pred_pcxr[label_col].astype(int)
    df_pred_pcxr["pred_calibrated"] = calibrated_preds
    counts = df_pred_pcxr.groupby("age_years").apply(
        lambda df: pd.DataFrame(df["pred_calibrated"].value_counts()))
    counts = counts.rename(columns={"count": "Pred. Count"})
    percs = df_pred_pcxr.groupby("age_years").apply(
        lambda df: pd.DataFrame((100*df["pred_calibrated"].value_counts() / len(df)).round(2)))
    percs = percs.rename(columns={"count": "Pred. Percentage"})
    df_pcxr_calib_counts = pd.concat([counts, percs], axis=1)

    # Stringify label column
    label_col_str = label_col.lower().replace(' ', '_')

    # Save metrics
    save_path = os.path.join(constants.DIR_FINDINGS, f"{label_col_str}-metrics.json")
    with open(save_path, "w") as f:
        json.dump(metrics_before_vs_after, f, indent=4)

    # Save table
    save_path = os.path.join(constants.DIR_FINDINGS, f"pcxr_{label_col_str}_calib_counts.csv")
    df_pcxr_calib_counts.to_csv(save_path)


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


def calculate_accuracy(df_pred, label_col="label", pred_col="pred"):
    """
    Given a table of predictions with columns "label" and "pred", compute
    accuracy rounded to 4 decimal places.

    Parameters
    ----------
    df_pred : pd.DataFrame
        Model predictions. Each row contains a label,
        prediction, and other patient and sequence-related metadata.
    label_col : str, optional
        Name of label column, by default "label"
    pred_col : str, optional
        Name of label column, by default "pred"

    Returns
    -------
    float
        Accuracy rounded to 4 decimal places
    """
    # Early return, if empty
    if df_pred.empty:
        return "N/A"

    # Compute accuracy
    acc = skmetrics.accuracy_score(df_pred[label_col], df_pred[pred_col])

    # Round decimals
    acc = round(acc, 4)

    return acc


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
            method='bca',
            size=1-alpha,
            tail='two').flatten()
        # Round to 4 decimal places
        ci_bounds = np.round(ci_bounds, 4)
    except RuntimeError: # NOTE: May occur if all labels are predicted correctly
        ci_bounds = np.nan, np.nan

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
        dm = dataset.VinDr_DataModule(eval_hparams["data_hparams"])
        dm.setup()
        df_metadata = dm.filter_metadata(dset=eval_hparams["dset"], split=eval_hparams["split"])

        if len(df_metadata) == 0:
            assert False, "Unexpected error! Filtered metadata is empty..."
    # CASE 2: Dataset is different from the one in training
    else:
        # Load metadata
        # NOTE: When doing inference, do not filter negatives by default
        df_metadata = dataset.load_metadata(dset=eval_hparams["dset"], filter_negative=False)
        # Filter on split
        df_metadata = df_metadata[df_metadata["split"] == eval_hparams["split"]]

    # Reset index
    df_metadata = df_metadata.reset_index(drop=True)

    # Check if label exists in new dataset
    labels = []
    if exp_hparams["label_col"] in df_metadata.columns:
        labels = df_metadata[exp_hparams["label_col"]].tolist()

    # Store model predictions
    img_paths = (df_metadata["dirname"] + "/" + df_metadata["filename"]).tolist()
    df_preds = predict_on_images(model, img_paths, exp_hparams, labels)
    df_metadata = pd.concat([df_metadata, df_preds], axis=1)

    # Save predictions
    df_metadata.to_csv(pred_save_path, index=False)


################################################################################
#                                User Interface                                #
################################################################################
if __name__ == '__main__':
    Fire(EvalHparams)
