"""
load_model.py

Description: Contains utility functions for instantiating model classes/objects.

Note: `hparams` is a direct dependence on arguments in `model_training.py`.
"""

# Standard libraries
import logging
import os
import re
from collections import defaultdict
from pathlib import Path

# Non-standard libraries
import lightning as L
import timm
import torch
import torchmetrics
import torchvision
import yaml
from torchvisions.transforms import v2
from torchvision.models.feature_extraction import create_feature_extractor

# Custom libraries
from configs import constants


################################################################################
#                                  Constants                                   #
################################################################################
# Configure logging
LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


################################################################################
#                                    Class                                     #
################################################################################
class ModelWrapper(L.LightningModule):
    """
    ModelWrapper class.

    Note
    ----
    Used to provide a simple interface for training an arbitrary model
    """

    def __init__(self, hparams):
        """
        Initialize EfficientNetPL object.

        Parameters
        ----------
        num_classes : int, optional
            Number of classes to predict, by default 5
        img_size : tuple, optional
            Expected image's (height, width), by default (256, 256)
        optimizer : str, optional
            Choice of optimizer, by default "adamw"
        lr : float, optional
            Optimizer learning rate, by default 0.0001
        momentum : float, optional
            If SGD optimizer, value to use for momentum during SGD, by
            default 0.9
        weight_decay : float, optional
            Weight decay value to slow gradient updates when performance
            worsens, by default 0.0005
        use_gradcam_loss : bool, optional
            If True, add auxiliary segmentation-attention GradCAM loss, by
            default False.
        use_mixup_aug : bool, optional
            If True, use Mixup augmentation during training, by default False
        freeze_weights : bool, optional
            If True, freeze convolutional weights, by default False.
        effnet_name : str, optional
            Name of EfficientNet backbone to use
        """
        # Save hyperparameters
        self.save_hyperparameters(hparams)

        # Load model
        self.network = load_network(hparams)

        # If specified, store training-specific augmentations
        # NOTE: These augmentations require batches of input
        self.train_aug = None
        if self.hparams["use_mixup_aug"]:
            self.train_aug = v2.MixUp(num_classes=self.hparams["num_classes"])

        # Define loss
        self.loss = torch.nn.CrossEntropyLoss()

        # Evaluation metrics
        self.split_to_acc = torch.nn.ModuleDict({
            f"{split}_acc": torchmetrics.Accuracy(
                num_classes={self.hparams.num_classes},
                task='multiclass'
            ) for split in ["train", "val", "test"]
        })
        # Store outputs
        self.dset_to_outputs = {"train": [], "val": [], "test": []}


    ############################################################################
    #                             Optimization                                 #
    ############################################################################
    def configure_optimizers(self):
        """
        Initialize and return optimizer (AdamW or SGD).

        Returns
        -------
        dict
            Contains optimizer and LR scheduler
        """
        # Get filtered or all parameters
        params = self.parameters()
        
        # Create optimizer
        if self.hparams.optimizer == "adamw":
            optimizer = torch.optim.AdamW(params,
                                          lr=self.hparams.lr,
                                          weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer == "sgd":
            optimizer = torch.optim.SGD(params,
                                        lr=self.hparams.lr,
                                        momentum=self.hparams.momentum,
                                        weight_decay=self.hparams.weight_decay)

        # Prepare return
        ret = {
            "optimizer": optimizer,
        }

        return ret


    def on_train_epoch_start(self):
        """
        Deal with Stochastic Weight Averaging (SWA) Issue in Lightning<=2.3.2
        """
        # HACK: Fix issue with Stochastic Weight Optimization on the last epoch
        if self.hparams.get("swa") and self.current_epoch == self.trainer.max_epochs - 1:
            # Workaround to always save the last epoch until the bug is fixed in lightning (https://github.com/Lightning-AI/lightning/issues/4539)
            self.trainer.check_val_every_n_epoch = 1

            # Disable backward pass for SWA until the bug is fixed in lightning (https://github.com/Lightning-AI/lightning/issues/17245)
            self.automatic_optimization = False
        else:
            self.automatic_optimization = True


    ############################################################################
    #                          Per-Batch Metrics                               #
    ############################################################################
    def training_step(self, train_batch, batch_idx):
        """
        Batch training step

        Parameters
        ----------
        train_batch : tuple
            Contains (img tensor, metadata dict)
        batch_idx : int
            Training batch index

        Returns
        -------
        torch.FloatTensor
            Loss for training batch
        """
        data, metadata = train_batch
        B = len(data)

        # Get label (and modify for loss if using MixUp)
        y_true = metadata["label"]

        # If specified, apply MixUp augmentation on images
        y_true_aug = y_true
        if self.hparams.get("use_mixup_aug"):
            data, y_true_aug = self.train_aug(data, y_true)

        # Get prediction
        out = self.network(data)
        y_pred = torch.argmax(out, dim=1)

        # Compute loss
        loss = self.loss(out, y_true_aug)

        # Log training metrics
        self.split_to_acc["train"].update(y_pred, y_true)

        # Prepare result
        ret = {
            "loss": loss.detach().cpu(),
        }
        self.dset_to_outputs["train"].append(ret)

        return loss


    def validation_step(self, val_batch, batch_idx):
        """
        Batch validation step

        Parameters
        ----------
        val_batch : tuple
            Contains (img tensor, metadata dict)
        batch_idx : int
            Validation batch index

        Returns
        -------
        torch.FloatTensor
            Loss for validation batch
        """
        data, metadata = val_batch

        # Get prediction
        out = self.network(data)
        y_pred = torch.argmax(out, dim=1)

        # Get label
        y_true = metadata["label"]

        # Get loss
        loss = self.loss(out, y_true)

        # Log validation metrics
        self.split_to_acc["val"].update(y_pred, y_true)

        # Prepare result
        ret = {
            "loss": loss.detach().cpu(),
            "y_pred": y_pred.detach().cpu(),
            "y_true": y_true.detach().cpu(),
        }
        self.dset_to_outputs["val"].append(ret)

        return ret


    def test_step(self, test_batch, batch_idx):
        """
        Batch test step

        Parameters
        ----------
        test_batch : tuple
            Contains (img tensor, metadata dict)
        batch_idx : int
            Test batch index

        Returns
        -------
        torch.FloatTensor
            Loss for test batch
        """
        data, metadata = test_batch

        # Get prediction
        out = self.network(data)
        y_pred = torch.argmax(out, dim=1)

        # Get label
        y_true = metadata["label"]

        # Get loss
        loss = self.loss(out, y_true)

        # Log test metrics
        self.split_to_acc["test"].update(y_pred, y_true)

        # Prepare result
        ret = {
            "loss": loss.detach().cpu(),
            "y_pred": y_pred.detach().cpu(),
            "y_true": y_true.detach().cpu(),
        }
        self.dset_to_outputs["test"].append(ret)

        return ret


    ############################################################################
    #                            Epoch Metrics                                 #
    ############################################################################
    def on_train_epoch_end(self):
        """
        Compute and log evaluation metrics for training epoch.
        """
        outputs = self.dset_to_outputs["train"]
        loss = torch.stack([d['loss'] for d in outputs]).mean()
        acc = self.split_to_acc["train"].compute()

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)

        self.split_to_acc["train"].reset()

        # Clean stored output
        self.dset_to_outputs["train"].clear()


    def on_validation_epoch_end(self):
        """
        Compute and log evaluation metrics for validation epoch.
        """
        outputs = self.dset_to_outputs["val"]
        loss = torch.tensor([o["loss"] for o in outputs]).mean()
        acc = self.split_to_acc["val"].compute()

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

        self.split_to_acc["val"].reset()

        # Create confusion matrix
        if self.hparams.get("use_comet_logger"):
            self.logger.experiment.log_confusion_matrix(
                y_true=torch.cat([o["y_true"] for o in outputs]),
                y_predicted=torch.cat([o["y_pred"] for o in outputs]),
                labels=constants.LABEL_PART_TO_CLASSES[self.hparams.label_part]["classes"],
                title="Validation Confusion Matrix",
                file_name="val_confusion-matrix.json",
                overwrite=False,
            )

        # Clean stored output
        self.dset_to_outputs["val"].clear()


    def on_test_epoch_end(self):
        """
        Compute and log evaluation metrics for test epoch.
        """
        outputs = self.dset_to_outputs["test"]
        dset = f'test'

        loss = torch.tensor([o["loss"] for o in outputs]).mean()
        acc = eval(f'self.{dset}_acc.compute()')

        self.log(f'{dset}_loss', loss)
        self.log(f'{dset}_acc', acc)

        exec(f'self.{dset}_acc.reset()')

        # Create confusion matrix
        if self.hparams.get("use_comet_logger"):
            self.logger.experiment.log_confusion_matrix(
                y_true=torch.cat([o["y_true"].cpu() for o in outputs]),
                y_predicted=torch.cat([o["y_pred"].cpu() for o in outputs]),
                labels=constants.LABEL_PART_TO_CLASSES[self.hparams.label_part]["classes"],
                title="Test Confusion Matrix",
                file_name="test_confusion-matrix.json",
                overwrite=False,
            )

        # Clean stored output
        self.dset_to_outputs["test"].clear()


    @torch.no_grad()
    def extract_embeds(self, inputs):
        """
        Extract embeddings from input image

        Parameters
        ----------
        inputs : torch.Tensor
            Model input

        Returns
        -------
        torch.Tensor
            Extracted embeddings
        """
        return extract_features(self.hparams, self.network, inputs)


################################################################################
#                               Helper Functions                               #
################################################################################
def extract_features(hparams, model, inputs):
    """
    Extract features from model.

    Parameters
    ----------
    hparams : dict
        Hyperparameters
    model : torch.nn.Module
        Neural network (not wrapper)
    inputs : torch.Tensor
        Model input

    Returns
    -------
    torch.Tensor
        Extracted features
    """
    model_name = hparams["model_name"]
    model_provider = hparams["model_provider"]
    extractor = None
    # CASE 1: Torchvision model
    if model_provider == "torchvision":
        if model_name == "resnet50":
            return_nodes = {"layer4": "layer4"}
            extractor = create_feature_extractor(model, return_nodes)
    # CASE 2: Timm model
    elif model_provider == "timm":
        extractor = model.forward_features

    # Raise error, if not implemented
    if extractor is None:
        raise NotImplementedError(
            "Feature extraction not implemented for "
            f"`{model_provider}/{model_name}`"
        )

    return extractor(inputs)


def load_network(hparams):
    """
    Load model in PyTorch

    Parameters
    ----------
    hparams : dict
        Experiment hyperparameters

    Returns
    ------
    torch.nn.Module
        Loaded model
    """
    # Load model backbone using torchvision or timm
    model_provider = hparams.get("model_provider", "torchvision")
    if model_provider == "torchvision":
        model_cls = getattr(torchvision.models, hparams["model_name"])
        model = model_cls(num_classes=hparams["num_classes"])
    elif model_provider == "timm":
        model = timm.create_model(
            model_name=hparams["model_name"],
            num_classes=hparams["num_classes"],
        )
    else:
        raise RuntimeError(f"Invalid model_provider specified! `{model_provider}`")

    return model


def load_pretrained_from_exp_name(exp_name, ckpt_option="best",
                                  **overwrite_hparams):
    """
    Load pretrained model from experiment name.

    Parameters
    ----------
    exp_name : str
        Name of experiment
    ckpt_option : str
        Choice of "best" checkpoint (based on validation set) or "last"
        checkpoint file, by default "best"

    Returns
    -------
    torch.nn.Module
        Pretrained model
    """
    # 0. Get experiment directory, where model was trained
    model_dir = get_exp_dir(exp_name)

    # 1 Get experiment hyperparameters
    hparams = get_hyperparameters(model_dir)
    hparams.update(overwrite_hparams)

    # 2. Load existing model and send to device
    # 2.1 Get checkpoint path
    if ckpt_option == "last":
        LOGGER.info("Loading `last` pre-trained model weights...")
        ckpt_path = find_last_ckpt_path(model_dir)
    else:
        LOGGER.info("Loading `best` pre-trained model weights...")
        ckpt_path = find_best_ckpt_path(model_dir)

    # 2.2 Load model
    model = ModelWrapper.load_from_checkpoint(checkpoint_path=ckpt_path)

    # If specified, compile model
    if hparams.get("torch_compile"):
        LOGGER.debug("Compiling model...")
        model = torch.compile(model)
        LOGGER.debug("Compiling model...DONE")

    return model


def get_exp_dir(exp_name, on_error="raise"):
    """
    Get experiment directory, given experiment name.

    Parameters
    ----------
    exp_name : str
        Experiment name
    on_error : str, optional
        If "raise", raises an error, if expected directory does not exist. If
        "ignore", simply returns None, by default "raise".

    Returns
    -------
    str
        Path to experiment directory, where model was trained
    """
    # INPUT: Verify provided `on_error` is valid
    assert on_error in ("raise", "ignore"), \
        "`on_error` must be one of ('raise', 'ignore')"

    # Create full path
    model_dir = os.path.join(constants.DIR_RESULTS, exp_name)

    # Raise error, if model directory does not exist
    if not os.path.exists(model_dir):
        if on_error == "raise":
            raise RuntimeError(f"`exp_name` ({exp_name}) provided does not lead"
                               " to a valid model training directory")
        model_dir = None
    return model_dir


def get_hyperparameters(hparam_dir=None, exp_name=None):
    """
    Load hyperparameters from model training directory. If not provided, return
    default hyperparameters.

    Parameters
    ----------
    hparam_dir : str
        Path to model training directory containing hyperparameters.
    exp_name : str, optional
        If `hparam_dir` not provided but `exp_name` is, use to find model
        directory, by default None.

    Returns
    -------
    dict
        Hyperparameters
    """
    try:
        # Get path to last checkpoint
        ckpt_path = find_last_ckpt_path(path_exp_dir=hparam_dir, exp_name=exp_name)

        # Load checkpoint and get pyerparameters
        hparams = torch.load(
            ckpt_path,
            map_location=lambda storage, loc: storage
        )["hyper_parameters"]
    except Exception as error_msg:
        display_msg = f"Failed to retrieve hyperparameters! \n\tError:{error_msg}"
        raise RuntimeError(display_msg)

    return hparams


def find_best_ckpt_path(path_exp_dir=None, exp_name=None):
    """
    Finds the path to the best model checkpoint.

    Parameters
    ----------
    path_exp_dir : str
        Path to a trained model directory
    exp_name : str
        Experiment name

    Returns
    -------
    str
        Path to PyTorch Lightning best model checkpoint

    Raises
    ------
    RuntimeError
        If no valid ckpt files found
    """
    # INPUT: Ensure at least one of `path_exp_dir` or `exp_name` is provided
    assert path_exp_dir or exp_name

    # If only `exp_name` provided, attempt to find experiment training directory
    if not path_exp_dir and exp_name:
        path_exp_dir = get_exp_dir(exp_name, on_error="raise")

    # Look for checkpoint files
    ckpt_paths = [str(path) for path in Path(path_exp_dir).rglob("*.ckpt")]

    # Remove last checkpoint. NOTE: The other checkpoint is for the best epoch
    ckpt_paths = [path for path in ckpt_paths if "last.ckpt" not in path]

    if not ckpt_paths:
        raise RuntimeError("No best epoch model checkpoint (.ckpt) found! "
                           f"\nDirectory: {path_exp_dir}")

    if len(ckpt_paths) > 1:
        raise RuntimeError("More than 1 checkpoint file (.ckpt) found besides "
                           f"last.ckpt! \nDirectory: {path_exp_dir}")

    return ckpt_paths[0]


def find_last_ckpt_path(path_exp_dir=None, exp_name=None):
    """
    Finds the path to the last model checkpoint.

    Parameters
    ----------
    path_exp_dir : str
        Path to a trained model directory
    exp_name : str
        Experiment name

    Returns
    -------
    str
        Path to PyTorch Lightning last model checkpoint

    Raises
    ------
    RuntimeError
        If no valid ckpt files found
    """
    # INPUT: Ensure at least one of `path_exp_dir` or `exp_name` is provided
    assert path_exp_dir or exp_name

    # If only `exp_name` provided, attempt to find experiment training directory
    if not path_exp_dir and exp_name:
        path_exp_dir = get_exp_dir(exp_name, on_error="raise")

    # Look for checkpoint files
    ckpt_paths = [str(path) for path in Path(path_exp_dir).rglob("*.ckpt")]

    # Get last checkpoint
    ckpt_paths = [path for path in ckpt_paths if "last.ckpt" in path]

    # Raise error, if no ckpt paths  found
    if not ckpt_paths:
        raise RuntimeError("No last epoch model checkpoint (.ckpt) found! "
                           f"\nDirectory: {path_exp_dir}")

    return ckpt_paths[0]


def overwrite_model(dst_model, src_model=None, src_state_dict=None):
    """
    Given a (new) model, overwrite its existing parameters based on a source
    model or its provided state dict.

    Note
    ----
    One of `src_model` or `src_state_dict` must be provided.

    Parameters
    ----------
    dst_model : torch.nn.Module
        Model whose weights to overwrite
    src_model : torch.nn.Module, optional
        Pretrained model whose weights to use in overwriting, by default None
    src_state_dict : dict, optional
        Pretrained model's state dict, by default None

    Returns
    -------
    torch.nn.Module
        Model whose weights were overwritten (in-place)
    """
    # INPUT: Ensure at least one of `src_model` or `src_state_dict` is provided
    assert src_model is not None or src_state_dict is not None, \
        "At least one of `src_model` or `src_state_dict` must be provided!"

    # Get model state dicts
    pretrained_weights = src_state_dict if src_state_dict is not None \
        else src_model.state_dict()
    new_weights = dst_model.state_dict()

    # Get names of overlapping weights
    pretrained_weight_names = set(pretrained_weights.keys())
    new_weight_names = set(new_weights.keys())
    overlapping_weights = pretrained_weight_names.intersection(new_weight_names)

    # Log skipped weights, due to incompatibility
    missing_weights = list(pretrained_weight_names.difference(new_weight_names))
    if missing_weights:
        LOGGER.warning("Loading pretrained model, where the following weights "
                       "were incompatible: \n\t%s",
                       "\n\t".join(missing_weights))

    # Overwrite overlapping weights with pretrained
    for weight_name in list(overlapping_weights):
        new_weights[weight_name] = pretrained_weights[weight_name]

    # Update the model's weights
    dst_model.load_state_dict(new_weights)

    LOGGER.info("Loaded weights from pretrained model successfully!")
    return dst_model


def find_layers_in_model(model, layer_type):
    """
    Find specified layers in model.

    Parameters
    ----------
    model : torch.nn.Module
        Model
    layer_type : class
        Class of layer desired

    Returns
    -------
    lists
        List of model indices containing layer
    """
    fc_idx = []
    for idx, layer in enumerate(model.children()):
        if isinstance(layer, layer_type):
            fc_idx.append(idx)

    return fc_idx
