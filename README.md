# Adult vs. Children CXR
An evaluation of chest-xray models trained on adults vs. children

---

## Pre-requisites
### 0. Install required Python packages
```
# (Optional) Create a new virtual environment with conda
conda create --name peds_cxr python=3.9.19
conda activate peds_cxr

# Install required dependencies
pip install requirements.txt
```

### 1. Prepare directories under `ped_vs_adult-cxr/`
```
# Create a data directory (Alternatively, create a symlink to a data directory)
# ln -s DATA_DIRECTORY data
mkdir data/
cd data/

# Create directories for CXR datasets, metadata and later save data
mkdir cxr_datasets
mkdir metadata
mkdir save_data
```

### 2. Download CXR data from PhysioNet
#### 2.1. From PhysioNet (VinDr-CXR, VinDr-PCXR, MIMIC-CXR)
```
# Go to CXR datasets directory
cd cxr_datasets

# Set PhysioNet username
USERNAME=(INSERT HERE)

# 1. Download VinDr-CXR on PhysioNet
wget -r -N -c -np --user $USERNAME --ask-password https://physionet.org/files/vindr-cxr/1.0.0/

# 2. Download VinDr-PCXR on PhysioNet
wget -r -N -c -np --user $USERNAME --ask-password https://physionet.org/files/vindr-pcxr/1.0.0/

# 3. Download MIMIC-CXR-JPEG on PhysioNet
wget -r -N -c -np --user $USERNAME --ask-password https://physionet.org/files/mimic-cxr-jpg/2.1.0/

# Move data to outside directory
# NOTE: Should now have the following dirs: (vindr-cxr, vindr-pcxr, mimic-cxr-jpg)
mv physionet.org/files/* .
rm -rf physionet.org
```

#### 2.2. From AcademicTorrents (NIH, PadChest)
```
# Follow torrent download instructions on `torchxrayvision` 
# 1. NIH Chest X-ray 18 : https://mlmed.org/torchxrayvision/datasets.html#torchxrayvision.datasets.NIH_Dataset
# 2. PadChest Dataset: https://mlmed.org/torchxrayvision/datasets.html#torchxrayvision.datasets.PC_Dataset
```

### 3. Prepare data

#### 3.1. VinDr-CXR & VinDr-PCXR Datasets
Processes metadata and converts DICOM to (224x224) PNG images in `train_test_processed`
```
# 1. Process VinDr-CXR metadata and images
python -m src.scripts.prep_data vindr_cxr_metadata_dicom
python -m src.scripts.prep_data vindr_images vindr_cxr
python -m src.scripts.prep_data vindr_cxr_metadata_post_process

# 2. Process VinDr-PCXR metadata and images
python -m src.scripts.prep_data vindr_pcxr_metadata
python -m src.scripts.prep_data vindr_images vindr_pcxr
```

#### 3.2. NIH & PadChest Datasets
```
# 1. NIH X-ray 18 Dataset
python -m src.scripts.prep_data nih_cxr18_metadata

# 2. PadChest Dataset
python -m src.scripts.prep_data padchest_metadata

```

---

## Model Training/Evaluation
### 1. Train Model
```
# Config file under `config/train_model/`
CONFIG_PATH = "param_sweep/exp_param_sweep-convnext_baseline.ini"

# Option 1. Run in current shell
python -m src.scripts.train_model -c $CONFIG_PATH

# Option 2. Submit job to SLURM server
sbatch slurm/train_model.sh
```
