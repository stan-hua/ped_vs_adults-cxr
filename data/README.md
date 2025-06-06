# Preparing Data


<p align="center">
  ⚡ Here, we provide details on how to download and process data for the chest X-ray cardiomegaly analyses.  ⚡ <br>
</p>

---

### 1. Prepare directories under `ped_vs_adult-cxr/`
```shell
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
```shell
# Go to CXR datasets directory
cd cxr_datasets

# Set PhysioNet username
USERNAME=(INSERT HERE)

# 1. Download VinDr-CXR on PhysioNet
wget -r -N -c -np --user $USERNAME --ask-password https://physionet.org/files/vindr-cxr/1.0.0/

# 2. Download VinDr-PCXR on PhysioNet
wget -r -N -c -np --user $USERNAME --ask-password https://physionet.org/files/vindr-pcxr/1.0.0/

# 3. Download MIMIC-CXR-JPEG on PhysioNet
wget -r -c -nc -np --user $USERNAME --ask-password https://physionet.org/files/mimic-cxr-jpg/2.1.0/
# NOTE: If resuming, you may want to change -c to -nc

# Move data to outside directory
# NOTE: Should now have the following dirs: (vindr-cxr, vindr-pcxr, mimic-cxr-jpg)
mv physionet.org/files/* .
rm -rf physionet.org
```

#### 2.2. From AcademicTorrents (NIH, PadChest)
```shell
# Follow torrent download instructions on `torchxrayvision` 
# 1. NIH Chest X-ray 18 : https://mlmed.org/torchxrayvision/datasets.html#torchxrayvision.datasets.NIH_Dataset
# 2. PadChest Dataset: https://mlmed.org/torchxrayvision/datasets.html#torchxrayvision.datasets.PC_Dataset
```

#### 2.3. From Azure (CheXpert)
```shell
# NOTE: Assuming you're in the repo home directory
DIR_CXR="$PWD/data/cxr_datasets"

# Get a download link from Stanford AIMI website
# Link: https://stanfordaimi.azurewebsites.net/datasets/8cbd9ed4-2eb9-4565-affc-111cf4f7ebe2
URL='[INSERT HERE]'

# Download to the data directory
# NOTE: Install azcopy following https://learn.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10
azcopy copy "$URL" "${DIR_CXR}/" --recursive

# Rename folder
# NOTE: Directory name may have changed
mv ${DIR_CXR}/chexpertchestxrays-u20210408 "${DIR_CXR}/chexbert"
DIR_CHEXBERT="${DIR_CXR}/chexbert"

# Unzip and remove the zip files
unzip "$DIR_CHEXBERT/*.zip"
# rm "$DIR_CHEXBERT/*.zip"          # Uncomment when it's safe to do so

# Create single directory for all training data
DIR_TRAIN="$DIR_CHEXBERT/train"
mkdir $DIR_TRAIN

# Move all training patients
cd ${DIR_CHEXBERT}
mv "${DIR_CHEXBERT}/CheXpert-v1.0 batch 2 (train 1)"/* $DIR_TRAIN/
mv "${DIR_CHEXBERT}/CheXpert-v1.0 batch 3 (train 2)"/* $DIR_TRAIN/
mv "${DIR_CHEXBERT}/CheXpert-v1.0 batch 4 (train 3)"/* $DIR_TRAIN/

# Extract validation set directory (N=200)
mv "${DIR_CHEXBERT}/CheXpert-v1.0 batch 1 (validate & csv)"/* ${DIR_CHEXBERT}/

# When ready delete the empty directories
find . -maxdepth 1 -type d -empty -delete
```


### 3. Prepare data

#### 3.1. VinDr-CXR & VinDr-PCXR Datasets
Processes metadata and converts DICOM to (224x224) PNG images in `train_test_processed`
```shell
# 1. Process VinDr-CXR metadata and images
python -m scripts.prep_data vindr_cxr_metadata_dicom
python -m scripts.prep_data vindr_images vindr_cxr
python -m scripts.prep_data vindr_cxr_metadata_post_process

# 2. Process VinDr-PCXR metadata and images
python -m scripts.prep_data vindr_pcxr_metadata
python -m scripts.prep_data vindr_images vindr_pcxr
```

#### 3.2. Other Datasets
```shell
# 1. NIH X-ray 18 Dataset
python -m scripts.prep_data nih_cxr18_metadata

# 2. PadChest Dataset
python -m scripts.prep_data padchest_metadata

# 3. CheXBERT Dataset
python -m scripts.prep_data chexbert_metadata
```