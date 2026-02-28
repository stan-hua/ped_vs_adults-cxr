


<h1 align="center">Lack of children in public medical imaging data points to growing age bias in biomedical AI</h1>


<p align="center">
  <img src="assets/summary.gif" alt="Logo" style="width: 80%; display: block; margin: auto;">
</p>

<p align="center">
  <a href="[OPT FILL: Path/link to paper]"><img src="https://img.shields.io/badge/medrXiv-2405.01535-blue.svg" alt="arXiv"></a>
  <a href="https://huggingface.co/stan-hua/peds_vs_adult-cxr"><img src="https://img.shields.io/badge/Hugging%20Face-Model-yellow" alt="Hugging Face Models"></a>
</p>

<p align="center">
  ⚡ In this repository, we provide all data and code artifacts from the analysis.  🚀 ⚡ <br>
</p>


---

## 🌲 About the Repo

<!-- OPTIONAL: Create Repository Structure Automatically
pip install rptree
rptree -d .
[OPT FILL: Copy structure to this README]
-->

```shell
ped_vs_adult-cxr/
├── config/                   # Data directory
│   ├── configspecs/            # Defines model training hyperparameters
│   └── train_model/            # Specified parameters for each model
├── data/                   # Data directory
│   ├── cxr_datasets/               # Contains raw/processed datasets
│   ├── metadata/                   # Contains dataset annotations and CXR metadata
│   └── save_data/              # Saved artifacts from training/inference
│       ├── train_runs/             # Contains training runs (model checkpoints)
│       └── figures/                # Contains all generated figures
├── scripts/                # Contains scripts to run
│  ├── data/                   # Data processing scripts
│  └── model/                  # Model training/evaluation scripts
├── slurm/                  # Contains sbatch scripts for running on SLURM server
└── src/
    └── utils/                  # Contains utility funcitons
        ├── data/                   # for data loading and visualization
        └── model/                  # for model loading
```



## 💴 About the Data

**A. Dataset Review**

> All dataset annotations are provided under [data/metadata/public_datasets_metadata.xlsx](./data/metadata/public_datasets_metadata.xlsx)
> 
> For analyzed pediatric datasets, please see [data/metadata/public_pediatric_datasets_metadata.xlsx](./data/metadata/public_pediatric_datasets_metadata.xlsx)

If you would like to re-do the analysis, please refer to step 1 below.

**B. Cardiomegaly Case Study**

> Please refer to [data/README.md](./data/README.md) for downloading and preprocessing the chest X-ray datasets used.

> We provide generated figures in [./data/save_data/figures](./data/save_data/figures) and [./data/save_data/findings/PedsVsAdult_CXR](./data/save_data/findings/PedsVsAdult_CXR), predictions saved in [./data/save_data/inference](./data/save_data/inference), and model hyperparameters in [./config/train_model](./data/config/train_model)



If you would like to re-train the models, please follow steps 2 and 3 below. If you would only like to re-generate the figures, skip to step 4.


## 🔧 Installation

**Clone Repository:**
```shell
git clone https://github.com/stan-hua/ped_vs_adults-cxr
cd ped_vs_adults-cxr
```

**Install Packages:**

***(Recommended) Option 1. Pixi*** (est. 3 mins)

Pixi is a light-weight package manager, localized to the project's directory.
```shell
# Install pixi
curl -fsSL https://pixi.sh/install.sh | sh

# Install packages (CPU)
pixi shell -e torch-cpu

# If a GPU is available for training, uncomment below
# pixi shell -e torch-gpu
```

***Option 2. Conda / Pip***
```shell
# a. conda
conda create --name peds_cxr python=3.9.19
conda activate peds_cxr

# b. pip
# pip install -r envs/requirements.txt
```

## 🏃 How to Run

**0. (Optional) Specify package manager for SLURM scripts**
```shell
export USE_PIXI=1     # (1 = use pixi, 0 = uses conda) in SLURM scripts
```

**1. Perform Analysis for Dataset Review**

```shell
# Option 1. Run in current shell
bash slurm/review_datasets.sh

# Option 2. Submit job to SLURM suerver
sbatch slurm/review_datasets.sh
```

**2.0 Train a Model**

> See [slurm/train_model.sh](./slurm/train_model.sh) for more examples

```shell
# To set up CometML for model logging
# Terminal: `pixi add comet-ml` or  `pip install comet-ml`
# Terminal: export COMET_API_KEY="YOUR_API_KEY"

# Option 1. Run in current shell
# NOTE: Config file must be defined under `/config/train_model/`
# NOTE: Replace `dummy` with `vindr_cxr`, `padchest`, `nih_cxr18`, `chexbert`
python -m scripts.train_model -c "exp_cardiomegaly-dummy-mixup-imb_sampler.ini"

# Option 2. Submit job to SLURM server
sbatch slurm/train_model.sh
```


**2.1. Download Pre-Trained Models**
```shell
git clone https://huggingface.co/stan-hua/peds_vs_adult-cxr
# mkdir data/save_data/train_runs         # If directory doesn't exist
mv peds_vs_adult-cxr/* data/save_data/train_runs/
```

**3. Generate Predictions with CXR Models**

> We **strongly recommend** using the bash script. It performs inference cross-dataset. We already provide the [raw predictions](./data/save_data/inference).

> See [slurm/eval_model.sh](./slurm/eval_model.sh) for more details.
```shell
# Option 1. Run in current shell
bash slurm/eval_model.sh

# Option 2. Submit job to SLURM server
# NOTE: Creates figures in `data/save_data/findings/PedsVsAdult_CXR`
sbatch slurm/eval_model.sh
```

**4. Create Figures from Paper**

> We **strongly recommend** using the bash script, since multiple figures are generated. The generated figures are stored in [./data/save_data/findings/PedsVsAdult_CXR](./data/save_data/findings/PedsVsAdult_CXR).

> See [slurm/eval_model.sh](./slurm/eval_model.sh) for more details.
```shell
# Option 1. Run in current shell
bash slurm/create_figures.sh

# Option 2. Submit job to SLURM server
sbatch slurm/create_figures.sh
```


## 👏 Acknowledgements

**Team**:
1. [Stanley Hua](https://stan-hua.github.io/) @ **The Hospital for Sick Children** (Former), **UC Berkeley** & **UCSF** (Now)
2. [Nicholas Heller](https://scholar.google.com/citations?user=gt3amx8AAAAJ&hl=en) @ **Cleveland Clinic**
3. [Ping He]() @ **The Hospital for Sick Children**
4. [Alexander Towbin](https://www.cincinnatichildrens.org/bio/t/alexander-towbin) @ **Cincinnati Children's Hospital**
5. [Irene Chen](https://irenechen.net/) @ **UC Berkeley** & **UCSF**
6. *[Alex Lu](https://www.alexluresearch.com/) @ **Microsoft Research**
7. *[Lauren Erdman](https://scholar.google.com/citations?user=bSKEpp8AAAAJ&hl=en) @ **Cincinnati Children's Hospital** & **University of Cincinnati**

**Co-senior authors*

For any questions, please email the corresponding author: [Lauren Erdman](mailto:lauren.erdman@cchmc.org).


## Citation

If you find our work useful, please consider citing our pre-print on medRxiv!

```bibtex
@article {Hua2025.06.06.25328913,
	author = {Hua, Stanley Bryan Zamora and Heller, Nicholas and He, Ping and Towbin, Alexander J. and Chen, Irene Y. and Lu, Alex X. and Erdman, Lauren},
	title = {Lack of children in public medical imaging data points to growing age bias in biomedical AI},
	elocation-id = {2025.06.06.25328913},
	year = {2025},
	doi = {10.1101/2025.06.06.25328913},
	publisher = {Cold Spring Harbor Laboratory Press},
	URL = {https://www.medrxiv.org/content/early/2025/06/07/2025.06.06.25328913},
	eprint = {https://www.medrxiv.org/content/early/2025/06/07/2025.06.06.25328913.full.pdf},
	journal = {medRxiv}
}

```
