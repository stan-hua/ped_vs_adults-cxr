


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

> All dataset annotations are provided under [data/metadata/open_data.xlsx](https://github.com/stan-hua/ped_vs_adults-cxr/blob/main/data/metadata/public_datasets_metadata.xlsx)


**B. Cardiomegaly Case Study**

> Please refer to [data/README.md](https://github.com/stan-hua/ped_vs_adults-cxr/blob/main/data/README.md) for downloading and preprocessing the chest X-ray datasets used.


---

## 🔧 Installation

**(Manual) Installation:**
```shell
# Get repository
git clone https://github.com/stan-hua/ped_vs_adults-cxr
cd ped_vs_adults-cxr

# (Optional) Create a new virtual environment with conda
# conda create --name peds_cxr python=3.9.19
# conda activate peds_cxr

# Install required dependencies
pip install -r envs/requirements.txt
```

## 🏃 How to Run

**1.0 Train a Model**

> See [slurm/train_model.sh](https://github.com/stan-hua/ped_vs_adults-cxr/blob/main/slurm/train_model.sh) for more examples

```shell
# Option 1. Run in current shell
# NOTE: Config file must be defined under `/config/train_model/`
python -m scripts.train_model -c "exp_cardiomegaly-vindr_cxr-mixup-imb_sampler.ini"

# Option 2. Submit job to SLURM server
sbatch slurm/train_model.sh
```


**1.1. Download Pre-Trained Models**
```shell
git clone https://huggingface.co/stan-hua/peds_vs_adult-cxr
# mkdir data/save_data/train_runs         # If directory doesn't exist
mv peds_vs_adult-cxr/* data/save_data/train_runs/
```

**2. Evaluate Models & Generate Figures**

> We **strongly recommend** using the bash script. It performs inference cross-dataset and generates figures from the paper. Additionally, we provide the [raw predictions](https://github.com/stan-hua/ped_vs_adults-cxr/tree/main/data/save_data/inference) and the [figures](https://github.com/stan-hua/ped_vs_adults-cxr/tree/main/data/save_data/figures/eda) for analysis.

> See [slurm/eval_model.sh](https://github.com/stan-hua/ped_vs_adults-cxr/blob/main/slurm/eval_model.sh) for more details.
```shell
# Option 1. Run in current shell
bash slurm/eval_model.sh

# Option 2. Submit job to SLURM suerver
sbatch slurm/eval_model.sh
```



## 👏 Acknowledgements

**Team**:
1. [Stanley Hua](https://stan-hua.github.io/) @ **The Hospital for Sick Children**
2. [Nicholas Heller](https://scholar.google.com/citations?user=gt3amx8AAAAJ&hl=en) @ **Cleveland Clinic**
3. [Ping He]() @ **The Hospital for Sick Children**
4. [Alexander Towbin](https://www.cincinnatichildrens.org/bio/t/alexander-towbin) @ **Cincinnati Children's Hospital**
5. [Irene Chen](https://irenechen.net/) @ **UC Berkeley** & **UCSF**
6. *[Alex Lu](https://www.alexluresearch.com/) @ **Microsoft Research**
7. *[Lauren Erdman](https://scholar.google.com/citations?user=bSKEpp8AAAAJ&hl=en) @ **Cincinnati Children's Hospital** & **University of Cincinnati**

**Co-senior authors*

For any questions, please email the corresponding author: [Lauren Erdman](mailto:lauren.erdman@cchmc.org).


## Citation

If you find our work useful, please consider citing our paper!

```bibtex
@article{YourName,
  title={Your Title},
  author={Your team},
  journal={Location},
  year={Year}
}
```
