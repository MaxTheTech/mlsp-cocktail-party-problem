# mlsp-cocktail-party-problem

Project in Machine Learning for Signal Processing. Goal: to solve the cocktail party problem.


## Setup
The `Makefile` in the project root contains various commands that simplify environment setup, dataset downloads, and running experiments. These can be run in a terminal using `make <command>` (inspect `Makefile`) from the project ROOT. This way everyone on the team can run the exact same commands with the correct arguments, reducing errors and ensuring consistency.

Any python script can also be run directly from a terminal using `python src/<script name>.py <arguments>` from the project ROOT with the appropriate arguments. This allows more specific arguments than the `Makefile` commands, and so is useful during development. An example would be
```bash
python -m src.data.load_dataset --dataset "acul3/voxceleb2" --out_name "voxceleb_subset --num_samples 50000 --split train
```

Use whichever you're more comfortable with.

### First Time Setup
1. Clone the repository:
```bash
git clone https://github.com/MaxTheTech/mlsp-cocktail-party-problem.git
cd <repo path>
```

2. Create conda environment `mlsp-project`, install requirements and activate it:
```bash
make environment-setup
conda activate mlsp-project
```
Or if environment already exists, you only need to activate it.

### Update requirements
If you only want to update the environment packages specified in `environment.yml`:
```bash
make environment-update
```



## Downloading datasets
All downloaded datasets are saved `data/raw`.

The `.gitignore` file makes sure that all files within `data/` directory are ignored, so that no actual dataset files are pushed to the remote repository (due to large file sizes).

### Raw training datasets
Single-speaker audio datasets, that will be pre-processed into mixed multi-speaker datasets with signal origin labels for training.

#### VoxCeleb2
VoxCeleb2 (audio only) dataset: https://huggingface.co/datasets/acul3/voxceleb2, has 462,850 data entries totalling 119 GB (compressed).

- Download subset of 10k samples for development (around 2.5 GB):
```bash
make download-voxceleb-10k
```

- Download full VoxCeleb2 dataset (WARNING: LARGE):
```bash
make download-voxceleb-full
```


### Raw testing datasets
Insert multi-speaker datasets used for testing.


#### VoxCeleb2
VoxCeleb2 (audio only) dataset: https://huggingface.co/datasets/acul3/voxceleb2, has 462,850 data entries totalling 119 GB (compressed)



## Pre-processing datasets



## Training models



## Evaluating and testing models








