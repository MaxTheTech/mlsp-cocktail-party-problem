# mlsp-cocktail-party-problem

## Environment

Install enviroment with requirements:
1. Create environment: `conda create -n mlsp-project python=3.10`
2. Activate environment `conda activate mlsp-project`
3. Install requirements: `pip install -r requirements.txt`


## Download datasets

Put links to download the raw datasets here.
- voxceleb2 audio only: https://huggingface.co/datasets/acul3/voxceleb2

Don't push the actual dataset files to the remote repo (too large), though this is accounted for in the .gitignore.




## Running Scripts

All scripts should be run from the repository root:
```bash
# Download data
python src/train_load.py

# Train model (example)
python src/train.py
```

