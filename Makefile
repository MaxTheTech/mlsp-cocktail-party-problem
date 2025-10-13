# Makefile


### UTILITY
environment-setup:
	conda env create -f environment.yml
	@echo ""
	@echo "Created environment and installed requirements. Activate with: conda activate mlsp-project"

environment-update:
	conda env update -f environment.yml --prune
	@echo ""
	@echo "Environment updated. Make sure you have activated: conda activate mlsp-project"

environment-delete:
	conda env remove --name mlsp-project
	@echo ""
	@echo "Environment mlsp-project deleted"


### DATASETS
# training datasets
download-voxceleb-10k:
	python -m src.data.load_dataset --dataset "acul3/voxceleb2" --out_name "voxceleb2_10k" --num_samples 10000

download-voxceleb-full:
	python -m src.data.load_dataset --dataset acul3/voxceleb2 --out_name voxceleb2_full --num_samples 0

# testing datasets
# WIP


### TRAINING
# train:
# 	python src/train.py


### EVALUATION AND TESTING
# eval:
# 	python src/evaluate.py

# test:
# 	python src/test.py


# add new commands here
.PHONY: environment-setup environment-update environment-delete download-voxceleb-10k download-voxceleb-full train eval test
