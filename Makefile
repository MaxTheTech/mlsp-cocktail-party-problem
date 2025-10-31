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

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

### DATASETS


### TRAINING
train-bilstm:
	python -m src.train.bilstm_train --root-dir-data data/Libri2Mix --config-data config/libri2mix_16k_2src.yaml --config-model config/bilstm.yaml


### EVALUATION AND TESTING
eval-oracle-irm-dev:
	python -m src.evaluate.eval_oracle_irm --root-dir-data data/Libri2Mix --config-data config/libri2mix_16k_2src.yaml --split dev

eval-oracle-irm-test:
	python -m src.evaluate.eval_oracle_irm --root-dir-data data/Libri2Mix --config-data config/libri2mix_16k_2src.yaml --split test


# add new commands here
.PHONY: environment-setup environment-update environment-delete clean eval-oracle-irm-dev eval-oracle-irm-test train-bilstm train-bilstm-debug train-bilstm-fast-dev-run eval-bilstm-dev eval-bilstm-test
