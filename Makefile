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
# train:
# 	python src/train.py


### EVALUATION AND TESTING
# eval:
# 	python src/evaluate.py


# test:
# 	python src/test.py


# add new commands here
.PHONY: environment-setup environment-update environment-delete clean
