# Facilitation-of-the-theorem-proving-in-Coq-using-transformer
This is the repository with the code used for bachelor research "Facilitation of the theorem proving in Coq using transformer". The main aim of this study is to investigate how good are transformers on the theorem-proving task in high-level formal environment, Coq.
## External Resources
Our repository contains the following external resources:
 - The coq_projects/ folder contains Coq projects collected by the authors of the CoqGym project: https://github.com/princeton-vl/CoqGym. This folder also contains Makefile from the same CoqGym repository, which we modify for our purposes. This folder also contains the Feit-Thompson Odd Order Theorem formalization project (https://github.com/math-comp/odd-order/tree/mathcomp-odd-order.2.0.0) as odd-order-mathcomp-odd-order/ subfolder.
 - projs_split.json contains the train/validation/test split and was also taken from the CoqGym project. We modified the train and validation split slightly but left the test split untouched.
 - json_data/ folder is cleaned for our purposes CoqGym dataset https://zenodo.org/records/8101883. This is the folder with JSON files that contain pre-extracted theorems and proofs from corresponding Coq source files from coq_projects directory.
## Project Structure Overview
- projs_split.json -> contains train/validation/test split of projects in the coq_projects/ directory.
- coq_projects/ -> contains Coq projects. This is our training data corpus. It also contains a Makefile for compiling these projects.
- json_data/ -> contains cleaned JSON files from CoqGym dataset with preextracted theorems and proofs from the corresponding Coq source files.
- scripts/ -> contains scripts that we used in our project.
- notebooks/ -> contains notebooks for training model and tokenizer, generating proofs. Note that these notebooks are the same as the corresponding scripts in scripts directory. scripts/training_tokenizer.py is the script version of notebooks/training_tokenizer.ipynb; scripts/training_model.py is the script version of notebooks/training_model.ipynb; scripts/generating_proofs.py is the script version of notebooks/generating_proofs.ipynb.
- images/ -> contains images that we used in our research.
- datasets/ -> contains train, validation, test datasets for training our models.
- configs/ -> contains configs for the scripts/training_tokenizer.py, scripts/training_model.py, scripts/generating_proofs.py (and corresponding notebooks).
- training_logs/ -> contains training logs.
- tensorboard_runs/ -> contains tensorboard data of our trainings.
- theorems/ -> contains theorem datasets. test_theorems.json contains all the theorems from the test split (without removed structures like Instance). However, as we mentioned in our work, we do not use the whole dataset. test_theorems_comp.json is our "comp" dataset and test_theorems_trunc.json is our "trunc" dataset.
- generated_proofs/ -> contains generated proofs for every model (n02, n04, n06, n08, n10, n12 models).
- tested_proofs/ -> contains the final result - information of which generated proofs were valid and which not and error messages.


