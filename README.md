# Facilitation-of-the-theorem-proving-in-Coq-using-transformer
This is the repository with the code used for bachelor research "Facilitation of the theorem proving in Coq using transformer". The main aim of this study is to investigate how good are transformers on the theorem-proving task in high-level formal environment, Coq.
## External Resources
Our repository contains the following external resources:
 The coq_projects/ folder contains Coq projects collected by the authors of the CoqGym project: https://github.com/princeton-vl/CoqGym. This folder also contains Makefile from the same CoqGym repository, which we modify for our purposes. This folder also contains the Feit-Thompson Odd Order Theorem formalization project (https://github.com/math-comp/odd-order/tree/mathcomp-odd-order.2.0.0) as odd-order-mathcomp-odd-order/ subfolder.
 Projs_split.json contains the train/validation/test split and was also taken from the CoqGym project. We modified the train and validation split slightly but left the test split untouched.
 - json_data/ folder is cleaned for our purposes CoqGym dataset https://zenodo.org/records/8101883. This is the folder with JSON files that contain pre-extracted theorems and proofs from corresponding Coq source files from coq_projects directory.
