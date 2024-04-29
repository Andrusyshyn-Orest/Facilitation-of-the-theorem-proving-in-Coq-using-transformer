# Facilitation-of-the-theorem-proving-in-Coq-using-transformer
This is the repository with the code used for bachelor research "Facilitation of the theorem proving in Coq using transformer". The main aim of this study is to investigate how good are transformers on the theorem-proving task in high-level formal environment, Coq.
## External Resources
Our repository contains the following external resources:
 The coq_projects/ folder contains Coq projects collected by the authors of the CoqGym project: https://github.com/princeton-vl/CoqGym. This folder also contains Makefile from the same CoqGym repository, which we modify for our purposes. This folder also contains the Feit-Thompson Odd Order Theorem formalization project (https://github.com/math-comp/odd-order/tree/mathcomp-odd-order.2.0.0) as odd-order-mathcomp-odd-order/ subfolder.
 Projs_split.json contains the train/validation/test split and was also taken from the CoqGym project. We modified the train and validation split slightly but left the test split untouched.
 - json_data/ folder is cleaned for our purposes CoqGym dataset https://zenodo.org/records/8101883. This is the folder with JSON files that contain pre-extracted theorems and proofs from corresponding Coq source files from coq_projects directory.
## Workflow
1) Firstly, we clean our json_folder/ using scripts/clean_data.py. This repository contains an already cleaned folder. Usage of the script:
```
Usage
-----
    python ./scripts/clean_data.py [<root_dir>]

    Argumets:
        <root_dir> - path to the folder to clean. Optional, default value: "./json_data/".

Examples
--------
    python ./scripts/clean_data.py
    python ./scripts/clean_data.py ./json_data/
```
2) We construct our training/validation/testing datasets using scripts/create_datasets.py. Usage of the script:
```
Usage
-----
    python ./scripts/create_datasets.py [OPTION...]

    Options:
        -h, --help                           Print help message.
        -c, --coq_projects <coq_projects>    Specify path to the directory with Coq projects.
                                             Default value is "./coq_projects/".
        -p, --projs_split  <projs_split>     Specify path to the split configuration file.
                                             Default value is "./projs_split.json".
        -d, --datasets_dir <datasets_dir>    Specify output directory. Datasets JSON files
                                             will be created here. Default value is "./datasets/".

Examples
--------
    python ./scripts/create_datasets.py
    python ./scripts/create_datasets.py -c "./coq_projects/" --projs_split "./projs_split.json" -d "./datasets/"
```
3) Having constructed these datasets, we can train our tokenizer and models.
   - During our work we used this notebook for training tokenizer: https://colab.research.google.com/drive/1iA12XfpytcU-blnLUEWWXgf4RwCkUGQp?usp=sharing. This is the same notebook as notebooks/training_tokenizer.ipynb but already loaded into Google Colab. This notebook (or corresponding script) has default config ./config/training_tokenizer_config.json:
     ```
     {
        "vocab_size"                   : 30000,    # vocablurary size of the tokenizer.
        "base_model"                   : "gpt2",   # base model from which we take base tokenizer
        "batch_size"                   : 20,       # batch size during training
    
        "raw_train_json"               : "./datasets/dataset_train.json",    # path to the training dataset
    
        # These options are used to commit trained tokenizer to our remote repository.
        "push_to_hub"                  : false,
        "tokenizer_repo_name"          : "Andrusyshyn/gpt2-coq-tokenizer",
        "tokenizer_output_dir"         : "gpt2-coq-tokenizer_local",        # This is the local directory which will be created and which contains tokenizer.
        "run_name"                     : "experimental",                    # This is the branch name
    
    
        # This is options which are used only by notebook.
        "drive_mounted"                : false,
        "drive_mounted_path"           : "/content/gdrive/",
        "train_data_archived"          : true,                    # indicates whether the data is archived
        "raw_train_archive"            : "./dataset_train.zip",   # if the data is archived, it will be in the current working directory after extraction. 
        # These are options for git configurations. This is only needed if push_to_hub is enabled.
        "user_email"                   : "user_email",
        "user_name"                    : "user_name"
     }
     ```  
     You can run this notebook with default provided options in the notebook. If you want to take options from config, specify config_file global variable in the notebook (path to the config file).
     Also you can run it through script:  
     ```
     Usage
     -----
     python ./scripts/training_tokenizer.py [<config_file>]
 
     Argumets:
         <config_file> - path to the config file. Optional.
 
     Examples
     --------
         python ./scripts/training_tokenizer.py
         python ./scripts/training_tokenizer.py ./configs/training_tokenizer_config.json
     ```
    
