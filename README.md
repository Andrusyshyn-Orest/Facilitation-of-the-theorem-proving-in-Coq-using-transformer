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
   - During our work we used this notebook for training tokenizer: https://colab.research.google.com/drive/1iA12XfpytcU-blnLUEWWXgf4RwCkUGQp?usp=sharing. This is the same notebook as notebooks/training_tokenizer.ipynb, but already loaded into Google Colab. This notebook (and corresponding script) has default config ./config/training_tokenizer_config.json:
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
    - During our work we used this notebook for models training: https://colab.research.google.com/drive/17-YH8_0xF8iVEIyoAHBNYeCnkRL71pXW?usp=sharing. this is the same notebook as notebooks/training_model.ipynb, but already uploaded to the Google Colab. This notebook (and corresponding script) has default config ./config/training_model_config.json:
      ```
      {
          # These are hyperparameters for training.
          "context_length"                 : 1024,
          "train_batch_size"               : 8,
          "eval_batch_size"                : 8,
          "weight_decay"                   : 0.1,
          "lr"                             : 8e-4,
          "lr_scheduler_func"              : "cosine",
          "adamw_b1"                       : 0.9,
          "adamw_b2"                       : 0.95,
          "adamw_e"                        : 1e-8,
          "num_warmup_steps"               : 30,
          "gradient_accumulation_steps"    : 4,
          "gradient_checkpointing"         : false,
          "eval_steps"                     : 100,
          "num_train_epochs"               : 10,
          "mixed_precision"                : "fp16",
      
      
          "save_json_logs"                 : true,    # if set to true, then we save training logs to the "train_logs_path"
          "train_logs_path"                : "./training_logs/test_run.json",
          "save_tensorboard_logs"          : true,    # if set to true, then we save training logs to the tensorboard run "tensorboard_run_path"
          "tensorboard_run_path"           : "./tensorboard_runs/test_run",
      
          "raw_train_json"                 : "./datasets/dataset_train.json",    # path to the train split
          "raw_valid_json"                 : "./datasets/dataset_valid.json",    # path to the validation split
      
          "tokenizer_repo_name"            : "Andrusyshyn/gpt2-coq-tokenizer",
          "tokenizer_commit_hash"          : "0e1383183b23c6764d83c88b83fa99de2a297199",
      
          "init_model"                     : "gpt2",
          "n_layer"                        : 6,         # These two fiels both mean number of heads.
          "n_head"                         : 6,
          "n_embd"                         : 384,       # This means embeddings size
          "model_repo_name"                : "Andrusyshyn/gpt2-pretrained-for-coq-pt-custom-train",
          "model_output_dir"               : "./gpt2-pretrained-for-coq-pt-custom-train-local",    # local directory for saving the model
      
          "push_to_hub"                    : false,             # We pushed our model to the repository during checkpoints.
          "run_name"                       : "testing_algo",    # branch name
      
          # This options were used to continue training of partly trained models.
          "partially_trained"              : false,
          "model_commit_hash"              : "",
          "previously_completed_steps"     : 0,
          "previous_global_steps"          : 0,
          "previous_step"                  : 0,
          "stopped_epoch"                  : 0,
      
          # Seeds
          "torch_seed"                     : 7,
          "data_seed"                      : 23,
      
      
          # This options are used only by notebooks.
          "have_git_write_access"          : false,
          "user_email"                     : "user_email",
          "user_name"                      : "user_name",
          "drive_mounted"                  : false,
          "drive_mounted_path"             : "/content/gdrive/",
          "data_archived"                  : true,                     # true when we need to extract data
          "raw_train_archive"              : "./dataset_train.zip",    # path to the archived train data
          "raw_valid_archive"              : "./dataset_valid.zip"     # path to the archived valid data
      }
      ```  
      Notebooks and script already contain default parameters from our experiment of training n06 model. To use config file in notebook set config_file global variable to the corresponding path.
      Script usage is the following:
      ```
      Usage
      -----
          python ./scripts/training_model.py [<config_file>]
      
          Argumets:
              <config_file> - path to the config file. Optional.

      Examples
      --------
          python ./scripts/training_model.py
          python ./scripts/training_model.py ./configs/training_model_config.json
      ```
      As we were using Colab notebook, we recommend using Colab notebook for reproducibility of training results.
      As a result of training, we get logs in the training_logs directory. Each log file contains hyperparameters that we used for the specific training.
      Also we get tensorboard_runs/ directory. You can check its contents using ```tensorboard --logdir=./tensorboard_runs/``` command.
