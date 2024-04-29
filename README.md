# Facilitation-of-the-theorem-proving-in-Coq-using-transformer
This is the repository with the code used for bachelor thesis "Facilitation of the theorem proving in Coq using transformer". The main aim of this study is to investigate how good are transformers on the theorem-proving task in high-level formal environment Coq.  
During this study we trained tokenizer which can be found here: https://huggingface.co/Andrusyshyn/gpt2-coq-tokenizer, and we trained 6 transformers which can be found here: https://huggingface.co/Andrusyshyn/gpt2-pretrained-for-coq-pt-custom-train.  
We compared our transformer models of different sizes with current solutions to the ATP (As was mentioned in the thesis, the results for "TacTok", "ASTactic", "CoqHammer" are taken from the TacTok paper https://dl.acm.org/doi/10.1145/3428299, page 22):
| Project              | **TacTok**      | **ASTactic**    | **CoqHammer**   | **n06**      | **n08**      | **n10**      | **n12**      | **Total Theorems** |
|----------------------|-----------------|-----------------|-----------------|--------------|--------------|--------------|--------------|--------------------|
| zchinese             | 5 (11.6%)       | 5 (11.6%)       | 12 (27.9%)      | 0 (0.0%)     | 0 (0.0%)     | 2 (4.7%)     | 1 (2.3%)     | 43                 |
| chinese              | 35 (26.7%)      | 31 (23.7%)      | 56 (42.7%)      | 18 (13.7%)   | 20 (15.3%)   | 24 (18.3%)   | 23 (17.6%)   | 131                |
| hoare-tut            | 5 (27.8%)       | 1 (5.5%)        | 6 (33.3%)       | 2 (11.1%)    | 3 (16.7%)    | 2 (11.1%)    | 2 (11.1%)    | 18                 |
| demos                | 53 (77.9%)      | 50 (73.5%)      | 54 (79.4%)      | 49 (72.1%)   | 52 (76.5%)   | 51 (75.0%)   | 49 (72.1%)   | 68                 |
| coqoban              | 0 (0.0%)        | 0 (0.0%)        | 0 (0.0%)        | 0 (0.0%)     | 0 (0.0%)     | 0 (0.0%)     | 0 (0.0%)     | 2                  |
| fundamental-arithmetics | 15 (10.6%)   | 11 (7.8%)       | 37 (26.1%)      | 8 (5.63%)    | 8 (5.63%)    | 8 (5.63%)    | 11 (7.7%)    | 142                |
| **Total**            | 113 (28.0%)     | 98 (24.3%)      | 165 (40.8%)     | 77 (19.1%)   | 83 (20.5%)   | 87 (21.5%)   | 86 (21.3%)   | 404                |
## Dependencies
In addition to Google Colab, we conducted our work on Ubuntu 22.04 LTS system. Here is the list of used dependencies:
1. Python dependencies: can be installed with ```pip install -r requirements.txt```. We installed the following dependencies:  
   ```pip install torch "transformers[sentencepiece]" datasets matplotlib accelerate tensorboard```
2. We used GNU Make 4.3
3. We were working with Coq 8.9.1 formal environment. To install Coq, we installed opam - OCaml package manager https://opam.ocaml.org/doc/Install.html
   We run the following set of command to install Coq on our system:
   ```
   sudo apt-get install opam
   opam init -y
   eval $(opam env) 
   opam switch create coq9_1 4.07.1+flambda
   eval $(opam env --switch=coq9_1)
   opam install coq.8.9.1
   ```
4. git-lfs. This is optional. Used for working with the Hugging Face repos. https://git-lfs.com/
## External Resources
Our repository contains the following external resources:
- The coq_projects/ folder contains Coq projects collected by the authors of the CoqGym project: https://github.com/princeton-vl/CoqGym. This folder also contains Makefile from the same CoqGym repository, which we modify for our purposes. This folder also contains the Feit-Thompson Odd Order Theorem formalization project (https://github.com/math-comp/odd-order/tree/mathcomp-odd-order.2.0.0) as odd-order-mathcomp-odd-order/ subfolder.
- projs_split.json contains the train/validation/test split and was also taken from the CoqGym project. We modified the train and validation split slightly but left the test split untouched.
- json_data/ folder is cleaned for our purposes CoqGym dataset https://zenodo.org/records/8101883. This is the folder with JSON files that contain pre-extracted theorems and proofs from corresponding Coq source files from coq_projects/ directory.
## Repository Structure
- coq_projects -> contains Coq projects. This is our training, validation, and test data. It also contains a Makefile with which we can build these projects.
- json_data -> contains cleaned JSON files from CoqGym dataset with extracted theorems and proofs.
- scripts/ -> contains scripts used in our work.
- notebooks/ -> contains notebooks used in our work. Note that generating_proofs.py, training_model.py, training_tokenizer.py scripts are just script versions of the notebooks.
- configs/ -> contains config files for generating_proofs.py, training_model.py, training_tokenizer.py scripts (and corresponding notebooks).
- projs_split.json -> contains train/validation/test split.
- datasets/ -> contains datasets that we use for training.
- training_logs -> contains training logs in the form of JSON files.
- tensorboard_runs -> contain training logs in the form of tensorboard.
- images/ -> contains images we used in our thesis.
- theorems/ -> contains theorem datasets. test_theorems.json contains all theorems (except removed structures like Instance). test_theorems_trunc.json is our "trunc" dataset, test_theorems_comp.json is our "comp" dataset.
- generated_proofs -> contains generated proofs.
- tested_proofs -> contains tested proofs and error messages.
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
   - During our work we used this notebook for training tokenizer: https://colab.research.google.com/drive/1iA12XfpytcU-blnLUEWWXgf4RwCkUGQp?usp=sharing. This is the same notebook as notebooks/training_tokenizer.ipynb, but already loaded into Google Colab. Corresponding script training_tokenizer.py has default config ./config/training_tokenizer_config.json:
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
     You can run this notebook with the default provided options in the notebook. If you want to take options from config, specify config_file global variable in the notebook (path to the config      file).
     Also you can use the script:  
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
    - During our work we used this notebook for models training: https://colab.research.google.com/drive/17-YH8_0xF8iVEIyoAHBNYeCnkRL71pXW?usp=sharing. this is the same notebook as          
      notebooks/training_model.ipynb, but already uploaded to the Google Colab. Corresponding script training_model.py has default config ./config/training_model_config.json:
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
      Notebooks and script already contain default parameters from our experiment of training n06 model (to run notebook you have to load dataset_train.zip and dataset_valid.zip into the Colab environment). To use config file in notebook set config_file global variable to the corresponding path.
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
  
4) We create theorem dataset using scripts/create_input_dataset.py script:
   ```
   Usage
   -----
      python ./scripts/create_input_dataset.py [OPTION...]
  
      Options:
          -h, --help                           Print help message.
          -r, --repo <repo_name>               Specify tokenizer repo name.
                                               Default value is "Andrusyshyn/gpt2-coq-tokenizer"
          -v, --revision <revision>            Specify commit hash or branch name of the tokenizer to use.
                                               Default value is "0e1383183b23c6764d83c88b83fa99de2a297199"
          -d, --train_dataset <dataset>        Specify path to the training dataset.
                                               Default value is "./datasets/dataset_train.json"
          -p, --projs_split <projs_split>      Specify path to the split configuration file.
                                               Default value is "./projs_split.json".
          -o, --output <output>                Specify output filepath for the theorem dataset.
                                               Default value is "./theorems/test_theorems.json"
          -c, --coq_projects <coq_projects>    Specify path to the directory with Coq projects.
                                               Default value is "./coq_projects/".
          -j, --json_data <json_date>          Specify cleaned CoqGym dataset directory.
                                               Default value is json_data_root_dir = "./json_data/".

    Examples
    --------
        python ./scripts/create_input_dataset.py
        python ./scripts/create_input_dataset.py -o "./theorems/test_theorems.json"
   ```
   The result is the ./theorems/test_theorems.json theorem dataset which contains every theorem from test set (except for the removed structures as "Instanse"). We than construct our "trunc" and    "comp" datasets (./theorems/test_theorems_trunc.json, ./theorems/test_theorems_comp.json). We do not include code for this but the content of these datasets is described in our thesis.
5) Having the theorem datasets, we now can generate proofs. We do this with the following notebook: https://colab.research.google.com/drive/1iXysomZDQIq-dIKUCbtaF2I7w_T3bmFS?usp=sharing. As in previous cases, this notebook is also provided in our repository as ./notebooks/generating_proofs.ipynb. The corresponding script ./scripts/generating_proofs.py has default config ./configs/generation_config.json:
   ```
   {
       # generation hyperparameters
       "do_generate_proofs"           : true,    # if true, then we do the generation
       "sequence_length"              : 1024,
       "max_new_tokens"               : 256,
       "batch_size"                   : 2,
       "proofs_per_theorem"           : 50,
       "temperature"                  : 1,
       "do_sample"                    : true,
       "top_p"                        : 0.95,
   
       "model_repo_name"              : "Andrusyshyn/gpt2-pretrained-for-coq-pt-custom-train",
       "model_commit_hash"            : "32c2695d0f5f0b6117529f2eaa7f240b95cc42eb",
   
       "theorems_input_file"          : "./theorems/test_theorems_comp.json",           # input therem dataset
       "theorems_output_file"         : "./generated_proofs/n06/experiment_gen.json",   # output generated data
   
       "do_test_loss"                 : false,   # if true, then we do the test loss calculation
       "test_batch_size"              : 4,
       "raw_test_json"                : "./datasets/dataset_test.json",
   
       "use_gpu"                      : true,    # Script-only parameter.
   
       "torch_seed"                   : 77,
   
       # Notebook-only parameters
       "drive_mounted"                : false,
       "drive_mounted_path"           : "/content/gdrive/",
       "test_data_archived"           : true,
       "raw_test_archive"             : "./dataset_test.zip"
   }
   ```
   Usage of the script:
   ```
   Usage
   -----
      python ./scripts/generating_proofs.py [<config_file>]
  
      Argumets:
          <config_file> - path to the config file. Optional.

    Examples
    --------
        python ./scripts/generating_proofs.py
        python ./scripts/generating_proofs.py ./configs/generation_config.json
   ```
   Notebook and script have already defined parameters in the code for generating k=50 proofs per theorem with temperature t=1 for n06 model on "comp" dataset (to run notebook you have to load test_theorems_comp.json in the Colab environment). If you want to parse config in the notebook, change the config_file global variable to the corresponding value.

   As a result of this step we get generated_proofs/ directory by running above notebook for different configurations. Each JSON file in the generated_proofs/ directory contains hyperparameters with which it was created. As we were working in the Colab notebook, we recommend to use it for results reproducibility.
6) The final step is testing the results. Before testing we compile our test project using ```make test-projects``` commands in the ./coq_projects/ directory. This commmand took more than 1 hour to execute. Then we use ./scripts/test_generated_proofs.py for testing generated proofs:
   ```
   Usage
   -----
      python ./scripts/test_generated_proofs.py <input_json_file> <output_json_file> <coq_projs_root_folder> [<all_proofs>]
  
      Argumets:
          <input_json_file>       - path to the dataset with generated proofs.
          <output_json_file>      - output path for the tested proofs.
          <coq_projs_root_folder> - path to the directory containing pre-compiled test Coq projects.
          <all_proofs>            - Optional. Set this argument to "True" to test every generated proof.
                                    Otherwise testing of every theorem will stop on the first correct
                                    proof and proceed with the next theorem.

    Examples
    --------
        python ./scripts/test_generated_proofs.py ./generated_proofs/n06/generated_comp_n06_k05.json ./tested_proofs/n06/tested_proofs_comp_n06_k05.json ./coq_projects/
        python ./scripts/test_generated_proofs.py ./generated_proofs/n06/generated_comp_n06_k50.json ./tested_proofs/n06/tested_proofs_comp_n06_k50.json ./coq_projects/ True
   ```
   Doing that for every file in the generated_proofs/ directory, we get the tested_proofs/ directory.

   Also, we build plots using ./scripts/build_plots.py:
   ```
   python ./scripts/build_plots.py
   ```
   The plots are located in images/ directory.

