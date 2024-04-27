"""
This is script version of the notebook https://colab.research.google.com/drive/1iA12XfpytcU-blnLUEWWXgf4RwCkUGQp?usp=sharing

This script trains tokenizer.

Usage
-----
    python ./scripts/training_tokenizer.py [<config_file>]

    Argumets:
        <config_file> - path to the config file. Optional.

Examples
--------
    python ./scripts/training_tokenizer.py
    python ./scripts/training_tokenizer.py ./configs/training_tokenizer_config.json
"""
import sys
import json
from datasets import load_dataset
from transformers import AutoTokenizer
from huggingface_hub import Repository


# HYPERPARAMS AND GLOBAL VARS (DEFAULT VALUES)
vocab_size                     = 30000                             # vocabulary size of the tokenizer
base_model                     = "gpt2"                            # base model Hugging Face name
batch_size                     = 20

raw_train_json                 = "./datasets/dataset_train.json"   # path to the train dataset

push_to_hub                    = False                             # set to True if you need to commit changes
tokenizer_repo_name            = "Andrusyshyn/gpt2-coq-tokenizer"  # only needed if push_to_hub == True
tokenizer_output_dir           = "gpt2-coq-tokenizer_local"        # local dir to save the tokenizer
run_name                       = "experimental"                    # branch name (only needed if push_to_hub == True)


def parse_config(config_file: str):
    """
    Parses config_file and sets global variables.

    Parameters
    ----------
    config_file : str
        path to config file.
    """
    global vocab_size, base_model, batch_size, raw_train_json, push_to_hub,\
        tokenizer_repo_name, tokenizer_output_dir, run_name

    with open(config_file, mode='r') as conf_file:
        conf_data = json.load(conf_file)

    vocab_size                     = conf_data["vocab_size"]
    base_model                     = conf_data["base_model"]
    batch_size                     = conf_data["batch_size"]

    raw_train_json                 = conf_data["raw_train_json"]

    push_to_hub                    = conf_data["push_to_hub"]
    tokenizer_repo_name            = conf_data["tokenizer_repo_name"]
    tokenizer_output_dir           = conf_data["tokenizer_output_dir"]
    run_name                       = conf_data["push_to_hub"]

def get_training_corpus():
    """
    Yields batch_size samples from training dataset.
    """
    train_dataset = tokenizer_dataset["train"]
    for ind in range(0, len(train_dataset), batch_size):
        samples = train_dataset[ind : ind + batch_size]
        yield samples["content"]


if __name__ == "__main__":
    error_msg =\
'''Error. Invalid CLI usage.

Usage
-----
    python ./scripts/training_tokenizer.py [<config_file>]

    Argumets:
        <config_file> - path to the config file. Optional.
'''
    if len(sys.argv) > 2:
        print(error_msg)
        sys.exit(1)
    if len(sys.argv) == 2:
        try:
            parse_config(sys.argv[1])
        except Exception as e:
            print(f"Error while parsing config file {sys.argv[1]}: {e}")
            sys.exit(1)

    # CONFIGURING GIT DIRECTORIES
    if push_to_hub:
        repo = Repository(tokenizer_output_dir, clone_from=tokenizer_repo_name)
        repo.git_checkout(run_name, create_branch_ok=True)

    # LOAD DATASET
    tokenizer_dataset = load_dataset("json", data_files=raw_train_json, field="data")
    print(tokenizer_dataset)

    # LOADING BASE TOKENIZER
    base_tokenizer = AutoTokenizer.from_pretrained(base_model)

    # GET TRAINING CORPUS
    training_corpus = get_training_corpus()

    # TRAINING TOKENIZER
    tokenizer = base_tokenizer.train_new_from_iterator(training_corpus, vocab_size)
    print("Tokenizer Vocab Size: ", len(tokenizer))

    # SAVING TOKENIZER
    tokenizer.save_pretrained(tokenizer_output_dir)
    if push_to_hub:
        repo.push_to_hub(
            commit_message=f"experimental commit", blocking=True
        )
