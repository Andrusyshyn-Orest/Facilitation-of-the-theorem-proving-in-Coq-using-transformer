"""
This is script version of the notebook https://colab.research.google.com/drive/17-YH8_0xF8iVEIyoAHBNYeCnkRL71pXW?usp=sharing

This script trains the model.

Usage
-----
    python ./scripts/training_model.py [<config_file>]

    Argumets:
        <config_file> - path to the config file. Optional.

Examples
--------
    python ./scripts/training_model.py
    python ./scripts/training_model.py ./configs/training_model_config.json
"""
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter

from huggingface_hub import Repository
from datasets import load_dataset, Dataset
from accelerate import Accelerator
from transformers import AutoTokenizer, GPT2LMHeadModel, AutoConfig, get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

import re
import json
import sys
import os
from tqdm import tqdm


# HYPER PARAMETERS (DEFAULT VALUES)
context_length                 = 1024
train_batch_size               = 8
eval_batch_size                = 8
weight_decay                   = 0.1
lr                             = 8e-4
lr_scheduler_func              = get_cosine_schedule_with_warmup
adamw_b1                       = 0.9
adamw_b2                       = 0.95
adamw_e                        = 1e-8
num_warmup_steps               = 30
gradient_accumulation_steps    = 4
gradient_checkpointing         = False
eval_steps                     = 100
num_train_epochs               = 10
mixed_precision                = "fp16"


#GLOBAL VARS (DEFAULT VALUES)
save_json_logs                 = True                             # set to True to save training results into json file (train_logs_path)
train_logs_path                = './training_logs/test_run.json'  # only needed if save_json_logs == True
save_tensorboard_logs          = True                             # set to True to save training results into tensorboard run (tensorboard_run_path)
tensorboard_run_path           = './tensorboard_runs/test_run'    # only needed if save_tensorboard_logs == True

raw_train_json                 = "./datasets/dataset_train.json"  # training dataset
raw_valid_json                 = "./datasets/dataset_valid.json"  # validation dataset

tokenizer_repo_name            = "Andrusyshyn/gpt2-coq-tokenizer"
tokenizer_commit_hash          = "0e1383183b23c6764d83c88b83fa99de2a297199"

init_model                     = "gpt2"                           # base model Hugging Face name
n_layer                        = 6
n_head                         = 6
n_embd                         = 384
model_repo_name                = "Andrusyshyn/gpt2-pretrained-for-coq-pt-custom-train"
model_output_dir               = "./gpt2-pretrained-for-coq-pt-custom-train-local"        # local dir to save the model

push_to_hub                    = False             # set to True to push the model (requires git-lfs to be installed)
run_name                       = "testing_algo"    # branch name (only needed if push_to_hub == True)

partially_trained              = False             # set to True to continue model training from specific commit hash (model_commit_hash)
model_commit_hash              = ""                # only needed if partially_trained == True
previously_completed_steps     = 0                 # only needed if partially_trained == True
previous_global_steps          = 0                 # only needed if partially_trained == True
previous_step                  = 0                 # only needed if partially_trained == True
stopped_epoch                  = 0                 # only needed if partially_trained == True

torch_seed                     = 7
data_seed                      = 23


def parse_config(config_file: str):
    """
    Parses config_file and sets global variables.

    Parameters
    ----------
    config_file : str
        path to config file.
    """
    global context_length, train_batch_size, eval_batch_size, weight_decay, lr,\
           lr_scheduler_func, adamw_b1, adamw_b2, adamw_e, num_warmup_steps, gradient_accumulation_steps,\
           gradient_checkpointing, eval_steps, num_train_epochs, mixed_precision, save_json_logs,\
           train_logs_path, save_tensorboard_logs, tensorboard_run_path, raw_train_json, raw_valid_json,\
           tokenizer_repo_name, tokenizer_commit_hash, init_model, n_layer, n_head, n_embd, model_repo_name,\
           model_repo_name, model_output_dir, push_to_hub, run_name, partially_trained, model_commit_hash,\
           previously_completed_steps, previous_global_steps, previous_step, stopped_epoch, torch_seed, data_seed
    with open(config_file, mode='r') as conf_file:
        conf_data = json.load(conf_file)

    context_length                 = conf_data["context_length"]
    train_batch_size               = conf_data["train_batch_size"]
    eval_batch_size                = conf_data["eval_batch_size"]
    weight_decay                   = conf_data["weight_decay"]
    lr                             = conf_data["lr"]
    if conf_data["lr_scheduler_func"] == "cosine":
        lr_scheduler_func = get_cosine_schedule_with_warmup
    elif conf_data["lr_scheduler_func"] == "linear":
        lr_scheduler_func = get_linear_schedule_with_warmup
    elif conf_data["lr_scheduler_func"] == "const":
        lr_scheduler_func = get_constant_schedule_with_warmup
    else:
        lr_scheduler_func = get_cosine_schedule_with_warmup
    adamw_b1                       = conf_data["adamw_b1"]
    adamw_b2                       = conf_data["adamw_b2"]
    adamw_e                        = conf_data["adamw_e"]
    num_warmup_steps               = conf_data["num_warmup_steps"]
    gradient_accumulation_steps    = conf_data["gradient_accumulation_steps"]
    gradient_checkpointing         = conf_data["gradient_checkpointing"]
    eval_steps                     = conf_data["eval_steps"]
    num_train_epochs               = conf_data["num_train_epochs"]
    mixed_precision                = conf_data["mixed_precision"]


    save_json_logs                 = conf_data["save_json_logs"]
    train_logs_path                = conf_data["train_logs_path"]
    save_tensorboard_logs          = conf_data["save_tensorboard_logs"]
    tensorboard_run_path           = conf_data["tensorboard_run_path"]

    raw_train_json                 = conf_data["raw_train_json"]
    raw_valid_json                 = conf_data["raw_valid_json"]

    tokenizer_repo_name            = conf_data["tokenizer_repo_name"]
    tokenizer_commit_hash          = conf_data["tokenizer_commit_hash"]

    init_model                     = conf_data["init_model"]
    n_layer                        = conf_data["n_layer"]
    n_head                         = conf_data["n_head"]
    n_embd                         = conf_data["n_embd"]
    model_repo_name                = conf_data["model_repo_name"]
    model_output_dir               = conf_data["model_output_dir"]

    push_to_hub                    = conf_data["push_to_hub"]
    run_name                       = conf_data["run_name"]

    partially_trained              = conf_data["partially_trained"]
    model_commit_hash              = conf_data["model_commit_hash"]
    previously_completed_steps     = conf_data["previously_completed_steps"]
    previous_global_steps          = conf_data["previous_global_steps"]
    previous_step                  = conf_data["previous_step"]
    stopped_epoch                  = conf_data["stopped_epoch"]

    torch_seed                     = conf_data["torch_seed"]
    data_seed                      = conf_data["data_seed"]

def loss_function(inputs: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """
    Calculates mean CrossEntropyLoss across samples in the batch.

    Parameters
    ----------
    inputs : torch.Tensor
        tensor of input sequences. Dimensions: batch_size X context_length.
    logits: torch.
        logits outputted by model. Dimensions: batch_size X context_length X vocab_size.

    Returns
    -------
    torch.Tensor
        mean CrossEntropyLoss loss across samples in the batch.
    """
    # inputs [batch_size X cl]
    # logits [btach_size X cl X vocab_size]
    # Our labels start from second sequence token because first one does not have preceding token.
    # We drop last logit because last sequence token does not have subsequent token, so no label to compare
    shifted_labels = inputs[..., 1:].contiguous()
    shifted_logits = logits[..., :-1, :].contiguous()

    loss_func = CrossEntropyLoss(reduction='none')
    # loss [batch_size * (cl-1)] = loss_fct([batch_size * (cl-1) X vocab_size], [batch_size * (cl-1)])
    loss = loss_func(shifted_logits.view(-1, shifted_logits.size(-1)), shifted_labels.view(-1))
    # loss_per_sequence [batch_size]
    loss_per_sequence = loss.view(shifted_logits.size(0), shifted_logits.size(1)).mean(axis=1)
    return loss_per_sequence.mean()

def get_wd_parameters(model: GPT2LMHeadModel, no_decay:list[str]=["bias", r"ln_.{1,2}\.weight"]) -> list[dict]:
    """
    Returns parameters with and without weight decay.

    Parameters
    ----------
    model : GPT2LMHeadModel
        model
    no_decay : list[str], optional
        list of subwords to look for in the model parameters names. This
        parameters will have no decay.
        Default value is ["bias", r"ln_.{1,2}\.weight"].

    Returns
    -------
    list[dict]
        {"params": wd_params,  "weight_decay": weight_decay},
        {"params": nwd_params, "weight_decay": 0.0},
        wd_params and nwd_params are of type Parameter.
    """
    wd_params = []
    nwd_params = []
    for name, params in model.named_parameters():
        if any(re.search(nd_reg, name) for nd_reg in no_decay):
            nwd_params.append(params)
        else:
            wd_params.append(params)
    return [
        {"params": wd_params,  "weight_decay": weight_decay},
        {"params": nwd_params, "weight_decay": 0.0},
    ]

def get_tokenized_dataset(p_raw_dataset: Dataset, p_context_length: int, p_tokenizer: AutoTokenizer) -> Dataset:
    """
    Tokenizes raw dataset p_raw_dataset.

    Parameters
    ----------
    p_raw_dataset : Dataset
        raw dataset ot tokenize
    p_context_length : int
        context length
    p_tokenizer : AutoTokenizer
        tokenizer

    Returns
    -------
    Dataset
        tokenized dataset, each entry is the input sequence of the
        context_length length.
    """
    concatenated_tokenized_samples = []
    for sample in p_raw_dataset:
        tokenized_sample = p_tokenizer(sample["content"], truncation=False)["input_ids"]
        concatenated_tokenized_samples.extend(tokenized_sample + [p_tokenizer.eos_token_id])

    tokenized_dataset_list = []
    for i in range(0, len(concatenated_tokenized_samples), p_context_length):
        input_ids = concatenated_tokenized_samples[i : i + p_context_length]
        if len(input_ids) == p_context_length:
            tokenized_dataset_list.append(torch.tensor(input_ids))

    return Dataset.from_dict({"input_ids": tokenized_dataset_list})

def save_results(filepath: str, split: str, results: list):
    """
    Save training logs to the filepath. Extends current logs in the filepath.

    Parameters
    ----------
    filepath : str
        path to the output JSON file with the following structure:
        {
            "hyperparams": {},
            "train": [
                results1,
                results2,
                ...
            ],
            "valid": [
                results1,
                results2,
                ...
            ]
        }
    split : str
        "train" or "valid"
    results : list
        list of training logs.
    """
    if not save_json_logs:
        return
    if split not in {"train", "valid"}:
        print("ERROR: INVALID SPLIT")
        return
    _run_name = run_name
    if not push_to_hub:
        _run_name = ""
    _tensorboard_run_path = tensorboard_run_path
    if not save_tensorboard_logs:
        _tensorboard_run_path = ""
    _lr_scheduler_type = "cosine"
    if lr_scheduler_func == get_linear_schedule_with_warmup:
        _lr_scheduler_type = "linear"
    elif lr_scheduler_func == get_constant_schedule_with_warmup:
        _lr_scheduler_type = "const"
    hyperparams_dict = {
        "context_length"                 : context_length,
        "train_batch_size"               : train_batch_size,
        "eval_batch_size"                : eval_batch_size,
        "weight_decay"                   : weight_decay,
        "lr"                             : lr,
        "lr_scheduler_type"              : _lr_scheduler_type,
        "adamw_b1"                       : adamw_b1,
        "adamw_b2"                       : adamw_b2,
        "adamw_e"                        : adamw_e,
        "num_warmup_steps"               : num_warmup_steps,
        "gradient_accumulation_steps"    : gradient_accumulation_steps,
        "gradient_checkpointing"         : gradient_checkpointing,
        "eval_steps"                     : eval_steps,
        "num_train_epochs"               : num_train_epochs,
        "mixed_precision"                : mixed_precision,
        "tokenizer_repo_name"            : tokenizer_repo_name,
        "tokenizer_commit_hash"          : tokenizer_commit_hash,
        "init_model"                     : init_model,
        "n_layer"                        : n_layer,
        "n_head"                         : n_head,
        "n_embd"                         : n_embd,
        "model_repo_name"                : model_repo_name,
        "run_name"                       : _run_name,
        "drive_train_res_path"           : train_logs_path,
        "tensorboard_run_path"           : _tensorboard_run_path,
    }
    json_data = {"train": [], "valid": []}
    if os.path.exists(filepath):
        with open(filepath, 'r') as json_file:
            json_data = json.load(json_file)
    json_data["hyperparams"] = hyperparams_dict
    json_data[split].extend(results)
    with open(filepath, 'w') as json_file:
        json.dump(json_data, json_file, indent=4)

def add_to_tensorboard(json_filepath: str, tensorboard_run_path: str):
    """
    Adds training logs from json_filepath to the tensorboard tensorboard_run_path.

    Parameters
    ----------
    json_filepath : str
        path to the JSON file with logs
    tensorboard_run_path : str
        path to the output tensorboard run
    """
    if not save_tensorboard_logs:
        return
    if not os.path.exists(json_filepath):
        print("ERROR: json_filepath DOES NOT EXIST")
        return
    with open(json_filepath, mode="r") as json_file:
        json_data = json.load(json_file)
        one_step_tokens = json_data["hyperparams"]["train_batch_size"] * json_data["hyperparams"]["gradient_accumulation_steps"] * json_data["hyperparams"]["context_length"]
        writer = SummaryWriter(tensorboard_run_path)
        prev_comleted_steps = json_data["train"][0]["completed_steps"]
        prev_lr = json_data["train"][0]["lr"][0]
        train_losses = []
        cs = 0
        for entry in json_data["train"]:
            cs = entry["completed_steps"]
            if cs == prev_comleted_steps:
                train_losses.append(entry["loss/train"])
                continue
            else:
                writer.add_scalar("Loss/Train", sum(train_losses)/len(train_losses), prev_comleted_steps * one_step_tokens)
                writer.add_scalar("Learning Rate", prev_lr, prev_comleted_steps * one_step_tokens)
                train_losses = [entry["loss/train"]]
                prev_comleted_steps = cs
                prev_lr = entry["lr"][0]
        writer.add_scalar("Loss/Train", sum(train_losses)/len(train_losses), cs * one_step_tokens)
        writer.add_scalar("Learning Rate", prev_lr, cs * one_step_tokens)

        for entry in json_data["valid"]:
            cs = entry["completed_steps"]
            writer.add_scalar("Loss/Eval", entry["loss/eval"], cs * one_step_tokens)
            writer.add_scalar("Perplexity/Eval", entry["perplexity"], cs * one_step_tokens)
        writer.close()

def evaluate(p_model: GPT2LMHeadModel, p_valid_dataloader: DataLoader) -> tuple[float, float]:
    """
    Calculates validation loss and perplexity of the p_model on the validation dataset from p_valid_dataloader.

    Parameters
    ----------
    p_model : GPT2LMHeadModel
        model to evaluate.
    p_valid_dataloader : DataLoader
        Dataloader with validation data.

    Returns
    -------
    tuple[float, float]
        validation loss, validation perplexity
    """
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    p_model.eval()
    losses = []
    with torch.no_grad():
        for batch in p_valid_dataloader:
            with torch.no_grad():
                logits = p_model(batch["input_ids"]).logits
                loss = loss_function(batch["input_ids"], logits)
                losses.append(loss.item())
    loss = torch.mean(torch.Tensor(losses))
    try:
        perplexity = torch.exp(loss)
    except OverflowError:
        perplexity = float("inf")

    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return loss.item(), perplexity.item()


if __name__ == "__main__":
    error_msg =\
"""Error. Invalid CLI usage.

Usage
-----
    python ./scripts/training_model.py [<config_file>]

    Argumets:
        <config_file> - path to the config file. Optional.
"""
    if len(sys.argv) > 2:
        print(error_msg)
        sys.exit(1)
    if len(sys.argv) == 2:
        try:
            parse_config(sys.argv[1])
        except Exception as e:
            print(f"Error while parsing config file {sys.argv[1]}: {e}")
            sys.exit(1)

    data_generator = torch.Generator().manual_seed(data_seed)
    torch.manual_seed(torch_seed)

    # CONFIGURING GIT DIRECTORIES
    if push_to_hub:
        repo = Repository(model_output_dir, clone_from=model_repo_name)
        repo.git_checkout(run_name, create_branch_ok=True)

    # LOAD DATASETS
    data_files = {"train": raw_train_json, "validation": raw_valid_json}
    raw_datasets = load_dataset("json", data_files=data_files, field="data")
    print(raw_datasets)

    # LOADING TOKENIZER
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_repo_name, revision=tokenizer_commit_hash)
    print("tokenizer vocab size: ", len(tokenizer))
    # TOKENIZE RAW DATASETS
    train_dataset = get_tokenized_dataset(raw_datasets["train"],      context_length, tokenizer)
    valid_dataset = get_tokenized_dataset(raw_datasets["validation"], context_length, tokenizer)
    train_dataset.set_format("torch")
    valid_dataset.set_format("torch")

    # CREATE DATALOADERS
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, generator=data_generator)
    valid_dataloader = DataLoader(valid_dataset, batch_size=eval_batch_size)
    print(train_dataset)
    print(valid_dataset)
    print("len(train_dataloader): ", len(train_dataloader))
    print("len(valid_dataloader): ", len(valid_dataloader))

    # CONFIGURING MODEL
    model = None
    if partially_trained:
        print("Loading partially trained model")
        model = GPT2LMHeadModel.from_pretrained(model_repo_name, revision=model_commit_hash)
    else:
        print("Training from scratch")
        config = AutoConfig.from_pretrained(
            init_model,
            vocab_size=len(tokenizer),
            n_ctx=context_length,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            n_layer = n_layer,
            n_head = n_head,
            n_embd = n_embd
        )
        model = GPT2LMHeadModel(config)
    print()
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    optimizer = AdamW(get_wd_parameters(model), lr=lr, betas=(adamw_b1, adamw_b2), eps=adamw_e)
    accelerator = Accelerator(mixed_precision=mixed_precision)
    model, optimizer, train_dataloader, valid_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, valid_dataloader
    )

    num_steps_per_epoch = len(train_dataloader)
    print("Num steps per epoch: ", num_steps_per_epoch)
    num_training_completed_steps = (num_train_epochs * num_steps_per_epoch) // gradient_accumulation_steps
    if ((num_train_epochs * num_steps_per_epoch) % gradient_accumulation_steps != 0):
        num_training_completed_steps += 1
    print("Num optimizer steps: ", num_training_completed_steps)

    if lr_scheduler_func == get_constant_schedule_with_warmup:
        lr_scheduler = lr_scheduler_func(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps
        )
    else:
        lr_scheduler = lr_scheduler_func(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_completed_steps
        )


    if partially_trained:
        with torch.no_grad():
            for ind in range(previously_completed_steps):
                if ((ind < num_warmup_steps) or (lr_scheduler.get_lr()[0] > (0.1 * lr))):
                    lr_scheduler.step()

    print()
    print(f"Model size:                                        {model.num_parameters()}")
    print(f"Model size (only trainable params):                {model.num_parameters(only_trainable=True)}")
    print(f"Model size (only trainable non-embeddings params): {model.num_parameters(only_trainable=True, exclude_embeddings=True)}")



    # TRAINING LOOP
    log_buffer = []

    model.train()
    completed_steps = 0
    global_steps = 0
    if partially_trained:
        completed_steps = previously_completed_steps
        global_steps = previous_global_steps
    for epoch in range(num_train_epochs):
        for step, batch in tqdm(
            enumerate(train_dataloader, start=0), total=num_steps_per_epoch
        ):
            if partially_trained and ((epoch<stopped_epoch) or ((epoch==stopped_epoch) and (step <= previous_step))):
                continue

            logits = model(batch["input_ids"]).logits
            loss = loss_function(batch["input_ids"], logits)
    ################################################################################
            log_train = {
                    "loss/train": loss.item(),
                    "completed_steps": completed_steps,
                    "lr": lr_scheduler.get_lr(),
                    "global_steps" : global_steps,
                    "epoch": epoch,
                    "steps": step,
            }
            if ((completed_steps % 10 == 0) and (global_steps % gradient_accumulation_steps == 0)):
                accelerator.print(log_train)
            if save_json_logs:
                log_buffer.append(log_train)
    ################################################################################
            loss = loss / gradient_accumulation_steps
            accelerator.backward(loss)
            global_steps += 1
    ################################################################################
            if global_steps % gradient_accumulation_steps == 0:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if ((completed_steps < num_warmup_steps) or (lr_scheduler.get_lr()[0] > (0.1 * lr))):
                    lr_scheduler.step()
                optimizer.zero_grad()
                completed_steps += 1
            if (global_steps % (eval_steps * gradient_accumulation_steps)) == 0:
                if save_json_logs:
                    save_results(train_logs_path, "train", log_buffer)
                    log_buffer = []
                eval_loss, perplexity = evaluate(model, valid_dataloader)
                log_eval = {
                        "loss/eval": eval_loss,
                        "perplexity": perplexity,
                        "completed_steps": completed_steps,
                        "lr": lr_scheduler.get_lr(),
                        "global_steps" : global_steps,
                        "epoch": epoch,
                        "steps": step,
                        "loss/train": loss.item() * gradient_accumulation_steps,
                }
                accelerator.print(log_eval)
                save_results(train_logs_path, "valid", [log_eval])
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(model_output_dir, save_function=accelerator.save)
                tokenizer.save_pretrained(model_output_dir)
                if accelerator.is_main_process:
                    if push_to_hub:
                        repo.push_to_hub(
                            commit_message=f"Training in progress: completed_steps {completed_steps}; global_steps {global_steps};\
                                            epoch {epoch}; steps {step}; lr {lr_scheduler.get_lr()};\
                                            loss/eval {eval_loss}; perplexity {perplexity}; loss/train {loss.item() * gradient_accumulation_steps}",
                            blocking=False
                        )
                model.train()

    ################################################################################
    ################################################################################
    ################################################################################

    #GRADIENT UPDATE (In case (global_steps % gradient_accumulation_steps != 0)
    last_eval_log_train_loss = 0
    if (global_steps % gradient_accumulation_steps != 0):
        for step, batch in tqdm(
            enumerate(train_dataloader, start=0), total=(gradient_accumulation_steps - (global_steps % gradient_accumulation_steps)) - 1    # -1 here is purely for better visualisation of tqdm progress bar
        ):
            logits = model(batch["input_ids"]).logits
            loss = loss_function(batch["input_ids"], logits)
            last_eval_log_train_loss = loss.item()
            log_train = {
                    "loss/train": loss.item(),
                    "completed_steps": completed_steps,
                    "lr": lr_scheduler.get_lr(),
                    "global_steps" : global_steps,
                    "epoch": epoch,
                    "steps": step + num_steps_per_epoch,
            }
            if save_json_logs:
                log_buffer.append(log_train)
            loss = loss / gradient_accumulation_steps
            accelerator.backward(loss)
            global_steps += 1
            if global_steps % gradient_accumulation_steps == 0:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if ((completed_steps < num_warmup_steps) or (lr_scheduler.get_lr()[0] > (0.1 * lr))):
                    lr_scheduler.step()
                optimizer.zero_grad()
                completed_steps += 1
                break

    ################################################################################
    ################################################################################
    ################################################################################

    # FINAL EVALUATE AND SAVE
    additional_steps = 0
    if (global_steps % gradient_accumulation_steps != 0):
        additional_steps = gradient_accumulation_steps - (global_steps % gradient_accumulation_steps)
    with torch.no_grad():
        last_train_loss = 0
        for batch in train_dataloader:
            logits = model(batch["input_ids"]).logits
            loss = loss_function(batch["input_ids"], logits)
            last_train_loss = loss.item()
            break
    log_train = {
            "loss/train": last_train_loss,
            "completed_steps": completed_steps,
            "lr": lr_scheduler.get_lr(),
            "global_steps" : global_steps,
            "epoch": epoch,
            "steps": num_steps_per_epoch + additional_steps,
    }
    if save_json_logs:
        log_buffer.append(log_train)
        save_results(train_logs_path, "train", log_buffer)
        log_buffer = []
    accelerator.print(log_train)

    eval_loss, perplexity = evaluate(model, valid_dataloader)
    log_eval = {
            "loss/eval": eval_loss,
            "perplexity": perplexity,
            "completed_steps": completed_steps,
            "lr": lr_scheduler.get_lr(),
            "global_steps" : global_steps,
            "epoch": epoch,
            "steps": num_steps_per_epoch + additional_steps - 1,
            "loss/train": last_eval_log_train_loss,
    }
    accelerator.print(log_eval)
    save_results(train_logs_path, "valid", [log_eval])

    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(model_output_dir, save_function=accelerator.save)
    tokenizer.save_pretrained(model_output_dir)
    if accelerator.is_main_process:
        if push_to_hub:
            repo.push_to_hub(
                commit_message=f"Final model: completed_steps {completed_steps}; global_steps {global_steps};\
                                epoch {epoch}; steps {num_steps_per_epoch + additional_steps - 1}; lr {lr_scheduler.get_lr()};\
                                loss/eval {eval_loss}; perplexity {perplexity}; loss/train {last_eval_log_train_loss}",
                blocking=False
            )

    model.train()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    # END OF TRAINING LOOP



    add_to_tensorboard(train_logs_path, tensorboard_run_path)
