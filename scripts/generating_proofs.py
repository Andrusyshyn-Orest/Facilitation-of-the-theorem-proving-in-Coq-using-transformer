"""
This is script version of the notebook https://colab.research.google.com/drive/1iXysomZDQIq-dIKUCbtaF2I7w_T3bmFS?usp=sharing

This script generates theorem proofs and calculates test loss of the model.

Usage
-----
    python ./scripts/generating_proofs.py [<config_file>]

    Argumets:
        <config_file> - path to the config file. Optional.

Examples
--------
    python ./scripts/generating_proofs.py
    python ./scripts/generating_proofs.py ./configs/generation_config.json
"""
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader
from datasets import load_dataset, Dataset

from transformers import AutoTokenizer, GPT2LMHeadModel, pipeline, TextGenerationPipeline

import json
import sys
import time


# HYPERPARAMS AND GLOBAL VARS (DEFAULT VALUES)
do_generate_proofs             = True     # set to True to generate proofs
sequence_length                = 1024     # input sequence length of the model
max_new_tokens                 = 256      # only needed if do_generate_proofs == True
batch_size                     = 2        # only needed if do_generate_proofs == True
proofs_per_theorem             = 50       # only needed if do_generate_proofs == True
temperature                    = 1        # only needed if do_generate_proofs == True
do_sample                      = True     # only needed if do_generate_proofs == True
top_p                          = 0.95     # only needed if do_generate_proofs == True

model_repo_name                = "Andrusyshyn/gpt2-pretrained-for-coq-pt-custom-train"
model_commit_hash              = "32c2695d0f5f0b6117529f2eaa7f240b95cc42eb"

theorems_input_file            = "./theorems/test_theorems_comp.json"             # theorem dataset
theorems_output_file           = "./generated_proofs/n06/experiment_gen.json"     # output file for generated proofs

do_test_loss                   = False    # set ot True to calculate model loss on the test dataset
test_batch_size                = 4        # only needed if do_test_loss == True
raw_test_json                  = "./datasets/dataset_test.json"    # path to the test dataset

use_gpu                        = True

torch_seed                     = 77


def parse_config(config_file: str):
    """
    Parses config_file and sets global variables.

    Parameters
    ----------
    config_file : str
        path to config file.
    """
    global do_generate_proofs, sequence_length, max_new_tokens, batch_size, proofs_per_theorem,\
           temperature, do_sample, top_p, model_repo_name, model_commit_hash, theorems_input_file,\
           theorems_output_file, do_test_loss, test_batch_size, raw_test_json, use_gpu, torch_seed

    with open(config_file, mode='r') as conf_file:
        conf_data = json.load(conf_file)

    do_generate_proofs             = conf_data["do_generate_proofs"]
    sequence_length                = conf_data["sequence_length"]
    max_new_tokens                 = conf_data["max_new_tokens"]
    batch_size                     = conf_data["batch_size"]
    proofs_per_theorem             = conf_data["proofs_per_theorem"]
    temperature                    = conf_data["temperature"]
    do_sample                      = conf_data["do_sample"]
    top_p                          = conf_data["top_p"]

    model_repo_name                = conf_data["model_repo_name"]
    model_commit_hash              = conf_data["model_commit_hash"]

    theorems_input_file            = conf_data["theorems_input_file"]
    theorems_output_file           = conf_data["theorems_output_file"]

    do_test_loss                   = conf_data["do_test_loss"]
    test_batch_size                = conf_data["test_batch_size"]
    raw_test_json                  = conf_data["raw_test_json"]

    use_gpu                        = conf_data["use_gpu"]

    torch_seed                     = conf_data["torch_seed"]


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
    # logits [batch_size X cl X vocab_size]
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

def test_loss(p_model: GPT2LMHeadModel, p_test_dataloader: DataLoader, p_device: torch.device) -> tuple[float, float]:
    """
    Calculates test loss and perplexity of the p_model on the test dataset from p_test_dataloader.

    Parameters
    ----------
    p_model : GPT2LMHeadModel
        model to test.
    p_test_dataloader : DataLoader
        Dataloader with test data.
    p_device : torch.device
        cpu or cuda.

    Returns
    -------
    tuple[float, float]
        test loss, test perplexity
    """
    if use_gpu and torch.cuda.is_available(): torch.cuda.empty_cache()

    p_model.eval()
    losses = []
    with torch.no_grad():
        for batch in p_test_dataloader:
            with torch.no_grad():
                input_ids = batch["input_ids"].to(p_device)
                logits = p_model(input_ids).logits
                loss = loss_function(input_ids, logits)
                losses.append(loss.item())
    loss = torch.mean(torch.Tensor(losses))
    try:
        perplexity = torch.exp(loss)
    except OverflowError:
        perplexity = float("inf")

    if use_gpu and torch.cuda.is_available(): torch.cuda.empty_cache()
    return loss.item(), perplexity.item()

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

def extract_theorem_statement(theorem_with_proof: str) -> str:
    """
    Extracts theorem statement from theorem_with_proof.

    Parameters
    ----------
    theorem_with_proof : str
        theorem with proof.

    Returns
    -------
        theorem statement.
    """
    pos = theorem_with_proof.find("\nProof.")
    if pos != -1:
        return theorem_with_proof[:pos]

    print("THEOREM PROOF DOES NOT START WITH 'Proof.'")
    return theorem_with_proof

def truncate_on_Qed(generated_proof: str) -> tuple[str, bool]:
    """
    Truncate generated_proof on the first occurance of
    "Qed.", "Defined." or "Save".

    Parameters
    ----------
    generated_proof : str
        generated proof

    Returns
    -------
    tuple[str, bool]
        First element is the trucated proof. Second element is status.
        Status is False if no finalization command were found. In such
        case, first element is just whole generated_proof.
    """
    qed_stop = "Qed."
    defined_stop = "Defined."
    save_stop = "Save"

    pos_qed = generated_proof.find(qed_stop)
    pos_defined = generated_proof.find(defined_stop)
    pos_save = generated_proof.find(save_stop)

    poses_stops = []
    if (pos_qed != -1):
        poses_stops.append(pos_qed)
    if (pos_defined != -1):
        poses_stops.append(pos_defined)
    if (pos_save != -1):
        poses_stops.append(pos_save)
    if (poses_stops == []):
        return (generated_proof, False)

    return (generated_proof[:min(poses_stops)], True)

def generate_proofs(input_file: str, output_file: str, p_pipe: TextGenerationPipeline, num_proofs: int):
    """
    Iterates over each theorem entry in the input_file. Generates num_proofs per theorem.
    Uses generation hyperparameters from global config:
    batch_size, max_new_tokens, do_sample, top_p, temperature, sequence_length.

    Parameters
    ----------
    input_file : str
        path to the file with theorems (theorem dataset).
    output_file : str
        output path of the tested theorems.
    p_pipe : TextGenerationPipeline
        text-generation pipeline
    num_proofs : int
        number of generated proofs per theorem.
    """
    cuda_available = torch.cuda.is_available()
    new_json_data = None
    with open(input_file, mode='r') as json_input:
        new_json_data = json.load(json_input)

    theorems_processed = 0
    proofs_with_end = 0
    for project in new_json_data["projects"].keys():
        for i in range(0, len(new_json_data["projects"][project]), batch_size):
            theorems = new_json_data["projects"][project][i:i+batch_size]
            input_sequences = []
            theorem_declarations = []
            for theorem in theorems:
                theorem_declaration = extract_theorem_statement(theorem["proof"])
                theorem_declarations.append(theorem_declaration)
                input_sequence = theorem["context"] + theorem_declaration
                input_sequences.append(input_sequence)

            generated_texts = p_pipe(input_sequences, num_return_sequences=num_proofs,
                                     max_new_tokens=max_new_tokens,
                                     return_full_text=False,
                                     do_sample=True, top_p=top_p, temperature=temperature)
            if use_gpu and cuda_available:
                torch.cuda.empty_cache()

            ind = 0
            for generated_text in generated_texts:
                generated_proofs = []
                for proof in generated_text:
                    proof_with_no_context = theorem_declarations[ind] + proof['generated_text']
                    truncated_proof, found_end = truncate_on_Qed(proof_with_no_context)
                    if found_end:
                        proofs_with_end += 1
                    generated_proofs.append(truncated_proof + theorems[ind]["end_command"])
                new_json_data["projects"][project][ind+i]["generated_proofs"] = generated_proofs
                theorems_processed += 1
                ind += 1

            if theorems_processed % 10 == 0:
                print(theorems_processed)
                with open(output_file, mode='w') as json_output:
                    json.dump(new_json_data, json_output, indent=4)

    new_json_data["hyperparams"] = {
        "sequence_length": sequence_length,
        "max_new_tokens": max_new_tokens,
        "batch_size": batch_size,
        "proofs_per_theorem": proofs_per_theorem,
        "temperature": temperature,
        "do_sample": do_sample,
        "top_p": top_p,
        "model_repo_name": model_repo_name,
        "model_commit_hash": model_commit_hash,
        "torch_seed": torch_seed
    }
    with open(output_file, mode='w') as json_output:
        json.dump(new_json_data, json_output, indent=4)
    print("Theorems Processed: ", theorems_processed)
    print("Proofs with end:    ", proofs_with_end)


if __name__ == "__main__":
    error_msg =\
"""Error. Invalid CLI usage.

Usage
-----
    python ./scripts/generating_proofs.py [<config_file>]

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

    if use_gpu and torch.cuda.is_available():
        print("Running on GPU!")
    else:
        print("Running on CPU!")

    torch.manual_seed(torch_seed)

    # LOAD TEST DATASET
    if do_test_loss:
        raw_test_dataset = load_dataset("json", data_files=raw_test_json, field="data")
        print(raw_test_dataset)

    # LOAD MODEL AND TOKENIZER
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    coq_tokenizer = AutoTokenizer.from_pretrained(model_repo_name, revision=model_commit_hash)
    coq_model = GPT2LMHeadModel.from_pretrained(model_repo_name, revision=model_commit_hash).to(device)
    print(f"Tokenizer vocab size:                              {len(coq_tokenizer)}")
    print(f"Model size:                                        {coq_model.num_parameters()}")
    print(f"Model size (only trainable params):                {coq_model.num_parameters(only_trainable=True)}")
    print(f"Model size (only trainable non-embeddings params): {coq_model.num_parameters(only_trainable=True, exclude_embeddings=True)}")
    pipe = pipeline("text-generation", model=coq_model, tokenizer=coq_tokenizer, batch_size=batch_size, device=0 if use_gpu and torch.cuda.is_available() else -1)
    pipe.tokenizer.pad_token_id = coq_model.config.eos_token_id
    pipe.tokenizer.padding_side = 'left'

    if do_test_loss:
        # TOKENIZE RAW DATASET
        test_dataset = get_tokenized_dataset(raw_test_dataset["train"], sequence_length, coq_tokenizer)
        test_dataset.set_format("torch")

        # CREATE DATALOADER
        test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size)
        print(test_dataset)
        print("len(test_dataloader): ", len(test_dataloader))

    # EVALUATE TEST LOSS
    if do_test_loss:
        _loss,_perp = test_loss(coq_model, test_dataloader, device)
        print("Test Loss:       ", _loss)
        print("Test Perplexity: ", _perp)

    # PROOF GENERATION
    if do_generate_proofs:
        time_start = time.perf_counter()
        generate_proofs(theorems_input_file, theorems_output_file, pipe, proofs_per_theorem)
        time_end = time.perf_counter()
        print(f"Total time: {time_end - time_start} seconds")
