"""
This module creates theorem dataset from test split.

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
"""
import json
import os
import re
import sys
from getopt import getopt
from transformers import AutoTokenizer


max_len_input = 768    # context length is 1024
to_remove = ["Instance", "Global Instance", "Local Instance", "Polymorphic Instance",
             "Global Polymorphic Instance", "Local Polymorphic Instance",
             "Next Obligation",
             "Add Morphism", "Add Parametric Morphism",
             "Fixpoint", "Function"]


def find_positions_in_file(theorem_name: str, full_theorem: str,
                           file_content: str, dd_found: bool) -> list[tuple[int, int]]:
    """
    Finds offset of the theorem start.

    Parameters
    ----------
    theorem_name : str
        theorem name if dd_found==True, or theorem statement if dd_found=False.
    full_theorem : str
        theorem statement.
    file_content : str
        content of the Coq source file.
    dd_found : bool
        True if full_theorem contains ":", False otherwise.

    Returns
    -------
    list[tuple[int, int]]
        Can return multiple tuples if theorem is duplicated in the Coq source file.
        First element of the tuple is offset of the theorem start. Second is not important.
    """
    positions = []
    if dd_found:
        pattern = r'\s+'.join(re.escape(word) for word in theorem_name.split())
        pattern += r'\s*:'
        for match in re.finditer(pattern, file_content):
            positions.append((match.start(), match.end()))

        if len(positions) > 1:
            positions = []
            pattern = r'\s+'.join(re.escape(word) for word in full_theorem.split())
            for match in re.finditer(pattern, file_content):
                positions.append((match.start(), match.end()))
    else:
        pattern = r'\s+'.join(re.escape(word) for word in full_theorem.split())
        for match in re.finditer(pattern, file_content):
            positions.append((match.start(), match.end()))

    return positions

def get_average_chars_per_token(dataset_path: str, p_tokenizer: AutoTokenizer) -> float:
    """
    Calculates average number of character per token in the dataset.

    Parameters
    ----------
    dataset_path : str
        Path to the JSON dataset.
    p_tokenizer : AutoTokenizer
        tokenizer.

    Returns
    -------
    float
        an average number of character per token in the dataset.
    """
    total_chars = 0
    total_tokens = 0
    with open(dataset_path, mode='r') as json_file:
        json_data = json.load(json_file)["data"]
        for entry in json_data:
            total_chars += len(entry["content"])
            total_tokens += len(p_tokenizer(entry["content"], truncation=False)["input_ids"])
    return total_chars / total_tokens

def json_file_iterator(project: str, root_folder: str):
    '''
    Recursively iterates over all JSON files in folder root_folder/project and
    yields filepathes.

    Parameters
    ----------
    root_folder : str
        path to the folder to iterate
    project : str
        path to the project to iterate

    Yields
    ------
    str
        filepath inside the root_folder/project directory.
    '''
    project_path = os.path.join(root_folder, project)
    for dirpath, _, filenames in os.walk(project_path):
        for filename in filenames:
            if filename.endswith(".json"):
                yield os.path.join(dirpath, filename)

def transform_data_path_into_coq_path(data_folder: str, coq_projs_folder: str, filepath: str) -> str:
    """
    Transforms json filepath from data_folder directory into the Coq source filepath in the
    coq_projs_folder directory. For example, transforms "./json_data/demos/Demo.json" into the
    "./coq_projects/demos/Demo.v"

    Parameters
    ----------
    data_folder : str
        path to the directory containing cleaned CoqGym json files.
    coq_projs_folder : str
        path to the directory containing Coq projects.
    filepath : str
        filepath to transform

    Returns
    -------
    str
        Transformed filepath.
    """
    normalized_root = os.path.normpath(data_folder)
    normalized_filepath = os.path.normpath(filepath)
    trimmed_path = normalized_filepath[len(normalized_root)+1:]
    new_path = os.path.join(coq_projs_folder, trimmed_path)[:-5] + ".v"
    return new_path

def find_positions(proof_str: str, coq_projects_filepath: str) -> tuple[list[tuple[int, int]], str]:
    """
    Finds start offset of the proof_str theorem statement in the
    coq_projects_filepath. Also, returns the file content of the coq_projects_filepath

    Parameters
    ----------
    proof_str : str
        theorem statement.
    coq_projects_filepath : str
        filepath of the Coq source file

    Returns
    list[tuple[int, int]]
        in case of theorem duplication, can return a list of tuples.
        First element of the tuple is the start offset of the theorem
        statement. Second element of the tuple is not important.
    str
        file content of the coq_projects_filepath
    """
    positions = []

    coq_file_content = ""
    with open(coq_projects_filepath, mode='r') as coq_file:
        coq_file_content = coq_file.read()

    proof_name = proof_str
    pos_dd = proof_str.find(":")
    if pos_dd == -1:
        positions = find_positions_in_file(proof_name, proof_str, coq_file_content, False)
    else:
        proof_name = proof_str[:pos_dd]
        positions = find_positions_in_file(proof_name, proof_str, coq_file_content, True)

    return positions, coq_file_content

def truncate_context_length(entry: dict, proof_definition: str,
                            p_tokenizer: AutoTokenizer, max_input_tokens: int)-> dict:
    """
    Truncates context in the entry so it fits into the max_input_tokens together with
    proof_definition.

    Parameters
    ----------
    entry : dict
        theorem entry. Structure of the entry:
        {
            "filepath": "",
            "context": "",
            "context_tokens": 0,
            "proof_start_offset": -1,
            "proof_end_offset": -1,
            "proof": "",
            "end_command": ""
        }
    proof_definition : str
        theorem statement.
    p_tokenizer : AutoTokenizer
        tokenizer.
    max_input_tokens : int
        maximum length of context + theorem statement in tokens.

    Returns
    -------
    dict
        updated entry. "context" and "context_tokens" fields are updated.
    """
    proof_definition_len = len(p_tokenizer(proof_definition, truncation=False)["input_ids"])
    overflow = 11
    if proof_definition_len >= max_input_tokens-overflow:
        print("TOO FEW MAX INPUT TOKENS")
        return entry
    context_max_len = max_input_tokens - proof_definition_len-overflow
    tokenized_context = p_tokenizer(entry["context"], truncation=False)["input_ids"]
    tokenized_context_trunc = tokenized_context[-context_max_len: ]
    entry["context"] = p_tokenizer.decode(tokenized_context_trunc)
    entry["context_tokens"] = len(tokenized_context_trunc)
    return entry

def process_file(filepath: str, data_root_dir: str,
                 coq_projects_path: str, chars_per_token: float,
                 p_tokenizer: AutoTokenizer) -> list[dict]:
    """
    Extracts theorems and their proofs from the JSON file of the CoqGym dataset.
    Skips theorems that start with any of the keywords from to_remove global list.
    Saves file offsets of the theorem start and proof end within corresponding Coq source file.
    Add context preceding the theorem in the corresponding Coq source file.
    Maximum length of context + theorem statement is max_len_input (global var) tokens.
    Saves finalization command of the proof.

    Parameters
    ----------
    filepath : str
        path to the JSON file of the CoqGym dataset.
    data_root_dir : str
        path to the folder with JSON files from CoqGym dataset.
    coq_projects_path : str
        path to Coq projects directory.
    chars_per_token : float
        average number of characters per token.
    p_tokenizer : AutoTokenizer
        tokenizer.

    Returns
    -------
    list[dict]
        list of theorem entries. Structure of the entry:
        {
            "filepath": "",
            "context": "",
            "context_tokens": 0,
            "proof_start_offset": -1,
            "proof_end_offset": -1,
            "proof": "",
            "end_command": ""
        }
    """
    global max_len_input, to_remove
    coq_projects_filepath = transform_data_path_into_coq_path(
                    data_root_dir, coq_projects_path, filepath
                )
    theorems = []
    with open(filepath, mode='r') as json_file:
        json_data = json.load(json_file)
        vernac_cmds = json_data["vernac_cmds"]

        for proof in json_data["proofs"]:
            entry = {
                        "filepath": coq_projects_filepath,
                        "context": "",
                        "context_tokens": 0,
                        "proof_start_offset": -1,
                        "proof_end_offset": -1,
                        "proof": "",
                        "end_command": ""
                    }
            proof_str = vernac_cmds[proof["line_nb"]][0]
            if any(proof_str.startswith(keyword) for keyword in to_remove):
                continue

            poses, coq_file_content = find_positions(proof_str, coq_projects_filepath)
            if poses == []:
                continue
            else:
                pos = poses[0][0]
                entry["proof_start_offset"] = pos
                if len(proof_str) // chars_per_token > max_len_input:
                    continue
                max_context_size = max_len_input - (len(proof_str) // chars_per_token)
                context_offset = max(pos-int(chars_per_token*max_context_size), 0)
                context = coq_file_content[context_offset : pos]
                entry["context"] = context
                entry["context_tokens"] = int(len(context) // chars_per_token)
                entry = truncate_context_length(entry, proof_str, p_tokenizer, max_len_input)

            proof_str += "\nProof."
            for step in proof["steps"]:
                if step["command"][1] == "VernacBullet":
                    continue
                proof_str += f'\n{step["command"][0]}'
            if step["command"][1] != "VernacEndProof":
                print(f"\nERROR, last command is not VernacEndProof: {filepath}")
            entry["proof"] = proof_str
            entry["end_command"] = step["command"][0]

            pos_end = coq_file_content.find(step["command"][0], pos)
            entry["proof_end_offset"] = pos_end + len(step["command"][0])
            if (pos_end == -1):
                print("\nCAN NOT FIND POS END, SKIPPING PROOF:")
                print(coq_projects_filepath, proof_str)
                continue

            theorems.append(entry)
    return theorems

def create_dataset(dataset_path: str, coq_projects_path: str,
                    root_folder: str, projects: list[str],
                    chars_per_token: float, p_tokenizer: AutoTokenizer):
    """
    Creates theorem dataset of the following structure:
    {
        "projects": {
            "proj_name": [
                {
                    "filepath": "",
                    "context": "",
                    "context_tokens": 0,
                    "proof_start_offset": -1,
                    "proof_end_offset": -1,
                    "proof": "",
                    "end_command": ""
                }, ...
            ]
        }
    }

    Parameters
    ----------
    dataset_path : str
        output filepath of the dataset to create (or to rewrite).
    coq_projects_path: str
        path to Coq projects directory.
    root_folder : str
        path to the folder with JSON files from CoqGym dataset.
    projects : str
        projects to include in the dataset.
    chars_per_token : float
        average number of characters per token.
    p_tokenizer : AutoTokenizer
        tokenizer.
    """
    with open(dataset_path, mode='w') as json_file:
        json_dict = {"projects": dict()}
        for project in projects:
            theorems = []
            for filepath in json_file_iterator(project, root_folder):
                new_theorems = process_file(filepath, root_folder, coq_projects_path, chars_per_token, p_tokenizer)
                theorems.extend(new_theorems)
            json_dict["projects"][project] = theorems

        json.dump(json_dict, json_file, indent=4)

def main():
    """
    Main function that parse CLI args and run script.
    """
    error_message = "Error: Invalid CLI usage.\n"
    usage_message =\
'''
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
'''

    parse_error = False
    provide_help = False
    tokenizer_repo = "Andrusyshyn/gpt2-coq-tokenizer"
    tokenizer_revision = "0e1383183b23c6764d83c88b83fa99de2a297199"
    train_dataset_path = "./datasets/dataset_train.json"
    split_file = "./projs_split.json"
    output_filepath = "./theorems/test_theorems.json"
    coq_projects_root_dir = "./coq_projects/"
    json_data_root_dir = "./json_data/"
    try:
        opts, args = getopt(sys.argv[1:],'hr:v:d:p:o:c:j:',['help', 'repo=', "revision=",
                                                            "train_dataset=", "projs_split=",
                                                            "output=", "coq_projects=",
                                                            "json_data="])
        if len(args) != 0:
            parse_error = True

        for option, argument in opts:
            if parse_error: break

            if option == '-h' or option == '--help':
                provide_help = True
            elif option == '-r' or option == '--repo':
                tokenizer_repo = argument
            elif option == '-v' or option == '--revision':
                tokenizer_revision = argument
            elif option == '-d' or option == '--train_dataset':
                train_dataset_path = argument
            elif option == '-p' or option == '--projs_split':
                split_file = argument
            elif option == '-o' or option == '--output':
                output_filepath = argument
            elif option == '-c' or option == '--coq_projects':
                coq_projects_root_dir = argument
            elif option == '-j' or option == '--json_data':
                json_data_root_dir = argument
            else:
                parse_error = True

    except:
        parse_error = True

    if parse_error:
        print(error_message + usage_message)
    elif provide_help:
        print(usage_message)
    else:
        try:
            coq_tokenizer = AutoTokenizer.from_pretrained(tokenizer_repo, revision=tokenizer_revision)
            chars_per_token = get_average_chars_per_token(train_dataset_path, coq_tokenizer)    # 3.14
            print("Tokenizer vocab size: ", len(coq_tokenizer))
            print("Chars per token:      ", round(chars_per_token, 2))

            test_projs = []
            with open(split_file, mode='r') as json_file:
                test_projs = json.load(json_file)["projs_test"]

            create_dataset(output_filepath, coq_projects_root_dir, json_data_root_dir,
                test_projs, chars_per_token, coq_tokenizer)
        except Exception as e:
            print(f"Error, an exception occured:\n{e}")


if __name__ == "__main__":
    main()
