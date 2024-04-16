import json
import os
import re
from transformers import AutoTokenizer


max_len_input = 768    # context length is 1024
to_remove = ["Instance", "Global Instance", "Local Instance", "Polymorphic Instance",
             "Global Polymorphic Instance", "Local Polymorphic Instance",
             "Next Obligation",
             "Add Morphism", "Add Parametric Morphism",
             "Fixpoint", "Function"]


def find_positions_in_file(theorem_name, full_theorem, file_content, dd_found):
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

def get_average_chars_per_token(dataset_path: str, p_tokenizer):
    total_chars = 0
    total_tokens = 0
    with open(dataset_path, mode='r') as json_file:
        json_data = json.load(json_file)["data"]
        for entry in json_data:
            total_chars += len(entry["content"])
            total_tokens += len(p_tokenizer(entry["content"], truncation=False)["input_ids"])
    return total_chars / total_tokens

def json_file_iterator(project, root_folder):
    project_path = os.path.join(root_folder, project)
    for dirpath, _, filenames in os.walk(project_path):
        for filename in filenames:
            if filename.endswith(".json"):
                yield os.path.join(dirpath, filename)

def transform_data_path_into_coq_path(data_folder: str, coq_projs_folder: str, filepath: str):
    normalized_root = os.path.normpath(data_folder)
    normalized_filepath = os.path.normpath(filepath)
    trimmed_path = normalized_filepath[len(normalized_root)+1:]
    new_path = os.path.join(coq_projs_folder, trimmed_path)[:-5] + ".v"
    return new_path

def find_positions(proof_str, coq_projects_filepath):
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

def truncate_context_length(entry: dict, proof_definition: str, p_tokenizer, max_input_tokens: int):
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
                 coq_projects_path: str, chars_per_token:float,
                 p_tokenizer):
    global max_len_input
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
                    chars_per_token: float, p_tokenizer):
    with open(dataset_path, mode='w') as json_file:
        json_dict = {"projects": dict()}
        for project in projects:
            theorems = []
            for filepath in json_file_iterator(project, root_folder):
                new_theorems = process_file(filepath, root_folder, coq_projects_path, chars_per_token, p_tokenizer)
                theorems.extend(new_theorems)
            json_dict["projects"][project] = theorems

        json.dump(json_dict, json_file, indent=4)


if __name__ == "__main__":
    tokenizer_repo = "Andrusyshyn/gpt2-coq-tokenizer"
    tokenizer_revision = "0e1383183b23c6764d83c88b83fa99de2a297199"
    train_dataset_path = "./datasets/dataset_train.json"
    split_file = "./projs_split.json"
    output_filepath = "./theorems/test_theorems.json"
    coq_projects_root_dir = "./coq_projects"
    json_data_root_dir = "./json_data/"
    ###############################################################################################
    coq_tokenizer = AutoTokenizer.from_pretrained(tokenizer_repo, revision=tokenizer_revision)
    chars_per_token = get_average_chars_per_token(train_dataset_path, coq_tokenizer)    # 3.14
    print("Tokenizer vocab size: ", len(coq_tokenizer))
    print("Chars per token:      ", get_average_chars_per_token(train_dataset_path, coq_tokenizer))

    test_projs = []
    with open(split_file, mode='r') as json_file:
        test_projs = json.load(json_file)["projs_test"]

    create_dataset(output_filepath, coq_projects_root_dir, json_data_root_dir,
                    test_projs, chars_per_token, coq_tokenizer)
