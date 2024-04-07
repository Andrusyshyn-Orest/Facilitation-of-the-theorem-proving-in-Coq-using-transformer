import json
import os
import torch
import re
# from transformers import AutoTokenizer, GPT2LMHeadModel, pipeline


context_length = 1024
max_len_input = 768
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
    if project == "disel":
        pass
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

dd_removed = 0
poses_1_removed = 0
poses_2_removed = 0
pos_end_removed = 0
count_saved = 0
dp_proofs = 0
def find_positions(proof_str, coq_projects_filepath):
    global dd_removed
    positions = []

    coq_file_content = ""
    with open(coq_projects_filepath, mode='r') as coq_file:
        coq_file_content = coq_file.read()

    proof_name = proof_str
    pos_dd = proof_str.find(":")
    if pos_dd == -1:
        positions = find_positions_in_file(proof_name, proof_str, coq_file_content, False)
        dd_removed += 1
    else:
        proof_name = proof_str[:pos_dd]
        positions = find_positions_in_file(proof_name, proof_str, coq_file_content, True)

    return positions, coq_file_content

def process_file(filepath: str, data_root_dir: str,
                 coq_projects_path: str, chars_per_token:float):
    global max_len_input, context_length, poses_1_removed, poses_2_removed, pos_end_removed, count_saved, dp_proofs
    coq_projects_filepath = transform_data_path_into_coq_path(
                    data_root_dir, coq_projects_path, filepath
                )
    theorems = []
    duplicate_proofs = set()
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
            if proof_str in duplicate_proofs:
                dp_proofs += 1
                continue

            poses, coq_file_content = find_positions(proof_str, coq_projects_filepath)
            if poses == []:
                print(f"PROOF NOT FOUND FOR {coq_projects_filepath}")
                print(proof_str)
                poses_1_removed += 1
                continue
            # elif len(poses) > 1:
            #     poses_2_removed += 1
            #     if True:
            #         print(coq_projects_filepath)
            #         proof_str += "\nProof."
            #         for step in proof["steps"]:
            #             if step["command"][1] == "VernacBullet":
            #                 continue
            #             proof_str += f'\n{step["command"][0]}'
            #         print(proof_str)
            #         print("####################################################")
            #     continue
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

            proof_str += "\nProof."
            for step in proof["steps"]:
                if step["command"][1] == "VernacBullet":
                    continue
                proof_str += f'\n{step["command"][0]}'
            if step["command"][1] != "VernacEndProof":
                print(f"ERROR, last command is not VernacEndProof: {filepath}")
            entry["proof"] = proof_str
            entry["end_command"] = step["command"][0]

            pos_end = coq_file_content.find(step["command"][0], pos)
            entry["proof_end_offset"] = pos_end + len(step["command"][0])
            if (pos_end == -1):
                print("CAN NOT FIND POS END")
                print(coq_projects_filepath, proof_str)
                pos_end_removed += 1
                continue

            theorems.append(entry)
    return theorems

def create_dataset(dataset_path: str, coq_projects_path: str,
                    root_folder: str, projects: list[str],
                    chars_per_token: float):
    with open(dataset_path, mode='w') as json_file:
        json_dict = {"projects": dict()}
        for project in projects:
            theorems = []
            for filepath in json_file_iterator(project, root_folder):
                new_theorems = process_file(filepath, root_folder, coq_projects_path, chars_per_token)
                theorems.extend(new_theorems)
            json_dict["projects"][project] = theorems

        json.dump(json_dict, json_file, indent=4)


if __name__ == "__main__":
    if False:
        coq_tokenizer = AutoTokenizer.from_pretrained("Andrusyshyn/gpt2-pretrained-for-coq-pt-custom-train", revision="91aac830c5ff8417d8bf389eea271a9d3dabab9c")
        print(get_average_chars_per_token("./datasets/dataset_train.json", coq_tokenizer))

    chars_per_token = 3.16
    test_projs = []
    with open("./projs_split.json", mode='r') as json_file:
        test_projs = json.load(json_file)["projs_test"]

    if True:
        create_dataset("./theorems/test_theorems_exp.json", "./coq_projects", "./json_data/",
                       test_projs, chars_per_token)
        print(dp_proofs, dd_removed, poses_1_removed, poses_2_removed, pos_end_removed)
        print(count_saved)

    if False:
        with open("./theorems/test_theorems.json", mode='r') as json_file:
            json_data = json.load(json_file)["projects"]
            max_len = 0
            max_theorem = ""
            max_filepath = ""
            for project in json_data.keys():
                for entry in json_data[project]:
                    length = len(entry["context"]) + len(extract_theorem_statement(entry["proof"]))
                    if length > max_len:
                        max_len = length
                        max_theorem = extract_theorem_statement(entry["proof"])
                        max_filepath = entry["filepath"]
            print(max_len // chars_per_token)
            print(max_filepath)
            print(max_theorem)
