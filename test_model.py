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

def find_positions_in_file(input_string, file_content):
    pattern = r'\s+'.join(re.escape(word) for word in input_string.split())
    pattern += r'\s*:'

    positions = []
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

def extract_theorem_statement(theorem_with_proof: str):
    pos = theorem_with_proof.find("\nProof.")
    if pos != -1:
        return theorem_with_proof[:pos]

    print("THEOREM PROOF DOES NOT START WITH 'Proof.'")
    return theorem_with_proof

def truncate_on_Qed(generated_proof: str):
    qed_stop = "Qed."
    defined_stop = "Defined."
    pos = generated_proof.find(qed_stop)
    if pos != -1:
        return generated_proof[:pos+len(qed_stop)]

    pos = generated_proof.find(defined_stop)
    if pos != -1:
        return generated_proof[:pos+len(defined_stop)]

    return generated_proof

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

def process_file(filepath: str, data_root_dir: str,
                 coq_projects_path: str, chars_per_token:float, context_tokens: int):
    global max_len_input, context_length
    theorems = []

    with open(filepath, mode='r') as json_file:
        json_data = json.load(json_file)
        vernac_cmds = json_data["vernac_cmds"]
        for proof in json_data["proofs"]:
            entry = {
                        "filepath": filepath,
                        "context": "",
                        "proof_start_offset": -1,
                        "proof_end_offset": -1,
                        "proof": ""
                    }
            proof_str = vernac_cmds[proof["line_nb"]][0]
            if any(proof_str.startswith(keyword) for keyword in to_remove):
                continue
            pos_dd = proof_str.find(":")
            if pos_dd == -1:
                # print(f": NOT FOUND FOR {coq_projects_filepath}")
                # print(proof_str)
                # return []
                continue
            # proof_name = " ".join(proof_str.split()[:5])
            proof_name = proof_str[:pos_dd].strip()

            coq_projects_filepath = transform_data_path_into_coq_path(
                                data_root_dir, coq_projects_path, filepath
                            )
            with open(coq_projects_filepath, mode='r') as coq_file:
                coq_file_content = coq_file.read()
                poses = find_positions_in_file(proof_name, coq_file_content)
                # pos = coq_file_content.find(proof_name)
                if poses == []:
                    print(f"PROOF NOT FOUND FOR {coq_projects_filepath}")
                    print(proof_name)
                    continue
                elif len(poses) > 1:
                    # print(f"FEW MATCHES FOR {coq_projects_filepath}")
                    # print(proof_name)
                    continue
                else:
                    pos = poses[0][0]
                    entry["proof_start_offset"] = pos
                    # context_tokens = max_len_input - (len(proof_str) // chars_per_token)
                    context_offset = max(pos-int(chars_per_token*context_tokens), 0)
                    if (len(proof_str) > int(max_len_input * chars_per_token) - (pos - context_offset)):
                        print("WARNING: ", filepath, proof_name)
                        context_offset = pos - (int(max_len_input * chars_per_token) - len(proof_str))
                        print("Length of context: ", (pos - context_offset) // chars_per_token)
                        print("Length of proof declaration: ", len(proof_str) // chars_per_token)
                    context = coq_file_content[context_offset : pos]
                    entry["context"] = context

            proof_str += "\nProof."
            for step in proof["steps"]:
                if step["command"][1] == "VernacBullet":
                    continue
                if step["command"][1] == "VernacEndProof":
                    proof_str += "\nQed."
                    continue
                proof_str += f'\n{step["command"][0]}'
            entry["proof"] = proof_str
            if step["command"][1] != "VernacEndProof":
                print(f"ERROR, last command is not VernacEndProof: {filepath}")

            pos_end = coq_file_content.find(step["command"][0], pos)
            entry["proof_end_offset"] = pos_end
            if (pos_end == -1):
                print("CAN NOT FIND POS END")
                print(filepath, proof_name)
                continue

            theorems.append(entry)
    return theorems

def create_dataset(dataset_path: str, coq_projects_path: str,
                    root_folder: str, projects: list[str],
                    chars_per_token: float, context_tokens: int):
    with open(dataset_path, mode='w') as json_file:
        json_dict = {"projects": dict()}
        for project in projects:
            theorems = []
            for filepath in json_file_iterator(project, root_folder):
                new_theorems = process_file(filepath, root_folder, coq_projects_path, chars_per_token, context_tokens)
                # if new_theorems == []:
                #     return
                theorems.extend(new_theorems)
            json_dict["projects"][project] = theorems

        json.dump(json_dict, json_file, indent=4)


if __name__ == "__main__":
    if False:
        coq_tokenizer = AutoTokenizer.from_pretrained("Andrusyshyn/gpt2-pretrained-for-coq-pt-custom-train", revision="91aac830c5ff8417d8bf389eea271a9d3dabab9c")
        print(get_average_chars_per_token("./datasets/dataset_train.json", coq_tokenizer))

    chars_per_token = 3.16
    context_tokens = 512
    test_projs = []
    with open("./projs_split.json", mode='r') as json_file:
        test_projs = json.load(json_file)["projs_test"]

    if True:
        create_dataset("./theorems/test_theorems.json", "./coq_projects",
                       "../datasets/CoqGym/data/data/",
                       test_projs, chars_per_token, context_tokens)

    if False:
        with open("./coq_projects/buchberger/Preduce.v", mode='r') as f:
            fc = f.read()
            print(fc[14314:14314+28])
            print(fc[18318-20:18318+20])
            print(fc[30747:30747+20])

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

    if False:
        commit_hash = "91aac830c5ff8417d8bf389eea271a9d3dabab9c" # cl 1024  {'loss/eval': 1.7990537881851196, 'perplexity': 6.043925762176514}
        coq_model = GPT2LMHeadModel.from_pretrained("Andrusyshyn/gpt2-pretrained-for-coq-pt-custom-train", revision=commit_hash)
        # tokenizer_commit_hash = "2eb6a5643c6cfdf7baa7be11d56424941b67b0fc"
        coq_tokenizer = AutoTokenizer.from_pretrained("Andrusyshyn/gpt2-pretrained-for-coq-pt-custom-train", revision="91aac830c5ff8417d8bf389eea271a9d3dabab9c")
        pipe = pipeline(
            "text-generation", model=coq_model, tokenizer=coq_tokenizer, # device=1
        )

        theorems_file = "../datasets/CoqGym/theorems_dataset/theorems_valid.json"
        project = "./data/data/angles/"
        with open(theorems_file, mode="r") as json_file:
            json_data = json.load(json_file)["data"]
        for entry in json_data:
            if entry["filepath"].startswith(project):
                print(entry["content"])
                generated_text = pipe(entry["content"], num_return_sequences=2, max_length=1024)
                print(generated_text)
                break
