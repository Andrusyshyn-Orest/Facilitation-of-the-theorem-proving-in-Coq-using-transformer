import os
import json


def file_iterator(root_folder: str):
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            yield os.path.join(dirpath, filename)

def remove_redundant_files(root_dir: str):
    count = 0
    for filename in file_iterator(root_dir):
        if os.path.exists(filename):
            if not filename.endswith(".json"):
                os.remove(filename)
                count += 1
        else:
            print(f"Error: no file {filename}")
    print(f"Removed {count} files from {root_dir}")

def remove_redundant_fields(root_dir: str):
    count = 0
    for filename in file_iterator(root_dir):
        if filename.endswith(".json"):
            json_data = dict()
            new_json_data = dict()
            try:
                with open(filename, mode='r') as json_file:
                    json_data = json.load(json_file)
            except:
                print(f"Error while opening file: {filename}")
                continue

            try:
                extra_cmds = json_data["num_extra_cmds"]
                new_json_data["filename"] = json_data["filename"]
                new_json_data["coq_project"] = json_data["coq_project"]
                new_json_data["vernac_cmds"] = json_data["vernac_cmds"][extra_cmds:]
                new_json_data["proofs"] = []
                for proof in json_data["proofs"]:
                    new_proof = dict()
                    new_proof["name"] = proof["name"]
                    new_proof["line_nb"] = proof["line_nb"] - extra_cmds
                    new_proof["steps"] = []
                    for step in proof["steps"]:
                        new_proof["steps"].append({"command": step["command"]})
                    new_json_data["proofs"].append(new_proof)
            except:
                continue

            with open(filename, mode='w') as json_file:
                json.dump(new_json_data, json_file, indent=4)
                count += 1

    print(f"Cleaned {count} files")


if __name__ == "__main__":
    remove_redundant_files("./json_data/")
    remove_redundant_fields("./json_data/")
