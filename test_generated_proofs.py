import json
import subprocess
import os


def compile_coq_project(project_path: str):
    result = False
    project = os.path.basename(os.path.normpath(project_path))
    start_dir = os.getcwd()
    os.chdir(project_path)
    command = ["make"]
    if project == "coquelicot":
        command = ["./remake"]
    process = subprocess.run(command, capture_output=True, text=True)
    error_msg = ""
    if process.returncode != 0:
        error_msg = process.stderr
    else:
        result = True
        print(f"Compiled successfully.")
    os.chdir(start_dir)
    return result, error_msg

def test(input_json_file: str, output_json_file: str, coq_projs_root_folder: str):
    with open(input_json_file, mode='r') as proofs_file:
        json_proofs = json.load(proofs_file)

    new_json_content = json_proofs
    json_proofs = json_proofs["projects"]

    total_theorems = 0
    theorems_proved = 0
    for project in json_proofs.keys():
        print("Testing project: ", project)
        coq_file_content = ""
        current_coq_filepath = ""
        entry_ind = 0
        for entry in json_proofs[project]:
            total_theorems += 1
            new_json_content["projects"][project][entry_ind]["error_msgs"] = []
            new_json_content["projects"][project][entry_ind]["correct_proof"] = ""
            new_json_content["projects"][project][entry_ind]["correct_proof_ind"] = -1

            if entry["filepath"] != current_coq_filepath:
                # restore original content of a file
                if current_coq_filepath != "":
                    with open(current_coq_filepath, mode='w') as coq_file:
                        coq_file.write(coq_file_content)
                # read original content
                current_coq_filepath = entry["filepath"]
                with open(current_coq_filepath, mode='r') as coq_file:
                    coq_file_content = coq_file.read()

            gen_ind = 0
            for generated_proof in entry["generated_proofs"]:
                # replace original proof with generated proof
                new_coq_file_content = coq_file_content[:entry["proof_start_offset"]] +\
                                    generated_proof +\
                                    coq_file_content[entry["proof_end_offset"]:]
                with open(current_coq_filepath, mode='w') as coq_file:
                    coq_file.write(new_coq_file_content)

                # compile project
                compile_result, err_msg = compile_coq_project(os.path.join(coq_projs_root_folder, project))
                if compile_result:
                    print(generated_proof)
                    print()
                    new_json_content["projects"][project][entry_ind]["correct_proof"] = generated_proof
                    new_json_content["projects"][project][entry_ind]["correct_proof_ind"] = gen_ind
                    theorems_proved += 1
                    gen_ind += 1
                    break
                else:
                    new_json_content["projects"][project][entry_ind]["error_msgs"].append(err_msg)
                gen_ind += 1

            entry_ind += 1

        # restore original content of a file
        with open(current_coq_filepath, mode='w') as coq_file:
            coq_file.write(coq_file_content)

    with open(output_json_file, mode='w') as correct_proofs_file:
        json.dump(new_json_content, correct_proofs_file, indent=4)

    print("total_theorems:  ", total_theorems)
    print("theorems_proved: ", theorems_proved)


if __name__ == "__main__":
    test("./theorems/try_generate_output.json", "./theorems/correct_proofs.json", "./coq_projects")
