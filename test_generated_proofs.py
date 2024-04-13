import json
import subprocess
import os
import time
import sys
import re


def compile_coq_project(project_path: str, timeout_duration=5):
    result = False
    project = os.path.basename(os.path.normpath(project_path))
    start_dir = os.getcwd()
    os.chdir(project_path)
    command = ["make"]
    if project == "coquelicot":
        command = ["./remake"]

    try:
        process = subprocess.run(command, capture_output=True, text=True, timeout=timeout_duration)
        error_msg = ""
        if process.returncode != 0:
            result = False
            error_msg = process.stderr
        else:
            result = True
            print(f"Compiled successfully.")
    except subprocess.TimeoutExpired:
        result = False
        error_msg = f"Compilation timed out after {timeout_duration} seconds."
        print(error_msg)

    os.chdir(start_dir)
    return result, error_msg

def extract_subpath(full_path, root_folder):
    full_path = os.path.normpath(full_path)
    root_folder = os.path.normpath(root_folder)

    if full_path.startswith(root_folder):
        subpath = full_path[len(root_folder)+1:]
        return subpath
    else:
        return None

def remove_v_files(make_string: str):
    pattern = re.compile(r'^.*\.v *\n+', re.MULTILINE)
    match = pattern.search(make_string)
    insert_pos = -1
    if match:
        insert_pos = match.start()
    new_content = pattern.sub('', make_string)
    return new_content, insert_pos

def get_new_make_content(make_string: str, start_pos: int, v_filepath):
    if start_pos == -1:
        new_content = make_string + "\n" + v_filepath + "\n"
    else:
        new_content = make_string[:start_pos] + v_filepath + "\n"
        if start_pos < len(make_string):
            new_content += make_string[start_pos:]
    return new_content

def test(input_json_file: str, output_json_file: str, coq_projs_root_folder: str, all_proofs: bool):
    _CoqProject_projects = set(["disel", "huffman", "PolTac", "coq-procrastination", "coq-library-undecidability", "coqrel"])
    Other_projects = set(["UnifySL", "coquelicot", "verdi-raft", "verdi"])

    time_total_start = time.perf_counter()
    with open(input_json_file, mode='r') as proofs_file:
        json_proofs = json.load(proofs_file)

    new_json_content = json_proofs
    json_proofs = json_proofs["projects"]

    total_theorems = 0
    theorems_proved = 0
    for project in json_proofs.keys():
        print("Testing project: ", project)
        print()
        project_time_start = time.perf_counter()

        ###########################################################################
        if project not in Other_projects:
            if project in _CoqProject_projects:
                make_path = os.path.join(coq_projs_root_folder, project, "_CoqProject")
            else:
                make_path = os.path.join(coq_projs_root_folder, project, "Make")

            with open(make_path, mode='r') as make_file:
                make_original_content = make_file.read()
            make_truncated_content, start_pos = remove_v_files(make_original_content)
        ###########################################################################

        coq_file_content = ""
        current_coq_filepath = ""
        entry_ind = 0
        for entry in json_proofs[project]:
            total_theorems += 1

            if entry["filepath"] != current_coq_filepath:
                # restore original content of a file ############################################################
                if current_coq_filepath != "":
                    with open(current_coq_filepath, mode='w') as coq_file:
                        coq_file.write(coq_file_content)
                    if project not in Other_projects:
                        compile_coq_project(os.path.join(coq_projs_root_folder, project))
                # read original content #########################################################################
                current_coq_filepath = entry["filepath"]
                with open(current_coq_filepath, mode='r') as coq_file:
                    coq_file_content = coq_file.read()
                if project not in Other_projects:
                    coq_file_relative_path = extract_subpath(current_coq_filepath, os.path.join(coq_projs_root_folder, project))
                    if coq_file_relative_path is None:
                        print("ERROR IN EXTRACTING SUBPATH")
                        continue
                    new_make_content = get_new_make_content(make_truncated_content, start_pos, coq_file_relative_path)
                    with open(make_path, mode='w') as make_file1:
                        make_file1.write(new_make_content)
                #################################################################################################

            gen_ind = 0
            found_proof = False
            generated_proofs_entries = []
            print(f"testing file {current_coq_filepath}, theorem index: {entry_ind}")
            for generated_proof in entry["generated_proofs"]:
                print(f"Gen ind: {gen_ind}")
                gen_proof_entry = {"proof": generated_proof, "correct": False, "error_msg": ""}

                # replace original proof with generated proof
                new_coq_file_content = coq_file_content[:entry["proof_start_offset"]] +\
                                    generated_proof +\
                                    coq_file_content[entry["proof_end_offset"]:]
                with open(current_coq_filepath, mode='w') as coq_file:
                    coq_file.write(new_coq_file_content)

                # compile project
                compile_result, err_msg = compile_coq_project(os.path.join(coq_projs_root_folder, project))
                gen_proof_entry["correct"] = compile_result
                gen_proof_entry["error_msg"] = err_msg
                generated_proofs_entries.append(gen_proof_entry)
                if compile_result:
                    print(generated_proof)
                    print("##############################################\n")
                    found_proof = True
                    if not all_proofs:
                        gen_ind += 1
                        break
                gen_ind += 1

            new_json_content["projects"][project][entry_ind]["generated_proofs"] = generated_proofs_entries
            if found_proof:
                theorems_proved += 1

            # checkpoint file save
            if total_theorems % 30 == 0:
                with open(output_json_file, mode='w') as correct_proofs_file_checkpoint:
                    json.dump(new_json_content, correct_proofs_file_checkpoint, indent=4)

            entry_ind += 1

        # restore original content of a file #####################################################
        with open(current_coq_filepath, mode='w') as coq_file:
            coq_file.write(coq_file_content)
        compile_coq_project(os.path.join(coq_projs_root_folder, project))
        if project not in Other_projects:
            with open(make_path, mode='w') as make_file2:
                make_file2.write(make_original_content)
        ##########################################################################################

        project_time_end = time.perf_counter()
        print(f"Tested project {project}. Time: {project_time_end-project_time_start} seconds")
        print("##############################################\n")

    with open(output_json_file, mode='w') as correct_proofs_file:
        json.dump(new_json_content, correct_proofs_file, indent=4)

    time_total_end = time.perf_counter()

    print("\n##############################################")
    print("##############################################")
    print("##############################################")
    print("Total theorems:                 ", total_theorems)
    print("Theorems proved:                ", theorems_proved)
    print("Total execution time (seconds): ", time_total_end-time_total_start)


if __name__ == "__main__":
    # print(extract_subpath("./coq_projects/demos/Demo.v", "./coq_projects/demos"))

    # test("./theorems/try_generate_output.json", "./theorems/correct_proofs.json", "./coq_projects")
    #test("./theorems/try_generate_output.json", "./theorems/correct_proofs.json", "./coq_projects")
    # print(sys.argv)
    #test("./theorems/correct_proofs_k50_copy.json", "./theorems/correct_proofs_k50_try.json", "./coq_projects", True)

    all_proofs = False
    if (len(sys.argv) == 5) and (sys.argv[4] == "True"):
        all_proofs = True
    print(all_proofs)
    test(sys.argv[1], sys.argv[2], sys.argv[3], all_proofs)

    # test("./theorems/generated_test_theorems_trunc_k25.json", "./theorems/correct_proofs_k25.json", "./coq_projects1", False)

    # UnifySL, coquelicot, verdi-raft, verdi
    # filepath = "./coq_projects/weak-up-to/Make"
    # make_files = ["./coq_projects/weak-up-to/Make", "./coq_projects/buchberger/Make",
    #               "./coq_projects/jordan-curve-theorem/Make", "./coq_projects/dblib/Make",
    #               "./coq_projects/disel/_CoqProject", "./coq_projects/zchinese/Make",
    #               "./coq_projects/zfc/Make", "./coq_projects/dep-map/Make",
    #               "./coq_projects/chinese/Make", "./coq_projects/hoare-tut/Make",
    #               "./coq_projects/huffman/_CoqProject", "./coq_projects/PolTac/_CoqProject",
    #               "./coq_projects/angles/Make", "./coq_projects/coq-procrastination_CoqProject",
    #               "./coq_projects/coq-library-undecidability/_CoqProject",
    #               "./coq_projects/tree-automata/Make", "./coq_projects/fermat4/Make",
    #               "./coq_projects/demos/Make", "./coq_projects/coqoban/Make", "./coq_projects/goedel/Make",
    #               "./coq_projects/zorns-lemma/Make", "./coq_projects/coqrel/_CoqProject",
    #               "./coq_projects/fundamental-arithmetics/Make"]
