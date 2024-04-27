"""
This module tests generated proofs.
def test(input_json_file: str, output_json_file: str, coq_projs_root_folder: str, all_proofs: bool):


Usage
-----
    python ./scripts/test_generated_proofs.py <input_json_file> <output_json_file> <coq_projs_root_folder> [<all_proofs>]

    Argumets:
        <input_json_file>       - path to the dataset with generated proofs.
        <output_json_file>      - output path for the tested proofs.
        <coq_projs_root_folder> - path to the directory containing pre-compiled test Coq projects.
        <all_proofs>            - Optional. Set this argument to "True" to test every generated proof.
                                  Otherwise testing of every theorem will stop on the first correct
                                  proof and proceed with the next theorem.

Examples
--------
    python ./scripts/test_generated_proofs.py ./generated_proofs/n06/generated_comp_n06_k05.json ./tested_proofs/n06/tested_proofs_comp_n06_k05.json ./coq_projects/
    python ./scripts/test_generated_proofs.py ./generated_proofs/n06/generated_comp_n06_k50.json ./tested_proofs/n06/tested_proofs_comp_n06_k50.json ./coq_projects/ True
"""
import json
import subprocess
import os
import time
import sys
import re


def compile_coq_project(project_path: str, timeout_duration:int=10) -> tuple[bool, str]:
    """
    Compiles Coq project. If compilation does not complete
    within timeout_duration seconds, interrupts compilation.

    Parameters
    ----------
    project_path : str
        path to project to compile
    timeout_duration : int, optional
        compilation timeout. Default value is 10

    Returns
    -------
    tuple[bool, str]
        First tuple element represents compilation status.
        Second tuple element is error message. Empty string if no
        error occured.
    """
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

def extract_subpath(full_path: str, root_folder: str) -> str:
    """
    Extracts subpath from the full_path given root_folder.

    Parameters
    ----------
    full_path : str
        full path (starts with root_folder)
    root_folder : str
        root folder path

    Returns
    -------
    str
        extracted subpath
    """
    full_path = os.path.normpath(full_path)
    root_folder = os.path.normpath(root_folder)

    if full_path.startswith(root_folder):
        subpath = full_path[len(root_folder)+1:]
        return subpath
    else:
        return None

def remove_v_files(make_string: str) -> tuple[str, int]:
    """
    Removes all the ".v" files entries from the content of "Make"
    or "_CoqProject" file that is used to instruct which
    files to compile and with which options.

    Parameters
    ----------
    make_string : str
        content of the "Make" or "_CoqProject" file

    Returns
    -------
    tuple[str, int]
        First element of the tuple is the content of the "Make" or "_CoqProject" file
        without ".v" files entries.
        Second element of the tuple is the offset position of the first ".v" file entry
        within the "Make" or "_CoqProject" file.
    """
    pattern = re.compile(r'^.*\.v *\n+', re.MULTILINE)
    match = pattern.search(make_string)
    insert_pos = -1
    if match:
        insert_pos = match.start()
    new_content = pattern.sub('', make_string)
    return new_content, insert_pos

def get_new_make_content(make_string: str, start_pos: int, v_filepath: str) -> str:
    """
    Inserts v_filepath into the make_string on the given start_pos.

    Parameters
    ----------
    make_string : str
        content of the "Make" or "_CoqProject" file without ".v" files entries.
    start_pos : int
        the offset position of the first ".v" file entry within
        the original "Make" or "_CoqProject" file.
    v_filepath : str
        ".v" file entry to insert.

    Returns
    -------
    str
        new content of the "Make" or "_CoqProject" file.
    """
    if start_pos == -1:
        new_content = make_string + "\n" + v_filepath + "\n"
    else:
        new_content = make_string[:start_pos] + v_filepath + "\n"
        if start_pos < len(make_string):
            new_content += make_string[start_pos:]
    return new_content

def test(input_json_file: str, output_json_file: str, coq_projs_root_folder: str, all_proofs: bool):
    """
    Tests generated proofs. All test Coq projects inside the coq_projs_root_folder directory must be
    previously built. To build test Coq projects use Makefile. For example, directory "coq_projects"
    contains Makefile which compiles test projects using command "make test-projects".
    Iterates over each theorem entry in the input_json_file.
    Replaces ground proof truth in corresponding Coq source file with the generated one. Recompiles this
    file and in case of error, saves error message.

    Parameters
    ----------
    input_json_file : str
        path to the JSON file with generated proofs with the following structure:
        {
            "hyperparams": {...},

            "projects": {
                "proj_name": [
                    {
                        "filepath": "",
                        "context": "",
                        "context_tokens": 0,
                        "proof_start_offset": -1,
                        "proof_end_offset": -1,
                        "proof": "",
                        "end_command": "",
                        "generated_proofs": ["", ...]
                    }, ...
                ]
            }
        }
    output_json_file : str
        output path for the JSON file with the following structure:
        {
            "hyperparams": {...},

            "projects": {
                "proj_name": [
                    {
                        "filepath": "",
                        "context": "",
                        "context_tokens": 0,
                        "proof_start_offset": -1,
                        "proof_end_offset": -1,
                        "proof": "",
                        "end_command": "",
                        "generated_proofs": [
                            {
                                "proof": "",
                                "correct": true,
                                "error_msg": ""
                            }, ...
                        ]
                    }, ...
                ], ...
            }
        }
    coq_projs_root_folder : str
        directory with Coq projects
    all_proofs : bool
        if True, checks every generated proof. If False, stops on the first correct proof
        for a given theorem and move to the next theorem.
    """
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
            if total_theorems % 10 == 0:
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
    usage_message = \
'''Error: Wrong CLI usage.

Usage
-----
    python ./scripts/test_generated_proofs.py <input_json_file> <output_json_file> <coq_projs_root_folder> [<all_proofs>]

    Argumets:
        <input_json_file>       - path to the dataset with generated proofs.
        <output_json_file>      - output path for the tested proofs.
        <coq_projs_root_folder> - path to the directory containing pre-compiled test Coq projects.
        <all_proofs>            - Optional. Set this argument to "True" to test every generated proof.
                                  Otherwise testing of every theorem will stop on the first correct
                                  proof and proceed with the next theorem.
'''
    if (len(sys.argv) < 4):
        print(usage_message)
    elif (len(sys.argv) > 5):
        print(usage_message)
    else:
        all_proofs = False
        if (len(sys.argv) == 5) and (sys.argv[4] == "True"):
            all_proofs = True
        try:
            test(sys.argv[1], sys.argv[2], sys.argv[3], all_proofs)
        except Exception as e:
            print(f"Error, an exception occured:\n{e}")
