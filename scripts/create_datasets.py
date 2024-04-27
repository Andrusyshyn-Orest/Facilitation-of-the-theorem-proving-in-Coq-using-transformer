'''
Module for creating the train/validation/test split from
Coq projects source files. "dataset_train.json", "dataset_valid.json",
"dataset_test.json" files will be created in the specified directory.

Usage
-----
    python ./scripts/create_datasets.py [OPTION...]

    Options:
        -h, --help                           Print help message.
        -c, --coq_projects <coq_projects>    Specify path to the directory with Coq projects.
                                             Default value is "./coq_projects/".
        -p, --projs_split  <projs_split>     Specify path to the split configuration file.
                                             Default value is "./projs_split.json".
        -d, --datasets_dir <datasets_dir>    Specify output directory. Datasets JSON files
                                             will be created here. Default value is "./datasets/".

Examples
--------
    python ./scripts/create_datasets.py
    python ./scripts/create_datasets.py -c "./coq_projects/" --projs_split "./projs_split.json" -d "./datasets/"
'''
import json
import sys
import os
from getopt import getopt


def iterate_files(directory: str):
    '''
    Recursively iterates over all files in folder directory and
    yields filepathes.

    Parameters
    ----------
    directory : str
        path to the folder to iterate

    Yields
    ------
    str
        filepath inside the directory directory.
    '''
    for root, _, files in os.walk(directory):
        for file in files:
            yield os.path.join(root, file)

def create_dataset(root_directory_path: str, dataset_json_file_path: str, projs_list: list[str]):
    '''
    Recursively iterates over Coq source files (".v" files) in the root_directory_path directory,
    that belongs to the projects from projs_list.
    Creates JSON dataset with the following structure: {data: [{filepath: "", content: ""}, ...]}

    Parameters
    ----------
    root_directory_path : str
        path to the directory with Coq projects.
    dataset_json_file_path : str
        filepath of the JSON dataset.
    projs_list : list[str]
        list of projects to build the dataset from.
    '''

    data = []
    count = 0
    total_size = 0
    for project in projs_list:
        dir_path = os.path.join(root_directory_path, project)
        proj_size = 0
        for file_path in iterate_files(dir_path):
            if file_path[-2:] == '.v':
                try:
                    with open(file_path, 'r') as file:
                        content = file.read()
                        data.append({"filepath": file_path, "content": content})
                    count += 1
                    total_size += os.path.getsize(file_path)
                    proj_size += os.path.getsize(file_path)
                except Exception as e:
                    print("Error in filepath: ", file_path)
                    print(e)
    print(f"Total files in {dataset_json_file_path}: ", count)
    print(f"Total size of {dataset_json_file_path}:   {total_size/1024**2:.1f} MB")

    json_data = {"data": data}
    with open(dataset_json_file_path, 'w') as json_file:
        json.dump(json_data, json_file, indent=4)

def create_datasets(root_directory_path: str, proj_split_path: str, datasets_output_dir: str):
    """
    Creates train/validation/test datasets in the datasets_output_dir directory from
    the Coq source files (".v" files) in the root_directory_path directory.
    "dataset_train.json", "dataset_valid.json", "dataset_test.json" files will be
    created in the datasets_output_dir directory (or rewritten if already exist).
    Structure of the datasets is the following: {data: [{filepath: "", content: ""}, ...]}

    Parameters
    ----------
    root_directory_path : str
        path to the directory with Coq projects.
    proj_split_path : str
        path to the JSON file fith train/validation/test split.
    datasets_output_dir : str
        path to the output directory where JSON datasets will be created.
    """
    proj_split_data = []
    with open(proj_split_path, 'r') as json_file:
        proj_split_data = json.load(json_file)

    create_dataset(root_directory_path, os.path.join(datasets_output_dir, "dataset_train.json"), proj_split_data["projs_train"])
    create_dataset(root_directory_path, os.path.join(datasets_output_dir, "dataset_valid.json"), proj_split_data["projs_valid"])
    create_dataset(root_directory_path, os.path.join(datasets_output_dir, "dataset_test.json"),  proj_split_data["projs_test"])

def main():
    """
    Main function that parse CLI args and run script.
    """
    error_message = "Error: Invalid CLI usage.\n"
    usage_message =\
'''
Usage
-----
    python ./scripts/create_datasets.py [OPTION...]

    Options:
        -h, --help                           Print help message.
        -c, --coq_projects <coq_projects>    Specify path to the directory with Coq projects.
                                             Default value is "./coq_projects/".
        -p, --projs_split  <projs_split>     Specify path to the split configuration file.
                                             Default value is "./projs_split.json".
        -d, --datasets_dir <datasets_dir>    Specify output directory. Datasets JSON files
                                             will be created here. Default value is "./datasets/".
'''

    parse_error = False
    provide_help = False
    coq_projects = "./coq_projects/"
    projs_split = "./projs_split.json"
    datasets = "./datasets/"
    try:
        opts, args = getopt(sys.argv[1:],'hc:p:d:',['help', 'coq_projects=', 'projs_split=', 'datasets_dir='])
        if len(args) != 0:
            parse_error = True

        for option, argument in opts:
            if parse_error: break

            if option == '-h' or option == '--help':
                provide_help = True
            elif option == '-c' or option == '--coq_projects':
                coq_projects = argument
            elif option == '-p' or option == '--projs_split':
                projs_split = argument
            elif option == '-d' or option == '--datasets_dir':
                datasets = argument
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
            create_datasets(coq_projects, projs_split, datasets)
        except Exception as e:
            print(f"Error, an exception occured:\n{e}")


if __name__ == "__main__":
    main()
