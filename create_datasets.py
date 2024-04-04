import json
import os


def iterate_files(directory: str):
    for root, _, files in os.walk(directory):
        for file in files:
            yield os.path.join(root, file)

def create_dataset(root_directory_path: str, dataset_json_file_path: str, projs_list: list[str]):
    '''
    {data: [{filepath: "", content: ""}, ...]}
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
    proj_split_data = []
    with open(proj_split_path, 'r') as json_file:
        proj_split_data = json.load(json_file)

    create_dataset(root_directory_path, os.path.join(datasets_output_dir, "dataset_train.json"), proj_split_data["projs_train"])
    create_dataset(root_directory_path, os.path.join(datasets_output_dir, "dataset_valid.json"), proj_split_data["projs_valid"])
    create_dataset(root_directory_path, os.path.join(datasets_output_dir, "dataset_test.json"),  proj_split_data["projs_test"])


if __name__ == "__main__":
    create_datasets("./coq_projects/", "./projs_split.json", "./datasets/")
