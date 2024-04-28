"""
Builds plots of training process, success rate metric, and pass@k metric.
Run this script without any arguments from root directory.
Output files: "./images/n12_valid.png", "./images/n12_lr.png", "./images/nall_valid.png",
"./images/sr_trunc.png", "./images/sr_comp.png",
"./images/passk_trunc.png", "./images/passk_comp.png"
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import math


model_color = {"n02": "#1f77b4",
               "n04": "#ff7f0e",
               "n06": "#2ca02c",
               "n08": "#d62728",
               "n10": "#9467bd",
               "n12": "#8c564b"}


def get_x_y_from_log(json_filepath: str) -> dict:
    """
    Gets training and validation data from the
    training logs.

    Parameters
    ----------
    json_filepath : str
        path to the training log.

    Returns
    -------
    dict
        Returns training data of the following structure:
        {
            "train_loss": ([], []),
            "valid_loss": ([], []),
            "lr":         ([], [])
        }
        In every tuple, first list is the list of steps. Second list
        is the list of corresponding values.
    """
    if not os.path.exists(json_filepath):
        print("ERROR: json_filepath DOES NOT EXIST")
        return
    output = {
                "train_loss": ([], []),
                "valid_loss": ([], []),
                "lr":         ([], [])
            }
    with open(json_filepath, mode="r") as json_file:
        json_data = json.load(json_file)

    prev_comleted_steps = json_data["train"][0]["completed_steps"]
    prev_lr = json_data["train"][0]["lr"][0]
    train_losses = []
    cs = 0
    for entry in json_data["train"]:
        cs = entry["completed_steps"]
        if cs == prev_comleted_steps:
            train_losses.append(entry["loss/train"])
            continue
        else:
            output["train_loss"][0].append(prev_comleted_steps)
            output["train_loss"][1].append(sum(train_losses)/len(train_losses))
            output["lr"][0].append(prev_comleted_steps)
            output["lr"][1].append(prev_lr)
            train_losses = [entry["loss/train"]]
            prev_comleted_steps = cs
            prev_lr = entry["lr"][0]
    output["train_loss"][0].append(cs)
    output["train_loss"][1].append(sum(train_losses)/len(train_losses))
    output["lr"][0].append(cs)
    output["lr"][1].append(prev_lr)

    for entry in json_data["valid"]:
        cs = entry["completed_steps"]
        output["valid_loss"][0].append(cs)
        output["valid_loss"][1].append(entry["loss/eval"])

    return output

def build_training_n12():
    """
    Builds two plots. One is validation losses during training of n12 model.
    The othe one is cosine learning rate decay during training n12 model
    with lr=8e-4.
    Output files: "./images/n12_valid.png", "./images/n12_lr.png"
    """
    output_1em5 = get_x_y_from_log("./training_logs/v30_a95_lrcos1em5_mod12l12h768e.json")
    output_8em5 = get_x_y_from_log("./training_logs/v30_a95_lrcos8em5_mod12l12h768e.json")
    output_5em4 = get_x_y_from_log("./training_logs/v30_a95_lrcos5em4_mod12l12h768e.json")
    output_8em4 = get_x_y_from_log("./training_logs/v30_a95_lrcos8em4_mod12l12h768e.json")
    output_3em3 = get_x_y_from_log("./training_logs/v30_a95_lrcos3em3_mod12l12h768e.json")

    plt.figure()
    plt.figure(figsize=(5, 4))
    markersize = 2
    plt.plot(output_1em5["valid_loss"][0], output_1em5["valid_loss"][1], label='lr=1e-5', marker='o', markersize=markersize)
    plt.plot(output_8em5["valid_loss"][0], output_8em5["valid_loss"][1], label='lr=8e-5', marker='o', markersize=markersize)
    plt.plot(output_5em4["valid_loss"][0], output_5em4["valid_loss"][1], label='lr=5e-4', marker='o', markersize=markersize)
    plt.plot(output_8em4["valid_loss"][0], output_8em4["valid_loss"][1], label='lr=8e-4', marker='o', markersize=markersize)
    plt.plot(output_3em3["valid_loss"][0], output_3em3["valid_loss"][1], label='lr=3e-3', marker='o', markersize=markersize)

    plt.xlabel('steps')
    plt.ylabel('validation loss')
    plt.title('Validation loss of n12 model')
    plt.legend()

    plt.savefig("./images/n12_valid.png", dpi=500)
    #########################################################################################################3
    plt.figure()
    plt.figure(figsize=(5, 4))
    plt.plot(output_8em4["lr"][0], output_8em4["lr"][1], label='lr=8e-4')

    plt.xlabel('steps')
    plt.ylabel('learning rate')
    plt.title('Learning rate decay of n12 model')
    plt.legend()
    plt.subplots_adjust(left=0.17)
    plt.savefig("./images/n12_lr.png", dpi=500)

def build_training_all():
    """
    Builds plot of validation loss for n02, n04, n06, n08, n10, n12 models
    for lr=8e-4. Output file: "./images/nall_valid.png"
    """
    global model_color
    output_12 = get_x_y_from_log("./training_logs/v30_a95_lrcos8em4_mod12l12h768e.json")
    output_10 = get_x_y_from_log("./training_logs/v30_a95_lrcos8em4_mod10l10h640e.json")
    output_8 = get_x_y_from_log("./training_logs/v30_a95_lrcos8em4_mod8l8h512e.json")
    output_6 = get_x_y_from_log("./training_logs/v30_a95_lrcos8em4_mod6l6h384e.json")
    output_4 = get_x_y_from_log("./training_logs/v30_a95_lrcos8em4_mod4l4h256e.json")
    output_2 = get_x_y_from_log("./training_logs/v30_a95_lrcos8em4_mod2l2h128e.json")

    plt.figure()
    plt.figure(figsize=(5, 4))
    markersize = 2
    plt.plot(output_12["valid_loss"][0], output_12["valid_loss"][1], label='n12', marker='o', markersize=markersize, color=model_color["n12"])
    plt.plot(output_10["valid_loss"][0], output_10["valid_loss"][1], label='n10', marker='o', markersize=markersize, color=model_color["n10"])
    plt.plot(output_8["valid_loss"][0], output_8["valid_loss"][1], label='n08', marker='o', markersize=markersize, color=model_color["n08"])
    plt.plot(output_6["valid_loss"][0][:59], output_6["valid_loss"][1][:59], label='n06', marker='o', markersize=markersize, color=model_color["n06"])
    plt.plot(output_4["valid_loss"][0], output_4["valid_loss"][1], label='n04', marker='o', markersize=markersize, color=model_color["n04"])
    plt.plot(output_2["valid_loss"][0], output_2["valid_loss"][1], label='n02', marker='o', markersize=markersize, color=model_color["n02"])


    plt.xlabel('steps')
    plt.ylabel('validation loss')
    plt.title('Validation loss of different models')
    plt.legend()

    plt.savefig("./images/nall_valid.png", dpi=500)

def get_success_rate(filepath: str, all: bool=True, projects: list=[]) -> tuple[int, int]:
    """
    Calculates success rate of the theorems in the filepath.

    Parameters
    ----------
    filepath : str
        path to the file fith tested theorems.
    all : bool, optional
        If True, calculates success rate of all theorems.
        If False, includes only theorems from projects.
        Default value is True
    projects : list, optional
        If all==False, includes only theorems from this list.
        Default value is []

    Returns
    -------
    tuple[int, int]
        First element is total number of proofs included.
        Second element is number of proved theorems.
    """
    with open(filepath, mode='r') as file:
        file_data = json.load(file)

    total_proofs = 0
    proved = 0
    for project in file_data["projects"].keys():
        if ((not all) and (project not in projects)):
            continue
        for entry in file_data["projects"][project]:
            total_proofs += 1
            for gen_proof in entry["generated_proofs"]:
                if gen_proof["correct"]:
                    proved += 1
                    break
    return (total_proofs, proved)

def pass_k(n: int, c: int, k: int) -> float:
    """
    Calculates pass@k for one theorem. c must be <= n.
    k must be <= n.

    Parameters
    ----------
    n : int
        number of generations per theorem.
    c : int
        number of correct proofs.
    k : int
        sample size.

    Returns
    -------
    float
        pass@k for one theorem.
    """
    if n-c < k:
        return 1
    return (1 - (math.comb(n-c, k)/math.comb(n, k)))

def get_pass_k(filepath: str, k: int, all: bool=True, projects: list=[]) -> float:
    """
    Calculates pass@k for the theorems from filepath.

    Parameters
    ----------
    filepath : str
        pth to the file with tested theorems.
    k : int
        k in the pass@k formula.
    all : bool, optional
        If True, calculates pass@k of all theorems.
        If False, includes only theorems from projects.
        Default value is True
    projects : list, optional
        If all==False, includes only theorems from this list.
        Default value is []

    Returns
    -------
    float
        pass@k metric
    """
    with open(filepath, mode='r') as file:
        file_data = json.load(file)
    for project in file_data["projects"].keys():
        n = len(file_data["projects"][project][0]["generated_proofs"])
        break

    pass_ks = []
    for project in file_data["projects"].keys():
        if ((not all) and (project not in projects)):
            continue
        for entry in file_data["projects"][project]:
            correct = 0
            for gen_proof in entry["generated_proofs"]:
                if gen_proof["correct"]:
                    correct += 1
            pass_ks.append(pass_k(n, correct, k))
    return sum(pass_ks)/len(pass_ks)


def build_success_rate_all(experiment: str):
    """
    Build plot with success rate for different ks [1, 5, 10, 25, 50]
    for different models. Outputs file f"./images/sr_{experiment}.png"

    Parameters
    ----------
    experiment : str
        Either "comp" or "trunc". If "comp", does not
        include n02 and n04 models.
    """
    global model_color
    folder = "./tested_proofs/"
    prefix = "tested_proofs_"
    if experiment not in ("trunc", "comp"):
        return

    models = ["n12", "n10", "n08", "n06", "n04", "n02"]
    if experiment == "comp":
        models = models[:-2]
    ks = ["01", "05", "10", "25", "50"]
    xs = [1, 5, 10, 25, 50]

    plt.figure()
    plt.figure(figsize=(5, 4))
    markersize = 3
    for model in models:
        ys = []
        for k in ks:
            filepath = os.path.join(folder, model, f"{prefix}{experiment}_{model}_k{k}.json")
            total, proven = get_success_rate(filepath)
            ys.append(round((proven/total)*100,2))
        plt.plot(xs, ys, label=model, marker='o', markersize=markersize, color=model_color[model])

    plt.xlabel('k')
    plt.ylabel('success rate, %')
    plt.title('success rate depending on k')
    plt.legend()

    plt.savefig(f"./images/sr_{experiment}.png", dpi=500)

def build_pass_k_all(experiment: str):
    """
    Build plot with pass@k for different ks [1, 5, 10, 25, 50]
    for different models. Outputs file f"./images/passk_{experiment}.png"

    Parameters
    ----------
    experiment : str
        Either "comp" or "trunc". If "comp", does not
        include n02 and n04 models.
    """
    global model_color
    folder = "./tested_proofs/"
    prefix = "tested_proofs_"
    if experiment not in ("trunc", "comp"):
        return

    models = ["n12", "n10", "n08", "n06", "n04", "n02"]

    if experiment == "comp":
        models = models[:-2]
    ks = ["01", "05", "10", "25", "50"]
    xs = [1, 5, 10, 25, 50]

    plt.figure()
    plt.figure(figsize=(5, 4))
    markersize = 3
    for model in models:
        filepath = os.path.join(folder, model, f"{prefix}{experiment}_{model}_k{50}.json")
        ys = []
        for k in ks:
            pass_k = get_pass_k(filepath, int(k))
            ys.append(round(pass_k*100,2))
        plt.plot(xs, ys, label=model, marker='o', markersize=markersize, color=model_color[model])

    plt.xlabel('k')
    plt.ylabel('pass@k, %')
    plt.title('pass@k depending on k')
    plt.legend()

    plt.savefig(f"./images/passk_{experiment}.png", dpi=500)


if __name__ == "__main__":
    try:
        build_training_n12()
        build_training_all()
        build_success_rate_all("trunc")
        build_pass_k_all("trunc")
        build_success_rate_all("comp")
        build_pass_k_all("comp")
    except Exception as e:
        print(f"Error, an exception occured:\n{e}")
