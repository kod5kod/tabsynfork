# --- tabsyn_executer.py ---
# This script loads configuration, prepares data, runs model training and sampling, and saves results.

import os
import socket
import subprocess
from copy import deepcopy
from datetime import datetime
from pprint import pp
from time import process_time

import polars as pl
import tomli

from utils import capture_subprocess_output, data_loader, get_pl_metadata, print_section, save_results

# --- Load configuration ---
print_section("Loading Configuration Files")
computer_name = socket.gethostname()
print(f"\nComputer/node Name: {computer_name}\n")

config_file_path = "config_moriah.toml"
with open(config_file_path, "rb") as f:
    config = tomli.load(f)

# Set data/output paths based on computer/node type
if computer_name in config["local"]["local_computer_name"]:
    config["data_path"] = config["local"]["data_path"]
    config["output_path"] = os.path.join(config["local"]["output_path"], config["date"], config["dataset_name"])
elif any(item in computer_name for item in config["moriah"]["moriah_node_types"]):
    config["data_path"] = config["moriah"]["data_path"]
    config["output_path"] = os.path.join(config["moriah"]["output_path"], config["date"], config["dataset_name"])
pp(config)

# --- Load output dictionary template ---
output_dict_template_path = "output_dict_template.toml"
with open(output_dict_template_path, "rb") as f:
    output_dict_template = tomli.load(f)
pp(output_dict_template)

# --- Create output folder ---
os.makedirs(config["output_path"], exist_ok=True)
print(f"Folder {config['output_path']} created or already exists.")

# --- Load data ---
print_section("Loading Data")
df_train, df_test, df_valid = data_loader(config["data_path"], config["dataset_name"])

# Cast target column to string for classification tasks
if config["prediction_task"] in ["binary_classification", "multiclass_classification"]:
    for df in [df_train, df_test, df_valid]:
        df = df.with_columns(pl.col(config["target_col_name"]).cast(pl.String))

sample_size = df_test.shape[0]
print(f"Train data size: {df_train.shape}, Test data size: {df_test.shape}, Validation data size: {df_valid.shape}")

# --- Get metadata and column types ---
print_section("Get Metadata and Column Types")
pl_metadata = get_pl_metadata(df_train)
pp(pl_metadata)

catCols = [k for k, v in pl_metadata["fields"].items() if v["type"] == "categorical"]
intCols = [k for k, v in pl_metadata["fields"].items() if v["type"] == "numerical" and v["subtype"] == "int"]
floatCols = [k for k, v in pl_metadata["fields"].items() if v["type"] == "numerical" and v["subtype"] == "float"]
binCols = [k for k, v in pl_metadata["fields"].items() if v["type"] == "categorical" and v["subtype"] == "binary"]
numCols = intCols + floatCols
print(
    f"catCols: {catCols}\nintCols: {intCols}\nfloatCols: {floatCols}\nbinCols: {binCols}\nnumCols: {numCols}\ntarget_col: {config['target_col_name']}"
)

# --- Update result dict template with dataset info ---
dttm_job_started = datetime.today().strftime("%Y%m%d_%H%M%S")
output_dict_template.update(
    {
        "dataset_name": config["dataset_name"],
        "dttm_job_started": dttm_job_started,
        "prediction_task": config["prediction_task"],
        "column_types": {
            "catCols": catCols,
            "intCols": intCols,
            "floatCols": floatCols,
            "binCols": binCols,
            "target_col": config["target_col_name"],
        },
        "data_splits_shapes": {
            "data_train": df_train.shape,
            "data_test": df_test.shape,
            "data_validate": df_valid.shape,
        },
        "computer_name": computer_name,
    }
)
pp(output_dict_template)

# --- Main execution loop ---
print_section("Executing Main Script")
python = "python"
main_file_name = "main.py"

for model in config["models_list"]:
    print_section(f"Processing Model: {model}")
    dttm = datetime.today().strftime("%Y%m%d_%H%M%S")
    results = deepcopy(output_dict_template)
    results["model"] = model
    results["dttm_model"] = dttm

    for run in config["runs"]:
        print(f"\nRun: {run}\n")
        train_time = sample_time = None
        try:
            # Train the model
            train_start = process_time()
            if model == "tabsyn":
                # Train VAE for tabsyn
                print(
                    "subprocess string:",
                    [
                        python,
                        main_file_name,
                        "--dataname",
                        config["dataset_name"],
                        "--method",
                        "vae",
                        "--mode",
                        "train",
                    ],
                )
                capture_subprocess_output(
                    [python, main_file_name, "--dataname", config["dataset_name"], "--method", "vae", "--mode", "train"]
                )
            capture_subprocess_output(
                [python, main_file_name, "--dataname", config["dataset_name"], "--method", model, "--mode", "train"]
            )
            train_stop = process_time()
            train_time = round(train_stop - train_start, 2)
            print(f"Done with training {model} in {train_time} seconds.")
        except subprocess.CalledProcessError as e:
            print(f"Training failed: {e}\nStderr: {e.stderr}")

        try:
            # Sample from the trained model
            sample_start = process_time()
            capture_subprocess_output(
                [
                    python,
                    main_file_name,
                    "--dataname",
                    config["dataset_name"],
                    "--method",
                    model,
                    "--mode",
                    "sample",
                    "--sample_size",
                    str(sample_size),
                ]
            )
            sample_stop = process_time()
            sample_time = round(sample_stop - sample_start, 2)
            print(f"Done with sampling {model} in {sample_time} seconds.")
        except subprocess.CalledProcessError as e:
            print(f"Sampling failed: {e}\nStderr: {e.stderr}")

        # Store timing and results
        if "learning_time" not in results:
            results["learning_time"] = {}
        if "sample_time" not in results:
            results["sample_time"] = {}
        if "samples" not in results:
            results["samples"] = {}
        results["learning_time"][run] = train_time
        results["sample_time"][run] = sample_time

        # Load synthetic data
        synth_path = f"synthetic/{config['dataset_name']}/{model}.csv"
        if os.path.exists(synth_path):
            synth_df = pl.read_csv(synth_path)
            results["samples"][run] = synth_df
        else:
            print(f"Synthetic data not found at {synth_path}")

    # Save results for this model
    save_results(results, config["output_path"], f"{config['dataset_name']}_{dttm_job_started}", model)
    print_section(f"Done with Model: {model}")

print_section("Done with all Models")

# Example CLI commands for reference:
# python3 main.py --dataname petfinder_tab --method tabddpm --mode train
# python3 main.py --dataname petfinder_tab --method tabddpm --mode sample --sample_size 2249
# python3 main.py --dataname petfinder_tab --method vae --mode train
# python3 main.py --dataname petfinder_tab --method tabsyn --mode train
# python3 main.py --dataname petfinder_tab --method tabsyn --mode sample --sample_size 2249
