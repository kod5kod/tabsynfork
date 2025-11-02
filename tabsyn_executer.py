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

print_section("Loading Configuration Files")
computer_name = socket.gethostname()
print(f"\nComputer/node Name: {computer_name}\n")

config_file_path = "config_moriah.toml"
with open(config_file_path, "rb") as f:
    config = tomli.load(f)
if computer_name in config["local"]["local_computer_name"]:
    config["data_path"] = config["local"]["data_path"]
    config["output_path"] = os.path.join(config["local"]["output_path"], config["date"], config["dataset_name"])
elif any(
    item in computer_name for item in config["moriah"]["moriah_node_types"]
):  # computer_name in config["moriah"]["moriah_node_types):
    config["data_path"] = config["moriah"]["data_path"]
    config["output_path"] = os.path.join(config["moriah"]["output_path"], config["date"], config["dataset_name"])
pp(config)

# Loading template for result output dictionary
output_dict_template_path = "output_dict_template.toml"
with open(output_dict_template_path, "rb") as f:
    output_dict_template = tomli.load(f)
pp(output_dict_template)

# Creating output folder
os.makedirs(config["output_path"], exist_ok=True)
print(f"""Folder {config["output_path"]} created or already exists.""")


print_section("Loading Data ")

df_train, df_test, df_valid = data_loader(config["data_path"], config["dataset_name"])
sample_size = df_test.shape[0]
print(
    "Train data size: ",
    df_train.shape,
    "Test data size: ",
    df_test.shape,
    "Validation data size: ",
    df_valid.shape,
)

print_section("Get Metadata and Column Types")
# Get metadata for Polars DataFrame
pl_metadata = get_pl_metadata(df_train)
pp(pl_metadata)

# Retrieves column types from metadata
catCols = [(k) for k, v in pl_metadata["fields"].items() if v["type"] == "categorical"]
intCols = [(k) for k, v in pl_metadata["fields"].items() if v["type"] == "numerical" and v["subtype"] == "int"]
floatCols = [(k) for k, v in pl_metadata["fields"].items() if v["type"] == "numerical" and v["subtype"] == "float"]
binCols = [(k) for k, v in pl_metadata["fields"].items() if v["type"] == "categorical" and v["subtype"] == "binary"]
numCols = intCols + floatCols
print("catCols: ", catCols)
print("intCols: ", intCols)
print("floatCols: ", floatCols)
print("binCols: ", binCols)
print("numCols: ", numCols)
print("target_col: ", config["target_col_name"])

# updating result dict template with dataset information:
dttm_job_started = datetime.today().strftime("%Y%m%d_%H%M%S")
output_dict_template["dataset_name"] = config["dataset_name"]
output_dict_template["dttm_job_started"] = dttm_job_started
output_dict_template["prediction_task"] = config["prediction_task"]
output_dict_template["column_types"] = {
    "catCols": catCols,
    "intCols": intCols,
    "floatCols": floatCols,
    "binCols": binCols,
    "target_col": config["target_col_name"],
}
output_dict_template["data_splits_shapes"] = {
    "data_train": df_train.shape,
    "data_test": df_test.shape,
    "data_validate": df_valid.shape,
}
output_dict_template["computer_name"] = computer_name
pp(output_dict_template)


print_section("Executing Main Script")

python = "python"
main_file_name = "main.py"

for model in config["models_list"]:
    
    print_section("Processing Model: " + model)
    
    # Initialize results dictionary for the current model
    dttm = datetime.today().strftime("%Y%m%d_%H%M%S")
    results = deepcopy(output_dict_template)
    results["model"] = model
    results["dttm_model"] = dttm
        
    # Loop over runs:
    for run in config['runs']:

        print(f"\nRun: {run}\n")
        try:
            # Train the model
            train_start = process_time()
            if model == "tabsyn":
                print("subprocess string: ,", 
                    [python, main_file_name, f"--dataname {config['dataset_name']}", "--method vae", "--mode train"],
                )
                capture_subprocess_output(
                    [python, main_file_name, "--dataname", config["dataset_name"], "--method", "vae", "--mode", "train"],
                )
            capture_subprocess_output(
                [python, main_file_name, "--dataname", config["dataset_name"], "--method", model, "--mode", "train"],
            )
            train_stop = process_time()
            train_time = round(train_stop - train_start, 2)
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
                ],
            )
            sample_stop = process_time()
            sample_time = round(sample_stop - sample_start, 2)
        except subprocess.CalledProcessError as e:
            print(f"Command failed with error: {e}")
            print(f"Standard Error: {e.stderr}")

        results["learning_time"][run] = train_time
        results["sample_time"][run] = sample_time

        # Load synthetic data to add to results dictionary:
        synth_df = pl.read_csv(f"synthetic/{config['dataset_name']}/{model}.csv")
        results["samples"][run] = synth_df

    # Save results
    save_results(results, config["output_path"], dttm_job_started, model)
        
        
    print_section("Done. Exiting.")

# python3 /sci/labs/yuvalb/lee.carlin/repos/tabsynfork/main.py --dataname petfinder_tab --method tabddpm --mode train
# python3 /sci/labs/yuvalb/lee.carlin/repos/tabsynfork/main.py --dataname petfinder_tab --method tabddpm --mode sample --sample_size 2249

# python3 /sci/labs/yuvalb/lee.carlin/repos/tabsynfork/main.py --dataname petfinder_tab --method vae --mode train
# python3 /sci/labs/yuvalb/lee.carlin/repos/tabsynfork/main.py --dataname petfinder_tab --method tabsyn --mode train
# python3 /sci/labs/yuvalb/lee.carlin/repos/tabsynfork/main.py --dataname petfinder_tab --method tabsyn --mode sample --sample_size 2249
