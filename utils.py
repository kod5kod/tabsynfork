import argparse
import importlib
import io
import os
import pickle
import selectors
import subprocess
import sys

import polars as pl


def execute_function(method, mode):
    if method == "vae":
        mode = "train"
    mode = "main" if mode == "train" else "sample"

    if method == "vae":
        module_name = f"tabsyn.vae.main"
    elif method == "tabsyn":
        module_name = f"tabsyn.{mode}"
    elif method == "tabddpm":
        module_name = f"baselines.tabddpm.main_train" if mode == "main" else f"baselines.tabddpm.main_sample"
    else:
        module_name = f"baselines.{method}.{mode}"

    try:
        train_module = importlib.import_module(module_name)
        train_function = getattr(train_module, "main")
    except ModuleNotFoundError:
        print(f"Module {module_name} not found.")
        exit(1)
    except AttributeError:
        print(f"Function 'main' not found in module {module_name}.")
        exit(1)
    return train_function


def get_args():
    parser = argparse.ArgumentParser(description="Pipeline")

    # General configs
    parser.add_argument("--dataname", type=str, default="adult", help="Name of dataset.")
    parser.add_argument("--mode", type=str, default="train", help="Mode: train or sample.")
    parser.add_argument("--method", type=str, default="tabsyn", help="Method: tabsyn or baseline.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index.")

    """ configs for CTGAN """

    parser.add_argument("-e", "--epochs", default=1000, type=int, help="Number of training epochs")
    parser.add_argument(
        "--no-header",
        dest="header",
        action="store_false",
        help="The CSV file has no header. Discrete columns will be indices.",
    )

    parser.add_argument("-m", "--metadata", help="Path to the metadata")
    parser.add_argument("-d", "--discrete", help="Comma separated list of discrete columns without whitespaces.")
    parser.add_argument(
        "-n", "--num-samples", type=int, help="Number of rows to sample. Defaults to the training data size"
    )

    parser.add_argument("--generator_lr", type=float, default=2e-4, help="Learning rate for the generator.")
    parser.add_argument("--discriminator_lr", type=float, default=2e-4, help="Learning rate for the discriminator.")

    parser.add_argument("--generator_decay", type=float, default=1e-6, help="Weight decay for the generator.")
    parser.add_argument("--discriminator_decay", type=float, default=0, help="Weight decay for the discriminator.")

    parser.add_argument("--embedding_dim", type=int, default=1024, help="Dimension of input z to the generator.")
    parser.add_argument(
        "--generator_dim",
        type=str,
        default="1024,2048,2048,1024",
        help="Dimension of each generator layer. Comma separated integers with no whitespaces.",
    )
    parser.add_argument(
        "--discriminator_dim",
        type=str,
        default="1024,2048,2048,1024",
        help="Dimension of each discriminator layer. Comma separated integers with no whitespaces.",
    )

    parser.add_argument("--batch_size", type=int, default=500, help="Batch size. Must be an even number.")
    parser.add_argument("--save", default=None, type=str, help="A filename to save the trained synthesizer.")
    parser.add_argument("--load", default=None, type=str, help="A filename to load a trained synthesizer.")

    parser.add_argument("--sample_condition_column", default=None, type=str, help="Select a discrete column name.")
    parser.add_argument(
        "--sample_condition_column_value",
        default=None,
        type=str,
        help="Specify the value of the selected discrete column.",
    )

    """ configs for GReaT """

    parser.add_argument("--bs", type=int, default=16, help="(Maximum) batch size")

    """ configs for CoDi """

    # General Options
    parser.add_argument("--logdir", type=str, default="./codi_exp", help="log directory")
    parser.add_argument("--train", action="store_true", help="train from scratch")
    parser.add_argument("--eval", action="store_true", help="load ckpt.pt and evaluate")

    # Network Architecture
    parser.add_argument("--encoder_dim", nargs="+", type=int, help="encoder_dim")
    parser.add_argument("--encoder_dim_con", type=str, default="512,1024,1024,512", help="encoder_dim_con")
    parser.add_argument("--encoder_dim_dis", type=str, default="512,1024,1024,512", help="encoder_dim_dis")
    parser.add_argument("--nf", type=int, help="nf")
    parser.add_argument("--nf_con", type=int, default=16, help="nf_con")
    parser.add_argument("--nf_dis", type=int, default=64, help="nf_dis")
    parser.add_argument("--input_size", type=int, help="input_size")
    parser.add_argument("--cond_size", type=int, help="cond_size")
    parser.add_argument("--output_size", type=int, help="output_size")
    parser.add_argument("--activation", type=str, default="relu", help="activation")

    # Training
    parser.add_argument("--training_batch_size", type=int, default=4096, help="batch size")
    parser.add_argument("--eval_batch_size", type=int, default=2100, help="batch size")
    parser.add_argument("--T", type=int, default=50, help="total diffusion steps")
    parser.add_argument("--beta_1", type=float, default=0.00001, help="start beta value")
    parser.add_argument("--beta_T", type=float, default=0.02, help="end beta value")
    parser.add_argument("--lr_con", type=float, default=2e-03, help="target learning rate")
    parser.add_argument("--lr_dis", type=float, default=2e-03, help="target learning rate")
    parser.add_argument("--total_epochs_both", type=int, default=20000, help="total training steps")  # codi
    parser.add_argument("--grad_clip", type=float, default=1.0, help="gradient norm clipping")
    parser.add_argument("--parallel", action="store_true", help="multi gpu training")

    # Sampling
    parser.add_argument("--sample_step", type=int, default=2000, help="frequency of sampling")
    parser.add_argument("--sample_size", type=int, default=2000, help="number of samples to generate")

    # Continuous diffusion model
    parser.add_argument(
        "--mean_type", type=str, default="epsilon", choices=["xprev", "xstart", "epsilon"], help="predict variable"
    )
    parser.add_argument(
        "--var_type", type=str, default="fixedsmall", choices=["fixedlarge", "fixedsmall"], help="variance type"
    )

    # Contrastive Learning
    parser.add_argument("--ns_method", type=int, default=0, help="negative condition method")
    parser.add_argument("--lambda_con", type=float, default=0.2, help="lambda_con")
    parser.add_argument("--lambda_dis", type=float, default=0.2, help="lambda_dis")
    ################

    # configs for TabDDPM
    parser.add_argument("--ddim", action="store_true", default=False, help="Whether use DDIM sampler")

    # configs for SMOTE
    parser.add_argument("--cat_encoding", type=str, default="one-hot", help="Encoding method for categorical features")

    # configs for traing TabSyn's VAE
    parser.add_argument("--max_beta", type=float, default=1e-2, help="Maximum beta")
    parser.add_argument("--min_beta", type=float, default=1e-5, help="Minimum beta.")
    parser.add_argument("--lambd", type=float, default=0.7, help="Batch size.")

    # configs for sampling
    parser.add_argument("--save_path", type=str, default=None, help="Path to save synthetic data.")
    parser.add_argument("--steps", type=int, default=50, help="NFEs.")

    args = parser.parse_args()

    return args


# Utility: Print section headers
def print_section(title):
    """Print a section header.

    Args:
        title (str): The title of the section.
    """
    print(f"\n{'#' * 25}\n{title}\n{'#' * 25}\n")


def capture_subprocess_output(subprocess_args):
    """
    Capture and display subprocess output in real-time while storing it in a buffer.
    This function runs a subprocess command and simultaneously:
    1. Displays the output to stdout as it's generated (line by line)
    2. Captures all output in a buffer for later retrieval
    Args:
        subprocess_args (list): A list of command arguments to pass to subprocess.Popen.
                               Example: ['python', 'script.py', '--arg1', 'value']
    Returns:
        tuple: A tuple containing:
            - success (bool): True if the subprocess exited with return code 0, False otherwise
            - output (str): The complete captured output from the subprocess (stdout and stderr combined)
    Example:
        >>> success, output = capture_subprocess_output(['echo', 'Hello World'])
        Hello World
        >>> print(success)
        True
        >>> print(output)
        Hello World
    Note:
        - stderr is redirected to stdout, so all output is captured together
        - Uses selectors for efficient I/O handling
        - Output is line-buffered for real-time display
    """

    # Start subprocess
    # bufsize = 1 means output is line buffered
    # universal_newlines = True is required for line buffering
    process = subprocess.Popen(
        subprocess_args, bufsize=1, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True
    )

    # Create callback function for process output
    buf = io.StringIO()

    def handle_output(stream, mask):
        # Because the process' output is line buffered, there's only ever one
        # line to read when this function is called
        line = stream.readline()
        buf.write(line)
        sys.stdout.write(line)

    # Register callback for an "available for read" event from subprocess' stdout stream
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ, handle_output)

    # Loop until subprocess is terminated
    while process.poll() is None:
        # Wait for events and handle them with their registered callbacks
        events = selector.select()
        for key, mask in events:
            callback = key.data
            callback(key.fileobj, mask)

    # Ensure all remaining output is processed
    while True:
        line = process.stdout.readline()
        if not line:
            break
        buf.write(line)
        sys.stdout.write(line)

    # Get process return code
    return_code = process.wait()
    selector.close()

    success = return_code == 0

    # Store buffered output
    output = buf.getvalue()
    buf.close()

    return (success, output)


def get_pl_metadata(dataframe: pl.DataFrame) -> dict:
    """
    Get the types metadata of a Polars dataframe.

    Args:
        dataframe (pl.DataFrame): A Polars dataframe

    Returns:
        dict: A dict with type metadata information

    Notes:
        The metadata will be a dictionary with the column name as the key and a
        dictionary with the type and subtype as the value. The type will be one
        of ['categorical', 'numerical'] and the subtype will be one of
        ['binary', 'multi'] or ['float', 'int'].
    """
    tmp = {}
    metadata = {}

    for col in dataframe.columns:
        dtype = dataframe.schema[col]
        col_data = dataframe[col].drop_nulls()

        if dtype == pl.String:
            unique_vals = col_data.unique()
            if unique_vals.len() == 2:
                tmp[col] = {"type": "categorical", "subtype": "binary"}
            else:
                tmp[col] = {"type": "categorical", "subtype": "multi"}

        elif dtype in [pl.Float32, pl.Float64]:
            tmp[col] = {"type": "numerical", "subtype": "float"}

        elif dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64]:
            tmp[col] = {"type": "numerical", "subtype": "int"}

        else:
            print(f"Didn't match on any data type for column: {col}")

    metadata["fields"] = tmp
    return metadata


def data_loader(data_path, dataset_name):
    """
    Load training, testing, and validation datasets from CSV files.
    This function reads preprocessed CSV files for a given dataset and returns them as
    Polars DataFrames. The files are expected to follow a specific naming convention and
    directory structure.
    Args:
        data_path (str): The base directory path where the dataset folders are located.
        dataset_name (str): The name of the dataset, which is used to construct the file
            paths and locate the dataset folder.
    Returns:
        tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]: A tuple containing three Polars
            DataFrames in the following order:
            - train_df: Training dataset
            - test_df: Testing dataset
            - valid_df: Validation dataset
    Raises:
        Exception: Re-raises any exception that occurs during file reading, such as
            FileNotFoundError if the CSV files don't exist, or parsing errors if the
            CSV format is invalid.
    Example:
        >>> train, test, valid = data_loader('/path/to/data', 'my_dataset')
        >>> print(train.shape)
        (1000, 10)
    Note:
        Expected file structure:
        {data_path}/{dataset_name}/{dataset_name}_processed_train.csv
        {data_path}/{dataset_name}/{dataset_name}_processed_test.csv
        {data_path}/{dataset_name}/{dataset_name}_processed_valid.csv
    """

    # Placeholder for actual data loading logic

    train_path = os.path.join(data_path, dataset_name, f"{dataset_name}_processed_train.csv")
    test_path = os.path.join(data_path, dataset_name, f"{dataset_name}_processed_test.csv")
    valid_path = os.path.join(data_path, dataset_name, f"{dataset_name}_processed_valid.csv")
    try:
        train_df = pl.read_csv(train_path)
        test_df = pl.read_csv(test_path)
        valid_df = pl.read_csv(valid_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        raise e

    return train_df, test_df, valid_df


def save_results(results, output_path, prefix, model_name):
    """Save results to a pickle file.

    Args:
        results (dict): The results dictionary to save.
        output_path (str): The directory where the results will be saved.
        prefix (str): A prefix for the output filename.
        model_name (str): The name of the model being evaluated.
    """
    fname = os.path.join(output_path, f"{prefix}_{model_name}_results.pkl")
    with open(fname, "wb") as f:
        pickle.dump(results, f)
    print(f"Results saved to {fname}")
