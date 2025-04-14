import json

#Third parties
import shap
import numpy as np
from qkeras.utils import load_qmodel
from sklearn.metrics import roc_curve, auc
from itertools import combinations
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import mplhep as hep
import tensorflow as tf
import tagger.plot.style as style
from tagger.data.tools import load_data, to_ML
import os
from .common import PT_BINS
from .common import plot_histo
from scipy.stats import norm
from tagger.plot.basic_plots import ROC_binary
import argparse

# Helper functions for signal specific plotting
def filter_process(test_data, process_dir):
    """
    Filter jets from specific signal process to create plots for specified signal processes.
    Comparison done through concatenation of sets to be compared and np unique to check for duplicates.
    """
    train, test, class_labels = load_data(os.path.join("signal_process_data", process_dir), percentage=100)[:3]
    train, test = to_ML(train, class_labels), to_ML(test, class_labels)

    # apply unique to sets to be compared, since there tend to be duplicates
    process_data = np.unique(np.concatenate((train[0], test[0]), axis=0), axis=0)
    unique_test_data, indices_unique_test_data = np.unique(test_data, axis=0, return_index=True)
    comparison_data = np.concatenate((unique_test_data, process_data), axis=0)
    u, index, counts = np.unique(comparison_data, axis=0, return_index=True, return_counts=True)
    process_indices = index[counts == 2]
    filtered_indices = indices_unique_test_data[process_indices]

    return filtered_indices

# fancy signal process labels
def process_labels(process_key):
    processes = {
        'TT_PU200': r't$\bar{t}$ (PU200)',
        'ggHHbbbb_PU200': r'gg $\rightarrow$ HH $\rightarrow$ b$\bar{b}$b$\bar{b}$ (PU200)',
        'VBFHtt_PU200': r'VBF $\rightarrow$ H $\rightarrow$ t$\bar{t}$ (PU200)',
        'ggHHbbtt_PU200': r'gg $\rightarrow$ HH $\rightarrow$ b$\bar{b}$t$\bar{t}$ (PU200)',
        'ggHtt_PU200': r'gg $\rightarrow$ HH $\rightarrow$ t$\bar{t}$ (PU200)',
        }

    return processes[process_key]

def plot_rocs(y_pred, y_test, signal_dir, class_pairs):
    # Make ROC binaries for complete test set and each signal process
    signal_indices = filter_process(X_test, signal_dir)
    y_p, y_t = y_pred[signal_indices], y_test[signal_indices]
    process_label = process_labels(signal_dir)
    binary_dir = os.path.join(plot_dir, signal_dir)

    # loop through all class pairs
    for class_pair in class_pairs:
        ROC_binary(y_p, y_t, class_labels, binary_dir, class_pair, process_label)


if __name__ == "__main__":
    # Load the model
    parser = ArgumentParser()
    parser.add_argument('--model-dir', type=str, default='qkeras_model.h5', help='Path to the model file')
    parser.add_argument('--test-data', type=str, default='test_data', help='Path to the test data directory')
    parser.add_argument('--sample-data', type=str, default='sample_data', help='Path to the sample data directory')
    model = load_qmodel(os.path.join("models", "qkeras_model.h5"))

    # Format the signal data
    signal_output = os.path.join("signal_process_data", signal_process)
    if not os.path.exists(signal_output):
        make_data(infile=args.sample_data, outdir=signal_output, step_size=args.step, extras=args.extras)

    #Load the metadata for class_label
    with open(f"{args.model_dir}/class_label.json", 'r') as file: class_labels = json.load(file)
    with open(f"{args.model_dir}/input_vars.json", 'r') as file: input_vars = json.load(file)

    ROC_dict = {class_label : 0 for class_label in class_labels}

    #Load the testing data
    X_test = np.load(f"{args.model_dir}/testing_data/X_test.npy")
    y_test = np.load(f"{args.model_dir}/testing_data/y_test.npy")
    truth_pt_test = np.load(f"{args.model_dir}/testing_data/truth_pt_test.npy")
    reco_pt_test = np.load(f"{args.model_dir}/testing_data/reco_pt_test.npy")

    #Load model
    model = load_qmodel(f"{args.model_dir}/model/saved_model.h5")
    y_pred = model.predict(X_test)[0]

    # Load the data
    X_train, X_test, y_train, y_test, class_labels = load_data(os.path.join("data", "test_data"), percentage=100)
    X_train, X_test = to_ML(X_train, class_labels), to_ML(X_test, class_labels)
    # Get the predictions
    y_pred = model.predict(X_test)

    # Define directories for signal processes
    plot_dir = os.path.join(args.model_dir, "plots/physics/sample_rocs")
    os.makedirs(plot_dir, exist_ok=True)

    # Define class pairs for ROC curves
    class_pairs = list(combinations(range(len(class_labels)), 2))

    # Plot ROC curves
    plot_rocs()
