import argparse
import torch
import os
import time
import json
import numpy as np

from datasets import select_dataset
from models import SSTHopfieldClassifier, BERTClassifier

from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings("ignore", category=UserWarning) 


def parse_arguments():
    """ Parse the command line arguments. """
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', default='2008', type=int)
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'], help="Device to train on")
    parser.add_argument('--dataset', required=True, type=str, choices=["SST", "UDPOS", "SNLI"], help="Dataset to train on")

    args = parser.parse_args()
    return args


def profile(model, args):
    device = torch.device(args.device)

    # Set seed for reproducibility
    #random.seed(seed)
    #np.random.seed(seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # Load the data.
    train_data, _, _, _ = select_dataset(args.dataset, 1, device)
    train_data = iter(train_data)

    model.to(device)
    model.train()

    times = []
    for i in range(1000):
        batch = next(train_data)
        text = batch.text
        
        stime = time.time()
        preds = model(text)#.view(-1, 18)
        times.append(time.time() - stime)

    print(f"Average prediction time (s): {np.mean(times)}")
    

if __name__ == '__main__':
    args = parse_arguments()

    # model = SSTHopfieldClassifier(reduction="none", num_classes=18)
    # model = SSTHopfieldClassifier(reduction="mean", num_classes=5)
    model = BERTClassifier(reduction="none", num_classes=18)
    profile(model, args)
