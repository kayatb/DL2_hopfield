import torch
import wandb
import argparse

from datasets import select_dataset
from models import SSTHopfieldClassifier, BERTClassifier

from sklearn.metrics import accuracy_score

import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def parse_arguments():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True,
        type=str,
        choices=["SST", "UDPOS", "SNLI"],
        help="Dataset to train on",
    )
    parser.add_argument(
        "--device", default="cuda", choices=["cuda", "cpu"], help="Device to train on"
    )
    parser.add_argument("--seed", default="2008", type=int)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="If filename given, start training from that checkpoint",
    )
    parser.add_argument(
        "--output_dir",
        default="output",
        help="Directory to save output (e.g. checkpoints) to",
    )
    parser.add_argument(
        "--save_every",
        default=-1,
        type=int,
        help="Save a checkpoint every x epochs. -1 denotes no intermediate saving",
    )

    # Training settings
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--lr", "--learning_rate", default=1e-3, type=float)
    parser.add_argument("--epochs", default=10, type=int)

    args = parser.parse_args()
    return args


def calc_accuracy(predictions, labels, pad_index):
    """Calculate the accuracy score. Ignore padding tokens."""
    predictions = torch.argmax(predictions, dim=-1).detach().cpu().numpy().flatten()
    labels = labels.detach().cpu().view(-1).numpy().flatten()

    predictions = predictions[predictions != pad_index]
    labels = labels[predictions != pad_index]
    return accuracy_score(labels, predictions)


def evaluate(model, data, pad_index):
    """Evaluate the model on the data by calculating the accuracy."""
    accuracy = 0
    model.eval()

    with torch.no_grad():
        for batch in data:
            text = batch.text
            labels = batch.label.long().view(-1)
            preds = model(text).view(-1, n_classes)

            acc = calc_accuracy(preds, labels, pad_index)
            accuracy += acc

    return accuracy / len(data)


def train(model, args):
    """Train the model and return it."""

    device = torch.device(args.device)

    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # Load the data.
    train_data, val_data, test_data, pad_index = select_dataset(
        args.dataset, args.batch_size, device
    )

    # Define optimizer and criterion.
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    # If a checkpoint was given, load it.
    if args.checkpoint:
        checkpoint = args.load(args.checkpoint)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        # Also load lr scheduler if one is used.

    # For outputting performance across epochs.

    model.to(device)
    model.train()

    print(
        "Number of parameters: ",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )

    wandb.init(project="hopfield")
    wandb.config.update(vars(args))
    evaluate(model, val_data, pad_index)

    for e in range(args.epochs):
        epoch_loss = 0
        epoch_acc = 0

        for batch in train_data:
            text = batch.text
            labels = batch.label.view(-1).long().view(-1)

            optimizer.zero_grad()

            preds = model(text).view(-1, n_classes)
            loss = criterion(preds, labels)

            loss.backward()
            optimizer.step()

            acc = calc_accuracy(preds, labels, pad_index)

            epoch_loss += loss.item()
            epoch_acc += acc

        train_loss = epoch_loss / len(train_data)
        train_acc = epoch_acc / len(train_data)
        val_acc = evaluate(model, val_data, pad_index)

        wandb.log(
            {"train_loss": train_loss, "train_acc": train_acc, "val_acc": val_acc}
        )


if __name__ == "__main__":
    args = parse_arguments()

    n_classes = 0
    reduction = "mean"
    if args.dataset == "SST":
        n_classes = 5
    elif args.dataset == "UDPOS":
        n_classes = 18
        reduction = "none"

    model = SSTHopfieldClassifier(reduction=reduction, num_classes=n_classes)
    train(model, args)
