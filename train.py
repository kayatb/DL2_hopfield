import argparse
import torch
import os
import json

from datasets import select_dataset


def parse_arguments():
    """ Parse the command line arguments. """
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', required=True, type=str, choices=["SST", "UDPOS", "SNLI"], help="Dataset to train on")
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'], help="Device to train on")
    parser.add_argument('--seed', default='2008', type=int)
    parser.add_argument('--checkpoint', default=None, help="If filename given, start training from that checkpoint")
    parser.add_argument('--output_dir', default='output', help="Directory to save output (e.g. checkpoints) to")
    parser.add_argument('--save_every', default=-1, type=int, help="Save a checkpoint every x epochs. -1 denotes no intermediate saving")
    
    # Training settings
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('-lr', '--learning_rate', default=1e-4, type=float)
    parser.add_argument('--epochs', default=10, type=int)

    # TODO: model settings

    args = parser.parse_args()
    return args


# TODO:
def calc_accuracy(predictions, labels, pad_index):
    """ Calculate the accuracy score. Ignore padding tokens. """
    return 0


def evaluate(model, data, pad_index):
    """ Evaluate the model on the data by calculating the accuracy. """
    accuracy = 0
    model.eval()

    with torch.no_grad():
        for batch in data:
            # FIXME: SNLI dataset contains hypothesis and premis instead of text.
            text = batch.text
            labels = batch.label

            preds = model(text)
            acc = calc_accuracy(preds, labels, pad_index)
            accuracy += acc
    
    return accuracy / len(data)


def train(model, args):
    """ Train the model and return it. """
    print("=== Starting Training ===")

    device = torch.device(args.device)

    # Set seed for reproducibility
    #random.seed(seed)
    #np.random.seed(seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # Load the data.
    train_data, val_data, test_data, pad_index = select_dataset(args.dataset, args.batch_size, device)
    
    # Define optimizer and criterion.
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  # Could add a lr-scheduler.
    criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_index).to(device)

    # TODO: calculate and print number of model params.

    # If a checkpoint was given, load it.
    if args.checkpoint:
        checkpoint = args.load(args.checkpoint)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        # Also load lr scheduler if one is used.
    
    # For outputting performance across epochs.
    # TODO: keep track of validation loss.
    performance_dict = {'train_loss': [], 'train_acc': [], 'val_acc': [], 'test_acc': []}

    os.makedirs(args.output_dir, exist_ok=True)

    model.to(device)
    model.train()

    best_val_acc = 0
    for e in range(args.epochs):
        epoch_loss = 0
        epoch_acc = 0

        for batch in train_data:
            # FIXME: the SNLI dataset contains hypothesis and premis instead of text.
            text = batch.text
            labels = batch.label

            optimizer.zero_grad()

            preds = model(text)
            loss = criterion(preds, labels)
            acc = calc_accuracy(preds, labels, pad_index)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc
        
        train_loss = epoch_loss / len(train_data)
        train_acc = epoch_loss / len(train_data)
        val_acc = evaluate(model, val_data, pad_index)

        performance_dict['train_loss'].append(train_loss)
        performance_dict['train_acc'].append(train_acc)
        performance_dict['val_acc'].append(val_acc)

        # Save the model with the best validation accuracy.
        if val_acc > best_val_acc:
            torch.save(model, os.path.join(args.output_dir, "best_model.pt"))
            best_val_acc = val_acc

        # Save a model checkpoint every x epochs (if so specified).
        if args.save_every > 0 and e % args.save_every == 0:
            torch.save(model, os.path.join(args.output_dir, f"model_checkpoint_{e}.pt")) 

        print(f"=> Epoch {e}/{args.epochs}")
        print(f"\ttrain loss: {train_loss}")
        print(f"\ttrain acc: {train_acc}")
        print(f"\tval acc: {val_acc}")

    print("=== Finished training ===")

    # Evaluate best model on the test set.
    best_model = torch.load(os.path.join(args.output_dir, "best_model.pt"))
    test_acc = evaluate(best_model, test_data, pad_index)
    print(f"=> Final test accuracy: {test_acc}")
    performance_dict['test_acc'].append(test_acc)

    # Write the performance across epochs to file.
    with open(os.path.join(args.output_dir, "performance.json"), 'w') as fp:
        json.dump(performance_dict, fp)

    return model, performance_dict
    

if __name__ == '__main__':
    args = parse_arguments()

    # model = ... # TODO:
    # train(model, args)
