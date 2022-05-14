from torchtext.legacy import data, datasets
import torch


def select_dataset(dataset, batch_size, device):
    """ Return the train, val, test data iterator for the specified dataset. """
    if dataset == "SST":
        # train, val, test = load_SST(batch_size, device)
        train, val, test = load_SST(batch_size, device)
    elif dataset == "UDPOS":
        train, val, test = load_UDPOS(batch_size, device)
    elif dataset == "SNLI":
        train, val, test = load_SNLI(batch_size, device)
    else:
        raise ValueError(f"Unknown dataset given `{dataset}`. Implementation available for SST, UDPOS and SNLI.")
    
    return train, val, test


def load_SST(batch_size, device, min_freq=2):
    """ Return train, val, test iterators for SST dataset containing the tokens and the
    binary sentiment label. 
    min_freq denotes the minimum frequency of a token to be contained in the vocabulary (otherwise <unk>)."""
    text_field = data.Field(lower=True)
    label_field = data.Field(dtype=torch.float)

    train_data, val_data, test_data = datasets.SST.splits(text_field, label_field)  # NOTE: `train_subtrees=True` to use all subtrees in training set.

    # Build a vocabulary based on the train data.
    text_field.build_vocab(train_data, min_freq=min_freq, vectors="glove.6B.100d", unk_init=torch.Tensor.normal_)
    label_field.build_vocab(train_data)  # {'<unk>': 0, '<pad>': 1, 'positive': 2, 'negative': 3, 'neutral': 4}

    train_iter, val_iter, test_iter = data.BucketIterator.splits(
        (train_data, val_data, test_data),
        batch_size=batch_size, device=device
    )
    # One datapoint in the iterator contains `.text` and `.label`
    return train_iter, val_iter, test_iter


def load_UDPOS(batch_size, device, min_freq=2):
    """ Return train, val, test iterators for the UDPOS dataset containing the tokens and their UD POS-tags. 
    min_freq denotes the minimum frequency of a token to be contained in the vocabulary (otherwise <unk>). """
    text_field = data.Field(lower=True)  # Lower case all tokens
    # The tags do not contain unknown tokens, since we have a finite and 
    # completely known vocabulary.
    ud_tags_field = data.Field(unk_token=None)
    # PTB_TAGS = data.Field(unk_token=None)  # Dataset also contain PennTreebank POS-tags.

    # fields = (("text", text_field), ("udtags", ud_tags_field), ("ptbtags", PTB_TAGS))
    fields = (("text", text_field), ("udtags", ud_tags_field), (None, None))

    train_data, val_data, test_data = datasets.UDPOS.splits(fields)

    # Build a vocabulary based on the train data.
    text_field.build_vocab(train_data, min_freq=min_freq, vectors="glove.6B.100d", unk_init=torch.Tensor.normal_)
    ud_tags_field.build_vocab(train_data)

    train_iter, val_iter, test_iter = data.BucketIterator.splits(
        (train_data, val_data, test_data), 
        batch_size=batch_size, device=device
    )
    # One datapoint in the iterator contains `.text` and `.udpos`
    return train_iter, val_iter, test_iter


def load_SNLI(batch_size, device, min_freq=2):
    """ Return train, val, test iterators for the SNLI dataset containing premise tokens, hypothesis tokens and labels. 
    min_freq denotes the minimum frequency of a token to be contained in the vocabulary (otherwise <unk>)."""
    text_field = data.Field(lower=True)
    label_field = data.Field(dtype=torch.float)

    train_data, val_data, test_data = datasets.SNLI.splits(text_field, label_field)

    text_field.build_vocab(train_data, min_freq=min_freq, vectors="glove.6B.100d", unk_init=torch.Tensor.normal_)
    label_field.build_vocab(train_data)  # {'<unk>': 0, '<pad>': 1, 'entailment': 2, 'contradiction': 3, 'neutral': 4, '-': 5}

    train_iter, val_iter, test_iter = data.BucketIterator.splits(
        (train_data, val_data, test_data),
        batch_size=batch_size, device=device
    )
    # One datapoint in the iterator contains `.premise`, `.hypothesis` and `.label`
    return train_iter, val_iter, test_iter
