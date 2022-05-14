from torchtext.legacy import data, datasets
import torch


def select_dataset(dataset, batch_size, device):
    """ Return the train, val, test data iterator for the specified dataset. """
    if dataset == "SST":
        train, val, test = load_SST(batch_size, device)
    if dataset == "UDPOS":
        train, val, test = load_UDPOS(batch_size, device)
    elif dataset == "SNLI":
        train, val, test = load_SNLI(batch_size, device)
    else:
        raise ValueError(f"Unknown dataset given `{dataset}`. Implementation available for SST, UDPOS and SNLI.")
    
    return train, val, test


def load_SST(batch_size, device):
    """ Return train, val, test iterators for SST dataset containing the tokens and the
    binary sentiment label. """
    raise NotImplementedError


def load_UDPOS(batch_size, device, min_freq=2):
    """ Return train, val, test iterators for the UDPOS dataset containing the
    tokens and their UD POS-tags. 
    min_freq denotes the minimum frequency of a token to be contained in the vocabulary (otherwise <unk>). """
    TEXT = data.Field(lower=True)  # Lower case all tokens
    # The tags do not contain unknown tokens, since we have a finite and 
    # completely known vocabulary.
    UD_TAGS = data.Field(unk_token=None)
    # PTB_TAGS = data.Field(unk_token=None)  # Dataset also contain PennTreebank POS-tags.

    # fields = (("text", TEXT), ("udtags", UD_TAGS), ("ptbtags", PTB_TAGS))
    fields = (("text", TEXT), ("udtags", UD_TAGS), (None, None))

    train_data, val_data, test_data = datasets.UDPOS.splits(fields)

    TEXT.build_vocab(train_data, min_freq=min_freq, vectors="glove.6B.100d", unk_init=torch.Tensor.normal_)
    UD_TAGS.build_vocab(train_data)

    train_iterator, val_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, val_data, test_data), 
        batch_size=batch_size, device=device)

    return train_iterator, val_iterator, test_iterator


def load_SNLI(batch_size, device):
    """ Return train, val, test iterators for the SNLI dataset. """
    raise NotImplementedError
