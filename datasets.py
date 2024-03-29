import torch
from torchtext.legacy import datasets, data
from torchtext import vocab
from torchtext.legacy.data import Field, BucketIterator


def select_dataset(dataset, batch_size, device):
    """ Return the train, val, test data iterator for the specified dataset. 
    Also return the vocabulary index of the padding token. """
    if dataset == "SST":
        # train, val, test = load_SST(batch_size, device)
        train, val, test, pad_index = load_SST(batch_size, device)
    elif dataset == "UDPOS":
        train, val, test, pad_index = load_UDPOS(batch_size, device)
    elif dataset == "SNLI":
        train, val, test, pad_index = load_SNLI(batch_size, device)
    else:
        raise ValueError(f"Unknown dataset given `{dataset}`. Implementation available for SST, UDPOS and SNLI.")
    
    return train, val, test, pad_index


def load_SST(batch_size, device, min_freq=2):
    """ Return train, val, test iterators for SST dataset containing the tokens and the
    binary sentiment label. 
    Also returns the index of the padding token in the vocabulary.
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

    # Obtain index of padding token.
    pad_index = label_field.vocab.stoi[label_field.pad_token]

    # One datapoint in the iterator contains `.text` and `.label`
    return train_iter, val_iter, test_iter, pad_index


def load_UDPOS(batch_size, device, min_freq=2):
    """ Return train, val, test iterators for the UDPOS dataset containing the tokens and their UD POS-tags. 
    Also returns the index of the padding token in the vocabulary.
    min_freq denotes the minimum frequency of a token to be contained in the vocabulary (otherwise <unk>). """
    text_field = data.Field(lower=True)  # Lower case all tokens
    # The tags do not contain unknown tokens, since we have a finite and 
    # completely known vocabulary.
    ud_tags_field = data.Field(unk_token=None)
    # PTB_TAGS = data.Field(unk_token=None)  # Dataset also contain PennTreebank POS-tags.

    # fields = (("text", text_field), ("udtags", ud_tags_field), ("ptbtags", PTB_TAGS))
    fields = (("text", text_field), ("label", ud_tags_field), (None, None))

    train_data, val_data, test_data = datasets.UDPOS.splits(fields)

    # Build a vocabulary based on the train data.
    text_field.build_vocab(train_data, min_freq=min_freq, vectors="glove.6B.100d", unk_init=torch.Tensor.normal_)
    ud_tags_field.build_vocab(train_data)

    train_iter, val_iter, test_iter = data.BucketIterator.splits(
        (train_data, val_data, test_data), 
        batch_size=batch_size, device=device
    )

    # Obtain index of padding token.
    pad_index = ud_tags_field.vocab.stoi[ud_tags_field.pad_token]

    # One datapoint in the iterator contains `.text` and `.label`
    return train_iter, val_iter, test_iter, pad_index


def load_SNLI(batch_size, device, min_freq=2):
    """ Return train, val, test iterators for the SNLI dataset containing premise tokens, hypothesis tokens and labels. 
    Also returns the index of the padding token in the vocabulary.
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

    # Obtain index of padding token.
    pad_index = label_field.vocab.stoi[label_field.pad_token]

    # One datapoint in the iterator contains `.premise`, `.hypothesis` and `.label`
    return train_iter, val_iter, test_iter, pad_index


def get_glove_embeddings():
    return vocab.GloVe(name='840B', dim=300)


def load_pl_snli(device=None, batch_size=64):
    glove_embeddings = get_glove_embeddings()

    data_root = '.data'

    text_field = Field(tokenize='spacy', lower=True, batch_first=True)
    label_field = Field(sequential=False, batch_first=True, is_target=True)
    train_dataset, dev_dataset, test_dataset = datasets.SNLI.splits(text_field=text_field,
                                                                    label_field=label_field,
                                                                    root=data_root)
    text_field.build_vocab(train_dataset, vectors=glove_embeddings)
    label_field.build_vocab(train_dataset, specials_first=False)
    vocabulary = text_field.vocab
    label_names = label_field.vocab
    train_iter, dev_iter, test_iter = BucketIterator.splits(
        (train_dataset, dev_dataset, test_dataset),
        batch_sizes=(batch_size, batch_size, batch_size),
        device=device)
    return vocabulary, label_names, train_iter, dev_iter, test_iter


