from torchtext.legacy import data, datasets
import torch


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

    train_data, val_data, test_data = datasets.SST.splits(text_field,
                                                          label_field)  # NOTE: `train_subtrees=True` to use all subtrees in training set.

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


def recreate_data(train_data, val_data, test_data, text_field):
    for index, data in enumerate(train_data):
        setattr(train_data[index], 'text', data.premise + ["<sep>"] + data.hypothesis)

    for index, data in enumerate(val_data):
        setattr(val_data[index], 'text', data.premise + ["<sep>"] + data.hypothesis)

    for index, data in enumerate(test_data):
        setattr(test_data[index], 'text', data.premise + ["<sep>"] + data.hypothesis)
    train_data.fields['text'] = text_field
    val_data.fields['text'] = text_field
    test_data.fields['text'] = text_field
    return train_data, val_data, test_data


def load_SNLI(batch_size, device, min_freq=2):
    """ Return train, val, test iterators for the SNLI dataset containing premise tokens, hypothesis tokens and labels. 
    Also returns the index of the padding token in the vocabulary.
    min_freq denotes the minimum frequency of a token to be contained in the vocabulary (otherwise <unk>)."""
    text_field = data.Field(lower=True)
    label_field = data.Field(sequential=False, batch_first=True, is_target=True)
    train_data, val_data, test_data = datasets.SNLI.splits(text_field, label_field)
    print("SEPARATOR TOKEN")
    # No override possible with the SNLI splits(or atleast i couldn't find any), brute forcing it to create a
    # .text field
    train_data, val_data, test_data = recreate_data(train_data, val_data, test_data, text_field)

    text_field.build_vocab(train_data, min_freq=min_freq,
                           vectors="glove.840B.300d", unk_init=torch.Tensor.normal_)
    label_field.build_vocab(train_data)
    for val in train_data:
        print(dir(val))
        break
    train_iter, val_iter, test_iter = data.BucketIterator.splits(
        (train_data, val_data, test_data),
        batch_size=batch_size, device=device
    )
    for val in train_iter:
        print(dir(val))
        break
    # Recreate, train, val, test iters
    # Obtain index of padding token.
    pad_index = label_field.vocab.stoi[label_field.pad_token]
    # One datapoint in the iterator contains `.premise`, `.hypothesis` and `.label`
    return train_iter, val_iter, test_iter, pad_index
