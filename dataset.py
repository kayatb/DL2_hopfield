from torchtext import datasets
from torchtext import vocab
from torchtext.data import Field, BucketIterator

# Methods useful for loading data in and out of the model. For now overriding the load_snli in datasets.py.
# TODO: Have to refactor


def get_glove_embeddings():
    return vocab.GloVe(name='840B', dim=300)


def load_snli(device=None, batch_size=64):
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


