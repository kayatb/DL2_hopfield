# This file contains the inference part for this project.
from __future__ import absolute_import, division, unicode_literals

import sys
import io
import logging

# import own files
from main import *

# Set PATHs
PATH_TO_SENTEVAL = 'SentEval'
PATH_TO_DATA = 'SentEval/data'
PATH_TO_VEC = 'SentEval/pretrained/glove.840B.300d.txt'

# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
from SentEval.senteval import engine

# added global variables for the model and embedding dimension
MODEL = None
EMBED_DIM = 300


# Create dictionary
def create_dictionary(sentences, threshold=0):
    words = {}
    for s in sentences:
        for word in s:
            words[word] = words.get(word, 0) + 1

    if threshold > 0:
        newwords = {}
        for word in words:
            if words[word] >= threshold:
                newwords[word] = words[word]
        words = newwords
    words['<s>'] = 1e9 + 4
    words['</s>'] = 1e9 + 3
    words['<p>'] = 1e9 + 2

    sorted_words = sorted(words.items(), key=lambda x: -x[1])  # inverse sort
    id2word = []
    word2id = {}
    for i, (w, _) in enumerate(sorted_words):
        id2word.append(w)
        word2id[w] = i

    return id2word, word2id


# Get word vectors from vocabulary (glove, word2vec, fasttext ..)
def get_wordvec(path_to_vec, word2id):
    word_vec = {}

    with io.open(path_to_vec, 'r', encoding='utf-8') as f:
        # if word2vec or fasttext file : skip first line "next(f)"
        for line in f:
            word, vec = line.split(' ', 1)
            if word in word2id:
                word_vec[word] = np.fromstring(vec, sep=' ')

    logging.info('Found {0} words with word vectors, out of \
        {1} words'.format(len(word_vec), len(word2id)))
    return word_vec


def prepare(params, samples):
    _, params.word2id = create_dictionary(samples)
    params.word_vec = get_wordvec(PATH_TO_VEC, params.word2id)
    params.wvec_dim = EMBED_DIM
    return


def batcher(params, batch):
    batch = [sent if sent != [] else ['.'] for sent in batch]
    embeddings = []

    for sent in batch:
        sentvec = []
        for word in sent:
            if word in params.word_vec:
                sentvec.append(torch.tensor(params.word_vec[word]))
        if not sentvec:
            vec = torch.zeros((1, 300))
            sentvec.append(vec)
        sentvec = torch.stack(sentvec, dim=0)
        embeddings.append(sentvec)

    # pad into tensor
    sentence_lengths = torch.tensor([x.shape[0] for x in embeddings])
    embeddings = torch.nn.utils.rnn.pad_sequence(embeddings, padding_value=0.0, batch_first=True)

    # pass through the model
    embeddings = MODEL.encoder(embeddings.float(), sentence_lengths)

    # cast back to numpy
    embeddings = embeddings.cpu().detach().numpy()

    # return the embeddings
    return embeddings


params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10,
                   'classifier': {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                                  'tenacity': 5, 'epoch_size': 4}}

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

# For command line activation
if __name__ == "__main__":
    # added parser for selecting the model
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', default='Hopfield', type=str,
                        help='What model to use. Default is AWE',
                        choices=['Hopfield', 'BERT'])
    args = parser.parse_args()

    if args.model == 'Hopfield':
        MODEL = TransformerModel.load_from_checkpoint('trained_models/Hopfield/epoch=10.ckpt')
        EMBED_DIM = 300
    else:
        # Enter the path for the model checkpoint here
        MODEL = TransformerModel.load_from_checkpoint('')
        EMBED_DIM = 300

    # run the senteval
    se = engine.SE(params_senteval, batcher, prepare)
    transfer_tasks = ['CR', 'MR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC', 'SICKEntailment']
    results = se.eval(transfer_tasks)

    # save the results
    torch.save(results, args.model + "SentenceEvalRes.pt")

    # print the results
    print(results)
