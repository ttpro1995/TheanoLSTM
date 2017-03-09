import pytreebank
import nltk
import itertools
from numpy import array
import numpy as np
SENTENCE_START_TOKEN = 'eos'
UNKNOWN_TOKEN = 'unk'


def word2index(sentences, vocabulary_size):
    tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]
    # Count the word frequencies
    word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    #print "Found %d unique words tokens." % len(word_freq.items())
    # Get the most common words and build index_to_word and word_to_index vectors
    vocab = word_freq.most_common(vocabulary_size-2)
    index_to_word = [x[0] for x in vocab]
    index_to_word.append(UNKNOWN_TOKEN)
    index_to_word.append(SENTENCE_START_TOKEN)
    word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])
    return (word_to_index, index_to_word)

def word2vec(sent, word2index_dict):
    '''
    Word2vec of a sentence
    :param sent: input sentence
    :param word2vec_dict: dict of word2vec
    :param maxlen: max len of sentence in dataset
    :return: vector of sentence (list vector of words)
    '''
    sent = "%s %s" % (SENTENCE_START_TOKEN, sent)
    words_in_sent = [x for x in nltk.word_tokenize(sent)]
    i = len(words_in_sent)
    array_sent=[0]*i
    sample_weight = [0]*i
    for j in range(i):
        if words_in_sent[j].lower() not in word2index_dict.keys():
            words_in_sent[j] = UNKNOWN_TOKEN
        array_sent[j] = (word2index_dict[words_in_sent[j].lower()])
        sample_weight[j] = 1
    array_sent = np.asarray(array_sent)
    return ((array_sent),array(sample_weight))

def demo_tree():
    small_trees = pytreebank.import_tree_corpus('./trees/dev.txt')
    small_trees = small_trees[:100]
    label = []
    sentences = []


    tree = small_trees[6]
    for l, sent in tree.to_labeled_lines():
        label.append(l)
        sentences.append(sent)
        print(l, sent)

    print('breakpoint')

def preprocess_full(vocabulary_size):
    trees = pytreebank.load_sst('trees')
    trees_train = trees["train"]
    trees_dev = trees["dev"]
    trees_test = trees["test"]



def preprocess(vocabulary_size):
    # trees = pytreebank.load_sst('trees')
    # trees_train = trees["train"]
    # trees_dev = trees["dev"]
    # trees_test = trees["test"]


    small_trees = pytreebank.import_tree_corpus('./trees/train.txt')
    label = []
    sentences = []

    for tree in small_trees:
        l, sent = tree.to_labeled_lines()[0]
        label.append(l)
        sentences.append(sent)

    #for tree in small_trees:
    #    for l, sent in tree.to_labeled_lines():
    #        label.append(l)
    #        sentences.append(sent)

    label = np.asarray(label)

    word_to_index, index_to_word = word2index(sentences,vocabulary_size)
    train_x = []
    for sent in sentences:
        x, _ = word2vec(sent,word_to_index)
        train_x.append(x)

    return (train_x, label)

if __name__ == "__main__":
    preprocess(4000)