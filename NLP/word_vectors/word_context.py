from utils import distinct_words
import numpy as np
from collections import Counter


def word_context_matrix(corpus):
    """ Compute word-context matrix.

        Params:
            corpus (list of list of strings): corpus of documents (contexts)
        Return:
            M (numpy matrix of shape (number of corpus words, number of contexts)): 
                word-context matrix of word-context co-occurrence count.
            word2Ind (dict): dictionary that maps word to index (i.e. row/column number) for matrix M.
    """
    # count word and context
    word_count = Counter([word for document in corpus for word in document])
    context_count = Counter(map(str, corpus))
    word2Ind = {w: i for i, w in enumerate(word_count.keys())}
    context2Ind = {c: i for i, c in enumerate(context_count.keys())}
    
    print(word2Ind)

    M = np.zeros((len(word_count), len(context_count)))
    # count co-occurrence
    for document in corpus:
        for word in document:
            M[word2Ind[word]][context2Ind[str(document)]] += 1
    print(M)
    # PMI: \text{PMI}(w, c)=\log\frac{#(w,c)\cdot|D|}{#(w)\cdot#(c)}
    for document in context_count.keys():
        for word in word_count.keys():
            M[word2Ind[word]][context2Ind[str(document)]] = np.log(M[word2Ind[word]][context2Ind[str(document)]]*len(context_count)
                                                                   /word_count[word]/context_count[str(document)])
    print(M)
    # PPMI: \text{PPMI}(w,c)=\max(\text{PMI}(w,c),0)
    for document in context_count.keys():
        for word in word_count.keys():
            M[word2Ind[word]][context2Ind[str(document)]] = max(M[word2Ind[word]][context2Ind[str(document)]], 0)
    print(M)
    
    return M, word2Ind


if __name__ == '__main__':
    test_corpus = ["START All that glitters isn't gold END".split(" "),
                   "START All that glitters isn't gold END".split(" "),
                   "START All's well that ends well END".split(" ")]
    M_test, word2Ind_test = word_context_matrix(test_corpus)