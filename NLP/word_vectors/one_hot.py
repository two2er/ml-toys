from utils import distinct_words


def one_hot(corpus):
    """ one-hot vector representation
    
        Params:
            corpus: list of strings (strings are lists of words)
        Return:
            vectors: one-hot embeddings of each word in the corpus. the dictionary is in lexicographical order
    """   
    corpus_words, num_corpus_words = distinct_words(corpus)
    embeddings = {}
    for i, word in enumerate(corpus_words):
        embeddings[word] = [0 for _ in range(len(corpus_words))]
        embeddings[word][i] = 1
    print('embeddings:', embeddings)
    vectors = [[None for _ in range(len(corpus[i]))] for i in range(len(corpus))]
    for i in range(len(corpus)):
        # for each sentence
        for j in range(len(corpus[i])):
            # map word to one-hot vector
            vectors[i][j] = embeddings[corpus[i][j]]

    return vectors


if __name__ == '__main__':
    test_corpus = ["START All that glitters isn't gold END".split(" "),
                   "START All's well that ends well END".split(" ")]
    one_hot_vectors = one_hot(test_corpus)
    print(one_hot_vectors)