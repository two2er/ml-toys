from utils import distinct_words
import numpy as np


def co_occurrence_matrix(corpus, window_size=4):
    """ Compute co-occurrence matrix for the given corpus and window_size (default of 4).
    
        Note: Each word in a document should be at the center of a window. Words near edges will have a smaller
              number of co-occurring words.
              
              For example, if we take the document "START All that glitters is not gold END" with window size of 4,
              "All" will co-occur with "START", "that", "glitters", "is", and "not".
    
        Params:
            corpus (list of list of strings): corpus of documents
            window_size (int): size of context window
        Return:
            M (numpy matrix of shape (number of corpus words, number of corpus words)): 
                Co-occurence matrix of word counts.
            word2Ind (dict): dictionary that maps word to index (i.e. row/column number) for matrix M.
    """
    words, num_words = distinct_words(corpus)
    word2Ind = {w: i for i, w in enumerate(words)}
    M = np.zeros((num_words, num_words))
    for document in corpus:
        for i, word in enumerate(document):
            for j in range(-window_size, window_size+1):
                if j == 0 or i + j < 0 or i + j >= len(document) or word == document[i+j]:
                    continue
                M[word2Ind[word]][word2Ind[document[i+j]]] += 1
                
    return M, word2Ind


if __name__ == '__main__':
    test_corpus = ["START All that glitters isn't gold END".split(" "), "START All's well that ends well END".split(" ")]
    M_test, word2Ind_test = co_occurrence_matrix(test_corpus, window_size=1)
    print(M_test)