import re
import nltk
nltk.download('brown')
from nltk.corpus import brown
from collections import Counter
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

corpus = brown.sents(categories='news')[:1500]
# preprocessing
def clean(text):
    # remove characters except for alphabets
    text = ' '.join(text)
    text = text.lower()
    text.replace('\n', ' ')
    text = re.sub('[^a-z]+', ' ', text).strip()
    return text.split()
corpus = [clean(text) for text in corpus]
print('totally {} texts in the corpus'.format(len(corpus)))

# subsampling
# https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf
# the discard probability is modified
def subsampling(corpus):
    # count the frequency of each word
    count = Counter([word for text in corpus for word in text])
    num_words = sum(count.values())
    count = {w: c / num_words for w, c in count.items()}
    result = []
    for text in corpus:
        result_text = []
        for word in text:
            # the keep probability
            prop = 1e-3 / count[word] * (np.sqrt(count[word] * 1e3) + 1)
            if prop > np.random.random():
                result_text.append(word)
        if result_text:
            result.append(result_text)
    return result
corpus = subsampling(corpus)

vocabulary = set([word for text in corpus for word in text])
word2index = {w:i for i, w in enumerate(vocabulary)}
index2word = dict(enumerate(vocabulary))

# negative sampling
def negative_sampling(corpus, sample_size=10):
    # count the frequency of each word
    count = Counter([word for text in corpus for word in text])
    # raise to 3/4 power
    count = {w: i**(0.75) for w, i in count.items()}
    # normalize
    normal_item = sum(count.values())
    count = {w: i/normal_item for w, i in count.items()}
    # sampling probability of each word
    prob = [0 for _ in range(len(vocabulary))]
    for word in vocabulary:
        prob[word2index[word]] = count[word]
    while True:
        # generate sample_size negative words
        yield np.random.choice(len(vocabulary), sample_size, p=prob)

# center word - context words pairs
negative_sampler = negative_sampling(corpus, sample_size=5)
window_size = 5
def generate_center_context_pairs(corpus):
    center_context = []
    for text in corpus:
        for i, word in enumerate(text):
            l = []
            for j in range(-window_size, window_size+1):
                idx = i + j
                if j == 0 or idx < 0 or idx >= len(text):
                    continue
                center_context.append((word2index[word], word2index[text[idx]], next(negative_sampler)))
    return center_context
center_context = generate_center_context_pairs(corpus)

def get_batch(center_context, batch_size=100):
    np.random.shuffle(center_context)
    batches = []
    i = 0
    while True:
        center_words, context_words, negative_samples = [], [], []
        if (i + 1) * batch_size >= len(center_context):
            for j in range(i*batch_size, len(center_context)):
                center_words.append(center_context[j][0])
                context_words.append(center_context[j][1])
                negative_samples.append(center_context[j][2])
            batches.append((center_words, context_words, negative_samples))
            return batches
        else:
            for j in range(i*batch_size, (i+1)*batch_size):
                center_words.append(center_context[j][0])
                context_words.append(center_context[j][1])
                negative_samples.append(center_context[j][2])
            batches.append((center_words, context_words, negative_samples))
        i += 1

class Skip_Gram(nn.Module):
    def __init__(self, vocab_size, embedding_dimen):
        super(Skip_Gram, self).__init__()
        # word representation for center words
        self.embedding_center = nn.Embedding(vocab_size, embedding_dimen)
        # word representation for context words
        self.embedding_context = nn.Embedding(vocab_size, embedding_dimen)
    
    def forward(self, center_word, context_word, negative_samples):
        # (batch_size, embed_dim)
        v_c = self.embedding_center(center_word)
        u_o = self.embedding_context(context_word)
        loss = torch.sum(torch.mul(v_c, u_o), dim=1)
        loss = -torch.sum(F.logsigmoid(loss))
        # loss of negative samples
        # (batch_size, negative_sample_size, embed_dim)
        ns = self.embedding_context(negative_samples)
        loss_ns = F.logsigmoid(-torch.bmm(ns, torch.unsqueeze(v_c, dim=2)))
        loss_ns = torch.sum(loss_ns)
        return loss - loss_ns

vocab_size = len(vocabulary)
embed_dim = 100
epoches = 20

loss_function = nn.CrossEntropyLoss()
model = Skip_Gram(vocab_size, embed_dim)
optimizer = optim.Adam(model.parameters())

for e in range(epoches):
    losses = []
    for center_words, context_words, negative_samples in get_batch(center_context, batch_size=100):
        model.zero_grad()
        center_words = torch.LongTensor(center_words)
        context_words = torch.LongTensor(context_words)
        negative_samples = torch.LongTensor(negative_samples)
        loss = model(center_words, context_words, negative_samples)
        loss.backward()
        optimizer.step()
        losses.append(loss.data)
    
    # training log
    print('epoch: {}, loss: {}'.format(e, np.mean(losses)))

# test
def get_closest_word(word, topn=5):
    word_distance = []
    emb = model.embedding_center
    pdist = nn.PairwiseDistance()
    i = word2index[word]
    lookup_tensor_i = torch.tensor([i], dtype=torch.long)
    v_i = emb(lookup_tensor_i)
    for j in range(len(vocabulary)):
        if j != i:
            lookup_tensor_j = torch.tensor([j], dtype=torch.long)
            v_j = emb(lookup_tensor_j)
            word_distance.append((index2word[j], float(pdist(v_i, v_j))))
    word_distance.sort(key=lambda x: x[1])
    return word_distance[:topn]

print(get_closest_word('war'))
