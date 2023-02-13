from nltk.tokenize import sent_tokenize
import re
import numpy as np
import networkx as nx
from scipy import spatial
from gensim.models import Word2Vec

# Note: this implementation uses unigrams in common instead of cosine distance
# adapted from https://medium.com/data-science-in-your-pocket/text-summarization-using-textrank-in-nlp-4bce52c5b390

def get_unigrams_in_common(s1, s2):
    common = 0
    for w1 in s1:
        if w1 in s2:
            common += 1
    return common

def textrank_summarize(document, reference, stopwords, lower=True):
    sum_sentences = len(sent_tokenize(reference))
    sentences = sent_tokenize(document)
    if lower:
        sentences_clean = [re.sub(r'[^\w\s]', '', s.lower()) for s in sentences]
    else:
        sentences_clean = [re.sub(r'[^\w\s]', '', s) for s in sentences]

    if stopwords:
        sentence_tokens = [[w for w in s.split(' ') if w not in stopwords] for s in sentences_clean]
    else:
        sentence_tokens = [[w for w in s.split(' ')] for s in sentences_clean]
    
    similarity_matrix = [
        [get_unigrams_in_common(s1, s2) for s2 in sentence_tokens] for s1 in sentence_tokens
    ]
    similarity_matrix = np.matrix(similarity_matrix)

    nx_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(nx_graph)

    top_sentences = {sentence:scores[index] for index,sentence in enumerate(sentences)}
    top = sorted(top_sentences.items(), key=lambda x: x[1], reverse=True)[:sum_sentences]
    top.sort(key=lambda x: sentences.index(x[0]))
    return ' '.join([t[0] for t in top])

def w2v_textrank_summarize(document, reference, stopwords, lower=True):
    sum_sentences = len(sent_tokenize(reference))
    sentences = sent_tokenize(document)
    if lower:
        sentences_clean = [re.sub(r'[^\w\s]', '', s.lower()) for s in sentences]
    else:
        sentences_clean = [re.sub(r'[^\w\s]', '', s) for s in sentences]
        
    if stopwords:
        sentence_tokens = [[w for w in s.split(' ') if w not in stopwords] for s in sentences_clean]
    else:
        sentence_tokens = [[w for w in s.split(' ')] for s in sentences_clean]
    
    w2v = Word2Vec(sentence_tokens, vector_size=1, min_count=1)
    sentence_embeddings = [[w2v.wv[word][0] for word in words] for words in sentence_tokens]
    max_len = max([len(tokens) for tokens in sentence_tokens])
    sentence_embeddings = [np.pad(embedding,(0,max_len-len(embedding)),'constant') for embedding in sentence_embeddings]

    similarity_matrix = np.zeros([len(sentence_tokens), len(sentence_tokens)])
    for i,row_embedding in enumerate(sentence_embeddings):
        for j,column_embedding in enumerate(sentence_embeddings):
            similarity_matrix[i][j]=1-spatial.distance.cosine(row_embedding,column_embedding)

    nx_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank_numpy(nx_graph)

    top_sentences = {sentence:scores[index] for index,sentence in enumerate(sentences)}
    top = sorted(top_sentences.items(), key=lambda x: x[1], reverse=True)[:sum_sentences]
    top.sort(key=lambda x: sentences.index(x[0]))
    return ' '.join([t[0] for t in top])