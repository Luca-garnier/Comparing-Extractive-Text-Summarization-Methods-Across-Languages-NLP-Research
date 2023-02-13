from nltk import sent_tokenize

### LUHN ###
# adapted from https://iq.opengenus.org/luhns-heuristic-method-for-text-summarization/

def get_important_words(sentences, th=0.1, stopwords=None):
    sws = [',', '.', '/', ';', ':', '(', ')', '\\', "'", '"', '?', '!', '...']
    if stopwords:
        sws.extend(stopwords)

    counts = {}
    for sentence in sentences:
        for word in sentence.split(' '):
            word = word.lower().strip('.!?,()\n:;\'"-')
            if word in sws or len(word) == 0:
                continue
            elif word not in counts:
                counts[word] = 1
            else:
                counts[word] += 1

    words = [k for k in counts.keys()]
    words.sort(reverse=True, key=lambda k: counts[k])
    return set(words[:int(len(words) * th)])

def calculate_score(sentence, meaningful_words):
    words = sentence.split()
    imp_words, total_words, end, begin = [0]*4
    for word in words:
        w = word.strip('.!?,();:\'"-\n').lower()
        end += 1
        if w in meaningful_words:
            imp_words += 1
            begin = total_words
            end = 0
        total_words += 1
    unimportant = total_words - begin - end
    if(unimportant != 0):
        return float(imp_words**2) / float(unimportant)
    return 0.0

def luhn_summarize(document, reference, stopwords, th):
    sum_sentences = len(sent_tokenize(reference))
    sentences = sent_tokenize(document)
    if not th:
        meaningful_words = get_important_words(sentences, stopwords=stopwords)
    else:
        meaningful_words = get_important_words(sentences, stopwords=stopwords, th=th)
    scores = {}
    for sentence in sentences:
        scores[sentence] = calculate_score(sentence, meaningful_words)
    top_sentences = list(sentences)                           
    top_sentences.sort(key=lambda x: scores[x], reverse=True)      # sort by score
    try:
        sum_sentences = top_sentences[:sum_sentences]
    except:
        sum_sentences = top_sentences
    sum_sentences.sort(key=lambda x: sentences.index(x))           # sort by occurrence
    return ' '.join(sum_sentences)