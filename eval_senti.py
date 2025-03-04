import nltk
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn
from nltk import pos_tag, word_tokenize
import string

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('V'):
        return wn.VERB
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    else:
        return None

def calculate_sentiment(sentence):
    tokens = word_tokenize(sentence)
    tokens = [word for word in tokens if word not in string.punctuation]
    tagged = pos_tag(tokens)

    pos_score = 0.0
    neg_score = 0.0
    count = 0

    for word, tag in tagged:
        wn_tag = get_wordnet_pos(tag)
        if wn_tag is not None:
            synsets = list(swn.senti_synsets(word, wn_tag))
            if synsets:
                synset_pos_score = sum(synset.pos_score() for synset in synsets) / len(synsets)
                synset_neg_score = sum(synset.neg_score() for synset in synsets) / len(synsets)
                
                pos_score += synset_pos_score
                neg_score += synset_neg_score
                count += 1

    if count > 0:
        pos_score /= count
        neg_score /= count

    return pos_score, neg_score


def get_emotion(sentence, task_name):
    pos_score, neg_score = calculate_sentiment(sentence)
    if pos_score > neg_score:
        return 1 if 'positive' in task_name else 0
    elif pos_score < neg_score:
        return 0 if 'positive' in task_name else 1
    else:
        return 0