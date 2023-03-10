from sentence_transformers import util, SentenceTransformer
import os

def load_file(input_file):
    pass

def get_similarity_sbert(s_pair, model=None):
    if s_pair[0] == None or s_pair[1] == None:
        return None

    if s_pair[0] == s_pair[1]:
        return 1.0
    if model is None:
        model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    embds = model.encode(s_pair)
    sim_score = util.cos_sim(embds[0], embds[1]).item()
    return sim_score


def get_similarity_keyval(dic1, dic2):
    key_set = set()
    tot_sim = 0
    for key, value1 in zip(dic1.keys(), dic1.values()):
        if key in dic2:
            value2 = dic2[key]
            if value2 == value1:
                tot_sim += 1

        key_set.add(key)

    for key2 in dic2:
        key_set.add(key2)

    return tot_sim / len(key_set)