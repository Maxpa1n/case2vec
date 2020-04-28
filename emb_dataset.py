from torch.utils.data import Dataset
import torch
from sklearn.feature_extraction.text import CountVectorizer
import json


def get_lable(accusation, num_lable):
    num_sample = len(accusation)
    accusation_index = []
    accsation_set_index = {}
    lable_set = list(set(accusation))
    for i in range(len(lable_set)):
        accsation_set_index[lable_set[i]] = i
    for i in accusation:
        for k, v in accsation_set_index.items():
            if i == k:
                accusation_index.append(v)
    lable = torch.LongTensor(accusation_index)
    one_hot = torch.zeros(num_sample, num_lable).scatter_(1, lable.unsqueeze(1), 1)
    return one_hot


def token(src, vocabulary_dic, max_length, stop_words):
    src_lenght = []
    for i in src:
        j = i.split()
        senten = ['<CLS>']
        if len(j) < max_length:
            for w in j:
                if w in vocabulary_dic.keys():
                    senten.append(w)
                elif w in stop_words:
                    pass
                else:
                    senten.append('<UNK>')
            senten.append('<EOT>')
            while len(senten) != max_length:
                senten.append('<PAD>')
        else:
            for w in j[:max_length - 2]:
                if w in vocabulary_dic.keys():
                    senten.append(w)
                elif w in stop_words:
                    pass
                else:
                    senten.append('<UNK>')
            senten.append('<EOT>')
            while len(senten) != max_length:
                senten.append('<PAD>')
        src_lenght.append(senten)
    return src_lenght


def token_to_id(all_sentences, vocabulary_dic):
    all_sentences_id = []
    for senten in all_sentences:
        senten_id = []
        for word in senten:
            senten_id.append(vocabulary_dic[word])
        all_sentences_id.append(senten_id)
    sentences_tensor = torch.tensor(all_sentences_id, dtype=torch.long)
    mask = sentences_tensor == torch.zeros_like(sentences_tensor)

    return sentences_tensor, mask


class Mydataset(Dataset):
    def __init__(self, src, accusation, max_length, vocabulary_dic, stop_word=None, vocabulary=None):
        self.vector = CountVectorizer(vocabulary=vocabulary, stop_words=None)
        self.tokens = self.vector.fit_transform(src)
        self.x_bow = torch.from_numpy(self.tokens.toarray()).float()
        self.x_seq, self.mask = token_to_id(token(src, vocabulary_dic, max_length, stop_word), vocabulary_dic)
        self.accusation = accusation

    def __getitem__(self, index):
        return self.x_bow[index], self.x_seq[index], self.mask[index], self.accusation[index]

    def __len__(self):
        return len(self.x_bow)


