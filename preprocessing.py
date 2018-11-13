"""文本预处理类"""
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals import joblib
import numpy as np
from collections import Counter


class Vocabulary(object):
    def __init__(self, max_size=None, lower=True, unk_token=True, specials=('<pad>',)):
        self._max_size = max_size
        self._lower = lower
        self._unk = unk_token
        self._token2id = {token: i for i, token in enumerate(specials)}
        self._id2token = list(specials)
        self._token_count = Counter()

    def __len__(self):
        return len(self._token2id)

    def add_token(self, token):
        token = self.process_token(token)
        self._token_count.update([token])

    def add_documents(self, docs):
        for sent in docs:
            sent = map(self.process_token, sent)
            self._token_count.update(sent)

    def doc2id(self, doc):
        # doc = map(self.process_token, doc)
        return [self.token_to_id(token) for token in doc]

    def id2doc(self, ids):
        return [self.id_to_token(idx) for idx in ids]

    def build(self):
        token_freq = self._token_count.most_common(self._max_size)
        idx = len(self.vocab)
        for token, _ in token_freq:
            self._token2id[token] = idx
            self._id2token.append(token)
            idx += 1
        if self._unk:
            unk = '<unk>'
            self._token2id[unk] = idx
            self._id2token.append(unk)

    def process_token(self, token):
        if self._lower:
            token = token.lower()

        return token

    def token_to_id(self, token):
        # token = self.process_token(token)
        return self._token2id.get(token, len(self._token2id) - 1)

    def id_to_token(self, idx):
        return self._id2token[idx]

    @property
    def vocab(self):
        return self._token2id

    @property
    def reverse_vocab(self):
        return self._id2token


class IndexTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, max_words=80, max_chars=8, lower=True,
                 use_char=False, bi_char=False, seg_info=False,
                 radical=False, initial_vocab=None):
        self.max_words = max_words
        self.max_chars = max_chars
        self._use_char = use_char
        self._bi_char = bi_char
        self._seg_info = seg_info
        self._radical = radical
        self._word_vocab = Vocabulary(lower=lower)
        self._label_vocab = Vocabulary(lower=False, unk_token=False)
        self._char_vocab = Vocabulary(lower=lower)
        self._bichar_vocab = Vocabulary(lower=lower)
        self._seg_vocab = Vocabulary(lower=False, unk_token=False)
        self._radical_vocab = Vocabulary(lower=False)
        if initial_vocab:
            self._word_vocab.add_documents([initial_vocab])
            self._char_vocab.add_documents(initial_vocab)

    def fit(self, X, y):
        self._word_vocab.add_documents(X['main_input'])
        self._word_vocab.build()

        self._label_vocab.add_documents(y)
        self._label_vocab.build()

        if self._bi_char:
            self._bichar_vocab.add_documents(X['bichar'])
            self._bichar_vocab.build()

        if self._seg_info:
            self._seg_vocab.add_documents(X['seg'])
            self._seg_vocab.build()

        if self._radical:
            self._radical_vocab.add_documents(X['radical'])
            self._radical_vocab.build()

        if self._use_char:
            for doc in X['main_input']:
                self._char_vocab.add_documents(doc)
            self._char_vocab.build()

        return self

    def transform(self, X, y=None):
        word_ids = [self._word_vocab.doc2id(doc) for doc in X['main_input']]
        word_ids = pad_sequences(word_ids, maxlen=self.max_words, padding='post')
        features = [word_ids]
        if self._use_char:
            char_ids = [[self._char_vocab.doc2id(w) for w in doc] for doc in X['main_input']]
            char_ids = pad_nested_sequences(char_ids, self.max_words, self.max_chars)
            features.append(char_ids)
        if self._bi_char:
            bichar_ids = [self._bichar_vocab.doc2id(doc) for doc in X['bichar']]
            bichar_ids = pad_sequences(bichar_ids, maxlen=self.max_words, padding='post')
            features.append(bichar_ids)
        if self._seg_info:
            seg_ids = [self._seg_vocab.doc2id(doc) for doc in X['seg']]
            seg_ids = pad_sequences(seg_ids, maxlen=self.max_words, padding='post')
            features.append(seg_ids)
        if self._radical:
            radical_ids = [self._radical_vocab.doc2id(doc) for doc in X['radical']]
            radical_ids = pad_sequences(radical_ids, maxlen=self.max_words, padding='post')
            features.append(radical_ids)
        lengths = np.array([len(doc) for doc in X['main_input']], dtype='int32')
        features.append(lengths)

        if y is not None:
            y = [self._label_vocab.doc2id(doc) for doc in y]
            y = pad_sequences(y, maxlen=self.max_words, padding='post')
            y = to_categorical(y, self.label_size).astype(float)
            return features, y
        else:
            return features

    def fit_transform(self, X, y=None, **params):
        return self.fit(X, y).transform(X, y)

    def inverse_transform(self, y, lengths=None):
        y = np.argmax(y, -1)
        inverse_y = [self._label_vocab.id2doc(ids) for ids in y]
        if lengths is not None:
            inverse_y = [iy[:l] for iy, l in zip(inverse_y, lengths)]
        return inverse_y

    @property
    def word_vocab_size(self):
        return len(self._word_vocab)

    @property
    def char_vocab_size(self):
        return len(self._char_vocab)

    @property
    def label_size(self):
        return len(self._label_vocab)

    @property
    def bichar_vocab_size(self):
        return len(self._bichar_vocab)

    @property
    def seg_vocab_size(self):
        return len(self._seg_vocab)

    @property
    def radical_vocab_size(self):
        return len(self._radical_vocab)

    def save(self, file_path):
        joblib.dump(self, file_path)

    @classmethod
    def load(cls, file_path):
        p = joblib.load(file_path)
        return p


def pad_nested_sequences(sequences, max_sent_len, max_word_len, dtype='int32'):
    x = np.zeros((len(sequences), max_sent_len, max_word_len)).astype(dtype)
    for i, sent in enumerate(sequences):
        if len(sent) > max_sent_len:
            sent = sent[:max_sent_len]
        for j, word in enumerate(sent):
            if len(word) < max_word_len:
                x[i, j, :len(word)] = word
            else:
                x[i, j, :] = word[:max_word_len]
    return x
