"""预测类"""
import numpy as np
from pathlib import Path
import re
from seqeval.metrics.sequence_labeling import get_entities
from chunk_segmentor.utils import flatten_gen, tag_by_dict, read_line, compare_idx
from collections import Counter


def check_in(check_list, filter_list):
    combine = set(check_list) & filter_list
    if len(combine) > 0:
        return True
    else:
        return False


def split_cn_en(word):
    new_word = []
    for char in word:
        if re.search(r'[\u4e00-\u9fa5]', char):
            new_word.append((char, 'cn'))
        elif re.search(r'[a-z]', char):
            new_word.append((char, 'en'))
        elif re.search(r'[0-9]', char):
            new_word.append((char, 'num'))
        else:
            new_word.append((char, 'other'))

    res, tmp = [], []
    l_w = len(new_word)
    for ix, item in enumerate(new_word):
        if item[1] in ['en', 'num']:
            if ix < l_w - 1:
                if new_word[ix+1][1] == item[1]:
                    tmp.append(item[0])
                else:
                    tmp.append(item[0])
                    res.append(''.join(tmp))
                    tmp = []
            else:
                tmp.append(item[0])
                res.append(''.join(tmp))
        else:
            res.append(item[0])
    return res


def split_word(word):
    word, pos = word.split('_')
    if len(word) == 1 or word in ['<s>', '<ss>', '<lan>'] or not re.search(r'[^a-z0-9]+', word):
        yield [word, word, pos, 'S']
    else:
        char_list = split_cn_en(word)
        l_c = len(char_list)
        word_list = [word] * l_c
        pos_list = [pos] * l_c
        seg_list = ['M'] * l_c
        seg_list[0] = 'B'
        seg_list[-1] = 'E'
        for i in range(l_c):
            yield [char_list[i], word_list[i], pos_list[i], seg_list[i]]


def word2char(word_list):
    return list(flatten_gen([list(split_word(word)) for word in word_list]))


def chunk_list(word_list, max_length):
    l_w = len(word_list)
    if l_w > max_length:
        for i in range(0, len(word_list), max_length):
            yield word_list[0+i: max_length+i]
    else:
        yield word_list


def split_long_sent(word_list, max_length):
    if len(word_list) <= max_length:
        return [word_list]

    num_split = int(len(word_list) / max_length)
    start = 0
    end = max_length
    split_text = []
    possible_split = [',', '.', '<s>', '、', '/']
    possible_idx = [idx for idx, item in enumerate(word_list) if item[0] in possible_split]
    if len(possible_idx) > 0:
        for _ in range(num_split):
            sub_possible_idx = [idx for idx in possible_idx if idx > start and idx <= end]
            if sub_possible_idx != []:
                end = max(sub_possible_idx, key=lambda x: x - end)
                split_text.append(word_list[start:end+1])
                start = end + 1
            end += max_length
        split_text.append(word_list[start:])
    else:
        split_text.append(word_list)
    new_list = list(flatten_gen([list(chunk_list(item, max_length)) for item in split_text]))
    return new_list


def get_bi_char(char_list):
    try:
        res = [char + char_list[idx+1] if idx < len(char_list)-1 else char + '<end>' for idx, char in enumerate(char_list)]
        return res
    except Exception as e:
        print(e)
        print(char_list)


def get_radical(d, char_list):
    return [d[char] if char in d else '<unk>' for char in char_list]


class Tagger(object):
    def __init__(self, model, preprocessor, char_input=False, pos=True,
                 batch_size=512, radical_file='', tree=None, qualifier=None):
        self.wrong = []
        self.model = model
        self.p = preprocessor
        self.pos = pos
        self.tree = tree
        self.qualifier = qualifier
        if self.p.bichar_vocab_size > 2:
            self.bichar = True
        else:
            self.bichar = False
        if self.p.radical_vocab_size > 2:
            self.radical = True
        else:
            self.radical = False
        if self.p.seg_vocab_size > 2:
            self.seg_info = True
        else:
            self.seg_info = False

        self.char_input = char_input
        self.char_tokenizer = word2char
        self.word_tokenizer = str.split
        self.batch_size = batch_size
        if self.seg_info or self.bichar or self.radical:
            assert self.char_input is True
        if self.radical:
            self.radical_dict = {item.split('\t')[0]: item.split('\t')[1] for item in read_line(radical_file)}
        dict_path = Path(__file__).parent / 'data' / 'dict'
        self.stopwords = set(read_line(dict_path / 'stopwords.txt'))
        self.stopwords_first = set(read_line(dict_path / 'stopwords_first_word.txt'))
        self.stopwords_last = set(read_line(dict_path / 'stopwords_last_word.txt'))
        self.pos_filter = set(read_line(dict_path / 'pos_filter.txt'))
        self.pos_filter_first = set(read_line(dict_path / 'pos_filter_first.txt'))
        self.MAIN_INPUT_IDX = 0
        self.POS_IDX = 1
        self.SEG_IDX = 2
        self.BICHAR_IDX = 3
        self.RADICAL_IDX = 4
        self.WORD_IDX = 5

    @property
    def get_tagger_info(self):
        params = {'char_input': self.char_input,
                  'pos': self.pos,
                  'batch_size': self.batch_size,
                  'seg_info': self.seg_info,
                  'bi_char': self.bichar,
                  'radical': self.radical}
        return params

    def data_generator(self, batch_input):
        input_data = {}
        batch_input = [self.preprocess_data(item) for item in batch_input]  # 73%(95%)
        text_pos_idx = [(idx, each, item['pos'][i]) for idx, item in enumerate(batch_input) for i, each in enumerate(item['main_input'])]
        sent_idx, sub_text, sub_pos = zip(*text_pos_idx)

        try:
            input_data['main_input'] = sub_text
            input_data['pos'] = sub_pos
            if self.seg_info:
                input_data['seg'] = [each for item in batch_input for each in item['seg']]
            if self.bichar:
                bichar = [each for item in batch_input for each in item['bichar']]
                input_data['bichar'] = bichar
            if self.radical:
                radical = [each for item in batch_input for each in item['radical']]
                input_data['radical'] = radical
            if self.char_input:
                word = [each for item in batch_input for each in item['word']]
                input_data['word'] = word
            cc = list(Counter(sent_idx).values())
            end_idx = [sum(cc[:i]) for i in range(len(cc)+1)]
            return end_idx, input_data
        except Exception as e:
            print(e)
            length = [len(each) for idx, item in enumerate(batch_input) for each in item['main_input']]
            print(len(batch_input), length, sub_text)
            self.wrong.append(len(batch_input), length, sub_text)

    def preprocess_data(self, seg_res):
        assert isinstance(seg_res, list)
        assert len(seg_res) > 0
        input_data = {}
        if self.char_input:
            string_c = self.char_tokenizer(seg_res)
            string_c = list(flatten_gen([split_long_sent(item, self.p.max_words) for item in string_c]))
            try:
                input_data['main_input'] = [item[0] for item in string_c]
                input_data['word'] = [item[1] for item in string_c]
                input_data['pos'] = [item[2] for item in string_c]
                if self.seg_info:
                    input_data['seg'] = [item[3] for item in string_c]
            except Exception as e:
                print('char tokenizer error: ', e)
                print(string_c)
            if self.bichar:
                input_data['bichar'] = [get_bi_char(item) for item in input_data['main_input']]
            if self.radical:
                input_data['radical'] = [get_radical(self.radical_dict, item) for item in input_data['main_input']]
        else:
            string_w = split_long_sent([item.split('_') for item in seg_res], self.p.max_words)
            input_data['main_input'] = [[each[0] for each in item] for item in string_w]
            input_data['pos'] = [[each[1] for each in item] for item in string_w]
        return input_data

    def predict_proba_batch(self, batch_data):
        split_text = batch_data['main_input']
        pos = batch_data['pos']
        if self.char_input:
            segs = batch_data['seg']
            words = batch_data['word']
        else:
            segs = []
            words = []
        X = self.p.transform(batch_data)
        Y = self.model.model.predict_on_batch(X)
        return split_text, pos, Y, segs, words

    def _get_prob(self, pred):
        prob = np.max(pred, -1)
        return prob

    def _get_tags(self, pred):
        tags = self.p.inverse_transform([pred])
        tags = tags[0]
        return tags

    def _build_response(self, split_text, tags, poss, segs=[], words=[]):
        if self.char_input:
            res = {
                'words': split_text,
                'pos': poss,
                'char_pos': poss,
                'char_word': words,
                'seg': segs,
                'entities': []
            }
        else:
            res = {
                'words': split_text,
                'pos': poss,
                'entities': []
            }
        chunks = get_entities(tags)  # 67%
        for chunk_type, chunk_start, chunk_end in chunks:
            chunk = self.post_process_chunk(chunk_type, chunk_start, chunk_end, split_text, poss)  # 27%
            if chunk is not None:
                entity = {
                    'text': chunk,
                    'type': chunk_type,
                    'beginOffset': chunk_start,
                    'endOffset': chunk_end
                }
                res['entities'].append(entity)
        return res

    def post_process_chunk(self, chunk_type, chunk_start, chunk_end, split_text, pos):
        if chunk_type == 'Chunk':
            chunk_inner_words = split_text[chunk_start: chunk_end+1]
            chunk = ''.join(chunk_inner_words)
            check_char = not re.search(r'[^a-zA-Z0-9\u4e00-\u9fa5\.\+#]+', chunk)
            if len(chunk) < 15 and len(chunk) > 2 and check_char and len(chunk_inner_words) > 1:
                chunk_inner_poss = pos[chunk_start: chunk_end+1]
                filter_flag = any([check_in(chunk_inner_words, self.stopwords),
                                   check_in([chunk_inner_words[0]], self.stopwords_first),
                                   check_in([chunk_inner_words[-1]], self.stopwords_last),
                                   check_in(chunk_inner_poss, self.pos_filter),
                                   check_in([chunk_inner_poss[-1]], self.pos_filter_first)])
                if not filter_flag:
                    return chunk
            else:
                return None
        else:
            return None

    def get_chunk(self, chunk):
        if self.qualifier and chunk in self.qualifier:
            return self.qualifier[chunk]
        return chunk

    def output(self, res):
        words = res['words']
        poss = res['pos']
        dict_idx = tag_by_dict(words, self.tree)  # 62%
        model_idx = [[item['beginOffset'], item['endOffset']] for item in res['entities']]
        new_idx = sorted(list(compare_idx(dict_idx, model_idx)), key=lambda x: x[1])  # 5.6%
        new_word = []
        new_pos = []
        new_chunk = []
        if self.char_input:
            seg = res['seg']
            tag = ['O'] * len(seg)
            char_pos = res['char_pos']
            char_word = res['char_word']
            assert len(char_pos) == len(seg) == len(char_word)
            for s, e in new_idx:
                tag[s:e] = ['B-Chunk'] + ['I-Chunk'] * (e-s-1) + ['E-Chunk']
            chunks = {e: ''.join(words[s:e+1]) for s, e in new_idx}
            start = 0
            mid = 0
            for j, item_BEMS in enumerate(seg):
                if tag[j] == 'O':
                    if item_BEMS == 'S':
                        new_word.append(char_word[j])
                        new_pos.append(char_pos[j])
                    elif item_BEMS == 'E':
                        if not tag[j-1].endswith('Chunk'):
                            if not tag[start].endswith('Chunk'):
                                new_word.append(char_word[j])
                            else:
                                new_word.append(''.join(words[mid:j]))
                        else:
                            new_word.append(words[j])
                        new_pos.append(char_pos[j])
                    else:
                        if item_BEMS == 'B':
                            start = j
                        if tag[j+1].endswith('Chunk'):
                            new_word.append(''.join(words[start:j]))
                            new_pos.append(char_pos[j])
                        if tag[j-1].endswith('Chunk') and item_BEMS == 'M':
                            mid = j
                elif tag[j] == 'E-Chunk':
                    try:
                        chunk = chunks[j]
                        if self.qualifier:
                            if chunk in self.qualifier:
                                qualifier_word = self.qualifier[chunk]
                                new_word.append(qualifier_word)
                                new_chunk.append(qualifier_word)
                            else:
                                new_word.append(chunk)
                                new_chunk.append(chunk)
                        else:
                            new_word.append(chunk)
                            new_chunk.append(chunk)
                    except Exception as e:
                        print(e)
                    new_pos.append('np')
        else:
            chunks = {item[1]: ''.join(words[item[0]: item[1]+1]) for item in new_idx}
            chunk_idx = [i for item in new_idx for i in range(item[0], item[1]+1)]
            for i, item in enumerate(words):
                if i not in chunk_idx:
                    new_word.append(item)
                    new_pos.append(poss[i])
                else:
                    if i in chunks.keys():
                        chunk = chunks[i]
                        if self.qualifier:
                            if chunk in self.qualifier:
                                qualifier_word = self.qualifier[chunk]
                                new_word.append(qualifier_word)
                                new_chunk.append(qualifier_word)
                            else:
                                new_word.append(chunk)
                                new_chunk.append(chunk)
                        else:
                            new_word.append(chunk)
                            new_chunk.append(chunk)
                        new_pos.append('np')
        try:
            assert len(new_word) == len(new_pos)
        except Exception as e:
            print('new word list length not equals with new pos list')
            print(new_word, len(new_word))
            print(new_pos, len(new_pos))
            print(chunks)
            print(dict_idx, model_idx, new_idx)
        return (new_word, new_pos, new_chunk)  # C_WORD=0 C_POS=1 C_CHUNK=2

    def analyze(self, text):
        assert isinstance(text, list) or isinstance(text, tuple)
        final_res = []
        sent_idx, batch_data = self.data_generator(text)
        split_text, split_pos, pred, segs, word = self.predict_proba_batch(batch_data)
        split_text = [split_text[sent_idx[i]:sent_idx[i+1]] for i in range(len(sent_idx)-1)]
        split_pos = [split_pos[sent_idx[i]:sent_idx[i+1]] for i in range(len(sent_idx)-1)]
        pred = [np.array(pred[sent_idx[i]:sent_idx[i+1]]) for i in range(len(sent_idx)-1)]
        if self.char_input:
            segs = [segs[sent_idx[i]:sent_idx[i+1]] for i in range(len(sent_idx)-1)]
            word = [word[sent_idx[i]:sent_idx[i+1]] for i in range(len(sent_idx)-1)]
            assert len(segs) == len(split_text) == len(pred)
        for k, item in enumerate(pred):
            tmp_y = [y[:len(x)] for x, y in zip(split_text[k], item)]
            Y = np.concatenate(tmp_y)
            words = list(flatten_gen(split_text[k]))
            poss = list(flatten_gen(split_pos[k]))
            # assert len(words) == len(poss)
            if self.char_input:
                split_segs = list(flatten_gen(segs[k]))
                split_words = list(flatten_gen(word[k]))
            else:
                split_segs = []
                split_words = []
            tags = self._get_tags(Y)
            # prob = self._get_prob(Y)
            res = self._build_response(words, tags, poss, split_segs, split_words)
            final_res.append(self.output(res))
        return final_res, self.wrong
