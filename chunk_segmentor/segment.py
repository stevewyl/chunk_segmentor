# ======主程序========
import os
import gc
import pickle
from pathlib import Path
from collections import Counter
from chunk_segmentor.trie import Trie
from chunk_segmentor.utils import read_line, flatten_gen, sent_split, preprocess, hanlp_cut
from chunk_segmentor.segmentor import SafeJClass
from chunk_segmentor.preprocessing import IndexTransformer
from chunk_segmentor.model import WORD_RNN, CHAR_RNN
from chunk_segmentor.tagger import Tagger
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
set_session(tf.Session(config=tf_config))
gc.disable()

global model_loaded
global last_model_name
global Tree
global Labeler
last_model_name = ''
tree_loaded = False
Labeler = None
Tree = None


class Chunk_Labeler(object):
    def __init__(self, model_name='word-rnn', tagger=None):
        self.model_name = model_name
        self.tagger = tagger

    def analyze(self, text, has_seq=True, char_input=False,
                mode='batch', batch_size=256, radical_file=''):
        if mode == 'single':
            batch_size = 1
        if not self.tagger:
            if self.model_name in ['char-rnn', 'idcnn']:
                char_input = True
            self.tagger = Tagger(self.model, self.p, char_input,
                                 mode, batch_size, radical_file)
        # print(self.tagger.get_tagger_info)
        return self.tagger.analyze(text)

    @classmethod
    def load(cls, model_name, weight_file, params_file, preprocessor_file):
        self = cls(model_name=model_name)
        self.p = IndexTransformer.load(preprocessor_file)
        if model_name == 'word-rnn':
            self.model = WORD_RNN.load(weight_file, params_file)
        elif model_name == 'char-rnn':
            self.model = CHAR_RNN.load(weight_file, params_file)
        else:
            print('No other available models for chunking')
            print('Please use word-rnn or char-rnn')
        return self


class Chunk_Segmentor(object):
    def __init__(self, model_name='word-rnn', mode='accurate', verbose=0):
        self.pos = True
        self.mode = mode
        self.qualifier = True
        self.verbose = verbose
        self.path = os.path.abspath(os.path.dirname(__file__))
        if model_name != '':
            self.model_name = model_name
        else:
            try:
                self.model_name = read_line(Path(self.path) / 'data' / 'best_model.txt')[0]
            except Exception:
                self.model_name = model_name

        # hanlp变量
        self.hanlp_static = Path(self.path) / 'segmentor' / 'static'
        self.hanlp_config = self.hanlp_static / 'hanlp.properties'
        self.hanlp_dictionary = self.hanlp_static / 'data' / 'dictionary' / 'custom'
        custom_dict = SafeJClass('com.hankcs.hanlp.dictionary.CustomDictionary')
        if self.mode == 'fast':
            dict_path = self.hanlp_static / 'data_withchunk' / 'CustomDictionary.txt'
            custom_dict.loadDat(str(dict_path))
        elif self.mode == 'accurate':
            dict_path = self.hanlp_static / 'data_nochunk' / 'CustomDictionary.txt'
            custom_dict.loadDat(str(dict_path))
        self.seg = SafeJClass('com.hankcs.hanlp.HanLP')

        # model变量
        self.weight_file = os.path.join(self.path, 'data/model/%s_weights.h5' % self.model_name)
        self.param_file = os.path.join(self.path, 'data/model/%s_parms.h5' % self.model_name)
        self.preprocess_file = os.path.join(self.path, 'data/model/%s_preprocess.h5' % self.model_name)
        self.define_tagger()

    def define_tagger(self):
        if self.qualifier:
            qualifier_word_path = os.path.join(self.path, 'data/dict/chunk_qualifier.dict')
            self.qualifier_word = pickle.load(open(qualifier_word_path, 'rb'))
        else:
            self.qualifier_word = None
        if self.mode == 'fast' and self.qualifier:
            self.fast_qualifier = True
        else:
            self.fast_qualifier = False

        char_input = True if self.model_name[:4] == 'char' else False

        # acc模式变量
        if self.mode == 'accurate':
            global tree_loaded
            global last_model_name
            global Labeler
            global Tree
            if self.verbose:
                print('Model and Trie Tree are loading. It will cost 10-20s.')
            if self.model_name != last_model_name:
                self.labeler = Chunk_Labeler.load(
                    self.model_name, self.weight_file, self.param_file, self.preprocess_file)
                if self.verbose:
                    print('load model succeed')
                last_model_name = self.model_name
                Labeler = self.labeler
            else:
                self.labeler = Labeler
            if not tree_loaded:
                chunk_dict = read_line(os.path.join(self.path, 'data/dict/chunk.txt'))
                self.tree = Trie()
                for chunk in chunk_dict:
                    self.tree.insert(chunk)
                if self.verbose:
                    print('trie tree succeed')
                tree_loaded = True
                Tree = self.tree
            else:
                self.tree = Tree
            radical_file = os.path.join(self.path, 'data/dict/radical.txt')
            self.tagger = Tagger(self.labeler.model, self.labeler.p,
                                 char_input=char_input, radical_file=radical_file,
                                 tree=self.tree, qualifier=self.qualifier_word)

    @property
    def get_segmentor_info(self):
        params = {'model_name': self.model_name,
                  'mode': self.mode,
                  'pos': self.pos,
                  'qualifier': self.qualifier}
        return params

    def extract_item(self, item):
        C_CUT_WORD, C_CUT_POS, C_CUT_CHUNK = 0, 1, 2
        complete_words = [sub[C_CUT_WORD] for sub in item]
        complete_poss = [sub[C_CUT_POS] for sub in item]
        if self.mode == 'fast':
            all_chunks = [x for sub in item for x, y in zip(sub[C_CUT_WORD], sub[C_CUT_POS]) if y == 'np']
        else:
            all_chunks = list(flatten_gen([sub[C_CUT_CHUNK] for sub in item]))
        if self.pos:
            d = (list(flatten_gen(complete_words)),   # C_CUT_WORD
                 list(flatten_gen(complete_poss)),    # C_CUT_POS
                 list(dict.fromkeys(all_chunks)))   # C_CUT_CHUNK
        else:
            d = (list(flatten_gen(complete_words)), list(dict.fromkeys(all_chunks)))
        return d

    def output(self, data):
        idx_list, strings = zip(
            *[[idx, sub] for idx, item in enumerate(data) for sub in sent_split(preprocess(item))])
        cc = list(Counter(idx_list).values())
        end_idx = [sum(cc[:i]) for i in range(len(cc)+1)]

        if self.fast_qualifier:
            seg_res = hanlp_cut(strings, self.seg,
                                self.qualifier_word, mode=self.mode)
        else:
            seg_res = hanlp_cut(strings, self.seg, mode=self.mode)

        if self.mode == 'accurate':
            outputs, _ = self.tagger.analyze(seg_res)
        else:
            outputs = [list(zip(*item)) for item in seg_res]

        new_res = (outputs[end_idx[i]: end_idx[i+1]]
                   for i in range(len(end_idx)-1))
        for item in new_res:
            yield self.extract_item(item)

    def cut(self, data, batch_size=512, pos=True, qualifier=True):
        if isinstance(data, str):
            data = [data]
        if not pos:
            self.pos = False
        else:
            self.pos = True
        if not qualifier:
            self.qualifier = False
            self.define_tagger()
        else:
            self.qualifier = True
            self.define_tagger()
        assert isinstance(data, list)
        data_cnt = len(data)
        num_batches = int(data_cnt / batch_size) + 1
        if self.verbose:
            print('total_batch_num: ', num_batches)
        for batch_num in range(num_batches):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_cnt)
            batch_input = data[start_index:end_index]
            for res in self.output(batch_input):
                yield res


if __name__ == "__main__":
    cutter = Chunk_Segmentor()
    cutter.cut('这是一个能够输出名词短语的分词器，欢迎试用！')
