"""chunk model"""

import keras.backend as K
from keras.models import Model
from keras.layers import Embedding, Input, Dense, TimeDistributed, Activation, Dropout
from keras.layers import Lambda, multiply, add, subtract
from keras.layers import LSTM, GRU, Bidirectional
from keras.layers import Conv1D, GlobalMaxPooling1D, BatchNormalization
from keras.layers.merge import Concatenate
from keras_contrib.layers import CRF
from keras.engine.topology import Layer
from keras import regularizers
import json

"""
main structure: bi-lstm + crf
1. baseline：word-emb + main structure (done)
2. model 1：char-emb || word-emb attention + main structure (done)
3. model 2: char-emb || bichar-emb || seg-emb + main structure (done)
4. model 3: lattice lstm
5. model 4: word-emb || pos-emb + conv1d (done)
5. model 5: seq2seq + pointer
"""


class Base_Model(object):
    def __init__(self):
        self.model = None

    def save(self, weights_file, params_file):
        self.save_weights(weights_file)
        self.save_params(params_file)

    def save_params(self, file_path):
        with open(file_path, 'w') as f:
            invalid_params = {'_loss', 'model', 'word_embeddings',
                              'char_embeddings', '_acc', 'word_embed',
                              'char_feature', 'cosine_loss'}
            params = {name.lstrip('_'): val for name, val in vars(self).items()
                      if name not in invalid_params}
            print(params)
            json.dump(params, f, sort_keys=True, indent=4)

    def save_weights(self, filepath):
        self.model.save_weights(filepath)

    @classmethod
    def load(cls, weights_file, params_file):
        params = cls.load_params(params_file)
        self = cls(**params)
        self.build()
        self.model.load_weights(weights_file)
        return self

    @classmethod
    def load_params(cls, file_path):
        with open(file_path) as f:
            params = json.load(f)
        return params


def last_layer(input_layer, num_labels, use_crf=True):
    if use_crf:
        crf = CRF(num_labels, sparse_target=False)
        loss = crf.loss_function
        acc = [crf.accuracy]
        output = crf(input_layer)
    else:
        loss = 'categorical_crossentropy'
        acc = ['acc']
        output = Dense(num_labels, activation='softmax')(input_layer)
    return output, loss, acc


class WORD_RNN(Base_Model):
    def __init__(self,
                 num_labels,
                 word_vocab_size, char_vocab_size=None,
                 word_embedding_dim=128, char_embedding_dim=32,
                 word_rnn_size=128, char_rnn_size=32,
                 word_embeddings=None, char_embeddings=None,
                 use_char=False, use_crf=True,
                 char_feature_method='rnn',
                 integration_method='concat',
                 rnn_type='lstm',
                 num_rnn_layers=1,
                 num_filters=32,
                 conv_kernel_size=2,
                 drop_rate=0.5,
                 re_drop_rate=0.15):
        super(WORD_RNN).__init__()
        self.num_labels = num_labels
        self.word_vocab_size = word_vocab_size
        self.word_embedding_dim = word_embedding_dim
        self.word_rnn_size = word_rnn_size
        if word_embeddings is not None:
            self.word_embeddings = [word_embeddings]
        else:
            self.word_embeddings = word_embeddings
        self.use_char = use_char
        if self.use_char:
            self.char_vocab_size = char_vocab_size
            self.char_embedding_dim = char_embedding_dim
            self.integration_method = integration_method
            self.char_embeddings = char_embeddings
            self.char_feature_method = char_feature_method
            if self.char_feature_method == 'rnn':
                if self.integration_method == 'attention':
                    self.char_rnn_size = int(self.word_embedding_dim / 2)
                else:
                    self.char_rnn_size = char_rnn_size
            elif self.char_feature_method == 'cnn':
                self.num_filters = num_filters
                self.conv_kernel_size = conv_kernel_size
                if self.integration_method == 'attention':
                    self.num_filters = self.word_embedding_dim
        self.rnn_type = rnn_type
        self.num_rnn_layers = num_rnn_layers
        self.drop_rate = drop_rate
        self.re_drop_rate = re_drop_rate
        self.use_crf = use_crf

    def build(self):
        word_ids = Input(batch_shape=(None, None), dtype='int32', name='word_input')
        lengths = Input(batch_shape=(None, None), dtype='int32', name='length_input')
        input_data = [word_ids]
        self.word_embed = Embedding(input_dim=self.word_vocab_size,
                                    output_dim=self.word_embedding_dim,
                                    weights=self.word_embeddings,
                                    mask_zero=True,
                                    name='word_embed')(word_ids)

        # char信息提取模块
        if self.use_char:
            char_ids = Input(batch_shape=(None, None, None), dtype='int32', name='char_input')
            input_data.append(char_ids)
            mask_zero = False
            if self.char_feature_method == 'rnn':
                mask_zero = True
            char_embed = TimeDistributed(
                Embedding(input_dim=self.char_vocab_size,
                          output_dim=self.char_embedding_dim,
                          weights=self.char_embeddings,
                          mask_zero=mask_zero,
                          name='char_embed'))(char_ids)
            if self.char_feature_method == 'rnn':
                if self.rnn_type == 'lstm':
                    self.char_feature = TimeDistributed(
                        Bidirectional(LSTM(self.char_rnn_size,
                                           return_sequences=False)), name="char_lstm")(char_embed)
                else:
                    self.char_feature = TimeDistributed(
                        Bidirectional(GRU(self.char_rnn_size,
                                          return_sequences=False)), name="char_gru")(char_embed)
            elif self.char_feature_method == 'cnn':
                conv1d_out = TimeDistributed(Conv1D(kernel_size=self.conv_kernel_size, filters=self.num_filters,
                                                    padding='same'), name='char_cnn')(char_embed)
                self.char_feature = TimeDistributed(GlobalMaxPooling1D(), name='char_pooling')(conv1d_out)
            if self.integration_method == 'concat':
                concat_tensor = Concatenate(axis=-1, name='concat_feature')([self.word_embed, self.char_feature])
            elif self.integration_method == 'attention':
                word_embed_dense = Dense(self.word_embedding_dim,
                                         kernel_initializer="glorot_uniform",
                                         activation='tanh')(self.word_embed)
                char_embed_dense = Dense(self.word_embedding_dim,
                                         kernel_initializer="glorot_uniform",
                                         activation='tanh')(self.char_feature)
                attention_evidence_tensor = add([word_embed_dense, char_embed_dense])
                attention_output = Dense(self.word_embedding_dim, activation='sigmoid')(attention_evidence_tensor)
                part1 = multiply([attention_output, self.word_embed])
                tmp = subtract([Lambda(lambda x: K.ones_like(x))(attention_output), attention_output])
                part2 = multiply([tmp, self.char_feature])
                concat_tensor = add([part1, part2], name='attention_feature')

        input_data.append(lengths)

        # rnn编码模块
        if not self.use_char:
            enc = self.word_embed
        else:
            enc = concat_tensor
        for i in range(self.num_rnn_layers):
            if self.rnn_type == 'lstm':
                enc = Bidirectional(LSTM(self.word_rnn_size, dropout=self.drop_rate,
                                         recurrent_dropout=self.re_drop_rate,
                                         return_sequences=True), name='word_lstm_%d' % (i+1))(enc)
            elif self.rnn_type == 'gru':
                enc = Bidirectional(GRU(self.word_rnn_size, dropout=self.drop_rate,
                                        recurrent_dropout=self.re_drop_rate,
                                        return_sequences=True), name='word_gru_%d' % (i+1))(enc)

        # 标签预测模块
        output, self._loss, self._acc = last_layer(enc, self.num_labels, self.use_crf)

        self.model = Model(inputs=input_data, outputs=output)

    def get_loss(self):
        return self._loss

    def get_metrics(self):
        return self._acc

    def cosine_loss(self):
        return 0

    def crf_cosine_loss(self, y_true, y_pred):
        loss1 = self._loss(y_true, y_pred)
        loss2 = self.cosine_loss()
        return loss1 + loss2


class CHAR_RNN(Base_Model):
    def __init__(self,
                 num_labels,
                 char_vocab_size,
                 seg_vocab_size=None,
                 bichar_vocab_size=None,
                 radical_vocab_size=None,
                 char_embedding_dim=64,
                 char_embeddings=None,
                 seg_embedding_dim=8,
                 bichar_embedding_dim=64,
                 bichar_embeddings=None,
                 radical_embedding_dim=32,
                 char_rnn_size=128,
                 seg_info=False,
                 bi_char=False,
                 radical=False,
                 rnn_type='lstm',
                 num_rnn_layers=1,
                 drop_rate=0.5,
                 re_drop_rate=0.15,
                 use_crf=True):
        super(CHAR_RNN).__init__()
        self.num_labels = num_labels
        self.char_vocab_size = char_vocab_size
        self.seg_vocab_size = seg_vocab_size
        self.bichar_vocab_size = bichar_vocab_size
        self.radical_vocab_size = radical_vocab_size
        self.char_embedding_dim = char_embedding_dim
        if char_embeddings is not None:
            self.char_embeddings = [char_embeddings]
        else:
            self.char_embeddings = char_embeddings
        self.seg_embedding_dim = seg_embedding_dim
        self.radical_embedding_dim = radical_embedding_dim
        self.bichar_embedding_dim = bichar_embedding_dim
        if bichar_embeddings is not None:
            self.bichar_embeddings = [bichar_embeddings]
        else:
            self.bichar_embeddings = bichar_embeddings
        self.char_rnn_size = char_rnn_size
        self.seg_info = seg_info
        self.bi_char = bi_char
        self.radical = radical
        self.rnn_type = rnn_type
        self.num_rnn_layers = num_rnn_layers
        self.drop_rate = drop_rate
        self.re_drop_rate = re_drop_rate
        self.use_crf = use_crf

    def build(self):
        char_ids = Input(batch_shape=(None, None), dtype='int32', name='char_input')
        lengths = Input(batch_shape=(None, None), dtype='int32', name='length_input')
        input_data = [char_ids]
        char_embed = Embedding(input_dim=self.char_vocab_size,
                               output_dim=self.char_embedding_dim,
                               weights=self.char_embeddings,
                               mask_zero=True,
                               name='char_embed')(char_ids)
        other_features = [char_embed]

        if self.bi_char:
            bichar_ids = Input(batch_shape=(None, None), dtype='int32', name='bi_char_input')
            input_data.append(bichar_ids)
            bichar_embed = Embedding(input_dim=self.bichar_vocab_size,
                                     output_dim=self.bichar_embedding_dim,
                                     weights=self.bichar_embeddings,
                                     mask_zero=True,
                                     name='bichar_embed')(bichar_ids)
            other_features.append(bichar_embed)
        if self.seg_info:
            seg_ids = Input(batch_shape=(None, None), dtype='int32', name='seg_input')
            input_data.append(seg_ids)
            seg_embed = Embedding(input_dim=self.seg_vocab_size+1,
                                  output_dim=self.seg_embedding_dim,
                                  mask_zero=True,
                                  name='seg_embed')(seg_ids)
            other_features.append(seg_embed)
        if self.radical:
            radical_ids = Input(batch_shape=(None, None), dtype='int32', name='radical_input')
            input_data.append(radical_ids)
            radical_embed = Embedding(input_dim=self.radical_vocab_size+1,
                                      output_dim=self.radical_embedding_dim,
                                      mask_zero=True,
                                      name='radical_embed')(radical_ids)
            other_features.append(radical_embed)

        input_data.append(lengths)

        if self.seg_info or self.bi_char or self.radical:
            enc = Concatenate(axis=-1, name='concat_features')(other_features)
        else:
            enc = char_embed
        enc = BatchNormalization()(enc)
        for i in range(self.num_rnn_layers):
            if self.rnn_type == 'lstm':
                enc = Bidirectional(LSTM(self.char_rnn_size, dropout=self.drop_rate,
                                         recurrent_dropout=self.re_drop_rate,
                                         return_sequences=True), name='char_lstm_%d' % (i+1))(enc)
            elif self.rnn_type == 'gru':
                enc = Bidirectional(GRU(self.char_rnn_size, dropout=self.drop_rate,
                                        recurrent_dropout=self.re_drop_rate,
                                        return_sequences=True), name='char_gru_%d' % (i+1))(enc)

        output, self._loss, self._acc = last_layer(enc, self.num_labels, self.use_crf)

        self.model = Model(inputs=input_data, outputs=output)

    def get_loss(self):
        return self._loss

    def get_metrics(self):
        return self._acc


class CNN_POS(Base_Model):
    def __init__(self,
                 num_labels,
                 word_vocab_size,
                 word_embedding_dim=128,
                 word_embeddings=None,
                 num_filters=128,
                 conv_kernel_size=3,
                 num_cnn_block=4,
                 drop_rate=0.1,
                 use_crf=True):
        super(CNN_POS).__init__()
        self.num_labels = num_labels
        self.word_vocab_size = word_vocab_size
        self.word_embedding_dim = word_embedding_dim
        if word_embeddings is not None:
            self.word_embeddings = [word_embeddings]
        else:
            self.word_embeddings = word_embeddings
        self.num_filters = num_filters
        self.conv_kernel_size = conv_kernel_size
        self.drop_rate = drop_rate
        self.num_cnn_block = num_cnn_block
        self.use_crf = use_crf

    def build(self):
        word_ids = Input(batch_shape=(None, None), dtype='int32', name='word_input')
        lengths = Input(batch_shape=(None, None), dtype='int32', name='length_input')
        input_data = [word_ids, lengths]
        word_embed = Embedding(input_dim=self.word_vocab_size,
                               output_dim=self.word_embedding_dim,
                               weights=self.word_embeddings,
                               name='word_embed')(word_ids)
        pos_embed = Position_Embedding()(word_embed)
        for idx in range(self.num_cnn_block):
            if idx == 0:
                conv_block = self.cnn_block(pos_embed, idx)
            else:
                conv_block = self.cnn_block(conv_block, idx)

        output, self._loss, self._acc = last_layer(conv_block, self.num_labels, self.use_crf)

        self.model = Model(inputs=input_data, outputs=output)

    def cnn_block(self, input_tensor, idx):
        conv = Conv1D(self.num_filters, self.conv_kernel_size, padding='same', name='conv_%d' % (idx+1))(input_tensor)
        relu = Activation('relu')(conv)
        drop = Dropout(self.drop_rate)(relu)
        bn = BatchNormalization()(drop)
        return bn

    def get_loss(self):
        return self._loss

    def get_metrics(self):
        return self._acc


class IDCNN(Base_Model):
    def __init__(self, num_labels,
                 char_vocab_size,
                 seg_vocab_size=None,
                 char_embedding_dim=64,
                 seg_embedding_dim=8,
                 drop_rate=0.25,
                 num_filters=64,
                 conv_kernel_size=3,
                 dilation_rate=[1, 1, 2],
                 repeat_times=4,
                 use_crf=True,
                 ):
        super(IDCNN).__init__()

        self.num_labels = num_labels
        self.char_vocab_size = char_vocab_size
        self.char_embedding_dim = char_embedding_dim
        self.seg_vocab_size = seg_vocab_size
        self.seg_embedding_dim = seg_embedding_dim
        self.drop_rate = drop_rate
        self.num_filters = num_filters
        self.conv_kernel_size = conv_kernel_size
        self.dilation_rate = dilation_rate
        self.repeat_times = repeat_times
        self.use_crf = use_crf

    def build(self):
        char_ids = Input(batch_shape=(None, None), dtype='int32', name='char_input')
        seg_ids = Input(batch_shape=(None, None), dtype='int32', name='seg_input')
        lengths = Input(batch_shape=(None, None), dtype='int32', name='length_input')
        input_data = [char_ids, seg_ids, lengths]
        char_embed = Embedding(input_dim=self.char_vocab_size,
                               output_dim=self.char_embedding_dim,
                               name='char_embed')(char_ids)
        seg_embed = Embedding(input_dim=self.seg_vocab_size,
                              output_dim=self.seg_embedding_dim,
                              name='seg_embed')(seg_ids)
        embed = Concatenate(axis=-1, name='concat_features')([char_embed, seg_embed])
        embed = Dropout(self.drop_rate)(embed)
        layerInput = Conv1D(
            self.num_filters, self.conv_kernel_size, padding='same', name='conv_1')(embed)
        dilation_layers = []
        totalWidthForLastDim = 0
        for j in range(self.repeat_times):
            for i in range(len(self.dilation_rate)):
                islast = True if i == len(self.dilation_rate) - 1 else False
                conv = Conv1D(self.num_filters, self.conv_kernel_size, use_bias=True,
                              padding='same', dilation_rate=self.dilation_rate[i],
                              name='atrous_conv_%d_%d' % (j, i))(layerInput)
                conv = Activation('relu')(conv)
                if islast:
                    dilation_layers.append(conv)
                    totalWidthForLastDim += self.num_filters
                layerInput = conv
        dilation_conv = Concatenate(axis=-1)(dilation_layers)
        enc = Dropout(0.5)(dilation_conv)

        output, self._loss, self._acc = last_layer(enc, self.num_labels, self.use_crf)

        self.model = Model(inputs=input_data, outputs=output)

    def get_loss(self):
        return self._loss

    def get_metrics(self):
        return self._acc


# 位置嵌入向量
class Position_Embedding(Layer):
    def __init__(self, size=None, mode='sum', **kwargs):
        self.size = size  # 必须为偶数
        self.mode = mode
        super(Position_Embedding, self).__init__(**kwargs)

    def call(self, x):
        if (self.size is None) or (self.mode == 'sum'):
            self.size = int(x.shape[-1])
        position_j = 1. / K.pow(10000., 2 * K.arange(self.size / 2, dtype='float32') / self.size)
        position_j = K.expand_dims(position_j, 0)
        position_i = K.cumsum(K.ones_like(x[:, :, 0]), 1) - 1  # K.arange不支持变长，只好用这种方法生成
        position_i = K.expand_dims(position_i, 2)
        position_ij = K.dot(position_i, position_j)
        position_ij = K.concatenate([K.cos(position_ij), K.sin(position_ij)], 2)
        if self.mode == 'sum':
            return position_ij + x
        elif self.mode == 'concat':
            return K.concatenate([position_ij, x], 2)

    def compute_output_shape(self, input_shape):
        if self.mode == 'sum':
            return input_shape
        elif self.mode == 'concat':
            return (input_shape[0], input_shape[1], input_shape[2]+self.size)


class Attention_Concat(Layer):
    def __init__(self, weight_w12, weight_w3, **kwargs):
        self.weight_w12 = weight_w12
        self.weight_w3 = weight_w3
        super(Attention_Concat, self).__init__(**kwargs)

    def build(self, input_shape):
        self.w1 = self.add_weight(name="W_{:s}".format(self.name),
                                  shape=(input_shape[0][-1], self.weight_w12),
                                  initializer="glorot_normal",
                                  trainable=True)
        self.w2 = self.add_weight(name="W_{:s}".format(self.name),
                                  shape=(input_shape[1][-1], self.weight_w12),
                                  initializer="glorot_normal",
                                  trainable=True)
        self.w3 = self.add_weight(name="W_{:s}".format(self.name),
                                  shape=(self.weight_w12, self.weight_w3),
                                  initializer="glorot_normal",
                                  trainable=True)
        super(Attention_Concat, self).build(input_shape)

    def call(self, x):
        tmp = K.tanh(K.dot(x[0], self.w1) + K.dot(x[1], self.w2))
        z = K.sigmoid(K.dot(tmp, self.w3))
        x_new = z * x[0] + (1 - z) * x[1]
        return x_new

    def compute_mask(self, input, input_mask=None):
        return None

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], input_shape[0][2])
