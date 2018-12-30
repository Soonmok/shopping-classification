# -*- coding: utf-8 -*-
# Copyright 2017 Kakao, Recommendation Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf

import keras
from keras.models import Model
from keras.layers.merge import dot
from keras.layers import Dense, Input
from keras.layers.core import Reshape

from keras.layers.embeddings import Embedding
from keras.layers.core import Dropout, Activation

from misc import get_logger, Option
opt = Option('./config.json')


def top1_acc(x, y):
    return keras.metrics.top_k_categorical_accuracy(x, y, k=1)


class TextOnly:
    def __init__(self):
        self.logger = get_logger('textonly')

    # input1(embeded) * input2(reshaped) --> dropout --> dense layer -> output
    def get_model(self, num_classes, activation='sigmoid'):
        max_len = opt.max_len
        voca_size = opt.unigram_hash_size + 1

        with tf.device('/gpu:0'):
            # voca_size = 32
            # embd_size = 128
            embd = Embedding(voca_size,
                             opt.embd_size,
                             name='uni_embd')

            # (None, 32)
            t_uni = Input((max_len,), name="input_1")
            # (None, 32, 128)
            t_uni_embd = embd(t_uni)  # token

            # (None, 32)
            w_uni = Input((max_len,), name="input_2")
            # (None, 32, 1)
            w_uni_mat = Reshape((max_len, 1))(w_uni)  # weight

            # (None, 128, 1)
            uni_embd_mat = dot([t_uni_embd, w_uni_mat], axes=1)
            # (None, 128)
            uni_embd = Reshape((opt.embd_size, ))(uni_embd_mat)

            # (None, 128)
            embd_out = Dropout(rate=0.5)(uni_embd)
            # (None, 128)
            relu = Activation('relu', name='relu1')(embd_out)
            # (None, 4215)
            outputs = Dense(num_classes, activation=activation)(relu)
            model = Model(inputs=[t_uni, w_uni], outputs=outputs)
            optm = keras.optimizers.Nadam(opt.lr)
            model.compile(loss='binary_crossentropy',
                        optimizer=optm,
                        metrics=[top1_acc])
            model.summary(print_fn=lambda x: self.logger.info(x))
        return model

class OnlyImage:
    def __init__(self):
        self.logger = get_logger("imageOnly")

    def get_model(self, num_classes, activation='sigmoid'):
        #TODO: 이미지 피쳐 shape 일치 시키기 받아서 (None, num_classes) output 반환 
        with tf.device('/gpu:0'):
            img_feature = Input((100), name="image_input")
            relu = Activation('relu', name='relu1')(embd_out)
            outputs = Dense(num_classes, activation=activation)(relu)
            model = model(inputs=img_feature, outputs=outputs)
            optm - keras.optimizers.Nadam(opt, lr)
            model.compile(loss='binary_crossentropy',
                            optimizer=optm,
                            metrics=[top1_acc])
        return model


class TwoBranch:
    def __init__(self):
        self.logger = get_logger("TwoBranch")
    def get_model(self, num_classes, Activation="sigmoid"):
        
        #TODO: 이미지 피쳐와 텍스트 피쳐를 각각 Dense 로 처리한뒤 concat 한것을 다시 Dense 로 통과 시킨것을 outputs으로 (논문 적용)
        with tf.device('/gpu:0'):
            img_feature = Input((100, name="image_input"))
            text_feature = Input((32, name="text_input"))
            relu = Activation('relu', name='relu1')(mixed_feature)
            outputs = Dense(num_classes, activation=Activation)(relu)
            model = Model(inputs=mixed_feature, outputs=outputs)
            model.compile(loss='binary_crossentropy',
                optimizer=optm,
                metrics=[top1_acc])
        
        return model