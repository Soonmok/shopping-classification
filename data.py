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
import os
os.environ['OMP_NUM_THREADS'] = '1'
import re
import sys
import traceback
from collections import Counter
from multiprocessing import Pool

import tqdm
import fire
import h5py
import numpy as np
import mmh3
import six
from keras.utils.np_utils import to_categorical
from six.moves import cPickle

from misc import get_logger, Option
opt = Option('./config.json')

re_sc = re.compile('[\!@#$%\^&\*\(\)-=\[\]\{\}\.,/\?~\+\'"|]')


class Reader(object):
    def __init__(self, data_path_list, div, begin_offset, end_offset):
        self.div = div
        self.data_path_list = data_path_list
        self.begin_offset = begin_offset
        self.end_offset = end_offset

    # 현재 데이터의 인덱스가 chunk offset 범위안에 있는지 여부를 따짐
    # input --> ex) 20034
    # output --> ex) True (현재 begin_offset이 20000, 현재 end_offset이 40000일 경우)
    def is_range(self, i):
        if self.begin_offset is not None and i < self.begin_offset:
            return False
        if self.end_offset is not None and self.end_offset <= i:
            return False
        return True

    # 가져온 데이터의 총 크기 계산 뒤 반환
    def get_size(self):
        offset = 0
        count = 0
        for data_path in self.data_path_list:
            h = h5py.File(data_path, 'r')
            sz = h[self.div]['pid'].shape[0]
            if not self.begin_offset and not self.end_offset:
                offset += sz
                count += sz
                continue
            if self.begin_offset and offset + sz < self.begin_offset:
                offset += sz
                continue
            if self.end_offset and self.end_offset < offset:
                break
            for i in range(sz):
                if not self.is_range(offset + i):
                    continue
                count += 1
            offset += sz
        return count

    # 해당하는 인덱스의 데이터 클래스 반환 -> ex) '133>244>5>1'
    def get_class(self, h, i):
        b = h['bcateid'][i]
        m = h['mcateid'][i]
        s = h['scateid'][i]
        d = h['dcateid'][i]
        return '%s>%s>%s>%s' % (b, m, s, d)

    # 해당하는 데이터 chunk(split으로 나누어진 데이터)의 class(분류 카테고리)를 h5py파일에서 가져옴
    # input --> ex) None
    # ouput --> ex) None  --> 왜냐면 yield로 지속적으로 변수를 업데이트하고있어서 리턴할 필요 x
    def generate(self):
        offset = 0
        for data_path in self.data_path_list:
            h = h5py.File(data_path, 'r')[self.div]
            sz = h['pid'].shape[0]
            if self.begin_offset and offset + sz < self.begin_offset:
                offset += sz
                continue
            if self.end_offset and self.end_offset < offset:
                break
            for i in range(sz):
                if not self.is_range(offset + i):
                    continue
                class_name = self.get_class(h, i)
                yield h['pid'][i], class_name, h, i
            offset += sz

    # {클래스 이름 : 키값} y_vocab 해쉬 생성 
    # input --> ex) 'data/train'
    # output --> ex) {'133>2224>1>2' : 1232}
    def get_y_vocab(self, data_path):
        y_vocab = {}
        h = h5py.File(data_path, 'r')[self.div]
        sz = h['pid'].shape[0]
        for i in tqdm.tqdm(range(sz), mininterval=1):
            class_name = self.get_class(h, i)
            if class_name not in y_vocab:
                y_vocab[class_name] = len(y_vocab)
        return y_vocab


def preprocessing_helper(data):
    try:
        cls, data_path_list, div, out_path, begin_offset, end_offset = data
        data = cls()
        data.load_y_vocab()
        data.preprocessing(data_path_list, div, begin_offset, end_offset, out_path)
    except Exception:
        raise Exception("".join(traceback.format_exception(*sys.exc_info())))


# data경로와 종류를 받아 그 데이터들에 해당하는 y_vocab 생성
# input --> ex) ('data/train', 'train')
# output --> ex) 
def build_y_vocab_helper(data):
    try:
        data_path, div = data
        reader = Reader([], div, None, None)
        y_vocab = reader.get_y_vocab(data_path)
    except Exception:
        raise Exception("".join(traceback.format_exception(*sys.exc_info())))
    return y_vocab


class Data:
    y_vocab_path = './data/y_vocab.cPickle' if six.PY2 else './data/y_vocab.py3.cPickle'
    tmp_chunk_tpl = 'tmp/base.chunk.%s'

    def __init__(self):
        self.logger = get_logger('data')

    # 제공되는 pickle에서 카테고리 vocab(모음집) 가져옴 (pickle 모르면 검색 ㄱ)
    # y_vocab --> ex) {'14>13>235>-1': 0, '51>545>-1>-1': 1, ......}
    # '14>13>235>-1' --> 14 = 대분류, 13 = 중분류 ... 
    def load_y_vocab(self):
        self.y_vocab = cPickle.loads(open(self.y_vocab_path, 'rb').read())

    # 비동기식으로 build_y_vocab_helper 함수를 이용하여 전체 데이터의 y_vocab를 생성함
    def build_y_vocab(self):
        pool = Pool(opt.num_workers)
        try:
            rets = pool.map_async(build_y_vocab_helper,
                                  [(data_path, 'train')
                                   for data_path in opt.train_data_list]).get(99999999)
            pool.close()
            pool.join()
            y_vocab = set()
            for _y_vocab in rets:
                for k in six.iterkeys(_y_vocab):
                    y_vocab.add(k)
            self.y_vocab = {y: idx for idx, y in enumerate(y_vocab)}
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()
            raise
        self.logger.info('size of y vocab: %s' % len(self.y_vocab))
        cPickle.dump(self.y_vocab, open(self.y_vocab_path, 'wb'), 2)

    # 해당 데이터 경로에 있는 모든 데이터를 일정 크기로 나누기 위한 index 처리 (데이터를 나누는게 아니라 개수 파악)
    # input --> ex) "/data/train", "train", 20000
    # output --> ex) [(0, 20000), (20001, 40000), ..... (total_size - 20000, total_size)] 
    def _split_data(self, data_path_list, div, chunk_size):
        total = 0
        for data_path in data_path_list:
            h = h5py.File(data_path, 'r')
            sz = h[div]['pid'].shape[0]
            total += sz
        chunks = [(i, min(i + chunk_size, total))
                  for i in range(0, total, chunk_size)]
        return chunks

    # 해당되는 데이터 chunk를 읽어와서 parse_data함수로 processing 처리
    def preprocessing(self, data_path_list, div, begin_offset, end_offset, out_path):
        self.div = div
        reader = Reader(data_path_list, div, begin_offset, end_offset)
        rets = []
        for pid, label, h, i in reader.generate():
            y, x = self.parse_data(label, h, i)
            if y is None:
                continue
            rets.append((pid, y, x))
        self.logger.info('sz=%s' % (len(rets)))
        open(out_path, 'wb').write(cPickle.dumps(rets, 2))
        self.logger.info('%s ~ %s done. (size: %s)' % (begin_offset, end_offset, end_offset - begin_offset))

    # split_data로 나누어진 데이터들을 비동기적으로 preprocessing 처리함
    # input --> ex) Data(class), "data/train", "train", 20000
    # output --> ex) 47 (나누어진 데이터 chunk 갯수)
    def _preprocessing(self, cls, data_path_list, div, chunk_size):
        chunk_offsets = self._split_data(data_path_list, div, chunk_size)
        num_chunks = len(chunk_offsets)
        self.logger.info('split data into %d chunks, # of classes=%s' % (num_chunks, len(self.y_vocab)))
        pool = Pool(opt.num_workers)
        try:
            pool.map_async(preprocessing_helper, [(cls,
                                            data_path_list,
                                            div,
                                            self.tmp_chunk_tpl % cidx,
                                            begin,
                                            end)
                                           for cidx, (begin, end) in enumerate(chunk_offsets)]).get(9999999)
            pool.close()
            pool.join()
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()
            raise
        return num_chunks

    # TODO: 임베딩 하는 방식을 바꾸어 볼수 있음 (kor_char_parser preprocessing 함수 이용하여 반영하기)

    # 해당하는 label에 대한 데이터 쌍을 꺼내서 해당되는 x, y processing하고 리턴함
    # input --> ex) '14>13>235>-1', h(h5py 파일), 34(index)
    # output --> ex) 5, ([132, 22, 115, 223, ...], [22, 155, 223, 444, ...])
    # 5 = 카테고리 값, [132, 22, 115, 223, ...] = 단어들의 해쉬값(순서 많이나온 순), [22, 155, 223, 444, ...] = 앞 벡터 한칸 뒤로 민것 
    def parse_data(self, label, h, i):
        Y = self.y_vocab.get(label)
        if Y is None and self.div in ['dev', 'test']:
            Y = 0
        if Y is None and self.div != 'test':
            return [None] * 2
        Y = to_categorical(Y, len(self.y_vocab))

        product = h['product'][i]
        if six.PY3:
            product = product.decode('utf-8')
        product = re_sc.sub(' ', product).strip().split()
        words = [w.strip() for w in product]
        words = [w for w in words
                 if len(w) >= opt.min_word_length and len(w) < opt.max_word_length]
        if not words:
            return [None] * 2

        hash_func = hash if six.PY2 else lambda x: mmh3.hash(x, seed=17)
        x = [hash_func(w) % opt.unigram_hash_size + 1 for w in words]
        xv = Counter(x).most_common(opt.max_len)

        x = np.zeros(opt.max_len, dtype=np.float32)
        v = np.zeros(opt.max_len, dtype=np.int32)
        for i in range(len(xv)):
            x[i] = xv[i][0]
            v[i] = xv[i][1]
        return Y, (x, v)

    # h5py 형식의 데이터 셋 초기화 (4가지 종류의 데이터셋)
    # input --> ex) train_group, 16000, len(y_vocab) 
    # output --> None (train_group에 데이터셋 초기화)
    def create_dataset(self, g, size, num_classes):
        shape = (size, opt.max_len)
        g.create_dataset('uni', shape, chunks=True, dtype=np.int32)
        g.create_dataset('w_uni', shape, chunks=True, dtype=np.float32)
        g.create_dataset('cate', (size, num_classes), chunks=True, dtype=np.int32)
        g.create_dataset('pid', (size,), chunks=True, dtype='S12')

    def init_chunk(self, chunk_size, num_classes):
        chunk_shape = (chunk_size, opt.max_len)
        chunk = {}
        chunk['uni'] = np.zeros(shape=chunk_shape, dtype=np.int32)
        chunk['w_uni'] = np.zeros(shape=chunk_shape, dtype=np.float32)
        chunk['cate'] = np.zeros(shape=(chunk_size, num_classes), dtype=np.int32)
        chunk['pid'] = []
        chunk['num'] = 0
        return chunk

    def copy_chunk(self, dataset, chunk, offset, with_pid_field=False):
        num = chunk['num']
        dataset['uni'][offset:offset + num, :] = chunk['uni'][:num]
        dataset['w_uni'][offset:offset + num, :] = chunk['w_uni'][:num]
        dataset['cate'][offset:offset + num] = chunk['cate'][:num]
        if with_pid_field:
            dataset['pid'][offset:offset + num] = chunk['pid'][:num]

    def copy_bulk(self, A, B, offset, y_offset, with_pid_field=False):
        num = B['cate'].shape[0]
        y_num = B['cate'].shape[1]
        A['uni'][offset:offset + num, :] = B['uni'][:num]
        A['w_uni'][offset:offset + num, :] = B['w_uni'][:num]
        A['cate'][offset:offset + num, y_offset:y_offset + y_num] = B['cate'][:num]
        if with_pid_field:
            A['pid'][offset:offset + num] = B['pid'][:num]

    # train_ratio만큼의 데이터를 뽑기위해 train_indices 랜덤으로 설정하고 그 크기를 반환
    # input --> ex) 20000, 0.8
    # output --> ex) [True, False, True, True, False ...], 16000
    def get_train_indices(self, size, train_ratio):
        train_indices = np.random.rand(size) < train_ratio
        train_size = int(np.count_nonzero(train_indices))
        return train_indices, train_size


    # 데이터들을 가져와서 학습 및 테스트용 데이터 베이스를 생성 
    # input --> ex) train (db 이름)
    # output --> ex) dataset (hash)
    # dataset['uni'] == 단어 임베딩 --> ex) [132, 22, 115, 223, ...]
    # dataset['w_uni'] == 단어 임베딩 --> ex) [22, 115, 223, 444, ...]
    # dataset['cate'] == 카테고리 종류 --> ex) [0, 0, 0, 1, 0, 0, 0, ...]
    def make_db(self, data_name, output_dir='/media/kakao/Kakao-arena/data/train', train_ratio=0.8):
        if data_name == 'train':
            div = 'train'
            data_path_list = opt.train_data_list
        elif data_name == 'dev':
            div = 'dev'
            data_path_list = opt.dev_data_list
        elif data_name == 'test':
            div = 'test'
            data_path_list = opt.test_data_list
        else:
            assert False, '%s is not valid data name' % data_name

        all_train = train_ratio >= 1.0
        all_dev = train_ratio == 0.0

        np.random.seed(17)
        self.logger.info('make database from data(%s) with train_ratio(%s)' % (data_name, train_ratio))

        self.load_y_vocab()
        num_input_chunks = self._preprocessing(Data,
                                               data_path_list,
                                               div,
                                               chunk_size=opt.chunk_size)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        data_fout = h5py.File(os.path.join(output_dir, 'data.h5py'), 'w')
        meta_fout = open(os.path.join(output_dir, 'meta'), 'wb')

        reader = Reader(data_path_list, div, None, None)
        tmp_size = reader.get_size()
        train_indices, train_size = self.get_train_indices(tmp_size, train_ratio)

        dev_size = tmp_size - train_size
        if all_dev:
            train_size = 1
            dev_size = tmp_size
        if all_train:
            dev_size = 1
            train_size = tmp_size

        train = data_fout.create_group('train')
        dev = data_fout.create_group('dev')
        self.create_dataset(train, train_size, len(self.y_vocab))
        self.create_dataset(dev, dev_size, len(self.y_vocab))
        self.logger.info('train_size ~ %s, dev_size ~ %s' % (train_size, dev_size))

        sample_idx = 0
        dataset = {'train': train, 'dev': dev}
        num_samples = {'train': 0, 'dev': 0}
        chunk_size = opt.db_chunk_size
        chunk = {'train': self.init_chunk(chunk_size, len(self.y_vocab)),
                 'dev': self.init_chunk(chunk_size, len(self.y_vocab))}
        chunk_order = list(range(num_input_chunks))
        np.random.shuffle(chunk_order)
        for input_chunk_idx in chunk_order:
            path = os.path.join(self.tmp_chunk_tpl % input_chunk_idx)
            self.logger.info('processing %s ...' % path)
            data = list(enumerate(cPickle.loads(open(path, 'rb').read())))
            np.random.shuffle(data)
            for data_idx, (pid, y, vw) in data:
                if y is None:
                    continue
                v, w = vw
                is_train = train_indices[sample_idx + data_idx]
                if all_dev:
                    is_train = False
                if all_train:
                    is_train = True
                if v is None:
                    continue
                c = chunk['train'] if is_train else chunk['dev']
                idx = c['num']
                c['uni'][idx] = v # 단어 임베딩_1 (parse_data 함수 참고)
                c['w_uni'][idx] = w # 단어 임베딩_2 (parse_data 함수 참고)
                c['cate'][idx] = y # 카테고리 y값 ex) [0,0,0,1,0,0,0,0,....]
                c['num'] += 1 # 
                if not is_train:
                    c['pid'].append(np.string_(pid))
                for t in ['train', 'dev']:
                    if chunk[t]['num'] >= chunk_size:
                        self.copy_chunk(dataset[t], chunk[t], num_samples[t],
                                        with_pid_field=t == 'dev')
                        num_samples[t] += chunk[t]['num']
                        chunk[t] = self.init_chunk(chunk_size, len(self.y_vocab))
            sample_idx += len(data)
        for t in ['train', 'dev']:
            if chunk[t]['num'] > 0:
                self.copy_chunk(dataset[t], chunk[t], num_samples[t],
                                with_pid_field=t == 'dev')
                num_samples[t] += chunk[t]['num']

        for div in ['train', 'dev']:
            ds = dataset[div]
            size = num_samples[div]
            shape = (size, opt.max_len)
            ds['uni'].resize(shape)
            ds['w_uni'].resize(shape)
            ds['cate'].resize((size, len(self.y_vocab)))

        data_fout.close()
        meta = {'y_vocab': self.y_vocab}
        meta_fout.write(cPickle.dumps(meta, 2))
        meta_fout.close()

        self.logger.info('# of classes: %s' % len(meta['y_vocab']))
        self.logger.info('# of samples on train: %s' % num_samples['train'])
        self.logger.info('# of samples on dev: %s' % num_samples['dev'])
        self.logger.info('data: %s' % os.path.join(output_dir, 'data.h5py'))
        self.logger.info('meta: %s' % os.path.join(output_dir, 'meta'))


if __name__ == '__main__':
    data = Data()
    # cli 에서 data.make_db 함수를 쓸수 있게 함 (google python-fire 참고)
    fire.Fire({'make_db': data.make_db,
               'build_y_vocab': data.build_y_vocab})
