#Generating News Headlines with Reccurent Neural Networks
#https://arxiv.org/abs/1512.01712
#データ読み込み
import json
import os
from ast import literal_eval
import re
import MeCab
import unicodedata
import sys
#import ngram
#import jcconv
import numpy as np
import argparse
import copy
import chainer
from chainer import cuda
#from normalizer import normalize,_convert_marks,_delete_cyclic_word
#from impala.dbapi import connect


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
args = parser.parse_args()

#xp = cuda.cupy if args.gpu >= 0 else np
args.gpu = 1
if args.gpu >= 0:
	xp = cuda.cupy
else:
	xp = np

#f = open("train1000.en")]
"""
f = open("yahoofinance/alltext0525.txt")
#f = open("train5000.en")
#englishdata = f.read()
yahooboarddata = f.read()
f.close()
#englishsentencset = englishdata.split("\n")
yahooboarddataset = yahooboarddata.split("\n")
yahooboarddataset = np.array(yahooboarddataset)
smallyahooboarddataset = yahooboarddataset[range(0,len(yahooboarddataset),2)]
smallsentencelist = []

vocablist = []
for sentences in smallyahooboarddataset:
	sentence = sentences.split(" ")
	smallsentencelist.append(sentence)
	vocablist += sentence
"""

import pickle 
#pickle.dump(sentenceslist_file, open("lyric/sentenceslist_file.dump", 'wb'), -1)
sentenceslist_file = pickle.load(open("lyric/sentenceslist_file.dump"))
vocablist = []
for sentences in sentenceslist_file:
	for sentence_id in sentences:
		for sentence in sentences[sentence_id]:
			for word in sentence:
				vocablist.append(word)

vocablist = list(set(vocablist))
vocabIDdic = dict(zip(vocablist,range(len(vocablist))))
vocabworddic = dict(zip(vocabIDdic.values(),vocabIDdic.keys()))


import gensim
from gensim import corpora, models, similarities
import os



#http://qiita.com/odashi_t/items/a1be7c4964fbea6a116e
from chainer import FunctionSet
from chainer.functions import *

import numpy as np
from chainer import Variable
from chainer.functions import *
from chainer.optimizers import *

SRC_VOCAB_SIZE = (len(vocablist) + 3)
SRC_EMBED_SIZE  = 200
HIDDEN_SIZE = 100
HIDDEN_SIZE2 = 50
TRG_VOCAB_SIZE = (len(vocablist) + 3)
TRG_EMBED_SIZE = 200
"""
model = FunctionSet(
	#encode
	w_xi = EmbedID(SRC_VOCAB_SIZE, SRC_EMBED_SIZE), # 入力層(one-hot) -> 入力埋め込み層
	w_ip = Linear(SRC_EMBED_SIZE, 4 * HIDDEN_SIZE), # 入力埋め込み層 -> 入力隠れ層
	w_iP = Linear(SRC_EMBED_SIZE, 4 * HIDDEN_SIZE2), # 入力埋め込み層 -> 入力attention隠れ層
	w_pp = Linear(HIDDEN_SIZE, 4 * HIDDEN_SIZE), # 入力隠れ層 -> 入力隠れ層
	w_PP = Linear(HIDDEN_SIZE2, 4 * HIDDEN_SIZE2), # 入力隠れ層 -> 入力隠れ層
	w_pq = Linear(HIDDEN_SIZE, 4 * HIDDEN_SIZE), # 入力隠れ層 -> 出力隠れ層
	#w_yq = EmbedID(TRG_VOCAB_SIZE, 4 * HIDDEN_SIZE), # 出力層(one-hot) -> 出力隠れ層
	# attentional weight estimator
	w_pw = Linear(HIDDEN_SIZE2, HIDDEN_SIZE),
	w_qw = Linear(HIDDEN_SIZE, HIDDEN_SIZE),
	w_we = Linear(HIDDEN_SIZE, 1),
	w_cp = Linear(HIDDEN_SIZE2, 4 * HIDDEN_SIZE),
	#decode
	w_iq = Linear(TRG_EMBED_SIZE, 4 * HIDDEN_SIZE), # 出力層(one-hot) -> 出力隠れ層
	w_qq = Linear(HIDDEN_SIZE, 4 * HIDDEN_SIZE), # 出力隠れ層 -> 出力隠れ層
	w_qj = Linear(HIDDEN_SIZE, TRG_EMBED_SIZE), # 出力隠れ層 -> 出力埋め込み層
	w_jy = Linear(TRG_EMBED_SIZE, TRG_VOCAB_SIZE), # 出力隠れ層 -> 出力隠れ層
)  
"""

import chainer
import chainer.functions as F
import chainer.links as L

from __future__ import print_function
import argparse
import math
import sys
import time

import numpy as np
import six

import six.moves.cPickle as pickle

import chainer
from chainer import cuda
import chainer.links as L
from chainer import optimizers
from chainer import serializers

class Encoder(chainer.Chain):

    """Recurrent neural net languabe model for penn tree bank corpus.
    This is an example of deep LSTM network for infinite length input.
    """
    def __init__(self, SRC_VOCAB_SIZE,SRC_EMBED_SIZE, HIDDEN_SIZE, HIDDEN_SIZE2,train=True):
        super(RNNLM, self).__init__(
            w_xi = L.EmbedID(SRC_VOCAB_SIZE, SRC_EMBED_SIZE), # 入力層(one-hot) -> 入力埋め込み層
            w_ip = L.LSTM(SRC_EMBED_SIZE, HIDDEN_SIZE), # 入力埋め込み層 -> 入力隠れ層
            w_iP = L.LSTM(SRC_EMBED_SIZE, HIDDEN_SIZE2), # 入力埋め込み層 -> 入力attention隠れ層
			)
        self.train = train

    def reset_state(self):
        self.w_ip.reset_state()
        self.w_iP.reset_state()
        self.w_pq.reset_state()
    def __call__(self, x):
        i = self.w_xi(x)
		p = self.w_ip(i)
		P = self.w_iP(i)
		return P,p

class Attention(Chain):
  def __init__(self, HIDDEN_SIZE2,HIDDEN_SIZE):
    super(Attention, self).__init__(
        w_pw = Linear(HIDDEN_SIZE2, HIDDEN_SIZE),
        w_qw = Linear(HIDDEN_SIZE, HIDDEN_SIZE),
        w_we = Linear(HIDDEN_SIZE, 1),
    )
    self.HIDDEN_SIZE2 = HIDDEN_SIZE2
    self.HIDDEN_SIZE = HIDDEN_SIZE
  def __call__(self, P_all, Pw_list,q): #p:h
    list_e = []
    sum_e = chainer.Variable(xp.zeros((1,1), dtype=np.float32))
    for P_w in Pw_list:
    	v_i_plus = model.w_we(tanh(model.w_qw(q))) #v_i: scalar
    	exp_v_i = exp(v_i_plus + P_w)
    	list_e.append(exp_v_i)
    	sum_e += exp_v_i
    # make attention vector
    for n, e in enumerate(list_e):
    	if n == 0:
    		e_all = (e / sum_e)
    	else:
    		e_all = Variable_hstack(e_all, e / sum_e)
	m_t = matmul(e_all,P_all)
    return m_t,q

class Decoder(chainer.Chain):

    """Recurrent neural net languabe model for penn tree bank corpus.
    This is an example of deep LSTM network for infinite length input.
    """
    def __init__(self, TRG_VOCAB_SIZE, TRG_EMBED_SIZE,HIDDEN_SIZE2, HIDDEN_SIZE,train=True):
        super(RNNLM, self).__init__(
        	w_cp = L.Linear(HIDDEN_SIZE2, 4 * HIDDEN_SIZE),
            w_jq = L.Linear(TRG_EMBED_SIZE, HIDDEN_SIZE), # 出力層(one-hot) -> 出力隠れ層
            w_qq = L.Linear(HIDDEN_SIZE, 4 * HIDDEN_SIZE),
            w_qj = L.Linear(HIDDEN_SIZE, TRG_EMBED_SIZE), # 出力隠れ層 -> 出力隠れ層
			w_jy = L.Linear(TRG_EMBED_SIZE, TRG_VOCAB_SIZE)
			w_yj = L.EmbedID(TRG_VOCAB_SIZE, TRG_EMBED_SIZE)
			)
        self.train = train
    def __call__(self, q,c, y,m_t):
    	e = functions.tanh(self.w_yj(y))
        c, q = functions.lstm(c, self.w_jq(e) + self.w_qq(q) + self.w_cp(m_t))
		j_out = functions.tanh(self.w_qj(q))
		return self.w_jy(j_out), c, q



class AttentionMT(Chain):
	def __init__(self, SRC_VOCAB_SIZE,SRC_EMBED_SIZE, HIDDEN_SIZE, HIDDEN_SIZE2):
		super(AttentionMT, self).__init__(
		enc = Encoder(SRC_VOCAB_SIZE,SRC_EMBED_SIZE, HIDDEN_SIZE, HIDDEN_SIZE2), #foward
		att = Attention(HIDDEN_SIZE2,HIDDEN_SIZE),
		w_pq = L.LSTM(HIDDEN_SIZE, HIDDEN_SIZE), # 入力隠れ層 -> 出力隠れ層
		dec = Decoder(TRG_VOCAB_SIZE, TRG_EMBED_SIZE,HIDDEN_SIZE2, HIDDEN_SIZE),
		)
		self.SRC_VOCAB_SIZE = SRC_VOCAB_SIZE
		self.SRC_EMBED_SIZE = SRC_EMBED_SIZE
		self.HIDDEN_SIZE = HIDDEN_SIZE
		self.HIDDEN_SIZE2 = HIDDEN_SIZE2
	def __call__(self, x_list, y_list):
		Pw_list  = []
		for i,x in enumerate(x_list):
			P,p = self.enc(x)
			Pw_list.append(model.w_we(model.w_pw(P)))
			if i == 0:
				P_all = P
			else:
				P_all = Variable_vstack(P_all,P)
		q = self.w_pq(p)
		t_list = []
		for y in y_list:
			m_t,q = self.att(P_all,Pw_list,q)
			t,c, q = self.dec(q,c, y,m_t)
			t_list.append(t)
		return t_list


args = parser.parse_args()
xp = cuda.cupy if args.gpu >= 0 else np

n_epoch = args.epoch   # number of epochs
n_units = args.unit  # number of units per layer
batchsize = args.batchsize   # minibatch size
bprop_len = args.bproplen   # length of truncated BPTT
grad_clip = args.gradclip    # gradient norm threshold to clip


lm = RNNLM(len(vocab), n_units)
encdecatt_simple = AttentionMT(SRC_VOCAB_SIZE,SRC_EMBED_SIZE, HIDDEN_SIZE, HIDDEN_SIZE2)
#model = L.Classifier(lm)
model = L.Classifier(encdecatt_simple)
model.compute_accuracy = False  # we only want the perplexity
for param in model.params():
    data = param.data
    data[:] = np.random.uniform(-0.1, 0.1, data.shape)
if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()


optimizer = optimizers.SGD(lr=1.)
optimizer.setup(model)
optimizer.add_hook(chainer.optimizer.GradientClipping(grad_clip))

# Init/Resume
if args.initmodel:
    print('Load model from', args.initmodel)
    serializers.load_npz(args.initmodel, model)
if args.resume:
    print('Load optimizer state from', args.resume)
    serializers.load_npz(args.resume, optimizer)


def evaluate(dataset):
    # Evaluation routine
    evaluator = model.copy()  # to use different state
    evaluator.predictor.reset_state()  # initialize state
    evaluator.predictor.train = False  # dropout does nothing

    sum_log_perp = 0
    for i in six.moves.range(dataset.size - 1):
        x = chainer.Variable(xp.asarray(dataset[i:i + 1]), volatile='on')
        t = chainer.Variable(xp.asarray(dataset[i + 1:i + 2]), volatile='on')
        loss = evaluator(x, t)
        sum_log_perp += loss.data
    return math.exp(float(sum_log_perp) / (dataset.size - 1))


# Learning loop
whole_len = train_data.shape[0]
jump = whole_len // batchsize
cur_log_perp = xp.zeros(())
epoch = 0
start_at = time.time()
cur_at = start_at
accum_loss = 0
batch_idxs = list(range(batchsize))
print('going to train {} iterations'.format(jump * n_epoch))

for i in six.moves.range(jump * n_epoch):
    x = chainer.Variable(xp.asarray(
        [train_data[(jump * j + i) % whole_len] for j in batch_idxs]))
    t = chainer.Variable(xp.asarray(
        [train_data[(jump * j + i + 1) % whole_len] for j in batch_idxs]))
    loss_i = model(x, t)
    accum_loss += loss_i
    cur_log_perp += loss_i.data

    if (i + 1) % bprop_len == 0:  # Run truncated BPTT
        model.zerograds()
        accum_loss.backward()
        accum_loss.unchain_backward()  # truncate
        accum_loss = 0
        optimizer.update()

    if (i + 1) % 10000 == 0:
        now = time.time()
        throuput = 10000. / (now - cur_at)
        perp = math.exp(float(cur_log_perp) / 10000)
        print('iter {} training perplexity: {:.2f} ({:.2f} iters/sec)'.format(
            i + 1, perp, throuput))
        cur_at = now
        cur_log_perp.fill(0)

    if (i + 1) % jump == 0:
        epoch += 1
        print('evaluate')
        now = time.time()
        perp = evaluate(valid_data)
        print('epoch {} validation perplexity: {:.2f}'.format(epoch, perp))
        cur_at += time.time() - now  # skip time of evaluation

        if epoch >= 6:
            optimizer.lr /= 1.2
            print('learning rate =', optimizer.lr)

    sys.stdout.flush()

# Evaluate on test dataset
print('test')
test_perp = evaluate(test_data)
print('test perplexity:', test_perp)

# Save the model and the optimizer
print('save the model')
serializers.save_npz('rnnlm.model', model)
print('save the optimizer')
serializers.save_npz('rnnlm.state', optimizer)






END_OF_SENTENCE = len(vocablist)
END_OF_SENTENCES = (len(vocablist) + 1)
import copy
sentenceslist = []
k  = 0
for j,sentences in enumerate(copy.deepcopy(sentenceslist_file)):
	if j == 100:
		print (k)
	k += len(sentences)
	for sentence_id in sentences:
		sentence_block = []
		for sentence in sentences[sentence_id]:
			sentence.append("End_sentence")
			sentence_block += sentence
		sentenceslist.append(sentence_block)

vocabIDdic["End_sentence"] = END_OF_SENTENCE
vocabworddic[END_OF_SENTENCE] = "End_sentence"

def Variable_vstack(v1,v2):
	#v1, v2: Variable(cp)
	stack_zero_1 = xp.zeros((v2.data.shape[0],v1.data.shape[0]), dtype=np.float32)
	stack_zero_2 = xp.zeros((v1.data.shape[0],v2.data.shape[0]), dtype=np.float32)
	stack_eye_1 = xp.eye(v1.data.shape[0], dtype=np.float32) 
	stack_eye_2 = xp.eye(v2.data.shape[0], dtype=np.float32)
	V_1 = chainer.Variable(xp.vstack([stack_eye_1,stack_zero_1]))
	V_2 = chainer.Variable(xp.vstack([stack_zero_2,stack_eye_2]))
	V = (matmul(V_1,v1) + matmul(V_2,v2))
	return V

"""
stack_zero_1 = xp.zeros((v2.data.shape[0],v1.data.shape[0]), dtype=np.float32)
stack_zero_2 = xp.zeros((v1.data.shape[0],v2.data.shape[0]), dtype=np.float32)
stack_eye_1 = xp.eye(v1.data.shape[0], dtype=np.float32) 
stack_eye_2 = xp.eye(v2.data.shape[0], dtype=np.float32)
V_1 = chainer.Variable(xp.vstack([stack_eye_1,stack_zero_1]))
V_2 = chainer.Variable(xp.vstack([stack_zero_2,stack_eye_2]))
V = (matmul(V_1,v1) + matmul(V_2,v2))
"""
def Variable_hstack(v1,v2):
	#v1, v2: Variable(cp)
	stack_zero_1 = xp.zeros((v1.data.shape[1],v2.data.shape[1]), dtype=np.float32) 
	stack_zero_2 = xp.zeros((v2.data.shape[1],v1.data.shape[1]), dtype=np.float32)
	stack_eye_1 = xp.eye(v1.data.shape[1], dtype=np.float32) 
	stack_eye_2 = xp.eye(v2.data.shape[1], dtype=np.float32)
	V_1 = chainer.Variable(xp.hstack([stack_eye_1,stack_zero_1]))
	V_2 = chainer.Variable(xp.hstack([stack_zero_2,stack_eye_2]))
	V = (matmul(v1,V_1) + matmul(v2,V_2))
	return V


v1 = chainer.Variable(xp.ones((1,2),dtype=np.float32))
v2 = chainer.Variable(xp.ones((3,2),dtype=np.float32))
Variable_vstack(v1,v2)

v1 = chainer.Variable(xp.ones((2,1),dtype=np.float32))
v2 = chainer.Variable(xp.ones((2,3),dtype=np.float32))
Variable_hstack(v1,v2)

# src_sentence: 翻訳したい単語列 e.g. ['彼', 'は', '走る']
# trg_sentence: 正解の翻訳を表す単語列 e.g. ['he', 'runs']
# training: 学習か予測か。デコーダの挙動に影響する。


state = {name: chainer.Variable(xp.zeros((1, HIDDEN_SIZE),
	dtype=np.float32),
	volatile=not train)
	for name in ('c1', 'c3', 'c4')}

state['c2'] = chainer.Variable(xp.zeros((1, HIDDEN_SIZE2),dtype=np.float32))

def forward(src_sentence, trg_sentence, model, training,state):
	# 単語IDへの変換（自分で適当に実装する）
	# 正解の翻訳には終端記号を追加しておく。
	src_sentence2 = []
	for word in src_sentence:
		src_sentence2.append(vocabIDdic[word])
	#src_sentence = [ japaneseIDdic[word.decode("utf-8")] for word in src_sentence]
	#trg_sentence = [ englishIDdic[word] for word in trg_sentence] + [END_OF_SENTENCE]
	src_sentence = src_sentence2
	trg_sentence2 = []
	for word in trg_sentence:
		trg_sentence2.append(vocabIDdic[word])
	#trg_sentence2 = (trg_sentence2 + [END_OF_SENTENCE])
	trg_sentence2 = (trg_sentence2 + [END_OF_SENTENCES])
	trg_sentence = trg_sentence2
	#print trg_sentence
	# LSTM内部状態の初期値
	c1 = state["c1"]
	c2 = state["c2"]
	c3 = state["c3"]
	c4 = state["c4"]
	#P_list = []
	Pw_list = []
	# エンコーダ
	x = Variable(xp.array([END_OF_SENTENCES], dtype=np.int32))
	i = tanh(model.w_xi(x))
	c1, p = lstm(c1, model.w_ip(i))
	c2, P = lstm(c2, model.w_iP(i))
	#P_list.append(P)
	P_all = P
	Pw_list.append(model.w_we(tanh(model.w_pw(P))))
	for word in reversed(src_sentence):
		x = Variable(xp.array([word], dtype=np.int32))
		i = tanh(model.w_xi(x))
		c1, p = lstm(c1, model.w_ip(i) + model.w_pp(p))
		c2, P = lstm(c2, model.w_iP(i) + + model.w_PP(P))
		#P_list.append(P)
		#print P_all.data.shape, P.data.shape
		#return P_all, P
		P_all = Variable_vstack(P_all,P)
		Pw_list.append(model.w_we(model.w_pw(P)))
		#print P_all.data.shape
	# エンコーダ -> デコーダ
	c3, q = lstm(c3, model.w_pq(p))
	t = Variable(xp.array([END_OF_SENTENCE], dtype=np.int32))
	# デコーダ
	if training:
		# 学習時はyとして正解の翻訳を使い、forwardの結果として累積損失を返す。
		accum_loss = Variable(xp.zeros((), dtype=np.float32))
		for word in trg_sentence:
			#attention
			list_e = []
			sum_e = chainer.Variable(xp.zeros((1,1), dtype=np.float32))
			#for P in P_list:
			for P_w in Pw_list:
				#v_i = model.w_we(tanh(model.w_qw(q) + model.w_pw(P))) #v_i: scalar
				v_i_plus = model.w_we(tanh(model.w_qw(q))) #v_i: scalar
				exp_v_i = exp(v_i_plus + P_w)
				list_e.append(exp_v_i)
				sum_e += exp_v_i
			# make attention vector
			for n, e in enumerate(list_e):
				if n == 0:
					e_all = (e / sum_e)
				else:
					e_all = Variable_hstack(e_all, e / sum_e)
			#print e_all.data.shape, P_all.data.shape
			m_t = matmul(e_all,P_all)
			#m_t = Variable(xp.zeros((1, HIDDEN_SIZE2),dtype=np.float32))
			#for n in range(len(P_list)):
				#a_i = list_e[n] / sum_e
				#m_t += matmul(a_i ,P_list[n])
			c4, q = lstm(c4, model.w_iq(model.w_xi(t)) + model.w_qq(q) + model.w_cp(m_t))
			j = tanh(model.w_qj(q))
			y = model.w_jy(j)
			t = Variable(xp.array([word], dtype=np.int32))
			loss = softmax_cross_entropy(y, t)
			accum_loss += softmax_cross_entropy(y, t)
		state = {'c1': c1, 'c2': c2, 'c3': c3, 'c4': c4}
		return accum_loss, state
	else:
	# 予測時には翻訳器が生成したyを次回の入力に使い、forwardの結果として生成された単語列を返す。
	# yの中で最大の確率を持つ単語を選択していくが、softmaxを取る必要はない。
		hyp_sentence = []
		while len(hyp_sentence) < 100: # 100単語以上は生成しないようにする
			#attention
			list_e = []
			sum_e = chainer.Variable(xp.zeros((1,1), dtype=np.float32))
			#for P in P_list:
			for P_w in Pw_list:
				#v_i = model.w_we(tanh(model.w_qw(q) + model.w_pw(P))) #v_i: scalar
				v_i_plus = model.w_we(tanh(model.w_qw(q))) #v_i: scalar
				exp_v_i = exp(v_i_plus + P_w)
				list_e.append(exp_v_i)
				sum_e += exp_v_i
			# make attention vector
			for n, e in enumerate(list_e):
				if n == 0:
					e_all = (e / sum_e)
				else:
					e_all = Variable_hstack(e_all, e / sum_e)
			#m_t = Variable(xp.zeros((1, HIDDEN_SIZE2),dtype=np.float32))
			#for n in range(len(P_list)):
				#a_i = list_e[n] / sum_e
				#m_t += matmul(a_i ,P_list[n])
			m_t = matmul(e_all,P_all)
			c4, q = lstm(c4, model.w_iq(model.w_xi(t)) + model.w_qq(q) + model.w_cp(m_t))
			j = tanh(model.w_qj(q))
			y = model.w_jy(j)
			word = int(y.data.argmax(1)[0])
			t = Variable(xp.array([word], dtype=np.int32))
			#print word
			if word == END_OF_SENTENCES:
				hyp_sentence.append("unk" + str(END_OF_SENTENCES))
				break # 終端記号が生成されたので終了
			hyp_sentence.append(vocabworddic[int(word)])
			#print word
		return hyp_sentence


def forward2(src_sentence, trg_sentence, model, training):
	# 単語IDへの変換（自分で適当に実装する）
	# 正解の翻訳には終端記号を追加しておく。
	src_sentence2 = []
	for word in src_sentence:
		src_sentence2.append(vocabIDdic[word])
	#src_sentence = [ japaneseIDdic[word.decode("utf-8")] for word in src_sentence]
	#trg_sentence = [ englishIDdic[word] for word in trg_sentence] + [END_OF_SENTENCE]
	src_sentence = src_sentence2
	trg_sentence2 = []
	for word in trg_sentence:
		trg_sentence2.append(vocabIDdic[word])
	#trg_sentence2 = (trg_sentence2 + [END_OF_SENTENCE])
	trg_sentence2 = (trg_sentence2 + [END_OF_SENTENCES])
	trg_sentence = trg_sentence2
	#print trg_sentence
	# LSTM内部状態の初期値
	c = chainer.Variable(xp.zeros((1, HIDDEN_SIZE),dtype=np.float32))
	c2 = chainer.Variable(xp.zeros((1, HIDDEN_SIZE2),dtype=np.float32))
	#P_list = []
	Pw_list = []
	# エンコーダ
	x = Variable(xp.array([END_OF_SENTENCES], dtype=np.int32))
	i = tanh(model.w_xi(x))
	c, p = lstm(c, model.w_ip(i))
	c2, P = lstm(c2, model.w_iP(i))
	#P_list.append(P)
	P_all = P
	Pw_list.append(model.w_we(tanh(model.w_pw(P))))
	for word in reversed(src_sentence):
		x = Variable(xp.array([word], dtype=np.int32))
		i = tanh(model.w_xi(x))
		c, p = lstm(c, model.w_ip(i) + model.w_pp(p))
		c2, P = lstm(c2, model.w_iP(i) + model.w_PP(P))
		#P_list.append(P)
		#print P_all.data.shape, P.data.shape
		#return P_all, P
		P_all = Variable_vstack(P_all,P)
		Pw_list.append(model.w_we(model.w_pw(P)))
		#print P_all.data.shape
	# エンコーダ -> デコーダ
	c, q = lstm(c, model.w_pq(p))
	t = Variable(xp.array([END_OF_SENTENCE], dtype=np.int32))
	# デコーダ
	if training:
		# 学習時はyとして正解の翻訳を使い、forwardの結果として累積損失を返す。
		accum_loss = Variable(xp.zeros((), dtype=np.float32))
		for word in trg_sentence:
			#attention
			list_e = []
			sum_e = chainer.Variable(xp.zeros((1,1), dtype=np.float32))
			#for P in P_list:
			for P_w in Pw_list:
				#v_i = model.w_we(tanh(model.w_qw(q) + model.w_pw(P))) #v_i: scalar
				v_i_plus = model.w_we(tanh(model.w_qw(q))) #v_i: scalar
				exp_v_i = exp(v_i_plus + P_w)
				list_e.append(exp_v_i)
				sum_e += exp_v_i
			# make attention vector
			for n, e in enumerate(list_e):
				if n == 0:
					e_all = (e / sum_e)
				else:
					e_all = Variable_hstack(e_all, e / sum_e)
			#print e_all.data.shape, P_all.data.shape
			m_t = matmul(e_all,P_all)
			#m_t = Variable(xp.zeros((1, HIDDEN_SIZE2),dtype=np.float32))
			#for n in range(len(P_list)):
				#a_i = list_e[n] / sum_e
				#m_t += matmul(a_i ,P_list[n])
			c, q = lstm(c, model.w_iq(model.w_xi(t)) + model.w_qq(q) + model.w_cp(m_t))
			j = tanh(model.w_qj(q))
			y = model.w_jy(j)
			t = Variable(xp.array([word], dtype=np.int32))
			loss = softmax_cross_entropy(y, t)
			accum_loss += softmax_cross_entropy(y, t)
		return accum_loss
	else:
	# 予測時には翻訳器が生成したyを次回の入力に使い、forwardの結果として生成された単語列を返す。
	# yの中で最大の確率を持つ単語を選択していくが、softmaxを取る必要はない。
		hyp_sentence = []
		while len(hyp_sentence) < 100: # 100単語以上は生成しないようにする
			#attention
			list_e = []
			sum_e = chainer.Variable(xp.zeros((1,1), dtype=np.float32))
			#for P in P_list:
			for P_w in Pw_list:
				#v_i = model.w_we(tanh(model.w_qw(q) + model.w_pw(P))) #v_i: scalar
				v_i_plus = model.w_we(tanh(model.w_qw(q))) #v_i: scalar
				exp_v_i = exp(v_i_plus + P_w)
				list_e.append(exp_v_i)
				sum_e += exp_v_i
			# make attention vector
			for n, e in enumerate(list_e):
				if n == 0:
					e_all = (e / sum_e)
				else:
					e_all = Variable_hstack(e_all, e / sum_e)
			#m_t = Variable(xp.zeros((1, HIDDEN_SIZE2),dtype=np.float32))
			#for n in range(len(P_list)):
				#a_i = list_e[n] / sum_e
				#m_t += matmul(a_i ,P_list[n])
			m_t = matmul(e_all,P_all)
			c, q = lstm(c, model.w_iq(model.w_xi(t)) + model.w_qq(q) + model.w_cp(m_t))
			j = tanh(model.w_qj(q))
			y = model.w_jy(j)
			word = int(y.data.argmax(1)[0])
			t = Variable(xp.array([word], dtype=np.int32))
			#print word
			if word == END_OF_SENTENCES:
				hyp_sentence.append("unk" + str(END_OF_SENTENCES))
				break # 終端記号が生成されたので終了
			hyp_sentence.append(vocabworddic[int(word)])
			#print word
		return hyp_sentence


opt = Adam()
opt.setup(model)
for textID in range(0,1000):
	opt.zero_grads()
	#accum_loss, state = forward(sentenceslist[textID], sentenceslist[textID], model, training = True,state = state) # 損失の計算
	accum_loss = forward2(sentenceslist[textID], sentenceslist[textID], model, training = True) # 損失の計算
	accum_loss.backward()
	accum_loss.unchain_backward()
	opt.update()
	print textID, accum_loss.data

forward(sentenceslist[0], sentenceslist[0], model, training = True, state = state)
forward2(sentenceslist[0], sentenceslist[0], model, training = True)

import time
def train(japansentencsetdoc,englishsentencsetdoc,model,N):
	#opt = SGD() # 確率的勾配法を使用
	opt = Adam()
	opt.setup(model) # 学習器の初期化
	#for sentence in sentence_set:
	accum_loss_sum = Variable(xp.zeros((), dtype=np.float32))
	first_time = time.time()
	for textID in range(N):
		if textID % 20 == 1:
			print "textID: ", textID, "time", (time.time() - first_time)
			print accum_loss_sum.data, accum_loss.data
		opt.zero_grads() # 勾配の初期化
		accum_loss,state = forward(japansentencsetdoc[textID], englishsentencsetdoc[textID], model, training = True,state = state) # 損失の計算
		print textID
		accum_loss.backward() # 誤差逆伝播
		accum_loss_sum += accum_loss
		#opt.clip_grads(10) # 大きすぎる勾配を抑制
		opt.update() # パラメータの更新
	print accum_loss_sum.data

def Test(n):
	text = ""
	hyp_sentence = forward(japantest[n],englishtest[n],model, training = False)
	for w in japantest[n]:
			text = text + w
	print "=====問題======"
	print text
	print "=====正解======"
	print englishtest[n]
	print "=====予測======"
	print hyp_sentence

def test(sentence):
	tagger = MeCab.Tagger( '-Owakati -u /usr/local/Cellar/mecab/0.996/lib/mecab/dic/ipadic/wikipedia-keyword.dic')
	wordarrays = tagger.parse(sentence).split(" ")[0:-1]
	#wordarrays = sentence
	print wordarrays
	print (wordarrays == japantest[0])
	hyp_sentence = forward(wordarrays,['did', 'you', 'clean', 'your', 'room', '?'],model,training = False)
	print hyp_sentence


for i in range(0,200):
	print i
	train(sentenceslist,sentenceslist,model,1067)
	hyp_sentence = forward(sentenceslist[0],sentenceslist[0],model, training = False)
	text = ""
	for w in sentenceslist[0]:
		text = text + w
	print "=====問題======"
	print text
	print "=====予測======"
	print ' '.join(hyp_sentence)

for i in range(20):
	hyp_sentence = forward(smallsentencelist[i],smallsentencelist[i],model, training = False)
	text = ""
	for w in smallsentencelist[i]:
		text = text + w
	print "=====問題======"
	print text
	print "=====予測======"
	print ' '.join(hyp_sentence)

# Save final model
import pickle
model.to_cpu()
pickle.dump(model, open("yahoofinance/alltext0525_s2s_model.dump", 'wb'), -1)
pickle.dump(japaneseIDdic,open("japaneseIDdic.dump", 'wb'), -1)
pickle.dump(englishIDdic,open("englishIDdic.dump", 'wb'), -1)
pickle.dump(unfiltered2,open("unfiltered2.dump", 'wb'), -1)

model.to_gpu()

