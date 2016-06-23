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
args.gpu = 0
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

"""
v1 = chainer.Variable(xp.ones((1,2),dtype=np.float32))
v2 = chainer.Variable(xp.ones((3,2),dtype=np.float32))
Variable_vstack(v1,v2)

v1 = chainer.Variable(xp.ones((2,1),dtype=np.float32))
v2 = chainer.Variable(xp.ones((2,3),dtype=np.float32))
Variable_hstack(v1,v2)
"""
# src_sentence: 翻訳したい単語列 e.g. ['彼', 'は', '走る']
# trg_sentence: 正解の翻訳を表す単語列 e.g. ['he', 'runs']
# training: 学習か予測か。デコーダの挙動に影響する。



model = FunctionSet(
	#encode
	w_xi = EmbedID(SRC_VOCAB_SIZE, SRC_EMBED_SIZE), # 入力層(one-hot) -> 入力埋め込み層
	w_ip = Linear(SRC_EMBED_SIZE, 4 * (HIDDEN_SIZE + HIDDEN_SIZE2)), # 入力埋め込み層 -> 入力隠れ層
	w_pp = Linear((HIDDEN_SIZE + HIDDEN_SIZE2), 4 * (HIDDEN_SIZE + HIDDEN_SIZE2)), # 入力隠れ層 -> 入力隠れ層
	w_pq = Linear((HIDDEN_SIZE + HIDDEN_SIZE2), 4 * (HIDDEN_SIZE + HIDDEN_SIZE2)), # 入力隠れ層 -> 出力隠れ層
	#w_yq = EmbedID(TRG_VOCAB_SIZE, 4 * HIDDEN_SIZE), # 出力層(one-hot) -> 出力隠れ層
	# attentional weight estimator
	w_pw = Linear(HIDDEN_SIZE2, HIDDEN_SIZE2),
	w_qw = Linear(HIDDEN_SIZE2, HIDDEN_SIZE2),
	w_we = Linear(HIDDEN_SIZE2, 1),
	w_cp = Linear(HIDDEN_SIZE, 4 * (HIDDEN_SIZE + HIDDEN_SIZE2)),
	#decode
	w_iq = Linear(TRG_EMBED_SIZE, 4 * (HIDDEN_SIZE + HIDDEN_SIZE2)), # 出力層(one-hot) -> 出力隠れ層
	w_qq = Linear((HIDDEN_SIZE + HIDDEN_SIZE2), 4 * (HIDDEN_SIZE + HIDDEN_SIZE2)), # 出力隠れ層 -> 出力隠れ層
	w_qj = Linear(HIDDEN_SIZE, TRG_EMBED_SIZE), # 出力隠れ層 -> 出力埋め込み層
	w_jy = Linear(TRG_EMBED_SIZE, TRG_VOCAB_SIZE), # 出力隠れ層 -> 出力隠れ層
)  

for param in model.parameters:
    param[:] = np.random.uniform(-0.1, 0.1, param.shape)

if args.gpu >= 0:
    cuda.check_cuda_available()
    cuda.get_device(args.gpu).use()
    model.to_gpu()

m1 = (np.r_[np.eye(HIDDEN_SIZE,dtype = np.float32),np.zeros((HIDDEN_SIZE2,HIDDEN_SIZE),dtype = np.float32)])
m2 = (np.r_[np.zeros((HIDDEN_SIZE,HIDDEN_SIZE2),dtype = np.float32),np.eye(HIDDEN_SIZE2,dtype = np.float32)])
M1 = Variable(xp.array(m1))
M2 = Variable(xp.array(m2))
#simple_attention モデル(attention部分少し細かい)
def forward3(src_sentence, trg_sentence, model, training):
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
	c = chainer.Variable(xp.zeros((1, (HIDDEN_SIZE + HIDDEN_SIZE2)),dtype=np.float32))
	# エンコーダ
	x = Variable(xp.array([END_OF_SENTENCES], dtype=np.int32))
	i = tanh(model.w_xi(x))
	c, p = lstm(c, model.w_ip(i))
	p_context = matmul(p, M1)
	P_attention = matmul(p, M2)
	#P_all = P_attention
	p_all = p_context
	Pw_list_Mat = model.w_pw(P_attention)
	for word in reversed(src_sentence):
		x = Variable(xp.array([word], dtype=np.int32))
		i = tanh(model.w_xi(x))
		c, p = lstm(c,model.w_ip(i) + model.w_pp(p))
		p_context = matmul(p, M1)
		P_attention = matmul(p, M2)
		#P_all = Variable_vstack(P_all,P_attention)
		p_all = Variable_vstack(p_all,p_context)
		Pw_list_Mat = Variable_vstack(Pw_list_Mat, model.w_pw(P_attention))
	# エンコーダ -> デコーダ
	c, q = lstm(c, model.w_pq(p))
	c = chainer.Variable(xp.zeros((1, (HIDDEN_SIZE + HIDDEN_SIZE2)),dtype=np.float32))
	# デコーダ
	if training:
		# 学習時はyとして正解の翻訳を使い、forwardの結果として累積損失を返す。
		accum_loss = Variable(xp.zeros((), dtype=np.float32))
		for word in trg_sentence:
			Q = matmul(q, M2)
			q_context = matmul(q, M1)
			j = tanh(model.w_qj(q_context))
			y = model.w_jy(j)
			t = Variable(xp.array([word], dtype=np.int32))
			loss = softmax_cross_entropy(y, t)
			accum_loss += softmax_cross_entropy(y, t)
			i = tanh(model.w_xi(t))
			#attention
			#multih = matmul(P_all,transpose(Q))
			v_i_plus = model.w_qw(Q) #v_i: vector
			v_i_plus_Mat = matmul(transpose(v_i_plus), Variable(xp.ones((1,len(src_sentence) + 1),dtype=np.float32)))
			list_e_Mat = model.w_we(tanh(Pw_list_Mat + transpose(v_i_plus_Mat)))
			a_t = softmax(transpose(list_e_Mat))
			#print a_t.data.sum(), p_all.data.shape, a_t.data.shape
			m_t = matmul(a_t, p_all)
			c,q = lstm(c, model.w_iq(i) + model.w_qq(q) + model.w_cp(m_t))
		#print list_e_Mat.data.sum()
		return accum_loss
	else:
	# 予測時には翻訳器が生成したyを次回の入力に使い、forwardの結果として生成された単語列を返す。
	# yの中で最大の確率を持つ単語を選択していくが、softmaxを取る必要はない。
		hyp_sentence = []
		while len(hyp_sentence) < 100: # 100単語以上は生成しないようにする
			Q = matmul(q, M2)
			q_context = matmul(q, M1)
			j = tanh(model.w_qj(q_context))
			y = model.w_jy(j)
			word = int(y.data.argmax(1)[0])
			t = Variable(xp.array([word], dtype=np.int32))
			i = tanh(model.w_xi(t))
			if word == END_OF_SENTENCES:
				hyp_sentence.append("unk" + str(END_OF_SENTENCES))
				break # 終端記号が生成されたので終了
			hyp_sentence.append(vocabworddic[int(word)])
			#attention
			v_i_plus = model.w_qw(Q) #v_i: vector
			v_i_plus_Mat = matmul(transpose(v_i_plus), Variable(xp.ones((1,len(src_sentence) + 1),dtype=np.float32)))
			list_e_Mat = model.w_we(tanh(Pw_list_Mat + transpose(v_i_plus_Mat)))
			a_t = softmax(transpose(list_e_Mat))
			#print a_t.data.sum(), p_all.data.shape, a_t.data.shape
			m_t = matmul(a_t, p_all)
			c,q = lstm(c, model.w_iq(tanh(model.w_xi(t))) + model.w_qq(q) + model.w_cp(m_t))
		return hyp_sentence

forward3(sentenceslist[0], sentenceslist[0], model, training = True)
#model.w_we(Variable(xp.ones((46,100),dtype = np.float32)))
"""
v = xp.array([[1,2]])
xp.dot(xp.ones((3 + 1,1)),v)
v = np.array([[1,2,3,4,5,6]])
HIDDEN_SIZE = 4
HIDDEN_SIZE2 = 2
M1 = np.r_[np.eye(HIDDEN_SIZE),np.zeros((HIDDEN_SIZE2,HIDDEN_SIZE))]
M2 = np.r_[np.zeros((HIDDEN_SIZE,HIDDEN_SIZE2)),np.eye(HIDDEN_SIZE2)]
np.dot(v,M1)
np.dot(v,M2)
"""
import time
from chainer import optimizer
def train(japansentencsetdoc,englishsentencsetdoc,model,N,forward):
	#opt = SGD() # 確率的勾配法を使用
	opt = Adam()
	#opt = AdaGrad(lr = 0.01)
	opt.setup(model) # 学習器の初期化
	#opt.add_hook(optimizer.GradientClipping(5))
	#for sentence in sentence_set:
	accum_loss_sum = 0
	first_time = time.time()
	for textID in range(N):
		accum_loss = xp
		if textID % 200 == 1:
			print "textID: ", textID, "time", (time.time() - first_time)
			#print accum_loss_sum.data, accum_loss.data
		opt.zero_grads() # 勾配の初期化
		accum_loss = forward(japansentencsetdoc[textID], englishsentencsetdoc[textID], model, training = True) # 損失の計算
		#print textID
		accum_loss.backward() # 誤差逆伝播
		accum_loss.unchain_backward() # 誤差逆伝播
		accum_loss_sum += (accum_loss.data)
		#opt.clip_grads(10) # 大きすぎる勾配を抑制
		#accum_loss_sum += accum_loss
		opt.update() # パラメータの更新
	print accum_loss_sum

def evaluate(japansentencsetdoc,englishsentencsetdoc,model,N,forward):
	accum_loss_sum = 0
	first_time = time.time()
	for textID in range(1067,N):
		#accum_loss = forward2(japansentencsetdoc[textID], englishsentencsetdoc[textID], model, training = True) # 損失の計算
		accum_loss = forward(japansentencsetdoc[textID], englishsentencsetdoc[textID], model, training = True) # 損失の計算
		accum_loss_sum += accum_loss.data
	print accum_loss_sum


for i in range(0,50):
	print i
	train(sentenceslist,sentenceslist,model,100,forward3)
	evaluate(sentenceslist,sentenceslist,model,1300,forward3)
	hyp_sentence = forward3(sentenceslist[0],sentenceslist[0],model, training = False)
	text = ""
	for w in sentenceslist[0]:
		text = text + w
	print "=====問題======"
	print text
	print "=====予測======"
	print ' '.join(hyp_sentence)


#simple_attention モデル(attention部分簡単)
model = FunctionSet(
	#encode
	w_xi = EmbedID(SRC_VOCAB_SIZE, SRC_EMBED_SIZE), # 入力層(one-hot) -> 入力埋め込み層
	w_ip = Linear(SRC_EMBED_SIZE, 4 * (HIDDEN_SIZE + HIDDEN_SIZE2)), # 入力埋め込み層 -> 入力隠れ層
	w_pp = Linear((HIDDEN_SIZE + HIDDEN_SIZE2), 4 * (HIDDEN_SIZE + HIDDEN_SIZE2)), # 入力隠れ層 -> 入力隠れ層
	w_pq = Linear((HIDDEN_SIZE + HIDDEN_SIZE2), 4 * (HIDDEN_SIZE + HIDDEN_SIZE2)), # 入力隠れ層2 -> 出力隠れ層
	#w_yq = EmbedID(TRG_VOCAB_SIZE, 4 * HIDDEN_SIZE), # 出力層(one-hot) -> 出力隠れ層
	# attentional weight estimator
	w_cp = Linear(HIDDEN_SIZE, 4 * (HIDDEN_SIZE + HIDDEN_SIZE2)),
	#decode
	w_iq = Linear(TRG_EMBED_SIZE, 4 * (HIDDEN_SIZE + HIDDEN_SIZE2)), # 出力層(one-hot) -> 出力隠れ層
	w_qq = Linear((HIDDEN_SIZE + HIDDEN_SIZE2), 4 * (HIDDEN_SIZE + HIDDEN_SIZE2)), # 出力隠れ層 -> 出力隠れ層
	w_qj = Linear(HIDDEN_SIZE, TRG_EMBED_SIZE), # 出力隠れ層 -> 出力埋め込み層
	w_jy = Linear(TRG_EMBED_SIZE, TRG_VOCAB_SIZE), # 出力隠れ層 -> 出力隠れ層
)  

for param in model.parameters:
    param[:] = np.random.uniform(-0.1, 0.1, param.shape)

if args.gpu >= 0:
    cuda.check_cuda_available()
    cuda.get_device(args.gpu).use()
    model.to_gpu()


def forward4(src_sentence, trg_sentence, model, training):
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
	c = chainer.Variable(xp.zeros((1, (HIDDEN_SIZE + HIDDEN_SIZE2)),dtype=np.float32))
	# エンコーダ
	x = Variable(xp.array([END_OF_SENTENCES], dtype=np.int32))
	i = tanh(model.w_xi(x))
	c, p = lstm(c, model.w_ip(i))
	p_context = matmul(p, M1)
	P_attention = matmul(p, M2)
	P_all = P_attention
	p_all = p_context
	for word in reversed(src_sentence):
		x = Variable(xp.array([word], dtype=np.int32))
		i = tanh(model.w_xi(x))
		c, p = lstm(c,model.w_ip(i) + model.w_pp(p))
		p_context = matmul(p, M1)
		P_attention = matmul(p, M2)
		P_all = Variable_vstack(P_all,P_attention)
		p_all = Variable_vstack(p_all,p_context)
		#Pw_list_Mat = Variable_vstack(Pw_list_Mat, model.w_pw(P))
	# エンコーダ -> デコーダ
	c, q = lstm(c, model.w_pq(p))
	c = chainer.Variable(xp.zeros((1, (HIDDEN_SIZE + HIDDEN_SIZE2)),dtype=np.float32))
	# デコーダ
	if training:
		# 学習時はyとして正解の翻訳を使い、forwardの結果として累積損失を返す。
		accum_loss = Variable(xp.zeros((), dtype=np.float32))
		for word in trg_sentence:
			Q = matmul(q, M2)
			q_context = matmul(q, M1)
			j = tanh(model.w_qj(q_context))
			y = model.w_jy(j)
			t = Variable(xp.array([word], dtype=np.int32))
			loss = softmax_cross_entropy(y, t)
			accum_loss += softmax_cross_entropy(y, t)
			i = tanh(model.w_xi(t))
			#attention
			multih = matmul(P_all,transpose(Q))
			a_t = softmax(transpose(multih))
			m_t = matmul(a_t, p_all)
			c,q = lstm(c, model.w_iq(i) + model.w_qq(q) + model.w_cp(m_t))
		#print list_e_Mat.data.sum()
		return accum_loss
	else:
	# 予測時には翻訳器が生成したyを次回の入力に使い、forwardの結果として生成された単語列を返す。
	# yの中で最大の確率を持つ単語を選択していくが、softmaxを取る必要はない。
		hyp_sentence = []
		while len(hyp_sentence) < 100: # 100単語以上は生成しないようにする
			Q = matmul(q, M2)
			q_context = matmul(q, M1)
			j = tanh(model.w_qj(q_context))
			y = model.w_jy(j)
			word = int(y.data.argmax(1)[0])
			t = Variable(xp.array([word], dtype=np.int32))
			if word == END_OF_SENTENCES:
				hyp_sentence.append("unk" + str(END_OF_SENTENCES))
				break # 終端記号が生成されたので終了
			hyp_sentence.append(vocabworddic[int(word)])
			#attention
			multih = matmul(P_all,transpose(Q))
			multi_exp = exp(multih)
			a_t = multi_exp / multi_exp.data.sum()
			m_t = matmul(transpose(a_t), p_all)
			c,q = lstm(c, model.w_iq(tanh(model.w_xi(t))) + model.w_qq(q) + model.w_cp(m_t))
		return hyp_sentence


forward4(sentenceslist[0], sentenceslist[0], model, training = True)
import time
from chainer import optimizer
"""
def train(japansentencsetdoc,englishsentencsetdoc,model,N):
	#opt = SGD() # 確率的勾配法を使用
	#opt = Adam()
	opt = AdaGrad(lr = 0.01)
	opt.setup(model) # 学習器の初期化
	opt.add_hook(optimizer.GradientClipping(5))
	#for sentence in sentence_set:
	accum_loss_sum = 0
	first_time = time.time()
	for textID in range(N):
		accum_loss = xp
		if textID % 100 == 0:
			print "textID: ", textID, "time", (time.time() - first_time)
			#print accum_loss_sum.data, accum_loss.data
		opt.zero_grads() # 勾配の初期化
		#accum_loss,state = forward(japansentencsetdoc[textID], englishsentencsetdoc[textID], model, training = True,state = state) # 損失の計算
		#accum_loss = forward2(japansentencsetdoc[textID], englishsentencsetdoc[textID], model, training = True) # 損失の計算
		accum_loss = forward4(japansentencsetdoc[textID], englishsentencsetdoc[textID], model, training = True) # 損失の計算
		#print textID
		accum_loss.backward() # 誤差逆伝播
		accum_loss.unchain_backward() # 誤差逆伝播
		accum_loss_sum += (accum_loss.data)
		#opt.clip_grads(10) # 大きすぎる勾配を抑制
		#accum_loss_sum += accum_loss
		opt.update() # パラメータの更新
	print accum_loss_sum


def evaluate(japansentencsetdoc,englishsentencsetdoc,model,N):
	accum_loss_sum = 0
	first_time = time.time()
	for textID in range(1067,N):
		#accum_loss = forward2(japansentencsetdoc[textID], englishsentencsetdoc[textID], model, training = True) # 損失の計算
		accum_loss = forward4(japansentencsetdoc[textID], englishsentencsetdoc[textID], model, training = True) # 損失の計算
		accum_loss_sum += accum_loss.data
	print accum_loss_sum
"""

for i in range(0,50):
	print i
	train(sentenceslist,sentenceslist,model,1000,forward4)
	evaluate(sentenceslist,sentenceslist,model,1300,forward4)
	#hyp_sentence = forward2(sentenceslist[0],sentenceslist[0],model, training = False)
	hyp_sentence = forward4(sentenceslist[0],sentenceslist[0],model, training = False)
	text = ""
	for w in sentenceslist[0]:
		text = text + w
	print "=====問題======"
	print text
	print "=====予測======"
	print ' '.join(hyp_sentence)

model = FunctionSet(
	#encode
	w_xi = EmbedID(SRC_VOCAB_SIZE, SRC_EMBED_SIZE), # 入力層(one-hot) -> 入力埋め込み層
	w_ip = Linear(SRC_EMBED_SIZE, 4 * HIDDEN_SIZE), # 入力埋め込み層 -> 入力隠れ層
	w_pp = Linear(HIDDEN_SIZE, 4 * HIDDEN_SIZE), # 入力隠れ層 -> 入力隠れ層
	w_pq = Linear(HIDDEN_SIZE, 4 * HIDDEN_SIZE), # 入力隠れ層2 -> 出力隠れ層
	#w_yq = EmbedID(TRG_VOCAB_SIZE, 4 * HIDDEN_SIZE), # 出力層(one-hot) -> 出力隠れ層
	# attentional weight estimator
	w_cp = Linear(HIDDEN_SIZE, 4 * HIDDEN_SIZE),
	#decode
	w_iq = Linear(TRG_EMBED_SIZE, 4 * HIDDEN_SIZE), # 出力層(one-hot) -> 出力隠れ層
	w_qq = Linear(HIDDEN_SIZE, 4 * HIDDEN_SIZE), # 出力隠れ層 -> 出力隠れ層
	w_qj = Linear(HIDDEN_SIZE, TRG_EMBED_SIZE), # 出力隠れ層 -> 出力埋め込み層
	w_jy = Linear(TRG_EMBED_SIZE, TRG_VOCAB_SIZE), # 出力隠れ層 -> 出力隠れ層
)  

for param in model.parameters:
    param[:] = np.random.uniform(-0.01, 0.01, param.shape)

if args.gpu >= 0:
    cuda.check_cuda_available()
    cuda.get_device(args.gpu).use()
    model.to_gpu()


#通常のattention
def forward5(src_sentence, trg_sentence, model, training):
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
	c = chainer.Variable(xp.zeros((1,HIDDEN_SIZE),dtype=np.float32))
	# エンコーダ
	x = Variable(xp.array([END_OF_SENTENCES], dtype=np.int32))
	i = tanh(model.w_xi(x))
	c, p = lstm(c, model.w_ip(i))
	p_all = p
	for word in reversed(src_sentence):
		x = Variable(xp.array([word], dtype=np.int32))
		i = tanh(model.w_xi(x))
		c, p = lstm(c,model.w_ip(i) + model.w_pp(p))
		p_all = Variable_vstack(p_all,p)
		#Pw_list_Mat = Variable_vstack(Pw_list_Mat, model.w_pw(P))
	# エンコーダ -> デコーダ
	c, q = lstm(c, model.w_pq(p))
	c2 = chainer.Variable(xp.zeros((1, HIDDEN_SIZE),dtype=np.float32))
	# デコーダ
	if training:
		# 学習時はyとして正解の翻訳を使い、forwardの結果として累積損失を返す。
		accum_loss = Variable(xp.zeros((), dtype=np.float32))
		for word in trg_sentence:
			j = tanh(model.w_qj(q))
			y = model.w_jy(j)
			t = Variable(xp.array([word], dtype=np.int32))
			loss = softmax_cross_entropy(y, t)
			accum_loss += softmax_cross_entropy(y, t)
			i = tanh(model.w_xi(t))
			#attention
			multih = matmul(p_all,transpose(q))
			#multi_exp = exp(multih)
			#a_t = multi_exp / multi_exp.data.sum()
			a_t = softmax(transpose(multih))
			#print a_t.data,a_t.data.sum(),
			m_t = matmul(a_t, p_all)
			c2,q = lstm(c2, model.w_iq(i) + model.w_qq(q) + model.w_cp(m_t))
		#print list_e_Mat.data.sum()
		return accum_loss
	else:
	# 予測時には翻訳器が生成したyを次回の入力に使い、forwardの結果として生成された単語列を返す。
	# yの中で最大の確率を持つ単語を選択していくが、softmaxを取る必要はない。
		hyp_sentence = []
		while len(hyp_sentence) < 100: # 100単語以上は生成しないようにする
			j = tanh(model.w_qj(q))
			y = model.w_jy(j)
			word = int(y.data.argmax(1)[0])
			t = Variable(xp.array([word], dtype=np.int32))
			i = tanh(model.w_xi(t))
			if word == END_OF_SENTENCES:
				hyp_sentence.append("unk" + str(END_OF_SENTENCES))
				break # 終端記号が生成されたので終了
			hyp_sentence.append(vocabworddic[int(word)])
			#attention
			multih = matmul(p_all,transpose(q))
			#multi_exp = exp(multih)
			#a_t = multi_exp / multi_exp.sum()
			a_t = softmax(transpose(multih))
			m_t = matmul(a_t, p_all)
			c2,q = lstm(c2, model.w_iq(i) + model.w_qq(q) + model.w_cp(m_t))
		return hyp_sentence


forward5(sentenceslist[0], sentenceslist[0], model, training = True)

for i in range(0,50):
	print i
	train(sentenceslist,sentenceslist,model,1000,forward5)
	evaluate(sentenceslist,sentenceslist,model,1300,forward5)
	hyp_sentence = forward5(sentenceslist[0],sentenceslist[0],model, training = False)
	text = ""
	for w in sentenceslist[0]:
		text = text + w
	print "=====問題======"
	print text
	print "=====予測======"
	print ' '.join(hyp_sentence)



model = FunctionSet(
	#encode
	w_xi = EmbedID(SRC_VOCAB_SIZE, SRC_EMBED_SIZE), # 入力層(one-hot) -> 入力埋め込み層
	w_ip = Linear(SRC_EMBED_SIZE, 4 * HIDDEN_SIZE), # 入力埋め込み層 -> 入力隠れ層
	w_pp = Linear(HIDDEN_SIZE, 4 * HIDDEN_SIZE), # 入力隠れ層 -> 入力隠れ層
	w_pq = Linear(HIDDEN_SIZE, 4 * HIDDEN_SIZE), # 入力隠れ層2 -> 出力隠れ層
	#w_yq = EmbedID(TRG_VOCAB_SIZE, 4 * HIDDEN_SIZE), # 出力層(one-hot) -> 出力隠れ層
	# attentional weight estimator
	w_pw = Linear(HIDDEN_SIZE, HIDDEN_SIZE),
	w_qw = Linear(HIDDEN_SIZE, HIDDEN_SIZE),
	w_we = Linear(HIDDEN_SIZE, 1),
	w_cp = Linear(HIDDEN_SIZE, 4 * HIDDEN_SIZE),
	#decode
	w_iq = Linear(TRG_EMBED_SIZE, 4 * HIDDEN_SIZE), # 出力層(one-hot) -> 出力隠れ層
	w_qq = Linear(HIDDEN_SIZE, 4 * HIDDEN_SIZE), # 出力隠れ層 -> 出力隠れ層
	w_qj = Linear(HIDDEN_SIZE, TRG_EMBED_SIZE), # 出力隠れ層 -> 出力埋め込み層
	w_jy = Linear(TRG_EMBED_SIZE, TRG_VOCAB_SIZE), # 出力隠れ層 -> 出力隠れ層
)  

for param in model.parameters:
    param[:] = np.random.uniform(-0.01, 0.01, param.shape)

if args.gpu >= 0:
    cuda.check_cuda_available()
    cuda.get_device(args.gpu).use()
    model.to_gpu()


#attentionモデル(attention少し複雑)
def forward6(src_sentence, trg_sentence, model, training):
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
	c = chainer.Variable(xp.zeros((1,HIDDEN_SIZE),dtype=np.float32))
	# エンコーダ
	x = Variable(xp.array([END_OF_SENTENCES], dtype=np.int32))
	i = tanh(model.w_xi(x))
	c, p = lstm(c, model.w_ip(i))
	p_all = p
	Pw_list_Mat = model.w_pw(p)
	for word in reversed(src_sentence):
		x = Variable(xp.array([word], dtype=np.int32))
		i = tanh(model.w_xi(x))
		c, p = lstm(c,model.w_ip(i) + model.w_pp(p))
		p_all = Variable_vstack(p_all,p)
		Pw_list_Mat = Variable_vstack(Pw_list_Mat, model.w_pw(p))
	# エンコーダ -> デコーダ
	c, q = lstm(c, model.w_pq(p))
	c2 = chainer.Variable(xp.zeros((1, HIDDEN_SIZE),dtype=np.float32))
	# デコーダ
	if training:
		# 学習時はyとして正解の翻訳を使い、forwardの結果として累積損失を返す。
		accum_loss = Variable(xp.zeros((), dtype=np.float32))
		for word in trg_sentence:
			j = tanh(model.w_qj(q))
			y = model.w_jy(j)
			t = Variable(xp.array([word], dtype=np.int32))
			loss = softmax_cross_entropy(y, t)
			accum_loss += softmax_cross_entropy(y, t)
			i = tanh(model.w_xi(t))
			#attention
			v_i_plus = model.w_qw(q) #v_i: vector
			v_i_plus_Mat = matmul(transpose(v_i_plus), Variable(xp.ones((1,len(src_sentence) + 1),dtype=np.float32)))
			list_e_Mat = model.w_we(tanh(Pw_list_Mat + transpose(v_i_plus_Mat)))
			a_t = softmax(transpose(list_e_Mat))
			#print a_t.data.sum(), p_all.data.shape, a_t.data.shape
			m_t = matmul(a_t, p_all)
			c2,q = lstm(c2, model.w_iq(i) + model.w_qq(q) + model.w_cp(m_t))
		#print list_e_Mat.data.sum()
		return accum_loss
	else:
	# 予測時には翻訳器が生成したyを次回の入力に使い、forwardの結果として生成された単語列を返す。
	# yの中で最大の確率を持つ単語を選択していくが、softmaxを取る必要はない。
		hyp_sentence = []
		while len(hyp_sentence) < 100: # 100単語以上は生成しないようにする
			j = tanh(model.w_qj(q))
			y = model.w_jy(j)
			word = int(y.data.argmax(1)[0])
			t = Variable(xp.array([word], dtype=np.int32))
			if word == END_OF_SENTENCES:
				hyp_sentence.append("unk" + str(END_OF_SENTENCES))
				break # 終端記号が生成されたので終了
			hyp_sentence.append(vocabworddic[int(word)])
			i = tanh(model.w_xi(t))
			#attention
			v_i_plus = model.w_qw(q)
			v_i_plus_Mat = matmul(transpose(v_i_plus), Variable(xp.ones((1,len(src_sentence) + 1),dtype=np.float32)))
			list_e_Mat = model.w_we(tanh(Pw_list_Mat + transpose(v_i_plus_Mat)))
			a_t = softmax(transpose(list_e_Mat))
			m_t = matmul(a_t, p_all)
			c2,q = lstm(c2, model.w_iq(i) + model.w_qq(q) + model.w_cp(m_t))
		return hyp_sentence


forward6(sentenceslist[0], sentenceslist[0], model, training = True)

for i in range(0,50):
	print i
	train(sentenceslist,sentenceslist,model,1000)
	evaluate(sentenceslist,sentenceslist,model,1300,forward6)
	hyp_sentence = forward6(sentenceslist[0],sentenceslist[0],model, training = False)
	text = ""
	for w in sentenceslist[0]:
		text = text + w
	print "=====問題======"
	print text
	print "=====予測======"
	print ' '.join(hyp_sentence)

"""
# Save final model
import pickle
model.to_cpu()
pickle.dump(model, open("yahoofinance/alltext0525_s2s_model.dump", 'wb'), -1)
pickle.dump(japaneseIDdic,open("japaneseIDdic.dump", 'wb'), -1)
pickle.dump(englishIDdic,open("englishIDdic.dump", 'wb'), -1)
pickle.dump(unfiltered2,open("unfiltered2.dump", 'wb'), -1)

model.to_gpu()
"""
