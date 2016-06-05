#encoding:utf-8
#データ読み込み
import json
import os
from ast import literal_eval
import re
#import MeCab
import unicodedata
import sys
#import ngram
#import jcconv
import chainer
from chainer import cuda
import argparse
import numpy as np
#from normalizer import normalize,_convert_marks,_delete_cyclic_word
#from impala.dbapi import connect

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
args = parser.parse_args()

#xp = cuda.cupy if args.gpu >= 0 else np
if args.gpu >= 0:
	xp = cuda.cupy
else:
	xp = np


#f = open("train1000.en")]
#f = open("train20000.en")
f = open("yahoofinance/alltext0525.txt")
#f = open("train5000.en")
#englishdata = f.read()
yahooboarddata = f.read()
f.close()
#englishsentencset = englishdata.split("\n")
yahooboarddataset = yahooboarddata.split("\n")
yahooboarddataset = np.array(yahooboarddataset)
smallyahooboarddataset = yahooboarddataset[range(0,len(yahooboarddataset),2)]

vocablist = []
for sentences in smallyahooboarddataset:
	sentence = sentences.split(" ")
	vocablist += sentence



vocablist = list(set(vocablist))
vocabIDdic = dict(zip(vocablist,range(len(vocablist))))


smallyahooboardsentences = []
for comments in smallyahooboarddataset:
	smallyahooboardsentence = []
	sentences = comments.split("。")
	for kuarray in sentences:
		IDsentence = []
		ku = kuarray.split("、")
		for wordsarray in ku:
			words = wordsarray.split(" ")
			for word in words:
				try:
					IDsentence.append(vocabIDdic[word])
				except:
					vocablist.append(word)
					vocabIDdic[word] = len(vocablist)
					IDsentence.append(vocabIDdic[word])
		smallyahooboardsentence.append(IDsentence)
	smallyahooboardsentences.append(smallyahooboardsentence)

vocabworddic = dict(zip(vocabIDdic.values(),vocabIDdic.keys()))

#http://qiita.com/odashi_t/items/a1be7c4964fbea6a116e
from chainer import FunctionSet
from chainer.functions import *
"""
VOCAB_SIZE = len(unfiltered)
HIDDEN_SIZE = 100
model = FunctionSet(
  w_xh = EmbedID(VOCAB_SIZE, HIDDEN_SIZE), # 入力層(one-hot) -> 隠れ層
  w_hh = Linear(HIDDEN_SIZE, HIDDEN_SIZE), # 隠れ層 -> 隠れ層
  w_hy = Linear(HIDDEN_SIZE, VOCAB_SIZE), # 隠れ層 -> 出力層
)  
"""


from chainer import Variable


"""
def forward(sentence, model,exity = False):
	accum_loss = Variable(np.zeros((), dtype=np.float32)) # 累積損失の初期値
	sentence = [ japaneseIDdic[word.decode("utf-8")] for word in sentence]
	h = Variable(np.zeros((1, HIDDEN_SIZE), dtype=np.float32)) # 隠れ層の初期値
	log_joint_prob = float(0) # 文の結合確率
	for word in sentence:
		x = Variable(np.array([word], dtype=np.int32)) # 次回の入力層 (=今回の正解)
		u = model.w_hy(h)
		accum_loss += softmax_cross_entropy(u, x) # 損失の蓄積
		y = softmax(u)
		log_joint_prob += np.log(y.data[0][word]) # 結合確率の更新
		h = tanh(model.w_xh(x) + model.w_hh(h)) # 隠れ層の更新
	return log_joint_prob, accum_loss # 累積損失も一緒に返す
"""
from chainer.optimizers import *
"""
def train(japansentencsetdoc,model):
	opt = SGD() # 確率的勾配法を使用
	opt.setup(model) # 学習器の初期化
	#for sentence in sentence_set:
	for textID in range(900):
		sentence = japansentencsetdoc[textID]
		opt.zero_grads(); # 勾配の初期化
		log_joint_prob, accum_loss = forward(sentence, model) # 損失の計算
		accum_loss.backward() # 誤差逆伝播
		opt.clip_grads(10) # 大きすぎる勾配を抑制
		opt.update() # パラメータの更新

def test(japansentencsetdoc,model):
	for textID in range(900,1000):
		log_joint_prob, accum_loss = forward(japansentencsetdoc[textID], model) # 損失の計算
	return 	log_joint_prob,accum_loss.data

for i in range(0,100):
	train(japansentencsetdoc,model)
	log_joint_prob,accum_loss = test(japansentencsetdoc,model)
	print log_joint_prob,accum_loss
"""

SRC_VOCAB_SIZE = (len(vocablist) + 3)
SRC_EMBED_SIZE  = (len(vocablist) + 3)
HIDDEN_SIZE = 100
TRG_VOCAB_SIZE = (len(vocablist) + 3)
TRG_EMBED_SIZE = (len(vocablist) + 3)
model = FunctionSet(
	w_xi = EmbedID(SRC_VOCAB_SIZE, SRC_EMBED_SIZE), # 入力層(one-hot) -> 入力埋め込み層
	w_ip = Linear(SRC_EMBED_SIZE, 4 * HIDDEN_SIZE), # 入力埋め込み層 -> 入力隠れ層
	w_pp = Linear(HIDDEN_SIZE, 4 * HIDDEN_SIZE), # 入力隠れ層 -> 入力隠れ層
	#w_pI = Linear(HIDDEN_SIZE, HIDDEN_SIZE), # 入力隠れ層 -> 文入力埋め込み層
	w_IP = Linear(HIDDEN_SIZE, 4 * HIDDEN_SIZE), # 文入力埋め込み層 -> 文入力隠れ層
	w_PP = Linear(HIDDEN_SIZE, 4 * HIDDEN_SIZE), # 文入力隠れ層 -> 文入力隠れ層
	w_PQ = Linear(HIDDEN_SIZE, 4 * HIDDEN_SIZE), # 文入力隠れ層 -> 文出力隠れ層
	w_QQ = Linear(HIDDEN_SIZE, 4 * HIDDEN_SIZE), # 文出力隠れ層 -> 文出力隠れ層
	#w_Qq = Linear(HIDDEN_SIZE, HIDDEN_SIZE), # 文出力隠れ層 -> 出力埋め込み層
	#w_qQ = Linear(HIDDEN_SIZE, HIDDEN_SIZE), # 出力埋め込み層 -> 文出力隠れ層
	w_yq = EmbedID(TRG_VOCAB_SIZE, 4 * HIDDEN_SIZE), # 出力層(one-hot) -> 出力隠れ層
	w_qq = Linear(HIDDEN_SIZE, 4 * HIDDEN_SIZE), # 出力隠れ層 -> 出力隠れ層
	w_qj = Linear(HIDDEN_SIZE, TRG_EMBED_SIZE), # 出力隠れ層 -> 出力埋め込み層
	w_jy = Linear(TRG_EMBED_SIZE, TRG_VOCAB_SIZE), # 出力隠れ層 -> 出力隠れ層
)  

if args.gpu >= 0:
    cuda.check_cuda_available()
    cuda.get_device(args.gpu).use()
    model.to_gpu()

print "model_making"
END_OF_SENTENCE = len(vocablist)
END_OF_SENTENCES = (len(vocablist) + 1)

# src_sentence: 翻訳したい単語列 e.g. ['彼', 'は', '走る']
# trg_sentence: 正解の翻訳を表す単語列 e.g. ['he', 'runs']
# training: 学習か予測か。デコーダの挙動に影響する。
def forward(src_sentences, trg_sentences, model, training):
	# 単語IDへの変換（自分で適当に実装する）
	# 正解の翻訳には終端記号を追加しておく。
	src_sentenceList = []
	for src_sentence in src_sentences:
		src_sentence2 = []
		for word in src_sentence:
			try:
				#src_sentence2.append(vocabIDdic[word])
				src_sentence2.append(word)
			except:
				print word
				src_sentence2.append(SRC_VOCAB_SIZE - 1)
		#src_sentence = [ japaneseIDdic[word.decode("utf-8")] for word in src_sentence]
		#trg_sentence = [ englishIDdic[word] for word in trg_sentence] + [END_OF_SENTENCE]
		src_sentence = src_sentence2
		src_sentenceList.append(src_sentence)
	trg_sentenceList = []
	for i,trg_sentence in enumerate(trg_sentences):
		trg_sentence2 = []
		for word in trg_sentence:
			try:
				#trg_sentence2.append(vocabIDdic[word])
				trg_sentence2.append(word)
			except:
				print word
				trg_sentence2.append(TRG_VOCAB_SIZE - 1)
		if i < len(trg_sentences) - 1:
			trg_sentence2 = (trg_sentence2 + [END_OF_SENTENCE])
		else:
			trg_sentence2 = (trg_sentence2 + [END_OF_SENTENCES])
		trg_sentence = trg_sentence2
		trg_sentenceList.append(trg_sentence)
		#print trg_sentence
		# LSTM内部状態の初期値
	X_list = []
	for src_sentence in src_sentenceList:
		c = Variable(xp.zeros((1, HIDDEN_SIZE),dtype=np.float32))
		# エンコーダ
		x = Variable(xp.array([END_OF_SENTENCE], dtype=np.int32))
		i = tanh(model.w_xi(x))
		c, p = lstm(c, model.w_ip(i))
		for word in reversed(src_sentence):
			x = Variable(xp.array([word], dtype=np.int32))
			i = tanh(model.w_xi(x))
			c, p = lstm(c, model.w_ip(i) + model.w_pp(p))
		# エンコーダ埋め込み -> 文エンコーダ埋め込み
		#X = model.w_pI(c)
		#X_list.append(X)
		X_list.append(c)
	#文エンコーダ
	C = Variable(xp.zeros((1, HIDDEN_SIZE),dtype=np.float32))
	X = X_list[0]
	C, P = lstm(C, model.w_IP(X))
	for X in X_list[1::]:
		C, P = lstm(C, model.w_IP(X) + model.w_PP(P))
	#文エンコーダ -> 文デコーダ
	C, Q = lstm(C, model.w_PQ(P)) #Q: 文ベクトル
	#文デコーダ
	if training:
		for trg_sentence in trg_sentenceList:
			#accum_loss = xp.zeros((), dtype=np.float32)
			accum_loss = chainer.Variable(xp.zeros((), dtype=np.float32))
			for word in trg_sentence:
				#q = model.w_Qq(Q)
				#j = tanh(model.w_qj(q))
				#j = tanh(model.w_qj(Q))
				y = model.w_jy(j)
				t = Variable(xp.array([word], dtype=np.int32))
				loss = softmax_cross_entropy(y, t)
				accum_loss += loss
				#print loss
				c, q = lstm(c, model.w_yq(t) + model.w_qq(q))
			C, Q = lstm(C, model.w_PQ(model.w_qQ(q)) + model.w_QQ(Q))
		return accum_loss
		# エンコーダ -> デコーダ
		#c, q = lstm(c, model.w_pq(p))
		# デコーダ
	else:
		hyp_sentences = []
		hyp_sentence = []
	# 予測時には翻訳器が生成したyを次回の入力に使い、forwardの結果として生成された単語列を返す。
	# yの中で最大の確率を持つ単語を選択していくが、softmaxを取る必要はない。
		while len(hyp_sentences) < 2: #10文以上は生成しないようにする
			#q = model.w_Qq(Q)
			#j = tanh(model.w_qj(q))
			j = tanh(model.w_qj(Q))
			y = model.w_jy(j)
			word = y.data.argmax(1)[0]
			#print word
			if word == END_OF_SENTENCES:
				break # 終端記号が生成されたので終了
			if (word != END_OF_SENTENCE) & (len(hyp_sentence) < 20): # 100単語以上は生成しないようにする
				hyp_sentence.append(word)
				#except:
					#hyp_sentence.append("<UNKNOWN>" + str(word))
				s_y = Variable(xp.array([word], dtype=np.int32))
				c, q = lstm(c, model.w_yq(s_y) + model.w_qq(q))
			else:
				hyp_sentences.append(hyp_sentence)
				#C, Q = lstm(C, model.w_PQ(model.w_qQ(q)) + model.w_QQ(Q))
				C, Q = lstm(C, model.w_PQ(q) + model.w_QQ(Q))
				#print len(hyp_sentence)
				hyp_sentence = []
			#print len(hyp_sentences)
			s_y = Variable(xp.array([word], dtype=np.int32))
			c, q = lstm(c, model.w_yq(s_y) + model.w_qq(q))
		return hyp_sentences

forward(smallyahooboardsentences[0], smallyahooboardsentences[0], model, training = True)
N = 1000

def train(japansentencsetdoc,englishsentencsetdoc,model,N = N,batchsize = 10):
	perm = np.random.permutation(N)
	#opt = SGD() # 確率的勾配法を使用
	opt = Adam()
	opt.setup(model) # 学習器の初期化
	#for sentence in sentence_set:
	perm = np.random.permutation(N)
	accum_loss_sum = chainer.Variable(xp.zeros((), dtype=np.float32))
	accum_loss_sum2 = chainer.Variable(xp.zeros((), dtype=np.float32))
	batch_size_array = xp.array(batchsize, dtype=xp.float32)
	for i, textID in enumerate(np.array(range(N))[perm]):
		opt.zero_grads() # 勾配の初期化
		accum_loss = forward(japansentencsetdoc[textID], englishsentencsetdoc[textID], model, training = True) # 損失の計算
		#accum_loss_sum = accum_loss_sum / chainer.Variable(batch_size_array)
		accum_loss_sum.backward() # 誤差逆伝播
		#opt.clip_grads(10) # 大きすぎる勾配を抑制
		opt.update() # パラメータの更新
		accum_loss = chainer.Variable(xp.zeros((), dtype=np.float32))
		print accum_loss.data

def train(japansentencsetdoc,englishsentencsetdoc,model,N = N,batchsize = 10):
	perm = np.random.permutation(N)
	#opt = SGD() # 確率的勾配法を使用
	opt = Adam()
	opt.setup(model) # 学習器の初期化
	#for sentence in sentence_set:
	perm = np.random.permutation(N)
	accum_loss_sum = chainer.Variable(xp.zeros((), dtype=np.float32))
	accum_loss_sum2 = chainer.Variable(xp.zeros((), dtype=np.float32))
	opt.zero_grads() # 勾配の初期化
	batch_size_array = xp.array(batchsize, dtype=xp.float32)
	for i, textID in enumerate(np.array(range(N))[perm]):
		accum_loss = forward(japansentencsetdoc[textID], englishsentencsetdoc[textID], model, training = True) # 損失の計算
		accum_loss_sum += accum_loss
		print i, textID
		if i % batchsize == 0:
			accum_loss_sum = accum_loss_sum / chainer.Variable(batch_size_array)
			accum_loss_sum.backward() # 誤差逆伝播
			#opt.clip_grads(10) # 大きすぎる勾配を抑制
			opt.update() # パラメータの更新
			print accum_loss_sum.data
			accum_loss_sum = chainer.Variable(xp.zeros((), dtype=np.float32))
			opt.zero_grads() # 勾配の初期化
	print accum_loss_sum.data

def Test(japantest,englishtest,n):
	text = ""
	hyp_sentencelist = []
	hyp_sentence = forward(japantest[n],englishtest[n],model, training = False)
	for sentence in japantest[n]:
		for w in sentence:
			text = text + vocabworddic[w]
	print "=====問題======"
	print text
	print "=====正解======"
	print englishtest[n]
	print "=====予測======"
	for s in hyp_sentence:
		for w in s:
			hyp_sentencelist.append(vocabworddic[int(w)])
	print ' '.join(hyp_sentencelist)

hyp_sentence = forward(smallyahooboardsentences[0],smallyahooboardsentences[0],model, training = False)
Test(smallyahooboardsentences, smallyahooboardsentences,0)
vocabworddic

def test(sentence):
	tagger = MeCab.Tagger( '-Owakati -u /usr/local/Cellar/mecab/0.996/lib/mecab/dic/ipadic/wikipedia-keyword.dic')
	wordarrays = tagger.parse(sentence).split(" ")[0:-1]
	#wordarrays = sentence
	print wordarrays
	print (wordarrays == japantest[0])
	hyp_sentence = forward(wordarrays,['did', 'you', 'clean', 'your', 'room', '?'],model,training = False)
	print hyp_sentence

for i in range(0,5):
	print i
	train(smallyahooboardsentences[0:100],smallyahooboardsentences[0:100],model,N = 100)
	#hyp_sentence = forward(smallyahooboardsentences[0],smallyahooboardsentences[0],model, training = False)
	hyp_sentence = forward(smallyahooboardsentences[N],smallyahooboardsentences[N],model, training = False)
	Test(smallyahooboardsentences, smallyahooboardsentences,0)
	Test(smallyahooboardsentences, smallyahooboardsentences,N)


# Save final model
import pickle
#pickle.dump(model, open("model20000_2.dump", 'wb'), -1)
pickle.dump(model.to_cpu(), open("model20000_2.dump", 'wb'), -1)
pickle.dump(japaneseIDdic,open("japaneseIDdic.dump", 'wb'), -1)
pickle.dump(englishIDdic,open("englishIDdic.dump", 'wb'), -1)
pickle.dump(unfiltered2,open("unfiltered2.dump", 'wb'), -1)
