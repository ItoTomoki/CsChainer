#データ読み込み
import json
import os
from ast import literal_eval
import re
import MeCab
import unicodedata
import sys
import ngram
import jcconv
#from normalizer import normalize,_convert_marks,_delete_cyclic_word
#from impala.dbapi import connect

#f = open("train1000.en")]
f = open("train20000.en")
#f = open("train5000.en")
englishdata = f.read()
f.close()
englishsentencset = englishdata.split("\n")
englishsentencsetdoc = {}
for i in range(len(englishsentencset)):
	englishsentencset[i] = englishsentencset[i].split(" ")
	englishsentencsetdoc[i] = englishsentencset[i]

#f = open("train1000.ja")
f = open("train20000.ja")
#f = open("train5000.ja")
japandata = f.read()
f.close()
japansentencset = japandata.split("\n")

tagger = MeCab.Tagger( '-Owakati -u /usr/local/Cellar/mecab/0.996/lib/mecab/dic/ipadic/wikipedia-keyword.dic')
japansentencsetdoc = {}
for i in range(len(japansentencset)):
	japansentencset[i] = japansentencset[i].replace(" ","")
	japansentencset[i] = tagger.parse(japansentencset[i]).split(" ")[0:-1]
	japansentencsetdoc[i] = japansentencset[i]

N = 20000
f = open("test10.en")
englishdatatest = f.read()
f.close()
englishtest = englishdatatest.split("\n")
englishtestdoc = {}
for i in range(len(englishtest)):
	englishtest[i] = englishtest[i].split(" ")
	englishsentencsetdoc[i + N] = englishtest[i]

f = open("test10.ja")
japandatatest = f.read()
f.close()
japantest = japandatatest.split("\n")
tagger = MeCab.Tagger( '-Owakati -u /usr/local/Cellar/mecab/0.996/lib/mecab/dic/ipadic/wikipedia-keyword.dic')
japantestdoc = {}
for i in range(len(japantest)):
	japantest[i] = japantest[i].replace(" ","")
	japantest[i] = tagger.parse(japantest[i]).split(" ")[0:-1]
	japansentencsetdoc[i + N] = japantest[i]


import gensim
from gensim import corpora, models, similarities
import os

def vec2dense(vec, num_terms):
	'''Convert from sparse gensim format to dense list of numbers'''
	return list(gensim.matutils.corpus2dense([vec], num_terms=num_terms).T[0])

def createtvectorMat(bow_docs,dct):
	vectorMatlist = []
	for name in bow_docs.keys():
		if (np.array(vectorMat2)).shape[0] == 0:
			sparse = bow_docs[name]
			vectorMat2 = vec2dense(sparse, num_terms=len(dct))
			vectorMatlist.append(vectorMat2)
		else:
			sparse = bow_docs[name]
			vectorMat2 = np.c_[vectorMat2,vec2dense(sparse, num_terms=len(dct))]
	return(vectorMat2.T)

preprocessed_docs = japansentencsetdoc
dct = gensim.corpora.Dictionary(preprocessed_docs.values())
unfiltered = dct.token2id.keys()
#dct.filter_extremes(no_below=3, no_above=0.6)
#filtered = dct.token2id.keys()
#filtered_out = set(unfiltered) - set(filtered)
bow_docs = {}
bow_docs_all_zeros = {}
for name in preprocessed_docs.keys():
	sparse = dct.doc2bow(preprocessed_docs[name])
	bow_docs[name] = sparse
	dense = vec2dense(sparse, num_terms=len(dct))
	bow_docs_all_zeros[name] = all(d == 0 for d in dense)

datajapanese = createtvectorMat(bow_docs,dct)
japaneseIDdic = dict(zip(unfiltered,range(0,len(unfiltered))))

preprocessed_docs2 = englishsentencsetdoc
dct2 = gensim.corpora.Dictionary(preprocessed_docs2.values())
unfiltered2 = dct2.token2id.keys()
bow_docs2 = {}
bow_docs_all_zeros2 = {}
for name in preprocessed_docs2.keys():
	sparse2 = dct2.doc2bow(preprocessed_docs2[name])
	bow_docs2[name] = sparse2
	dense2 = vec2dense(sparse2, num_terms=len(dct2))
	bow_docs_all_zeros2[name] = all(d == 0 for d in dense2)

dataenglish = createtvectorMat(bow_docs2,dct2)
englishIDdic = dict(zip(unfiltered2,range(0,len(unfiltered2))))


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

import numpy as np
from chainer import Variable
from chainer.functions import *

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

SRC_VOCAB_SIZE = (len(unfiltered) + 1)
SRC_EMBED_SIZE  = (len(unfiltered) + 1)
HIDDEN_SIZE = 100
TRG_VOCAB_SIZE = (len(unfiltered2) + 2)
TRG_EMBED_SIZE = (len(unfiltered2) + 2)
model = FunctionSet(
	w_xi = EmbedID(SRC_VOCAB_SIZE, SRC_EMBED_SIZE), # 入力層(one-hot) -> 入力埋め込み層
	w_ip = Linear(SRC_EMBED_SIZE, 4 * HIDDEN_SIZE), # 入力埋め込み層 -> 入力隠れ層
	w_pp = Linear(HIDDEN_SIZE, 4 * HIDDEN_SIZE), # 入力隠れ層 -> 入力隠れ層
	w_pq = Linear(HIDDEN_SIZE, 4 * HIDDEN_SIZE), # 入力隠れ層 -> 出力隠れ層
	w_yq = EmbedID(TRG_VOCAB_SIZE, 4 * HIDDEN_SIZE), # 出力層(one-hot) -> 出力隠れ層
	w_qq = Linear(HIDDEN_SIZE, 4 * HIDDEN_SIZE), # 出力隠れ層 -> 出力隠れ層
	w_qj = Linear(HIDDEN_SIZE, TRG_EMBED_SIZE), # 出力隠れ層 -> 出力埋め込み層
	w_jy = Linear(TRG_EMBED_SIZE, TRG_VOCAB_SIZE), # 出力隠れ層 -> 出力隠れ層
)  


END_OF_SENTENCE = len(unfiltered2)

# src_sentence: 翻訳したい単語列 e.g. ['彼', 'は', '走る']
# trg_sentence: 正解の翻訳を表す単語列 e.g. ['he', 'runs']
# training: 学習か予測か。デコーダの挙動に影響する。
def forward(src_sentence, trg_sentence, model, training):
	# 単語IDへの変換（自分で適当に実装する）
	# 正解の翻訳には終端記号を追加しておく。
	src_sentence2 = []
	for word in src_sentence:
		try:
			src_sentence2.append(japaneseIDdic[word.decode("utf-8")])
		except:
			print word
			src_sentence2.append(SRC_VOCAB_SIZE - 1)
	#src_sentence = [ japaneseIDdic[word.decode("utf-8")] for word in src_sentence]
	#trg_sentence = [ englishIDdic[word] for word in trg_sentence] + [END_OF_SENTENCE]
	src_sentence = src_sentence2
	trg_sentence2 = []
	for word in trg_sentence:
		try:
			trg_sentence2.append(englishIDdic[word])
		except:
			print word
			trg_sentence2.append(TRG_VOCAB_SIZE - 1)
	trg_sentence2 = (trg_sentence2 + [END_OF_SENTENCE])
	trg_sentence = trg_sentence2
	#print trg_sentence
	# LSTM内部状態の初期値
	c = Variable(np.zeros((1, HIDDEN_SIZE),dtype=np.float32))
	# エンコーダ
	x = Variable(np.array([END_OF_SENTENCE], dtype=np.int32))
	i = tanh(model.w_xi(x))
	c, p = lstm(c, model.w_ip(i))
	for word in reversed(src_sentence):
		x = Variable(np.array([word], dtype=np.int32))
		i = tanh(model.w_xi(x))
		c, p = lstm(c, model.w_ip(i) + model.w_pp(p))
	# エンコーダ -> デコーダ
	c, q = lstm(c, model.w_pq(p))
	# デコーダ
	if training:
		# 学習時はyとして正解の翻訳を使い、forwardの結果として累積損失を返す。
		accum_loss = np.zeros((), dtype=np.float32)
		for word in trg_sentence:
			j = tanh(model.w_qj(q))
			y = model.w_jy(j)
			t = Variable(np.array([word], dtype=np.int32))
			accum_loss += softmax_cross_entropy(y, t)
			c, q = lstm(c, model.w_yq(t) + model.w_qq(q))
		return accum_loss
	else:
	# 予測時には翻訳器が生成したyを次回の入力に使い、forwardの結果として生成された単語列を返す。
	# yの中で最大の確率を持つ単語を選択していくが、softmaxを取る必要はない。
		hyp_sentence = []
		while len(hyp_sentence) < 100: # 100単語以上は生成しないようにする
			j = tanh(model.w_qj(q))
			y = model.w_jy(j)
			word = y.data.argmax(1)[0]
			#print word
			if word == END_OF_SENTENCE:
				break # 終端記号が生成されたので終了
			hyp_sentence.append(unfiltered2[word])
			#print word
			s_y = Variable(np.array([word], dtype=np.int32))
			c, q = lstm(c, model.w_yq(s_y) + model.w_qq(q))
		return hyp_sentence

def train(japansentencsetdoc,englishsentencsetdoc,model):
	opt = SGD() # 確率的勾配法を使用
	opt.setup(model) # 学習器の初期化
	#for sentence in sentence_set:
	for textID in range(N):
		if textID % 500 == 0:
			print "textID", textID
		opt.zero_grads(); # 勾配の初期化
		accum_loss = forward(japansentencsetdoc[textID], englishsentencsetdoc[textID], model, training = True) # 損失の計算
		accum_loss.backward() # 誤差逆伝播
		opt.clip_grads(10) # 大きすぎる勾配を抑制
		opt.update() # パラメータの更新

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

for i in range(0,2):
	print i
	train(japansentencsetdoc,englishsentencsetdoc,model)
	hyp_sentence = forward(japantest[0],englishtest[0],model, training = False)
	text = ""
	for w in japantest[0]:
		text = text + w
	print "=====問題======"
	print text
	print "=====正解======"
	print englishtest[0]
	print "=====予測======"
	print hyp_sentence


# Save final model
import pickle
pickle.dump(model, open("model20000_2.dump", 'wb'), -1)
pickle.dump(japaneseIDdic,open("japaneseIDdic.dump", 'wb'), -1)
pickle.dump(englishIDdic,open("englishIDdic.dump", 'wb'), -1)
pickle.dump(unfiltered2,open("unfiltered2.dump", 'wb'), -1)
