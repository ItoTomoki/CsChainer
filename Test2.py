#encoding:utf-8
import json
import os
from ast import literal_eval
import re
import MeCab
import unicodedata
import sys
import ngram
import jcconv
from chainer import FunctionSet
from chainer.functions import *
import numpy as np
from chainer import Variable
from chainer.functions import *
from chainer.optimizers import *

import pickle
#model = pickle.load(open("modelsample2/model20000_2.dump", 'r'))
model = pickle.load(open("model20000_2.dump", 'r'))
japaneseIDdic = pickle.load(open("modelsample2/japaneseIDdic.dump", 'r'))
englishIDdic = pickle.load(open("modelsample2/englishIDdic.dump", 'r'))
unfiltered2 = pickle.load(open("modelsample2/unfiltered2.dump", 'r'))
SRC_VOCAB_SIZE = model.w_xi.W.shape[0]
TRG_VOCAB_SIZE = model.w_yq.W.shape[0]
HIDDEN_SIZE = 100
END_OF_SENTENCE = (TRG_VOCAB_SIZE - 2)
def forward(src_sentence, trg_sentence, model, training):
	# 単語IDへの変換（自分で適当に実装する）
	# 正解の翻訳には終端記号を追加しておく。
	src_sentence2 = []
	for word in src_sentence:
		try:
			src_sentence2.append(japaneseIDdic[word.decode("utf-8")])
		except:
			print ("未知語：" + word)
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
	"""
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
	"""
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


f = open("test10.en")
englishdatatest = f.read()
f.close()
englishtest = englishdatatest.split("\n")
englishtestdoc = {}
for i in range(len(englishtest)):
	englishtest[i] = englishtest[i].split(" ")

f = open("test10.ja")
japandatatest = f.read()
f.close()
japantest = japandatatest.split("\n")
tagger = MeCab.Tagger( '-Owakati -u /usr/local/Cellar/mecab/0.996/lib/mecab/dic/ipadic/wikipedia-keyword.dic')
japantestdoc = {}
for i in range(len(japantest)):
	japantest[i] = japantest[i].replace(" ","")
	japantest[i] = tagger.parse(japantest[i]).split(" ")[0:-1]

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
	wordarrays = tagger.parse(sentence.encode("utf-8")).split(" ")[0:-1]
	#wordarrays = sentence
	#print wordarrays
	#print (wordarrays == japantest[0])
	hyp_sentence = forward(wordarrays,['did', 'you', 'clean', 'your', 'room', '?'],model,training = False)
	print hyp_sentence

#for n in range(0,10):
	#Test(n)

import codecs
import sys

def jikkou():
	print "Please Text:(if you want to quit, please input”やめる” )"
	x = raw_input().decode("utf-8")
	#x = sys.stdin.read()
	if x == u"やめる":
		print "Bye"
	else:
		#print x
		test(x)
		jikkou()

jikkou()