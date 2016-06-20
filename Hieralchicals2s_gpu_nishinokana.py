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
import MeCab
#from normalizer import normalize,_convert_marks,_delete_cyclic_word
#from impala.dbapi import connect

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
args = parser.parse_args()

args.gpu = 2
#xp = cuda.cupy if args.gpu >= 0 else np
if args.gpu >= 0:
	xp = cuda.cupy
else:
	xp = np

"""
tagger = MeCab.Tagger( '-Owakati -u /usr/local/Cellar/mecab/0.996/lib/mecab/dic/ipadic/wikipedia-keyword.dic')

files = os.listdir('lyric/')
vocablist = []
sentenceslist_file = []
for file_name in files:
	f = open('lyric/' + file_name)
	data = f.read()
	f.close()
	sentences = data.split("\r")
	sentences = data.split("\r")
	sentenceslist = {}
	k = 0
	for sentence in sentences:
		#print sentence
		wordarrays = tagger.parse(sentence).split(" ")[0:-1]
		vocablist += wordarrays
		if sentence != '':
			if k > 1:
				sentenceslist[k-2].append(wordarrays)
		else:
			if k > 0:
				sentenceslist[k-1] = []
			k += 1
	try:
		sentenceslist_file.append(sentenceslist)
	except:
		sentenceslist_file = sentenceslist
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

#http://qiita.com/odashi_t/items/a1be7c4964fbea6a116e
from chainer import FunctionSet
from chainer.functions import *

from chainer import Variable
from chainer.optimizers import *

SRC_VOCAB_SIZE = (len(vocablist) + 3)
SRC_EMBED_SIZE  = 200
HIDDEN_SIZE = 100
TRG_VOCAB_SIZE = (len(vocablist) + 3)
TRG_EMBED_SIZE = 200
model = FunctionSet(
	w_xi = EmbedID(SRC_VOCAB_SIZE, SRC_EMBED_SIZE), # 入力層(one-hot) -> 入力埋め込み層
	w_ip = Linear(SRC_EMBED_SIZE, 4 * HIDDEN_SIZE), # 入力埋め込み層 -> 入力隠れ層
	w_pp = Linear(HIDDEN_SIZE, 4 * HIDDEN_SIZE), # 入力隠れ層 -> 入力隠れ層
	w_pI = Linear(HIDDEN_SIZE, HIDDEN_SIZE), # 入力隠れ層 -> 文入力埋め込み層
	w_IP = Linear(HIDDEN_SIZE, 4 * HIDDEN_SIZE), # 文入力埋め込み層 -> 文入力隠れ層
	w_PP = Linear(HIDDEN_SIZE, 4 * HIDDEN_SIZE), # 文入力隠れ層 -> 文入力隠れ層
	w_PQ = Linear(HIDDEN_SIZE, 4 * HIDDEN_SIZE), # 文入力隠れ層 -> 文出力隠れ層
	w_QQ = Linear(HIDDEN_SIZE, 4 * HIDDEN_SIZE), # 文出力隠れ層 -> 文出力隠れ層
	w_Qq = Linear(HIDDEN_SIZE, HIDDEN_SIZE), # 文出力隠れ層 -> 出力埋め込み層
	w_qQ = Linear(HIDDEN_SIZE, 4 * HIDDEN_SIZE), # 出力埋め込み層 -> 文出力隠れ層
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
				src_sentence2.append(vocabIDdic[word])
				#src_sentence2.append(word)
			except:
				print word
				src_sentence2.append(SRC_VOCAB_SIZE - 1)
		#print src_sentence
		src_sentence = src_sentence2
		#print src_sentence
		src_sentenceList.append(src_sentence)
	trg_sentenceList = []
	for i,trg_sentence in enumerate(trg_sentences):
		trg_sentence2 = []
		for word in trg_sentence:
			try:
				trg_sentence2.append(vocabIDdic[word])
				#trg_sentence2.append(word)
			except:
				print word
				trg_sentence2.append(TRG_VOCAB_SIZE - 1)
		if i < len(trg_sentences) - 1:
			trg_sentence2 = (trg_sentence2 + [END_OF_SENTENCE])
		else:
			trg_sentence2 = (trg_sentence2 + [END_OF_SENTENCES])
		#print trg_sentence
		trg_sentence = trg_sentence2
		#print trg_sentence
		trg_sentenceList.append(trg_sentence)
		# LSTM内部状態の初期値
	#X_list = []
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
		X = model.w_pI(p)
		X_list.append(X)
		#X_list.append(p)
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
		hyp_sentences = []
		hyp_sentence = []
		for trg_sentence in trg_sentenceList:
			accum_loss = chainer.Variable(xp.zeros((), dtype=np.float32))
			for word in trg_sentence:
				q = model.w_Qq(Q)
				j = tanh(model.w_qj(q))
				#j = tanh(model.w_qj(Q))
				y = model.w_jy(j)
				word2 = y.data.argmax(1)[0]
				t = Variable(xp.array([word], dtype=np.int32))
				loss = softmax_cross_entropy(y, t)
				#print loss.data
				accum_loss += loss
				c, q = lstm(c, model.w_yq(t) + model.w_qq(q))
				hyp_sentence.append(word2)
			hyp_sentences.append(hyp_sentence)
			C, Q = lstm(C, model.w_qQ(q) + model.w_QQ(Q))
		return accum_loss, hyp_sentences, trg_sentenceList
	else:
		hyp_sentences = []
		hyp_sentence = []
	# 予測時には翻訳器が生成したyを次回の入力に使い、forwardの結果として生成された単語列を返す。
	# yの中で最大の確率を持つ単語を選択していくが、softmaxを取る必要はない。
		while len(hyp_sentences) < 2: #10文以上は生成しないようにする
			q = model.w_Qq(Q)
			j = tanh(model.w_qj(q))
			#j = tanh(model.w_qj(Q))
			y = model.w_jy(j)
			word = y.data.argmax(1)[0]
			t = Variable(xp.array([word], dtype=np.int32))
			c, q = lstm(c, model.w_yq(t) + model.w_qq(q))
			#print word
			if word == END_OF_SENTENCES:
				hyp_sentences.append(hyp_sentence)
				break # 終端記号が生成されたので終了
			if (word == END_OF_SENTENCE) | (len(hyp_sentence) >= 20): # 20単語以上は生成しないようにする
				hyp_sentences.append(hyp_sentence)
				C, Q = lstm(C, model.w_qQ(q) + model.w_QQ(Q))
				hyp_sentence = []
		return hyp_sentences

forward(sentenceslist_file[1][2], sentenceslist_file[1][2], model, training = True)
N = 1000

def train(japansentencsetdocList,englishsentencsetdocList,model,N = N):
	#perm = np.random.permutation(N)
	opt = SGD() # 確率的勾配法を使用
	#opt = Adam()
	opt.setup(model) # 学習器の初期化
	#for sentence in sentence_set:
	#perm = np.random.permutation(N)
	accum_loss_sum = chainer.Variable(xp.zeros((), dtype=np.float32))
	#for i, textID in enumerate(np.array(range(N))[perm]):
	for k in range(1,N):
		japansentencsetdoc = japansentencsetdocList[k]
		#print japansentencsetdocList[k]
		englishsentencsetdoc = englishsentencsetdocList[k]
		for i, textID in enumerate(np.array(range(1,len(japansentencsetdoc)))):
			opt.zero_grads() # 勾配の初期化
			#print textID
			accum_loss,_,_ = forward(japansentencsetdoc[textID], englishsentencsetdoc[textID], model, training = True) # 損失の計算
			accum_loss_sum += accum_loss
			accum_loss.backward() # 誤差逆伝播
			opt.clip_grads(10) # 大きすぎる勾配を抑制
			opt.update() # パラメータの更新
	print accum_loss_sum.data

train(sentenceslist_file, sentenceslist_file,model,N = 2)

"""
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
		accum_loss,hyps, trgs = forward(japansentencsetdoc[textID], englishsentencsetdoc[textID], model, training = True) # 損失の計算
		accum_loss_sum += accum_loss
		accum_loss_sum2 += accum_loss
		#print i, textID
		if i % batchsize == 0:
			#accum_loss_sum = accum_loss_sum / chainer.Variable(batch_size_array)
			accum_loss_sum.backward() # 誤差逆伝播
			#opt.clip_grads(10) # 大きすぎる勾配を抑制
			opt.update() # パラメータの更新
			#print accum_loss_sum.data
			accum_loss_sum = chainer.Variable(xp.zeros((), dtype=np.float32))
			opt.zero_grads() # 勾配の初期化
	accum_loss_sum2 = accum_loss_sum2/ chainer.Variable(xp.array(N, dtype=xp.float32))
	print accum_loss_sum2.data
	print hyps
	print trgs
"""

def Test(japantest,englishtest,n):
	text = ""
	hyp_sentencelist = []
	hyp_sentences = forward(japantest[n],englishtest[n],model, training = False)
	for sentence in japantest[n]:
		for w in sentence:
			text = text + w
		text += "\n"
			#text = text + vocabworddic[w]
	print "=====問題======"
	print text
	print "=====正解======"
	print text
	print "=====予測======"
	for s in hyp_sentences:
		for w in s:
			try:
				hyp_sentencelist.append(vocabworddic[int(w)])
			except:
				hyp_sentencelist.append(str(w) + "unk")
		print ' '.join(hyp_sentencelist)
		hyp_sentencelist = []

hyp_sentence = forward(sentenceslist_file[1][2],sentenceslist_file[1][2],model, training = True)
Test(sentenceslist_file[1], sentenceslist_file[1],0)
#vocabworddic



for i in range(0,200):
	print i
	train(sentenceslist_file, sentenceslist_file,model,N = 100)
	#hyp_sentence = forward(smallyahooboardsentences[0],smallyahooboardsentences[0],model, training = False)
	hyp_sentence = forward(sentenceslist_file[1][2],sentenceslist_file[1][2],model, training = True)
	print hyp_sentence[0]
	print hyp_sentence[1]
	print hyp_sentence[2]
	Test(sentenceslist_file[1], sentenceslist_file[1],0)
	Test(sentenceslist_file[100], sentenceslist_file[100],0)


# Save final model
import pickle
pickle.dump(model, open("lyric/Hieralchicals2s.dump", 'wb'), -1)
#pickle.dump(model, open("model20000_2.dump", 'wb'), -1)
pickle.dump(model.to_cpu(), open("model20000_2.dump", 'wb'), -1)
pickle.dump(japaneseIDdic,open("japaneseIDdic.dump", 'wb'), -1)
pickle.dump(englishIDdic,open("englishIDdic.dump", 'wb'), -1)
pickle.dump(unfiltered2,open("unfiltered2.dump", 'wb'), -1)
