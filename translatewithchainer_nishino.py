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
TRG_VOCAB_SIZE = (len(vocablist) + 3)
TRG_EMBED_SIZE = 200
model = FunctionSet(
	w_xi = EmbedID(SRC_VOCAB_SIZE, SRC_EMBED_SIZE), # 入力層(one-hot) -> 入力埋め込み層
	w_ip = Linear(SRC_EMBED_SIZE, 4 * HIDDEN_SIZE), # 入力埋め込み層 -> 入力隠れ層
	w_pp = Linear(HIDDEN_SIZE, 4 * HIDDEN_SIZE), # 入力隠れ層 -> 入力隠れ層
	w_pq = Linear(HIDDEN_SIZE, 4 * HIDDEN_SIZE), # 入力隠れ層 -> 出力隠れ層
	#w_yq = EmbedID(TRG_VOCAB_SIZE, 4 * HIDDEN_SIZE), # 出力層(one-hot) -> 出力隠れ層
	w_iq = Linear(TRG_EMBED_SIZE, 4 * HIDDEN_SIZE), # 出力層(one-hot) -> 出力隠れ層
	w_qq = Linear(HIDDEN_SIZE, 4 * HIDDEN_SIZE), # 出力隠れ層 -> 出力隠れ層
	w_qj = Linear(HIDDEN_SIZE, TRG_EMBED_SIZE), # 出力隠れ層 -> 出力埋め込み層
	w_jy = Linear(TRG_EMBED_SIZE, TRG_VOCAB_SIZE), # 出力隠れ層 -> 出力隠れ層
)  

if args.gpu >= 0:
    cuda.check_cuda_available()
    cuda.get_device(args.gpu).use()
    model.to_gpu()


END_OF_SENTENCE = len(vocablist)
END_OF_SENTENCES = (len(vocablist) + 1)
import copy
sentenceslist = []
k  = 0
for j,sentences in enumerate(copy.deepcopy(sentenceslist_file)):
	if j == 100:
		print k
	k += len(sentences)
	for sentence_id in sentences:
		sentence_block = []
		for sentence in sentences[sentence_id]:
			sentence.append("End_sentence")
			sentence_block += sentence
		sentenceslist.append(sentence_block)

vocabIDdic["End_sentence"] = END_OF_SENTENCE
vocabworddic[END_OF_SENTENCE] = "End_sentence"
			
# src_sentence: 翻訳したい単語列 e.g. ['彼', 'は', '走る']
# trg_sentence: 正解の翻訳を表す単語列 e.g. ['he', 'runs']
# training: 学習か予測か。デコーダの挙動に影響する。
def forward(src_sentence, trg_sentence, model, training):
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
	c = Variable(xp.zeros((1, HIDDEN_SIZE),dtype=np.float32))
	# エンコーダ
	x = Variable(xp.array([END_OF_SENTENCES], dtype=np.int32))
	i = tanh(model.w_xi(x))
	c, p = lstm(c, model.w_ip(i))
	for word in reversed(src_sentence):
		x = Variable(xp.array([word], dtype=np.int32))
		i = tanh(model.w_xi(x))
		c, p = lstm(c, model.w_ip(i) + model.w_pp(p))
	# エンコーダ -> デコーダ
	c, q = lstm(c, model.w_pq(p))
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
			c, q = lstm(c, model.w_iq(model.w_xi(t)) + model.w_qq(q))
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
			#print word
			if word == END_OF_SENTENCES:
				hyp_sentence.append("unk" + str(END_OF_SENTENCES))
				break # 終端記号が生成されたので終了
			hyp_sentence.append(vocabworddic[int(word)])
			#print word
			c, q = lstm(c, model.w_iq(model.w_xi(t)) + model.w_qq(q))
		return hyp_sentence

forward(sentenceslist[0], sentenceslist[0], model, training = True)

def train(japansentencsetdoc,englishsentencsetdoc,model,N):
	#opt = SGD() # 確率的勾配法を使用
	opt = Adam()
	opt.setup(model) # 学習器の初期化
	#for sentence in sentence_set:
	accum_loss_sum = Variable(xp.zeros((), dtype=np.float32))
	for textID in range(N):
		#if textID % 500 == 0:
			#print "textID", textID
		opt.zero_grads(); # 勾配の初期化
		accum_loss = forward(japansentencsetdoc[textID], englishsentencsetdoc[textID], model, training = True) # 損失の計算
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

