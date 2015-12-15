#http://qiita.com/odashi_t/items/a1be7c4964fbea6a116e
from chainer import FunctionSet
from chainer.functions import *

model = FunctionSet(
  w_xh = EmbedID(VOCAB_SIZE, HIDDEN_SIZE), # 入力層(one-hot) -> 隠れ層
  w_hh = Linear(HIDDEN_SIZE, HIDDEN_SIZE), # 隠れ層 -> 隠れ層
  w_hy = Linear(HIDDEN_SIZE, VOCAB_SIZE), # 隠れ層 -> 出力層
)  

import math
import numpy as np
from chainer import Variable
from chainer.functions import *

def forward(sentence, model): # sentenceはstrの配列。MeCabなどの出力を想定。
  sentence = [convert_to_your_word_id(word) for word in sentence] # 単語をIDに変換。自分で適当に実装する。
  h = Variable(np.zeros((1, HIDDEN_SIZE), dtype=np.float32)) # 隠れ層の初期値
  log_joint_prob = float(0) # 文の結合確率

  for word in sentence:
    x = Variable(np.array([[word]], dtype=np.int32)) # 次回の入力層
    y = softmax(model.w_hy(h)) # 次の単語の確率分布
    log_joint_prob += math.log(y.data[0][word]) # 結合確率の更新
    h = tanh(model.w_xh(x) + model.w_hh(h)) # 隠れ層の更新

  return log_joint_prob # 結合確率の計算結果を返す

  def forward(sentence, model):
  """
  accum_loss = Variable(np.zeros((), dtype=np.float32)) # 累積損失の初期値
  """

  for word in sentence:
    x = Variable(np.array([[word]], dtype=np.int32)) # 次回の入力層 (=今回の正解)
    u = model.w_hy(h)
    accum_loss += softmax_cross_entropy(u, x) # 損失の蓄積
    y = softmax(u)
    ...

  return log_joint_prob, accum_loss # 累積損失も一緒に返す

from chainer.optimizers import *
...

def train(sentence_set, model):
  opt = SGD() # 確率的勾配法を使用
  opt.setup(model) # 学習器の初期化
  for sentence in sentence_set:
    opt.zero_grad(); # 勾配の初期化
    log_joint_prob, accum_loss = forward(sentence, model) # 損失の計算
    accum_loss.backward() # 誤差逆伝播
    opt.clip_grads(10) # 大きすぎる勾配を抑制
    opt.update() # パラメータの更新



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




# src_sentence: 翻訳したい単語列 e.g. ['彼', 'は', '走る']
# trg_sentence: 正解の翻訳を表す単語列 e.g. ['he', 'runs']
# training: 学習か予測か。デコーダの挙動に影響する。
def forward(src_sentence, trg_sentence, model, training):

  # 単語IDへの変換（自分で適当に実装する）
  # 正解の翻訳には終端記号を追加しておく。
  src_sentence = [convert_to_your_src_id(word) for word in src_sentence]
  trg_sentence = [convert_to_your_trg_id(word) for wprd in trg_sentence] + [END_OF_SENTENCE]

  # LSTM内部状態の初期値
  c = np.zeros(1, HIDDEN_SIZE)

  # エンコーダ
  for word in reversed(src_sentence):
    x = Variable(np.array([[word]], dtype=np.int32))
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
      t = Variable(np.array([[word]], dtype=np.int32))
      accum_loss += softmax_cross_entropy(y, t)
      c, q = lstm(c, model.w_yq(t), model.w_qq(q))
    return accum_loss
  else:
    # 予測時には翻訳器が生成したyを次回の入力に使い、forwardの結果として生成された単語列を返す。
    # yの中で最大の確率を持つ単語を選択していくが、softmaxを取る必要はない。
    hyp_sentence = []
    while len(hyp_sentence) < 100: # 100単語以上は生成しないようにする
      j = tanh(model.w_qj(q))
      y = model.w_jy(j)
      word = y.data.argmax(1)[0]
      if word == END_OF_SENTENCE:
        break # 終端記号が生成されたので終了
      hyp_sentence.append(convert_to_your_trg_str(word))
      c, q = lstm(c, model.w_yq(y), model.w_qq(q))
    return hyp_sentence