from __future__ import print_function
import argparse
import math
import sys
import time
from numpy import random

import numpy as np
import six

import chainer
from chainer import cuda
import chainer.functions as F
from chainer import optimizers


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
args = parser.parse_args()
xp = cuda.cupy if args.gpu >= 0 else np

n_epoch = 39   # number of epochs
n_units = 650  # number of units per layer
batchsize = 20   # minibatch size
bprop_len = 35   # length of truncated BPTT
grad_clip = 5    # gradient norm threshold to clip

# Prepare dataset (preliminary download dataset by ./download.py)
vocab = {}


def load_data(filename):
    global vocab, n_vocab
    wordsarray = open(filename).read().split('\n')
    dataset = {}
    for i1,s in enumerate(wordsarray):
        words = s.strip().split()
        dataset[i1] = np.ndarray((len(words),), dtype=np.int32)
        for i, word in enumerate(words):
            if word not in vocab:
                vocab[word] = len(vocab)
            dataset[i1][i] = vocab[word]
    return np.array(dataset.values())

#train_data = load_data('ptb.train.txt')
#valid_data = load_data('ptb.valid.txt')
#test_data = load_data('ptb.test.txt')
enata = load_data('train20000.en')
train_data  = enata[0:5000]
valid_data = enata[5000:6000]
valid_data = enata[6000:7000]
print('#vocab =', len(vocab))

# Prepare RNNLM model
model = chainer.FunctionSet(embed=F.EmbedID(len(vocab), n_units),
                            l1_x=F.Linear(n_units, 4 * n_units),
                            l1_h=F.Linear(n_units, 4 * n_units),
                            l2_x=F.Linear(n_units, 4 * n_units),
                            l2_h=F.Linear(n_units, 4 * n_units),
                            l3=F.Linear(n_units, len(vocab)))

for param in model.parameters:
    param[:] = np.random.uniform(-0.1, 0.1, param.shape)

if args.gpu >= 0:
    cuda.check_cuda_available()
    cuda.get_device(args.gpu).use()
    model.to_gpu()


def forward_one_step(x_data, y_data, state, train=True):
    # Neural net architecture
    x = chainer.Variable(x_data, volatile=not train)
    t = chainer.Variable(y_data, volatile=not train)
    h0 = model.embed(x)
    h1_in = model.l1_x(F.dropout(h0, train=train)) + model.l1_h(state['h1'])
    c1, h1 = F.lstm(state['c1'], h1_in)
    h2_in = model.l2_x(F.dropout(h1, train=train)) + model.l2_h(state['h2'])
    c2, h2 = F.lstm(state['c2'], h2_in)
    y = model.l3(F.dropout(h2, train=train))
    state = {'c1': c1, 'h1': h1, 'c2': c2, 'h2': h2}
    return state, F.softmax_cross_entropy(y, t)


def make_initial_state(batchsize=batchsize, train=True):
    return {name: chainer.Variable(xp.zeros((batchsize, n_units),
                                            dtype=np.float32),
                                   volatile=not train)
            for name in ('c1', 'h1', 'c2', 'h2')}

# Setup optimizer
#optimizer = optimizers.SGD(lr=1.)
optimizer = optimizers.Adam()
optimizer.setup(model)


# Evaluation routine


def evaluate(datasets):
    sum_log_perp = xp.zeros(())
    state = make_initial_state(batchsize=1, train=False)
    all_length = 0
    for dataset in datasets:
        for i in six.moves.range(dataset.size - 1):
            x_batch = xp.asarray(dataset[i:i + 1])
            y_batch = xp.asarray(dataset[i + 1:i + 2])
            state, loss = forward_one_step(x_batch, y_batch, state, train=False)
            sum_log_perp += loss.data.reshape(())
        all_length += (dataset.size - 1)
    return math.exp(cuda.to_cpu(sum_log_perp) / (all_length))


# Learning loop
whole_len = train_data.shape[0]
jump = whole_len // batchsize
cur_log_perp = xp.zeros(())
epoch = 0
start_at = time.time()
cur_at = start_at
state = make_initial_state(batchsize=1)
accum_loss = chainer.Variable(xp.zeros((), dtype=np.float32))
print('going to train {} iterations'.format(jump * n_epoch))
#for i in six.moves.range(jump * n_epoch):
    #x_batch = xp.array([train_data[(jump * j + i) % whole_len]
                        #for j in six.moves.range(batchsize)])
    #y_batch = xp.array([train_data[(jump * j + i + 1) % whole_len]
                        #for j in six.moves.range(batchsize)])

for i in six.moves.range(n_epoch):
    perm = random.permutation(len(train_data))
    for j in xrange(jump):
        alllength = 0
        train_data_batch = np.array(train_data[perm][j:(j+batchsize)])
        for words in train_data_batch:
            for cur_word,next_word  in zip(words, words[1:]):
                cur_word = xp.array(cur_word).reshape(1,)
                next_word = xp.array(next_word).reshape(1,)
                state, loss_i = forward_one_step(cur_word, next_word, state)
                accum_loss += loss_i
                cur_log_perp += loss_i.data.reshape(())
            alllength += (len(words) - 1)
        batch_size_array = np.array(batchsize, dtype=np.float32)
        if args.gpu >= 0:
            batch_size_array = cuda.to_gpu(batch_size_array)
        accum_loss = accum_loss / chainer.Variable(batch_size_array)
        #epoch_loss += accum_loss.data * batchsize
        # Run truncated BPTT
        optimizer.zero_grads()
        accum_loss.backward()
        accum_loss.unchain_backward()  # truncate
        accum_loss = chainer.Variable(xp.zeros((), dtype=np.float32))
        optimizer.clip_grads(grad_clip)
        optimizer.update()
    if (i) % 1 == 0:
        now = time.time()
        throuput = i / (now - cur_at)
        perp = evaluate(train_data)
        print('iter {} training perplexity: {:.2f} ({:.2f} iters/sec)'.format(
            i + 1, perp, throuput))
        cur_at = now
        cur_log_perp.fill(0)
    if (i) % 1 == 0:
        epoch += 1
        print('evaluate')
        now = time.time()
        perp = evaluate(valid_data)
        print('epoch {} validation perplexity: {:.2f}'.format(epoch, perp))
        #cur_at += time.time() - now  # skip time of evaluation
        #if epoch >= 6:
            #optimizer.lr /= 1.2
            #print('learning rate =', optimizer.lr)
    sys.stdout.flush()

# Evaluate on test dataset
print('test')
test_perp = evaluate(test_data)
print('test perplexity:', test_perp)