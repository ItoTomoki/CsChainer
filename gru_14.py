# Prepare RNNLM model
model = chainer.FunctionSet(embed=F.EmbedID(len(vocab), n_units),
                            l1_x=F.Linear(n_units, 4 * n_units),
                            l1_h=F.Linear(n_units, 4 * n_units),
                            l2_x=F.Linear(n_units, 4 * n_units),
                            l2_h=F.Linear(n_units, 4 * n_units),
                            l3=F.Linear(n_units, len(vocab)))

#gru_model
model = chainer.FunctionSet(embed=F.EmbedID(len(vocab), n_units),
                            l1_x=F.Linear(n_units, 6 * n_units),
                            l1_h=F.Linear(n_units, 6 * n_units),
                            l2_x=F.Linear(n_units, 6 * n_units),
                            l2_h=F.Linear(n_units, 6 * n_units),
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
    c1, h1 = GRU(state['c1'],h0, h1_in)
    h2_in = model.l2_x(F.dropout(h1, train=train)) + model.l2_h(state['h2'])
    c2, h2 = GRU(state['c2'], h2_in)
    y = model.l3(F.dropout(h2, train=train))
    state = {'c1': c1, 'h1': h1, 'c2': c2, 'h2': h2}
    return state, F.softmax_cross_entropy(y, t)


def make_initial_state(batchsize=batchsize, train=True):
    return {name: chainer.Variable(xp.zeros((batchsize, n_units),
                                            dtype=np.float32),
                                   volatile=not train)
            for name in ('c1', 'h1', 'c2', 'h2')}

# Setup optimizer
optimizer = optimizers.SGD(lr=1.)
#optimizer = optimizers.Adam()
optimizer.setup(model)


# Evaluation routine


def evaluate(dataset):
    sum_log_perp = xp.zeros(())
    state = make_initial_state(batchsize=1, train=False)
    for i in six.moves.range(dataset.size - 1):
        x_batch = xp.asarray(dataset[i:i + 1])
        y_batch = xp.asarray(dataset[i + 1:i + 2])
        state, loss = forward_one_step(x_batch, y_batch, state, train=False)
        sum_log_perp += loss.data.reshape(())
    return math.exp(cuda.to_cpu(sum_log_perp) / (dataset.size - 1))