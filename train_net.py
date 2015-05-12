#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals, absolute_import
import brainstorm as bs
from sacred import Experiment
import numpy as np
import h5py
import tempfile


ex = Experiment('nascence_simdata')


@ex.config
def cfg():
    num_layers = 5
    layer_size = 50
    act_func = 'rel'
    filename = 'simdata2.hdf5'
    batch_size = 20
    learning_rate = 0.003
    in_size = 8
    max_epochs = 100


@ex.capture
def build_network(num_layers, layer_size, act_func, in_size):
    inp = bs.InputLayer(out_shapes={'input_data': ('T', 'B', in_size),
                                    'targets': ('T', 'B', 1)})
    err = bs.SquaredDifferenceLayer()

    current_layer = inp - 'input_data'

    for i in range(num_layers):
        current_layer = current_layer >> bs.FullyConnectedLayer(layer_size, activation_function=act_func)

    current_layer >> bs.FullyConnectedLayer(1) >> 'inputs_1' - err >> bs.LossLayer()
    inp - 'targets' >> 'inputs_2' - err
    net = bs.build_net(inp)

    scaling = {
        'rel': np.sqrt(12),
        'tanh': np.sqrt(6),
        'sigmoid': 4 * np.sqrt(6),
        'linear': 1
    }[act_func]

    net.initialize(bs.DenseSqrtFanInOut(scaling),
                   fallback=bs.Gaussian(0.1))

    return net


@ex.capture
def get_data_iters(filename, batch_size):
    ds = h5py.File(filename, 'r')
    train_data = ds['training']['input_data'][()]
    train_targets = ds['training']['targets'][()]
    val_data = ds['validation']['input_data'][()]
    val_targets = ds['validation']['targets'][()]
    train_iter = bs.Minibatches(batch_size, input_data=train_data, targets=train_targets)
    val_iter = bs.Minibatches(200, input_data=val_data, targets=val_targets)
    return train_iter, val_iter


@ex.capture
def get_trainer(learning_rate, _run, max_epochs):
    tr = bs.Trainer(bs.SgdStep(learning_rate=learning_rate), verbose=True,
                    double_buffering=True)
    tr.add_monitor(bs.MaxEpochsSeen(max_epochs))
    tr.add_monitor(bs.MonitorLoss('val_iterator'))
    tr.add_monitor(bs.SaveBestWeights('MonitorLoss', filename='best_weights.npy'))
    tr.add_monitor(bs.ErrorRises('MonitorLoss', delay=5))
    tr.add_monitor(bs.InfoUpdater(run=_run))
    tr.add_monitor(bs.StopOnNan('MonitorLoss'))
    return tr


@ex.automain
def run():
    net = build_network()
    train_iter, val_iter = get_data_iters()
    tr = get_trainer()
    tr.train(net, train_iter, val_iterator=val_iter)

    ex.add_artifact('best_weights.npy')
    return np.min(tr.logs['MonitorLoss'])

