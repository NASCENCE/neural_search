#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals, absolute_import
import brainstorm as bs
from brainstorm.training.steppers import TrainingStepper
from sacred import Experiment
import numpy as np
import h5py
import tempfile
import pymongo


db = pymongo.MongoClient().nascence_reloaded
ex = Experiment('NascenceReloaded')

@ex.config
def cfg():
    num_layers = 3
    layer_size = 20
    act_func = 'tanh'
    net_spec = "F{}{} ".format(act_func[0], layer_size) * num_layers

    filename = 'data50K.h5'
    net_filename = None
    batch_size = 128
    learning_rate = 0.01
    patience = 5
    in_size = 8
    max_epochs = 50
    use_adam = False

    verbose = True

@ex.named_config
def best_net():
    filename = 'data50K.h5'
    layer_size = 120
    act_func = 'el'
    num_layers = 5
    use_adam = True
    net_filename = 'best_net.h5'


@ex.capture
def build_network(net_spec, in_size, _log):
    _log.info('Creating network with spec "{}"'.format(net_spec))
    net = bs.tools.create_net_from_spec("regression", in_size, 1, net_spec)
    #net.set_handler(bs.handlers.PyCudaHandler())
    return net


@ex.capture
def get_data_iters(filename, batch_size):
    ds = h5py.File(filename, 'r')
    train_data = ds['training']['default'][()]
    train_targets = ds['training']['targets'][()]
    val_data = ds['validation']['default'][()]
    val_targets = ds['validation']['targets'][()]
    bs.data_iterators
    train_iter = bs.data_iterators.Minibatches(batch_size, default=train_data,
                                               targets=train_targets)
    val_iter = bs.data_iterators.Minibatches(256, default=val_data,
                                             targets=val_targets)
    return train_iter, val_iter


@ex.capture
def get_trainer(learning_rate, max_epochs, verbose, net_filename, patience, use_adam, _run):
    if use_adam:
        tr = bs.Trainer(AdamStepper(), verbose=verbose)
    else:
        tr = bs.Trainer(bs.training.SgdStepper(learning_rate=learning_rate),
                        verbose=verbose)
    if verbose:
        tr.add_hook(bs.hooks.ProgressBar())
    tr.add_hook(bs.hooks.StopAfterEpoch(max_epochs))
    tr.add_hook(bs.hooks.MonitorLoss('val_iterator'))
    tr.add_hook(bs.hooks.StopOnNan(['MonitorLoss.total_loss']))
    if net_filename:
        tr.add_hook(bs.hooks.SaveBestNetwork('MonitorLoss.total_loss', net_filename,
                                             criterion='min'))
    else:
        tr.add_hook(bs.hooks.SaveBestNetwork('MonitorLoss.total_loss', 
                                             criterion='min'))
    tr.add_hook(bs.hooks.EarlyStopper('MonitorLoss.total_loss', patience=patience))
    tr.add_hook(bs.hooks.InfoUpdater(run=_run))
    return tr



class AdamStepper(TrainingStepper):
    """
    Adam optimizer.
    Decay rate lamb (lambda) decays beta1, thus slowly increasing momentum.
    For more detailed information see "Adam: A Method for Stochastic Optimization" by Kingma and Ba.
    """
    __undescribed__ = {'m_0', 'v_0'}

    def __init__(self, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, lamb=1-1e-8):
        super(AdamStepper, self).__init__()
        self.m_0 = None
        self.v_0 = None
        self.time_step = None
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.lamb = lamb

    def start(self, net):
        super(AdamStepper, self).start(net)
        self.m_0 = net.handler.zeros(net.buffer.parameters.shape)
        self.v_0 = net.handler.zeros(net.buffer.parameters.shape)
        self.time_step = 0

    def run(self):
        self.time_step += 1
        self.beta1 *= self.lamb
        t = self.time_step
        learning_rate = self.alpha
        self.net.forward_pass(training_pass=True)
        self.net.backward_pass()

        gradient = self.net.buffer.gradients
        temp = self.net.handler.allocate(gradient.shape)
        temp_m0 = self.net.handler.allocate(self.m_0.shape)
        temp_v0 = self.net.handler.allocate(self.v_0.shape)

        # m_t <- beta_1*m_{t-1} + (1-beta1) *gradient
        self.net.handler.mult_st(self.beta1, self.m_0, out=self.m_0)
        self.net.handler.mult_add_st(1.0-self.beta1, gradient, out=self.m_0)
        # v_t <- beta_2*v_{t-1} + (1-beta2) *gradient^2
        self.net.handler.mult_st(self.beta2, self.v_0, out=self.v_0)
        self.net.handler.mult_tt(gradient, gradient, temp)  # gradient^2
        self.net.handler.mult_add_st(1.0-self.beta2, temp, out=self.v_0)
        # m_hat_t <- m_t/(1-beta1^t)
        self.net.handler.mult_st(1.0/(1.0-pow(self.beta1, t)), self.m_0, out=temp_m0)
        # v_hat_t <- v_t/(1-beta2^t)
        self.net.handler.mult_st(1.0/(1.0-pow(self.beta2, t)), self.v_0, out=temp_v0)

        self.net.handler.sqrt_t(temp_v0, temp_v0)
        self.net.handler.add_st(self.epsilon, temp_v0, out=temp_v0)

        self.net.handler.mult_st(learning_rate, temp_m0, out=temp_m0)

        self.net.handler.divide_tt(temp_m0, temp_v0, temp)

        self.net.handler.subtract_tt(self.net.buffer.parameters, temp, out=self.net.buffer.parameters)


@ex.automain
def run(net_filename):
    net = build_network()
    train_iter, val_iter = get_data_iters()
    tr = get_trainer()
    tr.train(net, train_iter, val_iterator=val_iter)
    
    if net_filename:
        ex.add_artifact(net_filename)

    return np.min(tr.logs['MonitorLoss']['total_loss'])
