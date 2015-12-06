import theano
import theano.tensor as tensor
import theano.nnet as nn
#from recur.mixins.parameterizable import Parameterizable
#from recur.initializations import *

from affine import Affine

class MLP():
    def __init__(self, input_size, output_size, hidden_dims=[],
                 nonlinearity=tensor.nnet.sigmoid,
                 name=None, weight_init=Constant(0),
                 bias_init=Constant(0)):
        self.nonlinearity = nonlinearity

        name = "unnamed" if name is None else name

        self.layers = []

        # add intermediate layers
        cur_dim = input_size
        for hidden_dim in hidden_dims:
            layer_name = "{:s}_layer{:d}".format(name, len(self.layers))
            self.layers.append(Affine(cur_dim, hidden_dim, name=layer_name,
                                      weight_init=weight_init, bias_init=bias_init))
            cur_dim = hidden_dim

        # add output layer
        layer_name = "{:s}_layer{:d}".format(name, len(self.layers))
        self.layers.append(Affine(cur_dim, output_size, name=layer_name,
                                  weight_init=weight_init, bias_init=bias_init))


        self.local_params = []
        self.local_static_shared_vars = []
        self.children = self.layers

    def __call__(self, x):
        out = x
        cost
        for layer in self.layers:
            out = self.nonlinearity(layer(out))

        return out

    def mlp_grad(self, inputs, targets):
        preds = self.__call__(inputs)
        cost = nn.categorical_crossentropy(preds, targets)
        grads = theano.grad(cost, params)
