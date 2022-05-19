from __future__ import division
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import math

class GraphAttention(nn.Module):
    def __init__(self, in_features, out_features, dropout, edge_dropout, n_layers, activation, highway, bias=True):
        super(GraphAttention, self).__init__()        
        self.dropout = nn.Dropout(dropout)
        self.in_features = in_features
        self.n_layers = n_layers
        self.out_features = out_features
        self.edge_dropout = edge_dropout
        self.activation = activation
        self.highway = highway
        self._layers = nn.ModuleList([
            GraphConvolutionLayer(in_features,
                                  out_features,
                                  edge_dropout,
                                  activation,
                                  highway,
                                  bias) for _ in range(n_layers)])

    def forward(self, inputs, adj):
        features = inputs
        for i in range(len(self._layers)):
            features = self.dropout(self._layers[i](features, adj))
        return features
            
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
                + str(self.in_features) + ' -> ' \
                + str(self.out_features) + ', ' \
                + 'activation=' + str(self.activation) + ', ' \
                + 'highway=' + str(self.highway) + ', ' \
                + 'layers=' + str(self.n_layers) + ', ' \
                + 'dropout=' + str(self.dropout.p) + ', ' \
                + 'edge_dropout=' + str(self.edge_dropout) + ')'             

class GraphConvolutionLayer(nn.Module):
    """
    From https://github.com/tkipf/pygcn.
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, edge_dropout, activation, highway, bias=True):
        super(GraphConvolutionLayer, self).__init__()
        self.in_features = in_features
        out_features = int(out_features)
        self.out_features = out_features
        self.edge_dropout = nn.Dropout(edge_dropout)
        self.attn_dropout = nn.Dropout(0.2)
        self.alpha = 0.2
        self.highway = highway
        self.activation = activation
        self.weight = Parameter(torch.Tensor(3, in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(3, 1, out_features))
        if highway != "":
            assert(in_features == out_features)
            self.weight_highway = Parameter(torch.Tensor(in_features, out_features))
            self.bias_highway = Parameter(torch.Tensor(1, out_features))
        else:
            self.bias = None
            self.register_parameter('bias', None)         
        self.rnn = torch.nn.GRUCell(out_features, out_features, bias=bias)        ### attention
        self.a = nn.Parameter(torch.empty(size=(3, 2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)           
            
    def forward(self, inputs, adj):
        #print("INPUT SIZE: {}".format(inputs.shape))
        #print("ADJ SIZE: {}".format(adj.shape))
        features = self.edge_dropout(inputs)
        outputs = []
        #print("FEATURE SIZE: {}".format(features.shape))
        for i in range(features.size()[1]):
            support = torch.bmm(
                features[:, i, :].unsqueeze(0).expand(self.weight.size(0), *features[:, i, :].size()), 
                self.weight
            )
            #print("SUPPORT SIZE: {}".format(support.shape))
            if self.bias is not None:
                support += self.bias.expand_as(support)
            ### Attention
            N = support.size()[1]
            a = []
            for j in range(support.size()[0]):
                e = self._prepare_attentional_mechanism_input(support[j,...], self.a[j,...])
                #a_input = self._prepare_attentional_mechanism_input(support[j,...])  # (N, N, 2 * out_features)
                #a_input = torch.cat([support[j,...].repeat(1, N).view(N * N, -1), support[j,...].repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
                #e = self.leakyrelu(torch.matmul(a_input, self.a[j, ...]).squeeze(2))
                #e = torch.matmul(self.leakyrelu(a_input), self.a[j,...]).squeeze(2)
                # Masked Attention
                zero_vec = -9e15 * torch.ones_like(e)
                attention = torch.where(adj[j, i, :] > 0, e, zero_vec)
                attention = F.softmax(attention, dim=1)
                attention = self.attn_dropout(attention)
                h_prime = torch.matmul(attention, support[j,...])
                a.append(h_prime)
            support = torch.stack(a, dim = 0)
            output = torch.sum(support, dim = 0)
            outputs.append(output)
        if self.activation == "leaky_relu":
            output = F.leaky_relu(torch.stack(outputs, 1))
        elif self.activation == "relu":
            output = F.relu(torch.stack(outputs, 1))
        elif self.activation == "tanh":
            output = torch.tanh(torch.stack(outputs, 1))
        elif self.activation == "sigmoid":
            output = torch.sigmoid(torch.stack(outputs, 1))
        else:
            assert(False)

        if self.highway != "":
            transform = []
            for i in range(features.size()[1]):
                transform_batch = torch.mm(features[:, i, :], self.weight_highway)
                transform_batch += self.bias_highway.expand_as(transform_batch)
                transform.append(transform_batch)
            if self.highway == "leaky_relu":
                transform = F.leaky_relu(torch.stack(transform, 1))  
            elif self.highway == "relu":
                transform = F.relu(torch.stack(transform, 1))  
            elif self.highway == "tanh":
                transform = torch.tanh(torch.stack(transform, 1))
            elif self.highway == "sigmoid":
                transform = torch.sigmoid(torch.stack(transform, 1))
            else:
                assert(False)                
            carry = 1 - transform
            output = output * transform + features * carry
        return output


    def _prepare_attentional_mechanism_input(self, Wh, a):
        Wh1 = torch.matmul(Wh, a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)
 
