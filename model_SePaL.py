#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import relabel


class MultiScaleGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, params):
        super(MultiScaleGAT, self).__init__()
        self.gat = [GAT(nfeat, nhid, nclass, dropout, alpha, nheads, params).to(params['device']) for _ in range(params['scale'])]
        self.l = nn.Linear(nfeat * params['scale'], nfeat)
    def forward(self, x, adj):
        x = torch.cat([gat(x, adj) for gat in self.gat], dim=1).squeeze(0)
        return F.softmax(self.l(x), dim=1)

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, params):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.params = params
        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):

        x = x.squeeze(0)
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1).squeeze(0)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.softmax(x, dim=1)

class GraphAttentionLayer(nn.Module):

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):

        h = torch.mm(input, self.W)
        N = h.size()[0]
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class Event_encoder(nn.Module):

    def __init__(self, params):
        super(Event_encoder, self).__init__()

        H = params['H_dim']
        self.h_dim = params['h_dim']
        self.x_dim = params['x_dim']

        self.h = torch.nn.Sequential(
            torch.nn.Linear(self.x_dim, H),
            torch.nn.PReLU(),
            torch.nn.Linear(H, self.h_dim),
        )

    def forward(self, x):
        return self.h(x)


class SePaL(nn.Module):

    def __init__(self, params):

        super(SePaL, self).__init__()

        self.params = params
        self.previous_n = 0
        self.previous_K = 1

        self.token_embeddings = nn.Embedding(params['event_num'], params['x_dim'])

        self.f_dim = params['f_dim']
        self.h_dim = params['h_dim']
        self.c_dim = params['c_dim']
        self.x_dim = params['x_dim']
        H = params['H_dim']

        self.device = params['device']

        if self.params['model'] == 'Event':
            self.h = Event_encoder(params)
            self.f = Event_encoder(params)
            self.c = Event_encoder(params)
        else:
            raise NameError('Unknown model ' + self.params['model'])

        self.u = torch.nn.Sequential(
            torch.nn.Linear(self.f_dim + self.h_dim + self.c_dim, H),
            torch.nn.PReLU(),
            torch.nn.Linear(H, 1, bias=False),
        )

        self.g = torch.nn.Sequential(
            torch.nn.Linear(self.f_dim, H),
            torch.nn.PReLU(),
            torch.nn.Linear(H, self.f_dim, bias=False),
        )

        self.ms_gat = MultiScaleGAT(nfeat=self.x_dim, nhid=params['h1_dim'], nclass=self.x_dim, dropout=0.0, nheads=params['nheads'], alpha=0.1, params=params)

        self.lsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, data, cs, n, adj):

        assert (n == self.previous_n + 1)
        self.previous_n = self.previous_n + 1

        K = len(set(cs[:n]))  # num of already created clusters

        if n == 1:

            self.batch_size = 1 #data.shape[0]
            self.N = data.shape[0]
            assert (cs == relabel(cs)).all()
            self.hs = self.ms_gat(data, adj).view([self.batch_size, self.N, self.x_dim])
            self.Hs = torch.zeros([self.batch_size, 1, self.x_dim]).to(self.device)
            self.Hs[:, 0, :] = self.hs[:, 0, :]

            self.fs = self.f(data).view([self.batch_size, self.N, self.h_dim])
            self.G = self.fs[:, 2:, ].sum(dim=1)  # [batch_size,h_dim]
            self.previous_v_K = torch.zeros([K, self.batch_size, self.f_dim]).to(self.device)

        else:
            if K == self.previous_K:
                self.Hs[:, cs[n - 1], :] += self.hs[:, n - 1, :]
            else:
                self.Hs = torch.cat((self.Hs, self.hs[:, n - 1, :].unsqueeze(1)), dim=1)

            if n == self.N - 1:
                self.G = torch.zeros([self.batch_size, self.h_dim]).to(self.device)  # [batch_size,h_dim]
                self.previous_n = 0
            else:
                self.G -= self.fs[:, n, ]

        self.previous_K = K

        assert self.Hs.shape[1] == K

        logprobs = torch.zeros([self.batch_size, K + 1]).to(self.device)

        # loop over the K existing clusters for datapoint n to join
        for k in range(K):
            Hs2 = self.Hs.clone().view([self.batch_size, K, self.x_dim])
            Hs2[:, k, :] += self.hs[:, n, :]

            Hs2 = Hs2.view([self.batch_size * K, self.x_dim])

            hs = self.h(Hs2).view([self.batch_size, K, self.f_dim])
            Hk = hs.sum(dim=1)  # [batch_size,f_dim]

            cs = self.c(Hs2).view([self.batch_size, K, self.f_dim])
            Ck = cs.sum(dim=1)  # [batch_size,c_dim]

            R = torch.cat((Hk, Ck, self.G), dim=1)  # prepare argument for the call to u()
            logprobs[:, k] = torch.squeeze(self.u(R))

        # consider datapoint n creating a new cluster
        Hs2 = torch.cat((self.Hs, self.hs[:, n, :].unsqueeze(1)), dim=1)
        Hs2 = Hs2.view([self.batch_size * (K + 1), self.x_dim])

        hs = self.h(Hs2).view([self.batch_size, K + 1, self.f_dim])
        Hk = hs.sum(dim=1)

        cs = self.c(Hs2).view([self.batch_size, K + 1, self.f_dim])
        Ck = cs.sum(dim=1)

        encode_samples = cs.view([K + 1, self.batch_size, self.f_dim])

        R = torch.cat((Hk, Ck, self.G), dim=1)  # prepare argument for the call to u()
        logprobs[:, K] = torch.squeeze(self.u(R))
        nce = 0  # average over timestep and batch
        # normalize
        for i in range(K):

            total = torch.mm(encode_samples[i], torch.transpose(self.g(self.previous_v_K[i]), 0, 1))
            nce += torch.sum(torch.diag(self.lsoftmax(total)))

        self.previous_v_K = encode_samples.clone()

        nce /= -1. * self.batch_size * K
        m, _ = torch.max(logprobs, 1, keepdim=True)  # [batch_size,1]
        logprobs = logprobs - m - torch.log(torch.exp(logprobs - m).sum(dim=1, keepdim=True))

        return logprobs, nce



