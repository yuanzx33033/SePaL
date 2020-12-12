#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import torch
import pandas as pd
from utils import relabel, get_adj
from sklearn.cluster import MeanShift, estimate_bandwidth


def get_generator(params):
    if params['model'] == 'Event':
        return Event_generator(params)
    else:
        raise NameError('Unknown model ' + params['model'])


class Event_generator():

    def __init__(self, params):
        self.params = params
        self.dataset = pd.read_pickle('./sample_data/event_sample')
        self.firm = self.dataset['STOCK_ID'].unique()
        self.event = self.dataset['EVENT'].unique()
        print('Num of considered firms: ', len(self.firm))
        print('Total num of assumed event types : ', len(self.event))
        print('Note that the uploaded event data is only for demo purpose')

        self.adj, self.embeddings = get_adj(self.dataset, self.firm, self.event, self.event)

    def generate(self, model, N=None, batch_size=1):

        K = len(self.event) + 1

        data = torch.LongTensor(range(len(self.event))).unsqueeze(0).to('cuda')
        data = model.token_embeddings(data).squeeze(0)
        a = torch.from_numpy(self.adj).float().to('cuda')
        graph_emb = model.ms_gat(data, a).squeeze(0).detach().cpu().numpy()

        while K > len(self.event):
            clusters, N, K = generate_graph_emb(graph_emb, len(self.event))

        cumsum = np.cumsum(clusters)
        data = np.empty([batch_size, N])
        Adj = np.empty([batch_size, len(self.event), len(self.event)])
        cs = np.empty(N, dtype=np.int32)

        counter = 0
        for i in range(batch_size):
            for k in range(K):
                nk = int(clusters[k+1])
                event_ind = range(counter, counter + nk)
                data[i, cumsum[k]:cumsum[k + 1]] = event_ind
                cs[cumsum[k]:cumsum[k + 1]] = k + 1
                Adj[i, :, :] = self.adj
                counter += nk

        cs = relabel(cs)

        return data, cs, clusters, K, self.event, Adj, graph_emb


def generate_graph_emb(data, N, no_ones=False):

    bw = estimate_bandwidth(data, quantile=0.2, n_samples=10)
    Labels = MeanShift(bandwidth=bw).fit_predict(data)
    K = len(set(Labels))
    clusters = np.zeros(K+1, dtype=int)
    for i in range(N):
        clusters[Labels[i]+1] += 1
    return clusters, N, K
