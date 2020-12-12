#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import math


def get_parameters(args):

    if args.model == 'Event':
        params = {}
        params['model'] = 'Event'
        params['event_num'] = args.event_num
        params['x_dim'] = 600
        params['h1_dim'] = 200
        params['h_dim'] = 200
        params['f_dim'] = 200
        params['c_dim'] = 200
        params['H_dim'] = 400
        params['nheads'] = 4
        params['scale'] = 5

    else:
        raise NameError('Unknown model ' + model)

    return params


def relabel(cs):
    cs = cs.copy()
    d = {}
    k = 0

    for i in range(len(cs)):
        j = cs[i]
        if j not in d:
            d[j] = k
            k += 1
        cs[i] = d[j]

    return cs


def get_adj(data, selected_firm, selected_event, event):
    event_to_idx = {e: i for i, e in enumerate(event)}
    W = np.zeros((len(event), len(event)))

    for f in selected_firm:
        events = data[data['STOCK_ID'] == f]['EVENT'].tolist()
        events = [e for e in events if e in selected_event]

        for i, e1 in enumerate(events):
            for j, e2 in enumerate(events):
                if abs(i - j) <= 23:
                    W[event_to_idx[e1], event_to_idx[e2]] += math.exp(-abs(i - j))

    embeddings = []

    for i, e in enumerate(event):
        embeddings.append(W[event_to_idx[e], :])

    return np.array(W), np.array(embeddings)










