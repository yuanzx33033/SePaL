#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# color = sns.color_palette('bright', 10) + sns.color_palette("cubehelix", 1) + sns.xkcd_palette(['windows blue'])

np.set_printoptions(threshold=np.inf)

def plot_samples_Event(sepal, data_generator, N=40, seed=None):
    if seed:
        np.random.seed(seed=seed)

    data, cs, clusters, num_clusters, events, adj, graph_emb = data_generator.generate(sepal, N, batch_size=1)

    from sklearn.manifold import TSNE
    from sklearn.cluster import MeanShift, estimate_bandwidth
    idx_to_event = {i: e for i, e in enumerate(events)}

    pca = TSNE(n_components=2, init='pca')

    new_x = pca.fit_transform(graph_emb)
    x_min, x_max = np.max(new_x), np.min(new_x)
    new_x = (new_x - x_min) / (x_max - x_min)

    bw = estimate_bandwidth(new_x, quantile=0.2, n_samples=10)
    # print(bw)
    Label = MeanShift(bandwidth=bw).fit_predict(new_x)
    plt.figure(1, figsize=(8, 8))
    plt.clf()
    fig, ax = plt.subplots(ncols=1, nrows=1, num=1)

    K = len(set(Label))
    color = sns.color_palette('bright', K)
    Labels = {}
    for j in range(N):
        if Label[j] in Labels:
            Labels[Label[j]][0].append(new_x[j][0])
            Labels[Label[j]][1].append(new_x[j][1])
            Labels[Label[j]][2].append(color[Label[j]])
            Labels[Label[j]][3].append(Label[j])
        else:
            Labels[Label[j]] = [[], [], [], []]
    for i in range(K):
        plt.scatter(Labels[i][0], Labels[i][1], color=Labels[i][2],
                    label='$es_{%s}$'% (i+1), s=35)

    fontsize = 21
    ax.set_title(str(K) + ' Clusters', fontsize=fontsize)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(0.8)
    plt.show()

    # print('Clustering Methods: Mean Shift:')
    for k in set(Label):
        e_list = []
        for item in np.nonzero(Label == k)[0]:
            e_list.append(idx_to_event[item])
        # print(k, len(e_list), e_list)

    return


