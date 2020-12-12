#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import argparse
import time
import os
import torch
from model_SePaL import SePaL
from data_generators import get_generator
from plot_functions import plot_samples_Event
from utils import relabel, get_parameters


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)


    params = get_parameters(args)
    params['device'] = torch.device("cuda:0" if args.cuda else "cpu")
    params['path'] = args.path

    print(params['device'], params['model'])

    sepal = SePaL(params).to(params['device'])
    data_generator = get_generator(params)

    # define containers to collect statistics
    losses = []  # NLLs
    accs = []  # Accuracy of the classification prediction
    perm_vars = []  # permutation variance

    it = 0  # iteration counter
    learning_rate = 1e-4
    weight_decay = 0.0
    optimizer = torch.optim.Adam(sepal.parameters(), lr=learning_rate, weight_decay=weight_decay)

    perms = 2  # Number of permutations for each mini-batch.
    # In each permutation, the order of the event sequences is shuffled.

    batch_size = args.batch_size
    max_it = args.iterations


    if params['model'] == 'Event':
        if not os.path.isdir('saved_models/Event'):
            os.makedirs('saved_models/Event')
        if not os.path.isdir('figures/Event'):
            os.makedirs('figures/Event')

    end_name = params['model']
    learning_rates = {1200: 5e-5, 2200: 1e-5}
    criterion = torch.nn.MSELoss()

    t_start = time.time()
    itt = it
    while True:
        it += 1

        if it == max_it:
            break

        sepal.train()

        if it % args.plot_interval == 0:

            torch.cuda.empty_cache()

            if params['model'] == 'Event':
                plot_samples_Event(sepal, data_generator, N=params['event_num'], seed=it)

        if it % 100 == 0:
            if 'fname' in vars():
                os.remove(fname)
            sepal.params['it'] = it
            fname = 'saved_models/' + end_name + '/' + end_name + '_' + str(it) + '.pt'
            torch.save(sepal, fname)

        if it in learning_rates:
            optimizer = torch.optim.Adam(sepal.parameters(), lr=learning_rates[it], weight_decay=weight_decay)

        data, cs, clusters, K, _, Adj, graph_emb = data_generator.generate(sepal, batch_size=batch_size)
        graph_emb = torch.tensor(graph_emb).to(params['device'])
        # print('Num of clusters: ', K)
        N = data.shape[1]
        loss_values = np.zeros(perms)
        accuracies = np.zeros([N - 1, perms])

        # The memory requirements change in each iteration according to the random values of N and K.
        # If both N and K are big, an out of memory RuntimeError exception might be raised.
        # When this happens, we capture the exception, reduce the batch_size to 3/4 of its value, and try again.

        while True:

            loss = 0
            reconst_adj = torch.matmul(graph_emb, graph_emb.t())
            org_adj = torch.from_numpy(Adj[0]).float().to(params['device'])

            reconst_loss = criterion(reconst_adj, org_adj)

            for perm in range(perms):
                # print(perm, perms)
                arr = np.arange(N)
                np.random.shuffle(arr)  # permute the order in which the events are queried
                cs = cs[arr]
                data = graph_emb[arr, :]

                cs = relabel(cs)  # this makes ep labels appear in cs[] in increasing order

                this_loss = 0
                sepal.previous_n = 0


                for n in range(1, N):
                    # points up to (n-1) are already assigned, the point n is to be assigned
                    logprobs, nce = sepal(data, cs, n, org_adj)
                    c = cs[n]
                    accuracies[n - 1, perm] = np.sum(np.argmax(logprobs.detach().to('cpu').numpy(), axis=1) == c) / \
                                              logprobs.shape[0]

                    this_loss = this_loss + nce - 0.5 * logprobs[:, c].mean()

                this_loss += 0.3 * reconst_loss/ (N*N)

                this_loss.backward()  # this accumulates the gradients for each permutation
                loss_values[perm] = this_loss.item()
                loss += this_loss

            perm_vars.append(loss_values.var())
            losses.append(loss.item() / N)
            accs.append(accuracies.mean())

            optimizer.step()  # the gradients used in this step are the sum of the gradients for each permutation
            optimizer.zero_grad()

            print(
                '{0:4d}  N:{1:2d}  K:{2}  Mean NLL:{3:.3f}   Mean Acc:{4:.3f}   Mean Permutation Variance: {5:.7f}  Mean Time/Iteration: {6:.1f}' \
                .format(it, N, K, np.mean(losses[-50:]), np.mean(accs[-50:]), np.mean(perm_vars[-50:]),
                        (time.time() - t_start) / (it - itt)))
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SePaL')

    parser.add_argument('--model', type=str, default='Event', metavar='S',
                        choices=['Event'], help='Event (default: Event)')
    parser.add_argument('--batch-size', type=int, default=32, metavar='BS',
                        help='batch size for training (default: 32)')
    parser.add_argument('--iterations', type=int, default=400, metavar='IT',
                        help='number of iterations to train (default: 3500)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=2, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--path', type=str, default='./data', metavar='P',
                        help='project directory')
    parser.add_argument('--plot_interval', type=int, default=50, metavar='PI',
                        help='how many iterations between training plots')
    parser.add_argument('--event_num', type=int, default=176, metavar='EN',
                        help='the number of different event types')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    main(args)

