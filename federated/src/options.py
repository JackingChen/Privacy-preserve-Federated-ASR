#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--epochs', type=int, default=10,
                        help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=100,
                        help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.1,
                        help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=10,
                        help="the number of local epochs: E")
    #parser.add_argument('--local_bs', type=int, default=10,
    #                    help="local batch size: B")
    #parser.add_argument('--lr', type=float, default=0.01,
    #                    help='learning rate')
    #parser.add_argument('--momentum', type=float, default=0.5,
    #                    help='SGD momentum (default: 0.5)')

    # model arguments
    parser.add_argument('--model', type=str, default='data2vec', help='model name')
    #parser.add_argument('--kernel_num', type=int, default=9,
    #                    help='number of each kind of kernel')
    #parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
    #                    help='comma-separated kernel size to \
    #                    use for convolution')
    #parser.add_argument('--num_channels', type=int, default=1, help="number \
    #                    of channels of imgs")
    #parser.add_argument('--norm', type=str, default='batch_norm',
    #                    help="batch_norm, layer_norm, or None")
    #parser.add_argument('--num_filters', type=int, default=32,
    #                    help="number of filters for conv nets -- 32 for \
    #                    mini-imagenet, 64 for omiglot.")
    #parser.add_argument('--max_pool', type=str, default='True',
    #                    help="Whether use max pooling rather than \
    #                    strided convolutions")

    # other arguments
    parser.add_argument('--dataset', type=str, default='adress', help="name \
                        of dataset")
    #parser.add_argument('--num_classes', type=int, default=10, help="number \
    #                    of classes")
    parser.add_argument('--gpu', default=None, help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")
    #parser.add_argument('--optimizer', type=str, default='sgd', help="type \
    #                    of optimizer")
    #parser.add_argument('--iid', type=int, default=1,
    #                    help='Default set to IID. Set to 0 for non-IID.')
    #parser.add_argument('--unequal', type=int, default=0,
    #                    help='whether to use unequal data splits for  \
    #                    non-i.i.d setting (use 0 for equal splits)')
    #parser.add_argument('--stopping_rounds', type=int, default=10,
    #                    help='rounds of early stopping')
    #parser.add_argument('--verbose', type=int, default=1, help='verbose')
    #parser.add_argument('--seed', type=int, default=1, help='random seed')

    # additional arguments
    parser.add_argument('--pretrain_name', type=str, default='facebook/data2vec-audio-large-960h', help="str used to load pretrain model")

    parser.add_argument('-lam', '--LAMBDA', type=float, default=0.5, help="Lambda for GRL")
    parser.add_argument('-st', '--STAGE', type=int, default=1, help="Current training stage")
    parser.add_argument('-fl_st', '--FL_STAGE', type=int, default=1, help="Current FL training stage")
    parser.add_argument('-GRL', '--GRL', action='store_true', default=False, help="True: GRL")
    parser.add_argument('-model_in', '--model_in_path', type=str, default="./saves/wav2vec2-base-960h_GRL_0.5/checkpoint-14010/", help="Where the global model is saved")
    parser.add_argument('-model_out', '--model_out_path', type=str, default="./saves/wav2vec2-base-960h_linear_GRL", help="Where to save the model")
    parser.add_argument('-log', '--log_path', type=str, default="wav2vec2-base-960h_linear_GRL.txt", help="name for the txt file")
    parser.add_argument('-csv', '--csv_path', type=str, default="wav2vec2-base-960h_GRL_0.5", help="name for the csv file")
    # 2023/01/08: loss type
    parser.add_argument('-ad_loss', '--AD_loss', type=str, default="cel", help="loss to use for AD classifier")
    # 2023/01/18: ckpt
    parser.add_argument('-ckpt', '--checkpoint', type=str, default=None, help="path to checkpoint")
    # 2023/02/13: TOGGLE_RATIO
    parser.add_argument('-toggle_rt', '--TOGGLE_RATIO', type=float, default=0, help="To toggle more or less")
    # 2023/02/15: GS_TAU, loss weight
    parser.add_argument('-gs_tau', '--GS_TAU', type=float, default=1, help="Tau for gumbel_softmax")
    parser.add_argument('-w_loss', '--W_LOSS', type=float, default=None, nargs='+', help="weight for HC and AD")
    # 2023/04/20
    parser.add_argument('-EXTRACT', '--EXTRACT', action='store_true', default=False, help="True: extract embs")
    parser.add_argument('-client_id', '--client_id', type=str, default="public", help="client_id: public, 0, or 1")
    # 2023/04/24
    parser.add_argument('--global_ep', type=int, default=30, help="number for global model")
    
    args = parser.parse_args()
    return args
