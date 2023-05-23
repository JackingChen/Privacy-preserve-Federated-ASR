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
    parser.add_argument('--train_batch_size', type=int, default=6, help="")
    parser.add_argument('--eval_batch_size', type=int, default=8, help="")
    # model arguments
    parser.add_argument('--model', type=str, default='data2vec', help='model name')
    # other arguments
    parser.add_argument('--dataset', type=str, default='adress', help="name \
                        of dataset")
    parser.add_argument('--gpu', default=None, help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")
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
    parser.add_argument('--GPU_batchsize', type=str, default=None, help="如果cpu滿了就用GPU")
    # 2023/05/18
    parser.add_argument('-sl', '--supervised_level', type=float, default=1, help="0=fully unsupervised; 0.5=unsupervised+supervised; 1=fully supervised")
    parser.add_argument('--num_lms', type=int, default=5, help="")
    # 2023/5/200
    parser.add_argument('--eval_steps', type=int, default=1000, help="")
    args = parser.parse_args()
    return args
