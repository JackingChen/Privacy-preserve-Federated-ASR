#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse
import multiprocessing as mp
import os

import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference, ASRLocalUpdate
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, Data2VecAudioForCTC, DataCollatorCTCWithPadding
from utils import get_dataset, average_weights, exp_details

from transformers import Data2VecAudioConfig, Wav2Vec2Processor
from multiprocessing import Pool
from collections import OrderedDict

from client_train import client_train
import multiprocessing
parser = argparse.ArgumentParser()
# federated arguments (Notation for the arguments followed from paper)
parser.add_argument('--epochs', type=int, default=2,
                    help="number of rounds of training")
parser.add_argument('--num_users', type=int, default=2,
                    help="number of users: K")
parser.add_argument('--frac', type=float, default=1.0,
                    help='the fraction of clients: C')
parser.add_argument('--local_ep', type=int, default=1,
                    help="the number of local epochs: E")

parser.add_argument('--model', type=str, default='data2vec', help='model name')


# other arguments
parser.add_argument('--dataset', type=str, default='adress', help="name \
                    of dataset") #cifar
#parser.add_argument('--num_classes', type=int, default=10, help="number \
#                    of classes")
parser.add_argument('--gpu', default=1, help="To use cuda, set \
                    to a specific GPU ID. Default set to use CPU.")

# additional arguments
parser.add_argument('--pretrain_name', type=str, default='facebook/data2vec-audio-large-960h', help="str used to load pretrain model")
parser.add_argument('-lam', '--LAMBDA', type=float, default=0.5, help="Lambda for GRL")
parser.add_argument('-st', '--STAGE', type=int, default=2, help="Current training stage")
parser.add_argument('-GRL', '--GRL', action='store_true', default=False, help="True: GRL")
parser.add_argument('-model_in', '--model_in_path', type=str, default="/mnt/Internal/FedASR/weitung/HuggingFace/Pretrain/saves/data2vec-audio-large-960h_new1_recall/final/", help="Where the model is saved")
parser.add_argument('-model_out', '--model_out_path', type=str, default="./save/data2vec-audio-large-960h_new2_recall_FL", help="Where to save the model")
parser.add_argument('-log', '--log_path', type=str, default="data2vec-audio-large-960h_new2_recall_FL.txt", help="name for the txt file")
# 2023/01/08: loss type
parser.add_argument('-ad_loss', '--AD_loss', type=str, default="recall", help="loss to use for AD classifier")
# 2023/01/18: ckpt
parser.add_argument('-ckpt', '--checkpoint', type=str, default=None, help="path to checkpoint")
# 2023/02/13: TOGGLE_RATIO
parser.add_argument('-toggle_rt', '--TOGGLE_RATIO', type=float, default=0, help="To toggle more or less")
# 2023/02/15: GS_TAU, loss weight
parser.add_argument('-gs_tau', '--GS_TAU', type=float, default=1, help="Tau for gumbel_softmax")
parser.add_argument('-w_loss', '--W_LOSS', type=float, default=None, nargs='+', help="weight for HC and AD")

args = parser.parse_args(args=[]) # for jupyter notebook


# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3" # 或者其他你想要使用的 GPU 編號
lock = mp.Lock()
logger = SummaryWriter('../logs')

train_loss, test_wer = [], []
val_acc_list, net_list = [], []
cv_loss, cv_acc = [], []
print_every = 2
val_loss_pre, counter = 0, 0
global_weights = None                                                           # initial global_weights
train_dataset, test_dataset, user_groups = get_dataset(args)
if __name__ == "__main__":
    # multiprocessing.set_start_method('spawn')
    multiprocessing.set_start_method('spawn', force=True) #!!! [NOTE] 2.需要把multiprocessing的method從fork改成spawn，並且client train需要獨立到別的模塊然後用import的方式叫進來
    for epoch in range(2):
        m = max(int(args.frac * args.num_users), 1)                                 # num of clients to train, min:1
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)      # select by client_id
        pool = multiprocessing.Pool(processes=m)
        try:
            # final_result = pool.starmap_async(
            #     client_train, [(args, train_dataset, logger,
            #                     test_dataset, idx, epoch, global_weights)
            #                 for idx in idxs_users])
            #!!! [NOTE] 1.傳logger會出現RuntimeError: Queue objects should only be shared between processes through inheritance
            final_result = pool.starmap_async(
                client_train, [(args, train_dataset, None,
                                test_dataset, idx, epoch, global_weights)
                            for idx in idxs_users])
        except Exception as e:
            print(f"An error occurred while running local_model.update_weights(): {str(e)}")
        finally:
            final_result.wait()
            results = final_result.get()
        
        local_weights = []
        local_losses = []
        for idx in range(len(results)):
            w, loss = results[idx]
            local_weights.append(w)
            local_losses.append(loss)
        print("local weights: ", local_weights)
        # get global weights by averaging local weights
        global_weights = average_weights(local_weights)
        print("global wegiths: ", global_weights)
        # update global weights
        #global_model.load_state_dict(global_weights)
        loss_avg = sum(local_losses) / len(local_losses)                # average losses from participated client
        train_loss.append(loss_avg)                                     # save loss for this round
        print("All results done")
