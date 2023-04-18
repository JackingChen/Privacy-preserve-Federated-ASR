#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


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
import multiprocessing
from update import update_network, update_network_weight, get_model_weight

from training import client_train, centralized_training

def FL_training_rounds(args, model_in_path_root, model_out_path, train_dataset, test_dataset):
    train_loss = []                                                                 # list for training loss
    global_weights = None                                                           # initial global_weights

    multiprocessing.set_start_method('spawn', force=True)
    for epoch in tqdm(range(args.epochs)):                                          # train for given global rounds
        print(f'\n | Global Training Round : {epoch+1} |\n')                        # print current round

        m = max(int(args.frac * args.num_users), 1)                                 # num of clients to train, min:1
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)      # select by client_id
        pool = multiprocessing.Pool(processes=m)

        if args.STAGE == 0:                                                         # train ASR
            local_weights_en = []                                                   # weight list for ASR encoder
            local_weights_de = []                                                   # weight list for ASR decoder
        else:                                                                       # train AD classifier or toggling network
            local_weights = []                                                      # only 1 weight list needed
        local_losses = []                                                           # losses of training clients of this round

        try:
            if (epoch == 0) and (args.STAGE == 2):                                  # start from global model to train toggling network
                global_weights = get_model_weight(args=args, source_path=model_out_path + "_global/final/", network="toggling_network")
                                                                                    # local ASR and AD with global toggling network
                                                                                    # get toggling_network weights from model in model_out_path + "_global/final/"
            final_result = pool.starmap_async(client_train, [(args, model_in_path_root, model_out_path, train_dataset, test_dataset, idx,
                                                                  epoch, global_weights) for idx in idxs_users])
                                                                                    # train from model in model_in_path 
                                                                                    #                                 + "_global/final/", when stage=0
                                                                                    #                                 + "_client" + str(idx) + "_round" + str(args.epochs-1) + "/final/", o.w.
                                                                                    # or model in last round
                                                                                    # final result in model_out_path + "_client" + str(client_id) + "_round" + str(global_round)
        except Exception as e:
            print(f"An error occurred while running local_model.update_weights(): {str(e)}")
        
        finally:
            final_result.wait()                                                     # wait for all clients end
            results = final_result.get()                                            # get results
        
        for idx in range(len(results)):                                             # for each participated clients
            w, loss = results[idx]                                                  # function client_train returns w & loss
            if args.STAGE == 0:                                                     # train ASR
                local_weights_en.append(copy.deepcopy(w[0]))                        # save encoder weight for this client
                local_weights_de.append(copy.deepcopy(w[1]))                        # save decoder weight for this client
            else:                                                                   # train AD classifier or toggling network
                local_weights.append(copy.deepcopy(w))
            local_losses.append(loss)

        # aggregate weights
        if args.STAGE == 0:                                                         # train ASR
            global_weights = [average_weights(local_weights_en), average_weights(local_weights_de)]
        else:                                                                       # train AD classifier or toggling network
            global_weights = average_weights(local_weights)

        loss_avg = sum(local_losses) / len(local_losses)                            # average losses from participated client
        train_loss.append(loss_avg)                                                 # save loss for this round
    return global_weights

# FL stage 1: ASR & AD Classifier
def stage1_training(args, train_dataset, test_dataset):
    ##########################################################
    # Centralized Training: train global ASR & AD Classifier #
    ##########################################################
    args.STAGE = 0                                                                  # train ASR first
    centralized_training(args=args, model_in_path=args.pretrain_name, model_out_path=args.model_out_path+"_finetune", 
                         train_dataset=train_dataset, test_dataset=test_dataset, epoch=0)
                                                                                    # train from pretrain, final result in args.model_out_path + "_finetune" + "_global/final"
    args.STAGE = 1                                                                  # then train AD classifier
    centralized_training(args=args, model_in_path=args.model_out_path+"_finetune_global/final/", 
                         model_out_path=args.model_out_path, train_dataset=train_dataset, test_dataset=test_dataset, epoch=0)
                                                                                    # train from final result from last line, final result in args.model_out_path + "_global/final"
    ##########################################################
    # FL: train local ASR & AD Classifier federally          #
    ##########################################################
    args.STAGE = 0                                                                  # train ASR first
    global_weights = FL_training_rounds(args=args, model_in_path_root=args.model_out_path, model_out_path=args.model_out_path+"_finetune",
                                        train_dataset=train_dataset, test_dataset=test_dataset)

    # update global model
    model = update_network_weight(args=args, source_path=args.model_out_path+"_global/final/", target_weight=global_weights, network="ASR") 
                                                                                    # update ASR in source_path with given weights
    model.save_pretrained(args.model_out_path+"_FLASR_global/final")
    
    args.STAGE = 1                                                                  # then train AD classifier
    global_weights = FL_training_rounds(args=args, model_in_path_root=args.model_out_path+"_finetune", model_out_path=args.model_out_path,
                                        train_dataset=train_dataset, test_dataset=test_dataset)

    # update global model
    model = update_network_weight(args=args, source_path=args.model_out_path+"_FLASR_global/final", target_weight=global_weights, network="AD")
                                                                                    # update AD classifier in source_path with given weights
    model.save_pretrained(args.model_out_path+"_FLAD_global/final")

    
# FL stage 2: Toggling Network
def stage2_training(args, train_dataset, test_dataset):
    ##########################################################
    # Centralized Training: train global Toggling Network    #
    ##########################################################
    centralized_training(args=args, model_in_path=args.model_in_path + "_FLAD_global/final/", model_out_path=args.model_out_path, 
                         train_dataset=train_dataset, test_dataset=test_dataset, epoch=0)
                                                                                    # train from model_in_path + "_FLAD_global/final/" (aggregated ASR & AD)
                                                                                    # final result in args.model_out_path + "_global/final"
    ##########################################################
    # FL: train local Toggling Network federally             #
    ##########################################################
    global_weights = FL_training_rounds(args=args, model_in_path_root=args.model_in_path, model_out_path=args.model_out_path,
                                        train_dataset=train_dataset, test_dataset=test_dataset)
    # update global model
    model = update_network_weight(args=args, source_path=args.model_out_path+"_global/final", target_weight=global_weights, network="toggling_network")
                                                                                    # update toggling_network in source_path with given weights
    model.save_pretrained(args.model_out_path+"_final_global/final")


if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    
    args = args_parser()                                                            # get configuration
    exp_details(args)                                                               # print out details based on configuration

    train_dataset, test_dataset = get_dataset(args)                                 # get dataset

    # Training
    if args.FL_STAGE == 1:
        print("| Start FL Training Stage 1|")
        stage1_training(args, train_dataset, test_dataset)                          # Train ASR & AD Classifier
        print("| FL Training Stage 1 Done|")

    elif args.FL_STAGE == 2:
        print("| Start FL Training Stage 2|")
        args.STAGE = 2
        stage2_training(args, train_dataset, test_dataset)                          # Train Toggling Network
        print("| FL Training Stage 2 Done|")
    
    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))