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

def client_train(return_dict, args, train_dataset, logger, 
                 test_dataset, idx, epoch, global_weights=None):                    # train function for each client
    # BUILD MODEL for every process
    if args.model == 'data2vec':
        mask_time_prob = 0                                                          # change config to avoid training stopping
        config = Data2VecAudioConfig.from_pretrained(args.pretrain_name, mask_time_prob=mask_time_prob)
        model = Data2VecAudioForCTC.from_pretrained(args.model_in_path, config=config, args=args)
        print("model loaded")                                                       # load/initialize global model
        model.config.ctc_zero_infinity = True                                       # to avoid inf values

        global_model = copy.deepcopy(model.arbitrator)                              # only has global toggling network
        if global_weights != None:                                                  # if given global_weights
            global_model.load_state_dict(global_weights)                            # load it
        #else:
        #    # copy weights
        #    global_weights = copy.deepcopy(global_model.state_dict())                       # save global weight
        processor = Wav2Vec2Processor.from_pretrained(args.pretrain_name)
        data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    device = 'cuda' if args.gpu else 'cpu'
    global_model.to(device)
    global_model.train()
    #print(global_model)

    ####################
    # 'use client_id generate sub-dataset' to be done
    ####################
    #print("call ASRLocalUpdate")
    local_model = ASRLocalUpdate(args=args, dataset=train_dataset, logger=logger,
                        data_collator=data_collator, global_test_dataset=test_dataset, 
                        processor=processor, client_id=idx)
                                                                                    # initial dataset of current client
    ####################
    # 'use client_id load local model' to be done
    # 'save model in final round' to be done
    ####################
    #print("perform update_weight")
    w, loss = local_model.update_weights(
        global_arbitrator=copy.deepcopy(global_model), global_round=epoch)          # from global model to train
    
    #send_end.send([w, loss])                                                        # save model weights and average round loss
    return_dict[str(idx)] = [w, loss]
    #return w, loss

if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()                                                            # get configuration
    exp_details(args)                                                               # print out details based on configuration

    #device = 'cuda' if args.gpu else 'cpu'                                          # use gpu or cpu

    # load dataset and user groups
    ########################################
    # 'user_groups' need to be done
    ########################################
    train_dataset, test_dataset, user_groups = get_dataset(args)                    # get dataset
                                                                                    # user_groups: dict of client_id: sample_id
    # Training
    train_loss, test_wer = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0
    global_weights = None                                                           # initial global_weights
    for epoch in tqdm(range(args.epochs)):                                          # train for given global rounds
        #local_weights, local_losses = [], []                                        # weights and losses of training clients of this round
        print(f'\n | Global Training Round : {epoch+1} |\n')                        # print current round

        m = max(int(args.frac * args.num_users), 1)                                 # num of clients to train, min:1
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)      # select by client_id

        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        jobs = []
        for idx in idxs_users:                                                      # for each training client
            """ original code for training clients 1 by 1
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
                                      
            ####################
            # 'use client_id generate sub-dataset' to be done
            ####################
            local_model = ASRLocalUpdate(args=args, dataset=train_dataset, logger=logger,
                                data_collator=data_collator, global_test_dataset=test_dataset, 
                                processor=processor, client_id=idx)
                                                                        # initial dataset of current client
            ####################
            # 'use client_id load local model' to be done
            # 'save model in final round' to be done
            ####################
            w, loss = local_model.update_weights(
                global_arbitrator=copy.deepcopy(global_model), global_round=epoch)  # from global model to train
                                                                        # return model weights and average round loss
            """

            print("start of client #", idx)                                         # show current client id
            p = multiprocessing.Process(target=client_train, args=(return_dict,
                                        args, train_dataset, logger, test_dataset, idx, epoch, global_weights))
            jobs.append(p)
            p.start()
            
            #local_weights.append(copy.deepcopy(w))                      # save weight for this client
            #local_losses.append(copy.deepcopy(loss))                    # save loss for this client
        for proc in jobs:
            proc.join()                                                             # wait for process end
            #proc.close()

        local_weights = []
        local_losses = []
        for key in return_dict.keys():
            w, loss = return_dict[key]
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

        # Calculate avg training accuracy over all users at every epoch
        # 如果local model is saved, 不需要在train的時候eval吧?
        """
        list_wer = []
        global_model.eval()
        for c in range(args.num_users):                                 # for ALL users
            local_model = ASRLocalUpdate(args=args, dataset=train_dataset, logger=logger,
                                data_collator=data_collator, global_test_dataset=test_dataset, 
                                processor=processor)
                                                                        # initial dataset of current client
            wer = local_model.inference(global_arbitrator=global_model)       # get acc. & total loss on clients' test set
            list_wer.append(wer)                                        # save acc.
            #list_loss.append(loss)                                      # save loss
        test_wer.append(sum(list_wer)/len(list_wer))              # acc average over all clients

        # print global training loss after every 'i' rounds
        if (epoch+1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100*test_wer[-1])) # on testing set of clients though
        """
    # Test inference after completion of training
    #test_acc, test_loss = test_inference(args, global_model, test_dataset)  # test on actual testing set

    # show個簡單結果?
    #print(f' \n Results after {args.epochs} global rounds of training:')
    #print("|---- Avg Test WER: {:.2f}%".format(100*test_wer[-1]))
    #print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

    # Saving the objects train_loss and train_accuracy:
    #file_name = './save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
    #    format(args.dataset, args.model, args.epochs, args.frac, args.iid,
    #           args.local_ep, args.local_bs) # use configuration to generate file name

    #with open(file_name, 'wb') as f:
    #    pickle.dump([train_loss, train_accuracy], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

    # PLOTTING (optional)
    # import matplotlib
    # import matplotlib.pyplot as plt
    # matplotlib.use('Agg')

    # Plot Loss curve
    # plt.figure()
    # plt.title('Training Loss vs Communication rounds')
    # plt.plot(range(len(train_loss)), train_loss, color='r')
    # plt.ylabel('Training loss')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))
    #
    # # Plot Average Accuracy vs Communication rounds
    # plt.figure()
    # plt.title('Average Accuracy vs Communication rounds')
    # plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    # plt.ylabel('Average Accuracy')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))
