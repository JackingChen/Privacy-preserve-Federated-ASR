#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import os
import copy
import time
import numpy as np
from tqdm import tqdm

from options import args_parser
from utils import get_dataset, average_weights, exp_details

import multiprocessing
from update import update_network_weight, get_model_weight

from training import client_train, centralized_training, unsupervised_client_train
from update import ASRLocalUpdate
from datasets import load_dataset, concatenate_datasets
from utils import prepare_dataset
from transformers import Wav2Vec2Processor
def FL_training_rounds(args, model_in_path_root, model_out_path, train_dataset_supervised, train_dataset_unsupervised, test_dataset, supervised_level):
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
            if supervised_level == 1:                                               # fully supervised
                final_result = pool.starmap_async(client_train, [(args, model_in_path_root, model_out_path, train_dataset_supervised, train_dataset_unsupervised, test_dataset, idx,
                                                                  epoch, global_weights) for idx in idxs_users])

                                                                                    # train from model in model_in_path 
                                                                                    #                                 + "_global/final/", when stage=0
                                                                                    #                                 + "_client" + str(idx) + "_round" + str(args.epochs-1) + "/final/", o.w.
                                                                                    # or model in last round
                                                                                    # final result in model_out_path + "_client" + str(client_id) + "_round" + str(global_round)
            """
            elif supervised_level == 0.5:                                           # unsupervised, then supervised training
                final_result = pool.starmap_async(unsupervised_client_train, [(args, model_in_path_root, model_out_path, train_dataset_supervised, train_dataset_unsupervised,
                                                test_dataset, idx, epoch, False, True, global_weights) for idx in idxs_users])
                                                                                    # train from model in model_in_path 
                                                                                    #                                 + "_global/final/", when stage=0
                                                                                    #                                 + "_client" + str(idx) + "_round" + str(args.epochs-1) + "/final/", o.w.
                                                                                    # or model in last round
                                                                                    # final result in model_out_path + "_client" + str(client_id) + "_round" + str(global_round)
            elif supervised_level == 0:                                             # fully unsupervised
                final_result = pool.starmap_async(unsupervised_client_train, [(args, model_in_path_root, model_out_path, train_dataset_supervised, train_dataset_unsupervised,
                                                test_dataset, idx, epoch, True, False, global_weights) for idx in idxs_users])
                                                                                    # train from model in model_in_path 
                                                                                    #                                 + "_global/final/", when stage=0
                                                                                    #                                 + "_client" + str(idx) + "_round" + str(args.epochs-1) + "_unsuper/final/", o.w.
                                                                                    # or model in last round
                                                                                    # final result in model_out_path + "_client" + str(client_id) + "_round" + str(global_round) + "_unsuper"     
        """                                                                                                                                            
        except Exception as e:
            print(f"An error occurred while running local_model.update_weights(): {str(e)}")
        
        finally:
            final_result.wait()                                                     # wait for all clients end
            results = final_result.get()                                            # get results
        
        for idx in range(len(results)):                                             # for each participated clients
            # w, loss = results[idx]                                                  # function client_train returns w & loss
            w = results[idx]
            if args.STAGE == 0:                                                     # train ASR
                local_weights_en.append(copy.deepcopy(w[0]))                        # save encoder weight for this client
                local_weights_de.append(copy.deepcopy(w[1]))                        # save decoder weight for this client
            else:                                                                   # train AD classifier or toggling network
                local_weights.append(copy.deepcopy(w))
            # local_losses.append(loss)

        # aggregate weights
        if args.STAGE == 0:                                                         # train ASR
            global_weights = [average_weights(local_weights_en), average_weights(local_weights_de)]
        else:                                                                       # train AD classifier or toggling network
            global_weights = average_weights(local_weights)

        # loss_avg = sum(local_losses) / len(local_losses)                            # average losses from participated client
        # train_loss.append(loss_avg)                                                 # save loss for this round
    return global_weights

# FL stage 1: ASR & AD Classifier
def stage1_training(args, train_dataset_supervised, train_dataset_unsupervised, test_dataset, supervised_level):
    local_epoch = args.local_ep                                                     # save given number of local epoch
    ##########################################################
    # Centralized Training: train global ASR & AD Classifier #
    ##########################################################
    
    args.local_ep = args.global_ep                                                  # use number of global epoch for global model
    skip_centralizeTraining=True
    if not skip_centralizeTraining:
        args.STAGE = 0                                                                  # train ASR first
        centralized_training(args=args, model_in_path=args.pretrain_name, model_out_path=args.model_out_path+"_finetune", 
                            train_dataset=train_dataset_supervised, test_dataset=test_dataset, epoch=0)
                                                                                    # train from pretrain, final result in args.model_out_path + "_finetune" + "_global/final"
    # args.STAGE = 1                                                                  # then train AD classifier
    # centralized_training(args=args, model_in_path=args.model_out_path+"_finetune_global/final/", 
    #                      model_out_path=args.model_out_path, train_dataset=train_dataset_supervised, test_dataset=test_dataset, epoch=0)
                                                                                    # train from final result from last line, final result in args.model_out_path + "_global/final"
    
    ##########################################################
    # FL: train local ASR & AD Classifier federally          #
    ##########################################################
    args.local_ep = local_epoch                                                     # use the given number of local epoch
    args.STAGE = 0                                                                  # train ASR first

    global_weights = FL_training_rounds(args=args, model_in_path_root=args.model_out_path, model_out_path=args.model_out_path+"_finetune",
                                        train_dataset_supervised=train_dataset_supervised, train_dataset_unsupervised=train_dataset_unsupervised,
                                        test_dataset=test_dataset, supervised_level=supervised_level)

    # update global model
    model = update_network_weight(args=args, source_path=args.model_out_path+"_global/final/", target_weight=global_weights, network="ASR") 
    model.save_pretrained(args.model_out_path+"_FLASR_global/final")
    
    # args.STAGE = 1                                                                  # then train AD classifier
    # supervised_level = 1                                                            # train supervised
    # global_weights = FL_training_rounds(args=args, model_in_path_root=args.model_out_path+"_finetune", model_out_path=args.model_out_path,
    #                                     train_dataset_supervised=train_dataset_supervised, train_dataset_unsupervised=train_dataset_unsupervised,
    #                                     test_dataset=test_dataset, supervised_level=supervised_level)

    # # update global model
    # model = update_network_weight(args=args, source_path=args.model_out_path+"_FLASR_global/final", target_weight=global_weights, network="AD")
    #                                                                                 # update AD classifier in source_path with given weights
    # model.save_pretrained(args.model_out_path+"_FLAD_global/final")
    
    
# FL stage 2: Toggling Network
def stage2_training(args, train_dataset_supervised, train_dataset_unsupervised, test_dataset, supervised_level):
    local_epoch = args.local_ep                                                     # save given number of local epoch
    ##########################################################
    # Centralized Training: train global Toggling Network    #
    ##########################################################
    
    args.local_ep = args.global_ep                                                  # use number of global epoch for global model
    centralized_training(args=args, model_in_path=args.model_in_path + "_FLAD_global/final/", model_out_path=args.model_out_path, 
                         train_dataset=train_dataset_supervised, test_dataset=test_dataset, epoch=0)
                                                                                    # train from model_in_path + "_FLAD_global/final/" (aggregated ASR & AD)
                                                                                    # final result in args.model_out_path + "_global/final"
    
    ##########################################################
    # FL: train local Toggling Network federally             #
    ##########################################################
    args.local_ep = local_epoch                                                     # use the given number of local epoch
    supervised_level = 1                                                            # train supervised
    global_weights = FL_training_rounds(args=args, model_in_path_root=args.model_in_path, model_out_path=args.model_out_path,
                                        train_dataset_supervised=train_dataset_supervised, train_dataset_unsupervised=train_dataset_unsupervised,
                                        test_dataset=test_dataset, supervised_level=supervised_level)
    # update global model
    model = update_network_weight(args=args, source_path=args.model_out_path+"_global/final", target_weight=global_weights, network="toggling_network")
                                                                                    # update toggling_network in source_path with given weights
    model.save_pretrained(args.model_out_path+"_final_global/final")

def extract_emb(args, train_dataset, test_dataset):
    if args.client_id == "public":
        idx = "public"
    else:
        idx = int(args.client_id)
    local_model = ASRLocalUpdate(args=args, dataset=train_dataset, global_test_dataset=test_dataset, 
                                 client_id=idx, model_in_path=args.model_in_path, model_out_path=None)
                                                                                      # initial dataset of current client
    local_model.extract_embs()
                                                                                      # from model_in_path model, update certain part using given weight

##########################################################
# Global Train w/ 50% of Training Set then the other 50% #
##########################################################
# FL stage 1: ASR & AD Classifier
def stage1_training_5050(args, train_dataset, test_dataset):
    #local_epoch = args.local_ep                                                     # save given number of local epoch
    ##########################################################
    # Centralized Training: train global ASR & AD Classifier #
    ##########################################################
    args.local_ep = args.global_ep                                                  # use number of global epoch for global model
    #args.STAGE = 0                                                                  # train ASR first
    #centralized_training(args=args, model_in_path=args.pretrain_name, model_out_path=args.model_out_path+"_finetune", 
    #                     train_dataset=train_dataset, test_dataset=test_dataset, epoch=0)
                                                                                    # train from pretrain, final result in args.model_out_path + "_finetune" + "_global/final"
    #args.STAGE = 1                                                                  # then train AD classifier
    #centralized_training(args=args, model_in_path=args.model_out_path+"_finetune_global/final/", 
    #                     model_out_path=args.model_out_path, train_dataset=train_dataset, test_dataset=test_dataset, epoch=0)
                                                                                    # train from final result from last line, final result in args.model_out_path + "_global/final"
    
    # use the other 50% to train
    args.STAGE = 0                                                                  # train ASR first
    centralized_training(args=args, model_in_path=args.model_out_path+"_global/final/", model_out_path=args.model_out_path+"_finetune2", 
                         train_dataset=train_dataset, test_dataset=test_dataset, epoch=0, client_id="public2")
                                                                                    # train from args.model_out_path+"_global/final/", final result in args.model_out_path + "_finetune2" + "_global/final"
    args.STAGE = 1                                                                  # then train AD classifier
    centralized_training(args=args, model_in_path=args.model_out_path+"_finetune2_global/final/", 
                         model_out_path=args.model_out_path+"2", train_dataset=train_dataset, test_dataset=test_dataset, epoch=0, client_id="public2")
                                                                                    # train from final result from last line, final result in args.model_out_path+"2" + "_global/final"
    
# FL stage 2: Toggling Network
def stage2_training_5050(args, train_dataset, test_dataset):
    local_epoch = args.local_ep                                                     # save given number of local epoch
    ##########################################################
    # Centralized Training: train global Toggling Network    #
    ##########################################################
    
    args.local_ep = args.global_ep                                                  # use number of global epoch for global model
    centralized_training(args=args, model_in_path=args.model_in_path + "2_global/final/", model_out_path=args.model_out_path+"1", 
                         train_dataset=train_dataset, test_dataset=test_dataset, epoch=0)
                                                                                    # train from model_in_path + "2_global/final/"
                                                                                    # final result in args.model_out_path+"1" + "_global/final"
    
    args.local_ep = args.global_ep                                                  # use number of global epoch for global model
    centralized_training(args=args, model_in_path=args.model_out_path + "1_global/final/", model_out_path=args.model_out_path+"2", 
                         train_dataset=train_dataset, test_dataset=test_dataset, client_id="public2")
                                                                                    # train from model_out_path + "1_global/final/"
                                                                                    # final result in args.model_out_path+"2" + "_global/final"


import pickle
import whisper
import os
import pandas as pd
from datasets import Dataset
class TeacherStudentLearning:
    def __init__(self, loadpath=None, savepath=None, load_mdl='large-v2'):
        self.transcript = []
        self.loadpath = loadpath
        self.savepath = savepath
        self.DACS_dataRoot = '/mnt/Internal/FedASR/Data/ADReSSo21/diagnosis/train'
        out_root='/mnt/Internal/FedASR/Data/ADReSSo21/diagnosis'
        out_dirname='transcript_whisper'
        out_filename='train.csv'
        self.out_file=f"{out_root}/{out_dirname}/{out_filename}"
        self.model = whisper.load_model(load_mdl)            
    def add_transcript_to_dataset(self, dataset, transcript_in):
        dataset = dataset.add_column("text", transcript_in)
        return dataset
    
    def save_transcript(self, dataset,outFile):
        df = pd.DataFrame(dataset)
        df.to_csv(outFile, index=False)
    def load_transcript(self, in_file):
        df = pd.read_csv(in_file)
        dataset = Dataset.from_pandas(df)
        return dataset
    def transcribe(self, dataset):
        transcript=[]
        for i,batch in tqdm(enumerate(dataset)):
            singleFile=batch['path']
            file=f'{self.DACS_dataRoot}/clips/{singleFile}'
            result = self.model.transcribe(file, language="en")
            pred_text=result['text'].upper().strip()
            print(pred_text)
            transcript.append(pred_text)
        return transcript
    def transcribe_n_Merge(self, dataset):
        transcript = self.transcribe(dataset)
        ds=self.add_transcript_to_dataset(self, dataset, transcript)
        return ds
    def FilterAvailAudios(self, dataset):
            dataset_enoughLen = dataset.filter(lambda example: len(example['array']) >= 1600)
            dataset_enoughLen_enoughtext = dataset_enoughLen.filter(lambda example: len(example['text']) > 0)
            return dataset_enoughLen_enoughtext


if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    
    args = args_parser()                                                            # get configuration
    exp_details(args)                                                               # print out details based on configuration


    # TODO 用args來選要不要放這個資料庫
    args.dataset = "adress"                                                         # get supervised dataset (adress)
    train_dataset_supervised, test_dataset = get_dataset(args)                      # get dataset

    args.dataset = "adresso"                                                        # get unsupervised dataset (adresso)
    train_dataset_unsupervised, _ = get_dataset(args)                               # get dataset w.o. testing set

    # 創建一個 TranscriptDataset 物件
    TSL = TeacherStudentLearning()
    out_root='/mnt/Internal/FedASR/Data'
    out_dirname='transcript_whisper'
    out_filename='transcript_train.csv'
    out_path=f"{out_root}/{out_dirname}"
    in_file=f"{out_root}/{out_dirname}/{out_filename}"
    import json

    with open('/home/FedASR/dacs/federated/src/transcript.json') as f:
        transcript = json.load(f)


    train_dataset_addresso=TSL.add_transcript_to_dataset(train_dataset_unsupervised, transcript)


    train_dataset_addresso_validated=TSL.FilterAvailAudios(train_dataset_addresso)
    
    processor = Wav2Vec2Processor.from_pretrained(args.pretrain_name)
    train_dataset_addresso_validated = train_dataset_addresso_validated.map(lambda x: prepare_dataset(x, processor=processor, with_transcript=True), num_proc=10)

    train_dataset_unsupervised=train_dataset_addresso_validated
    # 获取数据集 A 和 B 的列名
    # columns_A = train_dataset_supervised.column_names
    # columns_B = train_dataset_addresso_validated.column_names
    # 找到重叠的列
    # overlapping_columns = set(columns_A) & set(columns_B)
    # Remove_columns_A=set(columns_A) - overlapping_columns
    # Remove_columns_B=set(columns_B) - overlapping_columns
    # train_dataset_supervised=train_dataset_supervised.remove_columns(list(Remove_columns_A))
    # train_dataset_addresso_validated=train_dataset_addresso_validated.remove_columns(list(Remove_columns_B))
    
    
    # 加一個fake的零矩陣符合資料格式
    # zeros_list = np.zeros(len(train_dataset_addresso_validated))
    # train_dataset_addresso_validated = train_dataset_addresso_validated.add_column("labels", zeros_list)
    
    # train_dataset_supervised = concatenate_datasets([train_dataset_supervised, train_dataset_addresso_validated])
    # train_dataset_supervised = concatenate_datasets([train_dataset_addresso_validated])
    print(train_dataset_unsupervised)
    if args.EXTRACT != True:                                                        # Training
        if args.FL_STAGE == 1:
            print("| Start FL Training Stage 1|")
            stage1_training(args=args, train_dataset_supervised=train_dataset_supervised, train_dataset_unsupervised=train_dataset_unsupervised, 
                            test_dataset=test_dataset, supervised_level=args.supervised_level)                      
                                                                                    # Train ASR & AD Classifier
            print("| FL Training Stage 1 Done|")
            #print("| Start 50-50 Training Stage 1|")
            #stage1_training_5050(args, train_dataset, test_dataset)                      # Train ASR & AD Classifier
            #print("| 50-50 Training Stage 1 Done|")
        elif args.FL_STAGE == 2:
            print("| Start FL Training Stage 2|")
            args.STAGE = 2
            stage2_training(args=args, train_dataset_supervised=train_dataset_supervised, train_dataset_unsupervised=train_dataset_unsupervised, 
                            test_dataset=test_dataset, supervised_level=args.supervised_level)                          
                                                                                    # Train Toggling Network
            print("| FL Training Stage 2 Done|")
            #print("| Start 50-50 Training Stage 2|")
            #stage2_training_5050(args, train_dataset, test_dataset)                      # Train ASR & AD Classifier
            #print("| 50-50 Training Stage 2 Done|")
    else:
        # TODO: need some modification!!! 要改成supervised + unsupervised的case?
        extract_emb(args, train_dataset, test_dataset)
    
    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))
