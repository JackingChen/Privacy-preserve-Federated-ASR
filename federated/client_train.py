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
import argparse

def client_train(args, train_dataset, logger,
                 test_dataset, idx, epoch, global_weights=None):
    if args.model == 'data2vec':
        mask_time_prob = 0                                                          # change config to avoid training stopping
        config = Data2VecAudioConfig.from_pretrained(args.pretrain_name, mask_time_prob=mask_time_prob)
        print("load from ", args.model_in_path)
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
    # w, loss = global_model.state_dict(), 0
    #send_end.send([w, loss])                                                        # save model weights and average round loss
    #return_dict[str(idx)] = [w, loss]
    print("PID {} Getting ".format(os.getpid()), "Done")
    return w, loss        