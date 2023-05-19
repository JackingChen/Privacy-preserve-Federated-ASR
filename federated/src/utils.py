#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import cifar_iid, cifar_noniid

from transformers import Wav2Vec2Processor
from datasets import Dataset
import librosa
import numpy as np
import pandas as pd
import os
from datasets import load_from_disk
import scipy
import argparse

# 一些參數
parser = argparse.ArgumentParser()
#parser.add_argument('-model', '--model_path', type=str, default="./saves/wav2vec2-base-960h_GRL_0.5", help="Where the model is saved")
parser.add_argument('-opt', '--optimizer', type=str, default="adamw_hf", help="The optimizer to use: adamw_hf, adamw_torch, adamw_apex_fused, or adafactor")
parser.add_argument('-MGN', '--max_grad_norm', type=float, default=1.0, help="Maximum gradient norm (for gradient clipping)")
parser.add_argument('-model_type', '--model_type', type=str, default="data2vec", help="Type of the model")
parser.add_argument('-sr', '--sampl_rate', type=float, default=16000, help="librosa read smping rate")
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4, help="Learning rate")
parser.add_argument('-RD', '--root_dir', default='/mnt/Internal/FedASR/Data/ADReSS-IS2020-data', help="Learning rate")
parser.add_argument('--AudioLoadFunc', default='librosa', help="用scipy function好像可以比較快")
args = parser.parse_args(args=[])


def prepare_dataset(batch, processor, with_transcript=True):
    audio = batch["array"]

    # batched output is "un-batched" to ensure mapping is correct
    batch["input_values"] = processor(audio, sampling_rate=16000).input_values[0]
    
    if with_transcript:                                                     # if given transcript
        with processor.as_target_processor():
            batch["labels"] = processor(batch["text"]).input_ids            # generate labels
        
    return batch

def ID2Label(ID,
            spk2label = np.load("/mnt/Internal/FedASR/weitung/HuggingFace/Pretrain/dataset/test_dic.npy", allow_pickle=True).tolist()):
    name = ID.split("_")                                                    #  from file name to spkID
    if (name[1] == 'INV'):                                                  # interviewer is CC
        label = 0
    else:                                                                   # for participant
        label = spk2label[name[0]]                                          # label according to look-up table
    return label                                                            # return dementia label for this file

"""
def csv2dataset(PATH = '/mnt/Internal/FedASR/Data/ADReSS-IS2020-data/clips/',
                path = '/mnt/Internal/FedASR/Data/ADReSS-IS2020-data/mid_csv/test.csv'):
    stored = "./dataset/" + path.split("/")[-1].split(".")[0]
    if (os.path.exists(stored)):
        print("Load data from local...")
        return load_from_disk(stored)
 
    data = pd.read_csv(path)                                                # read desired csv
    dataset = Dataset.from_pandas(data)                                     # turn into class dataset
    
    # initialize a dictionary
    my_dict = {}
    my_dict["path"] = []                                                    # path to audio
    my_dict["array"] = []                                                   # waveform in array
    my_dict["text"] = []                                                    # ground truth transcript
    my_dict["dementia_labels"] = []

    i = 1
    for path in dataset['path']:                                            # for all files
        if dataset['sentence'][i-1] != None:                                # only the non-empty transcript
            sig, s = librosa.load(PATH + path, sr=16000, dtype='float32')   # read audio w/ 16k sr
            if len(sig) > 1600:                                             # get rid of audio that's too short
                my_dict["path"].append(path)                                # add path
                my_dict["array"].append(sig)                                # add audio wave
                my_dict["text"].append(dataset['sentence'][i-1].upper())    # transcript to uppercase
                my_dict["dementia_labels"].append(ID2Label(path))
        print(i, end="\r")                                                  # print progress
        i += 1
    print("There're ", len(my_dict["path"]), " non-empty files.")

    result_dataset = Dataset.from_dict(my_dict)
    result_dataset.save_to_disk(stored)                                     # save for later use
    
    return result_dataset
"""
# 改版的 
def csv2dataset(audio_path = '{}/clips/'.format(args.root_dir),
                csv_path = '{}/mid_csv/test.csv'.format(args.root_dir),
                dataset_path = "./dataset/", with_transcript=True):
    stored = dataset_path + csv_path.split("/")[-1].split(".")[0]
    if (os.path.exists(stored)):
        print("Load data from local...")
        return load_from_disk(stored)
 
    data = pd.read_csv(csv_path)                                            # read desired csv
    dataset = Dataset.from_pandas(data)                                     # turn into class dataset
    
    # initialize a dictionary
    my_dict = {}
    my_dict["path"] = []                                                    # path to audio
    my_dict["array"] = []                                                   # waveform in array
    if with_transcript:
        my_dict["text"] = []                                                # ground truth transcript if given
    my_dict["dementia_labels"] = []

    if with_transcript:
        spk2label=np.load("/mnt/Internal/FedASR/weitung/HuggingFace/Pretrain/dataset/test_dic.npy", allow_pickle=True).tolist()
    else:
        spk2label=np.load("/mnt/Internal/FedASR/Data/ADReSSo21/diagnosis/spk2label_train.npy", allow_pickle=True).tolist()

    i = 1
    for file_path in dataset['path']:                                            # for all files
        if (dataset['sentence'][i-1] != None) or (with_transcript == False):     # only the non-empty transcript for dataset w/ with_transcript
                                                                                 # or dataset w.o. transcript
            if args.AudioLoadFunc == 'librosa':
                sig, s = librosa.load('{0}/{1}'.format(audio_path,file_path), sr=args.sampl_rate, dtype='float32')  
                                                                                 # read audio w/ 16k sr
            else:
                s, sig = scipy.io.wavfile.read('{0}/{1}'.format(audio_path,file_path))
                sig=librosa.util.normalize(sig)
            if len(sig) > 1600:                                                  # get rid of audio that's too short
                my_dict["path"].append(file_path)                                # add path
                my_dict["array"].append(sig)                                     # add audio wave
                if with_transcript:
                    my_dict["text"].append(dataset['sentence'][i-1].upper())     # transcript to uppercase
                my_dict["dementia_labels"].append(ID2Label(ID=file_path,
                                                           spk2label=spk2label))
        print(i, end="\r")                                                       # print progress
        i += 1
    print("There're ", len(my_dict["path"]), " non-empty files.")

    result_dataset = Dataset.from_dict(my_dict)
    result_dataset.save_to_disk(stored)                                          # save for later use
    
    return result_dataset


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args.dataset == 'cifar':
        data_dir = '../data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups = cifar_noniid(train_dataset, args.num_users)

    elif (args.dataset == 'mnist') or (args.dataset == 'fmnist'):
        if args.dataset == 'mnist':
            data_dir = '../data/mnist/'
        else:
            data_dir = '../data/fmnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = mnist_noniid(train_dataset, args.num_users)
    elif args.dataset == 'adress':                                               # for ADReSS dataset
        processor = Wav2Vec2Processor.from_pretrained(args.pretrain_name)

        # load train / test data
        train_data = csv2dataset(csv_path = "/mnt/Internal/FedASR/Data/ADReSS-IS2020-data/mid_csv/train.csv")
        #dev_data = csv2dataset(path = "/mnt/Internal/FedASR/Data/ADReSS-IS2020-data/mid_csv/dev.csv")
        test_data = csv2dataset(csv_path = "/mnt/Internal/FedASR/Data/ADReSS-IS2020-data/mid_csv/test.csv")

        # map to desired form
        #train_data = train_data.map(prepare_dataset, num_proc=10)
        train_dataset = train_data.map(lambda x: prepare_dataset(x, processor=processor), num_proc=10)
        #dev_data = dev_data.map(prepare_dataset, num_proc=10)
        #train_dataset = dev_data.map(lambda x: prepare_dataset(x, processor=processor), num_proc=10)
        #test_data = test_data.map(prepare_dataset, num_proc=10)
        test_dataset = test_data.map(lambda x: prepare_dataset(x, processor=processor), num_proc=10)
    elif args.dataset == 'adresso':
        processor = Wav2Vec2Processor.from_pretrained(args.pretrain_name)

        # load train / test data
        train_data = csv2dataset(audio_path = "/mnt/Internal/FedASR/Data/ADReSSo21/diagnosis/train/clips/",
                                 csv_path = "/mnt/Internal/FedASR/Data/ADReSSo21/diagnosis/train/train_ADReSSo.csv",
                                 dataset_path = "./dataset/ADReSSo/", with_transcript=False)
        # map to desired form
        train_dataset = train_data.map(lambda x: prepare_dataset(x, processor=processor, with_transcript=False), num_proc=10)
        test_dataset=None                                                        # No testing set for ADResso

    return train_dataset, test_dataset


def average_weights(w):                             # given list of clients' weights
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])                     # save 1st client's model weight
    for key in w_avg.keys():                        # each layer
        for i in range(1, len(w)):                  # for each participated client
            w_avg[key] += w[i][key]                 # sum up weight for this layer
        w_avg[key] = torch.div(w_avg[key], len(w))  # take average (element-wise divide)
    return w_avg


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Global Rounds   : {args.epochs}\n')
    print(f'    Current Stage   : {args.STAGE}\n')
    print(f'    Loss Type       : {args.AD_loss}\n')

    print('    Federated parameters:')
    print(f'    Number of users    : {args.num_users}')
    print(f'    Fraction of users  : {args.frac}')

    return
