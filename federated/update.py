#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from transformers.training_args import TrainingArguments
from transformers import Trainer
from typing import Any, Dict, List, Optional, Union
import json
import numpy as np
import os
import pandas as pd
from transformers import Data2VecAudioConfig
from models import Data2VecAudioForCTC, Data2VecAudioForCTC_eval
from datasets import Dataset, load_from_disk
import librosa
from jiwer import wer
import copy

LOG_DIR = './' #log/'

from datasets import load_metric
wer_metric = load_metric("wer")
def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

class CustomTrainer(Trainer):    
    def compute_loss(self, model, inputs, return_outputs=False):
            """
            How the loss is computed by Trainer. By default, all models return the loss in the first element.
            Subclass and override for custom behavior.
            """
            #dementia_labels = inputs.pop("dementia_labels") # pop 出來就會不見?
            
            if self.label_smoother is not None and "labels" in inputs:
                labels = inputs.pop("labels")
            else:
                labels = None
            
            outputs = model(**inputs)
            # Save past state if it exists
            # TODO: this needs to be fixed and made cleaner later.
            if self.args.past_index >= 0:
                self._past = outputs[self.args.past_index]

            if labels is not None:
                loss = self.label_smoother(outputs, labels)
            else:
                # We don't use .loss here since the model may return tuples instead of ModelOutput.
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

            return (loss, outputs) if return_outputs else loss
    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training.
        Subclass and override this method to inject custom behavior.
        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        
        # write to txt file
        file_object = open(LOG_DIR + log_path, 'a')
        # Append at the end of file
        file_object.write(json.dumps(output) + '\n')
        # Close the file
        file_object.close()

        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)

def csv2dataset(PATH = '/mnt/Internal/FedASR/Data/ADReSS-IS2020-data/clips/',
                path = '/mnt/Internal/FedASR/Data/ADReSS-IS2020-data/mid_csv/test.csv'):
    stored = "./dataset/" + path.split("/")[-1].split(".")[0]
    if (os.path.exists(stored)):
        print("Load data from local...")
        return load_from_disk(stored)
 
    data = pd.read_csv(path)                                               # read desired csv
    dataset = Dataset.from_pandas(data)                                    # turn into class dataset
    
    # initialize a dictionary
    my_dict = {}
    my_dict["path"] = []                                                   # path to audio
    my_dict["array"] = []                                                  # waveform in array
    my_dict["text"] = []                                                   # ground truth transcript
    my_dict["dementia_labels"] = []

    i = 1
    for path in dataset['path']:                                           # for all files
        if dataset['sentence'][i-1] != None:                               # only the non-empty transcript
            sig, s = librosa.load(PATH + path, sr=16000, dtype='float32')  # read audio w/ 16k sr
            if len(sig) > 1600:                                            # get rid of audio that's too short
                my_dict["path"].append(path)                                   # add path
                my_dict["array"].append(sig)                                   # add audio wave
                my_dict["text"].append(dataset['sentence'][i-1].upper())       # transcript to uppercase
                my_dict["dementia_labels"].append(ID2Label(path))
        print(i, end="\r")                                                 # print progress
        i += 1
    print("There're ", len(my_dict["path"]), " non-empty files.")

    result_dataset = Dataset.from_dict(my_dict)
    result_dataset.save_to_disk(stored)
    
    return result_dataset

def prepare_dataset(batch):
    audio = batch["array"]

    # batched output is "un-batched" to ensure mapping is correct
    batch["input_values"] = processor(audio, sampling_rate=16000).input_values[0]
    
    with processor.as_target_processor():
        batch["labels"] = processor(batch["text"]).input_ids
        
    return batch

def map_to_result(batch, processor, model, idx):
    with torch.no_grad():
        input_values = torch.tensor(batch["input_values"]).unsqueeze(0)            
        logits = model(input_values).logits                                     # includes ASR logits, dementia logits, hidden_states
        asr_lg = logits['ASR logits']
        AD_lg = logits['dementia logits'][0]                                    # (1, time-step, 2) --> (time-step, 2)
    
    pred_ad_tstep = torch.argmax(AD_lg, dim=-1)                                 # pred of each time-step
    pred_ad = pred_ad_tstep.sum() / pred_ad_tstep.size()[0]                     # average result
    if pred_ad > 0.5:                                                           # if over half of the time pred AD
        batch["pred_AD"] = 1                                                    # save final result as AD
    else:
        batch["pred_AD"] = 0
    
    pred_ids = torch.argmax(asr_lg, dim=-1)
    batch["pred_str"] = processor.batch_decode(pred_ids)[0]                     # predicted transcript
    batch["text"] = processor.decode(batch["labels"], group_tokens=False)       # ground truth transcript
    
    # for toggle
    
    df = pd.DataFrame({'path': batch["path"],                                   # to know which sample
                       'array': str(batch["array"]),
                       'text': batch["text"],                                   # ground truth transcript
                       'dementia_labels': batch["dementia_labels"],
                       'input_values': str(batch["input_values"]),              # input of the model
                       'labels': str(batch["labels"]),
                       'ASR logits': str(logits["ASR logits"].tolist()),
                       'dementia logits': str(logits["dementia logits"].tolist()),
                       'hidden_states': str(logits["hidden_states"].tolist()),
                       'pred_AD': batch["pred_AD"],                             # AD prediction
                       'pred_str': batch["pred_str"],                           # predicted transcript
                       'dementia_mask': str(logits["dementia_mask"].tolist()),  # ASR-free mask for AD classification
                       'lm_mask': str(logits["lm_mask"].tolist())},             # AD-free mask for ASR task
                      index=[idx])
    
    """
    # for model w.o. FSM
    df = pd.DataFrame({'path': batch["path"],                                    # to know which sample
                    'array': str(batch["array"]),
                    'text': batch["text"],
                    'dementia_labels': batch["dementia_labels"],
                    'input_values': str(batch["input_values"]),               # input of the model
                    'labels': str(batch["labels"]),
                    'ASR logits': str(logits["ASR logits"].tolist()),
                    'dementia logits': str(logits["dementia logits"].tolist()),
                    'hidden_states': str(logits["hidden_states"].tolist()),
                    'pred_AD': batch["pred_AD"],                             # AD prediction
                    'pred_str': batch["pred_str"]},
                    index=[idx])
    """
    return df

def ID2Label(ID,
            spk2label = np.load("/mnt/Internal/FedASR/weitung/HuggingFace/Pretrain/dataset/test_dic.npy", allow_pickle=True).tolist()):
    name = ID.split("_")                                  #  from file name to spkID
    if (name[1] == 'INV'):                                # interviewer is CC
        label = 0
    else:                                                 # for participant
        label = spk2label[name[0]]                        # label according to look-up table
    return label                                          # return dementia label for this file

class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, logger):
        self.args = args                                                        # given configuration
        self.logger = logger
        self.trainloader, self.validloader, self.testloader = self.train_val_test(
            dataset, list(idxs))                                                # get subset of training set (dataset of THIS client), divide into train-dev-test
                                                                                # idxs: ID of samples
        self.device = 'cuda' if args.gpu else 'cpu'                             # use gpu or cpu
        # Default criterion set to NLL loss function
        self.criterion = nn.NLLLoss().to(self.device)                           # loss function

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (80, 10, 10)
        idxs_train = idxs[:int(0.8*len(idxs))]                                  # 80% as training
        idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]                  # 10% as validation
        idxs_test = idxs[int(0.9*len(idxs)):]                                   # 10% as testing

        # to data loader
        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                 batch_size=int(len(idxs_val)/10), shuffle=False) # 沒用到
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=int(len(idxs_test)/10), shuffle=False)
        return trainloader, validloader, testloader

    def update_weights(self, model, global_round):                              # given global model
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        for iter in range(self.args.local_ep):                                  # train for local epochs
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()                                               # reset gradient
                log_probs = model(images)                                       # training data feed to model
                loss = self.criterion(log_probs, labels)                        # compute loss
                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 10 == 0):                 # print every 10 steps
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())                                  # save loss for each step
            epoch_loss.append(sum(batch_loss)/len(batch_loss))                  # average losses to loss for this epoch

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)            # return weight, average losses to loss for this round

    def inference(self, model):                                                 # given global model
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)                                             # testing data feed to model
            batch_loss = self.criterion(outputs, labels)                        # compute loss
            loss += batch_loss.item()                                           # sum up losses

            # Prediction
            _, pred_labels = torch.max(outputs, 1)                              # make prediction
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()          # sum up num of correct predictions
            total += len(labels)                                                # sum up num of evaluated samples

        accuracy = correct/total
        return accuracy, loss                                                   # return acc. & total loss


def test_inference(args, model, test_dataset):                                  # given global model and global testing set
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = 'cuda' if args.gpu else 'cpu'                                      # use cpu or gpu
    criterion = nn.NLLLoss().to(device)                                         # loss function
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)                                      # data loader of global testing set

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)                                                 # testing data feed to model
        batch_loss = criterion(outputs, labels)                                 # compute loss
        loss += batch_loss.item()                                               # sum up loss

        # Prediction
        _, pred_labels = torch.max(outputs, 1)                                  # make prediction
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()              # sum up num of correct prediction
        total += len(labels)                                                    # sum up num of evaluated samples

    accuracy = correct/total
    return accuracy, loss                                                       # return acc. & total loss

class ASRLocalUpdate(object):
    def __init__(self, args, dataset, logger, data_collator, global_test_dataset, processor, client_id):
        print("initialize ASRLocalUpdate")
        self.args = args                                                        # given configuration
        self.logger = logger
        self.client_train_dataset = self.client_train(dataset, client_id)       # get subset of training set (dataset of THIS client)
                                                                                # idxs: ID of samples
        self.device = 'cuda' if args.gpu else 'cpu'                             # use gpu or cpu
        
        self.data_collator = data_collator
        self.global_test_dataset = global_test_dataset
        self.processor = processor
        self.client_id = client_id

    def client_train(self, dataset, client_id):
        # generate sub- training set for given user-ID
        if client_id == "public":                                               # get spk_id for public dataset, 54 PAR (50% of all training set)
            client_spks = ['S086', 'S021', 'S018', 'S156', 'S016', 'S077', 'S027', 'S116', 'S143', 'S082', 'S039', 'S150', 'S004', 'S126', 'S137', 
            'S097', 'S128', 'S059', 'S096', 'S081', 'S135', 'S094', 'S070', 'S049', 'S080', 'S040', 'S076', 'S093', 'S141', 'S034', 'S056', 'S090', 
            'S130', 'S092', 'S055', 'S019', 'S154', 'S017', 'S114', 'S100', 'S036', 'S029', 'S127', 'S073', 'S089', 'S051', 'S005', 'S151', 'S003', 
            'S033', 'S007', 'S084', 'S043', 'S009']                             # 27 AD + 27 HC
        elif client_id == 0:                                                    # get spk_id for client 1, 27 PAR (25% of all training set)
            client_spks = ['S058', 'S030', 'S064', 'S104', 'S048', 'S118', 'S122', 'S001', 'S087', 'S013', 'S025', 'S083', 'S067', 'S068', 'S111', 
            'S028', 'S015', 'S108', 'S095', 'S002', 'S072', 'S020', 'S148', 'S144', 'S110', 'S124', 'S129']
                                                                                # 13 AD + 14 HC
        elif client_id == 1:                                                    # get spk_id for client 2, 27 PAR (25% of all training set)  
            client_spks = ['S071', 'S136', 'S140', 'S145', 'S032', 'S101', 'S103', 'S139', 'S038', 'S153', 'S035', 'S011', 'S132', 'S006', 'S149', 
            'S041', 'S079', 'S107', 'S063', 'S061', 'S125', 'S062', 'S012', 'S138', 'S024', 'S052', 'S142']
                                                                                # 14 AD + 13 HC
        else:
            print("Train with whole dataset!!")
            return dataset
        print("Generating client training set for client ", str(client_id), "...")
        client_train_dataset = dataset.filter(lambda example: example["path"].startswith(tuple(client_spks)))
        
        return client_train_dataset

    def update_weights(self, global_arbitrator, global_round):                              # given global model
        print("load model")
        # load accoring to client_id
        mask_time_prob = 0                                                      # change config to avoid training stopping
        config = Data2VecAudioConfig.from_pretrained(self.args.pretrain_name, mask_time_prob=mask_time_prob)
        #print("load global!!!!!!!!!!!!! should change to client model")
        model = Data2VecAudioForCTC.from_pretrained(self.args.model_in_path, config=config, args=self.args)
        model.config.ctc_zero_infinity = True                                   # to avoid inf values
        #print("before: ", model.arbitrator.state_dict())
        #model.arbitrator = global_arbitrator                        # replace w/ global_arbitrator
        model.arbitrator.load_state_dict(global_arbitrator.state_dict())        # update arbitrator's weight
        #print("after: ", model.arbitrator.state_dict())

        # Set mode to train model
        #print('| Global Round :', str(global_round), ' |')
        global log_path
        log_path = self.args.log_path

        model.train()
        if self.args.STAGE == 1:
            lr = 1e-4
        elif self.args.STAGE == 2:
            lr = 1e-3

        training_args = TrainingArguments(
            output_dir=self.args.model_out_path + "_" + str(self.client_id), # use client_num to name model
            group_by_length=True,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            evaluation_strategy="steps",
            num_train_epochs=self.args.local_ep, #self.args.local_ep
            fp16=True,
            gradient_checkpointing=True, 
            save_steps=500, # 500
            eval_steps=500, # 500
            logging_steps=500, # 500
            learning_rate=lr, # self.args.lr
            weight_decay=0.005,
            warmup_steps=1000,
            save_total_limit=2,
            log_level='debug',
            logging_strategy="steps",
            #adafactor=True,            # default:false. Whether or not to use transformers.Adafactor optimizer instead of transformers.AdamW
            #fp16_full_eval=True,      # to save memory
            #max_grad_norm=0.5
        )
        global processor
        processor = self.processor

        trainer = CustomTrainer(
            model=model,
            data_collator=self.data_collator,
            args=training_args,
            compute_metrics=compute_metrics,
            train_dataset=self.client_train_dataset,
            eval_dataset=self.global_test_dataset,
            tokenizer=self.processor.feature_extractor,
        )

        print(str(self.client_id), " ready to train!")
        trainer.train()
        save_path = self.args.model_out_path + "_client" + str(self.client_id) + "_round" + str(global_round) + "/final"
        trainer.save_model(save_path)                                                       # save local model

        # reload local model to return toggling network's weight
        mask_time_prob = 0                                                          # change config to avoid training stopping
        config = Data2VecAudioConfig.from_pretrained(self.args.pretrain_name, mask_time_prob=mask_time_prob)
        model = Data2VecAudioForCTC.from_pretrained(save_path, config=config, args=self.args)
                                                                                    # load local model
        model.config.ctc_zero_infinity = True                                       # to avoid inf values
 
        arbitrator = copy.deepcopy(model.arbitrator)                                # return weight for toggling network only

        return_weights = copy.deepcopy(arbitrator.state_dict())                       # save global weight
        return return_weights, trainer.state.log_history[-1]["train_loss"]       # return weight, average losses to loss for this round

    def inference(self, global_arbitrator):                                                 # given global model
        """ Returns the inference accuracy and loss.
        """
        # load accoring to client_id
        mask_time_prob = 0                                         # change config to avoid training stopping
        config = Data2VecAudioConfig.from_pretrained(self.args.pretrain_name, mask_time_prob=mask_time_prob)
        #print("load global!!!!!!!!!!!!! should change to client model")
        model = Data2VecAudioForCTC_eval.from_pretrained(self.args.model_out_path + "_" + str(self.client_id) + "/final", config=config, args=self.args) 
                                                                                            # load local model model for eval
        model.config.ctc_zero_infinity = True                      # to avoid inf values


        model.eval()

        # evaluate on global testing set
        #result = self.global_test_dataset.map(lambda x: map_to_result(x, processor=processor, model=model), num_proc=10)
        df = map_to_result(self.global_test_dataset[0], processor, model, 0)
        for i in range(len(self.global_test_dataset) - 1):
            df2 = map_to_result(self.global_test_dataset[i+1], processor, model, i+1)
            df = pd.concat([df, df2], ignore_index=True)
            print("\r"+ str(i), end="")

        csv_path = "./save/results/" + self.args.csv_path + "_" + str(self.client_id) + "_GlobalTest.csv"
        df.to_csv(csv_path)
        WER = wer(df["text"].to_list(), df["pred_str"].to_list())

        return WER                                                   # return WER