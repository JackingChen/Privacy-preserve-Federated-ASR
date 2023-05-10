#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from transformers.training_args import TrainingArguments
from transformers import Trainer
from typing import Dict
import json
import numpy as np
import os
import pandas as pd
from transformers import Data2VecAudioConfig
from models import Data2VecAudioForCTC, Data2VecAudioForCTC_eval, DataCollatorCTCWithPadding
from datasets import Dataset, load_from_disk
import librosa
from jiwer import wer
import copy
from transformers import Data2VecAudioConfig, Wav2Vec2Processor
import copy
from tensorboardX import SummaryWriter

LOG_DIR = './' #log/'

DACS_codeRoot = os.environ.get('DACS_codeRoot')
DACS_dataRoot = os.environ.get('DACS_dataRoot')

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
        file_object = open(LOG_DIR , 'a')
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

def csv2dataset(PATH = f'{DACS_codeRoot}/clips/',
                path = f'{DACS_codeRoot}/mid_csv/test.csv'):
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
                       #'array': str(batch["array"]),
                       'text': batch["text"],                                   # ground truth transcript
                       'dementia_labels': batch["dementia_labels"],
                       #'input_values': str(batch["input_values"]),              # input of the model
                       #'labels': str(batch["labels"]),
                       #'ASR logits': str(logits["ASR logits"].tolist()),
                       #'dementia logits': str(logits["dementia logits"].tolist()),
                       'hidden_states': str(logits["hidden_states"].tolist()),
                       #'pred_AD': batch["pred_AD"],                             # AD prediction
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

def update_network(args, source_path, target_path, network):                        # replace "network" in source_path with that in target_path
    # read source model                                                             # return model
    mask_time_prob = 0                                                              # change config to avoid training stopping
    config = Data2VecAudioConfig.from_pretrained(args.pretrain_name, mask_time_prob=mask_time_prob)
                                                                                    # use pre-trained config
    model = Data2VecAudioForCTC.from_pretrained(source_path, config=config, args=args)
    model.config.ctc_zero_infinity = True                                           # to avoid inf values

    target_model = Data2VecAudioForCTC.from_pretrained(target_path, config=config, args=args)
    target_model.config.ctc_zero_infinity = True                                    # to avoid inf values

    if network == "ASR":                                                            # replace ASR's weight
        model.data2vec_audio.load_state_dict(target_model.data2vec_audio.state_dict())
        model.lm_head.load_state_dict(target_model.lm_head.state_dict())

    elif network == "AD":                                                           # replace AD classifier's weight
        model.dementia_head.load_state_dict(target_model.dementia_head.state_dict())         

    elif network == "toggling_network":                                             # replace toggling network's weight
        model.arbitrator.load_state_dict(target_model.arbitrator.state_dict())

    return copy.deepcopy(model)

def update_network_weight(args, source_path, target_weight, network):               # update "network" in source_path with given weights
    # read source model                                                             # return model   
    mask_time_prob = 0                                                              # change config to avoid training stopping
    config = Data2VecAudioConfig.from_pretrained(args.pretrain_name, mask_time_prob=mask_time_prob)
                                                                                    # use pre-trained config
    model = Data2VecAudioForCTC.from_pretrained(source_path, config=config, args=args)
    model.config.ctc_zero_infinity = True                                           # to avoid inf values

    if network == "ASR":                                                            # given weight from ASR
        data2vec_audio, lm_head = target_weight

        model.data2vec_audio.load_state_dict(data2vec_audio)                        # replace ASR encoder's weight
        model.lm_head.load_state_dict(lm_head)                                      # replace ASR decoder's weight

    elif network == "AD":                                                           # given weight from AD
        model.dementia_head.load_state_dict(target_weight)                          # replace AD classifier's weight

    elif network == "toggling_network":                                             # given weight from toggling network
        model.arbitrator.load_state_dict(target_weight)                             # replace toggling network's weight

    return copy.deepcopy(model)

def get_model_weight(args, source_path, network):                                   # get "network" weights from model in source_path
    mask_time_prob = 0                                                              # change config to avoid training stopping
    config = Data2VecAudioConfig.from_pretrained(args.pretrain_name, mask_time_prob=mask_time_prob)
                                                                                    # use pre-trained config
    model = Data2VecAudioForCTC.from_pretrained(source_path, config=config, args=args)
                                                                                    # load from source
    model.config.ctc_zero_infinity = True                                           # to avoid inf values

    if network == "ASR":                                                            # get ASR weights
        return_weights = [copy.deepcopy(model.data2vec_audio.state_dict()), copy.deepcopy(model.lm_head.state_dict())]
    elif network == "AD":                                                           # get AD classifier weights
        return_weights = copy.deepcopy(model.dementia_head.state_dict())
    elif network == "toggling_network":                                             # get toggling network weights
        return_weights = copy.deepcopy(model.arbitrator.state_dict())  
    
    return return_weights

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
    def __init__(self, args, dataset, global_test_dataset, client_id, 
                 model_in_path, model_out_path):
        self.args = args                                                            # given configuration
        self.client_train_dataset = self.train_split(dataset, client_id)            # get subset of training set (dataset of THIS client)
        
        self.device = 'cuda' if args.gpu else 'cpu'                                 # use gpu or cpu
        
        self.global_test_dataset = global_test_dataset
        self.client_test_dataset = self.test_split(global_test_dataset, client_id)  # get subset of testing set (dataset of THIS client)
        self.processor = Wav2Vec2Processor.from_pretrained(args.pretrain_name)
        self.data_collator = DataCollatorCTCWithPadding(processor=self.processor, padding=True)
        self.client_id = client_id

        self.model_in_path = model_in_path                                          # no info for client_id & global_round
        self.model_out_path = model_out_path                                        # no info for client_id & global_round

    def train_split(self, dataset, client_id):
        # generate sub- training set for given user-ID
        if client_id == "public":                                                   # get spk_id for public dataset, 54 PAR (50% of all training set)
            client_spks = ['S086', 'S021', 'S018', 'S156', 'S016', 'S077', 'S027', 'S116', 'S143', 'S082', 'S039', 'S150', 'S004', 'S126', 'S137', 
            'S097', 'S128', 'S059', 'S096', 'S081', 'S135', 'S094', 'S070', 'S049', 'S080', 'S040', 'S076', 'S093', 'S141', 'S034', 'S056', 'S090', 
            'S130', 'S092', 'S055', 'S019', 'S154', 'S017', 'S114', 'S100', 'S036', 'S029', 'S127', 'S073', 'S089', 'S051', 'S005', 'S151', 'S003', 
            'S033', 'S007', 'S084', 'S043', 'S009']                                 # 27 AD + 27 HC

        elif client_id == 0:                                                        # get spk_id for client 1, 27 PAR (25% of all training set)
            client_spks = ['S058', 'S030', 'S064', 'S104', 'S048', 'S118', 'S122', 'S001', 'S087', 'S013', 'S025', 'S083', 'S067', 'S068', 'S111', 
            'S028', 'S015', 'S108', 'S095', 'S002', 'S072', 'S020', 'S148', 'S144', 'S110', 'S124', 'S129']
                                                                                    # 13 AD + 14 HC
        elif client_id == 1:                                                        # get spk_id for client 2, 27 PAR (25% of all training set)  
            client_spks = ['S071', 'S136', 'S140', 'S145', 'S032', 'S101', 'S103', 'S139', 'S038', 'S153', 'S035', 'S011', 'S132', 'S006', 'S149', 
            'S041', 'S079', 'S107', 'S063', 'S061', 'S125', 'S062', 'S012', 'S138', 'S024', 'S052', 'S142']
                                                                                    # 14 AD + 13 HC
        else:
            print("Train with whole dataset!!")
            return dataset
        
        print("Generating client training set for client ", str(client_id), "...")
        client_train_dataset = dataset.filter(lambda example: example["path"].startswith(tuple(client_spks)))
        
        return client_train_dataset
    
    def test_split(self, dataset, client_id):
        # generate sub- testing set for given user-ID
        if client_id == "public":                                                   # get spk_id for public dataset, 24 PAR (50% of all testing set)
            client_spks = ['S197', 'S163', 'S193', 'S169', 'S196', 'S184', 'S168', 'S205', 'S185', 'S171', 'S204', 'S173', 'S190', 'S191', 'S203', 
                           'S180', 'S165', 'S199', 'S160', 'S175', 'S200', 'S166', 'S177', 'S167']                                # 12 AD + 12 HC

        elif client_id == 0:                                                        # get spk_id for client 1, 12 PAR (25% of all testing set)
            client_spks = ['S198', 'S182', 'S194', 'S161', 'S195', 'S170', 'S187', 'S192', 'S178', 'S201', 'S181', 'S174']
                                                                                    # 6 AD + 6 HC
        elif client_id == 1:                                                        # get spk_id for client 2, 12 PAR (25% of all testing set)  
            client_spks = ['S179', 'S188', 'S202', 'S162', 'S172', 'S183', 'S186', 'S207', 'S189', 'S164', 'S176', 'S206']
                                                                                    # 6 AD + 6 HC
        else:
            print("Test with whole dataset!!")
            return dataset
        
        print("Generating client testing set for client ", str(client_id), "...")
        client_test_dataset = dataset.filter(lambda example: example["path"].startswith(tuple(client_spks)))
        
        return client_test_dataset
    
    def record_result(self, trainer, result_folder):                                # save training loss, testing loss, and testing wer
        logger = SummaryWriter('../logs/' + result_folder.split("/")[-1])           # use name of this model as folder's name

        for idx in range(len(trainer.state.log_history)):
            if "loss" in trainer.state.log_history[idx].keys():                     # add in training loss, epoch*100 to obtain int
                logger.add_scalar('Loss/train', trainer.state.log_history[idx]["loss"], trainer.state.log_history[idx]["epoch"]*100)

            elif "eval_loss" in trainer.state.log_history[idx].keys():              # add in testing loss & WER, epoch*100 to obtain int
                logger.add_scalar('Loss/test', trainer.state.log_history[idx]["eval_loss"], trainer.state.log_history[idx]["epoch"]*100)
                logger.add_scalar('wer/test', trainer.state.log_history[idx]["eval_wer"], trainer.state.log_history[idx]["epoch"]*100)

            else:                                                                   # add in final training loss, epoch*100 to obtain int
                logger.add_scalar('Loss/train', trainer.state.log_history[idx]["train_loss"], trainer.state.log_history[idx]["epoch"]*100)
        logger.close()

    def update_weights(self, global_weights, global_round):
        if global_weights == None:                                                  # train from model from model_in_path
            mask_time_prob = 0                                                      # change config to avoid training stopping
            config = Data2VecAudioConfig.from_pretrained(self.args.pretrain_name, mask_time_prob=mask_time_prob)
                                                                                    # use pre-trained config
            model = Data2VecAudioForCTC.from_pretrained(self.model_in_path, config=config, args=self.args)
            model.config.ctc_zero_infinity = True                                   # to avoid inf values
        else:                                                                       # update train model using given weight
            if self.args.STAGE == 0:                                                # train ASR
                model = update_network_weight(args=self.args, source_path=self.model_in_path, target_weight=global_weights, network="ASR")                    
                                                                                    # from model from model_in_path, update ASR's weight
            elif self.args.STAGE == 1:                                              # train AD classifier
                model = update_network_weight(args=self.args, source_path=self.model_in_path, target_weight=global_weights, network="AD")           
                                                                                    # from model from model_in_path, update AD classifier's weight
            elif self.args.STAGE == 2:                                              # train toggling network
                model = update_network_weight(args=self.args, source_path=self.model_in_path, target_weight=global_weights, network="toggling_network")             
                                                                                    # from model from model_in_path, update arbitrator's weight
        global log_path
        log_path = self.log_path

        model.train()
        if self.args.STAGE == 0:                                                    # fine-tune ASR
            lr = 1e-5
        elif self.args.STAGE == 1:                                                  # train AD classifier
            lr = 1e-4
        elif self.args.STAGE == 2:                                                  # train toggling network
            lr = 1e-3

        if self.client_id == "public":                                              # model train with public dataset, name end with "_global"
            save_path = self.model_out_path + "_global"
        else:
            save_path = self.model_out_path + "_client" + str(self.client_id) + "_round" + str(global_round)
                                                                                    # for local models, record info for id & num_round
        training_args = TrainingArguments(
            output_dir=save_path,
            group_by_length=True,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            evaluation_strategy="steps",
            num_train_epochs=self.args.local_ep, #self.args.local_ep
            fp16=True,
            gradient_checkpointing=True, 
            save_steps=500, # 500
            eval_steps=500, # 500
            logging_steps=10, # 500
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

        print(" | Client ", str(self.client_id), " ready to train! |")
        trainer.train()
        trainer.save_model(save_path + "/final")                                    # save final model
        self.record_result(trainer, save_path)                           # save training loss, testing loss, and testing wer

        # get "network" weights from model in source_path
        if self.args.STAGE == 0:                                                    # train ASR
            return_weights = get_model_weight(args=self.args, source_path=save_path + "/final/", network="ASR")
        elif self.args.STAGE == 1:                                                  # train AD classifier
            return_weights = get_model_weight(args=self.args, source_path=save_path + "/final/", network="AD")
        elif self.args.STAGE == 2:                                                  # train toggling_network
            return_weights = get_model_weight(args=self.args, source_path=save_path + "/final/", network="toggling_network")  
         
        return return_weights, trainer.state.log_history[-1]["train_loss"]          # return weight, average losses for this round

    def extract_embs(self):                                                         # extract emb. using model in args.model_in_path
        # load model
        mask_time_prob = 0                                                          # change config to avoid code from stopping
        config = Data2VecAudioConfig.from_pretrained(self.args.pretrain_name, mask_time_prob=mask_time_prob)
        model = Data2VecAudioForCTC_eval.from_pretrained(self.args.model_in_path, config=config, args=self.args)
        processor = self.processor

        # get emb.s, masks... 1 sample by 1 sample for client test
        df = map_to_result(self.client_test_dataset[0], processor, model, 0)
        for i in range(len(self.client_test_dataset) - 1):
            df2 = map_to_result(self.client_test_dataset[i+1], processor, model, i+1)
            df = pd.concat([df, df2], ignore_index=True)
            print("\r"+ str(i), end="")

        csv_path = "./results/" + self.args.csv_path + ".csv"
        df.to_csv(csv_path)
        print("Testing data Done")

        # get emb.s, masks... 1 sample by 1 sample for client train
        df = map_to_result(self.client_train_dataset[0], processor, model, 0)
        for i in range(len(self.client_train_dataset) - 1):
            df2 = map_to_result(self.client_train_dataset[i+1], processor, model, i+1)
            df = pd.concat([df, df2], ignore_index=True)
            print("\r"+ str(i), end="")

        csv_path = "./results/" + self.args.csv_path + "_train.csv"
        df.to_csv(csv_path)
        print("Training data Done")

        print(self.args.csv_path + " All Done")

    



    
