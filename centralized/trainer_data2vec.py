from dataclasses import dataclass
from typing import Dict
import numpy as np
from transformers import Wav2Vec2Processor, Data2VecAudioModel
from transformers.training_args import TrainingArguments
from transformers.utils import logging
from transformers.pytorch_utils import torch_int_div
from transformers.models.data2vec.configuration_data2vec_audio import Data2VecAudioConfig

from transformers import TrainingArguments
from transformers import Data2VecAudioConfig
from datasets import load_metric
import argparse
from utils import csv2dataset
from Models import (DataCollatorCTCWithPadding, 
                    Data2VecAudioForCTC,
                    DementiaGRLTrainer)
logger = logging.get_logger(__name__)
# import librosa
# from datasets import load_from_disk
# from jiwer import wer
# import pandas as pd
# from datasets import Dataset
# from torch.utils.data import Dataset
# import os
# from torch import nn
# import torch
# import math
# import torch
# import torch.utils.checkpoint
def prepare_dataset(batch):
    audio = batch["array"]

    # batched output is "un-batched" to ensure mapping is correct
    batch["input_values"] = processor(audio, sampling_rate=16000).input_values[0]
    
    with processor.as_target_processor():
        batch["labels"] = processor(batch["text"]).input_ids
        
    return batch



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

parser = argparse.ArgumentParser()
parser.add_argument('-lam', '--LAMBDA', type=float, default=0.5, help="Lambda for GRL")
parser.add_argument('-st', '--STAGE', type=int, default=1, help="Current training stage")
parser.add_argument('-GRL', '--GRL', action='store_true', default=False, help="True: GRL")
parser.add_argument('-model_in', '--model_in_path', type=str, default="/mnt/Internal/FedASR/weitung/HuggingFace/Pretrain/saves/data2vec-audio-large-960h/final/", help="Where the model is saved")
parser.add_argument('-model_out', '--model_out_path', type=str, default="./saves/data2vec2-base-960h_linear_GRL", help="Where to save the model")
parser.add_argument('-log', '--log_path', type=str, default="data2vec2-base-960h_linear_GRL.txt", help="name for the txt file")
args = parser.parse_args()
LAMBDA = args.LAMBDA                    # lambda for GRL
REVERSE = args.GRL                      # not used in this version
STAGE = args.STAGE                      # stage 1: train AD classifier; stage 2: train toggling network
model_in_dir = args.model_in_path       # path to load the initial model
model_out_dir = args.model_out_path     # path to store the resulted model
log_file = args.log_path                # path to save log file



# threshold for maskes, not used here
AD_THRES = 0.5
LM_THRES = 0.5

# load model from huggingface hub, here data2vec model
name = "facebook/" + model_in_dir.split("/")[-3]
print("Current model: ", name)

mask_time_prob = 0                                         # change config to avoid training stopping
config = Data2VecAudioConfig.from_pretrained(name, mask_time_prob=mask_time_prob)
model = Data2VecAudioForCTC.from_pretrained(name, config=config,LAMBDA=LAMBDA)
processor = Wav2Vec2Processor.from_pretrained(name)
data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

# load train / test data
train_data = csv2dataset(path = "/mnt/Internal/FedASR/Data/ADReSS-IS2020-data/mid_csv/train.csv")
#dev_data = csv2dataset(path = "/mnt/Internal/FedASR/Data/ADReSS-IS2020-data/mid_csv/dev.csv")
test_data = csv2dataset(path = "/mnt/Internal/FedASR/Data/ADReSS-IS2020-data/mid_csv/test.csv")

# map to desired form
train_data = train_data.map(prepare_dataset, num_proc=10)
#dev_data = dev_data.map(prepare_dataset, num_proc=10)
test_data = test_data.map(prepare_dataset, num_proc=10)

        
training_args = TrainingArguments(
    output_dir=model_out_dir,
    group_by_length=True,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    evaluation_strategy="steps",
    num_train_epochs=30,                 # finetune & GRL
    fp16=True,
    gradient_checkpointing=True, 
    save_steps=500,
    eval_steps=500,
    logging_steps=100000000,
    learning_rate=1e-5,
    weight_decay=0.005,
    warmup_steps=1000,
    save_total_limit=2,
    log_level='debug',
    logging_strategy="steps",
    optim="adafactor", #adamw_hf, adamw_torch, adamw_apex_fused, or adafactor.
    #adafactor=False,            # default:false. Whether or not to use transformers.Adafactor optimizer instead of transformers.AdamW
    #fp16_full_eval=True,      # to save memory
    max_grad_norm=0.5
)

trainer = DementiaGRLTrainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_data,
    eval_dataset=test_data,
    tokenizer=processor.feature_extractor,
    log_file=log_file
)

trainer.train() #"./saves/data2vec-audio-large-960h_GRL/checkpoint-56000/"
trainer.save_model(model_out_dir + "/final")











# =======================================

# def ID2Label(ID,
#             spk2label = np.load("/mnt/Internal/FedASR/weitung/HuggingFace/Pretrain/dataset/test_dic.npy", allow_pickle=True).tolist()):
#     name = ID.split("_")                                                    #  from file name to spkID
#     if (name[1] == 'INV'):                                                  # interviewer is CC
#         label = 0
#     else:                                                                   # for participant
#         label = spk2label[name[0]]                                          # label according to look-up table
#     return label                                                            # return dementia label for this file

# def csv2dataset(PATH = '/mnt/Internal/FedASR/Data/ADReSS-IS2020-data/clips/',
#                 path = '/mnt/Internal/FedASR/Data/ADReSS-IS2020-data/mid_csv/test.csv'):
#     stored = "./dataset/" + path.split("/")[-1].split(".")[0]
#     if (os.path.exists(stored)):
#         print("Load data from local...")
#         return load_from_disk(stored)
 
#     data = pd.read_csv(path)                                                # read desired csv
#     dataset = Dataset.from_pandas(data)                                     # turn into class dataset
    
#     # initialize a dictionary
#     my_dict = {}
#     my_dict["path"] = []                                                    # path to audio
#     my_dict["array"] = []                                                   # waveform in array
#     my_dict["text"] = []                                                    # ground truth transcript
#     my_dict["dementia_labels"] = []

#     i = 1
#     for path in dataset['path']:                                            # for all files
#         if dataset['sentence'][i-1] != None:                                # only the non-empty transcript
#             sig, s = librosa.load(PATH + path, sr=16000, dtype='float32')   # read audio w/ 16k sr
#             if len(sig) > 1600:                                             # get rid of audio that's too short
#                 my_dict["path"].append(path)                                # add path
#                 my_dict["array"].append(sig)                                # add audio wave
#                 my_dict["text"].append(dataset['sentence'][i-1].upper())    # transcript to uppercase
#                 my_dict["dementia_labels"].append(ID2Label(path))
#         print(i, end="\r")                                                  # print progress
#         i += 1
#     print("There're ", len(my_dict["path"]), " non-empty files.")

#     result_dataset = Dataset.from_dict(my_dict)
#     result_dataset.save_to_disk(stored)                                     # save for later use
    
#     return result_dataset