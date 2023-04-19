import os
import json
import argparse
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from torch import nn
from datasets import Dataset, load_from_disk, load_metric
from jiwer import wer
from transformers import (
    Trainer,
    TrainingArguments,
    Wav2Vec2Processor,
)
from transformers.file_utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings_to_model_forward,
)
from transformers.modeling_outputs import CausalLMOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from Models import (
    Data2VecAudioForCTC,
    Data2VecAudioPreTrainedModel,
    DataCollatorCTCWithPadding,
    FSMatt_loss,
    RecallLoss,
    ReverseLayerF,
    gumbel_softmax,
)
from functions.models import AngularPenaltySMLoss
from utils import csv2dataset, compute_metrics_ADsupport as compute_metrics

from transformers.models.data2vec.configuration_data2vec_audio import Data2VecAudioConfig
from transformers.models.data2vec.modeling_data2vec_audio import (
    Data2VecAudioEncoder,
    Data2VecAudioFeatureEncoder,
    Data2VecAudioFeatureProjection,
    Data2VecAudioPositionalConvLayer,
)

# import necessary modules
import math
from typing import Optional

import numpy as np
import torch.utils.checkpoint
from torch import nn

from transformers import Data2VecAudioModel
def prepare_dataset(batch):
    audio = batch["array"]

    # batched output is "un-batched" to ensure mapping is correct
    batch["input_values"] = processor(audio, sampling_rate=16000).input_values[0]
    
    with processor.as_target_processor():
        batch["labels"] = processor(batch["text"]).input_ids
        
    return batch


class Data2VecAudioForCTC(Data2VecAudioPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.data2vec_audio = Data2VecAudioModel(config)
        self.dropout = nn.Dropout(config.final_dropout)

        if config.vocab_size is None:
            raise ValueError(
                f"You are trying to instantiate {self.__class__} with a configuration that "
                "does not define the vocabulary size of the language model head. Please "
                "instantiate the model as follows: `Data2VecAudioForCTC.from_pretrained(..., vocab_size=vocab_size)`. "
                "or define `vocab_size` of your model's configuration."
            )

        self.alpha=torch.tensor(LAMBDA)
        self.dementia_thres = torch.tensor(AD_THRES)
        self.lm_thres = torch.tensor(LM_THRES)
        print("lambda = ", self.alpha)
        print("dementia_thres = ", self.dementia_thres)
        print("lm_thres = ", self.lm_thres)

        # 加lm相關components
        self.lm_fsm = nn.Linear(config.hidden_size, config.hidden_size)          # 找出對lm重要的feat
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)          # output字母的"機率"
        self.lm_grl = nn.Linear(config.hidden_size, config.vocab_size)           # 加了GRL那條
        
        # 加dementia相關components
        self.dementia_fsm = nn.Linear(config.hidden_size, config.hidden_size)    # 找出對AD預測重要的feat
        self.dementia_head = nn.Linear(config.hidden_size, 2)                    # 辨識AD
        self.dementia_grl = nn.Linear(config.hidden_size, 2)                     # 加GRL那條
        
        # define similarity loss: AM-Softmax, aka div loss
        self.criterion_similar = AngularPenaltySMLoss(in_features=config.hidden_size, out_features=2, loss_type='cosface').to('cpu')
        
        # freeze feature_extractor    
        self.freeze_feature_encoder()

        # skip to stage 6
        if STAGE == 1:                                                           # train FSMs alone
            print("Current stage: 1")
            self.freeze_lm_grl()
            self.freeze_dementia_grl()
            self.freeze_lm_head()
            self.freeze_dementia_head()
        elif STAGE == 2:                                                         # train FSM + head
            print("Current stage: 2")
            self.freeze_lm_grl()
            self.freeze_dementia_grl()
        elif STAGE == 3:                                                         # train dementia GRL
            print("Current stage: 3")
            self.freeze_data2vec_audio()
            self.freeze_lm_fsm()
            self.freeze_dementia_fsm()
            self.freeze_lm_head()
            self.freeze_dementia_head()
            self.freeze_lm_grl()
        elif STAGE == 4:                                                         # train lm GRL
            print("Current stage: 4")
            self.freeze_data2vec_audio()
            self.freeze_lm_fsm()
            self.freeze_dementia_fsm()
            self.freeze_lm_head()
            self.freeze_dementia_head()
            self.freeze_dementia_grl()
        elif STAGE == 5:                                                         # train lm_FSM
            self.freeze_data2vec_audio()
            self.freeze_dementia_fsm()            
            self.freeze_criterion_similar()
            self.freeze_lm_head()
            self.freeze_dementia_head()            
            self.freeze_lm_grl()
            self.freeze_dementia_grl()
        elif STAGE == 6:                                                         # train 2 FSM
            print("Current stage: new 2")
            self.freeze_data2vec_audio()
            self.freeze_lm_head()
            self.freeze_dementia_head()
            self.freeze_criterion_similar()
            self.freeze_lm_grl()
            self.freeze_dementia_grl()
            
        # Initialize weights and apply final processing
        self.post_init()

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.data2vec_audio.feature_extractor._freeze_parameters()
    
    def freeze_data2vec_audio(self):
        self.data2vec_audio.eval()
        for param in self.data2vec_audio.parameters():
            param.requires_grad = False
    
    def freeze_criterion_similar(self):
        self.criterion_similar.eval()
        for param in self.criterion_similar.parameters():
            param.requires_grad = False
            
    def freeze_lm_fsm(self):
        self.lm_fsm.eval()
        for param in self.lm_fsm.parameters():
            param.requires_grad = False
            
    def freeze_dementia_fsm(self):
        self.dementia_fsm.eval()
        for param in self.dementia_fsm.parameters():
            param.requires_grad = False
            
    def freeze_lm_head(self):
        self.lm_head.eval()
        for param in self.lm_head.parameters():
            param.requires_grad = False

    def freeze_dementia_head(self):
        self.dementia_head.eval()
        for param in self.dementia_head.parameters():
            param.requires_grad = False
   
    def freeze_lm_grl(self):
        self.lm_grl.eval()
        for param in self.lm_grl.parameters():
            param.requires_grad = False
 
    def freeze_dementia_grl(self):
        self.dementia_grl.eval()
        for param in self.dementia_grl.parameters():
            param.requires_grad = False

    # @add_start_docstrings_to_model_forward(DATA2VEC_AUDIO_INPUTS_DOCSTRING)
    # @add_code_sample_docstrings(
    #     processor_class=_PROCESSOR_FOR_DOC,
    #     checkpoint=_CHECKPOINT_FOR_DOC,
    #     output_type=CausalLMOutput,
    #     config_class=_CONFIG_FOR_DOC,
    #     expected_output=_CTC_EXPECTED_OUTPUT,
    #     expected_loss=_CTC_EXPECTED_LOSS,
    # )
    def forward(
        self,
        input_values,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
        dementia_labels=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, target_length)`, *optional*):
            Labels for connectionist temporal classification. Note that `target_length` has to be smaller or equal to
            the sequence length of the output logits. Indices are selected in `[-100, 0, ..., config.vocab_size - 1]`.
            All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
            config.vocab_size - 1]`.
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.data2vec_audio(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)

        # hidden_states: data2vec_audio embedding
        # 製造mask
        m = nn.Sigmoid()
        dementia_score = m(self.dementia_fsm(hidden_states))            # score range from 0~1
        lm_score = m(self.lm_fsm(hidden_states))                        # score range from 0~1
        
        # if score >= thredhold, mask = 1
        dementia_mask = torch.where(dementia_score >= self.dementia_thres.to(dementia_score.device), torch.tensor(1.0).to(dementia_score.device), torch.tensor(0.0).to(dementia_score.device))  # if condition, 1. else, 0
        lm_mask = torch.where(lm_score >= self.lm_thres.to(lm_score.device), torch.tensor(1.0).to(lm_score.device), torch.tensor(0.0).to(lm_score.device))                                      # if condition, 1. else, 0
        lm_mask = lm_mask + 0 * self.lm_fsm(lm_mask)                    # to has grad?
        dementia_mask = dementia_mask + 0 * self.lm_fsm(lm_mask)        # to has grad?

        # 拿score vector 跟原本的hidden_states點乘
        #dementia_resored = dementia_score*hidden_states
        #lm_resored = lm_score*hidden_states

        ##################################
        # 拿mask跟原本的hidden_states點乘 #
        ##################################
        dementia_masked = dementia_mask*hidden_states
        lm_masked = lm_mask*hidden_states
        
        ##############
        # head(clf)
        ##############
        #dementia_logits = self.dementia_head(dementia_resored) #******************* torch.Size([2, 1327, 32]) (batchsize, timestep, feature_dimention)
        #logits = self.lm_head(lm_resored)
        dementia_logits = self.dementia_head(dementia_masked) #******************* torch.Size([2, 1327, 32]) (batchsize, timestep, feature_dimention)
        logits = self.lm_head(lm_masked)
        # del dementia_resored, lm_resored
        dementia_output_mean = torch.mean(dementia_logits,dim=1)

        ##############
        # grl(dis)
        ##############
        hidden_states_r = ReverseLayerF.apply(hidden_states, self.alpha)
        # get score from reversed embedding
        dementia_score_r = m(self.dementia_fsm(hidden_states_r))            # score range from 0~1
        lm_score_r = m(self.lm_fsm(hidden_states_r))                        # score range from 0~1
        # if score >= thredhold, mask = 1
        dementia_mask_r = torch.where(dementia_score_r >= self.dementia_thres.to(dementia_score_r.device), torch.tensor(1.0).to(dementia_score_r.device), torch.tensor(0.0).to(dementia_score_r.device)) # if condition, 1. else, 0
        lm_mask_r = torch.where(lm_score_r >= self.lm_thres.to(lm_score_r.device), torch.tensor(1.0).to(lm_score_r.device), torch.tensor(0.0).to(lm_score_r.device))                   # if condition, 1. else, 0
        
        del dementia_score_r, lm_score_r
        #####################################
        # 拿mask跟reversed hidden_states點乘 #
        #####################################
        dementia_masked_r = dementia_mask_r*hidden_states_r
        lm_masked_r = lm_mask_r*hidden_states_r
        
        del hidden_states_r, dementia_mask_r, lm_mask_r
        # grl(dis)
        dementia_logits_r = self.dementia_grl(lm_masked_r) #******************* torch.Size([2, 1327, 32]) (batchsize, timestep, feature_dimention)
        logits_r = self.lm_grl(dementia_masked_r)
        del dementia_masked_r, lm_masked_r
        
        dementia_output_mean_r = torch.mean(dementia_logits_r,dim=1)
        #del dementia_logits_r, dementia_logits
        del dementia_logits_r
        #*******************
        
        final_loss = None
        if labels is not None:

            if labels.max() >= self.config.vocab_size:
                raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")

            # retrieve loss input_lengths from attention_mask
            attention_mask = (
                attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
            )
            input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)

            # assuming that padded tokens are filled with -100
            # when not being attended to
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            # ctc_loss doesn't support fp16
            log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)
            log_probs_r = nn.functional.log_softmax(logits_r, dim=-1, dtype=torch.float32).transpose(0, 1)
            with torch.backends.cudnn.flags(enabled=False):
                loss = nn.functional.ctc_loss(
                    log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )
                # loss for lm_grl
                loss_r = nn.functional.ctc_loss(
                    log_probs_r,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )
                
                loss_fn = nn.CrossEntropyLoss()
                
                dementia_loss = loss_fn(dementia_output_mean, dementia_labels)        # loss for AD
                dementia_loss_rev = loss_fn(dementia_output_mean_r, dementia_labels)  # AD-GRL
                
                # FSM att loss
                # Scorematrix = append([dementia_mask,lm_mask]) # torch.Size([2, embedding_size])
                # Att_loss = Scorematrix*Scorematrix - Identity matrix
                #Att_loss = FSMatt_loss(lm_score, dementia_score)
                Att_loss = FSMatt_loss(lm_mask, dementia_mask)                        # use mask to compute attention loss
                # del lm_mask, dementia_mask
                # diversity loss: AM-Softmax
                lm_masked = hidden_states * lm_mask
                AD_masked = hidden_states * dementia_mask
                lm_masked = torch.reshape(lm_masked, (lm_masked.size()[0]*lm_masked.size()[1], lm_masked.size()[2])) # to size [batch_size*time-step, hidden_size]
                AD_masked = torch.reshape(AD_masked, (AD_masked.size()[0]*AD_masked.size()[1], AD_masked.size()[2])) # to size [batch_size*time-step, hidden_size]
                #print("lm_masked size: ", lm_masked.size())
                #print("AD_masked size: ", AD_masked.size())

                scores = torch.cat((lm_masked, AD_masked), dim=0) # size: [batch_size*time-step * 2, hidden_size]
                #print("score size: ", scores.size())
                am_labels = torch.cat((torch.zeros(len(lm_masked), dtype=torch.long), torch.ones(len(AD_masked), dtype=torch.long)), dim=0).to('cpu') # [batch_size*time-step * 2] w/ 1st half being 0s, and 2nd half being 1s
                #print("am_labels size: ", am_labels.size())
                #print(am_labels)

                # should feed x: [batch_size, hidden_size] & labels: [batch_size] simply use num, no need to one-hot
                similarity, _ = self.criterion_similar(scores, am_labels)
                score_loss = similarity # * args.w_score if args.w_score > 0. else torch.tensor(0.).to(device)

                if STAGE == 1:                                                  # train FSM
                    #print("Current stage: 1")
                    final_loss = loss + dementia_loss + score_loss + Att_loss
                elif STAGE == 2:                                                # train ASR
                    #print("Current stage: 2")
                    final_loss = loss + dementia_loss + score_loss + Att_loss
                elif STAGE == 3:                                                # train dementia GRL
                    #print("Current stage: 3")
                    final_loss = dementia_loss_rev
                elif STAGE == 4:
                    final_loss = loss_r
                elif STAGE == 5:
                    # train encoder
                    #final_loss = loss + dementia_loss + score_loss + Att_loss + dementia_loss_rev + loss_r
                    # train lm_FSM
                    final_loss = loss + dementia_loss_rev
                    # train dementia_FSM
                    #final_loss = dementia_loss + loss_r
                elif STAGE == 6:                                                # ASR loss, AD Loss (CE), diversity loss, and attention loss
                    final_loss = loss + dementia_loss + score_loss + Att_loss
                # ////
        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]

        return CausalLMOutput(
            loss=final_loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )
    
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
        file_object = open(LOG_DIR + log_file, 'a')
        # Append at the end of file
        file_object.write(json.dumps(output) + '\n')
        # Close the file
        file_object.close()

        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)

logger = logging.get_logger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('-lam', '--LAMBDA', type=float, default=0.5, help="Lambda for GRL")
parser.add_argument('-st', '--STAGE', type=int, default=1, help="Current training stage")
parser.add_argument('-GRL', '--GRL', action='store_true', default=False, help="True: GRL")
parser.add_argument('-model_in', '--model_in_path', type=str, default="/mnt/Internal/FedASR/weitung/HuggingFace/Pretrain/saves/data2vec-audio-large-960h_finetuned/final/", help="Where the model is saved")
parser.add_argument('-model_out', '--model_out_path', type=str, default="./saves/data2vec-audio-large-960h_linear_FSM_stage1", help="Where to save the model")
parser.add_argument('-log', '--log_path', type=str, default="data2vec-audio-large-960h_linear_FSM_stage1.txt", help="name for the txt file")
args = parser.parse_args()
LAMBDA = args.LAMBDA                    # lambda for GRL
REVERSE = args.GRL                      # not used in this version
STAGE = args.STAGE                      # stage 1: train AD classifier; stage 2: train toggling network
model_in_dir = args.model_in_path       # path to load the initial model
model_out_dir = args.model_out_path     # path to store the resulted model
log_file = args.log_path                # path to save log file

# 設定log file位置與名稱
LOG_DIR = './saves/log/'

# threshold for maskes
AD_THRES = 0.5
LM_THRES = 0.5

# load model from huggingface hub, here data2vec model
name = "facebook/data2vec-audio-large-960h"# + model_in_dir.split("/")[-3]
print("Current model: ", name)
from transformers import Data2VecAudioConfig
mask_time_prob = 0                                         # change config to avoid training stopping
config = Data2VecAudioConfig.from_pretrained(name, mask_time_prob=mask_time_prob)
model = Data2VecAudioForCTC.from_pretrained(model_in_dir, config=config)
model.config.ctc_zero_infinity = True                      # to avoid inf values

processor = Wav2Vec2Processor.from_pretrained(name)

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

# load train / test data
train_data = csv2dataset(csv_path = "/mnt/Internal/FedASR/Data/ADReSS-IS2020-data/mid_csv/train.csv")
#dev_data = csv2dataset(csv_path = "/mnt/Internal/FedASR/Data/ADReSS-IS2020-data/mid_csv/dev.csv")
test_data = csv2dataset(csv_path = "/mnt/Internal/FedASR/Data/ADReSS-IS2020-data/mid_csv/test.csv")

# map to desired form
train_data = train_data.map(prepare_dataset, num_proc=10)
#dev_data = dev_data.map(prepare_dataset, num_proc=10)
test_data = test_data.map(prepare_dataset, num_proc=10)

# skip to stage 6
if STAGE == 1:
    training_args = TrainingArguments(
        output_dir=model_out_dir,
        group_by_length=True,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        evaluation_strategy="steps",
        num_train_epochs=6,
        fp16=True,
        gradient_checkpointing=True, 
        save_steps=500,
        eval_steps=500,
        logging_steps=500,
        learning_rate=1e-5,
        weight_decay=0.005,
        warmup_steps=1000,
        save_total_limit=2,
        log_level='debug',
        logging_strategy="steps",
        #adafactor=True,            # default:false. Whether or not to use transformers.Adafactor optimizer instead of transformers.AdamW
        #fp16_full_eval=True,      # to save memory
        #max_grad_norm=0.5
    )
elif STAGE == 2:
    training_args = TrainingArguments(
        output_dir=model_out_dir,
        group_by_length=True,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        evaluation_strategy="steps",
        num_train_epochs=16,                 # finetune & GRL
        fp16=True,
        gradient_checkpointing=True, 
        save_steps=500,
        eval_steps=500,
        logging_steps=500,
        learning_rate=1e-5,
        weight_decay=0.005,
        warmup_steps=1000,
        save_total_limit=2,
        log_level='debug',
        logging_strategy="steps",
        adafactor=True,            # default:false. Whether or not to use transformers.Adafactor optimizer instead of transformers.AdamW
        #fp16_full_eval=True,      # to save memory
        #max_grad_norm=0.5
    )
elif STAGE == 6:
    training_args = TrainingArguments(
        output_dir=model_out_dir,
        group_by_length=True,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        evaluation_strategy="steps",
        num_train_epochs=30,                 # FSM alone
        fp16=True,
        gradient_checkpointing=True, 
        save_steps=500,
        eval_steps=500,
        logging_steps=500,
        learning_rate=1e-3, # 原本用1e-5
        weight_decay=0.005,
        warmup_steps=1000,
        save_total_limit=2,
        log_level='debug',
        logging_strategy="steps",
        #adafactor=True,            # default:false. Whether or not to use transformers.Adafactor optimizer instead of transformers.AdamW
        #fp16_full_eval=True,      # to save memory
        #max_grad_norm=0.5
    )
else:
    training_args = TrainingArguments(
        output_dir=model_out_dir,
        group_by_length=True,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        evaluation_strategy="steps",
        num_train_epochs=40,
        fp16=True,
        gradient_checkpointing=True, 
        save_steps=500,
        eval_steps=500,
        logging_steps=500,
        learning_rate=1e-5,
        weight_decay=0.005,
        warmup_steps=1000,
        save_total_limit=2,
        log_level='debug',
        logging_strategy="steps",
        adafactor=True,            # default:false. Whether or not to use transformers.Adafactor optimizer instead of transformers.AdamW
        #fp16_full_eval=True,      # to save memory
        #max_grad_norm=0.5,
    ) 

trainer = CustomTrainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_data,
    eval_dataset=test_data,
    tokenizer=processor.feature_extractor,
)

trainer.train()
# save resulted model as "final"
trainer.save_model(model_out_dir + "/final")
