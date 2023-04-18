import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from argparse import ArgumentParser
from transformers import (
    Wav2Vec2Processor,
    Data2VecAudioModel,
    PreTrainedModel,
    # CausalLMOutput,
    TrainingArguments,
    Trainer,
    logging,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput
from transformers.training_args import TrainingArguments
from transformers.utils import logging,add_code_sample_docstrings, add_start_docstrings_to_model_forward
from transformers.models.data2vec.configuration_data2vec_audio import Data2VecAudioConfig
from transformers.models.data2vec.modeling_data2vec_audio import (
    Data2VecAudioEncoder,
    Data2VecAudioFeatureEncoder,
    Data2VecAudioFeatureProjection,
    Data2VecAudioPositionalConvLayer,
)
from datasets import (
    Dataset,
    load_from_disk,
    load_metric,
)
from Models import (
    Data2VecAudioPreTrainedModel,
    DataCollatorCTCWithPadding,
    Data2VecAudioForCTC,
    ReverseLayerF,
    RecallLoss,
    FSMatt_loss,
    gumbel_softmax
)

from functions.models import (
    AngularPenaltySMLoss,
)
from utils import csv2dataset

from jiwer import wer
import argparse
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
        print("lambda = ", self.alpha)

        # 加toggle network, lm_model
        self.arbitrator = nn.Linear(config.hidden_size, config.hidden_size*2)    # 2條保護AD資訊（one-hot後用其中一條）
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)          # output字母的"機率"
        
        # 加dementia model
        self.dementia_head = nn.Linear(config.hidden_size, 2)                    # 辨識AD
        
        # define similarity loss: AM-Softmax, aka div loss (not used here)
        self.criterion_similar = AngularPenaltySMLoss(in_features=config.hidden_size, out_features=2, loss_type='cosface').to('cpu')
        
        # freeze feature_extractor    
        self.freeze_feature_encoder()

        if STAGE == 1:                                                  # freeze all, train AD classifier alone
            print("Current stage: 1")
            self.freeze_data2vec_audio()
            self.freeze_lm_head()
            #self.freeze_lm_fsm()
            self.freeze_arbitrator()
            self.freeze_criterion_similar()
        elif STAGE == 2:                                                # freeze all, train toggle network alone
            print("Current stage: 2")
            self.freeze_data2vec_audio()
            self.freeze_lm_head()
            self.freeze_dementia_head()
            self.freeze_criterion_similar()
        elif STAGE == 3:                                                # freeze all, train FSM + classifiers
            print("Current stage: 3")
            self.freeze_data2vec_audio()
            self.freeze_criterion_similar()           

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
    """        
    def freeze_lm_fsm(self):
        self.lm_fsm.eval()
        for param in self.lm_fsm.parameters():
            param.requires_grad = False
    """        
    def freeze_lm_head(self):
        self.lm_head.eval()
        for param in self.lm_head.parameters():
            param.requires_grad = False

    def freeze_dementia_head(self):
        self.dementia_head.eval()
        for param in self.dementia_head.parameters():
            param.requires_grad = False
    
    def freeze_arbitrator(self):
        self.arbitrator.eval()
        for param in self.arbitrator.parameters():
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

        # 沒過FSM，用來單獨train AD classifier
        dementia_logits_unmask = self.dementia_head(hidden_states) # for stage 1 training

        # hidden_states: data2vec_audio embedding
        ###################
        # 製造mask
        ###################
        """
        m = nn.Sigmoid()
        lm_score = m(self.lm_fsm(hidden_states))             # score range from 0~1
        lm_mask = torch.where(lm_score >= self.lm_thres.to(lm_score.device), torch.tensor(1.0).to(lm_score.device), torch.tensor(0.0).to(lm_score.device))                   # if condition, 1. else, 0
        lm_mask = lm_mask + 0 * self.lm_fsm(lm_mask) # to has grad?
        """
        #m = nn.Sigmoid()
        #all_score = m(self.arbitrator(hidden_states))             # score range from 0~1
        all_score = self.arbitrator(hidden_states)
        """
        all_mask = torch.where(all_score >= self.lm_thres.to(all_score.device), torch.tensor(1.0).to(all_score.device), torch.tensor(0.0).to(all_score.device))                   # if condition, 1. else, 0
        all_mask = all_mask + 0 * self.arbitrator(hidden_states) # to have grad?  
        """
        # use Gunbel softmax
        #print(all_score)
        lm_score = torch.stack((all_score[:, :, :self.config.hidden_size] , all_score[:, :, self.config.hidden_size:self.config.hidden_size*2]), -1)     # first part for lm, size = [batch_size, time-step, hidden_state, 2]
       
        #lm_mask = torch.nn.functional.gumbel_softmax(lm_score, hard=True, dim=-1)[:, :, :, 0] # back to [batch_size, time-step, hidden_state]
        lm_mask = gumbel_softmax(lm_score, hard=True, dim=-1)[:, :, :, 0]

        ##################################
        # 拿mask跟原本的hidden_states點乘 #
        ##################################
        """
        lm_masked = lm_mask*hidden_states
        """
        lm_masked = lm_mask*hidden_states

        ##############
        # head(clf)
        ##############
        """
        logits = self.lm_head(lm_masked)
        dementia_logits = self.dementia_head(lm_masked) # masked hidden state 過AD classifier
        dementia_output_mean_2r = torch.mean(dementia_logits,dim=1)
        dementia_output_mean_r = ReverseLayerF.apply(dementia_output_mean_2r, self.alpha)
        dementia_output_mean = torch.mean(dementia_logits_unmask,dim=1)
        """
        logits = self.lm_head(lm_masked)                                                    # ASR loss
        dementia_logits = self.dementia_head(lm_masked)                                     # for AD GRL
        
        dementia_output_mean_2r = torch.mean(dementia_logits,dim=1)
        dementia_output_mean_r = ReverseLayerF.apply(dementia_output_mean_2r, self.alpha)   # for AD GRL
        dementia_output_mean_unmask = torch.mean(dementia_logits_unmask,dim=1)              # unmask

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
                #  /////
                # gradient reversal layers(GRL)
                if AD_loss == "cel":
                    #print("loss: cel")
                    loss_fn = nn.CrossEntropyLoss()
                    
                    dementia_loss_unmask = loss_fn(dementia_output_mean_unmask, dementia_labels)        # unmask
                    dementia_loss_rev = loss_fn(dementia_output_mean_r, dementia_labels)                # reverse
                elif AD_loss == "recall":                 
                    loss_fn = RecallLoss(weight=[0.1, 0.9])                                             # true label = 1 (AD) 的預測"機率"越大越好
                    #loss = criterion(y_predict, y_target)
                    # predict: [N, C, *]    ; target: [N, *]
                    dementia_loss_unmask = loss_fn(dementia_output_mean_unmask, dementia_labels)        # unmask: [batch_size, 2], [batch_size,]
                    
                    dementia_loss_rev = loss_fn(dementia_output_mean_r, dementia_labels)                # reverse: [batch_size, 2], [batch_size,]

                elif AD_loss == "prec":                 
                    print("NO prec loss yet!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    #loss_fn = nn.CrossEntropyLoss(weight=[1, 0])
                    
                    #dementia_loss = loss_fn(dementia_output_mean, dementia_labels)                      # AD classifier
                    #dementia_loss_unmask = loss_fn(dementia_output_mean_unmask, dementia_labels)        # unmask
                    #dementia_loss_rev = loss_fn(dementia_output_mean_r, dementia_labels)                # reverse
                # att loss
                #Att_loss = FSMatt_loss(lm_mask, AD_mask)
                # diversity loss: AM-Softmax
                #scores = torch.cat((hidden_states * lm_mask, hidden_states * AD_mask), dim=0)
                #am_labels = torch.cat((torch.zeros(len(hidden_states), dtype=torch.long), torch.ones(len(hidden_states), dtype=torch.long)), dim=0).to('cpu')
                #print("scores size: ", scores.size())
                #print("labels size: ", am_labels.size())
                #del hidden_states
                #lm_masked = hidden_states * lm_mask
                #AD_masked = hidden_states * AD_mask
                #lm_masked = torch.reshape(lm_masked, (lm_masked.size()[0]*lm_masked.size()[1], lm_masked.size()[2])) # batch_size*time-step, hidden_size
                #AD_masked = torch.reshape(AD_masked, (AD_masked.size()[0]*AD_masked.size()[1], AD_masked.size()[2])) # batch_size*time-step, hidden_size
                #print("lm_masked size: ", lm_masked.size())
                #print("AD_masked size: ", AD_masked.size())

                #scores = torch.cat((lm_masked, AD_masked), dim=0) # batch_size*time-step * 2, hidden_size
                #print("score size: ", scores.size())
                #am_labels = torch.cat((torch.zeros(len(lm_masked), dtype=torch.long), torch.ones(len(AD_masked), dtype=torch.long)), dim=0).to('cpu') # batch_size*time-step * 2
                #print("am_labels size: ", am_labels.size())
                #print(am_labels)

                # should feed x: [batch_size, hidden_size] & labels: [batch_size] simply use num, no need to one-hot
                #similarity, _ = self.criterion_similar(scores, am_labels)
                #score_loss = similarity # * args.w_score if args.w_score > 0. else torch.tensor(0.).to(device)
                
                #print("========================")
                #print(AD_mask, lm_mask)
                #print(loss, dementia_loss_rev, loss_r, dementia_loss, Att_loss, score_loss)

                if STAGE == 1:                                                  # train AD classifier
                    final_loss = dementia_loss_unmask
                elif STAGE == 2:                                                # train FSM
                    final_loss = loss + dementia_loss_rev  #+ Att_loss #+ score_loss + loss_r + dementia_loss + score_loss
                    #final_loss = loss + dementia_loss_rev + loss_r + dementia_loss + l2_lambda * l2_norm
                    #final_loss = l2_lambda * l2_norm
                elif STAGE == 3:
                    final_loss = loss + dementia_loss_rev #+ loss_r + dementia_loss #+ Att_loss + score_loss
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

wer_metric = load_metric("wer")
def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    label_ids_asr , label_ids_AD=pred.label_ids

    label_ids_asr[label_ids_asr == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(label_ids_asr, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}        

def prepare_dataset(batch):
    audio = batch["array"]

    # batched output is "un-batched" to ensure mapping is correct
    batch["input_values"] = processor(audio, sampling_rate=16000).input_values[0]
    
    with processor.as_target_processor():
        batch["labels"] = processor(batch["text"]).input_ids
        
    return batch

parser = argparse.ArgumentParser()
parser.add_argument('-lam', '--LAMBDA', type=float, default=0.5, help="Lambda for GRL")
parser.add_argument('-st', '--STAGE', type=int, default=1, help="Current training stage")
parser.add_argument('-GRL', '--GRL', action='store_true', default=False, help="True: GRL")
parser.add_argument('-model_in', '--model_in_path', type=str, default="/mnt/Internal/FedASR/weitung/HuggingFace/Pretrain/saves/data2vec-audio-large-960h_finetuned/final/", help="Where the model is saved")
parser.add_argument('-model_out', '--model_out_path', type=str, default="./saves/data2vec-audio-large-960h_linear_SingleToggle", help="Where to save the model")
parser.add_argument('-log', '--log_path', type=str, default="data2vec-audio-large-960h_linear_SingleToggle.txt", help="name for the txt file")
# 2023/01/08: loss type
parser.add_argument('-ad_loss', '--AD_loss', type=str, default="cel", help="loss to use for AD classifier")

args = parser.parse_args()
LAMBDA = args.LAMBDA                    # lambda for GRL
REVERSE = args.GRL                      # not used in this version
STAGE = args.STAGE                      # stage 1: train AD classifier; stage 2: train toggling network
model_in_dir = args.model_in_path       # path to load the initial model
model_out_dir = args.model_out_path     # path to store the resulted model
log_file = args.log_path                # path to save log file
AD_loss = args.AD_loss                  # type of AD loss: cel, f1, recall, prec, (recall_ori, prec_ori)

# 設定log file位置與名稱
LOG_DIR = './saves/log/'

# threshold for maskes, not used here
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
#dev_data = csv2dataset(path = "/mnt/Internal/FedASR/Data/ADReSS-IS2020-data/mid_csv/dev.csv")
test_data = csv2dataset(csv_path = "/mnt/Internal/FedASR/Data/ADReSS-IS2020-data/mid_csv/test.csv")

# map to desired form
train_data = train_data.map(prepare_dataset, num_proc=10)
#dev_data = dev_data.map(prepare_dataset, num_proc=10)
test_data = test_data.map(prepare_dataset, num_proc=10)

if STAGE == 1:                                          # config to train AD classifier
    training_args = TrainingArguments(
        output_dir=model_out_dir,
        group_by_length=True,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        evaluation_strategy="steps",
        num_train_epochs=30,
        fp16=True,
        gradient_checkpointing=True, 
        save_steps=500,
        eval_steps=500,
        logging_steps=500,
        learning_rate=1e-4,
        weight_decay=0.005,
        warmup_steps=1000,
        save_total_limit=2,
        # log_level='debug',
        logging_strategy="steps",
        #adafactor=True,            # default:false. Whether or not to use transformers.Adafactor optimizer instead of transformers.AdamW
        #fp16_full_eval=True,      # to save memory
        #max_grad_norm=0.5
    )
elif STAGE == 2:                                        # config to train toggle network
    training_args = TrainingArguments(
        output_dir=model_out_dir,
        group_by_length=True,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        evaluation_strategy="steps",
        num_train_epochs=30,
        fp16=True,
        gradient_checkpointing=True, 
        save_steps=500,
        eval_steps=500,
        logging_steps=500,
        learning_rate=1e-3,
        weight_decay=0.005,
        warmup_steps=1000,
        save_total_limit=2,
        # log_level='debug',
        logging_strategy="steps",
        #adafactor=True,            # default:false. Whether or not to use transformers.Adafactor optimizer instead of transformers.AdamW
        #fp16_full_eval=True,      # to save memory
        #max_grad_norm=0.5
    )
elif STAGE == 3:                                    # not used in this version
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
        learning_rate=1e-5,
        weight_decay=0.005,
        warmup_steps=1000,
        save_total_limit=2,
        # log_level='debug',
        logging_strategy="steps",
        #adafactor=True,            # default:false. Whether or not to use transformers.Adafactor optimizer instead of transformers.AdamW
        #fp16_full_eval=True,      # to save memory
        #max_grad_norm=0.5
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
trainer.save_model(model_out_dir + "/final")
