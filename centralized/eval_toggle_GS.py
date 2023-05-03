# only data2vec model is used!!!!!!!!!!
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn

from transformers import (
    Wav2Vec2Processor, Wav2Vec2Model,
    Data2VecAudioModel, Data2VecAudioPreTrainedModel,
)
from transformers.modeling_outputs import CausalLMOutput
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
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
from datasets import (
    load_metric,
)
from transformers.models.data2vec.configuration_data2vec_audio import Data2VecAudioConfig
from utils import csv2dataset 
import pandas as pd
# from argparse import ArgumentParser
import argparse
import os
from tqdm import tqdm

from torch.nn.parallel import DataParallel
#2023/04/23 For using GPU
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Subset
from tqdm import tqdm
import pickle
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

def Extract_Emb(dataset, GPU_batchsize=16):
    if GPU_batchsize!=None:
        bs=int(GPU_batchsize)
        df=pd.DataFrame()
        for i in tqdm(range(0,len(dataset),bs)):
            idxs=list(range(i,min(i+bs,len(dataset))))
            subset_dataset = Subset(dataset, idxs)
            df_data=get_Embs(subset_dataset)
            df = pd.concat([df, df_data], ignore_index=True)
    else:
        # get emb.s, masks... 1 sample by 1 sample
        df = map_to_result(dataset[0], 0)
        for i in tqdm(range(len(dataset) - 1)):
            df2 = map_to_result(dataset[i+1], i+1)
            df = pd.concat([df, df2], ignore_index=True)
    return df

def get_Embs(subset_dataset):
    with torch.no_grad():
        # 將每個元素的 "input_values" 提取出來並組成一個列表
        input_sequences = [torch.tensor(sample['input_values']) for sample in subset_dataset]
        lengths = [len(sample['input_values']) for sample in subset_dataset]
        # 將列表中的序列進行填充
        padded_input_sequences = pad_sequence(input_sequences, batch_first=True)
        # # 打印填充後的序列張量的形狀
        # print(padded_input_sequences.shape)
        # input_values=padded_input_sequences.cuda()
        input_values=padded_input_sequences.to(device)
        logits=model(input_values).logits  
        asr_lg = logits['ASR logits']
        AD_lg = logits['dementia logits']                                    # (batchsize, time-step, 2) --> (time-step, 2)
        # 轉換length的度量從sample到output的timestep
        ratio=max(lengths)/asr_lg.shape[1]  # (batchsize, seqlength, logitsize)
        oupLens=[int(l/ratio) for l in lengths]
        # for l in lengths:
        #     oupLens.append(int(l/ratio))
        pred_ids = torch.argmax(asr_lg, dim=-1)
        # batch["pred_str"] = processor.batch_decode(pred_ids)[0]
        pred_str=processor.batch_decode(pred_ids)
        # batch["text"] = processor.decode(batch["labels"], group_tokens=False)
        # texts=[processor.decode(batch["labels"], group_tokens=False) for batch in subset_dataset]
        # text=processor.decode(batch["labels"], group_tokens=False)

    df = pd.DataFrame()
    for i in range(len(subset_dataset)):
        RealLength=oupLens[i]  #只要有從logits取出來的都要還原
        pred_ad_tstep = torch.argmax(AD_lg[i][:RealLength,:], dim=-1)# pred of each time-step
        pred_ad = pred_ad_tstep.sum() / pred_ad_tstep.size()[0]      # average result (batchsize, )
        AD_prediction = 1 if pred_ad > 0.5 else 0
        # subset_dataset[i]["pred_AD"] = 1 if pred_ad > 0.5 else 0

        df2 = pd.DataFrame({'path': subset_dataset[i]["path"],                                    # to know which sample
                # 'array': str(subset_dataset[i]["array"]),
                'text': subset_dataset[i]["text"],
                'dementia_labels': subset_dataset[i]["dementia_labels"],
                # 'input_values': str(subset_dataset[i]["input_values"]),               # input of the model
                # 'labels': str(subset_dataset[i]["labels"]),
                # 'ASR logits': str(logits["ASR logits"][i].tolist()),
                # 'hidden_states': str(logits["hidden_states"][i].tolist()), #原本的hidden state架構
                'dementia logits': [logits["dementia logits"][i][:RealLength,:].cpu().numpy()],
                'hidden_states': [logits["hidden_states"][i][:RealLength,:].cpu().numpy()],  #(time-step,node_dimension)
                'pred_AD': AD_prediction,                             # AD prediction
                'pred_str': pred_str[i],                           # predicted transcript
                'dementia_mask': [logits["dementia_mask"][i][:RealLength,:].cpu().numpy()],  # ASR-free mask for AD classification
                'lm_mask': [logits["lm_mask"][i][:RealLength,:].cpu().numpy()]
                },             # AD-free mask for ASR task
                index=[i])
        df = pd.concat([df, df2], ignore_index=True)
    return df


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
        self.lm_thres = torch.tensor(LM_THRES)
        print("lambda = ", self.alpha)
        print("lm_thres = ", self.lm_thres)

        # 加toggle network, lm_model
        #self.lm_fsm = nn.Linear(config.hidden_size, config.hidden_size)          # 找出對lm重要的feat
        self.arbitrator = nn.Linear(config.hidden_size, config.hidden_size*4)    # 2條保護AD資訊（one-hot後用其中一條），2條保護ASR資訊（one-hot後用其中一條）
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)          # output字母的"機率"
        
        # 加dementia model
        self.dementia_head = nn.Linear(config.hidden_size, 2)                    # 辨識AD
        
        # define similarity loss: AM-Softmax, aka div loss
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
        AD_score = torch.stack((all_score[:, :, self.config.hidden_size*2:self.config.hidden_size*3] , all_score[:, :, self.config.hidden_size*3:]), -1) # second part for AD, size = [batch_size, time-step, hidden_state, 2]

        # toggle ratio
        if TOGGLE_RATIO != 0:                                                           # if toggle ratio is set
            # lm_score
            y0 = lm_score[:, :, :, 0]                                                   # target vector
            y1 = lm_score[:, :, :, 1]                                                   # another vector
            lm_score[:, :, :, 0] = (y1 - y0) * TOGGLE_RATIO + y0                        # replace target vector
            # AD_score
            y0 = AD_score[:, :, :, 0]                                                   # target vector
            y1 = AD_score[:, :, :, 1]                                                   # another vector
            AD_score[:, :, :, 0] = (y1 - y0) * TOGGLE_RATIO + y0                        # replace target vector  
        # check scores
        # lm_score
        y0 = lm_score[:, :, :, 0]                                                   # target vector
        y1 = lm_score[:, :, :, 1]                                                   # another vector
        print("ASR y0 - y1 = ", (y0-y1)[0, 0, :])
        # AD_score
        y0 = AD_score[:, :, :, 0]                                                   # target vector
        y1 = AD_score[:, :, :, 1]                                                   # another vector
        print("AD y0 - y1 = ", (y0-y1)[0, 0, :])  
        # go through GS to form mask
        #lm_mask = torch.nn.functional.gumbel_softmax(lm_score, hard=True, dim=-1)[:, :, :, 0] # back to [batch_size, time-step, hidden_state]
        lm_mask = gumbel_softmax(lm_score, tau=GS_TAU, hard=True, dim=-1)[:, :, :, 0]
        #AD_mask = torch.nn.functional.gumbel_softmax(AD_score, hard=True, dim=-1)[:, :, :, 0] # back to [batch_size, time-step, hidden_state]
        AD_mask = gumbel_softmax(AD_score, tau=GS_TAU, hard=True, dim=-1)[:, :, :, 0]

        ##################################
        # 拿mask跟原本的hidden_states點乘 #
        ##################################
        """
        lm_masked = lm_mask*hidden_states
        """
        lm_masked = lm_mask*hidden_states
        AD_masked = AD_mask*hidden_states

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

        logits_r = self.lm_head(AD_masked)                                                  # for ASR GRL
        dementia_logits = self.dementia_head(AD_masked)                                     # for AD classifier
        dementia_output_mean = torch.mean(dementia_logits,dim=1)
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
            log_probs_r = nn.functional.log_softmax(logits_r, dim=-1, dtype=torch.float32).transpose(0, 1) # logit轉prob
            log_probs_r = ReverseLayerF.apply(log_probs_r, self.alpha) # ASR-GRL
            
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
                # ASR GRL
                loss_r = nn.functional.ctc_loss(
                    log_probs_r,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )
                
                if AD_loss == "cel":
                    print("loss: cel")
                    loss_fn = nn.CrossEntropyLoss()
                    
                    dementia_loss = loss_fn(dementia_output_mean, dementia_labels)                      # AD classifier
                    dementia_loss_unmask = loss_fn(dementia_output_mean_unmask, dementia_labels)        # unmask
                    dementia_loss_rev = loss_fn(dementia_output_mean_r, dementia_labels)                # reverse
                elif AD_loss == "recall":                 
                    #print("loss: recall")
                    loss_fn = RecallLoss(weight=W_LOSS)                                                 # W_LOSS=[w_HC, w_AD]
                    #loss = criterion(y_predict, y_target)
                    # predict: [N, C, *]    ; target: [N, *]
                    dementia_loss = loss_fn(dementia_output_mean, dementia_labels, AD_loss)                      # AD classifier: [batch_size, 2], [batch_size,]
                    dementia_loss_unmask = loss_fn(dementia_output_mean_unmask, dementia_labels, AD_loss)        # unmask: [batch_size, 2], [batch_size,]
                    #print("dementia_output_mean_unmask: ", dementia_output_mean_unmask)
                    #print("dementia_labels: ", dementia_labels)
                    #print("dementia_loss: ", dementia_loss_unmask)
                    
                    dementia_loss_rev = loss_fn(dementia_output_mean_r, dementia_labels, AD_loss)                # reverse: [batch_size, 2], [batch_size,]

                elif AD_loss == "prec":                 
                    #print("loss: precision")
                    loss_fn = RecallLoss(weight=[0.1, 0.9])                                                      # emphasize on AD PAR
                    
                    dementia_loss = loss_fn(dementia_output_mean, dementia_labels, AD_loss)                      # AD classifier
                    dementia_loss_unmask = loss_fn(dementia_output_mean_unmask, dementia_labels, AD_loss)        # unmask
                    dementia_loss_rev = loss_fn(dementia_output_mean_r, dementia_labels, AD_loss)                # reverse
                elif AD_loss == "f1":
                    #print("loss: f1")
                    loss_fn = RecallLoss(weight=[0.5, 0.5])
                    
                    dementia_loss = loss_fn(dementia_output_mean, dementia_labels, AD_loss)                      # AD classifier
                    dementia_loss_unmask = loss_fn(dementia_output_mean_unmask, dementia_labels, AD_loss)        # unmask
                    dementia_loss_rev = loss_fn(dementia_output_mean_r, dementia_labels, AD_loss)                # reverse         
                elif AD_loss == "prec_ori":     
                    #print("loss: prec_ori")
                    loss_fn = RecallLoss(weight=[0.5, 0.5])
                    
                    dementia_loss = loss_fn(dementia_output_mean, dementia_labels, AD_loss)                      # AD classifier
                    dementia_loss_unmask = loss_fn(dementia_output_mean_unmask, dementia_labels, AD_loss)        # unmask
                    dementia_loss_rev = loss_fn(dementia_output_mean_r, dementia_labels, AD_loss)                # reverse     
                elif AD_loss == "recall_ori":     
                    #print("loss: recall_ori")
                    loss_fn = RecallLoss(weight=[0.5, 0.5])
                    
                    dementia_loss = loss_fn(dementia_output_mean, dementia_labels, AD_loss)                      # AD classifier
                    dementia_loss_unmask = loss_fn(dementia_output_mean_unmask, dementia_labels, AD_loss)        # unmask
                    dementia_loss_rev = loss_fn(dementia_output_mean_r, dementia_labels, AD_loss)                # reverse     
                # att loss
                Att_loss = FSMatt_loss(lm_mask, AD_mask)                                                        # not used in this version
                # diversity loss: AM-Softmax
                #scores = torch.cat((hidden_states * lm_mask, hidden_states * AD_mask), dim=0)
                #am_labels = torch.cat((torch.zeros(len(hidden_states), dtype=torch.long), torch.ones(len(hidden_states), dtype=torch.long)), dim=0).to('cpu')
                #del hidden_states
                lm_masked = hidden_states * lm_mask
                AD_masked = hidden_states * AD_mask
                lm_masked = torch.reshape(lm_masked, (lm_masked.size()[0]*lm_masked.size()[1], lm_masked.size()[2])) # batch_size*time-step, hidden_size
                AD_masked = torch.reshape(AD_masked, (AD_masked.size()[0]*AD_masked.size()[1], AD_masked.size()[2])) # batch_size*time-step, hidden_size
                #print("lm_masked size: ", lm_masked.size())
                #print("AD_masked size: ", AD_masked.size())

                scores = torch.cat((lm_masked, AD_masked), dim=0) # batch_size*time-step * 2, hidden_size
                #print("score size: ", scores.size())
                am_labels = torch.cat((torch.zeros(len(lm_masked), dtype=torch.long), torch.ones(len(AD_masked), dtype=torch.long)), dim=0).to('cpu') # batch_size*time-step * 2
                #print("am_labels size: ", am_labels.size())
                #print(am_labels)

                similarity, _ = self.criterion_similar(scores, am_labels)
                score_loss = similarity # * args.w_score if args.w_score > 0. else torch.tensor(0.).to(device)
                #print("========================")
                #print(AD_mask, lm_mask)
                #print(loss, dementia_loss_rev, loss_r, dementia_loss, Att_loss, score_loss)

                if STAGE == 1:                                                  # train AD classifier
                    #print("Current stage: 1")
                    final_loss = dementia_loss_unmask
                elif STAGE == 2:                                                # train toggle network
                    #print("Current stage: 2")
                    final_loss = loss + dementia_loss_rev + loss_r + dementia_loss + score_loss #+ Att_loss #+ score_loss
                    #print(loss, dementia_loss_rev, loss_r, dementia_loss, l2_lambda * l2_norm)
                    #final_loss = loss + dementia_loss_rev + loss_r + dementia_loss + l2_lambda * l2_norm
                    #final_loss = l2_lambda * l2_norm
                elif STAGE == 3:
                    final_loss = loss + dementia_loss_rev + loss_r + dementia_loss #+ Att_loss + score_loss
                # ////
        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]

        # return info that we might need
        logits_all = {'ASR logits': logits, 'dementia logits': dementia_logits, 'hidden_states': hidden_states,
                    'lm_mask': lm_mask, "dementia_mask": AD_mask}

        return CausalLMOutput(
            loss=final_loss, logits=logits_all, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )

def map_to_result(batch, idx):
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
                    #    'array': str(batch["array"]),
                       'text': batch["text"],                                   # ground truth transcript
                       'dementia_labels': batch["dementia_labels"],
                    #    'input_values': str(batch["input_values"]),              # input of the model
                    #    'labels': str(batch["labels"]),
                    #    'ASR logits': str(logits["ASR logits"].tolist()),
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

##################################
# choose model type
##################################    
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-lam', '--LAMBDA', type=float, default=0.5, help="Lambda for GRL")
parser.add_argument('-st', '--STAGE', type=int, default=1, help="Current stage")
parser.add_argument('-model', '--model_path', type=str, default="/mnt/Internal/FedASR/weitung/HuggingFace/Pretrain/saves/data2vec-audio-large-960h_new1_recall_82/final/", help="Where the model is saved")
parser.add_argument('-csv', '--csv_path', type=str, default="data2vec-audio-large-960h_new1_recall_82", help="name for the csv file")
parser.add_argument('-thres', '--threshold', type=float, default=0.5, help="Threshold for AD & ASR")
parser.add_argument('-model_type', '--model_type', type=str, default="data2vec", help="Type of the model")
# 2023/01/08: loss type
parser.add_argument('-ad_loss', '--AD_loss', type=str, default="cel", help="loss to use for AD classifier")
# 2023/02/13: toggling ratio
parser.add_argument('-toggle_rt', '--TOGGLE_RATIO', type=float, default=0, help="To toggle more or less")
# 2023/02/15: TOGGLE_RATIO, loss weight
parser.add_argument('-gs_tau', '--GS_TAU', type=float, default=1, help="Tau for gumbel_softmax")
parser.add_argument('-w_loss', '--W_LOSS', type=float, default=None, nargs='+', help="weight for HC and AD")

parser.add_argument('-RD', '--root_dir', default='/mnt/Internal/FedASR/Data/ADReSS-IS2020-data', help="Learning rate")
parser.add_argument('--savepath', default='./saves/results/', help="用scipy function好像可以比較快")
parser.add_argument('--GPU_batchsize', type=str, default=None, help="如果cpu滿了就用GPU")


args = parser.parse_args()
LAMBDA = args.LAMBDA                    # lambda for GRL
STAGE = args.STAGE                      # stage 1: train AD classifier; stage 2: train toggling network
model_dir = args.model_path             # path to load the model
csv_name = args.csv_path                # path to store the result
model_type = args.model_type            # select different type of model (here only data2vec is ready to use)
AD_loss = args.AD_loss                  # type of AD loss: cel, f1, recall, prec, (recall_ori, prec_ori)
TOGGLE_RATIO = args.TOGGLE_RATIO        # for exp. to change toggle rate
GS_TAU = args.GS_TAU                    # temperature for gumbel_softmax
savePath = args.savepath
if args.W_LOSS == None:                 # weight for HC and AD
    W_LOSS = [0.1, 0.9]                 # default weight for HC and AD
else:
    W_LOSS = args.W_LOSS
print("weight for loss: ", W_LOSS)

# threshold for maskes
AD_THRES = args.threshold
LM_THRES = args.threshold

# load according to model type
# note that only data2vec is done for this version
if model_type == "wav2vec":
    model = Wav2Vec2ForCTC.from_pretrained(model_dir)
    name = "facebook/wav2vec2-base-960h" # + model_dir.split("/")[-3]
    print("Current model: ", name)
    processor = Wav2Vec2Processor.from_pretrained(name)
elif model_type == "data2vec":
    name = "facebook/data2vec-audio-large-960h" # + model_in_dir.split("/")[-3]
    print("Current model: ", name)
    mask_time_prob = 0                                                                     # change config to avoid code from stopping
    config = Data2VecAudioConfig.from_pretrained(name, mask_time_prob=mask_time_prob)
    model = Data2VecAudioForCTC.from_pretrained(model_dir, config=config)
    processor = Wav2Vec2Processor.from_pretrained(name)
elif model_type == "hubert":
    name = "facebook/hubert-xlarge-ls960-ft" # + model_in_dir.split("/")[-3]
    print("Current model: ", name)
    mask_time_prob = 0                                                                     # change config
    config = HubertConfig.from_pretrained(name, mask_time_prob=mask_time_prob)
    model = HubertForCTC.from_pretrained(model_dir, config=config)
    processor = Wav2Vec2Processor.from_pretrained(name)
elif model_type == "sewd":
    name = "asapp/sew-d-mid-400k-ft-ls100h" #+ model_in_dir.split("/")[-3]
    print("Current model: ", name)
    mask_time_prob = 0                                                                     # change config
    config = SEWDConfig.from_pretrained(name, mask_time_prob=mask_time_prob)
    model = SEWDForCTC.from_pretrained(model_dir, config=config)
    processor = Wav2Vec2Processor.from_pretrained(name)
elif model_type == "unispeech":
    name = "microsoft/unispeech-sat-base-100h-libri-ft" # + model_in_dir.split("/")[-3]
    print("Current model: ", name)
    mask_time_prob = 0                                                                     # change config
    config = UniSpeechSatConfig.from_pretrained(name, mask_time_prob=mask_time_prob)
    model = UniSpeechSatForCTC.from_pretrained(model_dir, config=config)
    processor = Wav2Vec2Processor.from_pretrained(name)   


data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
if args.GPU_batchsize != None:
    # ======================
    # model = model.cuda()
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
    if torch.cuda.device_count() > 1:
        model = DataParallel(model)

    # 將模型移動到GPU上
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # ======================
# store result of test data
test_data = csv2dataset(audio_path = '{}/clips/'.format(args.root_dir),
                        csv_path = "{}/mid_csv/test.csv".format(args.root_dir))
test_data = test_data.map(prepare_dataset, num_proc=10)

# get emb.s, masks... 1 sample by 1 sample
# df = map_to_result(test_data[0], 0)
# for i in range(len(test_data) - 1):
#     df2 = map_to_result(test_data[i+1], i+1)
#     df = pd.concat([df, df2], ignore_index=True)
#     print("\r"+ str(i), end="")

# csv_path = "./saves/results/" + csv_name + ".csv"
df_test=Extract_Emb(test_data,GPU_batchsize=args.GPU_batchsize)
with open(f"{savePath}/{csv_name}.pkl", "wb") as f:
    pickle.dump(df_test, f)
print("Testing data Done")


# store result of train data
train_data = csv2dataset(audio_path = '{}/clips/'.format(args.root_dir),
                         csv_path = "{}/mid_csv/train.csv".format(args.root_dir)) #!!! librosa在load的時候非常慢，大約7分47秒讀完1869個file
train_data = train_data.map(prepare_dataset, num_proc=10)

# get emb.s, masks... 1 sample by 1 sample
# df = map_to_result(train_data[0], 0)
# for i in range(len(train_data) - 1):
#     df2 = map_to_result(train_data[i+1], i+1)
#     df = pd.concat([df, df2], ignore_index=True)
#     print("\r"+ str(i), end="")

# csv_path = "./saves/results/" + csv_name + "_train.csv"
df_train=Extract_Emb(train_data,GPU_batchsize=args.GPU_batchsize)
with open(f"{savePath}/{csv_name}_train.pkl", "wb") as f:
    pickle.dump(df_train, f)
    
print("Training data Done")

# store result of dev data
"""
dev_data = csv2dataset(path = "/mnt/Internal/FedASR/Data/ADReSS-IS2020-data/mid_csv/dev.csv")
dev_data = dev_data.map(prepare_dataset, num_proc=10)

df = map_to_result(dev_data[0], 0)
for i in range(len(dev_data) - 1):
    df2 = map_to_result(dev_data[i+1], i+1)
    df = pd.concat([df, df2], ignore_index=True)
    print("\r"+ str(i), end="")

csv_path = "./saves/results/" + csv_name + "_dev.csv"
df.to_csv(csv_path)
"""
print(csv_name + " All Done")
