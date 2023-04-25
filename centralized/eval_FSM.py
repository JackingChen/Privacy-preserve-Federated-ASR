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
def prepare_dataset(batch):
    audio = batch["array"]

    # batched output is "un-batched" to ensure mapping is correct
    batch["input_values"] = processor(audio, sampling_rate=16000).input_values[0]
    
    with processor.as_target_processor():
        batch["labels"] = processor(batch["text"]).input_ids
        
    return batch


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
        df2 = pd.DataFrame({'path': subset_dataset[i]["path"],                                    # to know which sample
                # 'array': str(subset_dataset[i]["array"]),
                'text': subset_dataset[i]["text"],
                'dementia_labels': subset_dataset[i]["dementia_labels"],
                # 'input_values': str(subset_dataset[i]["input_values"]),               # input of the model
                # 'labels': str(subset_dataset[i]["labels"]),
                # 'ASR logits': str(logits["ASR logits"][i].tolist()),
                'hidden_states': str(logits["hidden_states"][i][:RealLength,:].tolist()),
                'pred_str': pred_str[i]},
                index=[i])
        df = pd.concat([df, df2], ignore_index=True)
    return df


##################################
# choose model type
##################################    
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-lam', '--LAMBDA', type=float, default=0.5, help="Lambda for GRL")
parser.add_argument('-st', '--STAGE', type=int, default=1, help="Current stage")
parser.add_argument('-model', '--model_path', type=str, default="/mnt/Internal/FedASR/weitung/HuggingFace/Pretrain/saves/data2vec-audio-large-960h_SingleToggle/final/", help="Where the model is saved")
parser.add_argument('-csv', '--csv_path', type=str, default="data2vec-audio-large-960h_FSM", help="name for the csv file")
parser.add_argument('-thres', '--threshold', type=float, default=0.5, help="Threshold for AD & ASR")
parser.add_argument('-model_type', '--model_type', type=str, default="data2vec", help="Type of the model")

parser.add_argument('-RD', '--root_dir', default='/mnt/Internal/FedASR/Data/ADReSS-IS2020-data', help="Learning rate")
parser.add_argument('--savepath', default='./EmbFeats/', help="用scipy function好像可以比較快")
parser.add_argument('--GPU_batchsize', type=str, default=None, help="如果cpu滿了就用GPU")

args = parser.parse_args()
LAMBDA = args.LAMBDA                    # lambda for GRL
STAGE = args.STAGE                      # stage 1: train AD classifier; stage 2: train toggling network
model_dir = args.model_path             # path to load the model
csv_name = args.csv_path                # path to store the result
model_type = args.model_type            # select different type of model (here only data2vec is ready to use)
savePath = args.savepath

# threshold for maskes
AD_THRES = args.threshold
LM_THRES = args.threshold

# from functions.OtherMdls_FSM import (
#     Wav2Vec2ForCTC,
#     Data2VecAudioForCTC,
#     HubertForCTC,
#     SEWDForCTC,
#     UniSpeechSatForCTC
# )
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
    
    # for FSM
    
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
                       #'dementia_resored': str(logits["dementia_resored"].tolist()),       # masked embedding
                       #'lm_resored': str(logits["lm_resored"].tolist()),                  # masked embedding
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
        if STAGE == 1:                                                           # train FSM
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

        # return info that we might need
        logits_all = {'ASR logits': logits, 'dementia logits': dementia_logits, 'hidden_states': hidden_states,
                    'lm_mask': lm_mask, 'dementia_mask': dementia_mask}

        return CausalLMOutput(
            loss=final_loss, logits=logits_all, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )

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
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
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

if not os.path.exists(savePath):
    os.makedirs(savePath)

# get emb.s, masks... 1 sample by 1 sample
# df = map_to_result(test_data[0], 0)
# for i in range(len(test_data) - 1):
#     df2 = map_to_result(test_data[i+1], i+1)
#     df = pd.concat([df, df2], ignore_index=True)
#     print("\r"+ str(i), end="")

# csv_path = "./saves/results/" + csv_name + ".csv"
df_test=Extract_Emb(test_data,GPU_batchsize=args.GPU_batchsize)
df_test.to_csv(f"{savePath}/{csv_name}.csv")
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
df_train.to_csv(f"{savePath}/{csv_name}_train.csv")
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
