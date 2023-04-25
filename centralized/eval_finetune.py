# only data2vec model is used!!!!!!!!!!
# from transformers.models.wav2vec2.configuration_wav2vec2 import Wav2Vec2Config
# from dataclasses import dataclass
# from typing import Optional, Tuple, Union

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
    # ReverseLayerF,
    # RecallLoss,
    # FSMatt_loss,
    # gumbel_softmax
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
from addict import Dict
import pickle
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

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)          # output字母的"機率"

        # Initialize weights and apply final processing
        self.post_init()

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.data2vec_audio.feature_extractor._freeze_parameters()

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

        logits = self.lm_head(hidden_states)
        
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

                final_loss = loss
        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
        # return info that we might need
        logits_all = {'ASR logits': logits,'hidden_states': hidden_states}

        return CausalLMOutput(
            loss=final_loss, logits=logits_all, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )


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
    dummy=None
    for i in range(len(subset_dataset)):
        RealLength=oupLens[i]  #只要有從logits取出來的都要還原
        df2 = pd.DataFrame({'path': subset_dataset[i]["path"],                                    # to know which sample
                # 'array': str(subset_dataset[i]["array"]),
                'text': subset_dataset[i]["text"],
                'dementia_labels': subset_dataset[i]["dementia_labels"],
                # 'input_values': str(subset_dataset[i]["input_values"]),               # input of the model
                # 'labels': str(subset_dataset[i]["labels"]),
                # 'ASR logits': str(logits["ASR logits"][i].tolist()),
                # 'hidden_states': str(logits["hidden_states"][i].tolist()), #原本的hidden state架構
                'hidden_states': logits["hidden_states"][i][:RealLength,:].cpu().numpy(),  #(time-step,node_dimension)
                'pred_str': pred_str[i]},
                index=[i])
        df = pd.concat([df, df2], ignore_index=True)
    
    return df

def map_to_result(batch, idx):
    with torch.no_grad():
        input_values = torch.tensor(batch["input_values"]).unsqueeze(0)            
        logits = model(input_values).logits                                     # includes ASR logits, dementia logits, hidden_states
        asr_lg = logits['ASR logits']
        #AD_lg = logits['dementia logits'][0]                                    # (1, time-step, 2) --> (time-step, 2)
    
    """
    pred_ad_tstep = torch.argmax(AD_lg, dim=-1)                                 # pred of each time-step
    pred_ad = pred_ad_tstep.sum() / pred_ad_tstep.size()[0]                     # average result
    if pred_ad > 0.5:                                                           # over half of the time pred AD
        batch["pred_AD"] = 1
    else:
        batch["pred_AD"] = 0
    """
    pred_ids = torch.argmax(asr_lg, dim=-1)
    batch["pred_str"] = processor.batch_decode(pred_ids)[0]
    batch["text"] = processor.decode(batch["labels"], group_tokens=False)
    
    # for toggle
    # for fine-tune model
    df = pd.DataFrame({'path': batch["path"],                                    # to know which sample
                # 'array': str(subset_dataset[i]["array"]),
                'text': batch["text"],
                'dementia_labels': batch["dementia_labels"],
                # 'input_values': str(subset_dataset[i]["input_values"]),               # input of the model
                # 'labels': str(subset_dataset[i]["labels"]),
                # 'ASR logits': str(logits["ASR logits"][i].tolist()),
                'hidden_states': str(logits["hidden_states"].tolist()),
                'pred_str': batch["pred_str"]},
                index=[idx])
    return df


parser = argparse.ArgumentParser()
parser.add_argument('-lam', '--LAMBDA', type=float, default=0.5, help="Lambda for GRL")
parser.add_argument('-st', '--STAGE', type=int, default=1, help="Current stage")
# parser.add_argument('-model', '--model_path', type=str, default="data2vec", help="Where the model is saved")
parser.add_argument('-thres', '--threshold', type=float, default=0.5, help="Threshold for AD & ASR")
parser.add_argument('-model_type', '--model_type', type=str, default="data2vec", help="Type of the model")
parser.add_argument('-RD', '--root_dir', default='/mnt/Internal/FedASR/Data/ADReSS-IS2020-data', help="Learning rate")
parser.add_argument('--AudioLoadFunc', default='librosa', help="用scipy function好像可以比較快")
parser.add_argument('--savepath', default='./saves/results/', help="用scipy function好像可以比較快")
parser.add_argument('--GPU_batchsize', type=str, default=None, help="如果cpu滿了就用GPU")

args = parser.parse_args()
LAMBDA = args.LAMBDA                    # lambda for GRL
STAGE = args.STAGE                      # stage 1: train AD classifier; stage 2: train toggling network
# model_dir = args.model_path             # path to load the model
model_type = args.model_type            # select different type of model (here only data2vec is ready to use)
savePath = args.savepath

# threshold for maskes
AD_THRES = args.threshold
LM_THRES = args.threshold

# load according to model type
# note that only data2vec is done for this version
if model_type == "wav2vec":
    name = "facebook/wav2vec2-base-960h" # + model_dir.split("/")[-3]
    print("Current model: ", name)
    csv_name=name.split('/')[-1]
    model = Wav2Vec2ForCTC.from_pretrained(name)
    processor = Wav2Vec2Processor.from_pretrained(name)
elif model_type == "data2vec":
    name = "facebook/data2vec-audio-large-960h" # + model_in_dir.split("/")[-3]
    print("Current model: ", name)
    csv_name=name.split('/')[-1]
    mask_time_prob = 0                                                                     # change config to avoid code from stopping
    config = Data2VecAudioConfig.from_pretrained(name, mask_time_prob=mask_time_prob)
    model = Data2VecAudioForCTC.from_pretrained(name, config=config)
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

df_test=Extract_Emb(test_data,GPU_batchsize=args.GPU_batchsize)
if not os.path.exists(savePath):
    os.makedirs(savePath)

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
