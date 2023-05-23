import torch
import pandas as pd
from transformers import Data2VecAudioConfig, Wav2Vec2Processor
from torch.utils.data import Subset
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from datasets import Dataset
from tensorboardX import SummaryWriter
from Data2VecAudioForCTCMultitask_model import Data2VecAudioForCTCMultitask
from update import compute_metrics, get_model_weight
from transformers.training_args import TrainingArguments
from transformers import Trainer
import ast
from models import Data2VecAudioForCTC, DataCollatorCTCWithPadding
import copy

@dataclass
class DataCollatorCTCWithPadding_forMT:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": ast.literal_eval(feature["input_values"])} for feature in features]
        AD_labels = [{"dementia_labels": feature["dementia_labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # labels of single transcript
        if "labels" in features[0].keys():
            label_features = [{"input_ids": feature["labels"]} for feature in features]

            with self.processor.as_target_processor():
                labels_batch = self.processor.pad(
                    label_features,
                    padding=self.padding,
                    max_length=self.max_length_labels,
                    pad_to_multiple_of=self.pad_to_multiple_of_labels,
                    return_tensors="pt",                                                            # to torch tensor
                )
            # replace padding with -100 to ignore loss correctly
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
            batch["labels"] = labels

        # labels of multiple transcript
        if "labels_lst" in features[0].keys():
            # 處理labels list
            all_label_features = []
            for feature in features:
                sample_label_features = [{"input_ids": feature["labels_lst"][i]} for i in range(len(feature["labels_lst"]))]
                all_label_features.extend(sample_label_features)                                    # 把每個label list拆開來

            with self.processor.as_target_processor():
                labels_batch = self.processor.pad(
                    all_label_features,
                    padding=True,
                    max_length=None,
                    pad_to_multiple_of=None,
                    return_tensors="pt",                                                            # to torch tensor
                )

            # replace padding with -100 to ignore loss correctly
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

            labels_lst = []
            num_lms = len(feature["labels_lst"])
            i = 0
            for feature in features:
                sample_labels = [torch.tensor(labels[i*num_lms + j]) for j in range(num_lms)]       # 抓這個sample的所有transcript的labels, labels should be tensor
                labels_lst.append(sample_labels)                                                    # 塞回每個sample
                i += 1                                                                              # 移到下一個starting idx
            batch["labels_lst"] = labels_lst

        batch["dementia_labels"] = torch.tensor([torch.tensor(d['dementia_labels']) for d in AD_labels]) 
                                                                                                    # list of dict to list of tensor
        return batch
    
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

def gen_Ntranscripts(dataset, model, processor, device, num_lms, TRAIN, GPU_batchsize=16):
    if GPU_batchsize!=None:
        bs=int(GPU_batchsize)
        df=pd.DataFrame()
        for i in tqdm(range(0,len(dataset),bs)):
            idxs=list(range(i,min(i+bs,len(dataset))))
            subset_dataset = Subset(dataset, idxs)
            df_data=get_Embs(subset_dataset, model, processor, device, num_lms, TRAIN)
            df = pd.concat([df, df_data], ignore_index=True)
    else:
        # get emb.s, masks... 1 sample by 1 sample
        df = map_to_result(dataset[0], 0, model, processor, num_lms)
        for i in tqdm(range(len(dataset) - 1)):
            df2 = map_to_result(dataset[i+1], i+1, model, processor, num_lms)
            df = pd.concat([df, df2], ignore_index=True)
    return df, Dataset.from_pandas(df)                                      # return dataframe & Dataset

def get_Embs(subset_dataset, model, processor, device, num_lms, TRAIN):
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

        #// TODO model改成可以預測 N-best hypotheses的形式
        logits=model(input_values, EXTRACT=True).logits                                             # get full info like mask, emb. ...
        asr_lg = logits['ASR logits']
        # 轉換length的度量從sample到output的timestep
        ratio=max(lengths)/asr_lg.shape[1]                                                          # (batchsize, seqlength, logitsize)
        oupLens=[int(l/ratio) for l in lengths]
        # for l in lengths:
        #     oupLens.append(int(l/ratio))
        pred_ids = torch.argmax(asr_lg, dim=-1)
        pred_str=processor.batch_decode(pred_ids)

        # 生成num_lms個transcript
        transcripts = []
        for i in range(num_lms): 
            logits=model(input_values, EXTRACT=True).logits                                         # logits for a batch
            asr_lg = logits['ASR logits']

            probs = torch.softmax(asr_lg, dim=-1)
            pred_ids = torch.argmax(asr_lg, dim=-1)
            pred_str=processor.batch_decode(pred_ids)                                               # predicted transcript

            confidence_scores = [probs[i].max().item() for i in range(len(probs))]                  # confidence scores

            with processor.as_target_processor():
                labels = processor(pred_str).input_ids                                              # get label from predicted transcript
                transcripts.append((pred_str, labels, confidence_scores))
        
    df = pd.DataFrame()
    dummy=None
    for i in range(len(subset_dataset)):
        RealLength=oupLens[i]                                                                       #只要有從logits取出來的都要還原
        if TRAIN:                                                                                   # info needed for training
            df2 = pd.DataFrame({'path': subset_dataset[i]["path"],                                  # to know which sample
                 'array': str(subset_dataset[i]["array"]),
                 #'text': subset_dataset[i]["text"],
                 'dementia_labels': subset_dataset[i]["dementia_labels"],
                 'input_values': str(subset_dataset[i]["input_values"]),                            # input of the model
                # 'labels': str(subset_dataset[i]["labels"]), # need to be removed!!!!!!!!!!!!!!!!!
                # 'ASR logits': str(logits["ASR logits"][i].tolist()),
                # 'hidden_states': str(logits["hidden_states"][i].tolist()), #原本的hidden state架構
                # 'hidden_states': [logits["hidden_states"][i][:RealLength,:].cpu().numpy()],  #(time-step,node_dimension)
                #'pred_str': pred_str[i],
                'pred_str_lst': [[transcripts[j][0][i] for j in range(num_lms)]],                   # j-th transcript(y=0) of i-th sample
                'labels_lst': [[transcripts[j][1][i] for j in range(num_lms)]],                     # j-th labels(y=1) of i-th sample
                'confidence_scores_lst': [[transcripts[j][2][i] for j in range(num_lms)]],          # j-th confidence_scores(y=1) of i-th sample
                },
                index=[i])
        else:                                                                                       # info needed for evaluation
            df2 = pd.DataFrame({'path': subset_dataset[i]["path"],                                  # to know which sample
                # 'array': str(subset_dataset[i]["array"]),
                # 'text': subset_dataset[i]["text"],
                # 'dementia_labels': subset_dataset[i]["dementia_labels"],
                # 'input_values': str(subset_dataset[i]["input_values"]),                           # input of the model
                # 'labels': str(subset_dataset[i]["labels"]),
                # 'ASR logits': str(logits["ASR logits"][i].tolist()),
                # 'hidden_states': str(logits["hidden_states"][i].tolist()), #原本的hidden state架構
                # 'hidden_states': [logits["hidden_states"][i][:RealLength,:].cpu().numpy()],  #(time-step,node_dimension)
                #'pred_str': pred_str[i],
                'pred_str_lst': [[transcripts[j][0][i] for j in range(num_lms)]],                   # j-th transcript(y=0) of i-th sample
                'labels_lst': [[transcripts[j][1][i] for j in range(num_lms)]],                     # j-th labels(y=1) of i-th sample
                'confidence_scores_lst': [[transcripts[j][2][i] for j in range(num_lms)]],          # j-th confidence_scores(y=1) of i-th sample
                },
                index=[i])
        df = pd.concat([df, df2], ignore_index=True)
    return df

def map_to_result(batch, idx, model, processor, num_lms):
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
    
    transcripts = []
    for i in range(num_lms):  # 假設要生成5個transcript
        predicted_ids = model(input_values).logits.argmax(-1)
        decoded_output = processor.decode(predicted_ids[0], skip_special_tokens=True)
        confidence = torch.softmax(model(input_values).logits, dim=-1).max().item()
        
        with processor.as_target_processor():
            labels = processor(decoded_output).input_ids
            #print("labels:", labels)
            #print("predicted_ids: ", predicted_ids)
            transcripts.append((decoded_output, torch.tensor(labels), confidence)) # labels should be tensor
        
    transcripts = sorted(transcripts, key=lambda x: x[2], reverse=True) # confidence由大到小
    #print(transcripts)
    transcripts_lst = [transcript[0] for transcript in transcripts]
    #print(transcripts_lst)
    labels_lst = [transcript[1] for transcript in transcripts]
    batch["labels_lst"] = labels_lst
    batch["transcripts_lst"] = transcripts_lst

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
                'pred_str': batch["pred_str"],
                "labels_lst":batch["labels_lst"],
                "transcripts_lst": batch["transcripts_lst"],
                },
                index=[idx])
    return df

def update_MTnetwork_weight(args, source_path, target_weight, network):             # update "network" in source_path with given weights
    # read source model                                                             # return model   
    mask_time_prob = 0                                                              # change config to avoid training stopping
    config = Data2VecAudioConfig.from_pretrained(args.pretrain_name, mask_time_prob=mask_time_prob)
                                                                                    # use pre-trained config
    model = Data2VecAudioForCTCMultitask.from_pretrained(source_path, config=config, args=args)
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

        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)

class ASRLocalUpdate_multi(object):
    def __init__(self, args, dataset_supervised, dataset_unsupervised, global_test_dataset, client_id, 
                 model_in_path, model_out_path):
        self.args = args                                                            # given configuration
        self.client_train_dataset_supervised = self.train_split_supervised(dataset_supervised, client_id)            
                                                                                    # get subset of training set (dataset of THIS client)
                                                                                    # for supervised training (w/ transcripts)
        self.client_train_dataset_unsupervised = self.train_split_unsupervised(dataset_unsupervised, client_id)            
                                                                                    # get subset of training set (dataset of THIS client)
                                                                                    # for unsupervised training (w.o. transcripts)
        self.device = 'cuda' if args.gpu else 'cpu'                                 # use gpu or cpu
        
        self.global_test_dataset = global_test_dataset                              # global testing set for evaluation
        #self.client_test_dataset = self.test_split(global_test_dataset, client_id)  # get subset of testing set (dataset of THIS client)
        self.processor = Wav2Vec2Processor.from_pretrained(args.pretrain_name)
        self.data_collator_unsupervised = DataCollatorCTCWithPadding_forMT(processor=self.processor, padding=True)
                                                                                    # data collator for audio w.o. transcript
        self.data_collator_supervised = DataCollatorCTCWithPadding(processor=self.processor, padding=True)
                                                                                    # data collator for audio w/ transcript
        self.client_id = client_id

        self.model_in_path = model_in_path                                          # no info for client_id & global_round
        self.model_out_path = model_out_path   
    
    def train_split_supervised(self, dataset, client_id):
        # generate sub- training set for given user-ID
        if client_id == "public":                                                   # get spk_id for public dataset, 54 PAR (50% of all training set)
            client_spks = ['S086', 'S021', 'S018', 'S156', 'S016', 'S077', 'S027', 'S116', 'S143', 'S082', 'S039', 'S150', 'S004', 'S126', 'S137', 
            'S097', 'S128', 'S059', 'S096', 'S081', 'S135', 'S094', 'S070', 'S049', 'S080', 'S040', 'S076', 'S093', 'S141', 'S034', 'S056', 'S090', 
            'S130', 'S092', 'S055', 'S019', 'S154', 'S017', 'S114', 'S100', 'S036', 'S029', 'S127', 'S073', 'S089', 'S051', 'S005', 'S151', 'S003', 
            'S033', 'S007', 'S084', 'S043', 'S009']                                 # 27 AD + 27 HC
        elif client_id == "public2":                                                # get spk_id for public dataset, 54 PAR (50% of all training set) from clients
            client_spks = ['S058', 'S030', 'S064', 'S104', 'S048', 'S118', 'S122', 'S001', 'S087', 'S013', 'S025', 'S083', 'S067', 'S068', 'S111', 
            'S028', 'S015', 'S108', 'S095', 'S002', 'S072', 'S020', 'S148', 'S144', 'S110', 'S124', 'S129', 'S071', 'S136', 'S140', 'S145', 'S032', 
            'S101', 'S103', 'S139', 'S038', 'S153', 'S035', 'S011', 'S132', 'S006', 'S149', 'S041', 'S079', 'S107', 'S063', 'S061', 'S125', 'S062', 
            'S012', 'S138', 'S024', 'S052', 'S142']                                 # 27 AD + 27 HC
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
    
    def train_split_unsupervised(self, dataset, client_id):
        # generate sub- training set for given user-ID
        if client_id == 0:                                                        # get spk_id for client 1, 80 PAR (50% of ADReSSo)
            client_spks = ['adrso089', 'adrso148', 'adrso134', 'adrso189', 'adrso205', 'adrso162', 'adrso281', 'adrso156', 'adrso144', 'adrso183', 
                           'adrso222', 'adrso126', 'adrso223', 'adrso045', 'adrso025', 'adrso182', 'adrso070', 'adrso283', 'adrso098', 'adrso233', 
                           'adrso071', 'adrso008', 'adrso068', 'adrso154', 'adrso072', 'adrso015', 'adrso274', 'adrso046', 'adrso248', 'adrso141', 
                           'adrso315', 'adrso027', 'adrso236', 'adrso276', 'adrso031', 'adrso130', 'adrso267', 'adrso090', 'adrso211', 'adrso186', 
                           'adrso265', 'adrso047', 'adrso259', 'adrso128', 'adrso245', 'adrso229', 'adrso152', 'adrso307', 'adrso151', 'adrso197', 
                           'adrso109', 'adrso247', 'adrso003', 'adrso054', 'adrso167', 'adrso178', 'adrso308', 'adrso316', 'adrso278', 'adrso300', 
                           'adrso277', 'adrso012', 'adrso198', 'adrso106', 'adrso158', 'adrso053', 'adrso010', 'adrso160', 'adrso296', 'adrso289', 
                           'adrso168', 'adrso170', 'adrso187', 'adrso234', 'adrso224', 'adrso280', 'adrso138', 'adrso123', 'adrso056', 'adrso043']
                                                                                    # 43 AD + 37 HC
        elif client_id == 1:                                                        # get spk_id for client 2, 81 PAR (50% of ADReSSo)  
            client_spks = ['adrso032', 'adrso039', 'adrso260', 'adrso110', 'adrso216', 'adrso005', 'adrso028', 'adrso122', 'adrso078', 'adrso285', 
                           'adrso292', 'adrso014', 'adrso063', 'adrso262', 'adrso036', 'adrso164', 'adrso298', 'adrso218', 'adrso232', 'adrso060', 
                           'adrso273', 'adrso024', 'adrso172', 'adrso033', 'adrso212', 'adrso173', 'adrso077', 'adrso250', 'adrso253', 'adrso244', 
                           'adrso092', 'adrso180', 'adrso192', 'adrso215', 'adrso264', 'adrso209', 'adrso309', 'adrso125', 'adrso268', 'adrso017', 
                           'adrso257', 'adrso302', 'adrso093', 'adrso112', 'adrso177', 'adrso246', 'adrso312', 'adrso249', 'adrso220', 'adrso266', 
                           'adrso055', 'adrso286', 'adrso237', 'adrso263', 'adrso206', 'adrso202', 'adrso200', 'adrso188', 'adrso142', 'adrso002', 
                           'adrso161', 'adrso291', 'adrso007', 'adrso059', 'adrso310', 'adrso270', 'adrso016', 'adrso075', 'adrso228', 'adrso159', 
                           'adrso261', 'adrso074', 'adrso169', 'adrso049', 'adrso116', 'adrso165', 'adrso157', 'adrso299', 'adrso190', 'adrso153', 'adrso035']
                                                                                    # 44 AD + 37 HC
        else:
            print("Train with whole dataset!!")
            return dataset
        
        print("Generating client training set for client ", str(client_id), "...")
        client_train_dataset = dataset.filter(lambda example: example["path"].startswith(tuple(client_spks)))
        
        return client_train_dataset
    
    def test_split(self, dataset, client_id):
        # generate sub- testing set for given user-ID
        client_test_dataset = dataset
        """
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
        """
        return client_test_dataset
    
    def record_result(self, trainer, result_folder):                                # save training loss, testing loss, and testing wer
        logger = SummaryWriter('./logs/' + result_folder.split("/")[-1])            # use name of this model as folder's name

        for idx in range(len(trainer.state.log_history)):
            if "loss" in trainer.state.log_history[idx].keys():                     # add in training loss, epoch*100 to obtain int
                logger.add_scalar('Loss/train', trainer.state.log_history[idx]["loss"], trainer.state.log_history[idx]["epoch"]*100)

            elif "eval_loss" in trainer.state.log_history[idx].keys():              # add in testing loss & WER, epoch*100 to obtain int
                logger.add_scalar('Loss/test', trainer.state.log_history[idx]["eval_loss"], trainer.state.log_history[idx]["epoch"]*100)
                logger.add_scalar('wer/test', trainer.state.log_history[idx]["eval_wer"], trainer.state.log_history[idx]["epoch"]*100)

            else:                                                                   # add in final training loss, epoch*100 to obtain int
                logger.add_scalar('Loss/train', trainer.state.log_history[idx]["train_loss"], trainer.state.log_history[idx]["epoch"]*100)
        logger.close()

    def update_weights_adapted(self, global_weights, global_round, init_head, fully_unsupervised):
        # 從server端更新seed model
        if global_weights == None:                                                  # train from model from model_in_path
            mask_time_prob = 0                                                      # change config to avoid training stopping
            config = Data2VecAudioConfig.from_pretrained(self.args.pretrain_name, mask_time_prob=mask_time_prob)
                                                                                    # use pre-trained config
            model = Data2VecAudioForCTCMultitask.from_pretrained(self.model_in_path, config=config, args=self.args)
            model.config.ctc_zero_infinity = True                                   # to avoid inf values
        else:                                                                       # update train model using given weight
            if self.args.STAGE == 0:                                                # train ASR
                model = update_MTnetwork_weight(args=self.args, source_path=self.model_in_path, target_weight=global_weights, network="ASR")                    
                                                                                    # from model from model_in_path, update ASR's weight
                init_head = 1                                                       # need to update weights for lm_heads
            elif self.args.STAGE == 1:                                              # train AD classifier
                model = update_MTnetwork_weight(args=self.args, source_path=self.model_in_path, target_weight=global_weights, network="AD")           
                                                                                    # from model from model_in_path, update AD classifier's weight
            elif self.args.STAGE == 2:                                              # train toggling network
                model = update_MTnetwork_weight(args=self.args, source_path=self.model_in_path, target_weight=global_weights, network="toggling_network")             
                                                                                    # from model from model_in_path, update arbitrator's weight
        if init_head:                                                               # initial heads when needed
            model.lm_heads_init()
        
        model.train()
        if self.args.STAGE == 0:                                                    # fine-tune ASR
            lr = 1e-5
        elif self.args.STAGE == 1:                                                  # train AD classifier
            lr = 1e-4
        elif self.args.STAGE == 2:                                                  # train toggling network
            lr = 1e-3

        # unsupervised training for clients only
        save_path = self.model_out_path + "_client" + str(self.client_id) + "_round" + str(global_round) + "_unsuper"
                                                                                    # for local models, record info for id & num_round
        # 訓練在沒有label的資料上    (self-supervise)
        # use starting model to generate transcripts
        _, self.client_train_dataset_unsupervised = gen_Ntranscripts(dataset=self.client_train_dataset_unsupervised, model=model, processor=self.processor, 
                                                                     device=self.device, num_lms=self.args.num_lms, TRAIN=0, GPU_batchsize=16)

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
            eval_steps=100, # 500
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

        trainer = CustomTrainer(
            model=model,
            data_collator=self.data_collator_unsupervised,
            args=training_args,
            compute_metrics=compute_metrics,
            train_dataset=self.client_train_dataset_unsupervised,
            eval_dataset=self.global_test_dataset,
            tokenizer=self.processor.feature_extractor,
            #callbacks=[WERCallback(self.client_train_dataset)], # eval at the end
        )

        print(" | Client ", str(self.client_id), " ready to train unsupervised! |")
        trainer.train()
        # 取第一個lm_head給後面supervised training
        trainer.model.lm_head.load_state_dict(copy.deepcopy(trainer.model.lm_heads[0].state_dict())) 
                                                                                    # update model's lm_head
        trainer.save_model(save_path + "/final")                                    # save final model
        self.record_result(trainer, save_path)                                      # save training loss, testing loss, and testing wer

        if fully_unsupervised == False:                                             # if not fully unsupervised, then perform supervised training
            # perform supervised training after unsupervised one
            # load unsupervised trained model into supervised one
            mask_time_prob = 0                                                      # change config to avoid training stopping
            config = Data2VecAudioConfig.from_pretrained(self.args.pretrain_name, mask_time_prob=mask_time_prob)
                                                                                    # use pre-trained config
            model = Data2VecAudioForCTC.from_pretrained(save_path + "/final", config=config, args=self.args)
            model.config.ctc_zero_infinity = True                                   # to avoid inf values
            model.train()

            save_path = self.model_out_path + "_client" + str(self.client_id) + "_round" + str(global_round)

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
                eval_steps=100, # 500
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

            trainer = CustomTrainer(
                model=model,
                data_collator=self.data_collator_supervised,
                args=training_args,
                compute_metrics=compute_metrics,
                train_dataset=self.client_train_dataset_supervised,
                eval_dataset=self.global_test_dataset,
                tokenizer=self.processor.feature_extractor,
                #callbacks=[WERCallback(self.client_train_dataset)], # eval at the end
            )
            print(" | Client ", str(self.client_id), " ready to train supervised! |")
            trainer.train()
            trainer.save_model(save_path + "/final")                                # save final model
            self.record_result(trainer, save_path)                                  # save training loss, testing loss, and testing wer


        # get "network" weights from model in source_path
        if self.args.STAGE == 0:                                                    # train ASR
            return_weights = get_model_weight(args=self.args, source_path=save_path + "/final/", network="ASR")
        elif self.args.STAGE == 1:                                                  # train AD classifier
            return_weights = get_model_weight(args=self.args, source_path=save_path + "/final/", network="AD")
        elif self.args.STAGE == 2:                                                  # train toggling_network
            return_weights = get_model_weight(args=self.args, source_path=save_path + "/final/", network="toggling_network")  
         
        return return_weights, trainer.state.log_history[-1]["train_loss"]          # return weight, average losses for this round
