# =============================================================
# 更新2023/04/10
# 1. csv2dataset函數裡面使用csv_path和root_path
# 2. 在讀音檔的時候增加一個選項：scipy.io，讀起來會快很多但是不知道會不會影響到原來的效果

# 大約10138MiB

# 訓練一次在CPU很滿的情況下7hr
# =============================================================

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import os.path
import pandas as pd
from datasets import Dataset, load_from_disk
import librosa
from datasets import load_dataset, load_metric
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Trainer, TrainingArguments
from transformers import Data2VecAudioConfig, HubertConfig, SEWDConfig, UniSpeechSatConfig
from transformers import Data2VecAudioForCTC, HubertForCTC, SEWDForCTC, UniSpeechSatForCTC
from jiwer import wer
import scipy.io
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
import argparse
# from update import compute_metrics



# set up trainer
@dataclass
class DataCollatorCTCWithPadding:
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
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch



parser = argparse.ArgumentParser()
#parser.add_argument('-model', '--model_path', type=str, default="./saves/wav2vec2-base-960h_GRL_0.5", help="Where the model is saved")
parser.add_argument('-opt', '--optimizer', type=str, default="adamw_hf", help="The optimizer to use: adamw_hf, adamw_torch, adamw_apex_fused, or adafactor")
parser.add_argument('-MGN', '--max_grad_norm', type=float, default=1.0, help="Maximum gradient norm (for gradient clipping)")
parser.add_argument('-model_type', '--model_type', type=str, default="data2vec", help="Type of the model")
parser.add_argument('-sr', '--sampl_rate', type=float, default=16000, help="librosa read smping rate")
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-5, help="Learning rate")
parser.add_argument('-RD', '--root_dir', default='/mnt/Internal/FedASR/Data/ADReSS-IS2020-data', help="Learning rate")
parser.add_argument('--epochs', type=int, default=10,
                        help="number of rounds of training")
parser.add_argument('--num_users', type=int, default=100,
                    help="number of users: K")
parser.add_argument('--frac', type=float, default=0.1,
                    help='the fraction of clients: C')
parser.add_argument('--local_ep', type=int, default=10,
                    help="the number of local epochs: E")
# model arguments
parser.add_argument('--model', type=str, default='data2vec', help='model name')
# other arguments
parser.add_argument('--dataset', type=str, default='adress', help="name \
                    of dataset")
parser.add_argument('--gpu', default=None, help="To use cuda, set \
                    to a specific GPU ID. Default set to use CPU.")
# additional arguments
parser.add_argument('--pretrain_name', type=str, default='facebook/data2vec-audio-large-960h', help="str used to load pretrain model")

parser.add_argument('-lam', '--LAMBDA', type=float, default=0.5, help="Lambda for GRL")
parser.add_argument('-st', '--STAGE', type=int, default=0, help="Current training stage")
parser.add_argument('-fl_st', '--FL_STAGE', type=int, default=1, help="Current FL training stage")
parser.add_argument('-GRL', '--GRL', action='store_true', default=False, help="True: GRL")
parser.add_argument('-model_in', '--model_in_path', type=str, default="/home/FedASR/dacs/federated/save/data2vec-audio-large-960h_new1_recall_FLASR", help="Where the global model is saved")
parser.add_argument('-model_out', '--model_out_path', type=str, default="/home/FedASR/dacs/federated/save/", help="Where to save the model")
# parser.add_argument('-log', '--log_path', type=str, default="wav2vec2-base-960h_linear_GRL.txt", help="name for the txt file")
parser.add_argument('-csv', '--csv_path', type=str, default="wav2vec2-base-960h_GRL_0.5", help="name for the csv file")
# 2023/01/08: loss type
parser.add_argument('-ad_loss', '--AD_loss', type=str, default=None, help="loss to use for AD classifier")
# 2023/01/18: ckpt
parser.add_argument('-ckpt', '--checkpoint', type=str, default=None, help="path to checkpoint")
# 2023/02/13: TOGGLE_RATIO
parser.add_argument('-toggle_rt', '--TOGGLE_RATIO', type=float, default=0, help="To toggle more or less")
# 2023/02/15: GS_TAU, loss weight
parser.add_argument('-gs_tau', '--GS_TAU', type=float, default=1, help="Tau for gumbel_softmax")
parser.add_argument('-w_loss', '--W_LOSS', type=float, default=None, nargs='+', help="weight for HC and AD")
# 2023/04/20
parser.add_argument('-EXTRACT', '--EXTRACT', action='store_true', default=False, help="True: extract embs")
parser.add_argument('-client_id', '--client_id', type=str, default="public", help="client_id: public, 0, or 1")
# 2023/04/24
parser.add_argument('--global_ep', type=int, default=30, help="number for global model")
    

args = parser.parse_args()

from utils import csv2dataset, get_dataset, average_weights, exp_details
from update import update_network_weight
#model_out_dir = args.model_path # where to save model
model_type = args.model_type                # what type of the model
lr = args.learning_rate                     # learning rate
optim = args.optimizer                      # opt
max_grad_norm = args.max_grad_norm          # max_grad_norm
args.log_path=os.path.basename(args.model_in_path)
args.log_path+="epoch-{}".format(args.local_ep)

def Write_log(content,LOG_DIR="/home/FedASR/dacs/federated/logs/"):
    # write to txt file
    file_object = open(LOG_DIR + args.log_path, 'a')
    # Append at the end of file
    file_object.write(json.dumps(content) + '\n')
    # Close the file
    file_object.close()

def map_to_result(batch, model,processor):
    with torch.no_grad():
        input_values = torch.tensor(batch["input_values"]).unsqueeze(0)
        logits = model(input_values).logits

    pred_ids = torch.argmax(logits, dim=-1)
    batch["pred_str"] = processor.batch_decode(pred_ids)[0]
    batch["text"] = processor.decode(batch["labels"], group_tokens=False)
  
    return batch


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

        DACS_codeRoot = os.getenv("DACS_codeRoot")

        output_path=f"{DACS_codeRoot}/federated/logs/"
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        Write_log(LOG_DIR=f"{DACS_codeRoot}/federated/logs/",content=output)

        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)


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
        log_path = self.args.log_path

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
        # self.debug_mdl=model
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
        return trainer.model

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


exp_details(args)                                                               # print out details based on configuration
# 吃資料
train_dataset, test_dataset = get_dataset(args)                                 # get dataset



# train_data = csv2dataset(PATH = '{}/clips/'.format(args.root_dir),
#                          path = "{}/mid_csv/train.csv".format(args.root_dir)) #!!! librosa在load的時候非常慢，大約7分47秒讀完1869個file
# dev_data = csv2dataset(PATH = '{}/clips/'.format(args.root_dir),
#                        path = "{}/mid_csv/dev.csv".format(args.root_dir))
# test_data = csv2dataset(PATH = '{}/clips/'.format(args.root_dir),
#                         path = "{}/mid_csv/test.csv".format(args.root_dir))
# 吃global

client_id=0
model_in_path=args.pretrain_name
global_mdl_path=args.model_in_path
model_out_path=args.model_out_path
global_weights = get_model_weight(args=args, source_path=global_mdl_path + "_global/final/", network="ASR")
                                                                    # local ASR and AD with global toggling network
                                                                    # get toggling_network weights from model in model_out_path + "_global/final/"
# 選資料
local_model = ASRLocalUpdate(args=args, dataset=train_dataset, global_test_dataset=test_dataset, 
                                 client_id=client_id, model_in_path=model_in_path, model_out_path=model_out_path)
                                                                                      # initial dataset of current client
# 訓練模型
trained_model = local_model.update_weights(global_weights=global_weights, global_round=0) 


args.model_out_path=args.model_out_path+"_epoch-{}".format(args.local_ep)
trained_model.save_pretrained(args.model_out_path+"/final")
# evaluate
map_to_result_with_model = lambda batch: map_to_result(batch, trained_model, local_model.processor)
result = test_dataset.map(map_to_result_with_model)
Write_log({"test_wer= ":wer(result["text"], result["pred_str"])},LOG_DIR="/home/FedASR/dacs/federated/logs/")
print("WER of ", args.pretrain_name, " : ", wer(result["text"], result["pred_str"]))


