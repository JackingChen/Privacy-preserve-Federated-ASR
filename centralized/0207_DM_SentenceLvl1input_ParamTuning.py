import os
import pandas as pd
import numpy as np
from pprint import pprint
from pathlib import Path
from collections import Counter
import pickle
import random
import argparse
import time
from datetime import datetime

# torch:
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss

from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,classification_report
from sklearn.model_selection import train_test_split

from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
from transformers import BertTokenizer, BertConfig, BertModel,XLMTokenizer, XLMModel

from Dementia_challenge_models import SingleForwardModel, BertPooler, Audio_pretrain, ModelArg, Model_settings_dict, Text_pretrain, SingleForwardModelRegression
import librosa

class Model(SingleForwardModel):
    def __init__(self, args, config):
        super().__init__(args, config)
        self.inpArg = args.inpArg
        self.inp_embed_type = self.config['inp_embed']
        self.inp_col_name = self.inpArg.inp_col_name
        self.inp_hidden_size = self.inpArg.inp_hidden_size
        self.hidden = int(self.inpArg.linear_hidden_size)
        self.inp_tokenizer, self.inp_model, self.pooler=self._setup_embedding(self.inp_embed_type, self.inp_hidden_size)
        self.clf1 = nn.Linear(self.hidden, int(self.hidden/2))
        self.clf2 = nn.Linear(int(self.hidden/2), self.num_labels)
    def _safe_outputStr_gen(self):
        self.outStr=self.config['params_tuning_str'].replace("/","_")
    def _save_results_to_csv(self, df_result, pred_dict, args, suffix):
        # Save df_result to CSV
        self._safe_outputStr_gen()
        df_result.to_csv(f'{args.Output_dir}/{self.outStr}{suffix}.csv')

        # Save pred_df to CSV
        pred_df = pd.DataFrame(pred_dict)
        pred_df.to_csv(f'{args.Output_dir}/{self.outStr}{suffix}_pred.csv')



class ModelRegression(SingleForwardModelRegression):
    def __init__(self, args, config):
        super().__init__(args, config)
        self.inpArg = args.inpArg
        self.inp_embed_type = self.config['inp_embed']
        self.inp_col_name = self.inpArg.inp_col_name
        self.inp_hidden_size = self.inpArg.inp_hidden_size
        self.hidden = int(self.inpArg.linear_hidden_size)
        self.inp_tokenizer, self.inp_model, self.pooler=self._setup_embedding(self.inp_embed_type, self.inp_hidden_size)
        self.clf1 = nn.Linear(self.hidden, int(self.hidden/2))
        self.clf2 = nn.Linear(int(self.hidden/2), self.num_labels)

def main(args,config):
    print("Using PyTorch Ver", torch.__version__)
    print("Fix Seed:", config['random_seed'])
    seed_everything( config['random_seed'])
    
    if config['task']=='regression':
        model=ModelRegression(args,config)
    elif config['task']=='classification':
        model = Model(args,config) 
    else:
        raise ValueError()
    model.preprocess_dataframe()

    early_stop_callback = EarlyStopping(
        monitor='val_acc',
        patience=config['patience'],
        verbose=True,
        mode='max'
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{SaveRoot}/Model/{config['params_tuning_str']}/checkpoints",
        monitor='val_acc',
        auto_insert_metric_name=True,
        verbose=True,
        mode='max', 
        save_top_k=1,
      )    

    print(":: Start Training ::")
    #     
    trainer = Trainer(
        logger=False,
        callbacks=[early_stop_callback,checkpoint_callback],
        enable_checkpointing = True,
        max_epochs=args.mdlArg.epochs,
        fast_dev_run=args.mdlArg.test_mode,
        num_sanity_val_steps=None if args.mdlArg.test_mode else 0,
        # deterministic=True, # True會有bug，先false
        deterministic=False,
        # For GPU Setup
        # gpus=[config['gpu']] if torch.cuda.is_available() else None,
        strategy='ddp_find_unused_parameters_true',
        precision=16 if args.mdlArg.fp16 else 32
    )
    trainer.fit(model)
    trainer.test(model,dataloaders=model.test_dataloader(),ckpt_path="best")
    
if __name__ == '__main__': 

    parser = argparse.ArgumentParser("main.py", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--gpu", type=int, default=1)
    
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=2e-5, help="learning rate")
    parser.add_argument("--lr_scheduler", type=str, default='exp', help="learning rate")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--random_seed", type=int, default=2023) 

    parser.add_argument("--inp_embed", type=str, default="mbert") 
    parser.add_argument("--SaveRoot", type=str, default='/mnt/External/Seagate/FedASR/LLaMa2/dacs') 
    parser.add_argument("--file_in", type=str, default='/home/FedASR/dacs/centralized/saves/results/data2vec-audio-large-960h') 
    parser.add_argument("--task", type=str, default="classification") 
    
    config = parser.parse_args()
    SaveRoot=config.SaveRoot
    script_path, file_extension = os.path.splitext(__file__)


    config.params_tuning_str='__'.join([config.inp_embed,str(config.epochs),str(config.lr),config.lr_scheduler,str(config.patience)])
    # 使用os.path模組取得檔案名稱
    script_name = os.path.basename(script_path)
    task_str='result_regression' if config.task=='regression' else 'result_classification'
    Output_dir=f"{SaveRoot}/{task_str}/{script_name}/" 
    
    os.makedirs(Output_dir, exist_ok=True)
    print(config)

    
    class InpArg:
        inp_hidden_size = Model_settings_dict[config.inp_embed]['inp_hidden_size']
        pool_hidden_size = inp_hidden_size # BERT-base: 768, BERT-large: 1024, BERT paper setting
        linear_hidden_size = inp_hidden_size
        inp_col_name = Model_settings_dict[config.inp_embed]['inp_col_name']
        file_in = Model_settings_dict[config.inp_embed]['file_in']
    
    
    
    class ModelArgTuning:
        version = 1
        # data
        epochs: int = config.epochs  # Max Epochs, BERT paper setting [3,4,5]
        max_length: int = 350  # Max Length input size
        report_cycle: int = 30  # Report (Train Metrics) Cycle
        cpu_workers: int = os.cpu_count()  # Multi cpu workers
        test_mode: bool = False  # Test Mode enables `fast_dev_run`
        lr_scheduler: str = config.lr_scheduler  # ExponentialLR vs CosineAnnealingWarmRestarts
        fp16: bool = False  # Enable train on FP16
        batch_size: int = 8

    
    class Arg:
        mdlArg=ModelArgTuning()
        inpArg=InpArg()
        Output_dir=Output_dir

    args = Arg()
    # args.mdlArg.epochs=config.epochs
    main(args,config.__dict__)       


"""

python 0207_DM_multi.py --gpu 1 --t_embed mbert --a_embed en
python 0207_DM_multi.py --gpu 1 --t_embed xlm --a_embed en

# don
python 0207_DM_multi.py --gpu 0 --t_embed xlm --a_embed gr
python 0207_DM_multi.py --gpu 1 --t_embed mbert --a_embed gr

"""