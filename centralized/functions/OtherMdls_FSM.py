# only data2vec model is used!!!!!!!!!!
from transformers.models.wav2vec2.configuration_wav2vec2 import Wav2Vec2Config
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
import os

from transformers import (
    Wav2Vec2Processor, Wav2Vec2Model, 
    Data2VecAudioModel, Data2VecAudioPreTrainedModel,
    HubertModel, HubertPreTrainedModel,
    SEWDModel, SEWDPreTrainedModel,
    UniSpeechSatModel, UniSpeechSatPreTrainedModel
)
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2GumbelVectorQuantizer, Wav2Vec2Encoder, Wav2Vec2EncoderStableLayerNorm
from transformers.configuration_utils import PretrainedConfig

from transformers.file_utils import (
    DUMMY_INPUTS,
    FLAX_WEIGHTS_NAME,
    TF2_WEIGHTS_NAME,
    TF_WEIGHTS_NAME,
    WEIGHTS_NAME,
    ModelOutput,
    PushToHubMixin,
    cached_path,
    copy_func,
    hf_bucket_url,
    is_offline_mode,
    is_remote_url,
    replace_return_docstrings,
    is_datasets_available,
    add_code_sample_docstrings,
)

from transformers.modeling_utils import PreTrainedModel

from transformers.file_utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
    CONFIG_NAME,
    WEIGHTS_NAME,
    get_full_repo_name,
    is_apex_available,
    is_datasets_available,
    is_in_notebook,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_tpu_available,
)

from transformers.modeling_outputs import BaseModelOutput, CausalLMOutput, MaskedLMOutput, SequenceClassifierOutput
from transformers.utils import logging
from transformers.deepspeed import deepspeed_config, is_deepspeed_zero3_enabled

from transformers.debug_utils import DebugOption
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
import os
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, IterableDataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from huggingface_hub import Repository

from transformers.trainer_pt_utils import (
    DistributedLengthGroupedSampler,
    DistributedSamplerWithLoop,
    DistributedTensorGatherer,
    IterableDatasetShard,
    LabelSmoother,
    LengthGroupedSampler,
    SequentialDistributedSampler,
    ShardSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    find_batch_size,
    get_parameter_names,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_truncate,
    nested_xla_mesh_reduce,
    reissue_pt_warnings,
)

from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalLoopOutput,
    EvalPrediction,
    HPSearchBackend,
    HubStrategy,
    IntervalStrategy,
    PredictionOutput,
    ShardedDDPOption,
    TrainerMemoryTracker,
    TrainOutput,
    default_compute_objective,
    default_hp_space,
    denumpify_detensorize,
    get_last_checkpoint,
    number_of_arguments,
    set_seed,
    speed_metrics,
)

from transformers.utils import logging

class ReverseLayerF(torch.autograd.Function):
    def __init__(self):
        super(ReverseLayerF, self).__init__()
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.save_for_backward(lambda_)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        lambda_, = ctx.saved_variables
        grad_input = grad_output.clone()
        return - lambda_ * grad_input, None
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2PreTrainedModel
      class Wav2Vec2ForCTC(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.wav2vec2 = Wav2Vec2Model(config)
        self.dropout = nn.Dropout(config.final_dropout)
        
        if config.vocab_size is None:
            raise ValueError(
                f"You are trying to instantiate {self.__class__} with a configuration that "
                "does not define the vocabulary size of the language model head. Please "
                "instantiate the model as follows: `Wav2Vec2ForCTC.from_pretrained(..., vocab_size=vocab_size)`."
                "or define `vocab_size` of your model's configuration."
            )
        
        self.alpha=torch.tensor(LAMBDA)
        self.dementia_thres = torch.tensor(AD_THRES)
        self.lm_thres = torch.tensor(LM_THRES)
        print("lambda = ", self.alpha)
        print("dementia_thres = ", self.dementia_thres)
        print("lm_thres = ", self.lm_thres)
        
        # 加lm_model
        self.lm_fsm = nn.Linear(config.hidden_size, config.hidden_size)          # 找出對lm重要的feat
        #self.lm_fsm = nn.LSTM(config.hidden_size, config.hidden_size, 1)          # 找出對lm重要的feat
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)          # output字母的"機率"
        self.lm_grl = nn.Linear(config.hidden_size, config.vocab_size)           # 加了GRL那條
        
        # 加dementia model
        self.dementia_fsm = nn.Linear(config.hidden_size, config.hidden_size)    # 找出對AD預測重要的feat
        self.dementia_head = nn.Linear(config.hidden_size, 2)                    # 辨識AD
        self.dementia_grl = nn.Linear(config.hidden_size, 2)                     # 加GRL那條
        
        # define similarity loss: AM-Softmax
        self.criterion_similar = AngularPenaltySMLoss(in_features=config.hidden_size, out_features=2, loss_type='cosface').to('cpu')
        
        # freeze feature_extractor
        self.freeze_feature_extractor()
        
        if STAGE == 1:                                                  # train FSM
            #print("Current stage: 1")
            self.freeze_lm_grl()
            self.freeze_dementia_grl()
            self.freeze_lm_head()
            self.freeze_dementia_head()
        elif STAGE == 2:                                                # train FSM + head
            #print("Current stage: 2")
            self.freeze_lm_grl()
            self.freeze_dementia_grl()
        elif STAGE == 3:                                                # train dementia GRL
            print("Current stage: 3")
            self.freeze_wav2vec2()
            self.freeze_lm_fsm()
            self.freeze_dementia_fsm()
            self.freeze_lm_head()
            self.freeze_dementia_head()
            self.freeze_lm_grl()
        elif STAGE == 4:                                                # train lm GRL
            print("Current stage: 4")
            self.freeze_wav2vec2()
            self.freeze_lm_fsm()
            self.freeze_dementia_fsm()
            self.freeze_lm_head()
            self.freeze_dementia_head()
            self.freeze_dementia_grl()
        elif STAGE == 5:                                                # train encoder
            # train encoder
            """
            self.freeze_lm_fsm()
            self.freeze_dementia_fsm()            
            self.freeze_criterion_similar()
            self.freeze_lm_head()
            self.freeze_dementia_head()            
            self.freeze_lm_grl()
            self.freeze_dementia_grl()
            """

            # train lm_FSM
            
            self.freeze_wav2vec2()
            self.freeze_dementia_fsm()            
            self.freeze_criterion_similar()
            self.freeze_lm_head()
            self.freeze_dementia_head()            
            self.freeze_lm_grl()
            self.freeze_dementia_grl()
            
            
            # train dementia_FSM
            """
            self.freeze_wav2vec2()
            self.freeze_lm_fsm()          
            self.freeze_criterion_similar()
            self.freeze_lm_head()
            self.freeze_dementia_head()            
            self.freeze_lm_grl()
            self.freeze_dementia_grl()
            """
            
        self.init_weights()
    
    def freeze_feature_extractor(self):
        """
        Calling this function will disable the gradient computation for the feature extractor so that its parameter
        will not be updated during training.
        """
        self.wav2vec2.feature_extractor._freeze_parameters()

    def freeze_wav2vec2(self):
        self.wav2vec2.eval()
        for param in self.wav2vec2.parameters():
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
    
    @add_start_docstrings_to_model_forward(WAV_2_VEC_2_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutput, config_class=_CONFIG_FOR_DOC)
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
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, target_length)`, `optional`):
            Labels for connectionist temporal classification. Note that ``target_length`` has to be smaller or equal to
            the sequence length of the output logits. Indices are selected in ``[-100, 0, ..., config.vocab_size -
            1]``. All labels set to ``-100`` are ignored (masked), the loss is only computed for labels in ``[0, ...,
            config.vocab_size - 1]``.
        Returns:
        Example::
            >>> import torch
            >>> from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
            >>> from datasets import load_dataset
            >>> import soundfile as sf
            >>> processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
            >>> model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
            >>> def map_to_array(batch):
            >>>     speech, _ = sf.read(batch["file"])
            >>>     batch["speech"] = speech
            >>>     return batch
            >>> ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
            >>> ds = ds.map(map_to_array)
            >>> input_values = processor(ds["speech"][0], return_tensors="pt").input_values  # Batch size 1
            >>> logits = model(input_values).logits
            >>> predicted_ids = torch.argmax(logits, dim=-1)
            >>> transcription = processor.decode(predicted_ids[0])
            >>> # compute loss
            >>> target_transcription = "A MAN SAID TO THE UNIVERSE SIR I EXIST"
            >>> # wrap processor as target processor to encode labels
            >>> with processor.as_target_processor():
            >>>     labels = processor(target_transcription, return_tensors="pt").input_ids
            >>> loss = model(input_values, labels=labels).loss
        """
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        hidden_states = outputs[0]                   # last_hidden_state
        hidden_states = self.dropout(hidden_states)

        
        # hidden_states: wav2vec embedding
        # 製造mask
        m = nn.Sigmoid()
        dementia_score = m(self.dementia_fsm(hidden_states)) # score range from 0~1
        lm_score = m(self.lm_fsm(hidden_states))             # score range from 0~1
        
        dementia_mask = torch.where(dementia_score >= self.dementia_thres.to(dementia_score.device), torch.tensor(1.0).to(dementia_score.device), torch.tensor(0.0).to(dementia_score.device)) # if condition, 1. else, 0
        lm_mask = torch.where(lm_score >= self.lm_thres.to(lm_score.device), torch.tensor(1.0).to(lm_score.device), torch.tensor(0.0).to(lm_score.device))                   # if condition, 1. else, 0
        
        # 拿score vector 跟原本的hidden_states點乘
        dementia_resored = dementia_score*hidden_states
        lm_resored = lm_score*hidden_states
        
        # head(clf)
        dementia_logits = self.dementia_head(dementia_resored) #******************* torch.Size([2, 1327, 32]) (batchsize, timestep, feature_dimention)
        logits = self.lm_head(lm_resored)
        
        
        # grl(dis)
        hidden_states_r = ReverseLayerF.apply(hidden_states, self.alpha)
        dementia_score_r = m(self.dementia_fsm(hidden_states_r)) # score range from 0~1
        lm_score_r = m(self.lm_fsm(hidden_states_r))             # score range from 0~1
        
        dementia_mask_r = torch.where(dementia_score_r >= self.dementia_thres.to(dementia_score_r.device), torch.tensor(1.0).to(dementia_score_r.device), torch.tensor(0.0).to(dementia_score_r.device)) # if condition, 1. else, 0
        lm_mask_r = torch.where(lm_score_r >= self.lm_thres.to(lm_score_r.device), torch.tensor(1.0).to(lm_score_r.device), torch.tensor(0.0).to(lm_score_r.device))                   # if condition, 1. else, 0
        
        # 拿mask跟hidden_states_r點乘
        dementia_masked_r = dementia_mask_r*hidden_states_r
        lm_masked_r = lm_mask_r*hidden_states_r
        
        # grl(dis)
        dementia_logits_r = self.dementia_grl(lm_masked_r) #******************* torch.Size([2, 1327, 32]) (batchsize, timestep, feature_dimention)
        logits_r = self.lm_grl(dementia_masked_r)
        
        dementia_output_mean = torch.mean(dementia_logits,dim=1)
        dementia_output_mean_r = torch.mean(dementia_logits_r,dim=1)
        
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
                #  /////
                # gradient reversal layers(GRL)
                loss_fn = nn.CrossEntropyLoss()
                
                dementia_loss = loss_fn(dementia_output_mean, dementia_labels)        # multi-task
                dementia_loss_rev = loss_fn(dementia_output_mean_r, dementia_labels)  # GRL
                
                # FSM att loss
                # Scorematrix = append([dementia_mask,lm_mask]) # torch.Size([2, embedding_size])
                # Att_loss = Scorematrix*Scorematrix - Identity matrix
                #Att_loss = FSMatt_loss(lm_score, dementia_score)
                Att_loss = FSMatt_loss(lm_mask, dementia_mask)

                # diversity loss: AM-Softmax
                scores = torch.cat((hidden_states * lm_score, hidden_states * dementia_score), dim=0)
                am_labels = torch.cat((torch.zeros(len(hidden_states), dtype=torch.long), torch.ones(len(hidden_states), dtype=torch.long)), dim=0).to('cpu')
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
                # ////
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        logits_all = {'ASR logits': logits, 'dementia logits': dementia_logits, 'hidden_states': hidden_states,
                    'dementia_mask': dementia_mask, 'lm_mask': lm_mask}

        return CausalLMOutput(
            loss=final_loss, logits=logits_all, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )
 
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

    @add_start_docstrings_to_model_forward(DATA2VEC_AUDIO_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_PROCESSOR_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=CausalLMOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_CTC_EXPECTED_OUTPUT,
        expected_loss=_CTC_EXPECTED_LOSS,
    )
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

class HubertForCTC(HubertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.hubert = HubertModel(config)
        self.dropout = nn.Dropout(config.final_dropout)

        if config.vocab_size is None:
            raise ValueError(
                f"You are trying to instantiate {self.__class__} with a configuration that "
                "does not define the vocabulary size of the language model head. Please "
                "instantiate the model as follows: `HubertForCTC.from_pretrained(..., vocab_size=vocab_size)`. "
                "or define `vocab_size` of your model's configuration."
            )
        output_hidden_size = (
            config.output_hidden_size if hasattr(config, "add_adapter") and config.add_adapter else config.hidden_size
        )

        self.alpha=torch.tensor(LAMBDA)
        self.dementia_thres = torch.tensor(AD_THRES)
        self.lm_thres = torch.tensor(LM_THRES)
        print("lambda = ", self.alpha)
        print("dementia_thres = ", self.dementia_thres)
        print("lm_thres = ", self.lm_thres)
        
        # 加lm_model
        self.lm_fsm = nn.Linear(config.hidden_size, config.hidden_size)          # 找出對lm重要的feat
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)                             # output字母的"機率"
        self.lm_grl = nn.Linear(config.hidden_size, config.vocab_size)           # 加了GRL那條
        
        # 加dementia model
        self.dementia_fsm = nn.Linear(config.hidden_size, config.hidden_size)    # 找出對AD預測重要的feat
        self.dementia_head = nn.Linear(config.hidden_size, 2)                    # 辨識AD
        self.dementia_grl = nn.Linear(config.hidden_size, 2)                     # 加GRL那條
        
        # define similarity loss: AM-Softmax
        self.criterion_similar = AngularPenaltySMLoss(in_features=config.hidden_size, out_features=2, loss_type='cosface').to('cpu')
        
        # freeze feature_extractor
        self.freeze_feature_encoder()
        
        if STAGE == 1:                                                  # train FSM
            print("Current stage: 1")
            self.freeze_lm_grl()
            self.freeze_dementia_grl()
            self.freeze_lm_head()
            self.freeze_dementia_head()
        elif STAGE == 2:                                                # train FSM + head
            print("Current stage: 2")
            self.freeze_lm_grl()
            self.freeze_dementia_grl()
        elif STAGE == 3:                                                # train dementia GRL
            print("Current stage: 3")
            self.freeze_hubert()
            self.freeze_lm_fsm()
            self.freeze_dementia_fsm()
            self.freeze_lm_head()
            self.freeze_dementia_head()
            self.freeze_lm_grl()
        elif STAGE == 4:                                                # train lm GRL
            print("Current stage: 4")
            self.freeze_hubert()
            self.freeze_lm_fsm()
            self.freeze_dementia_fsm()
            self.freeze_lm_head()
            self.freeze_dementia_head()
            self.freeze_dementia_grl()
        elif STAGE == 5:                                                # train encoder
            # train lm_FSM
            self.freeze_hubert()
            self.freeze_dementia_fsm()            
            self.freeze_criterion_similar()
            self.freeze_lm_head()
            self.freeze_dementia_head()            
            self.freeze_lm_grl()
            self.freeze_dementia_grl()
    
        # Initialize weights and apply final processing
        self.post_init()

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.hubert.feature_extractor._freeze_parameters()
    
    def freeze_hubert(self):
        self.hubert.eval()
        for param in self.hubert.parameters():
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
 

    @add_start_docstrings_to_model_forward(HUBERT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_PROCESSOR_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=CausalLMOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_CTC_EXPECTED_OUTPUT,
        expected_loss=_CTC_EXPECTED_LOSS,
    )
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

        outputs = self.hubert(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]                   # last_hidden_state
        hidden_states = self.dropout(hidden_states)

        # hidden_states: hubert embedding
        # 製造mask
        m = nn.Sigmoid()
        dementia_score = m(self.dementia_fsm(hidden_states)) # score range from 0~1
        lm_score = m(self.lm_fsm(hidden_states))             # score range from 0~1
        
        #model.recognize(inputs, input_lengths)
        dementia_mask = torch.where(dementia_score >= self.dementia_thres.to(dementia_score.device), torch.tensor(1.0).to(dementia_score.device), torch.tensor(0.0).to(dementia_score.device)) # if condition, 1. else, 0
        lm_mask = torch.where(lm_score >= self.lm_thres.to(lm_score.device), torch.tensor(1.0).to(lm_score.device), torch.tensor(0.0).to(lm_score.device))                   # if condition, 1. else, 0
        lm_mask = lm_mask + 0 * self.lm_fsm(lm_mask) # to has grad?
        dementia_mask = dementia_mask + 0 * self.lm_fsm(lm_mask) # to has grad?

        # 拿score vector 跟原本的hidden_states點乘
        dementia_resored = dementia_score*hidden_states
        lm_resored = lm_score*hidden_states
        
        # head(clf)
        dementia_logits = self.dementia_head(dementia_resored) #******************* torch.Size([2, 1327, 32]) (batchsize, timestep, feature_dimention)
        logits = self.lm_head(lm_resored)
        # del dementia_resored, lm_resored
        
        # grl(dis)
        hidden_states_r = ReverseLayerF.apply(hidden_states, self.alpha)
        dementia_score_r = m(self.dementia_fsm(hidden_states_r)) # score range from 0~1
        lm_score_r = m(self.lm_fsm(hidden_states_r))            # score range from 0~1

        dementia_mask_r = torch.where(dementia_score_r >= self.dementia_thres.to(dementia_score_r.device), torch.tensor(1.0).to(dementia_score_r.device), torch.tensor(0.0).to(dementia_score_r.device)) # if condition, 1. else, 0
        lm_mask_r = torch.where(lm_score_r >= self.lm_thres.to(lm_score_r.device), torch.tensor(1.0).to(lm_score_r.device), torch.tensor(0.0).to(lm_score_r.device))                   # if condition, 1. else, 0
        
        del dementia_score_r, lm_score_r
        # 拿mask跟hidden_states_r點乘
        dementia_masked_r = dementia_mask_r*hidden_states_r
        lm_masked_r = lm_mask_r*hidden_states_r
        
        del hidden_states_r, dementia_mask_r, lm_mask_r
        # grl(dis)
        dementia_logits_r = self.dementia_grl(lm_masked_r) #******************* torch.Size([2, 1327, 32]) (batchsize, timestep, feature_dimention)
        logits_r = self.lm_grl(dementia_masked_r)
        del dementia_masked_r, lm_masked_r
        dementia_output_mean = torch.mean(dementia_logits,dim=1)
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
                #  /////
                # gradient reversal layers(GRL)
                loss_fn = nn.CrossEntropyLoss()
                
                dementia_loss = loss_fn(dementia_output_mean, dementia_labels)        # multi-task
                dementia_loss_rev = loss_fn(dementia_output_mean_r, dementia_labels)  # GRL
                
                # FSM att loss
                # Scorematrix = append([dementia_mask,lm_mask]) # torch.Size([2, embedding_size])
                # Att_loss = Scorematrix*Scorematrix - Identity matrix
                #Att_loss = FSMatt_loss(lm_score, dementia_score)
                Att_loss = FSMatt_loss(lm_mask, dementia_mask)
                # del lm_mask, dementia_mask
                # diversity loss: AM-Softmax
                scores = torch.cat((hidden_states * lm_score, hidden_states * dementia_score), dim=0)
                del lm_score, dementia_score
                am_labels = torch.cat((torch.zeros(len(hidden_states), dtype=torch.long), torch.ones(len(hidden_states), dtype=torch.long)), dim=0).to('cpu')
                #del hidden_states
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
                    # train lm_FSM
                    final_loss = loss + dementia_loss_rev
                # ////
        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        logits_all = {'ASR logits': logits, 'dementia logits': dementia_logits, 'hidden_states': hidden_states,
                    'lm_mask': lm_mask, 'dementia_mask': dementia_mask}

        return CausalLMOutput(
            loss=final_loss, logits=logits_all, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )

class SEWDForCTC(SEWDPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.sew_d = SEWDModel(config)
        self.dropout = nn.Dropout(config.final_dropout)

        if config.vocab_size is None:
            raise ValueError(
                f"You are trying to instantiate {self.__class__} with a configuration that "
                "does not define the vocabulary size of the language model head. Please "
                "instantiate the model as follows: `SEWDForCTC.from_pretrained(..., vocab_size=vocab_size)`. "
                "or define `vocab_size` of your model's configuration."
            )
        output_hidden_size = (
            config.output_hidden_size if hasattr(config, "add_adapter") and config.add_adapter else config.hidden_size
        )

        self.alpha=torch.tensor(LAMBDA)
        self.dementia_thres = torch.tensor(AD_THRES)
        self.lm_thres = torch.tensor(LM_THRES)
        print("lambda = ", self.alpha)
        print("dementia_thres = ", self.dementia_thres)
        print("lm_thres = ", self.lm_thres)

        # 加lm_model
        self.lm_fsm = nn.Linear(config.hidden_size, config.hidden_size)          # 找出對lm重要的feat
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)                             # output字母的"機率"
        self.lm_grl = nn.Linear(config.hidden_size, config.vocab_size)           # 加了GRL那條
        
        # 加dementia model
        self.dementia_fsm = nn.Linear(config.hidden_size, config.hidden_size)    # 找出對AD預測重要的feat
        self.dementia_head = nn.Linear(config.hidden_size, 2)                    # 辨識AD
        self.dementia_grl = nn.Linear(config.hidden_size, 2)                     # 加GRL那條
        
        # define similarity loss: AM-Softmax
        self.criterion_similar = AngularPenaltySMLoss(in_features=config.hidden_size, out_features=2, loss_type='cosface').to('cpu')
        
        # freeze feature_extractor
        self.freeze_feature_encoder()

        
        if STAGE == 1:                                                  # train FSM
            print("Current stage: 1")
            self.freeze_lm_grl()
            self.freeze_dementia_grl()
            self.freeze_lm_head()
            self.freeze_dementia_head()
        elif STAGE == 2:                                                # train FSM + head
            print("Current stage: 2")
            self.freeze_lm_grl()
            self.freeze_dementia_grl()
        elif STAGE == 3:                                                # train dementia GRL
            print("Current stage: 3")
            self.freeze_sew_d()
            self.freeze_lm_fsm()
            self.freeze_dementia_fsm()
            self.freeze_lm_head()
            self.freeze_dementia_head()
            self.freeze_lm_grl()
        elif STAGE == 4:                                                # train lm GRL
            print("Current stage: 4")
            self.freeze_sew_d()
            self.freeze_lm_fsm()
            self.freeze_dementia_fsm()
            self.freeze_lm_head()
            self.freeze_dementia_head()
            self.freeze_dementia_grl()
        elif STAGE == 5:                                                # train encoder
            # train lm_FSM
            self.freeze_sew_d()
            self.freeze_dementia_fsm()            
            self.freeze_criterion_similar()
            self.freeze_lm_head()
            self.freeze_dementia_head()            
            self.freeze_lm_grl()
            self.freeze_dementia_grl()

            
        # Initialize weights and apply final processing
        self.post_init()
    
    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.sew_d.feature_extractor._freeze_parameters()

    def freeze_sew_d(self):
        self.sew_d.eval()
        for param in self.sew_d.parameters():
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
  

    @add_start_docstrings_to_model_forward(SEWD_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_PROCESSOR_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=CausalLMOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_CTC_EXPECTED_OUTPUT,
        expected_loss=_CTC_EXPECTED_LOSS,
    )
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

        outputs = self.sew_d(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]                   # last_hidden_state
        hidden_states = self.dropout(hidden_states)

        # hidden_states: sew_d embedding
        # 製造mask
        m = nn.Sigmoid()
        dementia_score = m(self.dementia_fsm(hidden_states)) # score range from 0~1
        lm_score = m(self.lm_fsm(hidden_states))             # score range from 0~1
        
        #model.recognize(inputs, input_lengths)
        dementia_mask = torch.where(dementia_score >= self.dementia_thres.to(dementia_score.device), torch.tensor(1.0).to(dementia_score.device), torch.tensor(0.0).to(dementia_score.device)) # if condition, 1. else, 0
        lm_mask = torch.where(lm_score >= self.lm_thres.to(lm_score.device), torch.tensor(1.0).to(lm_score.device), torch.tensor(0.0).to(lm_score.device))                   # if condition, 1. else, 0
        
        # 拿score vector 跟原本的hidden_states點乘
        dementia_resored = dementia_score*hidden_states
        lm_resored = lm_score*hidden_states
        
        # head(clf)
        dementia_logits = self.dementia_head(dementia_resored) #******************* torch.Size([2, 1327, 32]) (batchsize, timestep, feature_dimention)
        logits = self.lm_head(lm_resored)
        # del dementia_resored, lm_resored
        
        # grl(dis)
        hidden_states_r = ReverseLayerF.apply(hidden_states, self.alpha)
        dementia_score_r = m(self.dementia_fsm(hidden_states_r)) # score range from 0~1
        lm_score_r = m(self.lm_fsm(hidden_states_r))            # score range from 0~1

        dementia_mask_r = torch.where(dementia_score_r >= self.dementia_thres.to(dementia_score_r.device), torch.tensor(1.0).to(dementia_score_r.device), torch.tensor(0.0).to(dementia_score_r.device)) # if condition, 1. else, 0
        lm_mask_r = torch.where(lm_score_r >= self.lm_thres.to(lm_score_r.device), torch.tensor(1.0).to(lm_score_r.device), torch.tensor(0.0).to(lm_score_r.device))                   # if condition, 1. else, 0
        
        del dementia_score_r, lm_score_r
        # 拿mask跟hidden_states_r點乘
        dementia_masked_r = dementia_mask_r*hidden_states_r
        lm_masked_r = lm_mask_r*hidden_states_r
        
        del hidden_states_r, dementia_mask_r, lm_mask_r
        # grl(dis)
        dementia_logits_r = self.dementia_grl(lm_masked_r) #******************* torch.Size([2, 1327, 32]) (batchsize, timestep, feature_dimention)
        logits_r = self.lm_grl(dementia_masked_r)
        del dementia_masked_r, lm_masked_r
        dementia_output_mean = torch.mean(dementia_logits,dim=1)
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
                #  /////
                # gradient reversal layers(GRL)
                loss_fn = nn.CrossEntropyLoss()
                
                dementia_loss = loss_fn(dementia_output_mean, dementia_labels)        # multi-task
                dementia_loss_rev = loss_fn(dementia_output_mean_r, dementia_labels)  # GRL
                
                # FSM att loss
                # Scorematrix = append([dementia_mask,lm_mask]) # torch.Size([2, embedding_size])
                # Att_loss = Scorematrix*Scorematrix - Identity matrix
                #Att_loss = FSMatt_loss(lm_score, dementia_score)
                Att_loss = FSMatt_loss(lm_mask, dementia_mask)
                # del lm_mask, dementia_mask
                # diversity loss: AM-Softmax
                scores = torch.cat((hidden_states * lm_score, hidden_states * dementia_score), dim=0)
                del lm_score, dementia_score
                am_labels = torch.cat((torch.zeros(len(hidden_states), dtype=torch.long), torch.ones(len(hidden_states), dtype=torch.long)), dim=0).to('cpu')
                #del hidden_states
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
                    # train lm_FSM
                    final_loss = loss + dementia_loss_rev
                    # train dementia_FSM
                # ////

        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        logits_all = {'ASR logits': logits, 'dementia logits': dementia_logits, 'hidden_states': hidden_states,
                    'lm_mask': lm_mask, 'dementia_mask': dementia_mask}

        return CausalLMOutput(
            loss=final_loss, logits=logits_all, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )

class UniSpeechSatForCTC(UniSpeechSatPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.unispeech_sat = UniSpeechSatModel(config)
        self.dropout = nn.Dropout(config.final_dropout)

        if config.vocab_size is None:
            raise ValueError(
                f"You are trying to instantiate {self.__class__} with a configuration that "
                "does not define the vocabulary size of the language model head. Please "
                "instantiate the model as follows: `UniSpeechSatForCTC.from_pretrained(..., vocab_size=vocab_size)`. "
                "or define `vocab_size` of your model's configuration."
            )
        output_hidden_size = (
            config.output_hidden_size if hasattr(config, "add_adapter") and config.add_adapter else config.hidden_size
        )

        self.alpha=torch.tensor(LAMBDA)
        self.dementia_thres = torch.tensor(AD_THRES)
        self.lm_thres = torch.tensor(LM_THRES)
        print("lambda = ", self.alpha)
        print("dementia_thres = ", self.dementia_thres)
        print("lm_thres = ", self.lm_thres)
        
        # 加lm_model
        self.lm_fsm = nn.Linear(config.hidden_size, config.hidden_size)          # 找出對lm重要的feat
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)                             # output字母的"機率"
        self.lm_grl = nn.Linear(config.hidden_size, config.vocab_size)           # 加了GRL那條
        
        # 加dementia model
        self.dementia_fsm = nn.Linear(config.hidden_size, config.hidden_size)    # 找出對AD預測重要的feat
        self.dementia_head = nn.Linear(config.hidden_size, 2)                    # 辨識AD
        self.dementia_grl = nn.Linear(config.hidden_size, 2)                     # 加GRL那條
        
        # define similarity loss: AM-Softmax
        self.criterion_similar = AngularPenaltySMLoss(in_features=config.hidden_size, out_features=2, loss_type='cosface').to('cpu')
        
        # freeze feature_extractor
        self.freeze_feature_encoder()

        if STAGE == 1:                                                  # train FSM
            print("Current stage: 1")
            self.freeze_lm_grl()
            self.freeze_dementia_grl()
            self.freeze_lm_head()
            self.freeze_dementia_head()
        elif STAGE == 2:                                                # train FSM + head
            print("Current stage: 2")
            self.freeze_lm_grl()
            self.freeze_dementia_grl()
        elif STAGE == 3:                                                # train dementia GRL
            print("Current stage: 3")
            self.freeze_unispeech_sat()
            self.freeze_lm_fsm()
            self.freeze_dementia_fsm()
            self.freeze_lm_head()
            self.freeze_dementia_head()
            self.freeze_lm_grl()
        elif STAGE == 4:                                                # train lm GRL
            print("Current stage: 4")
            self.freeze_unispeech_sat()
            self.freeze_lm_fsm()
            self.freeze_dementia_fsm()
            self.freeze_lm_head()
            self.freeze_dementia_head()
            self.freeze_dementia_grl()
        elif STAGE == 5:                                                # train encoder
            # train lm_FSM
            self.freeze_unispeech_sat()
            self.freeze_dementia_fsm()            
            self.freeze_criterion_similar()
            self.freeze_lm_head()
            self.freeze_dementia_head()            
            self.freeze_lm_grl()
            self.freeze_dementia_grl()
         
        # Initialize weights and apply final processing
        self.post_init()

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.unispeech_sat.feature_extractor._freeze_parameters()

    def freeze_unispeech_sat(self):
        self.unispeech_sat.eval()
        for param in self.unispeech_sat.parameters():
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

    @add_start_docstrings_to_model_forward(UNISPEECH_SAT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_PROCESSOR_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=CausalLMOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_CTC_EXPECTED_OUTPUT,
        expected_loss=_CTC_EXPECTED_LOSS,
    )
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

        outputs = self.unispeech_sat(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]                   # last_hidden_state
        hidden_states = self.dropout(hidden_states)

        # hidden_states: unispeech_sat embedding
        # 製造mask
        m = nn.Sigmoid()
        dementia_score = m(self.dementia_fsm(hidden_states)) # score range from 0~1
        lm_score = m(self.lm_fsm(hidden_states))             # score range from 0~1
        
        #model.recognize(inputs, input_lengths)
        dementia_mask = torch.where(dementia_score >= self.dementia_thres.to(dementia_score.device), torch.tensor(1.0).to(dementia_score.device), torch.tensor(0.0).to(dementia_score.device)) # if condition, 1. else, 0
        lm_mask = torch.where(lm_score >= self.lm_thres.to(lm_score.device), torch.tensor(1.0).to(lm_score.device), torch.tensor(0.0).to(lm_score.device))                   # if condition, 1. else, 0
        
        # 拿score vector 跟原本的hidden_states點乘
        dementia_resored = dementia_score*hidden_states
        lm_resored = lm_score*hidden_states
        
        # head(clf)
        dementia_logits = self.dementia_head(dementia_resored) #******************* torch.Size([2, 1327, 32]) (batchsize, timestep, feature_dimention)
        logits = self.lm_head(lm_resored)
        # del dementia_resored, lm_resored
        
        # grl(dis)
        hidden_states_r = ReverseLayerF.apply(hidden_states, self.alpha)
        dementia_score_r = m(self.dementia_fsm(hidden_states_r)) # score range from 0~1
        lm_score_r = m(self.lm_fsm(hidden_states_r))            # score range from 0~1

        dementia_mask_r = torch.where(dementia_score_r >= self.dementia_thres.to(dementia_score_r.device), torch.tensor(1.0).to(dementia_score_r.device), torch.tensor(0.0).to(dementia_score_r.device)) # if condition, 1. else, 0
        lm_mask_r = torch.where(lm_score_r >= self.lm_thres.to(lm_score_r.device), torch.tensor(1.0).to(lm_score_r.device), torch.tensor(0.0).to(lm_score_r.device))                   # if condition, 1. else, 0
        
        del dementia_score_r, lm_score_r
        # 拿mask跟hidden_states_r點乘
        dementia_masked_r = dementia_mask_r*hidden_states_r
        lm_masked_r = lm_mask_r*hidden_states_r
        
        del hidden_states_r, dementia_mask_r, lm_mask_r
        # grl(dis)
        dementia_logits_r = self.dementia_grl(lm_masked_r) #******************* torch.Size([2, 1327, 32]) (batchsize, timestep, feature_dimention)
        logits_r = self.lm_grl(dementia_masked_r)
        del dementia_masked_r, lm_masked_r
        dementia_output_mean = torch.mean(dementia_logits,dim=1)
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
                #  /////
                # gradient reversal layers(GRL)
                loss_fn = nn.CrossEntropyLoss()
                
                dementia_loss = loss_fn(dementia_output_mean, dementia_labels)        # multi-task
                dementia_loss_rev = loss_fn(dementia_output_mean_r, dementia_labels)  # GRL
                
                # FSM att loss
                # Scorematrix = append([dementia_mask,lm_mask]) # torch.Size([2, embedding_size])
                # Att_loss = Scorematrix*Scorematrix - Identity matrix
                #Att_loss = FSMatt_loss(lm_score, dementia_score)
                Att_loss = FSMatt_loss(lm_mask, dementia_mask)
                # del lm_mask, dementia_mask
                # diversity loss: AM-Softmax
                scores = torch.cat((hidden_states * lm_score, hidden_states * dementia_score), dim=0)
                del lm_score, dementia_score
                am_labels = torch.cat((torch.zeros(len(hidden_states), dtype=torch.long), torch.ones(len(hidden_states), dtype=torch.long)), dim=0).to('cpu')
                #del hidden_states
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
                    # train lm_FSM
                    final_loss = loss + dementia_loss_rev
                # ////
        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        logits_all = {'ASR logits': logits, 'dementia logits': dementia_logits, 'hidden_states': hidden_states,
                    'lm_mask': lm_mask, 'dementia_mask': dementia_mask}

        return CausalLMOutput(
            loss=final_loss, logits=logits_all, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )
