import torch
import torch.nn as nn
import torch.nn.functional as F

class Data2VecAudioForCTCMultitask(Data2VecAudioForCTC):
    def __init__(self, config, args, num_lms=2):
        super().__init__(config, args)

        # Define multiple lm_heads
        self.lm_heads = nn.ModuleList([nn.Linear(config.hidden_size, config.vocab_size) for _ in range(num_lms)])
    def MaskProduce(self, hidden_states):
            ###################
            # 製造mask  
            # Input: hidden_states 
            # Output: lm_mask, AD_mask
            ###################
            """
            m = nn.Sigmoid()
            lm_score = m(self.lm_fsm(hidden_states))             # score range from 0~1
            lm_mask = torch.where(lm_score >= self.lm_thres.to(lm_score.device), torch.tensor(1.0).to(lm_score.device), torch.tensor(0.0).to(lm_score.device))                   # if condition, 1. else, 0
            lm_mask = lm_mask + 0 * self.lm_fsm(lm_mask) # to has grad?
            """
            all_score = self.arbitrator(hidden_states)
            """
            all_mask = torch.where(all_score >= self.lm_thres.to(all_score.device), torch.tensor(1.0).to(all_score.device), torch.tensor(0.0).to(all_score.device))                   # if condition, 1. else, 0
            all_mask = all_mask + 0 * self.arbitrator(hidden_states) # to have grad?  
            """
            # use Gunbel softmax
            lm_score = torch.stack((all_score[:, :, :self.config.hidden_size] , all_score[:, :, self.config.hidden_size:self.config.hidden_size*2]), -1)     # first part for lm, size = [batch_size, time-step, hidden_state, 2]
            AD_score = torch.stack((all_score[:, :, self.config.hidden_size*2:self.config.hidden_size*3] , all_score[:, :, self.config.hidden_size*3:]), -1) # second part for AD, size = [batch_size, time-step, hidden_state, 2]
            # toggle ratio
            if self.TOGGLE_RATIO != 0:                                                           # if toggle ratio is set
                # lm_score
                y0 = lm_score[:, :, :, 0]                                                   # target vector
                y1 = lm_score[:, :, :, 1]                                                   # another vector
                lm_score[:, :, :, 0] = (y1 - y0) * self.TOGGLE_RATIO + y0                        # replace target vector
                # AD_score
                y0 = AD_score[:, :, :, 0]                                                   # target vector
                y1 = AD_score[:, :, :, 1]                                                   # another vector
                AD_score[:, :, :, 0] = (y1 - y0) * self.TOGGLE_RATIO + y0                        # replace target vector      
            # go through GS to form mask
            #lm_mask = torch.nn.functional.gumbel_softmax(lm_score, hard=True, dim=-1)[:, :, :, 0] # back to [batch_size, time-step, hidden_state]
            lm_mask = gumbel_softmax(lm_score, tau=self.GS_TAU, hard=True, dim=-1)[:, :, :, 0]
            #AD_mask = torch.nn.functional.gumbel_softmax(AD_score, hard=True, dim=-1)[:, :, :, 0] # back to [batch_size, time-step, hidden_state]
            AD_mask = gumbel_softmax(AD_score, tau=self.GS_TAU, hard=True, dim=-1)[:, :, :, 0]
            return lm_mask, AD_mask
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
        # 以上就是encoder的部分後面就是把encoder出來的hiddenstate拿來分配給後面的layer

        # 沒過FSM，用來單獨train AD classifier
        dementia_logits_unmask = self.dementia_head(hidden_states) # for stage 1 training

        lm_mask, AD_mask=self.MaskProduce(hidden_states)
        ##############
        # // TODO: head(clf)
        # !!!!!!!!!!!!!!!!!  這一塊用for迴圈讓每個self.lm_head 都input同個hidden_states, output不同的logits
        # hint: #!!!!!
        # 像這樣處裡
        # logits_list = []
        # for lm_head in self.lm_heads:
        #     logits = lm_head(hidden_states)
        #     logits_list.append(logits)
        
        # Input: hidden_states
        # output: logits, -> list[array]  還有剩下等等的
        ##############
        


        logits_unmask = self.lm_head(hidden_states)
        """
        lm_masked = lm_mask*hidden_states
        """
        lm_masked = lm_mask*hidden_states
        AD_masked = AD_mask*hidden_states
                                                 # for fine-tune ASR
        logits = self.lm_head(lm_masked)                                                    # ASR loss
        dementia_logits = self.dementia_head(lm_masked)                                     # for AD GRL
        
        dementia_output_mean_2r = torch.mean(dementia_logits,dim=1)
        dementia_output_mean_r = ReverseLayerF.apply(dementia_output_mean_2r, self.alpha)   # for AD GRL
        dementia_output_mean_unmask = torch.mean(dementia_logits_unmask,dim=1)              # unmask

        logits_r = self.lm_head(AD_masked)                                                  # for ASR GRL
        dementia_logits = self.dementia_head(AD_masked)                                     # for AD classifier
        dementia_output_mean = torch.mean(dementia_logits,dim=1)
        #*******************
        
        if labels is not None:
            r"""
            labels (`torch.LongTensor` of shape `(batch_size, target_length)`, *optional*):
                Labels for connectionist temporal classification. Note that `target_length` has to be smaller or equal to
                the sequence length of the output logits. Indices are selected in `[-100, 0, ..., config.vocab_size - 1]`.
                All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
                config.vocab_size - 1]`.
            """

            # // TODO:  這邊要修改一下
            #!!!!!!!!!!!       改成for label 做以下事情，因為現在labels有好幾個
            if labels.max() >= self.config.vocab_size:
                raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")

            # retrieve loss input_lengths from attention_mask
            attention_mask = (
                attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
            )
            input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)

            # assuming that padded tokens are filled with -100
            # when not being attended to
            # // TODO:
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            # ctc_loss doesn't support fp16
            # // TODO:
            log_probs_unmask = nn.functional.log_softmax(logits_unmask, dim=-1, dtype=torch.float32).transpose(0, 1)
            log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)
            log_probs_r = nn.functional.log_softmax(logits_r, dim=-1, dtype=torch.float32).transpose(0, 1) # logit轉prob
            log_probs_r = ReverseLayerF.apply(log_probs_r, self.alpha) # ASR-GRL
            
            with torch.backends.cudnn.flags(enabled=False):
                loss_unmask = nn.functional.ctc_loss(
                    log_probs_unmask,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )
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
                
                if self.AD_loss == "cel":
                    print("loss: cel")
                    loss_fn = nn.CrossEntropyLoss()
                    
                    dementia_loss = loss_fn(dementia_output_mean, dementia_labels)                      # AD classifier
                    dementia_loss_unmask = loss_fn(dementia_output_mean_unmask, dementia_labels)        # unmask
                    dementia_loss_rev = loss_fn(dementia_output_mean_r, dementia_labels)                # reverse
                elif self.AD_loss == "recall":                 
                    #print("loss: recall")
                    loss_fn = RecallLoss(weight=self.W_LOSS)                                                 # W_LOSS=[w_HC, w_AD]
                    #loss = criterion(y_predict, y_target)
                    # predict: [N, C, *]    ; target: [N, *]
                    dementia_loss = loss_fn(dementia_output_mean, dementia_labels, self.AD_loss)                      # AD classifier: [batch_size, 2], [batch_size,]
                    dementia_loss_unmask = loss_fn(dementia_output_mean_unmask, dementia_labels, self.AD_loss)        # unmask: [batch_size, 2], [batch_size,]
                    #print("dementia_output_mean_unmask: ", dementia_output_mean_unmask)
                    #print("dementia_labels: ", dementia_labels)
                    #print("dementia_loss: ", dementia_loss_unmask)
                    
                    dementia_loss_rev = loss_fn(dementia_output_mean_r, dementia_labels, self.AD_loss)                # reverse: [batch_size, 2], [batch_size,]
                elif self.AD_loss == "prec":                 
                    #print("loss: precision")
                    loss_fn = RecallLoss(weight=[0.1, 0.9])                                                      # emphasize on AD PAR
                    
                    dementia_loss = loss_fn(dementia_output_mean, dementia_labels, self.AD_loss)                      # AD classifier
                    dementia_loss_unmask = loss_fn(dementia_output_mean_unmask, dementia_labels, self.AD_loss)        # unmask
                    dementia_loss_rev = loss_fn(dementia_output_mean_r, dementia_labels, self.AD_loss)                # reverse
                elif self.AD_loss == "f1":
                    #print("loss: f1")
                    loss_fn = RecallLoss(weight=[0.5, 0.5])
                    
                    dementia_loss = loss_fn(dementia_output_mean, dementia_labels, self.AD_loss)                      # AD classifier
                    dementia_loss_unmask = loss_fn(dementia_output_mean_unmask, dementia_labels, self.AD_loss)        # unmask
                    dementia_loss_rev = loss_fn(dementia_output_mean_r, dementia_labels, self.AD_loss)                # reverse         
                elif self.AD_loss == "prec_ori":     
                    #print("loss: prec_ori")
                    loss_fn = RecallLoss(weight=[0.5, 0.5])
                    
                    dementia_loss = loss_fn(dementia_output_mean, dementia_labels, self.AD_loss)                      # AD classifier
                    dementia_loss_unmask = loss_fn(dementia_output_mean_unmask, dementia_labels, self.AD_loss)        # unmask
                    dementia_loss_rev = loss_fn(dementia_output_mean_r, dementia_labels, self.AD_loss)                # reverse     
                elif self.AD_loss == "recall_ori":     
                    #print("loss: recall_ori")
                    loss_fn = RecallLoss(weight=[0.5, 0.5])
                    
                    dementia_loss = loss_fn(dementia_output_mean, dementia_labels, self.AD_loss)                      # AD classifier
                    dementia_loss_unmask = loss_fn(dementia_output_mean_unmask, dementia_labels, self.AD_loss)        # unmask
                    dementia_loss_rev = loss_fn(dementia_output_mean_r, dementia_labels, self.AD_loss)                # reverse     

                # // TODO:  這邊lm_masked 跟AD_masked似乎跟上面有重複 所以留一個就好
                lm_masked = hidden_states * lm_mask
                AD_masked = hidden_states * AD_mask
                lm_masked = torch.reshape(lm_masked, (lm_masked.size()[0]*lm_masked.size()[1], lm_masked.size()[2])) # batch_size*time-step, hidden_size
                AD_masked = torch.reshape(AD_masked, (AD_masked.size()[0]*AD_masked.size()[1], AD_masked.size()[2])) # batch_size*time-step, hidden_size

                # // TODO: am_labels改成跟lm_masked的第一個element (1-best hypotheses) 來算
                scores = torch.cat((lm_masked, AD_masked), dim=0) # batch_size*time-step * 2, hidden_size
                am_labels = torch.cat((torch.zeros(len(lm_masked), dtype=torch.long), torch.ones(len(AD_masked), dtype=torch.long)), dim=0).to('cpu') # batch_size*time-step * 2

                # should feed x: [batch_size, hidden_size] & labels: [batch_size] simply use num, no need to one-hot
                similarity, _ = self.criterion_similar(scores, am_labels)
                score_loss = similarity # * args.w_score if args.w_score > 0. else torch.tensor(0.).to(device)

                if self.STAGE == 0:                                                     # fine-tune ASR
                    final_loss = loss_unmask
                elif self.STAGE == 1:                                                  # train AD classifier
                    #print("Current stage: 1")
                    final_loss = dementia_loss_unmask
                    #print("final loss: ", final_loss)
                elif self.STAGE == 2:                                                # train toggle network
                    #print("Current stage: 2")
                    final_loss = loss + dementia_loss_rev + loss_r + dementia_loss + score_loss #+ Att_loss #+ score_loss
                    #print(loss, dementia_loss_rev, loss_r, dementia_loss, l2_lambda * l2_norm)
                    #final_loss = loss + dementia_loss_rev + loss_r + dementia_loss + l2_lambda * l2_norm
                    #final_loss = l2_lambda * l2_norm

                # ////
        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]

        # // TODO: 這邊的logits用第一個head就好了，第一個head是1-best hypotheses
        return CausalLMOutput(
            loss=final_loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )

import torch
import torch.nn as nn
import torch.nn.functional as F
from models import Data2VecAudioForCTC, gumbel_softmax, ReverseLayerF, RecallLoss
from transformers.modeling_outputs import CausalLMOutput

class Data2VecAudioForCTCMultitask(Data2VecAudioForCTC):
    def __init__(self, config, args):
        super().__init__(config, args)

        # Define multiple lm_heads
        self.lm_heads = nn.ModuleList([nn.Linear(config.hidden_size, config.vocab_size) for _ in range(args.num_lms)])
        self.num_lms = args.num_lms
    def lm_heads_init(self):
        # use pre-trained lm_head's weight as initial point
        for head in self.lm_heads:
            head.load_state_dict(self.lm_head.state_dict())
    def MaskProduce(self, hidden_states):
            ###################
            # 製造mask  
            # Input: hidden_states 
            # Output: lm_mask, AD_mask
            ###################

            all_score = self.arbitrator(hidden_states)

            # use Gunbel softmax
            lm_score = torch.stack((all_score[:, :, :self.config.hidden_size] , all_score[:, :, self.config.hidden_size:self.config.hidden_size*2]), -1)        # first part for lm, size = [batch_size, time-step, hidden_state, 2]
            AD_score = torch.stack((all_score[:, :, self.config.hidden_size*2:self.config.hidden_size*3] , all_score[:, :, self.config.hidden_size*3:]), -1)    # second part for AD, size = [batch_size, time-step, hidden_state, 2]
            
            # toggle ratio
            if self.TOGGLE_RATIO != 0:                                                                                                                          # if toggle ratio is set
                # lm_score
                y0 = lm_score[:, :, :, 0]                                                                                                                       # target vector
                y1 = lm_score[:, :, :, 1]                                                                                                                       # another vector
                lm_score[:, :, :, 0] = (y1 - y0) * self.TOGGLE_RATIO + y0                                                                                       # replace target vector
                # AD_score
                y0 = AD_score[:, :, :, 0]                                                                                                                       # target vector
                y1 = AD_score[:, :, :, 1]                                                                                                                       # another vector
                AD_score[:, :, :, 0] = (y1 - y0) * self.TOGGLE_RATIO + y0                                                                                       # replace target vector      
            
            # go through GS to form mask
            lm_mask = gumbel_softmax(lm_score, tau=self.GS_TAU, hard=True, dim=-1)[:, :, :, 0]                                                                  # back to [batch_size, time-step, hidden_state]
            AD_mask = gumbel_softmax(AD_score, tau=self.GS_TAU, hard=True, dim=-1)[:, :, :, 0]                                                                  # back to [batch_size, time-step, hidden_state]
            return lm_mask, AD_mask
    def AD_logit2loss(self, dementia_logits, reverse, dementia_labels):
        dementia_output_mean = torch.mean(dementia_logits,dim=1)
        if reverse:
            dementia_output_mean = ReverseLayerF.apply(dementia_output_mean, self.alpha)                                                            # for AD GRL

        with torch.backends.cudnn.flags(enabled=False):
            if self.AD_loss == "cel":
                loss_fn = nn.CrossEntropyLoss()
                dementia_loss = loss_fn(dementia_output_mean, dementia_labels)
            else:                                                                                                                                   # for home-made loss function
                if (self.AD_loss == "recall") or (self.AD_loss == "prec"):                                                                          # use given weights for HC & AD
                    loss_fn = RecallLoss(weight=self.W_LOSS)                                                                                        # W_LOSS=[w_HC, w_AD]
                else:
                    loss_fn = RecallLoss(weight=[0.5, 0.5])                                                                                         # default and for F1, it's [0.5, 0.5] 
                dementia_loss = loss_fn(dementia_output_mean, dementia_labels, self.AD_loss)                                                        # [batch_size, 2], [batch_size,] 
        return dementia_loss
    def LM_logit2loss(self, logits, reverse, labels, input_values, attention_mask):
        #print(labels)
        #labels = torch.tensor(labels)

        log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)
        if reverse:
            log_probs = ReverseLayerF.apply(log_probs, self.alpha) # ASR-GRL
        
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
        return loss
    def Div_loss(self, lm_masked, AD_masked):
        with torch.backends.cudnn.flags(enabled=False):
            lm_masked = torch.reshape(lm_masked, (lm_masked.size()[0]*lm_masked.size()[1], lm_masked.size()[2]))                                    # batch_size*time-step, hidden_size
            AD_masked = torch.reshape(AD_masked, (AD_masked.size()[0]*AD_masked.size()[1], AD_masked.size()[2]))                                    # batch_size*time-step, hidden_size

            # // TODO: am_labels改成跟lm_masked的第一個element (1-best hypotheses) 來算
            scores = torch.cat((lm_masked, AD_masked), dim=0)                                                                                       # batch_size*time-step * 2, hidden_size
            am_labels = torch.cat((torch.zeros(len(lm_masked), dtype=torch.long), torch.ones(len(AD_masked), dtype=torch.long)), dim=0).to('cpu')   # batch_size*time-step * 2

            # should feed x: [batch_size, hidden_size] & labels: [batch_size] simply use num, no need to one-hot
            similarity, _ = self.criterion_similar(scores, am_labels)
            score_loss = similarity # * args.w_score if args.w_score > 0. else torch.tensor(0.).to(device)
            return score_loss
        
    def forward(
        self,
        input_values,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels_lst=None,                                                                                # labels --> list of labels
        dementia_labels=None,
        EXTRACT=False,
    ):
        #print("labels_lst: ", labels_lst
        if labels_lst is not None:
            labels_lst = labels_lst[0]
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
        # 以上就是encoder的部分，後面就是把encoder出來的hidden_state拿來分配給後面的layer

        lm_mask, AD_mask = self.MaskProduce(hidden_states)                                              # produce mask
        lm_masked = lm_mask * hidden_states
        AD_masked = AD_mask * hidden_states
        ##############
        # // TODO: head(clf)
        # !!!!!!!!!!!!!!!!!  這一塊用for迴圈讓每個self.lm_head 都input同個hidden_states, output不同的logits
        # hint: #!!!!!
        # 像這樣處裡
        # logits_list = []
        # for lm_head in self.lm_heads:
        #     logits = lm_head(hidden_states)
        #     logits_list.append(logits)
        
        # Input: hidden_states
        # output: logits, -> list[array]  還有剩下等等的
        ##############
        # AD logits分3種：unmasked, AD_masked, LM_masked
        dementia_logits_unmask = self.dementia_head(hidden_states)                                      # for stage 1 training
        dementia_logits = self.dementia_head(AD_masked)                                                 # for AD classifier
        dementia_logits_r = self.dementia_head(lm_masked)                                               # for AD GRL

        # ASR logits分3種：unmasked, AD_masked, LM_masked
        # 各有N（=num_lms）個
        unmask_logits_list = []
        logits_list = []
        grl_logits_list = []
        for lm_head in self.lm_heads:                                                                   # for each lm_head
            logits_unmask = lm_head(hidden_states)                                                      # unmasked
            logits = lm_head(lm_masked)                                                                 # ASR loss
            logits_r = lm_head(AD_masked)                                                               # for ASR GRL
            # put into list
            unmask_logits_list.append(logits_unmask)
            logits_list.append(logits)
            grl_logits_list.append(logits_r)
        #logits_unmask = self.lm_head(hidden_states) # main lm_head
        #logits = self.lm_head(lm_masked)                                                               # ASR loss
        #logits_r = self.lm_head(AD_masked)                                                             # for ASR GRL
        
        # 算loss
        final_loss = None
        if labels_lst is not None:
            #print("labels_lst is not None")
            #*******************
            # 計算loss：ASR, AD, and diversity loss
            # ASR(LM)有3種loss：unmasked, masked, and reversed (grl)
            total_loss_unmask = 0
            total_loss = 0
            total_loss_r = 0
            for i in range(self.num_lms):                                                                   # 有N個lm_head，就會有N個"ground truth" transcript
                # i-th labels對應i-th lm_head
                labels = labels_lst[i]
                if labels.numel() != 0: # labels contain more than 0 elements
                    # 3種loss分別從logits_unmask, logits, logits_r計算得出
                    # self.LM_logit2loss(self, logits, reverse, labels, input_values, attention_mask)
                    total_loss_unmask += self.LM_logit2loss(unmask_logits_list[i], 0, labels, input_values, attention_mask)
                    total_loss += self.LM_logit2loss(logits_list[i], 0, labels, input_values, attention_mask)
                    total_loss_r += self.LM_logit2loss(grl_logits_list[i], 1, labels, input_values, attention_mask)
            # take average
            total_loss_unmask = total_loss_unmask / self.num_lms
            total_loss = total_loss / self.num_lms
            total_loss_r = total_loss_r / self.num_lms
            if self.STAGE == 0:                                                                             # fine-tune ASR
                final_loss = total_loss_unmask
            elif self.STAGE == 2:                                                                           # train toggle network
                final_loss = total_loss + total_loss_r

        if dementia_labels is not None:
            # AD有3種loss：unmasked, masked, and reversed (grl)
            # 從dementia_logits_unmask, dementia_logits, and dementia_logits_r計算得出
            #self.AD_logit2loss(self, dementia_logits, reverse, dementia_labels)
            dementia_loss = self.AD_logit2loss(dementia_logits, 0, dementia_labels)                         # AD classifier
            dementia_loss_unmask = self.AD_logit2loss(dementia_logits_unmask, 0, dementia_labels)           # unmask
            dementia_loss_rev = self.AD_logit2loss(dementia_logits_r, 1, dementia_labels)                   # reverse
            if self.STAGE == 1:                                                                           # train AD classifier
                final_loss = dementia_loss_unmask
            elif self.STAGE == 2:    
                if  final_loss == None:
                    final_loss = dementia_loss_rev + dementia_loss
                else:
                    final_loss += dementia_loss_rev + dementia_loss

        # diversity loss
        loss_div = self.Div_loss(lm_masked, AD_masked)

        if self.STAGE == 2:                                                                           # train toggle network
            if  final_loss == None:
                final_loss = loss_div
            else:
                final_loss += loss_div

        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]

        # // TODO: 這邊的logits用第一個head就好了，第一個head是1-best hypotheses
        
        if EXTRACT: # return vectors that we might need
            logits_all = {'ASR logits': logits_list[0], 'dementia logits': dementia_logits, 'hidden_states': hidden_states,
                    'lm_mask': lm_mask, "dementia_mask": AD_mask}
        else:
            logits_all = logits_list[0]
        #logits = self.lm_heads[0](lm_masked)
        return CausalLMOutput(
            loss=final_loss, logits=logits_all, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )

