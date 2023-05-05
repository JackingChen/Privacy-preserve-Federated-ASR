
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

        #// TODO model改成可以預測 N-best hypotheses的形式
        logits=model(input_values).logits  
        asr_lg = logits['ASR logits']
        # 轉換length的度量從sample到output的timestep
        ratio=max(lengths)/asr_lg.shape[1]  # (batchsize, seqlength, logitsize)
        oupLens=[int(l/ratio) for l in lengths]
        # for l in lengths:
        #     oupLens.append(int(l/ratio))
        pred_ids = torch.argmax(asr_lg, dim=-1)
        pred_str=processor.batch_decode(pred_ids)

    df = pd.DataFrame()
    dummy=None
    for i in range(len(subset_dataset)):
        RealLength=oupLens[i]  #只要有從logits取出來的都要還原
        df2 = pd.DataFrame({'path': subset_dataset[i]["path"],                                    # to know which sample
                # 'array': str(subset_dataset[i]["array"]),
                # 'text': subset_dataset[i]["text"],
                # 'dementia_labels': subset_dataset[i]["dementia_labels"],
                # 'input_values': str(subset_dataset[i]["input_values"]),               # input of the model
                # 'labels': str(subset_dataset[i]["labels"]),
                # 'ASR logits': str(logits["ASR logits"][i].tolist()),
                # 'hidden_states': str(logits["hidden_states"][i].tolist()), #原本的hidden state架構
                # 'hidden_states': [logits["hidden_states"][i][:RealLength,:].cpu().numpy()],  #(time-step,node_dimension)
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


class ASRLocalUpdate(object):
    def update_weights_adapted(self, global_weights, global_round):
        # 從server端更新seed model
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
        global processor
        processor = self.processor                                                                            
        # TODO: 訓練再沒有label的資料上    (self-supervise)
        training_args = TrainingArguments()
        Multitaskmodel= Data2VecAudioForCTCMultitask()
        result= Extract_Emb(test_data, GPU_batchsize)
        labels=result["pred_str"]
        trainer = CustomTrainer(model=Multitaskmodel)
        trainer.train()
        self_trainMdl=get_model_from_trainer()
        # ****************************************************

        # 訓練再有label的資料上                                                                    
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
