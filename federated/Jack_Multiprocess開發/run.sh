#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1,2 python src/federated_main.py --model=data2vec --dataset=adress --gpu=1 --pretrain_name "facebook/data2vec-audio-large-960h"\
   -model_out "./save/data2vec-audio-large-960h_new1_recall" -log "data2vec-audio-large-960h_new1_recall_FL.txt" \
   --AD_loss "recall" --frac=1.0  \
   --local_ep 5 --epochs=2 --num_users=2 \
   --FL_STAGE 1 
    # 用FL_STAGE取代STAGE
    
    #-csv "data2vec-audio-large-960h_new1_recall" (need to be checked)
    #-model_in "/mnt/Internal/FedASR/weitung/HuggingFace/Pretrain/saves/data2vec-audio-large-960h_new1_recall/final/" \

pCUDA_VISIBLE_DEVICES=0,1,2 python src/federated_main.py --model=data2vec --dataset=adress --gpu=1 --pretrain_name "facebook/data2vec-audio-large-960h"\
    -model_in "./save/data2vec-audio-large-960h_new1_recall" \
    -model_out "./save/data2vec-audio-large-960h_new2_recall" -log "data2vec-audio-large-960h_new2_recall_FL.txt" \
    --AD_loss "recall" --frac=1.0  \
    --local_ep 5 --epochs=2 --num_users=2 \
    --FL_STAGE 2 # 用FL_STAGE取代STAGE
    # -csv "data2vec-audio-large-960h_new2_recall_FL" (need to be checked)

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
