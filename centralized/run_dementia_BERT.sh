#!/bin/bash

# 讀取用戶輸入的 stage 變數
parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --stage)
                stage="$2"
                shift 2
                ;;
            *)
                echo "未知的選項: $1"
                exit 1
                ;;
        esac
    done
}

set -x
set -e
# 解析命令行參數
parse_args "$@"


# `:-`：是默认值设置的运算符。
stage="${stage:-0}"

audioInfile_root=./saves/results
# infile=(data2vec-audio-large-960h_test.csv data2vec-audio-large-960h_train.csv)
infile=(data2vec-audio-large-960h_train.csv data2vec-audio-large-960h_dev.csv data2vec-audio-large-960h_test.csv)
# infile=(data2vec-audio-large-960h_train.csv)

if [ "$stage" -le 0 ]; then
    for inp in ${infile[@]};do 
        python Extract_Session_text.py --input_file ${audioInfile_root}/$inp; 
    done
fi


a_mdl=(en gr multi wv)
t_mdl=(mbert_sentence xlm_sentence)
# t_mdl=(xlm)

# 試functionality 用的
# CUDA_VISIBLE_DEVICES=0,1,2,3 python 0207_DM_SentenceLvlmulti.py --gpu 0 --t_embed mbert --a_embed en
# CUDA_VISIBLE_DEVICES=0,1,2,3 python 0207_DM_SentenceLvltext.py --gpu 0 --t_embed mbert --a_embed en
# CUDA_VISIBLE_DEVICES=0,1,2,3 python 0207_DM_SessionLvltext.py --gpu 0 --t_embed mbert --a_embed en
# CUDA_VISIBLE_DEVICES=0,1,2,3 python 0207_DM_SessionLvlSummary.py --gpu 0 --t_embed mbert --a_embed en
# CUDA_VISIBLE_DEVICES=0,1,2,3 python 0207_DM_SessionLvltextnSummary.py --gpu 0 --t_embed mbert --a_embed en

if [ "$stage" -le 1 ]; then
    # test_mdls=(en gr multi wv)
    test_mdls=(en gr multi wv mbert_sentence xlm_sentence)
    # test_mdls=(gr wv mbert_sentence xlm_sentence)
    for inpm in "${test_mdls[@]}"; do
        CUDA_VISIBLE_DEVICES=0,1,2,3 python 0207_DM_SentenceLvl1input.py --inp_embed "$inpm" --epochs 5
        pids+=($!)
    done
fi

# if [ "$stage" -le 1 ]; then
#     for am in ${a_mdl[@]};do
#         for tm in ${t_mdl[@]};do
#             CUDA_VISIBLE_DEVICES=0,1,2,3 python 0207_DM_SentenceLvlmulti.py --gpu 0 --t_embed $tm --a_embed $am
#         done
#     done
# fi
# LexicalEmbedding_root=/mnt/External/Seagate/FedASR/LLaMa2/dacs/EmbFeats/Lexical/Embeddings
# if [ "$stage" -le 2 ]; then
#     am=en # whatever ~ not using audio model in this stage
#     for tm in ${t_mdl[@]};do 
#         CUDA_VISIBLE_DEVICES=0,1,2,3 python 0207_DM_SentenceLvltext.py --gpu 0 --t_embed $tm --a_embed $am
#         CUDA_VISIBLE_DEVICES=0,1,2,3 python 0207_DM_SessionLvltext.py --gpu 0 --t_embed $tm --a_embed $am
#         CUDA_VISIBLE_DEVICES=0,1,2,3 python 0207_DM_SessionLvlSummary.py --gpu 0 --t_embed $tm --a_embed $am
#         CUDA_VISIBLE_DEVICES=0,1,2,3 python 0207_DM_SessionLvltextnSummary.py --gpu 0 --t_embed $tm --a_embed $am
#     done
#     # python pred_AD_svm.py --mode 'text' --train_text $base_train_file --test_text $base_test_file
# fi

# if [ "$stage" -le 3 ]; then
#     am=en # whatever ~ not using audio model in this stage
#     for tm in ${t_mdl[@]};do 
#         CUDA_VISIBLE_DEVICES=0,1,2,3 python 0207_DM_SessionLvlSimilarityEmb.py --gpu 0 --t_embed $tm --a_embed $am
#     done
# fi

# if [ "$stage" -le 4 ]; then
#     am=en # whatever ~ not using audio model in this stage
#     for tm in ${t_mdl[@]};do 
#         CUDA_VISIBLE_DEVICES=0,1,2,3 python 0207_DM_SentenceLvlRAGSummary.py --gpu 0 --t_embed $tm --a_embed $am
#     done
# fi
# Datasource=/mnt/External/Seagate/FedASR/LLaMa2/dacs/EmbFeats/Lexical/Embeddings/text_data2vec-audio-large-960h_Phych-anomia
# Out_dir=/mnt/External/Seagate/FedASR/LLaMa2/dacs/EmbFeats/Lexical/Augment_data/
# if [ "$stage" -le 5 ]; then
#     python 0207_DM_Extact_dataAugmentation.py --Aug_k 5 --summary_dir_in $Datasource --Outpath_root $Out_dir
# fi

# if [ "$stage" -le 6 ]; then
#     am=en # whatever ~ not using audio model in this stage
#     for tm in ${t_mdl[@]};do 
#         CUDA_VISIBLE_DEVICES=0,1,2,3 python 0207_DM_SessionLvltext_aug.py --gpu 0 --t_embed $tm --a_embed $am
#     done
# fi