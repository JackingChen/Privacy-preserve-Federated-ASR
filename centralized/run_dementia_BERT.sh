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
device=3
audioInfile_root=./saves/results
# infile=(data2vec-audio-large-960h_test.csv data2vec-audio-large-960h_train.csv)
infile=(data2vec-audio-large-960h_train.csv data2vec-audio-large-960h_dev.csv data2vec-audio-large-960h_test.csv)
# infile=(data2vec-audio-large-960h_train.csv)

if [ "$stage" -le 0 ]; then
    for inp in ${infile[@]};do 
        python Extract_Session_text.py --input_file ${audioInfile_root}/$inp; 
    done
fi


a_mdl=(en gr)
t_mdl=(mbert_sentence xlm_sentence)
# t_mdl=(xlm)


# wv 跟 gr都會壞掉，不要跑
if [ "$stage" -le 1 ]; then
    test_mdls=(en multi mbert_sentence xlm_sentence)
    # test_mdls=(gr)
    # test_mdls=(wv)
    for inpm in "${test_mdls[@]}"; do
        CUDA_VISIBLE_DEVICES=$device python 0207_DM_SentenceLvl1input.py --inp_embed "$inpm" --epochs 5 
    done
fi


if [ "$stage" -le 2 ]; then
    # test_mdls=(mbert_session xlm_session)
    test_mdls=(xlm_session)
    for inpm in "${test_mdls[@]}"; do
        CUDA_VISIBLE_DEVICES=$device python 0207_DM_SessionLvl1input.py --inp_embed "$inpm" --epochs 5 
    done
fi

if [ "$stage" -le 3 ]; then
    test_mdls1=(en multi mbert_sentence xlm_sentence)
    # test_mdls2=(anomia)
    # test_mdls1=(en)
    test_mdls2=(anomia)
    for inpm1 in "${test_mdls1[@]}"; do
        for inpm2 in "${test_mdls2[@]}"; do
            CUDA_VISIBLE_DEVICES=$device python 0207_DM_SentenceLvl2inputHeterogeneous.py --inp1_embed "$inpm1" --inp2_embed "$inpm2" --epochs 5
        done
    done
fi

if [ "$stage" -le 4 ]; then
    test_mdls1=(en multi)
    test_mdls2=(mbert_sentence xlm_sentence)
    # test_mdls1=(en)
    # test_mdls2=(mbert_sentence)
    for inpm1 in "${test_mdls1[@]}"; do
        for inpm2 in "${test_mdls2[@]}"; do
            CUDA_VISIBLE_DEVICES=$device python 0207_DM_SentenceLvl2inputHomogeneous.py --inp1_embed "$inpm1" --inp2_embed "$inpm2" --epochs 5 
        done
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



# Wait for all background processes to finish
for pid in "${pids[@]}"; do
    wait "$pid"
    exit_code=$?
    
    if [ $exit_code -ne 0 ]; then
        command=$(ps -o cmd= -p "$pid")
        errors+=("Error: $command")
    fi
done

# Print any errors
if [ ${#errors[@]} -gt 0 ]; then
    echo "Errors occurred. Details written to $errors_file:"
    for error in "${errors[@]}"; do
        echo "$error"
    done
else
    echo "All processes completed successfully."
    rm -f "$errors_file"
fi