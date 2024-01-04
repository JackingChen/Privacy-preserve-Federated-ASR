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
# 解析命令行參數
parse_args "$@"


# `:-`：是默认值设置的运算符。
stage="${stage:-0}"

audioInfile_root=./saves/results
# infile=(data2vec-audio-large-960h_test.csv data2vec-audio-large-960h_train.csv)
infile=(data2vec-audio-large-960h_train.csv data2vec-audio-large-960h_dev.csv)
# infile=(data2vec-audio-large-960h_train.csv)

if [ "$stage" -le 0 ]; then
    for inp in ${infile[@]};do 
        python Extract_Session_text.py --input_file ${audioInfile_root}/$inp; 
    done
fi


LexicalEmbedding_root=/mnt/External/Seagate/FedASR/LLaMa2/dacs/EmbFeats/Lexical/Embeddings
if [ "$stage" -le 1 ]; then
    text_Emb_files=$(ls ${LexicalEmbedding_root})
    for file in ${text_Emb_files[@]};do
        python pred_AD_svm.py --mode 'text' --Lexical_dataIn $LexicalEmbedding_root/$file
    done
    # python pred_AD_svm.py --mode 'text' --train_text $base_train_file --test_text $base_test_file
fi


