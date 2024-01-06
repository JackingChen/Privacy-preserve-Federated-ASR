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


if [ "$stage" -le 1 ]; then
    # test_mdls=(xlm-roberta-base
    #     albert-base-v1
    #     xlnet-base-cased
    #     emilyalsentzer/Bio_ClinicalBERT
    #     dmis-lab/biobert-base-cased-v1.2
    #     YituTech/conv-bert-base)
    test_mdls=(emilyalsentzer/Bio_ClinicalBERT
        dmis-lab/biobert-base-cased-v1.2
        YituTech/conv-bert-base)
    for inpm in "${test_mdls[@]}"; do
        CUDA_VISIBLE_DEVICES=$device python 0207_DM_SentenceLvl1input.py --inp_embed "$inpm" --epochs 5 
    done
fi

if [ "$stage" -le 2 ]; then
    test_mdls1=(en multi)
    test_mdls2=(xlm-roberta-base
        albert-base-v1
        xlnet-base-cased
        emilyalsentzer/Bio_ClinicalBERT
        dmis-lab/biobert-base-cased-v1.2
        )
    # YituTech/conv-bert-bases
    # test_mdls1=(en)
    # test_mdls2=(mbert_sentence)
    for inpm1 in "${test_mdls1[@]}"; do
        for inpm2 in "${test_mdls2[@]}"; do
            CUDA_VISIBLE_DEVICES=$device python 0207_DM_SentenceLvl2inputHomogeneous.py --inp1_embed "$inpm1" --inp2_embed "$inpm2" --epochs 5 
        done
    done
fi

