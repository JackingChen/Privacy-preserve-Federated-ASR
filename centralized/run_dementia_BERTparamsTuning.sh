#!/bin/bash

# Default values
device=3
stage=0

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --stage)
                stage="$2"
                shift 2
                ;;
            --device)
                device="$2"
                shift 2
                ;;
            *)
                echo "未知的選項: $1"
                exit 1
                ;;
        esac
    done
}

# Enable debugging and exit on error
set -x
set -e

# Disable tokenizers parallelism
export TOKENIZERS_PARALLELISM=false

# Parse command line arguments
parse_args "$@"

# Rest of your script...

audioInfile_root=./saves/results
# infile=(data2vec-audio-large-960h_test.csv data2vec-audio-large-960h_train.csv)
infile=(data2vec-audio-large-960h_train.csv data2vec-audio-large-960h_dev.csv data2vec-audio-large-960h_test.csv)
# infile=(data2vec-audio-large-960h_train.csv)

if [ "$stage" -le 0 ]; then
    for inp in ${infile[@]};do 
        python Extract_Session_text.py --input_file ${audioInfile_root}/$inp; 
    done
fi



test_mdls=(albert-base-v1 mbert_sentence)
lrs=(1e-1 1e-2 1e-3 1e-4 1e-5)
patiences=(3 4 5 6 7 8 9)
lr_schedulers=(exp lin cos)
epochs=(5 10 20 30)

# wv 跟 gr都會壞掉，不要跑
if [ "$stage" -le 1 ]; then
    concurrent_processes=3
    # test_mdls=(albert-base-v1 mbert_sentence)
    # test_mdls=(albert-base-v1)
    test_mdls=(mbert_sentence)
    lrs=(1e-3 1e-4 2e-5 1e-5 1e-6 5e-7 1e-7)
    # lrs=(1e-1)
    patiences=(3 4 5 6 7 8 9)
    # patiences=(3 4)
    # lr_schedulers=(exp lin cos)
    lr_schedulers=(exp)
    # epoch=10
    epochs=(5 10 20)
    # epochs=(1)

    # Counter to keep track of the number of processes currently running
    current_processes=0
    for inpm in "${test_mdls[@]}"; do
        for lr in "${lrs[@]}"; do
            for patience in "${patiences[@]}"; do
                for lr_scheduler in "${lr_schedulers[@]}"; do
                    for epoch in "${epochs[@]}";do
                        CUDA_VISIBLE_DEVICES=$device python 0207_DM_SentenceLvl1input_ParamTuning.py --inp_embed "$inpm"  \
                                --epochs $epoch \
                                --lr $lr \
                                --lr_scheduler $lr_scheduler \
                                --patience $patience &
                        # Increment the current process counter
                        ((current_processes++))

                        # If the number of concurrent processes is reached, wait for them to finish
                        if [ $current_processes -eq $concurrent_processes ]; then
                            # Wait for all background processes to finish
                            wait
                            # Reset the current process counter
                            echo "Starting new processes"

                            current_processes=0
                        fi
                    done
                done
            done
        done
    done
    wait
fi


# if [ "$stage" -le 2 ]; then
#     # test_mdls=(mbert_session xlm_session)
#     test_mdls=(xlm_session)
#     for inpm in "${test_mdls[@]}"; do
#         CUDA_VISIBLE_DEVICES=$device python 0207_DM_SessionLvl1input.py --inp_embed "$inpm" --epochs 5 
#     done
# fi

# if [ "$stage" -le 3 ]; then
#     test_mdls1=(en multi mbert_sentence xlm_sentence)
#     # test_mdls2=(anomia)
#     # test_mdls1=(en)
#     test_mdls2=(anomia)
#     for inpm1 in "${test_mdls1[@]}"; do
#         for inpm2 in "${test_mdls2[@]}"; do
#             CUDA_VISIBLE_DEVICES=$device python 0207_DM_SentenceLvl2inputHeterogeneous.py --inp1_embed "$inpm1" --inp2_embed "$inpm2" --epochs 5
#         done
#     done
# fi

# if [ "$stage" -le 4 ]; then
#     test_mdls1=(en multi)
#     test_mdls2=(mbert_sentence xlm_sentence)
#     # test_mdls1=(en)
#     # test_mdls2=(mbert_sentence)
#     for inpm1 in "${test_mdls1[@]}"; do
#         for inpm2 in "${test_mdls2[@]}"; do
#             CUDA_VISIBLE_DEVICES=$device python 0207_DM_SentenceLvl2inputHomogeneous.py --inp1_embed "$inpm1" --inp2_embed "$inpm2" --epochs 5 
#         done
#     done
# fi
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


