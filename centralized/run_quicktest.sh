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

# if [ "$stage" -le 0 ]; then
#     for inp in ${infile[@]};do 
#         python Extract_Session_text.py --input_file ${audioInfile_root}/$inp; 
#     done
# fi

# Function to run a task with a semaphore
run_with_semaphore() {
    local sem_file="$1"
    local cmd="$2"
    
    # Acquire semaphore
    read -u 3 -n 3 sem_value && ((0 == sem_value)) || exit $sem_value
    
    # Run the command asynchronously
    (
        $cmd
        # Push the return code of the command to the semaphore
        printf '%.3d' $? >&3
    )&
}

# Set the number of processes in each batch
N=4

# Initialize a semaphore with N tokens
open_semaphore() {
    mkfifo pipe-$$
    exec 3<>pipe-$$
    rm pipe-$$
    local i=$1
    for ((; i > 0; i--)); do
        printf %s 000 >&3
    done
}

# Open the semaphore
open_semaphore $N
export CUDA_VISIBLE_DEVICES=$device
if [ "$stage" -le 1 ]; then
    test_mdls=(xlm-roberta-base
        albert-base-v1
        xlnet-base-cased
        emilyalsentzer/Bio_ClinicalBERT
        dmis-lab/biobert-base-cased-v1.2
        YituTech/conv-bert-base)
    # test_mdls=(emilyalsentzer/Bio_ClinicalBERT
    #     dmis-lab/biobert-base-cased-v1.2
    #     YituTech/conv-bert-base)
    # test_mdls=(albert-base-v1)
    for inpm in "${test_mdls[@]}"; do
        # CUDA_VISIBLE_DEVICES=$device python 0207_DM_SentenceLvl1input.py --inp_embed "$inpm" --epochs 5 
        cmd="python 0207_DM_SentenceLvl1input.py --inp_embed $inpm --epochs 5 "
        run_with_semaphore 3 "$cmd"
    done
fi
exit 0;

if [ "$stage" -le 2 ]; then
    # test_mdls1=(en multi)
    # test_mdls2=(emilyalsentzer/Bio_ClinicalBERT
    #     dmis-lab/biobert-base-cased-v1.2
    #     )
    # xlm-roberta-base
    # albert-base-v1
    # xlnet-base-cased
    # YituTech/conv-bert-bases
    test_mdls1=(multi)
    test_mdls2=(xlm-roberta-base
    xlnet-base-cased
    )
    for inpm1 in "${test_mdls1[@]}"; do
        for inpm2 in "${test_mdls2[@]}"; do
            CUDA_VISIBLE_DEVICES=$device python 0207_DM_SentenceLvl2inputHomogeneous.py --inp1_embed "$inpm1" --inp2_embed "$inpm2" --epochs 5 
        done
    done
fi

# if [ "$stage" -le 3 ]; then
#     # test_mdls=(xlm-roberta-base
#     #     albert-base-v1
#     #     xlnet-base-cased
#     #     emilyalsentzer/Bio_ClinicalBERT
#     #     dmis-lab/biobert-base-cased-v1.2
#     #     YituTech/conv-bert-base)
#     # test_mdls=(emilyalsentzer/Bio_ClinicalBERT
#     #     dmis-lab/biobert-base-cased-v1.2
#     #     YituTech/conv-bert-base)
#     test_mdls=(albert-base-v1)
#     for trial in 0 1 2 3 4 5 6;do
#         for inpm in "${test_mdls[@]}"; do
#             CUDA_VISIBLE_DEVICES=$device python 0207_DM_SessionLvl1input_consistencyTest.py --inp_embed "$inpm" --epochs 5 --trial $trial
#         done
#     done
# fi