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

errors_file="errors.log"

# Remove existing errors file
rm -f "$errors_file"
# if [ "$stage" -le 0 ]; then
#     # test_mdls=(en gr multi wv)
#     # test_mdls=(en gr multi wv mbert_sentence xlm_sentence)
#     test_mdls=(gr wv mbert_sentence xlm_sentence)
#     for inpm in "${test_mdls[@]}"; do
#         CUDA_VISIBLE_DEVICES=0,1,2,3 python 0207_DM_SentenceLvl1input.py --inp_embed "$inpm" --epochs 1 &
#         pids+=($!)
#     done
# fi
if [ "$stage" -le 1 ]; then
    # test_mdls1=(en gr multi wv)
    # test_mdls2=(mbert_sentence xlm_sentence)
    test_mdls1=(en)
    test_mdls2=(mbert_sentence)
    for inpm1 in "${test_mdls1[@]}"; do
        for inpm2 in "${test_mdls2[@]}"; do
            CUDA_VISIBLE_DEVICES=0,1,2,3 python 0207_DM_SentenceLvl2inputHomogeneous.py --inp1_embed "$inpm1" --inp2_embed "$inpm2" --epochs 1 &
              pids+=($!)
        done
    done
fi
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