export HF_DATASETS_CACHE="/root/.cache/huggingface/datasets"
export PRJ_NAME="conformerlarge"
export EXP_NAME="conformerlarge_tedlium_baseline"

# 1) 출력 디렉토리 생성
OUTPUT_DIR="./outputs/$EXP_NAME"
mkdir -p "$OUTPUT_DIR"

# 2) 학습 실행 및 로그 저장
CUDA_VISIBLE_DEVICES=0 python train_conformer_IRHP_libri.py \
--output_dir "$OUTPUT_DIR" \
--data_config_name train_100 \
--data_train_split train.clean.100 \
--data_val_split dev.clean \
--data_test_split test.clean \
--batch_size 4 \
--iterative_finetune_epochs 4 \
--final_finetune_epochs 100 \
--method "baseline" \
--dataset_name tedlium \
--model_name_or_path stt_en_conformer_ctc_large \
