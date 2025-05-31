export HF_DATASETS_CACHE="/root/.cache/huggingface/datasets"
export PRJ_NAME="conformer"
export EXP_NAME="conformer_tedlium_IRHP"

# 1) 출력 디렉토리 생성
OUTPUT_DIR="./outputs/$EXP_NAME"
mkdir -p "$OUTPUT_DIR"

# 2) 학습 실행 및 로그 저장
CUDA_VISIBLE_DEVICES=0 python train_conformer_IRHP_libri.py \
--output_dir "$OUTPUT_DIR" \
--batch_size 32 \
--epochs 100 \
--method "redundancy-based" \
--prune_ratio 0.5 \
--dataset_name "tedlium" \