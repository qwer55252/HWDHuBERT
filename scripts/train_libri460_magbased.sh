EXP_NAME=libri460_magbased

CUDA_VISIBLE_DEVICES=2 python train_IRHP_libri.py \
--per_device_train_batch_size 4 \
--output_dir outputs/$EXP_NAME \
--method "magnitude-based" \
--dataset "librispeech" \
--data_config_name "train_460" \
--data_train_split "train.clean.100+train.clean.360" \