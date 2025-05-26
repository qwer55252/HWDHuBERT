EXP_NAME=libri460_baseline

python train_baseline_libri.py \
--per_device_train_batch_size 12 \
--output_dir outputs/$EXP_NAME \
--dataset "librispeech" \
--data_config_name "train_460" \
--data_train_split "train.clean.100+train.clean.360" \