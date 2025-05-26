EXP_NAME=libri100_baseline

python train_baseline_libri.py \
--per_device_train_batch_size 12 \
--output_dir outputs/$EXP_NAME \
--dataset "librispeech" \
--data_config_name "train_100"