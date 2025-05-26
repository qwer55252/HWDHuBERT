EXP_NAME=libri100_redundancybased

python train_IRHP_libri.py \
--per_device_train_batch_size 4 \
--output_dir outputs/$EXP_NAME \
--method "redundancy-based" \
--dataset "librispeech" \
--data_config_name "train_100"