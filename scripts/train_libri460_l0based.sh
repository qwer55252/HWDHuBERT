EXP_NAME=libri460_l0based

CUDA_VISIBLE_DEVICES=3 python train_IRHP_libri.py \
--per_device_train_batch_size 4 \
--output_dir outputs/$EXP_NAME \
--method "l0-based" \
--dataset "librispeech" \
--data_config_name "train_460"