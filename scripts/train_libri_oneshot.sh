EXP_NAME=libri_oneshot

python train_oneshot_libri.py \
--per_device_train_batch_size 4 \
--output_dir outputs/$EXP_NAME \
--dataset "librispeech"