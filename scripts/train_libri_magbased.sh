EXP_NAME=libri_magbased

CUDA_VISIBLE_DEVICE=2 python train_IRHP_libri.py \
--per_device_train_batch_size 4 \
--output_dir outputs/$EXP_NAME \
--method "magnitude-based" \
--dataset "librispeech"