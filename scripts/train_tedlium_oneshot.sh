CUDA_VISIBLE_DEVICES=2 python train_IRHP_tedlium.py \
--per_device_train_batch_size 12 \
--output_dir outputs/tedlium_oneshot \
--method "one-shot" \
--dataset "tedlium"