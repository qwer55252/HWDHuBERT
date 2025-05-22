CUDA_VISIBLE_DEVICES=3 python train_IRHP_tedlium.py \
--per_device_train_batch_size 8 \
--output_dir outputs/tedlium_l0based \
--method "l0-based" \
--dataset "tedlium"