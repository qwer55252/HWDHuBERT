CUDA_VISIBLE_DEVICES=3 python train_baseline_tedlium.py \
--per_device_train_batch_size 12 \
--output_dir outputs/tedlium_baseline \
--dataset "tedlium"