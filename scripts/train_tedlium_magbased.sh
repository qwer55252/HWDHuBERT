python train_IRHP_tedlium.py \
--per_device_train_batch_size 8 \
--output_dir outputs/tedlium_magbased \
--method "magnitude-based" \
--dataset "tedlium"