EXP_NAME=tedlium_magbased

python train_IRHP_tedlium.py \
--per_device_train_batch_size 8 \
--output_dir outputs/${EXP_NAME} \
--method "magnitude-based" \
--dataset "tedlium"