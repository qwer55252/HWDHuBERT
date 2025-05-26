EXP_NAME=tedlium_l0based

python train_IRHP_tedlium.py \
--per_device_train_batch_size 8 \
--output_dir outputs/${EXP_NAME} \
--method "l0-based" \
--dataset "tedlium"