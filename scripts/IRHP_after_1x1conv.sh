otuput_dir=IRHP_after_1x1conv
mkdir -p results/$otuput_dir

python IRHP_after_distillation.py \
--output_dir results/$otuput_dir \
> results/$otuput_dir/log.txt 2>&1