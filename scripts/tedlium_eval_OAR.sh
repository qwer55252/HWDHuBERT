python tedlium_eval.py \
--model_name_or_path "facebook/wav2vec2-base-100h" \
--model_checkpoint /home/kobie/workspace/HWDHuBERT/results/OAR_implementation/checkpoint-356750 \
--output_dir result_eval/tedlium_Eval/tedlium_eval_test_OAR \
--test_split "test" > eval_tedlium_test_OAR_output.txt \

python tedlium_eval.py \
--model_name_or_path "facebook/wav2vec2-base-100h" \
--model_checkpoint /home/kobie/workspace/HWDHuBERT/results/OAR_implementation/checkpoint-356750 \
--output_dir result_eval/tedlium_Eval/tedlium_eval_val_OAR \
--test_split "validation" > eval_tedlium_val_OAR_output.txt