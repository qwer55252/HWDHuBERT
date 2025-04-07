python tedlium_eval.py \
--model_name_or_path "facebook/wav2vec2-base-100h" \
--model_checkpoint "/home/kobie/workspace/HWDHuBERT/results/IHP_init_param/checkpoint-356750" \
--output_dir result_eval/tedlium_Eval/tedlium_eval_test_IRHP \
--test_split "test"

python tedlium_eval.py \
--model_name_or_path "facebook/wav2vec2-base-100h" \
--model_checkpoint "/home/kobie/workspace/HWDHuBERT/results/IHP_init_param/checkpoint-356750" \
--output_dir result_eval/tedlium_Eval/tedlium_eval_val_IRHP \
--test_split "validation"