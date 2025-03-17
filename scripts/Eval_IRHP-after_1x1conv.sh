
python libri_eval.py \
--model_name_or_path "facebook/wav2vec2-base-100h" \
--model_checkpoint /home/kobie/workspace/HWDHuBERT/results/IRHP_after_1x1conv/checkpoint-3313850 \
--output_dir result_eval/Eval_IRHP-after_1x1conv \
--test_split "test.clean"

python libri_eval.py \
--model_name_or_path "facebook/wav2vec2-base-100h" \
--model_checkpoint /home/kobie/workspace/HWDHuBERT/results/IRHP_after_1x1conv/checkpoint-3313850 \
--output_dir result_eval/Eval_IRHP-after_1x1conv \
--test_split "test.other"

python libri_eval.py \
--model_name_or_path "facebook/wav2vec2-base-100h" \
--model_checkpoint /home/kobie/workspace/HWDHuBERT/results/IRHP_after_1x1conv/checkpoint-3313850 \
--output_dir result_eval/Eval_IRHP-after_1x1conv \
--test_split "validation.clean"

python libri_eval.py \
--model_name_or_path "facebook/wav2vec2-base-100h" \
--model_checkpoint /home/kobie/workspace/HWDHuBERT/results/IRHP_after_1x1conv/checkpoint-3313850 \
--output_dir result_eval/Eval_IRHP-after_1x1conv \
--test_split "validation.other"