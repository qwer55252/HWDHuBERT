python libri_eval.py \
--model_name_or_path "facebook/wav2vec2-base-100h" \
--model_checkpoint /home/kobie/workspace/HWDHuBERT/OAR_libri_clean460_3090/checkpoint-1590672 \
--output_dir Eval_OAR_clean460/libri_eval_testclean \
--test_split "test.clean"

python libri_eval.py \
--model_name_or_path "facebook/wav2vec2-base-100h" \
--model_checkpoint /home/kobie/workspace/HWDHuBERT/OAR_libri_clean460_3090/checkpoint-1590672 \
--output_dir Eval_OAR_clean460/libri_eval_testother \
--test_split "test.other"

python libri_eval.py \
--model_name_or_path "facebook/wav2vec2-base-100h" \
--model_checkpoint /home/kobie/workspace/HWDHuBERT/OAR_libri_clean460_3090/checkpoint-1590672 \
--output_dir Eval_OAR_clean460/libri_eval_valclean \
--test_split "validation.clean"

python libri_eval.py \
--model_name_or_path "facebook/wav2vec2-base-100h" \
--model_checkpoint /home/kobie/workspace/HWDHuBERT/OAR_libri_clean460_3090/checkpoint-1590672 \
--output_dir Eval_OAR_clean460/libri_eval_valother \
--test_split "validation.other"