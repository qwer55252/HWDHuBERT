
python libri_eval.py \
--model_name_or_path "facebook/wav2vec2-base-100h" \
--model_checkpoint /home/kobie/workspace/HWDHuBERT/IHP_layertokeep_pr05/checkpoint-356750 \
--output_dir libri_eval_testclean_IRHP_pr05 \
--test_split "test.clean"

python libri_eval.py \
--model_name_or_path "facebook/wav2vec2-base-100h" \
--model_checkpoint /home/kobie/workspace/HWDHuBERT/IHP_layertokeep_pr05/checkpoint-356750 \
--output_dir libri_eval_testother_IRHP_pr05 \
--test_split "test.other"

python libri_eval.py \
--model_name_or_path "facebook/wav2vec2-base-100h" \
--model_checkpoint /home/kobie/workspace/HWDHuBERT/IHP_layertokeep_pr05/checkpoint-356750 \
--output_dir libri_eval_valclean_IRHP_pr05 \
--test_split "validation.clean"

python libri_eval.py \
--model_name_or_path "facebook/wav2vec2-base-100h" \
--model_checkpoint /home/kobie/workspace/HWDHuBERT/IHP_layertokeep_pr05/checkpoint-356750 \
--output_dir libri_eval_valother_IRHP_pr05 \
--test_split "validation.other"