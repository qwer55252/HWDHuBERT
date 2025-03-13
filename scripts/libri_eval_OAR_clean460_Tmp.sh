$model_name_or_path="facebook/wav2vec2-base-100h"
$already_pruned_heads_dict="{\"0\": [2, 3, 4, 5, 6, 7, 10, 11], \"1\": [0, 1, 2, 3, 6, 8, 9, 10, 11], \"2\": [2, 3, 4, 7, 8, 9, 10, 11], \"3\": [1, 3, 4, 5, 6, 7, 8, 9], \"4\": [0, 1, 2, 4, 6, 7, 8, 9, 10, 11], \"5\": [1, 2, 3, 4, 5, 6, 7, 8, 10, 11], \"6\": [0, 1, 2, 3, 4, 5, 6, 7, 8, 10], \"7\": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], \"8\": [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11], \"9\": [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11], \"10\": [0, 1, 3, 4, 6, 7, 8, 9, 10, 11], \"11\": [2, 3, 5, 6, 7, 8, 9, 10, 11]}"
$model_checkpoint="/home/kobie/workspace/HWDHuBERT/IHP_init_param/checkpoint-356750"


python libri_eval.py \
--model_name_or_path $model_name_or_path \
--already_pruned_heads_dict $already_pruned_heads_dict \
--model_checkpoint "/home/kobie/workspace/HWDHuBERT/IHP_init_param/checkpoint-356750" \
--output_dir libri_eval_testclean_IRHP \
--test_split "test.clean"

python libri_eval.py \
--model_name_or_path $model_name_or_path \
--already_pruned_heads_dict "$already_pruned_heads_dict \
--model_checkpoint "/home/kobie/workspace/HWDHuBERT/IHP_init_param/checkpoint-356750" \
--output_dir libri_eval_testother_IRHP \
--test_split "test.other"

python libri_eval.py \
--model_name_or_path $model_name_or_path \
--already_pruned_heads_dict $already_pruned_heads_dict \
--model_checkpoint "/home/kobie/workspace/HWDHuBERT/IHP_init_param/checkpoint-356750" \
--output_dir libri_eval_valclean_IRHP \
--test_split "validation.clean"

python libri_eval.py \
--model_name_or_path $model_name_or_path \
--already_pruned_heads_dict $already_pruned_heads_dict \
--model_checkpoint "/home/kobie/workspace/HWDHuBERT/IHP_init_param/checkpoint-356750" \
--output_dir libri_eval_valother_IRHP \
--test_split "validation.other"