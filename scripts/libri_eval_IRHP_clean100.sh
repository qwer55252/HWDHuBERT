python libri_eval.py \
--model_name_or_path "facebook/wav2vec2-base-100h" \
--already_pruned_heads_dict "{\"0\": [3, 4, 5, 8, 11], \"1\": [4, 6, 10], \"2\": [10, 3, 5, 7, 8, 9], \"3\": [3, 5, 6, 8, 9, 10, 11], \"4\": [2, 3, 4, 5, 6, 7, 8, 9, 11], \"5\": [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11], \"6\": [0, 1, 2, 3, 5, 6, 7, 8, 10, 11], \"7\": [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11], \"8\": [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11], \"9\": [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11], \"10\": [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11], \"11\": [0, 2, 3, 6, 5, 11]}" \
--model_checkpoint "/home/kobie/workspace/HWDHuBERT/IHP_init_param/checkpoint-356750" \
--output_dir libri_eval_testclean_IRHP \
--test_split "test.clean"

python libri_eval.py \
--model_name_or_path "facebook/wav2vec2-base-100h" \
--already_pruned_heads_dict "{\"0\": [3, 4, 5, 8, 11], \"1\": [4, 6, 10], \"2\": [10, 3, 5, 7, 8, 9], \"3\": [3, 5, 6, 8, 9, 10, 11], \"4\": [2, 3, 4, 5, 6, 7, 8, 9, 11], \"5\": [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11], \"6\": [0, 1, 2, 3, 5, 6, 7, 8, 10, 11], \"7\": [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11], \"8\": [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11], \"9\": [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11], \"10\": [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11], \"11\": [0, 2, 3, 6, 5, 11]}" \
--model_checkpoint "/home/kobie/workspace/HWDHuBERT/IHP_init_param/checkpoint-356750" \
--output_dir libri_eval_testother_IRHP \
--test_split "test.other"

python libri_eval.py \
--model_name_or_path "facebook/wav2vec2-base-100h" \
--already_pruned_heads_dict "{\"0\": [3, 4, 5, 8, 11], \"1\": [4, 6, 10], \"2\": [10, 3, 5, 7, 8, 9], \"3\": [3, 5, 6, 8, 9, 10, 11], \"4\": [2, 3, 4, 5, 6, 7, 8, 9, 11], \"5\": [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11], \"6\": [0, 1, 2, 3, 5, 6, 7, 8, 10, 11], \"7\": [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11], \"8\": [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11], \"9\": [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11], \"10\": [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11], \"11\": [0, 2, 3, 6, 5, 11]}" \
--model_checkpoint "/home/kobie/workspace/HWDHuBERT/IHP_init_param/checkpoint-356750" \
--output_dir libri_eval_valclean_IRHP \
--test_split "validation.clean"

python libri_eval.py \
--model_name_or_path "facebook/wav2vec2-base-100h" \
--already_pruned_heads_dict "{\"0\": [3, 4, 5, 8, 11], \"1\": [4, 6, 10], \"2\": [10, 3, 5, 7, 8, 9], \"3\": [3, 5, 6, 8, 9, 10, 11], \"4\": [2, 3, 4, 5, 6, 7, 8, 9, 11], \"5\": [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11], \"6\": [0, 1, 2, 3, 5, 6, 7, 8, 10, 11], \"7\": [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11], \"8\": [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11], \"9\": [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11], \"10\": [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11], \"11\": [0, 2, 3, 6, 5, 11]}" \
--model_checkpoint "/home/kobie/workspace/HWDHuBERT/IHP_init_param/checkpoint-356750" \
--output_dir libri_eval_valother_IRHP \
--test_split "validation.other"