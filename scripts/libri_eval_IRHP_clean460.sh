
python libri_eval.py \
--model_name_or_path "facebook/wav2vec2-base-100h" \
--already_pruned_heads_dict "{\"0\": [4,5,7,8, 10, 11], \"1\": [4, 7], \"2\": [2, 3,4, 5,6,7, 8, 9, 10], \"3\": [5, 8, 9, 10, 11], \"4\": [2, 3, 4, 5, 6, 7, 8,9,11], \"5\": [1, 2, 3, 4, 5, 6, 7, 9, 10, 11], \"6\": [1, 2, 3, 4, 5, 6, 7, 9, 10], \"7\": [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], \"8\": [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11], \"9\": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,11], \"10\": [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11], \"11\": [0, 1, 2, 7, 8, 11]}" \
--model_checkpoint /home/kobie/workspace/HWDHuBERT/IRHP_libri_clean460/checkpoint-1656950 \
--output_dir libri_eval_testclean_IRHP \
--test_split "test.clean"

python libri_eval.py \
--model_name_or_path "facebook/wav2vec2-base-100h" \
--already_pruned_heads_dict "{\"0\": [4,5,7,8, 10, 11], \"1\": [4, 7], \"2\": [2, 3,4, 5,6,7, 8, 9, 10], \"3\": [5, 8, 9, 10, 11], \"4\": [2, 3, 4, 5, 6, 7, 8,9,11], \"5\": [1, 2, 3, 4, 5, 6, 7, 9, 10, 11], \"6\": [1, 2, 3, 4, 5, 6, 7, 9, 10], \"7\": [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], \"8\": [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11], \"9\": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,11], \"10\": [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11], \"11\": [0, 1, 2, 7, 8, 11]}" \
--model_checkpoint /home/kobie/workspace/HWDHuBERT/IRHP_libri_clean460/checkpoint-1656950 \
--output_dir libri_eval_testother_IRHP \
--test_split "test.other"

python libri_eval.py \
--model_name_or_path "facebook/wav2vec2-base-100h" \
--already_pruned_heads_dict "{\"0\": [4,5,7,8, 10, 11], \"1\": [4, 7], \"2\": [2, 3,4, 5,6,7, 8, 9, 10], \"3\": [5, 8, 9, 10, 11], \"4\": [2, 3, 4, 5, 6, 7, 8,9,11], \"5\": [1, 2, 3, 4, 5, 6, 7, 9, 10, 11], \"6\": [1, 2, 3, 4, 5, 6, 7, 9, 10], \"7\": [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], \"8\": [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11], \"9\": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,11], \"10\": [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11], \"11\": [0, 1, 2, 7, 8, 11]}" \
--model_checkpoint /home/kobie/workspace/HWDHuBERT/IRHP_libri_clean460/checkpoint-1656950 \
--output_dir libri_eval_valclean_IRHP \
--test_split "validation.clean"

python libri_eval.py \
--model_name_or_path "facebook/wav2vec2-base-100h" \
--already_pruned_heads_dict "{\"0\": [4,5,7,8, 10, 11], \"1\": [4, 7], \"2\": [2, 3,4, 5,6,7, 8, 9, 10], \"3\": [5, 8, 9, 10, 11], \"4\": [2, 3, 4, 5, 6, 7, 8,9,11], \"5\": [1, 2, 3, 4, 5, 6, 7, 9, 10, 11], \"6\": [1, 2, 3, 4, 5, 6, 7, 9, 10], \"7\": [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], \"8\": [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11], \"9\": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,11], \"10\": [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11], \"11\": [0, 1, 2, 7, 8, 11]}" \
--model_checkpoint /home/kobie/workspace/HWDHuBERT/IRHP_libri_clean460/checkpoint-1656950 \
--output_dir libri_eval_valother_IRHP \
--test_split "validation.other"