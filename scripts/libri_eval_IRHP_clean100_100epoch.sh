
# python libri_eval.py \
# --model_name_or_path "facebook/wav2vec2-base-100h" \
# --model_checkpoint /home/kobie/workspace/HWDHuBERT/IHP_init_param_100epochs/checkpoint-713500 \
# --output_dir Eval_IRHP_100epoch/libri_eval_testclean \
# --test_split "test.clean"

# python libri_eval.py \
# --model_name_or_path "facebook/wav2vec2-base-100h" \
# --model_checkpoint /home/kobie/workspace/HWDHuBERT/IHP_init_param_100epochs/checkpoint-713500 \
# --output_dir Eval_IRHP_100epoch/libri_eval_testother \
# --test_split "test.other"

# python libri_eval.py \
# --model_name_or_path "facebook/wav2vec2-base-100h" \
# --model_checkpoint /home/kobie/workspace/HWDHuBERT/IHP_init_param_100epochs/checkpoint-713500 \
# --output_dir Eval_IRHP_100epoch/libri_eval_valclean \
# --test_split "validation.clean"

# python libri_eval.py \
# --model_name_or_path "facebook/wav2vec2-base-100h" \
# --model_checkpoint /home/kobie/workspace/HWDHuBERT/IHP_init_param_100epochs/checkpoint-713500 \
# --output_dir Eval_IRHP_100epoch/libri_eval_valother \
# --test_split "validation.other"




# parameter initialization 하지 않은 거 같음... 위에는 한거
python libri_eval.py \
--model_name_or_path "facebook/wav2vec2-base-100h" \
--model_checkpoint /home/kobie/workspace/HWDHuBERT/IHP_100epochs/checkpoint-713500 \
--output_dir Eval_IRHP2_100epoch/libri_eval_testclean \
--test_split "test.clean"

python libri_eval.py \
--model_name_or_path "facebook/wav2vec2-base-100h" \
--model_checkpoint /home/kobie/workspace/HWDHuBERT/IHP_100epochs/checkpoint-713500 \
--output_dir Eval_IRHP2_100epoch/libri_eval_testother \
--test_split "test.other"

python libri_eval.py \
--model_name_or_path "facebook/wav2vec2-base-100h" \
--model_checkpoint /home/kobie/workspace/HWDHuBERT/IHP_100epochs/checkpoint-713500 \
--output_dir Eval_IRHP2_100epoch/libri_eval_valclean \
--test_split "validation.clean"

python libri_eval.py \
--model_name_or_path "facebook/wav2vec2-base-100h" \
--model_checkpoint /home/kobie/workspace/HWDHuBERT/IHP_100epochs/checkpoint-713500 \
--output_dir Eval_IRHP2_100epoch/libri_eval_valother \
--test_split "validation.other"

