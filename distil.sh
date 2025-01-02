# python distil.py \
# --num_train_epochs 50 \
# --repo_name Optimizer_AdamW \
# --temperature 3.0


# Train + Evaluate
python run_speech_recognition_ctc.py \
--num_train_epochs 50 \
--learning_rate 0.0001 \
--temperature 3.0 \
--save_steps 5000 \
--model_name_or_path facebook/wav2vec2-base-100h \
--dataset_name Sreyan88/librispeech_asr \
--output_dir hidden_state_loss \
--overwrite_output_dir True \
--hidden_size 432 \
--intermediate_size 976 \
--do_train True \
--do_eval True \
--alpha_cos 0.5
 # hidden_state distillation


# # Evaluate
# python run_speech_recognition_ctc.py \
# --num_train_epochs 50 \
# --temperature 3.0 \
# --model_name_or_path /home/kobie/workspace/HWDHuBERT/hidden_state_loss \
# --evaluate_checkpoint /home/kobie/workspace/HWDHuBERT/hidden_state_loss/checkpoint-178400 \
# --dataset_name Sreyan88/librispeech_asr \
# --output_dir hidden_state_loss_eval \
# --hidden_size 432 \
# --intermediate_size 976 \
# --do_train False \
# --do_eval True