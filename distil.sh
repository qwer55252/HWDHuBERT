# python distil.py \
# --num_train_epochs 50 \
# --repo_name Optimizer_AdamW \
# --temperature 3.0


# Train + Evaluate
python run_speech_recognition_ctc.py \
--num_train_epochs 50 \
--learning_rate 0.001 \
--temperature 3.0 \
--model_name_or_path facebook/wav2vec2-base-100h \
--dataset_name Sreyan88/librispeech_asr \
--output_dir lr0_001 \
--overwrite_output_dir True \
--hidden_size 432 \
--intermediate_size 976 \
--do_train True \
--do_eval True


# # Evaluate
# python run_speech_recognition_ctc.py \
# --num_train_epochs 50 \
# --temperature 3.0 \
# --model_name_or_path /home/kobie/workspace/practice/hyperparameter_tuning \
# --evaluate_checkpoint /home/kobie/workspace/practice/hyperparameter_tuning/checkpoint-178400 \
# --dataset_name Sreyan88/librispeech_asr \
# --output_dir hyperparameter_tuning \
# --hidden_size 432 \
# --intermediate_size 976 \
# --do_train False \
# --do_eval True