# python distil.py \
# --num_train_epochs 50 \
# --repo_name Optimizer_AdamW \
# --temperature 3.0


# Train + Evaluate
python run_speech_recognition_ctc.py \
--num_train_epochs 100 \
--learning_rate 0.0001 \
--temperature 3.0 \
--save_steps 5000 \
--model_name_or_path facebook/wav2vec2-base-100h \
--data_dir /home/kobie/workspace/data/LibriSpeech \
--split_name default \
--train_split_name train \
--output_dir train_360 \
--overwrite_output_dir True \
--hidden_size 768 \
--intermediate_size 3072 \
--do_train True \
--do_eval True
# --dataset_name openslr/librispeech_asr \
# --alpha_mse 0.5343
 # hidden_state distillation


# # Evaluate
# python run_speech_recognition_ctc.py \
# --num_train_epochs 50 \
# --temperature 3.0 \
# --model_name_or_path /home/kobie/workspace/HWDHuBERT/hyperparameter_tuning_2 \
# --evaluate_checkpoint /home/kobie/workspace/HWDHuBERT/hyperparameter_tuning_2/checkpoint-178400 \
# --dataset_name Sreyan88/librispeech_asr \
# --output_dir hyperparameter_tuning_2 \
# --hidden_size 768 \
# --intermediate_size 3072 \
# --do_train False \
# --do_eval True