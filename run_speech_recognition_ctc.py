#!/usr/bin/env python
# coding=utf-8

import functools
import json
import logging
import os
import re
import sys
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import datasets
import evaluate
import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import DatasetDict, load_dataset, DownloadConfig
from evaluate import load

import transformers
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForCTC,
    AutoProcessor,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    Wav2Vec2Processor,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
import wandb

# Will error if the minimal version of Transformers is not installed.
# check_min_version("4.48.0.dev0")
require_version("datasets>=1.18.0", "To fix: pip install -r examples/pytorch/speech-recognition/requirements.txt")

logger = logging.getLogger(__name__)


def list_field(default=None, metadata=None):
    return field(default_factory=lambda: default, metadata=metadata)


# ------------------------------------------------------------------------------------
# ---- distillation changes START: new arguments for distillation
# ------------------------------------------------------------------------------------
@dataclass
class DistillationArguments:
    """Additional arguments for teacher-student distillation."""
    # Teacher
    teacher_model_name: str = field(
        default="facebook/wav2vec2-base-100h",
        metadata={"help": "Path or identifier of the teacher model (e.g. 'facebook/wav2vec2-base-100h')."}
    )
    # Distillation hyperparams
    temperature: float = field(
        default=5.0,
        metadata={"help": "Distillation temperature for softmax smoothing."}
    )
    lambda_param: float = field(
        default=0.5,
        metadata={"help": "Lambda parameter for weighting distillation vs. supervised CTC loss."}
    )
    # Overwrite certain config parameters for the student
    hidden_size: int = field(
        default=None,
        metadata={"help": "Hidden size for student model. Overrides teacher config if set."}
    )
    intermediate_size: int = field(
        default=None,
        metadata={"help": "Intermediate size for student model. Overrides teacher config if set."}
    )
    # Copy teacher parameters (like feature_extractor)
    copy_feature_extractor: bool = field(
        default=True,
        metadata={"help": "Whether to copy teacher's feature extractor parameters to the student model."}
    )
    # ---- alpha_mse 추가
    alpha_mse: float = field(
        default=0.0,
        metadata={"help": "Weight for the cos embedding loss between teacher/student hidden states."}
    )
# ---- distillation changes END
# ------------------------------------------------------------------------------------

# @dataclass
# class TrainingArguments:
#     learning_rate: float = field(default=0.001)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to *student* pretrained model or model identifier from huggingface.co/models"}
    )
    evaluate_checkpoint: str = field(
        default=None,
        metadata={"help": "checkpoint to evaluate. If not set, will use model_name_or_path."}
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained tokenizer or identifier from huggingface.co/models"},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where you want to store the pretrained models from huggingface.co"},
    )
    freeze_feature_encoder: bool = field(
        default=True, 
        metadata={"help": "Whether to freeze the feature encoder layers of the student model."}
    )
    attention_dropout: float = field(default=0.0)
    activation_dropout: float = field(default=0.0)
    feat_proj_dropout: float = field(default=0.0)
    hidden_dropout: float = field(default=0.0)
    final_dropout: float = field(default=0.0)
    mask_time_prob: float = field(default=0.05)
    mask_time_length: int = field(default=10)
    mask_feature_prob: float = field(default=0.0)
    mask_feature_length: int = field(default=10)
    layerdrop: float = field(default=0.0)
    ctc_loss_reduction: Optional[str] = field(
        default="mean", 
        metadata={"help": "The way the ctc loss should be reduced. 'mean' or 'sum'."}
    )
    ctc_zero_infinity: Optional[bool] = field(default=False)
    add_adapter: Optional[bool] = field(default=False)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to data input for training and eval.
    """
    dataset_name: str = field(
        default=None,
        metadata={"help": "The dataset name to use from the 'datasets' library."}
    )
    data_dir: str = field(
        default=None,
        metadata={"help": "Path to the data directory. If not set, will use the dataset_name."}
    )
    split_name: str = field(
        default="clean",
        metadata={"help": "The dataset split to use."},
    )
    dataset_config_name: str = field( # 안쓰임
        default=None, 
        metadata={"help": "The configuration name of the dataset (via the datasets library)."}
    )
    train_split_name: str = field(
        default="train.100",
        metadata={"help": "Which split to use for training."},
    )
    eval_split_name: str = field(
        default="validation",
        metadata={"help": "Which split to use for evaluation."},
    )
    audio_column_name: str = field(
        default="audio",
        metadata={"help": "Name of the column in the dataset that contains the audio data."},
    )
    text_column_name: str = field(
        default="label", # "text" -> None
        metadata={"help": "Name of the column in the dataset that contains the text data."},
    )
    overwrite_cache: bool = field(default=False)
    preprocessing_num_workers: Optional[int] = field(default=None)
    max_train_samples: Optional[int] = field(default=None)
    max_eval_samples: Optional[int] = field(default=None)
    chars_to_ignore: Optional[List[str]] = list_field(default=None)
    eval_metrics: List[str] = list_field(
        default=["wer"],
        metadata={"help": "Metrics for evaluation, e.g. 'wer cer'"},
    )
    max_duration_in_seconds: float = field(default=20.0)
    min_duration_in_seconds: float = field(default=0.0)
    preprocessing_only: bool = field(default=False)
    token: str = field(default=None)
    trust_remote_code: bool = field(default=False)
    unk_token: str = field(default="[UNK]")
    pad_token: str = field(default="[PAD]")
    word_delimiter_token: str = field(default="|")
    phoneme_language: Optional[str] = field(default=None)


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    """
    processor: AutoProcessor
    padding: Union[bool, str] = "longest"
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None
    feature_extractor_input_name: Optional[str] = "input_values"

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [
            {self.feature_extractor_input_name: feature[self.feature_extractor_input_name]} for feature in features
        ]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        labels_batch = self.processor.pad(
            labels=label_features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of_labels,
            return_tensors="pt",
        )

        # replace padding with -100 to ignore loss
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels
        if "attention_mask" in batch:
            batch["attention_mask"] = batch["attention_mask"].to(torch.long)

        return batch


def create_vocabulary_from_data(
    datasets: DatasetDict,
    word_delimiter_token: Optional[str] = None,
    unk_token: Optional[str] = None,
    pad_token: Optional[str] = None,
):
    def extract_all_chars(batch):
        all_text = " ".join(batch["target_text"])
        vocab = list(set(all_text))
        return {"vocab": [vocab], "all_text": [all_text]}

    vocabs = datasets.map(
        extract_all_chars,
        batched=True,
        batch_size=-1,
        keep_in_memory=True,
        remove_columns=datasets["train"].column_names,
    )

    import functools
    vocab_set = functools.reduce(
        lambda v1, v2: set(v1["vocab"][0]) | set(v2["vocab"][0]), vocabs.values()
    )
    vocab_dict = {v: k for k, v in enumerate(sorted(vocab_set))}

    # replace white space with delimiter token
    if word_delimiter_token is not None and " " in vocab_dict:
        vocab_dict[word_delimiter_token] = vocab_dict[" "]
        del vocab_dict[" "]

    # add unk and pad token
    if unk_token is not None and unk_token not in vocab_dict:
        vocab_dict[unk_token] = len(vocab_dict)
    if pad_token is not None and pad_token not in vocab_dict:
        vocab_dict[pad_token] = len(vocab_dict)

    return vocab_dict


# ------------------------------------------------------------------------------------
# ---- distillation changes START: custom DistilCTCTrainer for teacher-student
# ------------------------------------------------------------------------------------
class DistilCTCTrainer(Trainer):
    """
    Custom Trainer that computes a distillation loss for teacher-student training.
    """
    def __init__(self, 
                 teacher_model=None, 
                 student_model=None, 
                 temperature=5.0, 
                 lambda_param=0.5,
                 alpha_mse=0.0, # ---- alpha_mse 추가
                 *args, 
                 **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        self.student = student_model
        
        # teacher는 gradient 업데이트를 하지 않도록
        self.teacher.to(self.args.device)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        
        self.temperature = temperature
        self.lambda_param = lambda_param
        
        # alpha_mse 및 cosine_loss_fct 초기화
        self.alpha_mse = alpha_mse
        # self.cosine_loss_fct = nn.CosineEmbeddingLoss(reduction="mean")
        self.mse_loss_fct = nn.MSELoss(reduction="mean")  
        
        # KLDivLoss for distillation
        self.loss_function = nn.KLDivLoss(reduction="batchmean")
        
        # 예시: Student hidden_state (B, T, 432) → Linear → (B, T, 768)
        self.student_hidden_size = self.student.config.hidden_size
        self.teacher_hidden_size = self.teacher.config.hidden_size
        self.student_projection = nn.Linear(self.student_hidden_size, self.teacher_hidden_size)  # trainable
        self.student_projection.to(self.args.device)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Compute the weighted combination of CTC loss (student) and distillation loss (KL divergence)."""
        
        # attention_mask가 있는 경우 꺼냄
        attention_mask = inputs.get("attention_mask", None)
        
        # 1) Student forward (hidden_states 반환을 위해 output_hidden_states=True)
        student_output = model(**inputs, output_hidden_states=True)
        student_logits = student_output.logits # (batch_size, time_steps, vocab_size)
        s_hidden_states = student_output.hidden_states  # list of all layer hidden states: [layer0, layer1, ..., layerN]


        # 2) Teacher forward (no gradient, hidden_states 반환)
        with torch.no_grad():
            teacher_input = {k: v for k, v in inputs.items() if k != "labels"}
            teacher_output = self.teacher(**teacher_input, output_hidden_states=True)
            teacher_logits = teacher_output.logits
            t_hidden_states = teacher_output.hidden_states # [layer0, layer1, ..., layerN]

        # 3) Logit distillation loss: teacher (soft targets) vs student (soft targets)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)
        distillation_loss = self.loss_function(soft_student, soft_teacher) * (self.temperature ** 2)

        # 4) Normal supervised CTC loss from the student
        student_target_loss = student_output.loss

        # Combine them
        loss = (1.0 - self.lambda_param) * student_target_loss + self.lambda_param * distillation_loss
        
        # 추가: alpha_mse > 0.0이면, 마지막 hidden_states를 이용한 코사인 임베딩 손실 계산
        mse_loss = 0.0
        if self.alpha_mse > 0.0:
            # student/teacher에서 마지막 hidden state를 가져옴
            # wav2vec2는 마지막이 s_hidden_states[-1], t_hidden_states[-1]
            
            ### 이부분 잘못됨
            num_layers = min(len(s_hidden_states), len(t_hidden_states))
            
            sum_loss_mse = 0.0
            for layer_idx in range(num_layers):
                s_hid = s_hidden_states[layer_idx]   # (bs, seq_len, student_dim)
                t_hid = t_hidden_states[layer_idx]   # (bs, seq_len, teacher_dim)
                assert s_hid.size() == t_hid.size(), (
                    f"Shape mismatch at layer {layer_idx}: {s_hid.size()} vs {t_hid.size()}"
                )

                # 마스킹 (attention_mask) 적용
                if attention_mask is not None:
                    mask = attention_mask.unsqueeze(-1).expand_as(s_hid)  # (B, T, D)
                    s_masked = torch.masked_select(s_hid, mask.bool())
                    t_masked = torch.masked_select(t_hid, mask.bool())
                    dim = s_hid.size(-1)

                    s_masked = s_masked.view(-1, dim)
                    t_masked = t_masked.view(-1, dim)
                    # MSE Loss
                    layer_loss = self.mse_loss_fct(s_masked, t_masked)

                else:
                    B, L, D = s_hid.shape
                    s_view = s_hid.view(B * L, D)
                    t_view = t_hid.view(B * L, D)

                    # MSE loss
                    layer_loss = self.mse_loss_fct(s_view, t_view)
                
                sum_loss_mse += layer_loss
        
            # 레이어별로 평균을 낼 수도 있고, 그냥 합산할 수도 있음(취향/실험에 따라 다름)
            mse_loss = sum_loss_mse / float(num_layers)

        # alpha_mse (이름만 동일) 가중치로 최종 로스에 합산
        loss += self.alpha_mse * mse_loss
            
        # print(f"loss: {loss} / student_target_loss: {student_target_loss} / distillation_loss: {distillation_loss} / loss_cos: {loss_cos}")

        if "wandb" in sys.modules:
            current_step = self.state.global_step
            wandb.log({
                "train/total_loss": loss.item(),
                "train/ctc_loss": student_target_loss.item(),
                "train/distill_loss": distillation_loss.item(),
                "train/hidden_state_loss": mse_loss,
                "train/global_step": current_step,
            })

        return (loss, student_output) if return_outputs else loss

# ---- distillation changes END
# ------------------------------------------------------------------------------------


def main():
    # Original argument parser includes ModelArguments, DataTrainingArguments, TrainingArguments
    # We'll add DistillationArguments as well.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, DistillationArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args, distil_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, distil_args = parser.parse_args_into_dataclasses()

    send_example_telemetry("run_speech_recognition_ctc", model_args, data_args)

    

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)
    set_seed(training_args.seed)

    # ------------------------------------------------------------------------------------
    # 1. Load teacher model if provided
    # ------------------------------------------------------------------------------------
    if not distil_args.teacher_model_name:
        raise ValueError("You must provide a --teacher_model_name for distillation.")

    logger.info(f"Loading teacher model from: {distil_args.teacher_model_name}")
    teacher_processor = AutoProcessor.from_pretrained(distil_args.teacher_model_name)
    teacher_model = AutoModelForCTC.from_pretrained(distil_args.teacher_model_name)
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False  # Freeze teacher

    # ------------------------------------------------------------------------------------
    # 2. Load student config from model_name_or_path
    # ------------------------------------------------------------------------------------
    logger.info(f"Loading student config from: {model_args.model_name_or_path}")
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        token=data_args.token,
        trust_remote_code=data_args.trust_remote_code,
    )
    # Update config with user-specified overrides (like dropout, ctc, etc.)
    config.update(
        {
            "feat_proj_dropout": model_args.feat_proj_dropout,
            "attention_dropout": model_args.attention_dropout,
            "hidden_dropout": model_args.hidden_dropout,
            "final_dropout": model_args.final_dropout,
            "mask_time_prob": model_args.mask_time_prob,
            "mask_time_length": model_args.mask_time_length,
            "mask_feature_prob": model_args.mask_feature_prob,
            "mask_feature_length": model_args.mask_feature_length,
            "gradient_checkpointing": training_args.gradient_checkpointing,
            "layerdrop": model_args.layerdrop,
            "ctc_loss_reduction": model_args.ctc_loss_reduction,
            "ctc_zero_infinity": model_args.ctc_zero_infinity,
            "add_adapter": model_args.add_adapter,
        }
    )

    # ---- distillation changes: Overwrite config from teacher if needed
    # Instead of using student config from scratch, let's start from teacher config and then override:
    logger.info("Cloning teacher config to create the student config ...")
    teacher_config = teacher_model.config
    student_config_dict = teacher_config.to_dict()
    # Now update with any fields we want to override from the teacher
    # so that the student mimics teacher's config *unless* user overrides
    if distil_args.hidden_size is not None:
        student_config_dict["hidden_size"] = distil_args.hidden_size
    if distil_args.intermediate_size is not None:
        student_config_dict["intermediate_size"] = distil_args.intermediate_size

    # We also incorporate any existing or newly updated fields from the base config
    # student_config_dict.update(config.to_dict())
    StudentConfigCls = type(teacher_config)  # the same class as teacher config
    student_config = StudentConfigCls(**student_config_dict)
    # End of config logic

    # ------------------------------------------------------------------------------------
    # 3. Create the student model from config
    # ------------------------------------------------------------------------------------
    logger.info("Creating student model from merged student_config ...")
    student_model = AutoModelForCTC.from_config(student_config)

    # Optionally copy feature_extractor from teacher model to student
    if distil_args.copy_feature_extractor:
        logger.info("Copying teacher feature extractor parameters to student ...")
        with torch.no_grad():
            teacher_state = teacher_model.state_dict()
            # student_state = student_model.state_dict()
            overwritten = []
            for n, p in student_model.named_parameters():
                if "feature_extractor" in n and n in teacher_state:
                    p.copy_(teacher_state[n].data)
                    overwritten.append(n)
            logger.info(f"Parameters overwritten from teacher: {overwritten}")

    # ------------------------------------------------------------------------------------
    # 4. Tokenizer & feature extractor
    # ------------------------------------------------------------------------------------
    # We'll load the *student* tokenizer & feature_extractor
    tokenizer_name_or_path = model_args.tokenizer_name_or_path or model_args.model_name_or_path
    tokenizer_kwargs = {}
    logger.info(f"Loading student tokenizer from: {tokenizer_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        token=data_args.token,
        trust_remote_code=data_args.trust_remote_code,
        **tokenizer_kwargs,
    )
    logger.info(f"Loading student feature extractor from: {model_args.model_name_or_path}")
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        token=data_args.token,
        trust_remote_code=data_args.trust_remote_code,
    )

    # The combined processor for the student (used in training, collators, etc.)
    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path)
    # If there's a mismatch (in practice you might rely on the student processor alone)
    # But let's ensure the student processor is consistent:
    if not isinstance(processor, Wav2Vec2Processor):
        warnings.warn("Processor loaded is not Wav2Vec2Processor. Using combined feature_extractor+tokenizer.")
        processor = Wav2Vec2Processor(feature_extractor, tokenizer)

    # Freeze encoder if requested (for the student)
    if model_args.freeze_feature_encoder:
        student_model.freeze_feature_encoder()

    # ------------------------------------------------------------------------------------
    # 5. Load datasets
    # ------------------------------------------------------------------------------------
    download_config = DownloadConfig(
        # timeout=24*3600,          # 24시간 정도로 크게 설정
        resume_download=True,     # 실패 시 이어받기
        max_retries=10            # 재시도 횟수 늘리기
    )
    raw_datasets = DatasetDict()
    
    
    if data_args.data_dir:
        if training_args.do_train:
            raw_datasets["train"] = load_dataset(
                data_args.data_dir,
                data_args.split_name,
                split=data_args.train_split_name,
                token=data_args.token,
                trust_remote_code=data_args.trust_remote_code,
                download_config=download_config,
            )
            if data_args.max_train_samples is not None:
                raw_datasets["train"] = raw_datasets["train"].select(range(data_args.max_train_samples))

        if training_args.do_eval:
            raw_datasets["dev_clean"] = load_dataset(
                data_args.data_dir,
                data_args.split_name,
                split=data_args.eval_split_name,
                token=data_args.token,
                trust_remote_code=data_args.trust_remote_code,
                download_config=download_config,
            )
            if data_args.max_eval_samples is not None:
                raw_datasets["dev_clean"] = raw_datasets["dev_clean"].select(range(data_args.max_eval_samples))
        
    else:
        if training_args.do_train:
            raw_datasets["train"] = load_dataset(
                data_args.dataset_name,
                data_args.split_name,
                split=data_args.train_split_name,
                token=data_args.token,
                trust_remote_code=data_args.trust_remote_code,
                download_config=download_config,
            )
            if data_args.max_train_samples is not None:
                raw_datasets["train"] = raw_datasets["train"].select(range(data_args.max_train_samples))

        if training_args.do_eval:
            raw_datasets["dev_clean"] = load_dataset(
                data_args.dataset_name,
                data_args.split_name,
                split=data_args.eval_split_name,
                token=data_args.token,
                trust_remote_code=data_args.trust_remote_code,
                download_config=download_config,
            )
            if data_args.max_eval_samples is not None:
                raw_datasets["dev_clean"] = raw_datasets["dev_clean"].select(range(data_args.max_eval_samples))

    # Remove special chars
    chars_to_ignore_regex = (
        f'[{"".join(data_args.chars_to_ignore)}]' if data_args.chars_to_ignore is not None else None
    )
    
    text_column_name = data_args.text_column_name
    if text_column_name is not None:
    
        def remove_special_characters(batch):
            if chars_to_ignore_regex is not None:
                batch["target_text"] = re.sub(chars_to_ignore_regex, "", batch[text_column_name]).lower() + " "
            else:
                # batch["target_text"] = batch[text_column_name].lower() + " "
                pass
            return batch

        with training_args.main_process_first(desc="dataset map special characters removal"):
            for k in list(raw_datasets.keys()):
                raw_datasets[k] = raw_datasets[k].map(
                    remove_special_characters,
                    remove_columns=[text_column_name],
                    desc="remove special characters",
                )

    # 6. Possibly create vocabulary from data if no tokenizer is given
    # (skipped here for brevity, or copied from original code snippet)

    # adapt config
    # We already integrated updates in student_config
    student_config.pad_token_id = tokenizer.pad_token_id
    student_config.vocab_size = len(tokenizer)

    # 7. Preprocess datasets
    dataset_sampling_rate = next(iter(raw_datasets.values())).features[data_args.audio_column_name].sampling_rate
    if dataset_sampling_rate != feature_extractor.sampling_rate:
        raw_datasets = raw_datasets.cast_column(
            data_args.audio_column_name, 
            datasets.features.Audio(sampling_rate=feature_extractor.sampling_rate)
        )

    max_input_length = data_args.max_duration_in_seconds * feature_extractor.sampling_rate
    min_input_length = data_args.min_duration_in_seconds * feature_extractor.sampling_rate

    audio_column_name = data_args.audio_column_name
    num_workers = data_args.preprocessing_num_workers
    feature_extractor_input_name = feature_extractor.model_input_names[0]
    phoneme_language = data_args.phoneme_language

    def prepare_dataset(batch):
        sample = batch[audio_column_name]
        inputs = feature_extractor(sample["array"], sampling_rate=sample["sampling_rate"])
        batch[feature_extractor_input_name] = getattr(inputs, feature_extractor_input_name)[0]
        batch["input_length"] = len(sample["array"].squeeze())
        additional_kwargs = {}
        if phoneme_language is not None:
            additional_kwargs["phonemizer_lang"] = phoneme_language
        batch["labels"] = tokenizer(batch["target_text"], **additional_kwargs).input_ids
        return batch

    with training_args.main_process_first(desc="dataset map preprocessing"):
        for k in list(raw_datasets.keys()):
            raw_datasets[k] = raw_datasets[k].map(
                prepare_dataset,
                remove_columns=raw_datasets[k].column_names,
                num_proc=num_workers,
                desc="preprocess datasets",
            )
            def is_audio_in_length_range(length):
                return length > min_input_length and length < max_input_length
            raw_datasets[k] = raw_datasets[k].filter(
                is_audio_in_length_range,
                num_proc=num_workers,
                input_columns=["input_length"],
            )

    # 8. Setup metrics
    eval_metrics = {metric: evaluate.load(metric, cache_dir=model_args.cache_dir) for metric in data_args.eval_metrics}

    def preprocess_logits_for_metrics(logits, labels):
        if len(logits) > 1:
            logits = logits[0]  # logits가 tuple인 경우 첫 번째 요소를 사용, 2 번째 요소는 hidden_state임
        pred_ids = torch.argmax(logits, dim=-1)
        return pred_ids, labels
    
    wer_metric = load("wer")
    
    '''
    def create_compute_metrics(trainer):
        def compute_metrics(pred):
            pred_logits = pred.predictions
            pred_ids = torch.argmax(torch.tensor(pred_logits), dim=-1)
            pred_str = processor.batch_decode(pred_ids)
            label_ids = pred.label_ids
            label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
            label_str = processor.batch_decode(label_ids, group_tokens=False)
            wer = wer_metric.compute(predictions=pred_str, references=label_str)

            # 현재 스텝과 에폭 번호 출력
            current_step = trainer.state.global_step

            return {"wer": wer}
        return compute_metrics
    '''
    
    def compute_metrics(pred):
        pred_ids = pred.predictions[0]
        pred.label_ids[pred.label_ids == tokenizer.pad_token_id] = -100

        pred_str = tokenizer.batch_decode(pred_ids)
        # we do not want to group tokens when computing the metrics
        label_str = tokenizer.batch_decode(pred.label_ids, group_tokens=False)

        metrics = {k: v.compute(predictions=pred_str, references=label_str) for k, v in eval_metrics.items()}

        return metrics

    if data_args.preprocessing_only:
        logger.info(f"Data preprocessing finished. Files cached at {raw_datasets.cache_files}")
        return

    # 9. Create data collator
    data_collator = DataCollatorCTCWithPadding(
        processor=processor, 
        feature_extractor_input_name=feature_extractor_input_name
    )

    # Save feature extractor, tokenizer, config
    with training_args.main_process_first():
        if is_main_process(training_args.local_rank):
            feature_extractor.save_pretrained(training_args.output_dir)
            tokenizer.save_pretrained(training_args.output_dir)
            student_config.save_pretrained(training_args.output_dir)

    # ------------------------------------------------------------------------------------
    # 10. Initialize custom DistilCTCTrainer
    # ------------------------------------------------------------------------------------
    trainer = DistilCTCTrainer(
        teacher_model=teacher_model,
        student_model=student_model,
        temperature=distil_args.temperature,
        lambda_param=distil_args.lambda_param,
        model=student_model,  # redundant but for clarity
        data_collator=data_collator,
        args=training_args,
        # compute_metrics=create_compute_metrics(trainer=None),
        compute_metrics=compute_metrics,
        train_dataset=raw_datasets["train"] if training_args.do_train else None,
        eval_dataset=raw_datasets["dev_clean"] if training_args.do_eval else None,
        processing_class=processor,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        alpha_mse=distil_args.alpha_mse,  # ---- alpha_mse 추가
    )
    # trainer.compute_metrics = create_compute_metrics(trainer)
    
    
    # 11. Train
    if training_args.do_train:
        # checkpoint = last_checkpoint if last_checkpoint else None
        if os.path.isdir(model_args.model_name_or_path):
            checkpoint = model_args.model_name_or_path
        train_result = trainer.train()
        trainer.save_model()  # Saves the student model

        metrics = train_result.metrics
        max_train_samples = (data_args.max_train_samples 
                             if data_args.max_train_samples is not None 
                             else len(raw_datasets["train"]))
        metrics["train_samples"] = min(max_train_samples, len(raw_datasets["train"]))
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # 12. Evaluate
    results = {}
    if training_args.do_eval:
        # Detecting last checkpoint
        last_checkpoint = None
        if (os.path.isdir(training_args.output_dir) 
            and training_args.do_train 
            and not training_args.overwrite_output_dir):
            last_checkpoint = get_last_checkpoint(training_args.output_dir)
            if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
                raise ValueError(
                    f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                    "Use --overwrite_output_dir to overcome."
                )
            elif last_checkpoint is not None:
                logger.info(
                    f"Checkpoint detected, resuming training at {last_checkpoint}."
                )
        logger.info("*** Evaluate ***")
        checkpoint = last_checkpoint if last_checkpoint else model_args.evaluate_checkpoint
        evaluate_model = AutoModelForCTC.from_pretrained(checkpoint)
        trainer.model = evaluate_model
        metrics = trainer.evaluate()
        max_eval_samples = (data_args.max_eval_samples 
                            if data_args.max_eval_samples is not None 
                            else len(raw_datasets["dev_clean"]))
        metrics["eval_samples"] = min(max_eval_samples, len(raw_datasets["dev_clean"]))
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        results.update(metrics)

    # (Optional) push_to_hub or create_model_card
    if training_args.push_to_hub:
        trainer.push_to_hub()
    else:
        trainer.create_model_card()

    return results


if __name__ == "__main__":
    main()
