"""
train_conformer_small.py

Transformers의 load_dataset으로 LibriSpeech 100h 불러와
halved-dimension Conformer CTC 모델 구조(student)를 NeMo로 생성,
Weights & Biases 로깅 포함
"""

import re
import os
import torch
import tempfile
import shutil
import json
import argparse
import torch.nn as nn
from ruamel.yaml import YAML
import nemo.collections.asr as nemo_asr
import lightning as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from datasets import load_dataset, DownloadConfig, config
import aiohttp
from omegaconf import OmegaConf
from copy import deepcopy
from nemo.utils.app_state import AppState
from torch.utils.data import DataLoader
import glob
import zipfile
import torch.nn.functional as F
from transformers import Wav2Vec2Config, Wav2Vec2Processor
from pruning_utils import (
    get_avg_attention_matrices,
    cluster_and_select_heads,
    prune_conformer_attention,
    find_remaining_heads,
    find_heads_to_keep,
    compute_pairwise_distances,
    get_token_based_distance,
    build_prune_dict
)
import numpy as np
from sklearn.cluster import SpectralClustering
from pruning_utils import prune_conformer_attention
from nemo.core.connectors.save_restore_connector import SaveRestoreConnector
from dataclasses import dataclass
from typing import Union, Optional
from transformers import Wav2Vec2FeatureExtractor
from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer

@dataclass
class ConformerCTCDataCollator:
    feature_extractor: Wav2Vec2FeatureExtractor
    tokenizer: SentencePieceTokenizer
    padding: Union[bool, str] = True  # 이 예제에서는 항상 True 로 가정
    sampling_rate: int = 16000

    def __call__(self, features):
        # 1) waveform → feature
        batch_audio = [f["audio"]["array"] for f in features]
        inputs = self.feature_extractor(
            batch_audio,
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )
        input_signal = inputs["input_values"].float()
        input_signal_length = inputs["attention_mask"].sum(dim=-1)


        # 2) text → token IDs 리스트 (no padding yet)
        raw_texts = [f["text"] for f in features]
        ids_list  = [self.tokenizer.text_to_ids(text) for text in raw_texts]

        # 3) 리스트를 동일 길이로 패딩
        batch_size = len(ids_list)
        max_len    = max(len(ids) for ids in ids_list)
        pad_id     = self.tokenizer.pad_id  # SentencePieceTokenizer 의 pad token ID

        labels = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
        for i, ids in enumerate(ids_list):
            labels[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)

        # 4) CTC ignore index(-100) 로 변환
        labels = labels.masked_fill(labels == pad_id, -100)
        label_lengths = (labels != -100).sum(-1)

        return (
            input_signal,        # Tensor [B, T], float32
            input_signal_length, # Tensor [B]
            labels,              # Tensor [B, L], with -100 padding
            label_lengths,       # Tensor [B]
        )


def build_manifest_from_hf(ds, manifest_path: str, cache_dir: str):
    """
    HuggingFace Dataset 객체(ds)를 순회하며
    NeMo 형식의 JSON manifest를 생성
    """
    # 기본 HF_DATASETS_CACHE (원본 오디오가 풀리던 위치)
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    # HF가 flac을 풀어놓는 최상위 디렉토리
    extract_root = os.path.join(cache_dir, "extracted")

    with open(manifest_path, "w") as fout:
        for sample in ds:
            audio     = sample["audio"]
            orig_path = audio["path"]  # HF가 알려준 경로 (존재하지 않을 수도 있음)

            # 1) 첫 시도: orig_path 에 파일이 실제로 존재하는지
            if not os.path.isfile(orig_path):
                filename = os.path.basename(orig_path)
                # 2) fallback: extract_root 이하를 재귀 검색
                pattern = os.path.join(extract_root, "**", filename)
                matches = glob.glob(pattern, recursive=True)
                if not matches:
                    raise FileNotFoundError(
                        f"Audio 파일을 찾을 수 없습니다: {filename} \n"
                        f"원경로: {orig_path}\n"
                        f"검색경로: {pattern}"
                    )
                # 검색 결과 중 첫 번째를 사용
                orig_path = matches[0]

            duration = len(audio["array"]) / audio["sampling_rate"]
            entry = {
                "audio_filepath": orig_path,
                "duration":        duration,
                "text":            sample["text"].lower().strip(),
            }
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")

def release_nemoAPI(model):
    # 1) .nemo 실제 경로 조회
    # 실제 .nemo 파일 경로는 model._cfg.restore_from 혹은 
    # model._pretrained_model_path 에 있습니다.
    meta = AppState().get_model_metadata_from_guid(model.model_guid)
    nemo_file = meta.restoration_path
    # 2) 압축 풀기
    connector = SaveRestoreConnector()
    connector._unpack_nemo_file(nemo_file, out_folder="/workspace/outputs/nemo_archive")
     # 3) 다음 복원 때 재활용할 디렉토리 지정
    model._save_restore_connector.model_extracted_dir = "/workspace/outputs/nemo_archive"
    AppState().nemo_file_folder = "/workspace/outputs/nemo_archive"

def save_weights_only_nemo(model, checkpoint_path, save_path):
    """
    model: 학습된 ModelPT 인스턴스
    checkpoint_path: trainer.best_model_path
    save_path: 최종 .nemo 파일 경로
    """
    # 1) 임시 디렉터리 생성
    pack_dir = tempfile.mkdtemp()
    try:
        # 2) config 저장
        cfg_file = os.path.join(pack_dir, "model_config.yaml")
        OmegaConf.save(config=model.cfg, f=cfg_file)

        # 3) 체크포인트 복사
        ckpt_name = os.path.basename(checkpoint_path)
        shutil.copy(checkpoint_path, os.path.join(pack_dir, ckpt_name))

        # 4) (선택) 토크나이저 등 추가 파일 복사
        # 예: shutil.copy("<vocab.json>", os.path.join(pack_dir, "vocab.json"))

        # 5) pack_dir → .nemo (Zip) 파일로 만들기
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with zipfile.ZipFile(save_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(pack_dir):
                for fname in files:
                    fullpath = os.path.join(root, fname)
                    relpath  = os.path.relpath(fullpath, pack_dir)
                    zf.write(fullpath, relpath)

        print(f"✅ Saved weights-only .nemo to {save_path}")
    finally:
        # 6) 임시 디렉터리 삭제
        shutil.rmtree(pack_dir)

def setup_nemo_datasets_and_cfg(model, train_ds, val_ds, test_ds, collator, train_manifest, val_manifest, test_manifest, args):
    """
    NeMo 모델의 cfg와 데이터셋, collator를 일관되게 할당하고 setup_training_data를 호출합니다.
    """
    # train_ds
    train_cfg = model.cfg.train_ds
    if isinstance(train_cfg, dict):
        is_tarred = train_cfg.get("is_tarred", False)
    else:
        is_tarred = getattr(train_cfg, "is_tarred", False)
    if is_tarred:
        if isinstance(train_cfg, dict):
            train_cfg["is_tarred"] = False
            train_cfg["tarred_audio_filepaths"] = None
        else:
            train_cfg.is_tarred = False
            train_cfg.tarred_audio_filepaths = None
    if isinstance(train_cfg, dict):
        train_cfg["manifest_filepath"] = train_manifest
        train_cfg["sample_rate"] = args.data_sample_rate
        train_cfg["batch_size"] = args.batch_size
    else:
        train_cfg.manifest_filepath = train_manifest
        train_cfg.sample_rate = args.data_sample_rate
        train_cfg.batch_size = args.batch_size

    # validation_ds
    val_cfg = model.cfg.validation_ds
    if isinstance(val_cfg, dict):
        is_tarred = val_cfg.get("is_tarred", False)
    else:
        is_tarred = getattr(val_cfg, "is_tarred", False)
    if is_tarred:
        if isinstance(val_cfg, dict):
            val_cfg["is_tarred"] = False
            val_cfg["tarred_audio_filepaths"] = None
        else:
            val_cfg.is_tarred = False
            val_cfg.tarred_audio_filepaths = None
    if isinstance(val_cfg, dict):
        val_cfg["manifest_filepath"] = val_manifest
        val_cfg["sample_rate"] = args.data_sample_rate
        val_cfg["batch_size"] = args.batch_size
    else:
        val_cfg.manifest_filepath = val_manifest
        val_cfg.sample_rate = args.data_sample_rate
        val_cfg.batch_size = args.batch_size

    # test_ds
    test_cfg = model.cfg.test_ds
    if isinstance(test_cfg, dict):
        is_tarred = test_cfg.get("is_tarred", False)
    else:
        is_tarred = getattr(test_cfg, "is_tarred", False)
    if is_tarred:
        if isinstance(test_cfg, dict):
            test_cfg["is_tarred"] = False
            test_cfg["tarred_audio_filepaths"] = None
        else:
            test_cfg.is_tarred = False
            test_cfg.tarred_audio_filepaths = None
    if isinstance(test_cfg, dict):
        test_cfg["manifest_filepath"] = test_manifest
        test_cfg["sample_rate"] = args.data_sample_rate
        test_cfg["batch_size"] = args.batch_size
    else:
        test_cfg.manifest_filepath = test_manifest
        test_cfg.sample_rate = args.data_sample_rate
        test_cfg.batch_size = args.batch_size

    model.train_dataset = train_ds
    model.val_dataset = val_ds
    model.test_dataset = test_ds
    model.batch_size = args.batch_size
    model.data_collator = collator

    model.setup_training_data(model.cfg.train_ds)
    model.setup_validation_data(model.cfg.validation_ds)
    model.setup_test_data(model.cfg.test_ds)

class My_EncDecCTCModelBPE(nemo_asr.models.EncDecCTCModelBPE):
    def __init__(self, cfg, trainer=None):
        # ① NeMo 부모 생성자 호출 시 반드시 cfg, trainer 인자를 전달
        super().__init__(cfg=cfg, trainer=trainer)
        # ② 데이터셋은 None 으로 초기화
        self.train_dataset = None
        self.val_dataset   = None
        self.test_dataset  = None
        self.data_collator = None

    @classmethod
    def from_pretrained(
        cls,
        *args,
        train_ds=None,
        val_ds=None,
        test_ds=None,
        **kwargs,
    ):
        # 부모 class 의 from_pretrained 로 모델 로드
        model = super().from_pretrained(*args, **kwargs)
        # 호출 시 전달된 데이터셋 속성으로 할당
        if train_ds is not None:
            model.train_dataset = train_ds
        if val_ds is not None:
            model.val_dataset = val_ds
        if test_ds is not None:
            model.test_dataset = test_ds
        return model

    # DataLoader 메서드들은 그대로 놔두시면 됩니다.


    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=self.data_collator,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=self.data_collator,
        )

    def test_dataloader(self):
        # 1) setup_test_data()로 _test_dl 이 세팅되어 있으면 그걸 쓰고,
        if hasattr(self, "_test_dl") and self._test_dl is not None:
            return self._test_dl
        # 2) 아니면 원래 self.test_dataset 기반 DataLoader
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=self.data_collator,
        )

def load_datasets(dataset_name):
    if dataset_name == "librispeech":
        train_dataset = load_dataset("openslr/librispeech_asr", "clean", split="train.100", trust_remote_code=True)
        dev_dataset = load_dataset("openslr/librispeech_asr", "clean", split="validation", trust_remote_code=True)
        test_dataset = load_dataset("openslr/librispeech_asr", "clean", split="test", trust_remote_code=True)
    elif dataset_name == "tedlium":
        train_dataset = load_dataset("./tedlium_test.py", "release1", split="train", trust_remote_code=True)
        dev_dataset = load_dataset("./tedlium_test.py", "release1", split="validation", trust_remote_code=True)
        test_dataset = load_dataset("./tedlium_test.py", "release1", split="test", trust_remote_code=True)
        MAX_SAMPLES = 320000  # 10초
        train_dataset = train_dataset.filter(lambda x: x["audio"]["array"].shape[0] <= MAX_SAMPLES)
        dev_dataset = dev_dataset.filter(lambda x: x["audio"]["array"].shape[0] <= MAX_SAMPLES)
        test_dataset = test_dataset.filter(lambda x: x["audio"]["array"].shape[0] <= MAX_SAMPLES)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return train_dataset, dev_dataset, test_dataset

def preprocess_function(batch, processor):
    # 1) 오디오 인풋 처리
    input_values = processor(
        batch["audio"]["array"],
        sampling_rate=16_000
    ).input_values[0]
    
    batch["text"] = clean_transcript(batch["text"])  # 텍스트 전처리
    # 2) 텍스트 라벨은 tokenizer를 직접 호출
    labels = processor.tokenizer(
        batch["text"]
    ).input_ids
    
    return {
        "input_values": input_values,
        "labels": labels,
    }

def clean_transcript(text):
    text = re.sub(r"\{.*?\}", "", text)              # {SMACK}, {BREATH} 등 제거
    text = re.sub(r"\(.*?\)", "", text)              # (2), (3) 등 제거
    text = text.replace("<sil>", "")                 # <sil> 제거
    text = text.upper()                              # 모두 대문자
    text = re.sub(r"[^A-Z' ]+", "", text)            # A–Z, 공백, ' 만 남기고 제거
    text = re.sub(r"\s+", " ", text).strip()         # 중복 공백 정리
    return text


def main():
    parser = argparse.ArgumentParser(
        description="Train halved-dimension Conformer CTC student on LibriSpeech 100h"
    )
    parser.add_argument("--data_dir", type=str, default="data", help="데이터 루트 디렉토리")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/conformer_ctc_bpe.yaml",
        help="원본 모델 config YAML 경로",
    )
    parser.add_argument("--gpus", type=int, default=1, help="사용할 GPU 개수")
    parser.add_argument("--batch_size", type=int, default=8, help="배치 크기")
    parser.add_argument("--data_sample_rate", type=int, default=16000, help="샘플링 주파수")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="로그·체크포인트·wandb 저장 디렉토리",
    )
    parser.add_argument(
        "--data_script_path",
        type=str,
        default="./librispeech_asr.py",
        help="HuggingFace LibriSpeech 데이터 스크립트 경로",
    )
    parser.add_argument(
        "--data_config_name",
        type=str,
        default="train_100",
        help="_DL_URLS 키값 설정(train_100 등)",
    )
    parser.add_argument(
        "--data_train_split",
        type=str,
        default="train.clean.100",
        help="훈련 split 이름",
    )
    parser.add_argument(
        "--data_val_split",
        type=str,
        default="dev.clean",
        help="평가 split 이름",
    )
    parser.add_argument(
        "--data_test_split",
        type=str,
        default="test.clean",
        help="평가 split 이름",
    )
    parser.add_argument(
        "--train_teacher_model",
        type=bool,
        default=False,
        help="True: teacher 모델 학습, False: student 모델 학습",
    )
    parser.add_argument(
        "--logit_distillation",
        type=bool,
        default=False,
        help="CTC loss 외에 teacher logits 와의 KL-divergence loss 를 추가"
    )
    parser.add_argument(
        "--kd_alpha",
        type=float,
        default=1.0,
        help="logit distillation loss 의 가중치"
    )
    parser.add_argument(
        "--kd_temperature",
        type=float,
        default=1.0,
        help="softmax 온도 파라미터"
    )
    parser.add_argument(
        "--layerwise_distillation", 
        type=bool, 
        default=False,
        help="레이어 단위 KD 실행 여부"
    )
    parser.add_argument(
        "--layer_kd_alpha", 
        type=float, 
        default=1.0,
        help="레이어 KD loss 가중치"
    )
    parser.add_argument(
        "--test_mode",
        action="store_true",
        help="테스트 모드일 때 True로 설정하면 데이터셋을 매우 적게 사용"
    )
    parser.add_argument(
        "--iterative_finetune_epochs",
        type=int,
        default=2,
        help="각 Iteration마다 미세튜닝할 epoch 수"
    )
    parser.add_argument(
        "--final_finetune_epochs",
        type=int,
        default=100,
        help="최종 Pruning 이후 미세튜닝 epoch 수"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="redundancy-based",
        choices=["redundancy-based", "magnitude-based", "l0-based", "baseline", "one-shot"],
        help="Pruning 방법 선택 (redundancy-based, magnitude-based, l0-based, baseline)"
    )
    parser.add_argument(
        "--prune_ratio",
        type=float,
        default=0.8,
        help="전체 헤드 중 제거할 비율 (0.0 ~ 1.0)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Iterative Pruning 반복 횟수"
    )
    parser.add_argument(
        "--distance_metric",
        type=str,
        default="cosine",
        help="token-based distance 지표('cosine', 'corr', 'js', 'bc') 또는 sentence-based('dCor', 'PC', 'CC') 중 선택"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="librispeech",
        help="[librispeech, tedlium]"
    )
    parser.add_argument(
        "--model_name_or_path", 
        type=str, 
        default="nvidia/stt_en_conformer_ctc_small",
        help="모델 이름 또는 경로 ['nvidia/stt_en_conformer_ctc_small', 'facebook/wav2vec2-base-100h', ...]"
    )
    args = parser.parse_args()

    # manifest 경로 설정
    os.makedirs(args.output_dir, exist_ok=True)
    manifest_dir = os.path.join(args.data_dir, "manifests")
    os.makedirs(manifest_dir, exist_ok=True)
    # train_manifest = os.path.join(args.data_dir, "manifests", "train-clean-100.json")
    # val_manifest = os.path.join(args.data_dir, "manifests", "validation.json")
    train_manifest = os.path.join(manifest_dir, "train.json")
    val_manifest = os.path.join(manifest_dir, "val.json")
    test_manifest = os.path.join(manifest_dir, "test.json")

    # 1) HuggingFace LibriSpeech 로드
    print("Datasets cache dir:", config.HF_DATASETS_CACHE)
    
    cache_dir = os.path.join(args.data_dir, "cache")
    config.HF_DATASETS_CACHE = cache_dir
    dl_cfg = DownloadConfig(
        cache_dir=cache_dir,
        force_download=False,
        resume_download=True,
        max_retries=10,
        disable_tqdm=False,
        download_desc="Downloading LibriSpeech ASR",
        storage_options={"client_kwargs": {"timeout": aiohttp.ClientTimeout(total=3600)}},
        delete_extracted=False,
        extract_compressed_file=True,
        force_extract=True,            
    )
    
    if args.dataset_name == "librispeech":
        train_ds = load_dataset(
            args.data_script_path,
            args.data_config_name,
            split=args.data_train_split,
            trust_remote_code=True,
            download_config=dl_cfg,
            cache_dir=cache_dir,
        )
        val_ds = load_dataset(
            args.data_script_path,
            args.data_config_name,
            split=args.data_val_split,
            trust_remote_code=True,
            download_config=dl_cfg,
            cache_dir=cache_dir,
        )
        test_ds = load_dataset(
            args.data_script_path,
            args.data_config_name,
            split=args.data_test_split,
            trust_remote_code=True,
            download_config=dl_cfg,
            cache_dir=cache_dir,
        )
    
    elif args.dataset_name == "tedlium":
        train_ds, val_ds, test_ds = load_datasets(dataset_name=args.dataset_name)
        wav2vec2config = Wav2Vec2Config.from_pretrained("facebook/wav2vec2-base-100h", output_attentions=True)
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-100h", config=wav2vec2config)
        if args.test_mode:
            # test_mode일 때는 데이터셋을 100개로 제한
            train_ds = train_ds.select(range(300))
            val_ds = val_ds.select(range(300))
            test_ds = test_ds.select(range(300))
        train_ds = train_ds.map(preprocess_function, fn_kwargs={"processor": processor}, num_proc=4)
        val_ds = val_ds.map(preprocess_function, fn_kwargs={"processor": processor}, num_proc=4)
        test_ds = test_ds.map(preprocess_function, fn_kwargs={"processor": processor}, num_proc=4)
        
    eval_datasets = {"dev": val_ds, "test": test_ds} # TODO: 정상작동하는지 확인
    
    print(f'train_ds.cache_files: {train_ds.cache_files}')  # [{'filename': '/home/you/.cache/huggingface/datasets/.../train.arrow', ...}, ...]
    # 2) NeMo manifest 생성

    print("building manifest files...")
    if not os.path.isfile(train_manifest):
        build_manifest_from_hf(train_ds, train_manifest, cache_dir)
        print(f"train_manifest DONE: {train_manifest}")
    if not os.path.isfile(val_manifest):
        build_manifest_from_hf(val_ds, val_manifest, cache_dir)
        print(f"val_manifest DONE: {val_manifest}")
    if not os.path.isfile(test_manifest):
        build_manifest_from_hf(test_ds, test_manifest, cache_dir)
        print(f"test_manifest DONE: {test_manifest}")
    print("manifest files built.")
    
    # test_mode일 때는 별도의 작은 manifest 파일을 생성
    if args.test_mode:
        train_ds = train_ds.select(range(300))
        val_ds = val_ds.select(range(300))
        test_ds = test_ds.select(range(300))
        args.iterative_finetune_epochs = 2
        args.final_finetune_epochs = 5

        # test_mode용 manifest 경로
        test_train_manifest = os.path.join(manifest_dir, "train_testmode.json")
        test_val_manifest = os.path.join(manifest_dir, "val_testmode.json")
        test_test_manifest = os.path.join(manifest_dir, "test_testmode.json")

        # 항상 새로 생성 (작으니 시간 부담 없음)
        build_manifest_from_hf(train_ds, test_train_manifest, cache_dir)
        build_manifest_from_hf(val_ds, test_val_manifest, cache_dir)
        build_manifest_from_hf(test_ds, test_test_manifest, cache_dir)

        # 이후 코드에서 사용할 manifest 경로를 test_mode용으로 교체
        train_manifest = test_train_manifest
        val_manifest = test_val_manifest
        test_manifest = test_test_manifest
        

    # 3) W&B logger 생성
    prj_name = os.getenv("PRJ_NAME")
    exp_name = os.getenv("EXP_NAME")
    iterative_checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_dir,
        filename="best",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min",
    )
    last_checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.output_dir, "final_finetune"),
        filename="last",
        save_top_k=0,
        verbose=True,
        save_last=True,
    )
    best_checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.output_dir, "final_finetune"),
        filename="best",
        save_top_k=2,
        verbose=True,
        monitor="val_wer",
        mode="min",
    )
    # 4) PyTorch Lightning Trainer
    cfg_dict = vars(args)
    wandb_logger = WandbLogger(
        project=prj_name,
        name=exp_name,
        save_dir=args.output_dir,
        config=cfg_dict,   # pure dict
    )
    trainer = pl.Trainer(
        devices=args.gpus,
        accelerator="gpu",
        max_epochs=args.final_finetune_epochs,
        default_root_dir=args.output_dir,
        logger=wandb_logger,
        callbacks=[last_checkpoint_callback, best_checkpoint_callback],
    )
    

    model = My_EncDecCTCModelBPE.from_pretrained(
        model_name=args.model_name_or_path,
        map_location="cuda:0",
        trainer=trainer,
    )
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
    tokenizer = model.tokenizer   # Nemo BPE tokenizer

    collator  = ConformerCTCDataCollator(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        padding=True,
        sampling_rate=args.data_sample_rate,
        # max_length=None,
        # pad_to_multiple_of=None,
    )
    
    

    # 새로운 manifest 경로로 덮어쓰기
    setup_nemo_datasets_and_cfg(
        model, train_ds, val_ds, test_ds, collator,
        train_manifest, val_manifest, test_manifest, args
    )
    
    # 파이썬에서 Nemo API로 풀어두는 함수 실행
    release_nemoAPI(model)
    
    # 올바른 속성 이름으로 변경
    model._save_restore_connector.model_extracted_dir = f"{args.output_dir}/nemo_archive"
    AppState().nemo_file_folder = f"{args.output_dir}/nemo_archive"
    
    
    # Stage 3: Iterative Pruning
    init_num_layers = model.cfg.encoder.n_layers  # conformer-small 예시: 16
    init_num_heads = model.cfg.encoder.n_heads  # conformer-small 예시: 4
    init_num_total_heads = init_num_layers * init_num_heads # conformer-small 예시: 64
    init_head_dim = model.encoder.layers[0].self_attn.d_k  # conformer-small 예시: 44
    heads_to_remove_per_iter = int(init_num_total_heads * args.prune_ratio / args.iterations)
    already_pruned_heads_dict = {i: set() for i in range(init_num_layers)}
    
    if args.method == "redundancy-based":
        avg_attn_matrices = get_avg_attention_matrices(
            model, 
            dataset=train_ds,   # batch마다 model(**batch, output_attentions=True) 필요
            data_collator=collator,
            sample_size=10,
            already_pruned_heads=already_pruned_heads_dict
        )
        heads_to_keep = find_heads_to_keep(avg_attn_matrices, already_pruned_heads_dict, init_num_total_heads, args) # [(head_idx, avg_similarity), ...]
        print(f' - 중요하다 판단해서 제거하지 않을 10개의 heads들: {heads_to_keep}')
    else:
        heads_to_keep = []
    
    
    
    if args.method == "baseline":
        print("▶ Baseline pruning: no pruning applied.")
        pass
    elif args.method == "one-shot":
        print("▶ One-shot pruning: computing average attention …")
        avg_attn_matrices = get_avg_attention_matrices(
            model,
            dataset=train_ds,
            data_collator=collator,
            sample_size=10,
            already_pruned_heads=already_pruned_heads_dict,
        )  # Tensor of shape [init_num_layers, init_num_heads, T, T]

        L, H, T, _ = avg_attn_matrices.shape
        flat_attn = avg_attn_matrices.view(-1, T, T)  # [(L*H), T, T]

        # 2) 헤드 간 pairwise distance → similarity
        head_distance_matrix = compute_pairwise_distances(
            flat_attn,
            distance_func=get_token_based_distance,
            mode="token",
            metric=args.distance_metric,
        )  # shape (L*H, L*H)
        head_distance_matrix = head_distance_matrix / head_distance_matrix.max()
        head_similarity_matrix = 1.0 - head_distance_matrix

        # 3) Spectral Clustering으로 대표 헤드 n_keep개 선택
        total_heads = init_num_total_heads
        n_keep = int(total_heads * (1.0 - args.prune_ratio))
        clustering = SpectralClustering(
            n_clusters=n_keep,
            affinity="precomputed",
            assign_labels="kmeans",
            random_state=42,
        )
        cluster_labels = clustering.fit_predict(head_similarity_matrix)  # shape (total_heads,)

        # 4) 클러스터별 첫 등장 헤드를 대표로 선택
        cluster_to_rep = {}
        for head_idx, c in enumerate(cluster_labels):
            cluster_to_rep.setdefault(c, head_idx)

        # 5) layer→keep할 head list 구성, 최소 한 개 보장
        layer_to_keep = {i: [] for i in range(init_num_layers)}
        for rep in cluster_to_rep.values():
            lyr = rep // init_num_heads
            h_in_layer = rep % init_num_heads
            layer_to_keep[lyr].append(h_in_layer)
        for i in range(init_num_layers):
            if not layer_to_keep[i]:
                layer_to_keep[i] = [0]

        # 6) prune dict 생성 및 적용
        heads_to_prune = build_prune_dict(layer_to_keep, init_num_layers, init_num_heads)
        print(f"▶ One-shot prune plan: {heads_to_prune}")
        prune_conformer_attention(model, heads_to_prune)
        print("▶ One-shot head pruning complete.")
    else:
        # # model 메모리 해제
        # del model
        # torch.cuda.empty_cache()
        
        # iterative 과정을 통해 already_pruned_heads_dict 생성
        for i in range(args.iterations):
            iter_last_checkpoint_callback = ModelCheckpoint(
                dirpath=os.path.join(args.output_dir, f"iteration_{i+1}"),
                filename=f"last",
                save_top_k=0,
                verbose=True,
                save_last=True,
            )
            trainer_i = pl.Trainer(
                devices=args.gpus,
                accelerator="gpu",
                max_epochs=args.iterative_finetune_epochs,
                default_root_dir=args.output_dir,
                logger=wandb_logger,
                callbacks=[iter_last_checkpoint_callback],
            )
            # model_i pruning 하고 short fine-tuning
            model_i = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(
                model_name=args.model_name_or_path,
                map_location="cuda:0",
                trainer=trainer_i,
            )
            setup_nemo_datasets_and_cfg(
                model_i, train_ds, val_ds, test_ds, collator,
                train_manifest, val_manifest, test_manifest, args
            )

            print(f' - {i+1}번째 itertaion pruning 할 head : {already_pruned_heads_dict}')
            prune_conformer_attention(model_i, already_pruned_heads_dict)
            
            _orig = wandb_logger.log_hyperparams
            wandb_logger.log_hyperparams = lambda *args, **kwargs: None
            trainer_i.fit(model_i)
            wandb_logger.log_hyperparams = _orig
            
            # 다음 iteration에서 제거할 헤드 추가
            if args.method == "redundancy-based":
                # 2) 평균 어텐션 계산 (train_ds에서 랜덤 샘플)
                avg_attn_matrices = get_avg_attention_matrices(
                    model_i, 
                    dataset=train_ds.select(range(1)),   # batch마다 model(**batch, output_attentions=True) 필요
                    data_collator=collator,
                    sample_size=1,
                    already_pruned_heads=already_pruned_heads_dict
                )
                del model_i
                del trainer_i
                torch.cuda.empty_cache()

                # 4) 클러스터링으로 제거할 헤드 선정
                n_remove = heads_to_remove_per_iter
                remove_candidates = cluster_and_select_heads(
                    avg_attn_matrices,
                    n_remove=n_remove,
                    init_num_total_heads=init_num_total_heads,
                    already_pruned_heads=already_pruned_heads_dict,
                    test_mode=args.test_mode,
                    init_num_heads=init_num_heads,
                    heads_to_keep=heads_to_keep
                )
                current_num_heads = find_remaining_heads(already_pruned_heads_dict, init_num_total_heads)
                print(f" - 현재 남아있는 전체 헤드 수: {current_num_heads}")            
                
                for (lyr_idx, h_idx) in remove_candidates:
                    if h_idx in already_pruned_heads_dict[lyr_idx]:
                        print(f' - 이미 제거된 head 이니 건너뜀: ({lyr_idx}, {h_idx})')
                        continue
                    h_g_idx = lyr_idx * init_num_heads + h_idx
                    if h_g_idx in heads_to_keep:
                        print(f' - 중요하다 판단해서 제거하지 않을 head 이니 건너뜀: ({lyr_idx}, {h_idx})')
                        continue
                    # 이 head까지 pruning하면 해당 layer에 더이상 남는 head가 없다면 skip
                    if len(already_pruned_heads_dict[lyr_idx]) + 1 == init_num_heads:
                        print(f' - 이 head까지 제거하면 더이상 남는 head가 없어서 건너뜀: ({lyr_idx}, {h_idx})')
                        continue
                    # already_pruned_heads_dict 갱신    
                    already_pruned_heads_dict[lyr_idx].add(h_idx)
            elif args.method == "magnitude-based":
                head_importance = []
                # 1) 각 layer, 각 head에 대해 Q-proj 가중치 L1-norm 계산
                for layer_idx in range(init_num_layers):
                    # HuggingFace Wav2Vec2 Attention 모듈
                    attn = model_i.encoder.layers[layer_idx].self_attn
                    q_weight = attn.linear_q.weight.data  # shape: [hidden_size, hidden_size]
                    k_weight = attn.linear_k.weight.data
                    v_weight = attn.linear_v.weight.data
                    o_weight = attn.linear_out.weight.data
                    # TODO: q_weight 만으로 magnitude를 계산해도 돼?
                    for h in range(init_num_heads):
                        # 이미 제거된 헤드는 스킵
                        if h in already_pruned_heads_dict[layer_idx]:
                            continue
                        start, end = h * init_head_dim, (h + 1) * init_head_dim
                        # L1-norm 을 중요도로 사용
                        q_norm = q_weight[start:end, :].abs().sum()
                        k_norm = k_weight[start:end, :].abs().sum()
                        v_norm = v_weight[start:end, :].abs().sum()

                        # Output projection: column block
                        o_norm = o_weight[:, start:end].abs().sum()

                        # 네 개 블록 합산하여 magnitude로 사용
                        magnitude = (q_norm + k_norm + v_norm + o_norm).item()
                        head_importance.append((layer_idx, h, magnitude))

                # 2) 중요도 오름차순 정렬
                head_importance.sort(key=lambda x: x[2])

                # 3) 이번 iteration 에 제거할 헤드 개수 결정
                current_num_heads = find_remaining_heads(already_pruned_heads_dict, init_num_total_heads)
                n_remove = heads_to_remove_per_iter
                # 최소 1개는 남기기
                if current_num_heads <= n_remove:
                    n_remove = current_num_heads - 1

                # 4) 제거 대상 후보 선택
                remove_candidates = head_importance[:n_remove]
                print(f" - magnitude-based 제거할 head 후보: {remove_candidates}")

                # 5) heads_to_prune_dict 및 already_pruned_heads_dict 업데이트
                cnt = 0
                for (lyr_idx, h_idx, _) in remove_candidates:
                    # 해당 layer 에 최소 1개는 남도록 보장
                    while len(already_pruned_heads_dict[lyr_idx]) + 1 == init_num_heads:
                        print(f" - layer {lyr_idx}에 남아있는 head가 1개 이하이므로 건너뜀: ({lyr_idx}, {h_idx}) 대신 다른 head로 대체")
                        lyr_idx, h_idx, _ = head_importance[n_remove + cnt]
                        cnt += 1
                    print(f" - 최종 remove head: ({lyr_idx}, {h_idx})")
                    already_pruned_heads_dict[lyr_idx].add(h_idx)
            elif args.method == "l0-based":
                eps = 1e-3
                head_importance = []
                # 1) 각 layer, 각 head에 대해 Q/K/V/O-proj 가중치 L0-노름 계산
                for layer_idx in range(init_num_layers):
                    attn = model_i.encoder.layers[layer_idx].self_attn
                    q_weight = attn.linear_q.weight.data   # [hidden, hidden]
                    k_weight = attn.linear_k.weight.data
                    v_weight = attn.linear_v.weight.data
                    o_weight = attn.linear_out.weight.data
                    for h in range(init_num_heads):
                        if h in already_pruned_heads_dict[layer_idx]:
                            continue
                        start, end = h * init_head_dim, (h + 1) * init_head_dim

                        # 각 블록에서 비제로 원소 개수 계산
                        sub_q = q_weight[start:end, :]
                        sub_k = k_weight[start:end, :]
                        sub_v = v_weight[start:end, :]
                        sub_o = o_weight[:, start:end]

                        # nonzero_q = torch.count_nonzero(sub_q)
                        # nonzero_k = torch.count_nonzero(sub_k)
                        # nonzero_v = torch.count_nonzero(sub_v)
                        # nonzero_o = torch.count_nonzero(sub_o)
                        
                        nonzero_q = torch.count_nonzero(sub_q.abs() > eps)
                        nonzero_k = torch.count_nonzero(sub_k.abs() > eps)
                        nonzero_v = torch.count_nonzero(sub_v.abs() > eps)
                        nonzero_o = torch.count_nonzero(sub_o.abs() > eps)

                        total_nonzero = (nonzero_q + nonzero_k + nonzero_v + nonzero_o).item()
                        head_importance.append((layer_idx, h, total_nonzero))

                # 2) 비제로 개수 오름차순 정렬 (가장 스파스한 헤드 우선)
                head_importance.sort(key=lambda x: x[2])

                # 3) 이번 iteration에 제거할 헤드 수 결정
                current_num_heads = find_remaining_heads(already_pruned_heads_dict, init_num_total_heads)
                n_remove = heads_to_remove_per_iter
                if current_num_heads <= n_remove:
                    n_remove = current_num_heads - 1  # 최소 1개는 남기기

                # 4) 제거 대상 후보 선택
                remove_candidates = head_importance[:n_remove]
                print(f" - l0-based 제거할 head 후보: {remove_candidates}")

                # 5) heads_to_prune_dict 및 already_pruned_heads_dict 업데이트
                cnt = 0
                for (lyr_idx, h_idx, _) in remove_candidates:
                    while len(already_pruned_heads_dict[lyr_idx]) + 1 == init_num_heads:
                        print(f" - layer {lyr_idx}에 남아있는 head가 1개 이하이므로 건너뜀: ({lyr_idx}, {h_idx}) 대신 다른 head로 대체")
                        lyr_idx, h_idx, _ = head_importance[n_remove + cnt]
                        cnt += 1
                    print(f" - 최종 remove head: ({lyr_idx}, {h_idx})")
                    already_pruned_heads_dict[lyr_idx].add(h_idx)
            else:
                raise ValueError(f"Unknown method: {args.method}")
            print(f" - {i+1}번째 iteration 최종 already_pruned_head_dict: {already_pruned_heads_dict}")
        
        
        trainer = pl.Trainer(
            devices=args.gpus,
            accelerator="gpu",
            max_epochs=args.final_finetune_epochs,
            default_root_dir=args.output_dir,
            logger=wandb_logger,
            callbacks=[best_checkpoint_callback, last_checkpoint_callback],
        )
        
        # 최종 모델 구조 선언
        model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(
            model_name=args.model_name_or_path,
            map_location="cuda:0",
            trainer=trainer,
        )
        setup_nemo_datasets_and_cfg(
            model, train_ds, val_ds, test_ds, collator,
            train_manifest, val_manifest, test_manifest, args
        )
        print(f' 최종 모델 구조의 already_pruned_heads_dict : {already_pruned_heads_dict}')
        prune_conformer_attention(model, already_pruned_heads_dict)
    
    # Stage 4: Final Fine-tuning
    _orig = wandb_logger.log_hyperparams
    wandb_logger.log_hyperparams = lambda *args, **kwargs: None
    trainer.fit(model)
    wandb_logger.log_hyperparams = _orig
    
    # 학습한 모델 저장
    final_epoch_ckpt_path = os.path.join(args.output_dir, "final_finetune", "final_epoch.ckpt")
    trainer.save_checkpoint(final_epoch_ckpt_path)
    print(f"✅ Final checkpoint saved to {final_epoch_ckpt_path}")
    
    # last_ckpt = last_checkpoint_callback.best_model_path
    # nemo_out  = os.path.join(args.output_dir, exp_name, f"result_weight_{exp_name}.nemo")
    # save_weights_only_nemo(model, last_ckpt, nemo_out)
    # print(f"✅ Saved weights‐only .nemo to {nemo_out}")
    
    
    # Stage 5: Evaluation
    # ================================
    # Stage 5: Evaluation
    # ================================
    # 1) 평가할 split 이름과 load_dataset 파라미터 분기
    MAX_SAMPLES = 160000  # 20초
    if args.dataset_name == "librispeech":
        split_names = ["dev.clean", "dev.other", "test.clean", "test.other"]
        script       = args.data_script_path
        config_name  = args.data_config_name
        load_kwargs = {
            "path":       script,
            "name":       config_name,
            "trust_remote_code": True,
            "download_config":   dl_cfg,
            "cache_dir":  cache_dir,
        }
    else:  # tedlium
        split_names = ["dev", "test"]
        # tedlium test split load
        load_kwargs = {
            "path":       "./tedlium_test.py",
            "name":       "release1",
            "trust_remote_code": True,
        }

    metrics = {}
    for split_name in split_names:
        print(f"\n===== Evaluating on split: {split_name} =====")
        model.eval()

        # 2) 테스트 데이터셋 로드 & (필요시) 필터 & 전처리
        test_i_ds = load_dataset(
            load_kwargs["path"],
            load_kwargs["name"],
            split=split_name,
            **{k:v for k,v in load_kwargs.items() if k not in ["path","name"]}
        )
        if args.dataset_name == "tedlium":
            # 최대 길이 필터 (10초)
            test_i_ds = test_i_ds.filter(lambda x: x["audio"]["array"].shape[0] <= MAX_SAMPLES)
            test_i_ds = test_i_ds.map(preprocess_function, fn_kwargs={"processor": processor}, num_proc=4)

        if args.test_mode:
            test_i_ds = test_i_ds.select(range(300))

        # 3) manifest 파일 생성
        json_name = split_name.replace(".", "_") + ".json"
        manifest_i = os.path.join(manifest_dir, json_name)
        build_manifest_from_hf(test_i_ds, manifest_i, cache_dir)

        # 4) NeMo 테스트 데이터 설정
        test_cfg = deepcopy(model.cfg.test_ds)
        test_cfg.manifest_filepath = manifest_i
        test_cfg.shuffle = False
        model.setup_test_data(test_cfg)
        dl = model.test_dataloader()
        
        _orig_log_hparams = wandb_logger.log_hyperparams
        wandb_logger.log_hyperparams = lambda *args, **kwargs: None

        results = trainer.test(
            model=model,
            dataloaders=[dl],
            ckpt_path=final_epoch_ckpt_path or None,
            verbose=True,
        )
        wandb_logger.log_hyperparams = _orig_log_hparams
        
        # trainer.test 는 리스트(dict) 반환, 첫 번째 원소에서 메트릭 추출
        res   = results[0]
        wer   = res.get("test_wer", res.get("wer", None))
        loss  = res.get("test_loss", res.get("loss", None))
        print(f"  → split={split_name} | loss={loss:.4f} | wer={wer:.2%}")
        
        # ① 메트릭 키에 split 이름을 붙여서 Wandb에 기록
        # #    dev.clean  → dev_clean/wer, dev_clean/loss
        key_prefix = split_name.replace(".", "_")
        metric = {
            f"{key_prefix}/wer":  wer,
            f"{key_prefix}/loss": loss,
        }
        metrics[f"{key_prefix}/wer"] = wer
        metrics[f"{key_prefix}/loss"] = loss
        # ② step을 epoch 기반으로 찍거나 global_step 을 사용
        wandb_logger.log_metrics(metric, step=trainer.current_epoch)
    print(f"metrics: {metrics}")
    wandb_logger.log_metrics(metrics, step=trainer.current_epoch)
    
if __name__ == "__main__":
    main()
