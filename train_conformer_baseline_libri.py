
"""
train_conformer_small.py

Transformers의 load_dataset으로 LibriSpeech 100h 불러와
halved-dimension Conformer CTC 모델 구조(student)를 NeMo로 생성,
Weights & Biases 로깅 포함
"""


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
from nemo.core.connectors.save_restore_connector import SaveRestoreConnector
import glob
import zipfile
import torch.nn.functional as F
from pruning_utils import (
    get_avg_attention_matrices,
    cluster_and_select_heads,
    prune_conformer_attention
)
from transformers import Wav2Vec2Processor
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
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)

    # 기본 HF_DATASETS_CACHE (원본 오디오가 풀리던 위치)
    default_root = config.HF_DATASETS_CACHE
    extract_marker = os.path.join("extracted")

    # with open(manifest_path, "w") as fout:
    #     for sample in ds:
    #         audio = sample["audio"]
    #         orig_path = audio["path"] 
    #         # sample["audio"]["path"] : '/workspace/data/cache/extracted/28e1f76d85906acbe5672f913bb405be336b2a2aa63d4db4a3d1546fd2728272/2277-149896-0000.flac'
    #         # 실제 데이터 경로 : '/workspace/data/cache/extracted/28e1f76d85906acbe5672f913bb405be336b2a2aa63d4db4a3d1546fd2728272/LibriSpeech/dev-clean/2277/149896/2277-149896-0000.flac'
            

    #         duration = len(audio["array"]) / audio["sampling_rate"]
    #         entry = {
    #             "audio_filepath": orig_path,  # 실제로 존재하는 절대/상대 경로
    #             "duration": duration,
    #             "text": sample["text"].lower().strip(),
    #         }
    #         fout.write(json.dumps(entry, ensure_ascii=False) + "\n")

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
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=self.data_collator,
        )


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
    parser.add_argument("--epochs", type=int, default=50, help="최대 epoch 수")
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
    
    if args.test_mode: # 100개 데이터만 사용
        train_ds = train_ds.select(range(100))
        val_ds = val_ds.select(range(100))
        test_ds = test_ds.select(range(100))
        

    # 3) W&B logger 생성
    exp_name = os.getenv("EXP_NAME")
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_dir,
        filename="best",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min",
    )

    # 4) PyTorch Lightning Trainer
    
    cfg_dict = vars(args)
    wandb_logger = WandbLogger(
        project=exp_name,
        save_dir=args.output_dir,
        config=cfg_dict,   # pure dict
    )
    trainer = pl.Trainer(
        devices=args.gpus,
        accelerator="gpu",
        max_epochs=args.epochs,
        default_root_dir=args.output_dir,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
    )
    

    model = My_EncDecCTCModelBPE.from_pretrained(
        model_name="stt_en_conformer_ctc_small",
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
    model.cfg.train_ds.is_tarred = False # manifest_filepath 기반으로 데이터 Load하기 위한 설정
    model.cfg.train_ds.tarred_audio_filepaths = None
    model.cfg.train_ds.manifest_filepath      = train_manifest
    model.cfg.train_ds.sample_rate            = args.data_sample_rate
    model.cfg.train_ds.batch_size             = args.batch_size

    model.cfg.validation_ds.is_tarred = False
    model.cfg.validation_ds.tarred_audio_filepaths = None
    model.cfg.validation_ds.manifest_filepath = val_manifest
    model.cfg.validation_ds.sample_rate       = args.data_sample_rate
    model.cfg.validation_ds.batch_size        = args.batch_size
    
    
    model.cfg.test_ds.is_tarred = False
    model.cfg.test_ds.tarred_audio_filepaths = None
    model.cfg.test_ds.manifest_filepath       = test_manifest
    model.cfg.test_ds.sample_rate             = args.data_sample_rate
    model.cfg.test_ds.batch_size              = args.batch_size
    
    model.train_dataset = train_ds
    model.val_dataset = val_ds
    model.test_dataset = test_ds
    model.batch_size = args.batch_size
    model.data_collator = collator
    
    model.setup_training_data(model.cfg.train_ds) 
    
    
    
    # 파이썬에서 Nemo API로 풀어두는 함수 실행
    release_nemoAPI(model)
    
    # 올바른 속성 이름으로 변경
    model._save_restore_connector.model_extracted_dir = f"{args.output_dir}/nemo_archive"
    
    AppState().nemo_file_folder = f"{args.output_dir}/nemo_archive"

    # 8) 학습 시작
    _orig = wandb_logger.log_hyperparams
    wandb_logger.log_hyperparams = lambda *args, **kwargs: None
    trainer.fit(model)
    wandb_logger.log_hyperparams = _orig
    
    # 9) Best checkpoint 로드 후 .nemo로 저장
    best_ckpt = checkpoint_callback.best_model_path
    nemo_out  = os.path.join(args.output_dir, exp_name,
                            f"result_weight_{exp_name}.nemo")
    save_weights_only_nemo(model, best_ckpt, nemo_out)
    
    # 10) 평가 시작
    split_names = ["dev.clean", "dev.other", "test.clean", "test.other"]
    metrics = {}
    for i, split_name in enumerate(split_names):
        print(f"\n===== Evaluating on split: {split_name} =====")
        model.eval()

        test_i_ds = load_dataset(
            args.data_script_path,
            args.data_config_name,
            split=split_name,
            trust_remote_code=True,
            download_config=dl_cfg,
            cache_dir=cache_dir,
        )
        if args.test_mode:
            test_i_ds = test_i_ds.select(range(100))
        json_name = split_name.replace(".", "_") + ".json"
        manifest_i = os.path.join(manifest_dir, json_name)
        build_manifest_from_hf(test_i_ds, manifest_i, cache_dir)

        test_data_config = deepcopy(model.cfg.test_ds)
        test_data_config.manifest_filepath = manifest_i
        # shuffle 옵션이 없으면 False 로 자동 설정되지만, 명시적으로 꺼줄 수도 있습니다.
        test_data_config.shuffle = False

        # NeMo API 호출: 내부에서 _test_dl 이 세팅되고,
        # 이후 test_dataloader() 호출 시 이 _test_dl 이 반환됩니다.
        model.setup_test_data(test_data_config)
        dl = model.test_dataloader()
        
        _orig_log_hparams = wandb_logger.log_hyperparams
        wandb_logger.log_hyperparams = lambda *args, **kwargs: None

        results = trainer.test(
            model=model,
            dataloaders=[dl],
            ckpt_path=best_ckpt or None,
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
