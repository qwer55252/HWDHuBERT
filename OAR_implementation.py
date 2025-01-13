import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2Config, Wav2Vec2Processor
import numpy as np

from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
from scipy.stats import pearsonr

import torch
import numpy as np
from scipy.special import softmax
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr

import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset, Audio
from transformers import Wav2Vec2Processor

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from transformers import Wav2Vec2Processor
from transformers import TrainingArguments, Trainer, Wav2Vec2ForCTC

# =============== #
# 1. 모델 로드 예시
# =============== #
# 요청에 따라 Wav2Vec2 모델과 프로세서를 예시로 로드합니다.
# (실제로는 BERT에서 attention을 추출해야 하므로, BERT 모델을 사용해야 함)
from transformers import Wav2Vec2Processor, Wav2Vec2Model

def cosine_distance(vec1, vec2):
    """1 - cosine similarity"""
    # scipy의 cosine()는 distance를 바로 반환하므로 그대로 사용해도 됨
    # distance = 1 - cosine_similarity => distance = cosine_distance
    return cosine(vec1, vec2)

def pearson_distance(vec1, vec2):
    """1 - Pearson correlation"""
    # stats.pearsonr 리턴값: (상관계수, p-value)
    corr, _ = pearsonr(vec1, vec2)
    # corr 범위는 -1 ~ 1, 거리로 쓰려면 [0,2] 범위가 되지만
    # 논문에서 0~1로 표준화한다고 했으니 1 - (corr+1)/2 로 매핑하는 식도 고려할 수 있음
    # 여기서는 간단히 1 - corr로만 사용(음의 상관도 처리 방법은 연구 맥락에 따라 달라짐)
    return 1 - corr

def jensen_shannon_distance(p, q):
    """Jensen-Shannon distance (보통 JS divergence의 제곱근을 distance로 사용)"""
    # 두 확률 분포 p, q가 들어온다고 가정 (합이 1)
    p = np.array(p, dtype=np.float64)
    q = np.array(q, dtype=np.float64)
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    # KL 계산 시 log(0) 방지를 위해 epsilon 추가 등 처리 가능
    def kl_div(a, b):
        a = np.where(a == 0, 1e-12, a)
        b = np.where(b == 0, 1e-12, b)
        return np.sum(a * np.log(a / b))
    js_div = 0.5 * kl_div(p, m) + 0.5 * kl_div(q, m)
    return np.sqrt(js_div)  # JS distance

def bhattacharyya_distance(p, q):
    """Bhattacharyya distance"""
    p = np.array(p, dtype=np.float64)
    q = np.array(q, dtype=np.float64)
    p /= p.sum()
    q /= q.sum()
    bc = np.sum(np.sqrt(p * q))
    # Bhattacharyya distance = -ln(bc)
    # 보통 0~∞ 범위, 논문에서는 [0,1] 범위로 정규화해서 사용
    # 여기서는 간단히 -ln(bc)만 반환, 필요 시 max/min 스케일링 가능
    return -np.log(bc + 1e-12)

def get_token_based_distance(matA, matB, metric="cosine"):
    """
    matA, matB shape = (seq_len, seq_len)
    각각의 '행(row)'을 해당 토큰(i)이 바라보는 attention distribution이라 가정.
    토큰 i마다 vecA = matA[i, :], vecB = matB[i, :]
    -> 두 벡터 간 distance 측정 -> 전체 토큰에 대해 평균.
    """
    distances = []
    for i in range(matA.shape[0]):
        vecA = matA[i, :]
        vecB = matB[i, :]

        if metric == "cosine":
            dist = cosine_distance(vecA, vecB)
        elif metric == "corr":
            dist = pearson_distance(vecA, vecB)
        elif metric == "js":
            dist = jensen_shannon_distance(vecA, vecB)
        elif metric == "bc":
            dist = bhattacharyya_distance(vecA, vecB)
        else:
            raise ValueError("Unknown token-based metric")
        
        distances.append(dist)

    return np.mean(distances)


# =============== #
# 4. Sentence-based distance 함수들
# =============== #
# 두 헤드의 n×n Attention matrix를 “전체 행렬” 단위로 비교해 거리를 구합니다.
# 실제 논문에서 쓰는 distance correlation, Procrustes, Canonical correlation은
# 구현이 조금 복잡하거나 추가 라이브러리가 필요합니다.
# 여기서는 간단한 버전/흉내를 보여드립니다.

def distance_correlation(A, B):
    """
    Distance correlation(Szekely 등)은 임의 차원 행렬 간 독립성/종속성을
    측정하기 위한 지표. 여기서는 간단화를 위해 행렬을 1D로 펼쳐서
    pairwise distance 기반 계산을 아주 간단히 흉내만 낸 예시입니다.
    실제 구현은 더 복잡한 절차가 필요합니다.
    """
    # 실제론 pairwise distance행렬 a_{ij}, b_{ij}를 만들어
    # double-centering 등을 거쳐 계산함.
    # 여기서는 예시로 임시 변환(행렬->1D벡터) 후 상관계수로 대체.
    A_flat = A.flatten()
    B_flat = B.flatten()
    corr, _ = pearsonr(A_flat, B_flat)
    # distance = 1 - |corr|
    # (실제 distance correlation과 동일하진 않지만, 예시로서...)
    return 1 - abs(corr)

def procrustes_distance(A, B):
    """
    Procrustes analysis:
    한 행렬을 회전/축척/직교변환하여 다른 행렬과 얼마나 잘 align되는지 보는 기법.
    실제론 scipy의 procrustes 등을 사용할 수 있음.
    여기서는 매우 단순화된 예시(행렬을 정규화해 차이만 보는 형태).
    """
    # 간단히 frobenius norm으로 차이를 보는 예시
    # 실제론 회전/직교 변환 등을 최적화로 구해야 함.
    A_norm = A / (np.linalg.norm(A) + 1e-12)
    B_norm = B / (np.linalg.norm(B) + 1e-12)
    fro_diff = np.linalg.norm(A_norm - B_norm)
    # fro_diff 범위가 [0,2] 정도 되므로, 0~1 사이로 스케일링하기 위해 /2
    return fro_diff / 2.0

def canonical_correlation_distance(A, B):
    """
    Canonical Correlation: 두 데이터 집합(행렬)의 선형 조합들이
    얼마나 상관관계가 높은지 측정하는 기법.
    실제론 차원 맞춤, SVD 등을 이용해야 함.
    여기서는 간단히 SVD 없이 열벡터를 이어 붙인 후 상관관계로 예시 대체.
    """
    # 예시로 (seq_len x seq_len)을 (seq_len^2)로 펴서 상관 계산
    A_flat = A.flatten()
    B_flat = B.flatten()
    corr, _ = pearsonr(A_flat, B_flat)
    # corr이 높을수록 유사 -> distance = 1 - corr
    return 1 - corr

def get_sentence_based_distance(matA, matB, metric="dCor"):
    """
    matA, matB shape = (seq_len, seq_len)
    한 문장 전체의 attention 행렬을 직접 비교.
    """
    if metric == "dCor":
        return distance_correlation(matA, matB)
    elif metric == "PC":
        return procrustes_distance(matA, matB)
    elif metric == "CC":
        return canonical_correlation_distance(matA, matB)
    else:
        raise ValueError("Unknown sentence-based metric")


# =============== #
# 5. 실제 Pairwise Distance 계산 예시
# =============== #
# 여기서는 head가 4개이므로 4x4=16개 쌍에 대해 거리를 구하여
# (4,4) 크기의 distance matrix를 얻을 수 있습니다.

def compute_pairwise_distances(attention_mats, distance_func, mode="token", metric="cosine"):
    """
    attention_mats: shape = (num_heads, seq_len, seq_len)
    distance_func: get_token_based_distance() or get_sentence_based_distance()
    mode: "token" or "sentence"
    metric: 사용할 세부 지표
    """
    num_heads = attention_mats.shape[0]
    dist_matrix = np.zeros((num_heads, num_heads))

    for i in range(num_heads):
        for j in range(num_heads):
            if i == j:
                dist_matrix[i, j] = 0.0
            elif j > i:
                dist_val = distance_func(attention_mats[i], attention_mats[j], metric)
                dist_matrix[i, j] = dist_val
                dist_matrix[j, i] = dist_val
            # j < i인 경우 이미 계산됨

    return dist_matrix



# 예: 샘플 레이트 16k, 1초짜리 랜덤 파형 1개(batch_size=1)
# TODO: 데이터 수 늘려야함 + 전체적인 하이퍼파라미터 수정 필요

sample_rate = 16000
duration = 1 # seconds
batch_size = 1
audio_tensor = torch.randn(batch_size, sample_rate * duration) # (B, T)
audio_tensor.shape


model_name = "facebook/wav2vec2-base-100h"

config = Wav2Vec2Config.from_pretrained(model_name, output_attentions=True)
processor = Wav2Vec2Processor.from_pretrained(model_name, config=config)
model = Wav2Vec2Model.from_pretrained(model_name, config=config)

# forward 시, attention을 함께 받아오기
with torch.no_grad():
    outputs = model(audio_tensor, output_attentions=True)
    # outputs.attentions: 튜플 형태, (num_layers, B, num_heads, seq_len, seq_len)
    attentions = outputs.attentions

num_layers = len(attentions)                    # 12
num_heads_per_layer = attentions[0].shape[1]    # 12
total_heads = num_layers * num_heads_per_layer  # 144

# batch_size=1이므로 dim=0 없애기
attention_matrices = torch.stack(attentions).squeeze(1) # torch.Size([12, 12, 104, 104])

# (num_heads * num_layers, seq_len, seq_len)으로 변환
seq_len = attention_matrices.shape[-1]
attention_matrices = attention_matrices.view(-1, seq_len, seq_len) # torch.Size([144, 104, 104])
                            
head_distance_matrix = compute_pairwise_distances(
    attention_matrices,
    distance_func=get_token_based_distance,
    mode="token",
    metric="corr"
)

# TODO: head_distance_matirx를 0~1로 정규화 할 필요 있음 ✅
head_distance_matrix = head_distance_matrix / 2.0 # head_distance_matrix를 0~1 범위로 정규화 (2로 나누기)
head_similarity_matrix = 1.0 - head_distance_matrix 


# 스펙트럴 클러스터링을 위해 precomputed affinity 사용을 가정
# SKlearn의 SpectralClustering을 사용하여 스펙트럴 클러스터링을 수행 -> 여기서는 유사도 행렬을 사용해야만 함

pruning_ratio = 0.1
# 원하는 클러스터 개수 (pruning_ratio 파라미터에 따라 결정되게 수정 필요)
n_clusters = int(total_heads * pruning_ratio)
clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', assign_labels='kmeans', random_state=42)

# 클러스터 레이블 예측 
cluster_labels = clustering.fit_predict(head_similarity_matrix) # head_similarity_matrix는 (144x144) 헤드간의 유사도 행렬

# Silhouette Score 계산
# 여기서는 유사도 행렬이 아니라 "거리 행렬"을 사용해야함

try:
    # metric='precomputed'로 해서 head_redundancy_matrix 자체를 거리 행렬로 간주
    score = silhouette_score(
        head_distance_matrix,
        cluster_labels,
        metric='precomputed'
    )
    print(f"Spectral Clustering Silhouette Score: {score:.4f}")
except Exception as e:
    print("실루엣 스코어 계산 중 오류:", e)


    
# 각 클러스터별 대표 Head 1개만 남기고 나머지는 모두 제거
# 예: 클러스터마다 첫 번째로 만나는 Head를 대표로 선정 <- 일단 이렇게 구현하자

cluster_to_rep = {} # cluster_id -> 대표 head_id
# TODO: 대표 head 선정 방법을 수정해야 함
for head_idx, c_id in enumerate(cluster_labels):
    if c_id not in cluster_to_rep:
        cluster_to_rep[c_id] = head_idx
# cluster_to_rep[c_id] = head_idx: c_id 클러스터의 대표 head는 head_idx번째 head

# 각 클러스터 대표만 남긴다고 가정
selected_head_indices = list(cluster_to_rep.values()) # values에는 대표 head index들이 들어있음
selected_head_indices.sort()

# (선택된 head만 남겨두기 위해) 레이어 단위로 묶어본다
layer_to_keep = {layer_idx: [] for layer_idx in range(num_layers)}

for c_id, rep_head_idx in cluster_to_rep.items():
    layer_id = rep_head_idx // num_heads_per_layer
    head_id_in_layer = rep_head_idx % num_heads_per_layer
    layer_to_keep[layer_id].append(head_id_in_layer)
# layer_to_keep[layer_id] = [...] -> 해당 layer에서 남길 head들의 index들
layer_to_keep




def build_prune_dict(model_config, layer_to_keep_dict):
    """
    layer_to_keep_dict: 예) {0: [0, 2], 1: [1, 3, 5], ...}
    model_config: model.config (Wav2Vec2Config)
    
    반환: heads_to_prune 형태의 dict
       예) { layer_i: [prune_head_idx1, prune_head_idx2, ...], ... }
    """
    num_attention_heads = model_config.num_attention_heads  # 보통 12
    heads_to_prune = {}
    
    for layer_idx in range(model_config.num_hidden_layers):
        keep_heads = set(layer_to_keep_dict.get(layer_idx, []))
        all_heads = set(range(num_attention_heads))
        # prune할 head는 all_heads - keep_heads
        prune_heads = sorted(list(all_heads - keep_heads))
        if prune_heads:
            heads_to_prune[layer_idx] = prune_heads
    
    return heads_to_prune

heads_to_prune = build_prune_dict(model.config, layer_to_keep)


from transformers.pytorch_utils import prune_linear_layer

def find_pruneable_heads_and_indices(heads, num_heads, head_size, already_pruned_heads=None):
    """
    BERT 모델 pruning 로직을 참고해 작성한 유틸 함수.
    `heads`: 제거할 head 번호들의 집합 (ex: {1, 3})
    `num_heads`: 현재 레이어의 전체 head 수 (예: 12)
    `head_size`: head당 dimension (예: 64)
    `already_pruned_heads`: 이미 제거된 head들의 집합
    """
    if already_pruned_heads is None:
        already_pruned_heads = set()

    # 현재까지 제거된 head들을 합집합으로
    heads = set(heads) - already_pruned_heads
    mask = torch.ones(num_heads, head_size)
    # heads_to_prune에 해당하는 row는 0으로 만든다
    for head in heads:
        mask[head] = 0
    mask = mask.view(-1).eq(1)
    
    index = torch.arange(num_heads * head_size)[mask]
    return heads, index


def prune_wav2vec2_attention_layer(attention_module, heads_to_prune):
    """
    wav2vec2의 single layer(Wav2Vec2Attention)에서 지정된 head들을 제거.
    attention_module: Wav2Vec2Attention 객체
                     (encoder.layers[i].attention)
    heads_to_prune: 리스트/집합 형태. 제거해야 할 head 번호들
    """
    if not heads_to_prune:
        return  # 제거할 head가 없으면 아무것도 안 함
    
    # (예) attention_module.num_heads=12, attention_module.head_dim=64
    num_heads, head_dim = attention_module.num_heads, attention_module.head_dim
    
    # 이미 prune된 head가 있다면, 그 정보를 반영
    already_pruned_heads = getattr(attention_module, "pruned_heads", set())
    
    # 제거할 head와 인덱스 계산
    heads, index = find_pruneable_heads_and_indices(
        heads_to_prune,
        num_heads,
        head_dim,
        already_pruned_heads
    )
    
    # Q, K, V, Out projection에 대해 prune
    # 1) q_proj ( in_features -> num_heads*head_dim, out_features -> hidden_size )
    attention_module.q_proj = prune_linear_layer(attention_module.q_proj, index, dim=1)
    # 2) k_proj
    attention_module.k_proj = prune_linear_layer(attention_module.k_proj, index, dim=1)
    # 3) v_proj
    attention_module.v_proj = prune_linear_layer(attention_module.v_proj, index, dim=1)
    # 4) out_proj
    #   out_proj에서 head(=channel) 방향 pruning은 weight의 "in_features" 차원 축소로 진행
    attention_module.out_proj = prune_linear_layer(attention_module.out_proj, index, dim=0)
    
    # heads 제거 후, num_heads를 업데이트
    attention_module.num_heads = attention_module.num_heads - len(heads)
    # 어떤 head들이 제거되었는지 기록
    attention_module.pruned_heads = already_pruned_heads.union(heads)


def prune_wav2vec2_attention(model, heads_to_prune_dict):
    """
    실제 Wav2Vec2Model에 대해 레이어 단위로 prune_wav2vec2_attention_layer(...) 호출.
    
    heads_to_prune_dict 형태 예:
       {
         0: [1, 2],   # layer 0에서 head 1,2 제거
         3: [0, 5, 7] # layer 3에서 head 0,5,7 제거
         ...
       }
    """
    for layer_idx, prune_head_list in heads_to_prune_dict.items():
        layer_module = model.encoder.layers[layer_idx]
        # layer_module.attention: Wav2Vec2Attention
        prune_wav2vec2_attention_layer(layer_module.attention, prune_head_list)

###
# 클러스터링 결과로부터 layer_to_keep(유지할 head) -> heads_to_prune(제거할 head) 변환
###

def build_heads_to_prune_dict(config, layer_to_keep_dict):
    """
    layer_to_keep_dict: 예) {0: [0, 2], 1: [1, 3, 5], ...}
    config: Wav2Vec2Config
    return: heads_to_prune 형태의 dict
       예) { layer_i: [head_i, head_j, ...], ... }
    """
    num_attention_heads = config.num_attention_heads  # (예: 12)
    num_hidden_layers = config.num_hidden_layers      # (예: 12)
    heads_to_prune = {}
    
    for layer_idx in range(num_hidden_layers):
        keep_heads = set(layer_to_keep_dict.get(layer_idx, []))
        all_heads = set(range(num_attention_heads))
        # 제거해야 할 head = 전체 - 유지
        prune_heads = sorted(list(all_heads - keep_heads))
        if prune_heads:
            heads_to_prune[layer_idx] = prune_heads
    return heads_to_prune


### 3) 커스텀 Pruning 함수 호출
prune_wav2vec2_attention(model, heads_to_prune)

# 잘 되었는지 확인(레이어별 num_heads, pruned_heads 상태)
for i, layer in enumerate(model.encoder.layers):
    attn = layer.attention
    print(f"Layer {i} -> num_heads:{attn.num_heads}, pruned_heads:{getattr(attn, 'pruned_heads', None)}")





# LibriSpeech ASR 데이터셋 로드 (train-clean-100 및 validation-clean)
raw_train_dataset = load_dataset("Sreyan88/librispeech_asr", "clean", split="train.100")
raw_eval_dataset = load_dataset("Sreyan88/librispeech_asr", "clean", split="validation")

chars_to_ignore_regex = r"[\,\?\.\!\-\;\:\"\“\%\‘\”\‘]"

def preprocess_function(batch):
        audio = batch["audio"]
        inputs = processor([audio["array"]], sampling_rate=audio["sampling_rate"], return_attention_mask=True)
        batch["input_values"] = inputs.input_values[0]
        batch["attention_mask"] = inputs.attention_mask[0]

        text_list = [batch["text"]] if isinstance(batch["text"], str) else batch["text"]
        text_encoding = processor.tokenizer(text_list)
        batch["labels"] = text_encoding.input_ids[0] if len(text_list) == 1 else text_encoding.input_ids
        return batch
    
train_dataset = raw_train_dataset.map(preprocess_function, remove_columns=raw_train_dataset.column_names)
eval_dataset = raw_eval_dataset.map(preprocess_function, remove_columns=raw_eval_dataset.column_names)

train_dataset = raw_train_dataset.map(
    preprocess_function,
    num_proc=4
)
eval_dataset = raw_eval_dataset.map(
    preprocess_function,
    num_proc=4
)

print(train_dataset)
print(eval_dataset)



@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        # Pad audio inputs (input_values)
        batch = self.processor.feature_extractor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt"
        )

        # Pad label inputs
        with self.processor.as_target_processor():
            labels_batch = self.processor.tokenizer.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt"
            )

        # Replace padding token ID with -100 for CTC loss
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["input_ids"] == self.processor.tokenizer.pad_token_id, -100
        )
        batch["labels"] = labels

        return batch

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)


# Pruning 정보 초기화
model.config.pruned_heads.clear()
# model.config.pruned_heads = {}

model_ft = Wav2Vec2ForCTC(config=model.config)
model_ft.load_state_dict(model.state_dict(), strict=False)  # Pruning된 weight 적용

training_args = TrainingArguments(
    output_dir="./test-ft",
    num_train_epochs=50,
    per_device_train_batch_size=4,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
    learning_rate=1e-4,
    fp16=True,  # GPU 사용 시
)

trainer = Trainer(
    model=model_ft,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,  # 준비된 데이터셋
    eval_dataset=eval_dataset,
    tokenizer=processor,
)

trainer.train()
