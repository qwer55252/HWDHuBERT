import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Config, Wav2Vec2Processor, TrainingArguments, Trainer, Wav2Vec2ForCTC
from transformers.pytorch_utils import prune_linear_layer

from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score

from scipy.spatial.distance import cosine
from scipy.stats import pearsonr

from datasets import load_dataset
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from evaluate import load

# =============== #
# Token-based distance 함수들
# =============== #
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
# Sentence-based distance 함수들
# =============== #
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

def compute_pairwise_distances(attention_mats, distance_func, mode="token", metric="cosine"):
    """
    실제 Pairwise Distance 계산 예시
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


# TODO: 데이터 수 늘려야함 + 전체적인 하이퍼파라미터 수정 필요
sample_rate = 16000
duration = 1 # seconds
batch_size = 10
audio_tensor = torch.randn(batch_size, sample_rate * duration) # (B, T)
output_dir="Baseline"

# Wav2Vec2 모델 및 프로세서 로드
model_name = "facebook/wav2vec2-base-100h"
config = Wav2Vec2Config.from_pretrained(model_name, output_attentions=True)
processor = Wav2Vec2Processor.from_pretrained(model_name, config=config)
model = Wav2Vec2ForCTC.from_pretrained(
    model_name, 
    output_attentions=True,
    ctc_loss_reduction="mean", 
    pad_token_id=processor.tokenizer.pad_token_id,
)

print(f'config: {config}')

# TODO: 주석 remove
'''
# forward 시, attention을 함께 받아오기
with torch.no_grad():
    outputs = model(audio_tensor, output_attentions=True)
    # outputs.attentions: 튜플 형태, (num_layers, B, num_heads, seq_len, seq_len)
    attentions = outputs.attentions

num_layers = len(attentions)                    # 12
num_heads_per_layer = attentions[0].shape[1]    # 12
total_heads = num_layers * num_heads_per_layer  # 144

if batch_size == 1:
    # batch_size=1이므로 dim=0 없애기
    attention_matrices = torch.stack(attentions).squeeze(1) # torch.Size([12, 12, 104, 104])
else:
    # batch_size=10이면, dim=1을 평균내어, torch.Size([12, 12, 104, 104]) 로 만듦
    attention_matrices = torch.stack(attentions).mean(dim=1) # torch.Size([12, 12, 104, 104])

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
print(f'min: {head_distance_matrix.min()}, max: {head_distance_matrix.max()}') # 0~2 사이인지 확인
head_distance_matrix = head_distance_matrix / 2.0 # head_distance_matrix를 0~1 범위로 정규화 (2로 나누기)
head_similarity_matrix = 1.0 - head_distance_matrix # TODO: 단순히 1에서 거리 뺀게 유사도라는거 수정해야함


# 스펙트럴 클러스터링을 위해 precomputed affinity 사용을 가정
# SKlearn의 SpectralClustering을 사용하여 스펙트럴 클러스터링을 수행 -> 여기서는 유사도 행렬을 사용해야만 함

pruning_ratio = 0.2
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
cluster_head_dict = {} # cluster_head_dict[cluster_id] -> [head_id1, head_id2, ...]

# TODO: 대표 head 선정 방법을 수정해야 함 -> 일단 클러스터별로 랜덤하게
for head_idx, c_id in enumerate(cluster_labels):
    cluster_head_dict[c_id] = cluster_head_dict.get(c_id, []) + [head_idx]

    # if c_id not in cluster_to_rep:
    #     cluster_to_rep[c_id] = head_idx
for c_id, head_indices in cluster_head_dict.items():
    # 랜덤하게 대표 head 선정
    cluster_to_rep[c_id] = np.random.choice(head_indices)
# cluster_to_rep[c_id] = head_idx: c_id 클러스터의 대표 head는 head_idx번째 head
print(f'cluster_to_rep: {cluster_to_rep}')

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
for layer_idx in range(num_layers):
    if not layer_to_keep[layer_idx]:
        layer_to_keep[layer_idx] = [0] # 해당 레이어에 대표 head가 없으면 첫 번째 Head 살림 # TODO: 수정 필요
print(f'layer_to_keep: {layer_to_keep}')



# 클러스터링 결과로부터 layer_to_keep(유지할 head) -> heads_to_prune(제거할 head) 변환
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
        # 만약 해당 레이어에 유지할 헤드가 없다면, 기본값으로 0번 헤드를 유지
        if not keep_heads:
            keep_heads = {0}
        all_heads = set(range(num_attention_heads))
        # prune할 head는 all_heads - keep_heads
        prune_heads = sorted(list(all_heads - keep_heads))
        if prune_heads:
            heads_to_prune[layer_idx] = prune_heads
    
    return heads_to_prune

heads_to_prune = build_prune_dict(model.config, layer_to_keep)

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
    attention_module.q_proj = prune_linear_layer(attention_module.q_proj, index, dim=0)
    # 2) k_proj
    attention_module.k_proj = prune_linear_layer(attention_module.k_proj, index, dim=0)
    # 3) v_proj
    attention_module.v_proj = prune_linear_layer(attention_module.v_proj, index, dim=0)
    # 4) out_proj
    #   out_proj에서 head(=channel) 방향 pruning은 weight의 "in_features" 차원 축소로 진행
    attention_module.out_proj = prune_linear_layer(attention_module.out_proj, index, dim=1)
    
    # heads 제거 후, num_heads를 업데이트
    attention_module.num_heads = attention_module.num_heads - len(heads)
    # 어떤 head들이 제거되었는지 기록
    attention_module.all_head_size = attention_module.num_heads * attention_module.head_dim
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
        layer_module = model.wav2vec2.encoder.layers[layer_idx]
        # layer_module = model.hubert.encoder.layers[layer_idx]
        # layer_module.attention: Wav2Vec2Attention
        prune_wav2vec2_attention_layer(layer_module.attention, prune_head_list)

 # Baseline 성능 뽑기 위해 일단 주석 처리 
# heads_to_prune = {0: [1,3,5,7,9,11], 1: [0,2,4,6,8,10], 2: [1,3,5,7,9,11], 3: [0,2,4,6,8,10], 4: [1,3,5,7,9,11], 5: [0,2,4,6,8,10], 6: [1,3,5,7,9,11], 7: [0,2,4,6,8,10], 8: [1,3,5,7,9,11], 9: [0,2,4,6,8,10], 10: [1,3,5,7,9,11], 11: [0,2,4,6,8,10]}
### 3) 커스텀 Pruning 함수 호출
prune_wav2vec2_attention(model, heads_to_prune)

# 잘 되었는지 확인(레이어별 num_heads, pruned_heads 상태)
for i, layer in enumerate(model.wav2vec2.encoder.layers):
    attn = layer.attention
    print(f"Layer {i} -> num_heads:{attn.num_heads}, pruned_heads:{getattr(attn, 'pruned_heads', None)}")
'''

# LibriSpeech ASR 데이터셋 로드 (train-clean-100 및 validation-clean)
raw_train_dataset = load_dataset("Sreyan88/librispeech_asr", "clean", split="train.100")
raw_eval_dataset = load_dataset("Sreyan88/librispeech_asr", "clean", split="validation")
# raw_train_dataset = load_dataset("librispeech_asr", "clean", split="train.100")
# raw_eval_dataset = load_dataset("librispeech_asr", "clean", split="test")

###### 데이터셋 크기 줄이기 ###### TODO: Remove
# raw_train_dataset = raw_train_dataset.select(range(1000))
# raw_eval_dataset = raw_eval_dataset.select(range(100))

chars_to_ignore_regex = r"[\,\?\.\!\-\;\:\"\“\%\‘\”\‘]"

def preprocess_function(batch):
    # audio = batch["audio"]
    # text = batch["text"]
    # # print(f'type(audio): {type(audio)}') # <class 'list'>
    # # print(f'len(audio): {len(audio)}') # len(audio): 1000
    # # print(f'type(audio[0]): {type(audio[0])}') # <class 'dict'>
    # # print(f'audio[0].keys(): {audio[0].keys()}') # dict_keys(['path', 'array', 'sampling_rate'])
    

    # # (1) 오디오 -> input_values
    # inputs = processor(
    #     audio,
    #     sampling_rate=audio[0]["sampling_rate"],
    #     return_attention_mask=True
    # )
    # # print(f'len(inputs["input_values"]): {len(inputs["input_values"])}') # 1
    # # print(f'inputs["input_values"][0].shape: {inputs["input_values"][0].shape}') # (seq_len,)
    # # print(f'len(inputs["attention_mask"]): {len(inputs["attention_mask"])}') # 1
    # # print(f'inputs["attention_mask"][0].shape: {inputs["attention_mask"][0].shape}') # (seq_len,)
    # batch["input_values"] = inputs["input_values"]     # (seq_len,)
    # batch["attention_mask"] = inputs["attention_mask"]  # (seq_len,)

    # # (2) 텍스트 -> 라벨
    # with processor.as_target_processor():
    #     labels = processor.tokenizer(
    #         text,
    #         padding=True, # 필요하다면 지정
    #         #  truncation=True, 
    #         # return_attention_mask=False,    # 굳이 생성할 필요 X
    #     )
    # # print(f'len(labels["input_ids"]): {len(labels["input_ids"])}') # 178 <- text를 토큰화한 결과 시퀀스 길이
    # batch["labels"] = labels["input_ids"]
    return batch
    
print(f'raw_train_dataset.column_names : {raw_train_dataset.column_names}')
train_dataset = raw_train_dataset.map(preprocess_function, batched=True)
eval_dataset = raw_eval_dataset.map(preprocess_function, batched=True)
# train_dataset = raw_train_dataset.map(preprocess_function, num_proc=4)
# eval_dataset = raw_eval_dataset.map(preprocess_function, num_proc=4)
print(train_dataset)
print(eval_dataset)


@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    sampling_rate: int = 16000
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        # features = [
        #   {"audio": {"array": ..., "sampling_rate": ...}, "text": ...},
        #   ...
        # ]
        audio_arrays = [f["audio"]["array"] for f in features]
        sampling_rates = [f["audio"]["sampling_rate"] for f in features]
        
        # 오디오 일괄 처리
        inputs = self.processor(
            audio_arrays,
            sampling_rate=sampling_rates[0],  # 가정: 전부 같은 샘플레이트
            return_attention_mask=True,
            padding=True,
            return_tensors="pt"
        )
        # 텍스트 라벨 일괄 처리
        with self.processor.as_target_processor():
            labels = self.processor.tokenizer([f["text"] for f in features],
                                              padding=True, return_tensors="pt")
        
        labels["input_ids"][labels["input_ids"] == self.processor.tokenizer.pad_token_id] = -100

        # 최종 batch(dict) 구성
        batch = {
            "input_values": inputs["input_values"],
            "attention_mask": inputs["attention_mask"],
            "labels": labels["input_ids"]
        }
        return batch

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)


# Pruning 정보 초기화
# model.config.pruned_heads.clear()
# model.config.pruned_heads = {}
model.config.output_attentions = False # Pruning 후에는 attention 출력 끔
# # model_ft = Wav2Vec2ForCTC(config=model.config)
# pruned_encoder = model

# # Wav2Vec2ForCTC 모델 초기화
# model_ft = Wav2Vec2ForCTC.from_pretrained(model_name, config=config)
# # 프루닝된 인코더로 교체
# model_ft.wav2vec2 = pruned_encoder

wer_metric = load("wer")

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    # processor.decode를 이용해 토큰 id를 텍스트로 변환
    pred_str = processor.batch_decode(pred_ids)
    # 라벨도 문자열로 변환
    label_ids = pred.label_ids
    # 패딩(-100) 제거 및 문자열 변환
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    print(f'pred_str: {pred_str}')
    print(f'label_str: {label_str}')
    return {"wer": wer}

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=50,
    per_device_train_batch_size=4,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
    learning_rate=1e-4,
    fp16=True,  # GPU 사용 시
    save_total_limit=4,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
    remove_unused_columns=False,
)


def my_compute_loss_ctc(outputs, labels, processor=None, blank_token_id=None, num_items_in_batch=None):
    """
    CTCLoss를 사용하여 손실을 계산하는 함수.
    outputs: 모델의 출력, logit 값을 포함.
    labels: 정답 시퀀스들 (패딩이 -100으로 표시됨).
    processor: Wav2Vec2Processor 또는 유사 객체 (토크나이저 포함).
    blank_token_id: CTC에서 사용할 blank 토큰 ID. None이면 processor.tokenizer.pad_token_id 사용.
    """
    # blank 토큰 ID 설정 (CTCLoss의 blank 인자로 사용)
    if blank_token_id is None:
        blank_token_id = processor.tokenizer.pad_token_id

    logits = outputs.logits  # shape: [batch_size, T, num_classes]
    
    # CTCLoss는 로그 확률 입력을 기대하므로 log_softmax 적용
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)  # [batch_size, T, num_classes]
    # CTCLoss는 [T, batch_size, num_classes] 형태를 기대하므로 차원 교환
    log_probs = log_probs.transpose(0, 1)  # [T, batch_size, num_classes]
    
    batch_size, T, _ = logits.shape
    device = logits.device

    # 레이블의 실제 길이 계산 (-100이 아닌 토큰의 수)
    label_lengths = (labels != -100).sum(dim=1)  # shape: [batch_size]

    # 각 배치의 레이블에서 패딩(-100)을 제거하고, 모두 연결(concatenate)
    targets_list = [label_seq[label_seq != -100] for label_seq in labels]
    # 빈 시퀀스가 있을 경우를 대비해 필터링
    targets_list = [t for t in targets_list if t.numel() > 0]
    targets = torch.cat(targets_list).to(device) if targets_list else torch.tensor([], dtype=torch.long, device=device)

    # 입력 길이는 모든 시퀀스에서 동일하다고 가정(T)
    input_lengths = torch.full(size=(batch_size,), fill_value=T, dtype=torch.long).to(device)

    # CTCLoss 함수 초기화
    ctc_loss_fn = nn.CTCLoss(blank=blank_token_id, zero_infinity=True)

    # CTCLoss 계산
    loss = ctc_loss_fn(log_probs, targets, input_lengths, label_lengths)
    return loss
    

class Custom_Trainer(Trainer):
    def __init__(self, processor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.processor = processor

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # labels 추출
        labels = inputs.pop("labels")
        # 모델 실행
        outputs = model(**inputs)
        # CTCLoss 계산에 self.processor 사용
        loss = my_compute_loss_ctc(outputs, labels, self.processor)
        return (loss, outputs) if return_outputs else loss    


trainer = Custom_Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,  # 준비된 데이터셋
    eval_dataset=eval_dataset,
    processing_class=processor,
    processor=processor,
    compute_metrics=compute_metrics,
    compute_loss_func=my_compute_loss_ctc,
)
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     data_collator=data_collator,
#     train_dataset=train_dataset,  # 준비된 데이터셋
#     eval_dataset=eval_dataset,
#     tokenizer=processor,
#     compute_metrics=compute_metrics,
#     compute_loss_func=my_compute_loss_ctc(processor=processor),
# )

trainer.train()
