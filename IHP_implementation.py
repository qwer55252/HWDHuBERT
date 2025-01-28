import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
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
@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    sampling_rate: int = 16000
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        '''
        features = [
          {"audio": {"array": ..., "sampling_rate": ...}, "text": ...},
          ...
        ]
        '''
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

    # 헤드 간 distance 계산
    for i in tqdm(range(num_heads)):
        for j in range(num_heads):
            dist_val = distance_func(attention_mats[i], attention_mats[j], metric)
            # 유효한 거리 값인지 확인
            if isinstance(dist_val, (int, float)) and np.isfinite(dist_val):
                dist_matrix[i, j] = dist_val
                dist_matrix[j, i] = dist_val
            else:
                # 거리 계산이 불가능한 경우, 최대 거리로 설정
                dist_matrix[i, j] = 2.0
                dist_matrix[j, i] = 2.0

    return dist_matrix

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

def find_pruneable_heads_and_indices(heads, num_heads, head_dim, already_pruned_heads=None, num_attention_heads=None):
    """
    BERT 모델 pruning 로직을 참고해 작성한 유틸 함수.
    `heads`: 제거할 head 번호들의 집합 (ex: {0})
    `num_heads`: 현재 레이어의 전체 head 수 (예: 12)
    `head_dim`: head당 dimension (예: 64)
    `already_pruned_heads`: 이미 제거된 head들의 집합 (ex: {6, 9})
    `num_attention_heads`: 초기 모델의 layer당 head 수 (예: 12)
    """
    if already_pruned_heads is None:
        already_pruned_heads = set()

    # 현재까지 제거된 head들을 합집합으로
    heads = set(heads) - already_pruned_heads
    mask = torch.ones(num_heads, head_dim)
    # heads_to_prune에 해당하는 row는 0으로 만든다
    for head in heads:
        # already_pruned_heads에 현재 head보다 적은 수 count
        less_than_head = 0
        less_than_head += sum(1 if h < head else 0 for h in already_pruned_heads)
        
        mask[head - less_than_head] = 0
        # mask[head] = 0
        
    mask = mask.view(-1).eq(1)
    
    index = torch.arange(num_heads * head_dim)[mask] # num_heads * head_dim == mask.size(0) 여야함
    return heads, index

def prune_wav2vec2_attention_layer(attention_module, heads_to_prune, model_config):
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
    num_attention_heads = model_config.num_attention_heads
    
    # 이미 prune된 head가 있다면, 그 정보를 반영
    already_pruned_heads = getattr(attention_module, "pruned_heads", set())
    
    # 제거할 head와 인덱스 계산
    heads, index = find_pruneable_heads_and_indices(
        heads_to_prune,
        num_heads,
        head_dim,
        already_pruned_heads,
        num_attention_heads
    )
    # `heads`: 제거할 head 번호들의 집합 (ex: {1, 3})
    # `index`: 남은 dim index들의 리스트 (ex: pruning head == 5이면, tensor([0,1,...,319, 384, 385, ... 767])) 이런식으로 320~383번째 dim 제거
    
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
        prune_wav2vec2_attention_layer(layer_module.attention, prune_head_list, model.config)

def preprocess_function(batch):
    return batch

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

def pad_attention_tensors_to_max(avg_attn_per_layer, model_config, already_pruned_heads):
    """
    모든 어텐션 텐서가 동일한 헤드 수를 가지도록 패딩을 적용합니다.
    
    Args:
        avg_attn_per_layer (torch.Tensor): [num_layers, max_heads, seq_len, seq_len]
        
        already_pruned_heads (set): 이미 제거된 헤드들의 집합 (key: layer_idx, value: 제거된 헤드 번호 리스트)
    
    Returns:
        torch.Tensor: 패딩된 텐서들의 스택, shape = [init_num_layer, init_num_head, seq_len, seq_len]
    """
    # TODO: 그냥 12layer, 12head로 padding 하도록 수정. 단 already_pruned_heads는 zero padding을 사이사이에 알맞은 인덱스에 넣어줘야함.
    # 1. 최대 헤드 수 찾기
    init_num_layer = model_config.num_hidden_layers
    init_num_head = model_config.num_attention_heads
    # max_heads = max(tensor.size(0) for tensor in attn_tensors)
    seq_len = avg_attn_per_layer[0].size(1)  # 모든 Head의 seq_len은 동일함
    device = avg_attn_per_layer[0].device
    dtype = avg_attn_per_layer[0].dtype
    
    # 2. 패딩된 텐서 리스트 생성
    padded_head_tensors = []
    padded_layer_tensors = []
    for layer_idx, tensor in enumerate(avg_attn_per_layer):
        # already_pruned_heads[0]: {0, 2, 4}
        head_tensor_list = []
        # 부족한 헤드 수만큼 제로 패딩하는데, 사이사이에 패딩을 넣어줘야 함
        
        cnt = 0
        for head_idx in range(init_num_head):
            if head_idx in already_pruned_heads[layer_idx]:
                cnt += 1
                head_tensor_list.append(torch.zeros(seq_len, seq_len, device=device, dtype=dtype))
            else:
                head_tensor_list.append(avg_attn_per_layer[layer_idx][head_idx - cnt])
        # padded_head_tensors.shape: [init_num_head, seq_len, seq_len]
        padded_head_tensors = torch.stack(head_tensor_list, dim=0)
        padded_layer_tensors.append(padded_head_tensors)
    
    # 3. 텐서 스택
    padded_tensors = torch.stack(padded_layer_tensors, dim=0)  # [init_num_layer, init_num_head, seq_len, seq_len]
    return padded_tensors

def get_avg_attention_matrices(model, processor, dataset, sample_size=10, already_pruned_heads=None): # TODO: sample_size config에서 수정하도록
    """
    랜덤하게 sample_size개를 뽑아, 모델 forward(attention) -> 평균 attention matrix 획득
    반환 shape: (num_layers, num_heads, seq_len, seq_len)
    """
    print(f'---------- START get_avg_attention_matrices ----------')
    # 1) sample_size개 추출
    idxs = random.sample(range(len(dataset)), sample_size)
    sampled_data = [dataset[i] for i in idxs]

    # 2) 전처리 후 모델 forward
    model.eval()
    inputs = data_collator(sampled_data)
    with torch.no_grad():
        outputs = model(
            input_values=inputs["input_values"].to(model.device),
            attention_mask=inputs["attention_mask"].to(model.device),
            output_attentions=True
        )
    # 3) 배치 차원 평균
    # outputs.attentions: (num_layers, B, num_heads, seq_len, seq_len)
    attn_tensors = outputs.attentions  # tuple of length num_layers
    # stack -> (num_layers, B, num_heads, seq_len, seq_len)
    
    avg_attn_per_layer = []
    
    for layer_idx, tensor in enumerate(attn_tensors):
        # tensor shape: [B, num_heads, seq_len, seq_len]
        # 배치 차원(B)에서 평균을 계산하여 [num_heads, seq_len, seq_len] 형태로 변환
        avg_attn = tensor.mean(dim=0)  # [num_heads, seq_len, seq_len]
        avg_attn_per_layer.append(avg_attn.cpu())
        print(f'Layer {layer_idx} avg_attn shape: {avg_attn.shape}')
    
    

    # 4) 패딩을 적용하여 스택
    padded_avg_attn = pad_attention_tensors_to_max(avg_attn_per_layer, model.config, already_pruned_heads)  # [num_layers, max_heads, seq_len, seq_len]
    print(f'padded_avg_attn shape: {padded_avg_attn.shape}') # padded_avg_attn shape: torch.Size([12, 12, 764, 764])
    print(f'---------- END get_avg_attention_matrices ----------')
    return padded_avg_attn

    # attn_stack = torch.stack(attn_tensors, dim=0)  # shape 동일
    # avg_attn = attn_stack.mean(dim=1)              # (num_layers, num_heads, seq_len, seq_len)
    # return avg_attn.cpu()

def cluster_and_select_heads(attn_matrices, distance_metric="cosine", n_remove=5, current_num_heads=None, already_pruned_heads=None):
    """
    attn_matrices: shape = (num_layers, num_heads, seq_len, seq_len)
    distance_metric: "corr", "cosine", ...
    n_remove: 이번 단계에서 제거할 head 개수 (전체를 통틀어)
    반환: 제거해야 할 head들의 (layer_idx, head_idx) 리스트
    """
    total_heads_current = current_num_heads
    seq_len = attn_matrices.size(-1)
    print(f'')

    # (num_layers*num_heads, seq_len, seq_len)로 reshape
    reshaped_attn = attn_matrices.view(-1, seq_len, seq_len)  # shape: (total_heads_current, seq_len, seq_len)
    # reshaped_attn 자체가 padding을 거쳐서 나와서 index 정보 훼손됨

    # 1) 헤드 간 distance matrix 계산
    print(f" - Calculating pairwise distances ...")
    distance_matrix = compute_pairwise_distances(
            reshaped_attn,
            distance_func=get_token_based_distance,
            mode="token",
            metric=distance_metric
        )
    # if not is_test:
    #     distance_matrix = compute_pairwise_distances(
    #         reshaped_attn,
    #         distance_func=get_token_based_distance,
    #         mode="token",
    #         metric=distance_metric
    #     )
    # else:
    #     # 테스트용: 0~2 범위의 랜덤 distance matrix 생성
    #     distance_matrix = np.random.rand(total_heads_current, total_heads_current) * 2.0
    print(f" - Distance matrix shape: {distance_matrix.shape}")
    # 2) 0~1 범위로 간단 정규화(예: 0~2 구간이라고 가정 후 /2)
    distance_matrix = distance_matrix / 2.0
    # 3) 유사도 행렬 = 1 - distance
    similarity_matrix = 1.0 - distance_matrix

    # 4) 스펙트럴 클러스터링
    #    이번에 제거할 head 개수가 n_remove -> 남길 head 개수 = total_heads_current - n_remove
    #    => 클러스터 수 = 남길 head 개수(각 클러스터에서 대표 head 1개)
    n_clusters = max(1, total_heads_current - n_remove)  # 최소 1
    clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', assign_labels='kmeans', random_state=42)
    cluster_labels = clustering.fit_predict(similarity_matrix)
    # TODO: 이미 pruned 된 similairy_matrix를 사용하면, cluster_labels가 이상하게 나올 수 있음
    # already_pruned_head를 참고해서 similarity_matrix에서 index i가 실제로 몇번째 layer의 몇번째 head인지 알아내야 함

    
    # # cluster_labels에 이미 prunede된 head는 -1로 표시
    # for i in range(total_heads_current):

    
    print(f'cluster_labels: {cluster_labels}')

    # 혹시 silhouette_score를 확인해볼 수 있음(거리 행렬 사용)
    try:
        sc_score = silhouette_score(distance_matrix, cluster_labels, metric='precomputed')
        print(f" - SpectralClustering Silhouette Score: {sc_score:.4f}")
    except:
        pass

    # 5) 클러스터마다 대표 head 1개씩 선택(간단히 클러스터마다 랜덤하게)
    cluster_to_rep = {}
    cluster_head_dict = {}

    for head_idx, c_id in enumerate(cluster_labels):
        if c_id == -1:
            continue
        cluster_head_dict[c_id] = cluster_head_dict.get(c_id, []) + [head_idx]
    print(f'cluster_head_dict: {cluster_head_dict}')

    for c_id, head_indices in cluster_head_dict.items():
        rep_head_idx = np.random.choice(head_indices)  # 랜덤 대표
        cluster_to_rep[c_id] = rep_head_idx
    print(f'cluster_to_rep: {cluster_to_rep}')

    # 6) 전체 head 중, 대표로 선택된 head만 keep. 나머지는 prune
    keep_set = set(cluster_to_rep.values())
    all_heads_set = set(range(total_heads_current))
    prune_set = all_heads_set - keep_set  # 이번 단계에서 제거할 head의 global index

    # (layer_idx, head_idx)로 변환
    remove_list = []
    num_heads = attn_matrices.size(1)
    # TODO: 이거 layer마다 head 개수가 다른데 이렇게 하면 안돼
    
    
    
    for g_idx in prune_set:
        layer_idx = g_idx // num_heads
        head_idx_in_layer = g_idx % num_heads
        remove_list.append((layer_idx, head_idx_in_layer))
    print(f" - Heads to remove: {remove_list}")
    return remove_list



# TODO: argparse 사용 + config 파일 사용하도록 수정 필요
# TODO: 데이터 수 늘려야함 + 전체적인 하이퍼파라미터 수정 필요

### Stage 1. 모델 및 데이터셋 설정
sample_rate = 16000
duration = 1 # seconds
batch_size = 10
audio_tensor = torch.randn(batch_size, sample_rate * duration) # (B, T)
output_dir="IHP_implementation"
is_test = False
total_pruning_iterations = 10
pruning_ratio = 0.2  # 최종적으로 전체 헤드의 20%만 남기고 싶다고 가정
iterative_finetune_epochs = 2 # total fine-tuning epochs: 2*10 epochs
final_finetune_epochs = 50
per_device_train_batch_size = 4
distance_metric = "cosine" # token -> "cosine", "corr", "js", "bc" 중 선택 / sentence -> "dCor", "PC", "CC" 중 선택

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


### Stage 2. 데이터셋 준비
# LibriSpeech ASR 데이터셋 로드 (train-clean-100 및 validation-clean)
raw_train_dataset = load_dataset("Sreyan88/librispeech_asr", "clean", split="train.100")
raw_eval_dataset = load_dataset("Sreyan88/librispeech_asr", "clean", split="validation") # TODO: 비교를 위해서 validation 쓰고 있는데, test로 바꿔야함

chars_to_ignore_regex = r"[\,\?\.\!\-\;\:\"\“\%\‘\”\‘]"

print(f'raw_train_dataset.column_names : {raw_train_dataset.column_names}')
# train_dataset = raw_train_dataset.map(preprocess_function, batched=True)
# eval_dataset = raw_eval_dataset.map(preprocess_function, batched=True)
train_dataset = raw_train_dataset.map(preprocess_function, num_proc=4)
eval_dataset = raw_eval_dataset.map(preprocess_function, num_proc=4)
if is_test: # 100개 데이터만 사용
    train_dataset = train_dataset.select(range(100))
    eval_dataset = eval_dataset.select(range(100))
print(train_dataset)
print(eval_dataset)

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

### Stage 3. Iterative Head Pruning 수행
model.config.output_attentions = False # Pruning 후에는 attention 출력 끔
wer_metric = load("wer")
training_args_1 = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=iterative_finetune_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
    learning_rate=1e-4,
    fp16=True,  # GPU 사용 시
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
    remove_unused_columns=False,
)


trainer_1 = Custom_Trainer(
    model=model,
    args=training_args_1,
    data_collator=data_collator,
    train_dataset=train_dataset,  # 준비된 데이터셋
    eval_dataset=eval_dataset,
    processing_class=processor,
    processor=processor,
    compute_metrics=compute_metrics,
    compute_loss_func=my_compute_loss_ctc,
)

# 1) 초기 설정
init_num_layers = model.config.num_hidden_layers
init_num_heads = model.config.num_attention_heads
initial_total_heads = init_num_layers * init_num_heads
# (전체 헤드 중 (1 - pruning_ratio)만큼을 10회에 걸쳐 제거)
heads_to_remove_per_step = int( initial_total_heads * (1 - pruning_ratio) / total_pruning_iterations )
print(f"[초기설정] iteration x {total_pruning_iterations}, 최종 pruning_ratio={pruning_ratio}")
print(f"한 번에 제거할 head 수: {heads_to_remove_per_step}")

# 각 단계별로 이미 제거된 heads를 추적하기 위해, layer 단위로 set 사용
already_pruned_heads_dict = {layer_idx: set() for layer_idx in range(init_num_layers)}

# ======================== #
# 실제 Iterative Pruning 루프
# ======================== #
current_model = model
current_model.config.output_attentions = True  # attention 추출 허용

for iteration in range(total_pruning_iterations):
    print(f"\n[Iteration {iteration+1}/{total_pruning_iterations}]")

    # 3) 헤드 평가용 attention 추출 (랜덤 10샘플)
    avg_attn_matrices = get_avg_attention_matrices(
        current_model, processor, train_dataset, sample_size=10, already_pruned_heads=already_pruned_heads_dict
    )
    # avg_attn_matrices: Tensor(init_num_layers, init_num_heads, seq_len, seq_len)

    # 현재 남아있는 총 head 수
    current_num_heads = 0
    for layer_idx in range(init_num_layers):
        # 레이어별로 실제 남은 head 수 = model.wav2vec2.encoder.layers[layer_idx].attention.num_heads
        current_num_heads += current_model.wav2vec2.encoder.layers[layer_idx].attention.num_heads
    print(f" - 현재 남아있는 전체 헤드 수: {current_num_heads}")

    # 이번 단계에서 제거할 head 수 계산(절대 개수)
    n_remove = heads_to_remove_per_step
    # 만약 현재 남은 head 수가 n_remove보다 작으면, 조정
    if current_num_heads <= n_remove:
        n_remove = current_num_heads - 1  # 최소 1개는 남김

    # 4) 클러스터링 -> 제거 대상 head 선정
    remove_candidates = cluster_and_select_heads(avg_attn_matrices, distance_metric=distance_metric, n_remove=n_remove, current_num_heads=current_num_heads, already_pruned_heads=already_pruned_heads_dict)
    # remove_candidates: [(layer_idx, head_idx_in_layer), ...]
    print(f''' - 제거할 head 후보: {remove_candidates}''')

    # heads_to_prune_dict에 반영
    # 이미 제거된 head는 빼고, 새롭게 제거될 head만 추가
    layer_to_prune_dict = {} # ex) {0: [1,3,5,7,9,11], 1: [0,2,4,6,8,10], ...}
    for (lyr_idx, h_idx) in remove_candidates:
        # 이미 제거된 head면 skip
        if h_idx in already_pruned_heads_dict[lyr_idx]:
            continue
        layer_to_prune_dict.setdefault(lyr_idx, []).append(h_idx)
    print(f" - Heads to prune: {layer_to_prune_dict}")
    # 실제 prune 실행 (5) 헤드 프루닝
    prune_wav2vec2_attention(current_model, layer_to_prune_dict)

    # already_pruned_heads_dict 갱신
    for lyr_idx, h_idxs in layer_to_prune_dict.items():
        for h_idx in h_idxs:
            already_pruned_heads_dict[lyr_idx].add(h_idx)
    print(f' - 현재까지 제거된 head: {already_pruned_heads_dict}')
    print(f" - Iteration {iteration+1} 프루닝 완료")
    
    # 5) 프루닝 후 미세조정(Fine-tuning)
    #    - 여기서는 시간 절약을 위해 적은 epoch로 실행
    current_model.config.output_attentions = False
    trainer_1.model = current_model  # Trainer에 프루닝된 모델 갱신
    print(f" - Iteration {iteration+1} Fine-Tuning 시작")
    trainer_1.train()

    # 6) 모델 평가
    eval_metrics = trainer_1.evaluate()
    print(f" - Iteration {iteration+1} 평가 결과(WER): {eval_metrics['eval_wer']:.4f}")

    # (다음 iteration에서 다시 attention 뽑을 때를 위해)
    current_model.config.output_attentions = True

current_model.config.output_attentions = False

print("\n[모든 Iteration 종료]")
print("최종 프루닝 완료 후 모델 저장/사용 등 후속 작업을 진행하세요.")

# 필요한 경우 최종 모델 저장
trainer_1.save_model(output_dir + "/final_pruned_model")



### Stage 4. Final Fine-Tuning
# 저장한 최종 모델로 최종 fine-tuning
print("\n[최종 Fine-tuning 시작]")
# 최종 fine-tuning을 위한 Trainer 설정
training_args_2 = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=final_finetune_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
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
trainer_2 = Custom_Trainer(
    model=trainer_1.model,
    args=training_args_2,
    data_collator=data_collator,
    train_dataset=train_dataset,  # 준비된 데이터셋
    eval_dataset=eval_dataset,
    processing_class=processor,
    processor=processor,
    compute_metrics=compute_metrics,
    compute_loss_func=my_compute_loss_ctc,
)

# 최종 fine-tuning 실행
trainer_2.train()

print("\n[모든 작업 완료]")

trainer_2 = Custom_Trainer(
    model=model,
    args=training_args_2,
    data_collator=data_collator,
    train_dataset=train_dataset,  # 준비된 데이터셋
    eval_dataset=eval_dataset,
    processing_class=processor,
    processor=processor,
    compute_metrics=compute_metrics,
    compute_loss_func=my_compute_loss_ctc,
)


trainer_2.train()
