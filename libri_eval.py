import torch
import numpy as np
import argparse
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, TrainingArguments, Trainer, Wav2Vec2Config
from dataclasses import dataclass
from evaluate import load
from safetensors.torch import load_file
from transformers.pytorch_utils import prune_linear_layer
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2PreTrainedModel
import json

# 기존 코드에서 사용한 DataCollatorCTCWithPadding 클래스 (변경 없이 사용)
@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: bool = True
    sampling_rate: int = 16000
    max_length: int = None
    pad_to_multiple_of: int = None

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
            if "text" in features[0].keys():
                labels = self.processor.tokenizer([f["text"] for f in features], padding=True, return_tensors="pt")
            else:
                label_lists = [
                    f["label"] if isinstance(f["label"], list) else [f["label"]]
                    for f in features
                ]
                labels = self.processor.tokenizer.pad(
                    {"input_ids": label_lists}, padding=True, return_tensors="pt"
                )
        
        labels["input_ids"][labels["input_ids"] == self.processor.tokenizer.pad_token_id] = -100 # pad 토큰을 -100으로 대체 (loss 계산시 무시)
        
        

        # 최종 batch(dict) 구성
        batch = {
            "input_values": inputs["input_values"],
            "attention_mask": inputs["attention_mask"],
            "labels": labels["input_ids"]
        }
        return batch

# 평가 시 사용될 compute_metrics 함수 (WER 계산)
def compute_metrics(pred, processor):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    # 토큰 id를 텍스트로 변환
    pred_str = processor.batch_decode(pred_ids)
    # 정답 label 변환 (패딩 토큰(-100)을 실제 pad_token_id로 변경)
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(label_ids, group_tokens=False)
    wer_metric = load("wer")
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

def preprocess_function(batch):
    # 여기서는 추가 전처리가 필요 없다면 원본 그대로 반환
    return batch

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




# 커스텀 어텐션 모듈: 각 레이어마다 다른 헤드 수를 지원
class CustomWav2Vec2Attention(nn.Module):
    def __init__(self, config, num_heads):
        super().__init__()
        self.num_heads = num_heads
        # 원래 모델은 config.hidden_size를 config.num_attention_heads(보통 12)로 나눈 head_dim을 사용
        # 프루닝 후에도 head_dim은 그대로 유지한다고 가정 (예: 768/12=64)
        self.head_dim = config.hidden_size // 12  
        self.all_head_size = self.num_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(config.hidden_size, self.all_head_size)
        self.k_proj = nn.Linear(config.hidden_size, self.all_head_size)
        self.v_proj = nn.Linear(config.hidden_size, self.all_head_size)
        self.out_proj = nn.Linear(self.all_head_size, config.hidden_size)
        self.dropout = nn.Dropout(config.attention_dropout)

    def forward(self, hidden_states, attention_mask=None):
        # hidden_states: (batch, seq_len, hidden_size)
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        batch_size, seq_length, _ = hidden_states.size()

        # (batch, seq_len, num_heads, head_dim) -> (batch, num_heads, seq_len, head_dim)
        query = query.view(batch_size, seq_length, self.num_heads, -1).transpose(1, 2)
        key = key.view(batch_size, seq_length, self.num_heads, -1).transpose(1, 2)
        value = value.view(batch_size, seq_length, self.num_heads, -1).transpose(1, 2)

        # attention score: (batch, num_heads, seq_len, seq_len)
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scaling
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # context: (batch, num_heads, seq_len, head_dim)
        context = torch.matmul(attn_probs, value)
        context = context.transpose(1, 2).reshape(batch_size, seq_length, self.all_head_size)
        output = self.out_proj(context)
        return output

# 커스텀 Encoder Layer: 커스텀 어텐션 모듈을 사용
class CustomWav2Vec2EncoderLayer(nn.Module):
    def __init__(self, config, num_heads):
        super().__init__()
        self.attention = CustomWav2Vec2Attention(config, num_heads)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 간단한 feed-forward block (원래 모델보다 단순화됨)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout),
            nn.Linear(config.intermediate_size, config.hidden_size),
        )
        self.final_dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, hidden_states, attention_mask=None):
        # Self-attention block
        attn_output = self.attention(hidden_states, attention_mask)
        hidden_states = self.layer_norm(hidden_states + self.dropout(attn_output))
        # Feed-forward block
        ff_output = self.feed_forward(hidden_states)
        hidden_states = self.layer_norm(hidden_states + self.final_dropout(ff_output))
        return hidden_states

# 커스텀 Encoder: 각 레이어마다 다른 헤드 수를 적용
class CustomWav2Vec2Encoder(nn.Module):
    def __init__(self, config, pruned_num_heads_list):
        super().__init__()
        self.layers = nn.ModuleList([
            CustomWav2Vec2EncoderLayer(config, num_heads)
            for num_heads in pruned_num_heads_list
        ])
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, attention_mask=None):
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        hidden_states = self.layer_norm(hidden_states)
        return hidden_states

# 커스텀 Wav2Vec2 Model: 기존 feature extractor는 그대로 사용 (여기서는 간략화)
class CustomWav2Vec2Model(Wav2Vec2PreTrainedModel):
    def __init__(self, config, pruned_num_heads_list):
        super().__init__(config)
        # feature extractor는 원래 모델과 동일한 역할을 수행 (예: convolutional layers)
        from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2FeatureExtractor
        self.feature_extractor = Wav2Vec2FeatureExtractor(config)
        self.encoder = CustomWav2Vec2Encoder(config, pruned_num_heads_list)
        self.init_weights()

    def forward(self, input_values, attention_mask=None):
        # feature extraction (출력 shape: (batch, seq_len, hidden_size))
        hidden_states = self.feature_extractor(input_values)
        hidden_states = self.encoder(hidden_states, attention_mask)
        return hidden_states

# 커스텀 Wav2Vec2ForCTC: CTC head를 붙여 최종 음성 인식 모델을 구성
class CustomWav2Vec2ForCTC(Wav2Vec2PreTrainedModel):
    def __init__(self, config, pruned_num_heads_list):
        super().__init__(config)
        self.wav2vec2 = CustomWav2Vec2Model(config, pruned_num_heads_list)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        self.init_weights()

    def forward(self, input_values, attention_mask=None, labels=None):
        hidden_states = self.wav2vec2(input_values, attention_mask)
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            # CTC loss 계산 (간략화된 예시)
            loss_fn = nn.CTCLoss(blank=self.config.pad_token_id, zero_infinity=True)
            log_probs = F.log_softmax(logits, dim=-1)
            input_lengths = torch.full((logits.size(0),), logits.size(1), dtype=torch.long)
            target_lengths = torch.sum(labels != -100, dim=-1)
            loss = loss_fn(log_probs.transpose(0, 1), labels, input_lengths, target_lengths)
        return {"loss": loss, "logits": logits}





def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned Wav2Vec2 model on LibriSpeech test datasets.")
    parser.add_argument(
        "--already_pruned_heads_dict", 
        type=json.loads,  # json.loads로 변경
        required=True,
        help="프루닝된 헤드 정보 딕셔너리"
    )
    parser.add_argument(
        "--model_name_or_path", 
        type=str, 
        default="facebook/wav2vec2-base-100h",
        help="모델 이름 또는 경로"
    )
    parser.add_argument(
        "--model_checkpoint", 
        type=str, 
        required=True,
        help="최종 fine-tuning 또는 프루닝 후 저장된 모델 체크포인트 경로"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True,
        help="평가 결과를 저장할 디렉토리 경로"
    )
    parser.add_argument(
        "--test_split",
        type=str,
        required=True,
        help="평가할 데이터셋 split 이름 (test.clean 또는 test.other)"        
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=4,
        help="각 디바이스 당 평가 배치 크기"
    )
    args = parser.parse_args()

    # 모델 및 프로세서 로드
    state_dict = load_file(f"{args.model_checkpoint}/model.safetensors") # TODO: dir로부터 load하게 수정
    

    
    for key, tensor in state_dict.items(): # key ex: wav2vec2.encoder.layers.0.attention.q_proj.bias
        if "attention.q_proj.weight" in key:
            num_heads = tensor.shape[0] // 64
            print(f"{key}: {tensor.shape}, remaining heads: {num_heads}")
    
    config = Wav2Vec2Config.from_pretrained(args.model_name_or_path, output_attentions=True)
    processor = Wav2Vec2Processor.from_pretrained(args.model_name_or_path, config=config)
    model = Wav2Vec2ForCTC.from_pretrained(
        args.model_name_or_path, 
        output_attentions=False,
        ctc_loss_reduction="mean", 
        pad_token_id=processor.tokenizer.pad_token_id,
    )
    
    already_pruned_heads_dict = args.already_pruned_heads_dict # dict인데 str로 받아서 dict로 변환 필요 
    already_pruned_heads_dict = {int(k): v for k, v in already_pruned_heads_dict.items()} # ex) {0: [1,3,5,7,9,11], 1: [0,2,4,6,8,10], ...}
    
    
    
    
    prune_wav2vec2_attention(model, already_pruned_heads_dict)
    model.config.output_attentions = False
    
    # pruning된 모델 구조 잘 반영됐는지 확인
    for i, layer in enumerate(model.wav2vec2.encoder.layers):
        head_dim = layer.attention.head_dim
        new_num_heads = layer.attention.q_proj.weight.shape[0] // head_dim
        print(f"Layer {i}: new_num_heads = {new_num_heads}")
    
    
    # model_checkpoint 가중치 덧씌우기
    model.load_state_dict(state_dict, strict=False)



    # LibriSpeech test-clean, test-other 데이터셋 로드
    test_dataset = load_dataset("./librispeech_asr_test.py", "clean_other_test", split=args.test_split) # test-clean, test-other 둘 다 사용할 경우

    # 필요한 경우 전처리 (예제에서는 identity mapping)
    test_dataset = test_dataset.map(preprocess_function, num_proc=4)
    print(f'test_dataset.column_names: {test_dataset.column_names}')
    print(f'test_dataset: {test_dataset}')

    # Data Collator 생성
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    model.eval()
    
    # Trainer에서 사용할 TrainingArguments (평가용이므로 output_dir 등은 간단하게 설정)
    eval_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        fp16=True,
        logging_steps=10,
        do_eval=True,
        remove_unused_columns=False,
    )

    # Trainer 인스턴스 생성 (compute_metrics는 lambda를 통해 processor 전달)
    trainer = Trainer(
        model=model,
        args=eval_args,
        data_collator=data_collator,
        compute_metrics=lambda pred: compute_metrics(pred, processor),
        eval_dataset=test_dataset,
    )
    
    

    # 평가: test
    print("== LibriSpeech test 데이터셋 평가 시작 ==")
    eval_result = trainer.evaluate(eval_dataset=test_dataset)
    print(f"{args.test_split} WER: {eval_result['eval_wer']:.4f}")

    
if __name__ == "__main__":
    main()
