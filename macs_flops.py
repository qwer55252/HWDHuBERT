import argparse
import torch
from ptflops import get_model_complexity_info
from safetensors.torch import load_file
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, TrainingArguments, Trainer, Wav2Vec2Config
from transformers.pytorch_utils import prune_linear_layer

# ──────────────────────────────────────────────────────────────#
# 프루닝 관련 유틸 함수 (코드 내부 주석 참고)
# ──────────────────────────────────────────────────────────────#
def find_pruneable_heads_and_indices(heads, num_heads, head_dim, already_pruned_heads=None, num_attention_heads=None):
    if already_pruned_heads is None:
        already_pruned_heads = set()

    heads = set(heads) - already_pruned_heads
    mask = torch.ones(num_heads, head_dim)
    for head in heads:
        less_than_head = sum(1 for h in already_pruned_heads if h < head)
        mask[head - less_than_head] = 0  # 해당 head의 row를 제거
    mask = mask.view(-1).eq(1)
    index = torch.arange(num_heads * head_dim)[mask]
    return heads, index

def prune_wav2vec2_attention_layer(attention_module, heads_to_prune, model_config):
    if not heads_to_prune:
        return  # 제거할 head가 없으면 아무 작업도 수행하지 않음

    num_heads, head_dim = attention_module.num_heads, attention_module.head_dim
    num_attention_heads = model_config.num_attention_heads
    already_pruned_heads = getattr(attention_module, "pruned_heads", set())
    heads, index = find_pruneable_heads_and_indices(
        heads_to_prune,
        num_heads,
        head_dim,
        already_pruned_heads,
        num_attention_heads
    )
    # Q, K, V, 그리고 Out projection에 대해 pruning 수행
    attention_module.q_proj = prune_linear_layer(attention_module.q_proj, index, dim=0)
    attention_module.k_proj = prune_linear_layer(attention_module.k_proj, index, dim=0)
    attention_module.v_proj = prune_linear_layer(attention_module.v_proj, index, dim=0)
    attention_module.out_proj = prune_linear_layer(attention_module.out_proj, index, dim=1)
    
    # pruned head 정보를 업데이트
    attention_module.num_heads -= len(heads)
    attention_module.all_head_size = attention_module.num_heads * attention_module.head_dim
    attention_module.pruned_heads = already_pruned_heads.union(heads)

def prune_wav2vec2_attention(model, heads_to_prune_dict):
    """
    heads_to_prune_dict 예시:
      {
          0: [1, 2],   # layer 0에서 head 1,2 제거
          3: [0, 5, 7] # layer 3에서 head 0,5,7 제거
      }
    """
    for layer_idx, prune_head_list in heads_to_prune_dict.items():
        layer_module = model.wav2vec2.encoder.layers[layer_idx]
        prune_wav2vec2_attention_layer(layer_module.attention, prune_head_list, model.config)

# ──────────────────────────────────────────────────────────────#
# 체크포인트의 state dict에서 이미 프루닝된 head 정보를 추출
# ──────────────────────────────────────────────────────────────#
def derive_already_pruned_heads(state_dict, original_num_heads=12, head_dim=64):
    already_pruned_heads_dict = {}
    for key, tensor in state_dict.items():
        if "attention.q_proj.weight" in key:
            # key 예: "wav2vec2.encoder.layers.0.attention.q_proj.weight"
            layer = int(key.split('.')[3])
            # tensor.shape[0] = (남은 head 수) * head_dim
            num_heads = tensor.shape[0] // head_dim
            # 원래 head 개수에서 남은 head 개수를 빼서 제거된 head 번호(0부터 순서대로)를 생성
            already_pruned_heads_dict[layer] = [i for i in range(original_num_heads - num_heads)]
    return already_pruned_heads_dict

# ptflops를 위한 더미 입력 생성자 정의
def input_constructor(input_res):
    # input_res: (input_length,)
    return {"input_values": torch.randn(1, input_res[0])}
    
# ──────────────────────────────────────────────────────────────#
# 모델 MACs 계산 코드
# ──────────────────────────────────────────────────────────────#
def main():
    parser = argparse.ArgumentParser(
        description="계산된 프루닝 정보를 반영한 Wav2Vec2 모델의 MACs를 계산합니다."
    )
    parser.add_argument(
        "--model_name_or_path", 
        type=str, 
        default="facebook/wav2vec2-base-100h",
        help="원본 모델 이름 또는 경로"
    )
    parser.add_argument(
        "--model_checkpoint", 
        type=str, 
        required=True,
        help="프루닝 후 저장된 체크포인트 디렉토리 (model.safetensors 파일 포함)"
    )
    parser.add_argument(
        "--input_length",
        type=int,
        default=16000,
        help="더미 입력의 길이 (샘플 수, 예: 16000 = 1초 오디오)"
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
    
    macs, params = get_model_complexity_info(
        model, 
        (args.input_length,), 
        input_constructor=input_constructor, 
        as_strings=True, 
        print_per_layer_stat=True
    )
    print("────────────────────────────")
    print("pruning 하기 전")
    print("모델 MACs:", macs)
    print("모델 Parameters:", params)
    print("────────────────────────────")
    
    already_pruned_heads_dict = {}
    
    for key, tensor in state_dict.items(): # key ex: wav2vec2.encoder.layers.0.attention.q_proj.bias
        if "attention.q_proj.weight" in key:
            layer = int(key.split('.')[3]) # ex) 0
            num_heads = tensor.shape[0] // 64
            already_pruned_heads_dict[layer] = [i for i in range(12-num_heads)]
    
    
    prune_wav2vec2_attention(model, already_pruned_heads_dict)
    model.config.output_attentions = False
    
    # pruning된 모델 구조 잘 반영됐는지 확인
    for i, layer in enumerate(model.wav2vec2.encoder.layers):
        head_dim = layer.attention.head_dim
        new_num_heads = layer.attention.q_proj.weight.shape[0] // head_dim
        print(f"Layer {i}: new_num_heads = {new_num_heads}")
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    

    # 모델 MACs와 parameter 수 계산 (입력은 1초 오디오로 가정)
    macs, params = get_model_complexity_info(
        model, 
        (args.input_length,), 
        input_constructor=input_constructor, 
        as_strings=True, 
        print_per_layer_stat=True
    )
    print("────────────────────────────")
    print("모델 MACs:", macs)
    print("모델 Parameters:", params)
    print("────────────────────────────")

if __name__ == "__main__":
    main()
