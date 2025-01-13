import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt
from transformers import Wav2Vec2Model, Wav2Vec2Processor

# 1. 모델과 프로세서 로드
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-100h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-100h")

# 모델을 evaluation 모드로 전환
model.eval()

# 2. 예시 오디오 파일 불러오기 (16kHz 모노 채널 권장).
#    만약 실제 음성 파일이 없다면 dummy tensor를 이용할 수도 있습니다.
#    아래는 librosa로 예시 파일을 불러오는 코드입니다.
audio_filepath = "/home/kobie/workspace/data/LibriSpeech/dev-clean/84/121123/84-121123-0000.flac"  # 실제 파일 경로로 수정
waveform, sr = librosa.load(audio_filepath, sr=16000, mono=True)

# 3. 입력 전처리
input_values = processor(waveform, sampling_rate=16000, return_tensors="pt").input_values

# 4. Attention 출력을 받기 위해 forward 시 output_attentions=True 설정
with torch.no_grad():
    outputs = model(
        input_values, 
        output_attentions=True  # <-- Multi-Head Attention 가중치를 반환
    )

# 5. 각 레이어별 Attention(=outputs.attentions)은 리스트 형태로 반환됩니다.
#    len(outputs.attentions) = num_hidden_layers (Wav2Vec2-base의 경우 12)
#    각 요소 shape = (batch_size, num_heads, seq_len, seq_len)
attentions = outputs.attentions

print(f"Number of transformer layers: {len(attentions)}")
print(f"Shape of attention in layer[0]: {attentions[0].shape}")

# 6. Head 간 redundancy를 살펴보기 위한 간단한 함수 예시
#    여기서는 'attention matrix'를 펼친 뒤 헤드 간 코사인 유사도를 계산해 봅니다.
def compute_head_redundancy(att_mats):
    """
    att_mats: 텐서 shape = (num_heads, seq_len, seq_len)
    return:   head 간 유사도 행렬, shape = (num_heads, num_heads)
    """
    num_heads = att_mats.size(0)
    seq_len = att_mats.size(1)
    
    # (num_heads, seq_len*seq_len)으로 reshape
    att_reshaped = att_mats.view(num_heads, -1)
    
    # 코사인 유사도 계산
    # 유클리디안 노름
    norm = att_reshaped.norm(dim=1, keepdim=True)  # shape = (num_heads, 1)
    att_normed = att_reshaped / (norm + 1e-9)

    # (num_heads, num_heads) = att_normed @ att_normed.T
    similarity_matrix = att_normed @ att_normed.transpose(0, 1)
    return similarity_matrix

# 7. 예시로 첫 번째 레이어의 Head들 간 유사도를 계산해보고 시각화
layer_idx = 0
att_mat = attentions[layer_idx][0]  # batch_size=1 가정 => (num_heads, seq_len, seq_len)
similarity = compute_head_redundancy(att_mat)

print(f"Layer {layer_idx} Head similarity matrix:\n{similarity}")

# 8. 시각화(Heatmap) 예시
plt.imshow(similarity.cpu().numpy(), cmap='viridis', interpolation='nearest')
plt.colorbar()
plt.title(f"Head Similarity Matrix (Layer {layer_idx})")
plt.xlabel("Head Index")
plt.ylabel("Head Index")
plt.show()

# 9. 전 레이어에 대해 평균적인 redundancy를 보는 간단한 예시
#    레이어별 모든 Head 쌍 간의 평균 코사인 유사도를 출력
all_layer_sims = []

for l_idx, att in enumerate(attentions):
    # att shape = (batch_size, num_heads, seq_len, seq_len)
    # 여기서는 batch_size=1 가정
    att_mat = att[0]  # (num_heads, seq_len, seq_len)
    sim_matrix = compute_head_redundancy(att_mat)
    # Head 쌍별 평균
    mean_sim = (sim_matrix.sum() - sim_matrix.diag().sum()) / (sim_matrix.numel() - sim_matrix.size(0))
    all_layer_sims.append(mean_sim.item())

    print(f"Layer {l_idx} - Average off-diagonal Head similarity: {mean_sim.item():.4f}")

# 10. 레이어별 유사도를 간단히 그래프로 표시
plt.plot(range(len(all_layer_sims)), all_layer_sims, marker='o')
plt.title("Average Off-Diagonal Head Similarity per Layer")
plt.xlabel("Layer")
plt.ylabel("Avg Head Similarity")
plt.show()
