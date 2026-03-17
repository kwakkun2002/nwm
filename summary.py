import torch
import torchinfo
from torchviz import make_dot
from models import CDiTBlock

# CDiTBlock 생성
model = CDiTBlock(
    hidden_size=384,
    num_heads=6,
)

# CDiTBlock forward에 필요한 입력: x, c, x_cond
# x: (batch, num_patches, hidden_size) - 임베딩된 패치들
# c: (batch, hidden_size) - conditioning 벡터
# x_cond: (batch, context_size * num_patches, hidden_size) - conditioning 컨텍스트

# input_size (1, 384, 32, 32)가 (batch, channels, height, width)를 의미한다고 가정
# patch_size=2로 patch embedding 후: num_patches = (32/2)^2 = 256
num_patches = (32 // 2) ** 2  # 256 패치
context_size = 2  # 일반적인 컨텍스트 크기

# 더미 입력 생성
dummy_x = torch.randn(1, num_patches, 384)
dummy_c = torch.randn(1, 384)
dummy_x_cond = torch.randn(1, context_size * num_patches, 384)

# torchinfo는 input_data 파라미터로 튜플을 받아 여러 입력을 처리할 수 있음
torchinfo.summary(
    model,
    input_data=(dummy_x, dummy_c, dummy_x_cond),
    col_names=("input_size", "output_size", "num_params", "mult_adds"),
)