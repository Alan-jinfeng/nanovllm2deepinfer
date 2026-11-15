import torch
from torch import nn
import torch.nn.functional as F


class SiluAndMul(nn.Module):

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 对前半的门控分支应用 SiLU，再与后半的 up 分支逐元素相乘。
        x, y = x.chunk(2, -1)
        return F.silu(x) * y
