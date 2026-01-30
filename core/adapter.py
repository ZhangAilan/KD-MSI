import torch.nn as nn

class DinoToClipProjector(nn.Module):
    def __init__(self, in_dim=1024, out_dim=768, hidden_dim=None):
        """
        Args:
            in_dim: DINO 特征维度 (默认为 1024)
            out_dim: CLIP 特征维度 (默认为 768)
            hidden_dim: 中间层维度，如果不指定，默认保持与输入维度一致 (1024)，以防止信息过早丢失
        """
        super().__init__()
        
        # 如果未指定中间维度，默认保持 1024 以保留足够的信息量
        if hidden_dim is None:
            hidden_dim = in_dim

        # 第一层：特征变换 + 非线性激活
        self.fc1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.LeakyReLU(inplace=False)
        )

        # 第二层：降维投影 + 非线性激活
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim, bias=False),
            nn.LeakyReLU(inplace=False) 
        )

    def forward(self, x):
        # 依次通过两层网络
        x = self.fc1(x)
        x = self.fc2(x)
        return x