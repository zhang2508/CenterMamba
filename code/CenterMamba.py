import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .ss2d import SS2D_my
    from .csms6s import CrossScan_2, CrossScan_1, CrossScan_3
    from .csms6s import CrossMerge_2, CrossMerge_1, CrossMerge_3
except:
    from ss2d import SS2D_my
    from csms6s import CrossScan_2, CrossScan_1, CrossScan_3
    from csms6s import CrossMerge_2, CrossMerge_1, CrossMerge_3

# 空间mamba
class Mambaspa(nn.Module):
    def __init__(self, in_features, scan_type="spa", d_conv=3, expand=1, d_state=16, bias=False,
                 conv_bias=True, ):
        super().__init__()
        d_inner = int(expand * in_features)
        # self.in_proj = nn.Linear(in_features, d_inner * 2, bias=bias)
        self.in_proj = nn.Linear(in_features, d_inner, bias=bias)
        self.in_proj_skip = nn.Conv1d(1, 1, kernel_size=5, padding=2, bias=True)
        self.act = nn.SiLU()
        self.out_proj = nn.Linear(d_inner, in_features, bias=bias)
        self.conv2d = nn.Conv2d(
            in_channels=d_inner,
            out_channels=d_inner,
            groups=d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
        )

        self.mamba = SS2D_my(
            d_model=d_inner,
            d_state=d_state,
            ssm_ratio=1,
            d_conv=d_conv,
            scan_type=scan_type,
            k_group=2
        )

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()

        z = self.in_proj_skip(torch.mean(x, dim=[1, 2]).unsqueeze(1)).unsqueeze(1)
        x = self.in_proj(x)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))

        x = self.mamba(x, CrossScan=CrossScan_2, CrossMerge=CrossMerge_2)
        x = x.permute(0,2,3,1)

        x = x * F.softmax(z)
        x = self.out_proj(x)
        return x.permute(0, 3, 1, 2).contiguous()

# 光谱mamba
class Mambaspe(nn.Module):
    def __init__(self, in_features, group_channel_num=8, scan_type="spe", d_conv=3, expand=1, d_state=16, bias=False,
                 conv_bias=True, ):
        super().__init__()
        d_inner = int(expand * in_features)
        self.channel_num = d_inner // group_channel_num
        # self.in_proj = nn.Linear(in_features, d_inner * 2, bias=bias)
        self.in_proj = nn.Linear(in_features, d_inner, bias=bias)
        self.in_proj_skip = nn.Conv1d(1, 1, kernel_size=5, padding=2, bias=True)
        self.act = nn.SiLU()
        self.out_proj = nn.Linear(d_inner, in_features, bias=bias)
        self.conv2d = nn.Conv2d(
            in_channels=d_inner,
            out_channels=d_inner,
            groups=d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
        )

        self.mamba = SS2D_my(
            d_model=self.channel_num,
            d_state=d_state,
            ssm_ratio=1,  # 注意这里与空间的不同，这里是1才能确保一致
            scan_type=scan_type,
            k_group=2
        )

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()

        z = self.in_proj_skip(torch.mean(x, dim=[1, 2]).unsqueeze(1)).unsqueeze(1)
        x = self.in_proj(x)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))  # (b, d, h, w)

        B, C, H, W = x.shape
        x = x.flatten(2, 3).transpose(dim0=1, dim1=2).reshape(B*H*W, self.channel_num, -1, 1)
        x = self.mamba(x, CrossScan=CrossScan_3, CrossMerge=CrossMerge_3)  # bhw channel_num group_channel_num 1
        x = x.reshape(B, H, W, -1)

        x = x * F.softmax(z)
        x = self.out_proj(x)
        return x.permute(0, 3, 1, 2).contiguous()


class Stem(nn.Module):
    def __init__(self, in_channels, stem_hidden_dim, group_num=4):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=stem_hidden_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(stem_hidden_dim),
            nn.SiLU())

    def forward(self, x):
        x = self.conv1(x)
        return x


class MSCM(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.spa_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),

            nn.Conv2d(in_channels, in_channels, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
        )

        self.spa_conv2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(5, 1), padding=(0, 2)),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),

            nn.Conv2d(in_channels, in_channels, kernel_size=(1, 5), padding=(2, 0)),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
        )

        self.spa_conv3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
        )

        self.spe_conv12 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.spe_conv12(F.gelu(self.spa_conv1(x) + self.spa_conv2(x) + self.spa_conv3(x)))


class ChanLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

# SS blocks
class Block_ssmamba(nn.Module):
    def __init__(self, in_features):
        super().__init__()

        self.dw = nn.Sequential(
            ChanLayerNorm(in_features),
            nn.Conv2d(in_features, in_features // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(in_features // 2, in_features, kernel_size=3, padding=1),
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=1),
        )

        self.spa = nn.Sequential(
            Mambaspa(in_features),
        )
        self.spe = nn.Sequential(
            Mambaspe(in_features),
        )

    def forward(self, x):
        spa_x = self.spa(x)
        spe_x = self.spe(x)

        stem = F.softmax(torch.mean(self.dw(spa_x + spe_x),dim=1,keepdim=True))
        stem = self.conv1(stem * spa_x + stem * spe_x)
        stem = spa_x + spe_x + stem
        return stem

class CenterMamba(nn.Module):
    def __init__(self, in_features=200, hidden_dim=64, num_classes=16):
        super().__init__()

        self.stem = Stem(in_features, hidden_dim)

        self.cnn_blocks = nn.Sequential(
            Block_ssmamba(hidden_dim),
        )

        self.lss = MSCM(hidden_dim)

        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)

        x = self.cnn_blocks(x)

        x = self.lss(x)
        x = self.cls_head(x)
        return x


if __name__ == '__main__':
    import time

    a = torch.randn((1, 200, 11, 11)).cuda()
    model = CenterMamba(in_features=200).cuda()

    time1 = time.time()
    print(model(a).shape)
    time2 = time.time()
    print(time2 - time1)
