import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange
from torch.utils.cpp_extension import load
from pytorch_wavelets import DWTForward

# RWKV
T_MAX = 512*512
wkv_cuda = load(
    name="wkv",
    sources=["/home/WACV/cuda/wkv_op.cpp", "/home/WACV/cuda/wkv_cuda.cu"],
    verbose=True,
    extra_cuda_cflags=['-res-usage', '--maxrregcount 60', f'-DTmax={T_MAX}']
)

class WKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, w, u, k, v):
        ctx.B = B
        ctx.T = T
        ctx.C = C
        assert T <= T_MAX
        assert B * C % min(C, 1024) == 0

        half_mode = (w.dtype == torch.half)
        bf_mode = (w.dtype == torch.bfloat16)
        ctx.save_for_backward(w, u, k, v)
        w = w.float().contiguous()
        u = u.float().contiguous()
        k = k.float().contiguous()
        v = v.float().contiguous()
        y = torch.empty((B, T, C), device='cuda', memory_format=torch.contiguous_format)
        wkv_cuda.forward(B, T, C, w, u, k, v, y)
        if half_mode:
            y = y.half()
        elif bf_mode:
            y = y.bfloat16()
        return y

    @staticmethod
    def backward(ctx, gy):
        B = ctx.B
        T = ctx.T
        C = ctx.C
        # assert T <= T_MAX
        assert B * C % min(C, 1024) == 0
        w, u, k, v = ctx.saved_tensors
        gw = torch.zeros((B, C), device='cuda').contiguous()
        gu = torch.zeros((B, C), device='cuda').contiguous()
        gk = torch.zeros((B, T, C), device='cuda').contiguous()
        gv = torch.zeros((B, T, C), device='cuda').contiguous()
        half_mode = (w.dtype == torch.half)
        bf_mode = (w.dtype == torch.bfloat16)
        wkv_cuda.backward(B, T, C,
                          w.float().contiguous(),
                          u.float().contiguous(),
                          k.float().contiguous(),
                          v.float().contiguous(),
                          gy.float().contiguous(),
                          gw, gu, gk, gv)
        if half_mode:
            gw = torch.sum(gw.half(), dim=0)
            gu = torch.sum(gu.half(), dim=0)
            return (None, None, None, gw.half(), gu.half(), gk.half(), gv.half())
        elif bf_mode:
            gw = torch.sum(gw.bfloat16(), dim=0)
            gu = torch.sum(gu.bfloat16(), dim=0)
            return (None, None, None, gw.bfloat16(), gu.bfloat16(), gk.bfloat16(), gv.bfloat16())
        else:
            gw = torch.sum(gw, dim=0)
            gu = torch.sum(gu, dim=0)
            return (None, None, None, gw, gu, gk, gv)
        
def RUN_CUDA(B, T, C, w, u, k, v):
    return WKV.apply(B, T, C, w.cuda(), u.cuda(), k.cuda(), v.cuda())

def q_shift(input, shift_pixel=1, gamma=1 / 4):
    assert gamma <= 1 / 4
    B, C, H, W = input.shape
    output = torch.zeros_like(input)
    output[:, 0:int(C * gamma), :, shift_pixel:W] = input[:, 0:int(C * gamma), :, 0:W - shift_pixel]
    output[:, int(C * gamma):int(C * gamma * 2), :, 0:W - shift_pixel] = input[:, int(C * gamma):int(C * gamma * 2), :,
                                                                         shift_pixel:W]
    output[:, int(C * gamma * 2):int(C * gamma * 3), shift_pixel:H, :] = input[:, int(C * gamma * 2):int(C * gamma * 3),
                                                                         0:H - shift_pixel, :]
    output[:, int(C * gamma * 3):int(C * gamma * 4), 0:H - shift_pixel, :] = input[:,
                                                                             int(C * gamma * 3):int(C * gamma * 4),
                                                                             shift_pixel:H, :]
    output[:, int(C * gamma * 4):, ...] = input[:, int(C * gamma * 4):, ...]
    return output

class VRWKV_SpatialMix_Coil(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id, init_mode='fancy', key_norm=False, scan_schemes=None):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        attn_sz = n_embd
        self.device = None
        self.recurrence = 2
        self.scan_schemes = scan_schemes or [
            ('top-left', 'True'), ('top-right', 'True'), ('bottom-left', 'True'), ('bottom-right', 'True'),
            ('top-left', 'False'), ('top-right', 'False'), ('bottom-left', 'False'), ('bottom-right', 'False')]
        self.dwconv = nn.Conv2d(n_embd, n_embd, kernel_size=3, stride=1, padding=1, groups=n_embd, bias=False)
        self.key = nn.Linear(n_embd, attn_sz, bias=False)
        self.value = nn.Linear(n_embd, attn_sz, bias=False)
        self.receptance = nn.Linear(n_embd, attn_sz, bias=False)
        if key_norm:
            self.key_norm = nn.LayerNorm(n_embd)
        else:
            self.key_norm = None
        self.output = nn.Linear(attn_sz, n_embd, bias=False)
        self.spatial_decay = nn.Parameter(torch.randn((self.recurrence, self.n_embd)))
        self.spatial_first = nn.Parameter(torch.randn((self.recurrence, self.n_embd)))
    
    def get_coil_indices(self, h, w, clockwise=True, start_corner='top-left'):
        visited = [[False for _ in range(w)] for _ in range(h)]
        indices = []

        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        if start_corner == 'top-left':
            row, col = 0, 0
            dir_idx = 0 if clockwise else 1
        elif start_corner == 'top-right':
            row, col = 0, w - 1
            dir_idx = 1 if clockwise else 2
        elif start_corner == 'bottom-right':
            row, col = h - 1, w - 1
            dir_idx = 2 if clockwise else 3
        elif start_corner == 'bottom-left':
            row, col = h - 1, 0
            dir_idx = 3 if clockwise else 0
        else:
            raise ValueError("Invalid start_corner")

        for _ in range(h * w):
            indices.append(row * w + col)
            visited[row][col] = True

            next_row = row + directions[dir_idx][0]
            next_col = col + directions[dir_idx][1]

            if 0 <= next_row < h and 0 <= next_col < w and not visited[next_row][next_col]:
                row, col = next_row, next_col
            else:
                dir_idx = (dir_idx + 1) % 4 if clockwise else (dir_idx - 1) % 4
                row += directions[dir_idx][0]
                col += directions[dir_idx][1]

        return torch.tensor(indices, dtype=torch.long, device=self.device)
    
    def jit_func(self, x, resolution, scan_scheme):
        h, w = resolution
        start, clockwise = scan_scheme
        spiral_order = self.get_coil_indices(h, w, start_corner=start, clockwise=clockwise)

        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = q_shift(x)

        x = rearrange(x, 'b c h w -> b c (h w)')
        x = x[..., spiral_order]
        x = rearrange(x, 'b c (h w) -> b (h w) c', h=h, w=w)

        k = self.key(x)
        v = self.value(x)
        r = self.receptance(x)
        sr = torch.sigmoid(r)
        return sr, k, v   
    
    def forward(self, x, resolution):
        B, T, C = x.size()
        self.device = x.device

        selected_scheme = self.scan_schemes[self.layer_id % len(self.scan_schemes)]
        sr, k, v = self.jit_func(x, resolution, selected_scheme)

        for j in range(self.recurrence):
            if j % 2 == 0:
                v = RUN_CUDA(B, T, C, self.spatial_decay[j] / T, self.spatial_first[j] / T, k, v)
            else:
                h, w = resolution
                new_h, new_w = (h, w) if selected_scheme[1] == 'True' else (w, h)
                spiral_order = self.get_coil_indices(new_h, new_w, start_corner=selected_scheme[0],
                                                       clockwise=selected_scheme[1])
                k = rearrange(k, 'b (h w) c -> b c h w', h=h, w=w)
                k = rearrange(k, 'b c h w -> b c (h w)')[..., spiral_order]
                k = rearrange(k, 'b c (h w) -> b (h w) c', h=new_h, w=new_w)

                v = rearrange(v, 'b (h w) c -> b c h w', h=h, w=w)
                v = rearrange(v, 'b c h w -> b c (h w)')[..., spiral_order]
                v = rearrange(v, 'b c (h w) -> b (h w) c', h=new_h, w=new_w)

                v = RUN_CUDA(B, T, C, self.spatial_decay[j] / T, self.spatial_first[j] / T, k, v)
                k = rearrange(k, 'b (h w) c -> b (h w) c', h=h, w=w)
                v = rearrange(v, 'b (h w) c -> b (h w) c', h=h, w=w)

        x = v
        if self.key_norm is not None:
            x = self.key_norm(x)
        x = sr * x
        x = self.output(x)
        return x

class VRWKV_ChannelMix(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id, hidden_rate=4, init_mode='fancy', key_norm=False):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        hidden_sz = int(hidden_rate * n_embd)
        self.key = nn.Linear(n_embd, hidden_sz, bias=False)
        if key_norm:
            self.key_norm = nn.LayerNorm(hidden_sz)
        else:
            self.key_norm = None
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(hidden_sz, n_embd, bias=False)
    def forward(self, x, resolution):
        h, w = resolution
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = q_shift(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        k = self.key(x)
        k = torch.square(torch.relu(k))
        if self.key_norm is not None:
            k = self.key_norm(k)
        kv = self.value(k)
        x = torch.sigmoid(self.receptance(x)) * kv

        return x

class Block(nn.Module):
    def __init__(self, outer_dim, inner_dim, layer_id, num_words,drop_path=0.,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.has_inner = inner_dim > 0
        if self.has_inner:
            self.inner_norm1 = norm_layer(num_words * inner_dim)
            self.inner_attn = VRWKV_SpatialMix_Coil(n_embd=inner_dim, n_layer=None, layer_id=layer_id)
            self.inner_norm2 = norm_layer(num_words * inner_dim)
            self.inner_ffn = VRWKV_ChannelMix(n_embd=inner_dim, n_layer=None, layer_id=None)
            self.proj_norm1 = norm_layer(num_words * inner_dim)
            self.proj = nn.Linear(num_words * inner_dim, outer_dim, bias=False)
            self.proj_norm2 = norm_layer(outer_dim)

        self.outer_norm1 = norm_layer(outer_dim)
        self.outer_attn = VRWKV_SpatialMix_Coil(n_embd=outer_dim, n_layer=None, layer_id=layer_id)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.outer_norm2 = norm_layer(outer_dim)
        self.outer_ffn = VRWKV_ChannelMix(n_embd=outer_dim, n_layer=None, layer_id=1)

    def forward(self, x, outer_tokens, H_out, W_out, H_in, W_in):
        B, N, C = outer_tokens.size()
        if self.has_inner:
            inner_patch_resolution = [H_in, W_in]
            x = x + self.drop_path(self.inner_attn(self.inner_norm1(x.reshape(B, N, -1)).reshape(B * N, H_in * W_in, -1), inner_patch_resolution))
            x = x + self.drop_path(self.inner_ffn(self.inner_norm2(x.reshape(B, N, -1)).reshape(B * N, H_in * W_in, -1), inner_patch_resolution))
            outer_tokens = outer_tokens + self.proj_norm2(self.proj(self.proj_norm1(x.reshape(B, N, -1))))
        outer_patch_resolution = [H_out, W_out]
        outer_tokens = outer_tokens + self.drop_path(self.outer_attn(self.outer_norm1(outer_tokens), outer_patch_resolution))
        outer_tokens = outer_tokens + self.drop_path(self.outer_ffn(self.outer_norm2(outer_tokens), outer_patch_resolution))
        return x, outer_tokens
    
class Stem(nn.Module):
    def __init__(self, img_size, in_chans=1, outer_dim=768, inner_dim=24):
        super().__init__()
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.inner_dim = inner_dim
        self.num_patches = img_size[0] // 8 * img_size[1] // 8
        self.num_words = 16

        self.common_conv = nn.Sequential(
            nn.Conv2d(in_chans, inner_dim * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(inner_dim * 2),
            nn.ReLU(inplace=True),
        )
        self.inner_convs = nn.Sequential(
            nn.Conv2d(inner_dim * 2, inner_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(inner_dim),
            nn.ReLU(inplace=False),
        )
        self.outer_convs = nn.Sequential(
            nn.Conv2d(inner_dim * 2, inner_dim * 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(inner_dim * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_dim * 4, inner_dim * 8, 3, stride=2, padding=1),
            nn.BatchNorm2d(inner_dim * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_dim * 8, outer_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(outer_dim),
            nn.ReLU(inplace=False),
        )
        self.unfold = nn.Unfold(kernel_size=4, padding=0, stride=4)
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.common_conv(x)
        H_out, W_out = H // 8, W // 8
        H_in, W_in = 4, 4
        inner_tokens = self.inner_convs(x)
        inner_tokens = self.unfold(inner_tokens).transpose(1, 2)
        inner_tokens = inner_tokens.reshape(B * H_out * W_out, self.inner_dim, H_in * W_in).transpose(1, 2)
        outer_tokens = self.outer_convs(x)
        outer_tokens = outer_tokens.permute(0, 2, 3, 1).reshape(B, H_out * W_out, -1)
        return inner_tokens, outer_tokens, (H_out, W_out), (H_in, W_in)

class Stage(nn.Module):
    def __init__(self, outer_dim, inner_dim, num_words, num_blocks, drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        blocks = []
        drop_path = drop_path if isinstance(drop_path, list) else [drop_path] * num_blocks

        for j in range(num_blocks):
            blocks.append(Block(
                outer_dim, inner_dim, layer_id=j, num_words=num_words, drop_path=drop_path[j], norm_layer=norm_layer))

        self.blocks = nn.ModuleList(blocks)

    def forward(self, inner_tokens, outer_tokens, H_out, W_out, H_in, W_in):
        for blk in self.blocks:
            inner_tokens, outer_tokens = blk(inner_tokens, outer_tokens, H_out, W_out, H_in, W_in)
        return inner_tokens, outer_tokens

# Edge light extraction
class Coil_RWKV(nn.Module):
    def __init__(self, img_size, in_chans, out_chans, outer_dim, inner_dim, num_blocks):
        super().__init__()
        self.patch_embed = Stem(img_size=img_size, in_chans=in_chans, outer_dim=outer_dim, inner_dim=inner_dim)
        num_works = self.patch_embed.num_words
        self.stage = Stage(outer_dim=outer_dim, inner_dim=inner_dim, num_words=num_works, num_blocks=num_blocks)
        self.out_proj = nn.Conv2d(outer_dim, out_chans, kernel_size=1)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'outer_pos', 'inner_pos'}

    def forward_feature(self, x):
        inner_tokens, outer_tokens, (H_out, W_out), (H_in, W_in) = self.patch_embed(x)
        inner_tokens, outer_tokens = self.stage(inner_tokens, outer_tokens, H_out, W_out, H_in, W_in)
        B, L, M = outer_tokens.shape
        mid_out = outer_tokens.reshape(B, int(math.sqrt(L)), int(math.sqrt(L)), M).permute(0, 3, 1, 2)
        mid_out = F.interpolate(mid_out, size=x.size()[-2:], mode='bilinear')
        output = self.out_proj(mid_out)
        return output
    
    def forward(self, x):
        x = self.forward_feature(x)
        return x

# Illumination Estimator
class Scharr(nn.Module):
    def __init__(self, channel):
        super(Scharr, self).__init__()
        scharr_x = torch.tensor([[-3., 0., 3.], [-10., 0., 10.], [-3., 0., 3.]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        scharr_y = torch.tensor([[-3., -10., -3.], [0., 0., 0.], [3., 10., 3.]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.conv_x = nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False)
        self.conv_y = nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False)
        self.conv_x.weight.data = scharr_x.repeat(channel, 1, 1, 1)
        self.conv_y.weight.data = scharr_y.repeat(channel, 1, 1, 1)
        self.norm = nn.BatchNorm2d(channel)
        self.conv_extra = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=16, kernel_size=1, stride=1, groups=1, dilation=1),
            nn.SiLU(),
            nn.Conv2d(in_channels=16, out_channels=channel, kernel_size=3, padding=1, stride=1, groups=1, dilation=1),
        )

    def forward(self, x):
        edges_x = self.conv_x(x)
        edges_y = self.conv_y(x)
        # scharr_edge = torch.sqrt(edges_x ** 2 + edges_y ** 2)   # L2
        scharr_edge = torch.abs(edges_x) + torch.abs(edges_y)   # L1
        scharr_edge = self.norm(scharr_edge)
        out = self.conv_extra(x + scharr_edge)

        return out
    
class Illumination(nn.Module):
    def __init__(self,):
        super().__init__()
        self.shared_stage = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1, groups=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1, groups=4, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, padding=1, groups=4, bias=True),
            nn.ReLU(inplace=True)
        )
        self.noise_head = nn.Conv2d(in_channels=8, out_channels=3, kernel_size=1)
        self.illumination_head = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=1)

    def forward(self, x):
        illum_map = x.mean(dim=1, keepdim=True)  # 灰度世界理论估计全局光照
        x_shared = self.shared_stage(x)
        noise = torch.tanh(self.noise_head(x_shared))   # 噪声估计
        illumination = torch.sigmoid(self.illumination_head(x_shared))  # 继续全局光照估计
        restored_reflection = (x - noise) / (illumination)
        restored = x * illumination + restored_reflection 
        return illum_map, noise, restored

# WTFDown
class WTFDown(nn.Module):#小波变化高低频分解下采样模块
    def __init__(self, in_ch):
        super(WTFDown, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.dconv_bn_relu = nn.Sequential(
                                    nn.Conv2d(in_ch*3, in_ch, kernel_size=1, stride=1, groups=in_ch),
                                    nn.BatchNorm2d(in_ch),
                                    nn.ReLU(inplace=True),
                                    )
        self.outdconv_bn_relu_L = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1, groups=in_ch),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
        )
        self.outdconv_bn_relu_H = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1, groups=in_ch),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:,:,0,::]
        y_LH = yH[0][:,:,1,::]
        y_HH = yH[0][:,:,2,::]
        yH = torch.cat([y_HL, y_LH, y_HH], dim=1)
        yH = self.dconv_bn_relu(yH)
        yL = self.outdconv_bn_relu_L(yL)
        yH = self.outdconv_bn_relu_H(yH)
        # return yL , yH
        return yL + yH #小波变化高低频分解下采样模块

# Concat
class LayerNorm(nn.Module):

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=4, bias=False):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        b, c, h, w = x.shape

        q = self.q_dwconv(self.q(x))
        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = nn.functional.softmax(attn, dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class IEL(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super(IEL, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)
        self.dwconv1 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                 groups=hidden_features, bias=bias)
        self.dwconv2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                 groups=hidden_features, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        self.Tanh = nn.Tanh()

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x1 = self.Tanh(self.dwconv1(x1)) + x1
        x2 = self.Tanh(self.dwconv2(x2)) + x2
        x = x1 * x2
        x = self.project_out(x)
        return x

class CFEM(nn.Module):
    def __init__(self, in_channels, num_heads=4, init_lambda=0.2):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1, 1))
        self.conv_Fn_v = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=True)
        self.conv_Fn_k = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=True)
        self.conv_Fr_q = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=True)
        self.init_lambda = init_lambda
        self.cam = CrossAttention(in_channels)
        self.iel = IEL(in_channels)
        self.norm = LayerNorm(in_channels)
        self.scharr = Scharr(in_channels)

    def forward(self, data):
        input1, input2 = data
        F_Q =  self.conv_Fr_q(input1)
        F_K =  self.conv_Fn_k(input2)
        F_V =  self.conv_Fn_v(input2)
        # 交叉注意力
        att = self.cam(F_Q,F_K)
        # DFR操作
        out = att @ F_V
        Fdfr = torch.sub(out * self.init_lambda ,F_V)
        out =  att @ F_Q + Fdfr

        # 特征增强操作➕残差连接
        out = self.iel(self.norm(out)) + self.scharr(input1)
        # 最后进行特征增强操作，在这里然后再进行残差操作，
        # 防止特征信息丢失，选择input1或是input2都可以
        # 举个例子：input1，input2分别代表高频和低频特征，如果想要保留更多高频信息，就使用input1高频信息，反之。或是使用input1+input2也行，可以多去跑跑使用验证。
        return out
       
# Net
class Net(nn.Module):
    def __init__(self, channels, out_dim, dim, img_size, num_blocks):
        super().__init__()
        self.illum = Illumination()
        self.proj = nn.Conv2d(in_channels=channels, out_channels=dim, kernel_size=3, stride=1, padding=1, bias=True)

        self.Coil_RWKV_Stage1 = Coil_RWKV(
            img_size=img_size[0], in_chans=dim, out_chans=dim*2, inner_dim=dim, outer_dim=dim*2, num_blocks=num_blocks[0]
        )   # 512
        self.WTD_1 = WTFDown(in_ch=dim*2)

        self.Coil_RWKV_Stage2 = Coil_RWKV(
            img_size=img_size[1], in_chans=dim*2, out_chans=dim*4, inner_dim=dim*2, outer_dim=dim*4, num_blocks=num_blocks[1]
        )   # 256
        self.WTD_2 = WTFDown(in_ch=dim*4)

        self.Coil_RWKV_Bottom = Coil_RWKV(
            img_size=img_size[2], in_chans=dim*4, out_chans=dim*4, inner_dim=dim*4, outer_dim=dim*4, num_blocks=num_blocks[1]
        )   # 128

        self.Coil_RWKV_Stage3 = Coil_RWKV(
            img_size=img_size[1], in_chans=dim*8, out_chans=dim*2, inner_dim=dim*8, outer_dim=dim*2, num_blocks=num_blocks[1]
        )   # 256

        self.Coil_RWKV_Stage4 = Coil_RWKV(
            img_size=img_size[0], in_chans=dim*4, out_chans=dim, inner_dim=dim*4, outer_dim=dim, num_blocks=num_blocks[0]
        )   # 512

        self.CFEM_1 = CFEM(dim*4)
        self.CFEM_2 = CFEM(dim*2)

        self.outproj = nn.Conv2d(in_channels=dim, out_channels=3, kernel_size=3, stride=1, padding=1, bias=True)
        

    def forward(self, x):
        # illum_map, noise, restored = self.illum(x)

        # Coil_RWKV_in = self.proj(restored * x)
        Coil_RWKV_in = self.proj(x)
        Coil_RWKV_Stage1 = self.Coil_RWKV_Stage1(Coil_RWKV_in)
        Coil_RWKV_Stage1_Down = self.WTD_1(Coil_RWKV_Stage1)

        Coil_RWKV_Stage2 = self.Coil_RWKV_Stage2(Coil_RWKV_Stage1_Down)
        Coil_RWKV_Stage2_Down = self.WTD_2(Coil_RWKV_Stage2)
        
        Coil_RWKV_Bottom = self.Coil_RWKV_Bottom(Coil_RWKV_Stage2_Down)

        Coil_RWKV_Stage3_Up = F.interpolate(Coil_RWKV_Bottom, scale_factor=2, mode='bilinear', align_corners=True)
        Coil_RWKV_Stage3_in = self.CFEM_1([Coil_RWKV_Stage2, Coil_RWKV_Stage3_Up])# torch.cat([Coil_RWKV_Stage2, Coil_RWKV_Stage3_Up],dim=1)
        Coil_RWKV_Stage3 = self.Coil_RWKV_Stage3(Coil_RWKV_Stage3_in)

        Coil_RWKV_Stage4_Up = F.interpolate(Coil_RWKV_Stage3, scale_factor=2, mode='bilinear', align_corners=True)
        Coil_RWKV_Stage4_in = torch.cat([Coil_RWKV_Stage1, Coil_RWKV_Stage4_Up],dim=1)# self.CFEM_2([Coil_RWKV_Stage1, Coil_RWKV_Stage4_Up])
        Coil_RWKV_Stage4 = self.Coil_RWKV_Stage4(Coil_RWKV_Stage4_in)

        return self.outproj(Coil_RWKV_Stage4)
        # Coil_RWKV_Out_illum = torch.sigmoid(self.outproj(Coil_RWKV_Stage4))

        # # R · (Sg + St) + N
        # total_illum = Coil_RWKV_Out_illum + illum_map
        # # reflectance = torch.clamp((((x - noise) / total_illum) * torch.pow(total_illum, 0.4)), min=0, max=1)
        # # out = reflectance + restored
        # return total_illum, noise, restored

if __name__ == '__main__':
    from torchinfo import summary
    input = torch.rand(1,3,512,512).cuda()
    model = Net(
        channels=3,
        out_dim=3,
        dim=16,
        img_size=[512, 256, 128],
        num_blocks=[4, 8],
        ).cuda()
    out = model(input)
    summary(model, input_size=(1, 3, 128, 128), col_names=["input_size", "output_size", "num_params"])
    print(out.shape)

    # model = Stem(img_size=512, in_chans=3, outer_dim=64, inner_dim=64).cuda()
    # inner_tokens, outer_tokens, (H_out, W_out), (H_in, W_in) = model(input)
    # print(inner_tokens.shape, outer_tokens.shape, (H_out, W_out), (H_in, W_in))

    # # model_1 = Block(outer_dim=64, inner_dim=64, layer_id=1, num_words=16).cuda()
    # # inner_tokens1, outer_tokens1 = model_1(inner_tokens, outer_tokens, H_out, W_out, H_in, W_in)
    # # print(inner_tokens1.shape, outer_tokens1.shape)

    # model_1 = Stage(num_blocks=8, outer_dim=64, inner_dim=64, num_words=16).cuda()
    # inner_tokens1, outer_tokens1 = model_1(inner_tokens, outer_tokens, H_out, W_out, H_in, W_in)
    # print(inner_tokens1.shape, outer_tokens1.shape)
    # model = Coil_RWKV(img_size=512, in_chans=3, out_chans=16, outer_dim=16, inner_dim=16, num_blocks=8).cuda()
    # out = model(input)
    # summary(model, input_size=(1, 3, 512, 512), col_names=["input_size", "output_size", "num_params"])
    # print(out.shape)