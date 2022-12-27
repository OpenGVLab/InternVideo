#!/usr/bin/env python

from collections import OrderedDict

from timm.models.layers import trunc_normal_, DropPath
import torch
import torch.nn as nn
import torch.nn.functional as F


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


def conv_1x1x1(inp, oup, groups=1):
    return nn.Conv3d(inp, oup, (1, 1, 1), (1, 1, 1), (0, 0, 0), groups=groups)

def conv_3x3x3(inp, oup, groups=1):
    return nn.Conv3d(inp, oup, (3, 3, 3), (1, 1, 1), (1, 1, 1), groups=groups)

def conv_1x3x3(inp, oup, groups=1):
    return nn.Conv3d(inp, oup, (1, 3, 3), (1, 1, 1), (0, 1, 1), groups=groups)

def bn_3d(dim):
    return nn.BatchNorm3d(dim)


class STM(nn.Module):
    def __init__(self, n_dim, reduction=4):
        super(STM, self).__init__()
        reduced_c = n_dim // reduction
        self.reduce = nn.Sequential(
            nn.Conv2d(n_dim, reduced_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduced_c)
        )
        self.shift = nn.Conv2d(reduced_c, reduced_c, kernel_size=3, padding=1, groups=reduced_c, bias=False)
        self.recover = nn.Sequential(
            nn.Conv2d(reduced_c, n_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(n_dim)
        )
        self.pad = (0, 0, 0, 0, 0, 0, 0, 1)

    def forward(self, x):
        # x: [L, N, T, C]
        cls_token, x = x[:1], x[1:]
        L, N, T, C = x.shape
        H = W = int(L**0.5)

        fea = x.permute(1, 2, 3, 0).reshape(N*T, C, H, W)
        bottleneck = self.reduce(fea) # NT, C//r, H, W

        # t feature
        reshape_bottleneck = bottleneck.view((-1, T) + bottleneck.size()[1:])  # N, T, C//r, H, W
        t_fea, __ = reshape_bottleneck.split([T-1, 1], dim=1) # N, T-1, C//r, H, W

        # apply transformation conv to t+1 feature
        conv_bottleneck = self.shift(bottleneck)  # NT, C//r, H, W
        # reshape fea: N, T, C//r, H, W
        reshape_conv_bottleneck = conv_bottleneck.view((-1, T) + conv_bottleneck.size()[1:])
        __, tPlusone_fea = reshape_conv_bottleneck.split([1, T-1], dim=1)  # N, T-1, C//r, H, W
        
        # motion fea = t+1_fea - t_fea
        # pad the last timestamp
        diff_fea = tPlusone_fea - t_fea # N, T-1, C//r, H, W
        # pad = (0,0,0,0,0,0,0,1)
        diff_fea_pluszero = F.pad(diff_fea, self.pad, mode="constant", value=0)  # N, T, C//r, H, W
        diff_fea_pluszero = diff_fea_pluszero.view((-1,) + diff_fea_pluszero.size()[2:])  # NT, C//r, H, W
        y = self.recover(diff_fea_pluszero)  # NT, C, H, W

        # reshape
        y = y.reshape(N, T, C, L).permute(3, 0, 1, 2)
        y = torch.cat([cls_token, y], dim=0)
        return y


class DSTM(nn.Module):
    def __init__(self, n_dim, reduction=4):
        super(DSTM, self).__init__()
        reduced_c = n_dim // reduction
        self.reduce = nn.Sequential(
            nn.Conv2d(n_dim, reduced_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduced_c)
        )
        # DW(T+1) - T
        self.shift_pre = nn.Conv2d(reduced_c, reduced_c, kernel_size=3, padding=1, groups=reduced_c, bias=False)
        self.recover_pre = nn.Sequential(
            nn.Conv2d(reduced_c, n_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(n_dim)
        )
        self.pad_pre = (0, 0, 0, 0, 0, 0, 0, 1)
        # DW(T-1) - T 
        self.shift_back = nn.Conv2d(reduced_c, reduced_c, kernel_size=3, padding=1, groups=reduced_c, bias=False)
        self.recover_back = nn.Sequential(
            nn.Conv2d(reduced_c, n_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(n_dim)
        )
        self.pad_back = (0, 0, 0, 0, 0, 0, 0, 1)

    def forward(self, x):
        # x: [L, N, T, C]
        cls_token, x = x[:1], x[1:]
        L, N, T, C = x.shape
        H = W = int(L**0.5)

        fea = x.permute(1, 2, 3, 0).reshape(N*T, C, H, W)
        bottleneck = self.reduce(fea) # NT, C//r, H, W

        # t feature
        reshape_bottleneck = bottleneck.view((-1, T) + bottleneck.size()[1:])  # N, T, C//r, H, W
        pre_t_fea, __ = reshape_bottleneck.split([T-1, 1], dim=1) # N, T-1, C//r, H, W
        back_t_fea, __ = reshape_bottleneck.split([1, T-1], dim=1) # N, T-1, C//r, H, W
        # apply transformation conv to t+1/t-1 feature
        pre_conv_bottleneck = self.shift_pre(bottleneck)  # NT, C//r, H, W
        back_conv_bottleneck = self.shift_back(bottleneck)  # NT, C//r, H, W
        # reshape fea: N, T, C//r, H, W
        pre_reshape_conv_bottleneck = pre_conv_bottleneck.view((-1, T) + pre_conv_bottleneck.size()[1:])
        back_reshape_conv_bottleneck = back_conv_bottleneck.view((-1, T) + back_conv_bottleneck.size()[1:])
        __, tPlusone_fea = pre_reshape_conv_bottleneck.split([1, T-1], dim=1)  # N, T-1, C//r, H, W
        tMinusone_fea, _ = back_reshape_conv_bottleneck.split([T-1, 1], dim=1)  # N, T-1, C//r, H, W
        # pre_fea = t+1_fea - t_fea
        # back_fea = t-1_fea - t_fea
        pre_diff_fea = tPlusone_fea - pre_t_fea # N, T-1, C//r, H, W
        back_diff_fea = tMinusone_fea - back_t_fea # N, T-1, C//r, H, W
        # pad the last/first timestamp
        pre_diff_fea_pluszero = F.pad(pre_diff_fea, self.pad_pre, mode="constant", value=0)  # N, T, C//r, H, W
        pre_diff_fea_pluszero = pre_diff_fea_pluszero.view((-1,) + pre_diff_fea_pluszero.size()[2:])  # NT, C//r, H, W
        back_diff_fea_pluszero = F.pad(back_diff_fea, self.pad_back, mode="constant", value=0)  # N, T, C//r, H, W
        back_diff_fea_pluszero = back_diff_fea_pluszero.view((-1,) + back_diff_fea_pluszero.size()[2:])  # NT, C//r, H, W
        # recover channel
        pre_y = self.recover_pre(pre_diff_fea_pluszero)  # NT, C, H, W
        back_y = self.recover_back(back_diff_fea_pluszero)  # NT, C, H, W
        # reshape
        y = (pre_y + back_y).reshape(N, T, C, L).permute(3, 0, 1, 2)

        # cat cls_token
        y = torch.cat([cls_token, y], dim=0)
        return y


class TDN(nn.Module):
    def __init__(self, channel, n_segment=8, index=1, reduction=4):
        super(TDN, self).__init__()
        self.channel = channel
        self.reduction = reduction
        self.n_segment = n_segment
        self.stride = 2**(index-1)
        self.conv1 = nn.Conv2d(in_channels=self.channel,
                out_channels=self.channel//self.reduction,
                kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=self.channel//self.reduction)
        self.conv2 = nn.Conv2d(in_channels=self.channel//self.reduction,
                out_channels=self.channel//self.reduction,
                kernel_size=3, padding=1, groups=self.channel//self.reduction, bias=False)

        self.avg_pool_forward2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.avg_pool_forward4 = nn.AvgPool2d(kernel_size=4, stride=4)
        
        self.sigmoid_forward = nn.Sigmoid()

        self.avg_pool_backward2 = nn.AvgPool2d(kernel_size=2, stride=2)#nn.AdaptiveMaxPool2d(1)
        self.avg_pool_backward4 = nn.AvgPool2d(kernel_size=4, stride=4)

        self.sigmoid_backward = nn.Sigmoid()

        self.pad1_forward = (0, 0, 0, 0, 0, 0, 0, 1)
        self.pad1_backward = (0, 0, 0, 0, 0, 0, 1, 0)

        self.conv3 = nn.Conv2d(in_channels=self.channel//self.reduction,
                 out_channels=self.channel, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=self.channel)

        self.conv3_smallscale2 = nn.Conv2d(in_channels=self.channel//self.reduction,
                                  out_channels=self.channel//self.reduction,padding=1, kernel_size=3, bias=False)
        self.bn3_smallscale2 = nn.BatchNorm2d(num_features=self.channel//self.reduction)
        
        self.conv3_smallscale4 = nn.Conv2d(in_channels = self.channel//self.reduction,
                                  out_channels=self.channel//self.reduction,padding=1, kernel_size=3, bias=False)
        self.bn3_smallscale4 = nn.BatchNorm2d(num_features=self.channel//self.reduction)

    def spatial_pool(self, x):
        nt, channel, height, width = x.size()
        input_x = x
        # [N, C, H * W]
        input_x = input_x.view(nt, channel, height * width)
        # [N, 1, C, H * W]
        input_x = input_x.unsqueeze(1)
        # [N, 1, H, W]
        context_mask = self.conv_mask(x)
        # [N, 1, H * W]
        context_mask = context_mask.view(nt, 1, height * width)
        # [N, 1, H * W]
        context_mask = self.softmax(context_mask)
        context_mask = context_mask.view(nt,1,height,width)
        return context_mask

    def forward(self, x):
        # x: [L, N, T, C]
        cls_token, x = x[:1], x[1:]
        L, N, T, C = x.shape
        H = W = int(L**0.5)
        fea = x.permute(1, 2, 3, 0).reshape(N*T, C, H, W)
        
        bottleneck = self.conv1(fea) # nt, c//r, h, w
        bottleneck = self.bn1(bottleneck) # nt, c//r, h, w
        reshape_bottleneck = bottleneck.view((-1, self.n_segment) + bottleneck.size()[1:]) # n, t, c//r, h, w
        
        t_fea_forward, _ = reshape_bottleneck.split([self.n_segment -1, 1], dim=1) # n, t-1, c//r, h, w
        _, t_fea_backward = reshape_bottleneck.split([1, self.n_segment -1], dim=1) # n, t-1, c//r, h, w
        
        conv_bottleneck = self.conv2(bottleneck) # nt, c//r, h, w
        reshape_conv_bottleneck = conv_bottleneck.view((-1, self.n_segment) + conv_bottleneck.size()[1:]) # n, t, c//r, h, w
        _, tPlusone_fea_forward = reshape_conv_bottleneck.split([1, self.n_segment-1], dim=1) # n, t-1, c//r, h, w
        tPlusone_fea_backward ,_ = reshape_conv_bottleneck.split([self.n_segment-1, 1], dim=1) # n, t-1, c//r, h, w
        diff_fea_forward = tPlusone_fea_forward - t_fea_forward # n, t-1, c//r, h, w
        diff_fea_backward = tPlusone_fea_backward - t_fea_backward# n, t-1, c//r, h, w
        diff_fea_pluszero_forward = F.pad(diff_fea_forward, self.pad1_forward, mode="constant", value=0) # n, t, c//r, h, w
        diff_fea_pluszero_forward = diff_fea_pluszero_forward.view((-1,) + diff_fea_pluszero_forward.size()[2:]) #nt, c//r, h, w
        diff_fea_pluszero_backward = F.pad(diff_fea_backward, self.pad1_backward, mode="constant", value=0) # n, t, c//r, h, w
        diff_fea_pluszero_backward = diff_fea_pluszero_backward.view((-1,) + diff_fea_pluszero_backward.size()[2:]) #nt, c//r, h, w
        y_forward_smallscale2 = self.avg_pool_forward2(diff_fea_pluszero_forward) # nt, c//r, 1, 1
        y_backward_smallscale2 = self.avg_pool_backward2(diff_fea_pluszero_backward) # nt, c//r, 1, 1

        y_forward_smallscale4 = diff_fea_pluszero_forward
        y_backward_smallscale4 = diff_fea_pluszero_backward
        y_forward_smallscale2 = self.bn3_smallscale2(self.conv3_smallscale2(y_forward_smallscale2))
        y_backward_smallscale2 = self.bn3_smallscale2(self.conv3_smallscale2(y_backward_smallscale2))

        y_forward_smallscale4 = self.bn3_smallscale4(self.conv3_smallscale4(y_forward_smallscale4))
        y_backward_smallscale4 = self.bn3_smallscale4(self.conv3_smallscale4(y_backward_smallscale4))
        
        y_forward_smallscale2 = F.interpolate(y_forward_smallscale2, diff_fea_pluszero_forward.size()[2:])
        y_backward_smallscale2 = F.interpolate(y_backward_smallscale2, diff_fea_pluszero_backward.size()[2:])
        
        y_forward = self.bn3(self.conv3(1.0/3.0*diff_fea_pluszero_forward + 1.0/3.0*y_forward_smallscale2 + 1.0/3.0*y_forward_smallscale4))# nt, c, 1, 1
        y_backward = self.bn3(self.conv3(1.0/3.0*diff_fea_pluszero_backward + 1.0/3.0*y_backward_smallscale2 + 1.0/3.0*y_backward_smallscale4)) # nt, c, 1, 1

        y_forward = self.sigmoid_forward(y_forward) - 0.5
        y_backward = self.sigmoid_backward(y_backward) - 0.5

        y = 0.5 * y_forward + 0.5 * y_backward
        attn = fea * y
        x = x + attn.reshape(N, T, C, L).permute(3, 0, 1, 2)
        x = torch.cat([cls_token, x], dim=0)
        return x


class CMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = conv_1x1x1(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = conv_1x1x1(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
    
class CBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4., dropout=0., drop_path=0., uni_type='3d', add_ffn=True):
        super().__init__()
        self.norm1 = bn_3d(dim)
        self.conv1 = conv_1x1x1(dim, dim, 1)
        self.conv2 = conv_1x1x1(dim, dim, 1)
        if uni_type == '3d':
            print('Use 3d conv for local MHRA')
            self.attn = conv_3x3x3(dim, dim, groups=dim)
        else:
            print('Use 2d conv for local MHRA')
            self.attn = conv_1x3x3(dim, dim, groups=dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.add_ffn = add_ffn
        if add_ffn:
            print('Add FFN in local MHRA')
            self.norm2 = bn_3d(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = CMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=dropout)
        
        print('Init zero')
        nn.init.constant_(self.conv2.weight, 0.)
        nn.init.constant_(self.conv2.bias, 0.)
        if add_ffn:
            nn.init.constant_(self.mlp.fc2.weight, 0.)
            nn.init.constant_(self.mlp.fc2.bias, 0.)

    def forward(self, x):
        x = x + self.drop_path(self.conv2(self.attn(self.conv1(self.norm1(x)))))
        if self.add_ffn:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x  


class ResidualDecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None,
                 mlp_factor: float = 4.0, dropout: float = 0.0, drop_path: float = 0.0, init_zero=True):
        super().__init__()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        print(f'Drop path rate: {drop_path}')
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        d_mlp = round(mlp_factor * d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_mlp)),
            ("gelu", QuickGELU()),
            ("dropout", nn.Dropout(dropout)),
            ("c_proj", nn.Linear(d_mlp, d_model))
        ]))
        self.ln_2 = nn.LayerNorm(d_model)
        self.ln_3 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask

        if init_zero:
            nn.init.xavier_uniform_(self.attn.in_proj_weight)
            nn.init.constant_(self.attn.in_proj_bias, 0.)
            # nn.init.xavier_uniform_(self.attn.out_proj.weight)
            nn.init.constant_(self.attn.out_proj.weight, 0.)
            nn.init.constant_(self.attn.out_proj.bias, 0.)
            nn.init.xavier_uniform_(self.mlp[0].weight)
            # nn.init.xavier_uniform_(self.mlp[-1].weight)
            nn.init.constant_(self.mlp[-1].weight, 0.)
            nn.init.constant_(self.mlp[-1].bias, 0.)
        else:
            nn.init.trunc_normal_(self.attn.in_proj_weight, std=.02)
            nn.init.constant_(self.attn.in_proj_bias, 0.)
            nn.init.trunc_normal_(self.attn.out_proj.weight, std=.02)
            nn.init.constant_(self.attn.out_proj.bias, 0.)
            nn.init.trunc_normal_(self.mlp.c_fc.weight, std=.02)
            nn.init.constant_(self.mlp.c_fc.bias, 0.)
            nn.init.trunc_normal_(self.mlp.c_proj.weight, std=.02)
            nn.init.constant_(self.mlp.c_proj.bias, 0.)

    def attention(self, x: torch.Tensor, y: torch.Tensor):
        #self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        # return self.attn(x, y, y, need_weights=False, attn_mask=self.attn_mask)[0]
        assert self.attn_mask is None  # not implemented
        # manual forward to add position information
        d_model = self.ln_1.weight.size(0)
        q = (x @ self.attn.in_proj_weight[:d_model].T) + self.attn.in_proj_bias[:d_model]

        k = (y @ self.attn.in_proj_weight[d_model:-d_model].T) + self.attn.in_proj_bias[d_model:-d_model]
        v = (y @ self.attn.in_proj_weight[-d_model:].T) + self.attn.in_proj_bias[-d_model:]
        Tx, Ty, N = q.size(0), k.size(0), q.size(1)
        q = q.view(Tx, N, self.attn.num_heads, self.attn.head_dim).permute(1, 2, 0, 3)
        k = k.view(Ty, N, self.attn.num_heads, self.attn.head_dim).permute(1, 2, 0, 3)
        v = v.view(Ty, N, self.attn.num_heads, self.attn.head_dim).permute(1, 2, 0, 3)
        aff = (q @ k.transpose(-2, -1) / (self.attn.head_dim ** 0.5))

        aff = aff.softmax(dim=-1)
        out = aff @ v
        out = out.permute(2, 0, 1, 3).flatten(2)
        out = self.attn.out_proj(out)
        return out

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        x = x + self.drop_path(self.attention(self.ln_1(x), self.ln_3(y)))
        x = x + self.drop_path(self.mlp(self.ln_2(x)))
        return x


class TransformerDecoder_uniformer_diff_conv_balance(nn.Module):
    def __init__(self, n_layers=4, 
                 uni_layer=4, uni_type='3d', add_ffn=True, t_conv_type='1d', pre_prompt=True,
                 n_dim=768, n_head=12, mlp_factor=4.0, drop_path_rate=0.,
                 mlp_dropout=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], cls_dropout=0.5, t_size=8, spatial_size=14,
                 balance=0.,
                 use_t_conv=True, after_me=True, before_me=False, me_type='dstm', me_reduction=4,
                 use_t_pos_embed=True, num_classes=400, init_zero=True):
        super().__init__()

        n_layers += uni_layer
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]

        self.uni_layer = uni_layer
        self.uni_dec = nn.ModuleList([
            CBlock(n_dim, mlp_ratio=mlp_factor, dropout=mlp_dropout[i], drop_path=dpr[i], uni_type=uni_type, add_ffn=add_ffn)
            for i in range(uni_layer)
        ])

        self.dec = nn.ModuleList([
            ResidualDecoderBlock(n_dim, n_head, mlp_factor=mlp_factor, dropout=mlp_dropout[i], drop_path=dpr[i], init_zero=init_zero)
            for i in range(n_layers)
        ])

        self.pre_prompt = pre_prompt
        if pre_prompt:
            print('Add pre prompt')
            self.pre_temporal_cls_token = nn.Parameter(torch.zeros(n_dim))
        self.temporal_cls_token = nn.Parameter(torch.zeros(n_dim))

        if use_t_conv:
            self.t_conv_type = t_conv_type
            if t_conv_type == '1d':
                print('Use 1d t_conv for CPE')
                self.tconv = nn.ModuleList([
                    nn.Conv1d(n_dim, n_dim, kernel_size=3, stride=1, padding=1, bias=True, groups=n_dim)
                    for i in range(n_layers)
                ])
                if init_zero:
                    for m in self.tconv:
                        nn.init.constant_(m.bias, 0.)
                        m.weight.data[...] = torch.Tensor([0, 1, 0])
            else:
                print('Use 3d t_conv for CPE')
                self.tconv = nn.ModuleList([
                    nn.Conv3d(n_dim, n_dim, kernel_size=3, stride=1, padding=1, bias=True, groups=n_dim)
                    for i in range(n_layers)
                ])
                if init_zero:
                    for m in self.tconv:
                        nn.init.constant_(m.bias, 0.)
        else:
            self.tconv = None

        self.before_me = before_me
        self.after_me = after_me
        if before_me or after_me:
            assert before_me != after_me
            print(f'Use {me_type} attention, Before {before_me}, After {after_me}')
            if me_type == 'stm':
                me_op = STM
            elif me_type == 'dstm':
                me_op = DSTM
            elif me_type == 'tdn':
                me_op = TDN
            self.me = nn.ModuleList([me_op(n_dim, reduction=me_reduction) for i in range(n_layers)])

        if use_t_pos_embed:
            self.pemb_t = nn.Parameter(torch.zeros([n_layers, t_size, n_dim]))
        else:
            self.pemb_t = None
            
        print(F'Balnce weight {balance}')
        self.balance = nn.Parameter(torch.ones((n_dim)) * balance)
        self.sigmoid = nn.Sigmoid()

        if not init_zero:
            nn.init.normal_(self.temporal_cls_token, std=1e-6)
            if self.pemb_t is not None:
                nn.init.trunc_normal_(self.pemb_t, std=.02)

    def forward(self, clip_feats_all, mode='video'):
        # clip_feats_all = clip_feats_all[-len(self.dec):]
        # only return n_layers features, save memory
        clip_feats = [x for x in clip_feats_all]
        if self.after_me:
            origin_clip_feats = [x for x in clip_feats_all]
        
        L, N, T, C = clip_feats[0].size()
        x = self.temporal_cls_token.view(1, 1, -1).repeat(1, N, 1)

        for i in range(len(clip_feats)):
            if self.before_me:
                # contain residual
                clip_feats[i] = self.me[i](clip_feats[i])
            if self.tconv is not None:
                L, N, T, C = clip_feats[i].shape
                if self.t_conv_type == '1d':
                    clip_feats[i] = clip_feats[i].permute(0, 1, 3, 2).flatten(0, 1)  # L * N, C, T
                    clip_feats[i] = self.tconv[i](clip_feats[i]).permute(0, 2, 1).contiguous().view(L, N, T, C)
                else:
                    H = W = int((L - 1) ** 0.5)
                    _, tmp_feats = clip_feats[i][:1], clip_feats[i][1:]
                    tmp_feats = tmp_feats.permute(1, 3, 2, 0).reshape(N, C, T, H, W)
                    tmp_feats = self.tconv[i](tmp_feats).view(N, C, T, L - 1).permute(3, 0, 2, 1)
                    clip_feats[i][1:] = clip_feats[i][1:] + tmp_feats
            if self.pemb_t is not None and mode == 'video':
                clip_feats[i] = clip_feats[i] + self.pemb_t[i]
            if self.after_me:
                clip_feats[i] = clip_feats[i] + self.me[i](origin_clip_feats[i])
            if i < self.uni_layer:
                # L, N, T, C
                L, N, T, C = clip_feats[i].shape
                H = W = int((L - 1) ** 0.5)
                _, tmp_feats = clip_feats[i][:1], clip_feats[i][1:]
                tmp_feats = tmp_feats.permute(1, 3, 2, 0).reshape(N, C, T, H, W)
                tmp_feats = self.uni_dec[i](tmp_feats).view(N, C, T, L - 1).permute(3, 0, 2, 1)
                clip_feats[i][1:] = clip_feats[i][1:] + tmp_feats
            clip_feats[i] = clip_feats[i].permute(2, 0, 1, 3).flatten(0, 1)  # T * L, N, C

        if self.pre_prompt:
            pre_x = self.pre_temporal_cls_token.view(1, 1, -1).repeat(1, N, 1)
            for i in range(len(self.dec)):
                if i < self.uni_layer:
                    pre_x = self.dec[i](pre_x, clip_feats[i])
                elif i == self.uni_layer:
                    clip_feats[i] = torch.cat([pre_x, clip_feats[i]], dim=0)
                    x = self.dec[i](x, clip_feats[i])
                else:
                    x = self.dec[i](x, clip_feats[i])
        else:
            for i in range(len(self.dec)):
                x = self.dec[i](x, clip_feats[i])

        # real residual
        # L, N, T, C
        residual = clip_feats_all[-1][0].mean(1)
        weight = self.sigmoid(self.balance)
        
        # return self.proj((1 - weight) * x[0, :, :] + weight * residual)
        return (1 - weight) * x[0, :, :] + weight * residual


if __name__ == '__main__':
    model = TransformerDecoder_uniformer_diff_conv_balance()

    # construct a fake input to demonstrate input tensor shape
    L, N, T, C = 197, 1, 8, 768  # num_image_tokens, video_batch_size, t_size, feature_dim
    # we use intermediate feature maps from multiple blocks, so input features should be a list
    input_features = []
    for i in range(8):  # vit-b has 12 blocks
        # every item in input_features contains features maps from a single block
        # every item is a tuple containing 3 feature maps:
        # (1) block output features (i.e. after mlp) with shape L, N, T, C
        # (2) projected query features with shape L, N, T, C
        # (3) projected key features with shape L, N, T, C
        input_features.append(
            tuple(torch.zeros([L, N, T, C]) for _ in range(3)))
        # some small optimizations:
        # (1) We only decode from the last $n$ blocks so it's good as long as the last $n$ items of input_features is valid and all previous items can be filled with None to save memory. By default $n=4$.
        # (2) projected query/key features are optional. If you are using an uncompatible image backbone without query/key (e.g. CNN), you can fill the position with None (i.e. the tuple should be (Tensor, None, None) and set use_image_attnmap=False when constructing the model.

    print(model)
    print(model(input_features).shape)  # should be N, 400
