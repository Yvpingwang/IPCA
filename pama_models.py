#This is the baseline model [PAMA](https://doi.org/10.1007/978-3-031-43987-2_69) used in this paper.

from functools import partial

import torch
import torch.nn as nn
from torch import einsum
import torch.nn.functional as F
from module.drop import DropPath
from einops import rearrange, repeat
import numpy as np


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PACA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # update polar head separately
        self.polar_emb = nn.Embedding(8, 1)
        self.dis_embed = nn.Embedding(64 + 2, num_heads)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def generate_main_orientation(self, k_x_attn, polar_pos):
        b, h, k_l, x_l = k_x_attn.shape

        # update polar head separately
        attn_8 = repeat((k_x_attn), 'b h k_l x_l -> b h k_l x_l o', o=8)
        # use torch.abs
        attn_8 = torch.abs(attn_8)
        polar_hot = F.one_hot(polar_pos.to(torch.int64))
        mul = attn_8 * polar_hot
        ori_sum = torch.sum(mul, dim=-2)
        main_ori = torch.argmax(ori_sum, dim=-1)
        new_polar = polar_pos - repeat(main_ori, 'b h k_l-> b h k_l x_l',
                                       x_l=x_l)
        new_polar[new_polar < 0] += 8
        return new_polar.to(torch.int64)

    def forward(self, x, kernal, rd, polar_pos, att_mask):
        c_qkv = self.qkv(x).chunk(3, dim=-1)
        k_kqv = self.qkv(kernal).chunk(3, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), c_qkv)
        k_q, k_k, k_v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), k_kqv)

        rd = self.dis_embed(rd)
        rd = rd.permute(0, 3, 1, 2)
        k_x_attn = (k_q @ k.transpose(-2, -1)) * self.scale  # shape=[b, h, k_l, x_l]   
        polar_pos = repeat(polar_pos.unsqueeze(1), 'b () i j -> b h i j', h=self.num_heads)

        # update polar head separately (polar_emb(8, 1))
        zeros_tensor = torch.zeros_like(polar_pos).cuda(polar_pos.device)
        max_tensor = torch.ones_like(polar_pos).cuda(polar_pos.device) * 7
        polar_pos = torch.where(polar_pos < 0, zeros_tensor, polar_pos)
        polar_pos = torch.where(polar_pos > 7, max_tensor, polar_pos)
        polar_pos = polar_pos.to(torch.int64)

        new_polar = self.generate_main_orientation(k_x_attn,
                                                   polar_pos)
        new_polar = self.polar_emb(new_polar).squeeze(-1)
        if att_mask is not None:
            k_x_attn = k_x_attn.masked_fill(att_mask.permute(0, 1, 3, 2), torch.tensor(-1e6))
        k_x_attn = (k_x_attn + rd + new_polar).softmax(dim=-1)
        k_out = k_x_attn @ v
        k_out = rearrange(k_out, 'b h n d -> b n (h d)')
        k_out = self.proj_drop(self.proj(k_out))

        x_k_attn = (q @ k_k.transpose(-2, -1)) * self.scale  # shape=[b, h, x_l, k_l]
        if att_mask is not None:
            x_k_attn = x_k_attn.masked_fill(att_mask, torch.tensor(-1e9))
        x_k_attn = (x_k_attn + rd.permute(0, 1, 3, 2) + new_polar.permute(0, 1, 3, 2)).softmax(dim=-1)
        x_out = x_k_attn @ k_v
        x_out = rearrange(x_out, 'b h n d -> b n (h d)')
        x_out = self.proj_drop(self.proj(x_out))

        return x_out, k_out



class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)

        self.attn = PACA(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, kx, rd, polar_pos, att_mask):
        x, kx = self.norm1(x), self.norm1(kx)
        x_, kx_ = self.attn(x, kx, rd, polar_pos, att_mask)
        x = x + x_
        kx = kx + kx_

        x = self.drop_path(self.mlp(self.norm2(x))) + x
        kx = self.drop_path(self.mlp(self.norm2(kx))) + kx
        return x, kx


class PamaViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, global_pool=False, num_kernel=64, in_chans=256,
                 embed_dim=1024, depth=24, num_heads=16,
                 mlp_ratio=4., num_classes=5, norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()

        # --------------------------------------------------------f------------------
        # pama encoder specifics
        self.patch_embed = nn.Linear(in_chans, embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.kernel_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.nk = num_kernel
        # self.k_pro = nn.Linear(512, embed_dim)
        self.num_heads = num_heads

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
            
        self.norm = norm_layer(embed_dim)
        self.global_pool = global_pool
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.kernel_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward_encoder(self, x, wsi_rd, wsi_polar, token_mask, kernel_mask, kernel_tokens, device, mask_ratio):
        # embed patches
        x = self.patch_embed(x)
        wsi_rd_mask = wsi_rd # shape[b, k_l, x_l]

        # append k_clstoken rd
        rd_mean = torch.mean(wsi_rd_mask.float(), dim=-1).to(torch.int64)
        wsi_rd_mask = torch.cat((rd_mean.unsqueeze(-1), wsi_rd_mask), dim=-1)
        wsi_polar_mask = wsi_polar
        # append k_clstoken polar
        # 1.use mean
        polar_mean = torch.mean(wsi_polar_mask.float(), dim=-1).to(torch.int64)
        wsi_polar_mask = torch.cat((polar_mean.unsqueeze(-1), wsi_polar_mask), dim=-1)
        # 2.use zero
        # wsi_polar_mask = torch.cat((torch.ones(wsi_polar_mask.shape[0], wsi_polar_mask.shape[1], 1).cuda(device), wsi_polar_mask), dim=-1)

        att_mask = einsum('b i d, b j d -> b i j', token_mask.float(), kernel_mask.float())
        # append cls_token mask
        att_mask = torch.cat((torch.ones(att_mask.shape[0], 1, att_mask.shape[2]).cuda(device), att_mask), dim=1)
        att_mask = repeat(att_mask.unsqueeze(1), 'b () i j -> b h i j', h=self.num_heads) < 0.5

        # append cls token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        # apply Transformer blocks
        for blk in self.blocks:
            x, kernel_tokens = blk(x, kernel_tokens, wsi_rd_mask, wsi_polar_mask,
                                               att_mask=att_mask)

        return x, kernel_tokens

    def forward(self, imgs, wsi_rd, wsi_polar, token_mask, kernel_mask, device, mask_ratio=0.75):
        b = imgs.shape[0]
        kernel_tokens = repeat(self.kernel_token, '() () d -> b k d', b=b, k=self.nk)
        latent, kernel_tokens = self.forward_encoder(imgs, wsi_rd, wsi_polar, token_mask, kernel_mask,
                                                                     kernel_tokens, device, mask_ratio)
        kernel_tokens = self.norm(kernel_tokens)
        if self.global_pool:
            x = latent[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(latent)
            outcome = x[:, 0]
        logits = self.head(outcome)
        return logits, kernel_tokens


def pama_vit_base_patch16_dec512d8b(**kwargs):
    model = PamaViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def pama_vit_large_patch16_dec512d8b(**kwargs):
    model = PamaViT(
        img_size=2048, patch_size=16, embed_dim=1024, depth=4, num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def pama_vit_huge_patch14_dec512d8b(**kwargs):
    model = PamaViT(
        patch_size=14, embed_dim=1024, depth=32, num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def pama_vit_base_dec512d8b(**kwargs):
    model = PamaViT(**kwargs)
    return model


# set recommended archs
pama_vit_base_patch16 = pama_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
pama_vit_large_patch16 = pama_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
pama_vit_huge_patch14 = pama_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
pama_vit_base = pama_vit_base_dec512d8b
