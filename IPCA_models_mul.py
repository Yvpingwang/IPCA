from functools import partial

import torch
import torch.nn as nn
from torch import einsum
import torch.nn.functional as F
import random
from module.drop import DropPath
from einops import rearrange, repeat


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


class IPCA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        # update polar head separately
        self.polar_emb = nn.Embedding(8, 1)
        self.dis_embed = nn.Embedding(64 + 2, num_heads)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_i = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_i2 = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_i = nn.Linear(dim, dim)
        self.proj_i2 = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.proj_drop_i = nn.Dropout(proj_drop)
        self.proj_drop_i2 = nn.Dropout(proj_drop)

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

    def forward(self, x, kernal, rd, polar_pos, att_mask,i_att_mask, cross_att_mask, i_x, i_kernal):
        c_qkv = self.qkv(x).chunk(3, dim=-1)
        k_kqv = self.qkv(kernal).chunk(3, dim=-1)
        i_c_qkv = self.qkv_i(i_x).chunk(3, dim=-1)
        i_k_kqv = self.qkv_i(i_kernal).chunk(3, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), c_qkv)
        k_q, k_k, k_v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), k_kqv)
        i_q, i_k, i_v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), i_c_qkv)
        i_k_q, i_k_k, i_k_v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), i_k_kqv)

        rd = self.dis_embed(rd)
        rd = rd.permute(0, 3, 1, 2)
        k_x_attn = (k_q @ k.transpose(-2, -1)) * self.scale  # shape=[b, h, k_l, x_l]   
        polar_pos = repeat(polar_pos.unsqueeze(1), 'b () i j -> b h i j', h=self.num_heads)

        # 1.fix new_polar (polar_emb(8, num_head))
        # new_polar = self.polar_emb(polar_pos.to(torch.int64)).permute(0, 3, 1, 2)
        # 2.ignore polar head (polar_emb(8, num_head))
        # new_polar = self.generate_main_orientation(k_x_attn, polar_pos)
        # new_polar = self.polar_emb(new_polar).permute(0, 3, 1, 2)

        # 3.update polar head separately (polar_emb(8, 1))
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
        x_ik_attn = (q @ i_k_k.transpose(-2, -1)) * self.scale  # shape=[b, h, x_l, k_l]
        if att_mask is not None:
            x_k_attn = x_k_attn.masked_fill(att_mask, torch.tensor(-1e9))
        if cross_att_mask is not None:
            x_ik_attn = x_ik_attn.masked_fill(cross_att_mask, torch.tensor(-1e9))
        x_k_attn = (x_k_attn + rd.permute(0, 1, 3, 2) + new_polar.permute(0, 1, 3, 2)).softmax(dim=-1)
        x_ik_attn = x_ik_attn.softmax(dim=-1)
        x_out1 = x_k_attn @ k_v 
        x_out1 = rearrange(x_out1, 'b h n d -> b n (h d)')
        x_out1 = self.proj_drop(self.proj(x_out1))
        x_out2 = x_ik_attn @ i_k_v
        x_out2 = rearrange(x_out2, 'b h n d -> b n (h d)')
        x_out2 = self.proj_drop(self.proj(x_out2))
        x_out = x_out1 + x_out2

        ik_ix_attn = (i_k_q @ i_k.transpose(-2, -1)) * self.scale  # shape=[b, h, k_l, x_l]
        if i_att_mask is not None:
            ik_ix_attn = ik_ix_attn.masked_fill(i_att_mask.permute(0, 1, 3, 2), torch.tensor(-1e9))
        ik_ix_attn = ik_ix_attn.softmax(dim=-1)
        i_k_out = ik_ix_attn @ i_v
        i_k_out = rearrange(i_k_out, 'b h n d -> b n (h d)')
        i_k_out = self.proj_drop_i(self.proj_i(i_k_out)) 
        
        i_k2_kqv = self.qkv_i2(i_k_out).chunk(3, dim=-1)
        i_k2_q, i_k2_k, i_k2_v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), i_k2_kqv)

        ix_ik2_attn = (i_q @ i_k2_k.transpose(-2, -1)) * self.scale  # shape=[b, h, x_l, k_l]
        if i_att_mask is not None:
            ix_ik2_attn = ix_ik2_attn.masked_fill(i_att_mask, torch.tensor(-1e9))
        ix_ik2_attn = ix_ik2_attn .softmax(dim=-1)
        i_x_out = ix_ik2_attn @ i_k2_v
        i_x_out = rearrange(i_x_out, 'b h n d -> b n (h d)')
        i_x_out = self.proj_drop_i2(self.proj_i2(i_x_out))

        return x_out, k_out, i_x_out


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., num_ik=32, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm1_i = norm_layer(dim)
        self.kernel_token_i = nn.Parameter(torch.randn(1, num_ik, dim))
        self.attn = IPCA(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm2_i = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp_i = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, i_x, kx, rd, polar_pos, att_mask, i_att_mask, cross_att_mask): 
        b = x.shape[0]
        i_kx = repeat(self.kernel_token_i, '() k d -> b k d', b=b)
        x, kx, i_x, i_kx = self.norm1(x), self.norm1(kx), self.norm1_i(i_x), self.norm1_i(i_kx)
        x_, kx_ ,i_x_= self.attn(x, kx, rd, polar_pos, att_mask, i_att_mask, cross_att_mask, i_x, i_kx) 
        x = x + x_
        kx = kx + kx_
        i_x = i_x + i_x_

        x = self.drop_path(self.mlp(self.norm2(x))) + x
        kx = self.drop_path(self.mlp(self.norm2(kx))) + kx
        i_x = self.drop_path(self.mlp_i(self.norm2_i(i_x))) + i_x
        return x, kx, i_x


class IpcaViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, global_pool=False, num_kernel=64, in_chans=256,
                 embed_dim=1024, depth=24, num_heads=16,
                 mlp_ratio=4., num_classes=5, num_ik=32, norm_layer=nn.LayerNorm, dropout=0., attn_dropout=0.,**kwargs):
        super().__init__()

        # --------------------------------------------------------------------------
        # encoder specifics
        self.patch_embed = nn.Linear(in_chans, embed_dim)
        self.patch_embed_i = nn.Linear(in_chans, embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token_i = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.kernel_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        self.nk = num_kernel
        self.num_heads = num_heads
        self.num_ik=num_ik

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, num_ik, qkv_bias=True, qk_scale=None, norm_layer=norm_layer,drop=dropout, attn_drop=attn_dropout)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.norm_i = norm_layer(embed_dim)
        self.global_pool = global_pool
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.head_i = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
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
        torch.nn.init.normal_(self.cls_token_i, std=.02)

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

    def random_masking(self, k, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = k.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=k.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        k_masked = torch.gather(k, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=k.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return k_masked, mask, ids_restore, ids_keep

    def forward_encoder(self, x, wsi_rd, wsi_polar, token_mask, kernel_mask, kernel_tokens, i_x, i_token_mask, i_kernel_mask, device, mask_ratio):
        # embed patches
        x = self.patch_embed(x)
        i_x = self.patch_embed_i(i_x)
        kernel_tokens, mask, ids_restore, ids_keep = self.random_masking(kernel_tokens, mask_ratio)

        wsi_rd_mask = torch.gather(wsi_rd, dim=1,index=ids_keep.unsqueeze(-1).repeat(1, 1, wsi_rd.shape[-1]))  # shape[b, k_l, x_l]

        # append k_clstoken rd
        rd_mean = torch.mean(wsi_rd_mask.float(), dim=-1).to(torch.int64)
        wsi_rd_mask = torch.cat((rd_mean.unsqueeze(-1), wsi_rd_mask), dim=-1)
        wsi_polar_mask = torch.gather(wsi_polar, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, wsi_polar.shape[-1]))
        # append k_clstoken polar
        # 1.use mean
        polar_mean = torch.mean(wsi_polar_mask.float(), dim=-1).to(torch.int64)
        wsi_polar_mask = torch.cat((polar_mean.unsqueeze(-1), wsi_polar_mask), dim=-1)
        # 2.use zero
        # wsi_polar_mask = torch.cat((torch.ones(wsi_polar_mask.shape[0], wsi_polar_mask.shape[1], 1).cuda(device), wsi_polar_mask), dim=-1)

        att_mask = einsum('b i d, b j d -> b i j', token_mask.float(), kernel_mask.float())
        att_mask = torch.gather(att_mask.transpose(-1, -2), dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, att_mask.shape[-2])).transpose(-1, -2)
        # append cls_token mask
        att_mask = torch.cat((torch.ones(att_mask.shape[0], 1, att_mask.shape[2]).cuda(device), att_mask), dim=1)
        att_mask = repeat(att_mask.unsqueeze(1), 'b () i j -> b h i j', h=self.num_heads) < 0.5

        i_att_mask = einsum('b i d, b j d -> b i j', i_token_mask.float(), i_kernel_mask.float())
        # append cls_token mask
        i_att_mask = torch.cat((torch.ones(i_att_mask.shape[0], 1, i_att_mask.shape[2]).cuda(device), i_att_mask), dim=1)
        i_att_mask = repeat(i_att_mask.unsqueeze(1), 'b () i j -> b h i j', h=self.num_heads) < 0.5

        cross_att_mask = einsum('b i d, b j d -> b i j', token_mask.float(), i_kernel_mask.float())
        # append cls_token mask
        cross_att_mask = torch.cat((torch.ones(cross_att_mask.shape[0], 1, cross_att_mask.shape[2]).cuda(device), cross_att_mask), dim=1)
        cross_att_mask = repeat(cross_att_mask.unsqueeze(1), 'b () i j -> b h i j', h=self.num_heads) < 0.5

        # append cls token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        cls_tokens_i = self.cls_token_i.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        i_x = torch.cat((cls_tokens_i, i_x), dim=1)
        # apply Transformer blocks
        for blk in self.blocks:
            x, kernel_tokens, i_x= blk(x, i_x, kernel_tokens, wsi_rd_mask, wsi_polar_mask,
                                               att_mask=att_mask, i_att_mask=i_att_mask, cross_att_mask= cross_att_mask) 

        return x, kernel_tokens, i_x

    def forward(self, imgs, wsi_rd, wsi_polar, token_mask, kernel_mask, ihcs, i_token_mask, device, mask_ratio):
        b = imgs.shape[0]
        i_kernel_mask = torch.ones((b, self.num_ik, 1)).cuda(device)
        kernel_tokens = repeat(self.kernel_token, '() () d -> b k d', b=b, k=self.nk)
        latent, kernel_tokens, i_latent= self.forward_encoder(imgs, wsi_rd, wsi_polar, token_mask, kernel_mask,
                                                                     kernel_tokens, ihcs, i_token_mask, i_kernel_mask, device, mask_ratio)
        kernel_tokens = self.norm(kernel_tokens)

        if self.global_pool:
            x = latent[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
            i_x = i_latent[:, 1:, :].mean(dim=1)  # global pool without cls token
            i_outcome = self.fc_norm(i_x)
        else:
            x = self.norm(latent)
            outcome = x[:, 0]
            i_x = self.norm_i(i_latent)
            i_outcome = i_x[:, 0]
        logits = self.head(outcome)
        i_logits = self.head_i(i_outcome)
        return logits, kernel_tokens, i_logits


def IPCA_vit_base_patch16_dec512d8b(**kwargs):
    model = IpcaViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def IPCA_vit_large_patch16_dec512d8b(**kwargs):
    model = IpcaViT(
        img_size=2048, patch_size=16, embed_dim=1024, depth=4, num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def IPCA_vit_huge_patch14_dec512d8b(**kwargs):
    model = IpcaViT(
        patch_size=14, embed_dim=1024, depth=32, num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def IPCA_vit_base_dec512d8b(**kwargs):
    model = IpcaViT(**kwargs)
    return model


# set recommended archs
IPCA_vit_base_patch16 = IPCA_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
IPCA_vit_large_patch16 = IPCA_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
IPCA_vit_huge_patch14 = IPCA_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
IPCA_vit_base = IPCA_vit_base_dec512d8b
