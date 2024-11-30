'''
 * DreamClear: High-Capacity Real-World Image Restoration with Privacy-Safe Dataset Curation
 * Modified from PixArt-alpha by Yuang Ai
 * 13/10/2024
'''
import re
import torch
import torch.nn as nn

from copy import deepcopy
from torch import Tensor
from torch.nn import Module, Linear, init
from typing import Any, Mapping

from diffusion.model.nets import PixArtMSBlock, PixArtMS, PixArt
from diffusion.model.utils import auto_grad_checkpoint, to_2tuple

import numpy as np


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0, lewei_scale=1.0, base_size=16):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    if isinstance(grid_size, int):
        grid_size = to_2tuple(grid_size)
    grid_h = np.arange(grid_size[0], dtype=np.float32) / (grid_size[0]/base_size) / lewei_scale
    grid_w = np.arange(grid_size[1], dtype=np.float32) / (grid_size[1]/base_size) / lewei_scale
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size[1], grid_size[0]])

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    return np.concatenate([emb_h, emb_w], axis=1)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    return np.concatenate([emb_sin, emb_cos], axis=1)

# The implementation of ControlNet-Half architrecture
# https://github.com/lllyasviel/ControlNet/discussions/188
class ControlT2IDitBlockHalf(Module):
    def __init__(self, base_block: PixArtMSBlock, block_index: 0) -> None:
        super().__init__()
        self.copied_block = deepcopy(base_block)
        self.block_index = block_index

        for p in self.copied_block.parameters():
            p.requires_grad_(True)

        self.copied_block.load_state_dict(base_block.state_dict())
        self.copied_block.train()
        
        self.hidden_size = hidden_size = base_block.hidden_size
        if self.block_index == 0:
            self.before_proj = Linear(hidden_size, hidden_size)
            init.zeros_(self.before_proj.weight)
            init.zeros_(self.before_proj.bias)
        self.after_proj = Linear(hidden_size, hidden_size) 
        init.zeros_(self.after_proj.weight)
        init.zeros_(self.after_proj.bias)

    def forward(self, x, y, t, mask=None, c=None):
        
        if self.block_index == 0:
            # the first block
            c = self.before_proj(c)
            c = self.copied_block(x + c, y, t, mask)
            c_skip = self.after_proj(c)
        else:
            # load from previous c and produce the c for skip connection
            c = self.copied_block(c, y, t, mask)
            c_skip = self.after_proj(c)
        
        return c, c_skip
        

# The implementation of ControlPixArtHalf net
class ControlPixArtHalf(Module):
    # only support single res model
    def __init__(self, base_model: PixArt, copy_blocks_num: int = 13) -> None:
        super().__init__()
        self.base_model = base_model.eval()
        self.controlnet = []
        self.copy_blocks_num = copy_blocks_num
        self.total_blocks_num = len(base_model.blocks)
        for p in self.base_model.parameters():
            p.requires_grad_(False)

        # Copy first copy_blocks_num block
        for i in range(copy_blocks_num):
            self.controlnet.append(ControlT2IDitBlockHalf(base_model.blocks[i], i))
        self.controlnet = nn.ModuleList(self.controlnet)
    
    def __getattr__(self, name: str) -> Tensor or Module:
        if name in ['forward', 'forward_with_dpmsolver', 'forward_with_cfg', 'forward_c', 'load_state_dict']:
            return self.__dict__[name]
        elif name in ['base_model', 'controlnet']:
            return super().__getattr__(name)
        else:
            return getattr(self.base_model, name)

    def forward_c(self, c):
        self.h, self.w = c.shape[-2]//self.patch_size, c.shape[-1]//self.patch_size
        pos_embed = torch.from_numpy(get_2d_sincos_pos_embed(self.pos_embed.shape[-1], (self.h, self.w), lewei_scale=self.lewei_scale, base_size=self.base_size)).unsqueeze(0).to(c.device).to(self.dtype)
        return self.x_embedder(c) + pos_embed if c is not None else c

    # def forward(self, x, t, c, **kwargs):
    #     return self.base_model(x, t, c=self.forward_c(c), **kwargs)
    def forward(self, x, timestep, y, mask=None, data_info=None, c=None, **kwargs):
        # modify the original PixArtMS forward function
        if c is not None:
            c = c.to(self.dtype)
            c = self.forward_c(c)
        """
        Forward pass of PixArt.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N, 1, 120, C) tensor of class labels
        """
        x = x.to(self.dtype)
        timestep = timestep.to(self.dtype)
        y = y.to(self.dtype)
        pos_embed = self.pos_embed.to(self.dtype)
        self.h, self.w = x.shape[-2]//self.patch_size, x.shape[-1]//self.patch_size
        x = self.x_embedder(x) + pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(timestep.to(x.dtype))  # (N, D)
        t0 = self.t_block(t)
        y = self.y_embedder(y, self.training)  # (N, 1, L, D)
        if mask is not None:
            if mask.shape[0] != y.shape[0]:
                mask = mask.repeat(y.shape[0] // mask.shape[0], 1)
            mask = mask.squeeze(1).squeeze(1)
            y = y.squeeze(1).masked_select(mask.unsqueeze(-1) != 0).view(1, -1, x.shape[-1])
            y_lens = mask.sum(dim=1).tolist()
        else:
            y_lens = [y.shape[2]] * y.shape[0]
            y = y.squeeze(1).view(1, -1, x.shape[-1])

        # define the first layer
        x = auto_grad_checkpoint(self.base_model.blocks[0], x, y, t0, y_lens, **kwargs)  # (N, T, D) #support grad checkpoint

        if c is not None:
            # update c
            for index in range(1, self.copy_blocks_num + 1):
                c, c_skip = auto_grad_checkpoint(self.controlnet[index - 1], x, y, t0, y_lens, c, **kwargs)
                x = auto_grad_checkpoint(self.base_model.blocks[index], x + c_skip, y, t0, y_lens, **kwargs)
        
            # update x
            for index in range(self.copy_blocks_num + 1, self.total_blocks_num):
                x = auto_grad_checkpoint(self.base_model.blocks[index], x, y, t0, y_lens, **kwargs)
        else:
            for index in range(1, self.total_blocks_num):
                x = auto_grad_checkpoint(self.base_model.blocks[index], x, y, t0, y_lens, **kwargs)

        x = self.final_layer(x, t)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        return x

    def forward_with_dpmsolver(self, x, t, y, data_info, c, **kwargs):
        model_out = self.forward(x, t, y, data_info=data_info, c=c, **kwargs)
        return model_out.chunk(2, dim=1)[0]

    # def forward_with_dpmsolver(self, x, t, y, data_info, c, **kwargs):
    #     return self.base_model.forward_with_dpmsolver(x, t, y, data_info=data_info, c=self.forward_c(c), **kwargs)

    def forward_with_cfg(self, x, t, y, cfg_scale, data_info, c, **kwargs):
        return self.base_model.forward_with_cfg(x, t, y, cfg_scale, data_info, c=self.forward_c(c), **kwargs)

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        if all((k.startswith('base_model') or k.startswith('controlnet')) for k in state_dict.keys()):
            return super().load_state_dict(state_dict, strict)
        else:
            new_key = {}
            for k in state_dict.keys():
                new_key[k] = re.sub(r"(blocks\.\d+)(.*)", r"\1.base_block\2", k)
            for k, v in new_key.items():
                if k != v:
                    print(f"replace {k} to {v}")
                    state_dict[v] = state_dict.pop(k)

            return self.base_model.load_state_dict(state_dict, strict)
    
    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        assert self.h * self.w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], self.h, self.w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, self.h * p, self.w * p))
        return imgs

    @property
    def dtype(self):
        # 返回模型参数的数据类型
        return next(self.parameters()).dtype


# The implementation for PixArtMS_Half + 1024 resolution
class ControlPixArtMSHalf(ControlPixArtHalf):
    # support multi-scale res model (multi-scale model can also be applied to single reso training & inference)
    def __init__(self, base_model: PixArtMS, copy_blocks_num: int = 13) -> None:
        super().__init__(base_model=base_model, copy_blocks_num=copy_blocks_num)

    def forward(self, x, timestep, y, mask=None, data_info=None, c=None, **kwargs):
        # modify the original PixArtMS forward function
        """
        Forward pass of PixArt.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N, 1, 120, C) tensor of class labels
        """
        if c is not None:
            c = c.to(self.dtype)
            c = self.forward_c(c)
        bs = x.shape[0]
        x = x.to(self.dtype)
        timestep = timestep.to(self.dtype)
        y = y.to(self.dtype)
        c_size, ar = data_info['img_hw'].to(self.dtype), data_info['aspect_ratio'].to(self.dtype)
        self.h, self.w = x.shape[-2]//self.patch_size, x.shape[-1]//self.patch_size

        pos_embed = torch.from_numpy(get_2d_sincos_pos_embed(self.pos_embed.shape[-1], (self.h, self.w), lewei_scale=self.lewei_scale, base_size=self.base_size)).unsqueeze(0).to(x.device).to(self.dtype)
        x = self.x_embedder(x) + pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(timestep)  # (N, D)
        csize = self.csize_embedder(c_size, bs)  # (N, D)
        ar = self.ar_embedder(ar, bs)  # (N, D)
        t = t + torch.cat([csize, ar], dim=1)
        t0 = self.t_block(t)
        y = self.y_embedder(y, self.training)  # (N, D)
        if mask is not None:
            if mask.shape[0] != y.shape[0]:
                mask = mask.repeat(y.shape[0] // mask.shape[0], 1)
            mask = mask.squeeze(1).squeeze(1)
            y = y.squeeze(1).masked_select(mask.unsqueeze(-1) != 0).view(1, -1, x.shape[-1])
            y_lens = mask.sum(dim=1).tolist()
        else:
            y_lens = [y.shape[2]] * y.shape[0]
            y = y.squeeze(1).view(1, -1, x.shape[-1])

        # define the first layer
        x = auto_grad_checkpoint(self.base_model.blocks[0], x, y, t0, y_lens, **kwargs)  # (N, T, D) #support grad checkpoint

        if c is not None:
            # update c
            for index in range(1, self.copy_blocks_num + 1):
                c, c_skip = auto_grad_checkpoint(self.controlnet[index - 1], x, y, t0, y_lens, c, **kwargs)
                x = auto_grad_checkpoint(self.base_model.blocks[index], x + c_skip, y, t0, y_lens, **kwargs)
        
            # update x
            for index in range(self.copy_blocks_num + 1, self.total_blocks_num):
                x = auto_grad_checkpoint(self.base_model.blocks[index], x, y, t0, y_lens, **kwargs)
        else:
            for index in range(1, self.total_blocks_num):
                x = auto_grad_checkpoint(self.base_model.blocks[index], x, y, t0, y_lens, **kwargs)

        x = self.final_layer(x, t)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        return x

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

class PatchEmbed_Zero(Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(
            self,
            patch_size=16,
            in_chans=3,
            embed_dim=768,
            norm_layer=None,
            flatten=True,
            bias=True,
    ):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.flatten = flatten
        self.proj = zero_module(nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias))
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(
            self,
            patch_size=16,
            in_chans=3,
            embed_dim=768,
            norm_layer=None,
            flatten=True,
            bias=True,
    ):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.flatten = flatten
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x

import xformers.ops

class Zero_MultiHeadCrossAttentionDe(Module):
    def __init__(self, d_model, num_heads, attn_drop=0., proj_drop=0., **block_kwargs):
        super(Zero_MultiHeadCrossAttentionDe, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_linear = Linear(d_model, d_model)
        self.kv_linear = Linear(d_model, d_model*2)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = zero_module(Linear(d_model, d_model))
        self.proj_drop = nn.Dropout(proj_drop)
        self.de_proj = Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # self.norm3 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)

    def forward(self, x, cond, mask=None):
        # query/value: img tokens; key: condition; mask: if padding tokens
        B, N, C = x.shape
        x_in = x

        q = self.q_linear(self.norm1(x)).view(1, -1, self.num_heads, self.head_dim)
        kv = self.kv_linear(self.norm2(cond)).view(1, -1, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(2)
        attn_bias = None
        if mask is not None:
            attn_bias = xformers.ops.fmha.BlockDiagonalMask.from_seqlens([N] * B, mask)
        x = xformers.ops.memory_efficient_attention(q, k, v, p=self.attn_drop.p, attn_bias=attn_bias)
        x = x.view(B, -1, C)
        de_map = self.de_proj(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, de_map

class ControlT2IDitBlockHalfSR(Module):
    def __init__(self, base_block: PixArtMSBlock, block_index: 0) -> None:
        super().__init__()
        self.copied_block = deepcopy(base_block)
        self.block_index = block_index

        for p in self.copied_block.parameters():
            p.requires_grad_(True)

        self.copied_block.load_state_dict(base_block.state_dict())
        self.copied_block.train()
        
        self.hidden_size = hidden_size = base_block.hidden_size

    def forward(self, x, y, t, mask=None, c=None):
        if self.block_index == 0:
            c = self.copied_block(x + c, y, t, mask)      
        else:
            c = self.copied_block(c, y, t, mask)
        return c

class Adaptive_Modulator(Module):
    def __init__(self, base_block: PixArtMSBlock) -> None:
        super().__init__()
        
        self.hidden_size = hidden_size = base_block.hidden_size
        self.shared_proj = Linear(hidden_size, hidden_size//2)
        self.act =  nn.GELU(approximate="tanh")
        self.proj_add = Linear(hidden_size//2, hidden_size)
        self.proj_mul = Linear(hidden_size//2, hidden_size)
        init.zeros_(self.proj_add.weight)
        init.zeros_(self.proj_mul.weight)
        init.zeros_(self.proj_add.bias)
        init.zeros_(self.proj_mul.bias)

    def forward(self, c):
        actv = self.act(self.shared_proj(c))
        gamma = self.proj_mul(actv)
        beta = self.proj_add(actv)   
        return gamma, beta

class Adaptive_Modulator_MoE(Module):
    def __init__(self, base_block: PixArtMSBlock, num_experts=3) -> None:
        super().__init__()
        
        self.hidden_size = hidden_size = base_block.hidden_size
        self.experts = nn.ModuleList([Adaptive_Modulator(base_block) for _ in range(num_experts)])
        self.gate = nn.Sequential(nn.Linear(hidden_size,num_experts*8),nn.GELU(approximate="tanh"),zero_module(nn.Linear(num_experts*8,num_experts))) 

    def forward(self, c, de_map):
        #c: B, N, C; de_map: B, N, C
        gate_score = self.gate(de_map) # B N NUM_EXP
        gate_score = gate_score.softmax(dim=-1)
        gamma_list = []
        beta_list = []
        for expert in self.experts:
            gamma, beta = expert(c)
            gamma_list.append(gamma.unsqueeze(-1)) #B N C 1
            beta_list.append(beta.unsqueeze(-1))
        gamma = torch.cat(gamma_list,dim=-1) # B N C NUM_EXP
        beta = torch.cat(beta_list,dim=-1) # B N C NUM_EXP
        gate_score = gate_score.unsqueeze(-2) #B, N, 1, num_exp
        gamma = gamma*gate_score #B, N, C, num_exp
        gamma = torch.sum(gamma,dim=-1) #B, N, C
        beta = beta*gate_score #B, N, C, num_exp
        beta = torch.sum(beta,dim=-1) #B, N, C
        return gamma, beta


class ControlPixArtMSHalfSR2Branch(Module):
    # support multi-scale res model (multi-scale model can also be applied to single reso training & inference)
    def __init__(self, base_model: PixArtMS, copy_blocks_num: int = 13) -> None:
        super().__init__()
        self.base_model = base_model.eval()
        self.controlnet_lq = []
        self.controlnet_pre = []
        self.am_lq = []
        self.am_pre = []
        self.cross_de = []
        self.copy_blocks_num = copy_blocks_num
        self.total_blocks_num = len(base_model.blocks)
        for p in self.base_model.parameters():
            p.requires_grad_(False)

        # Copy first copy_blocks_num block
        for i in range(copy_blocks_num):
            self.controlnet_lq.append(ControlT2IDitBlockHalfSR(base_model.blocks[i], i))
        self.controlnet_lq = nn.ModuleList(self.controlnet_lq)

        for i in range(copy_blocks_num):
            self.am_lq.append(Adaptive_Modulator_MoE(base_model.blocks[i]))
        self.am_lq = nn.ModuleList(self.am_lq)

        for i in range(copy_blocks_num):
            self.controlnet_pre.append(ControlT2IDitBlockHalfSR(base_model.blocks[i], i))
        self.controlnet_pre = nn.ModuleList(self.controlnet_pre)

        for i in range(copy_blocks_num):
            self.am_pre.append(Adaptive_Modulator(base_model.blocks[i]))
        self.am_pre = nn.ModuleList(self.am_pre)

        for i in range(copy_blocks_num):
            self.cross_de.append(Zero_MultiHeadCrossAttentionDe(1152,16))
        self.cross_de = nn.ModuleList(self.cross_de)    

        self.c_embedder_lq = PatchEmbed_Zero(patch_size=2,in_chans=4,embed_dim=1152,bias=True)
        for p in self.c_embedder_lq.parameters():
            p.requires_grad_(True)
        self.c_embedder_lq.train()

        self.c_embedder_pre = PatchEmbed_Zero(patch_size=2,in_chans=4,embed_dim=1152,bias=True)
        for p in self.c_embedder_pre.parameters():
            p.requires_grad_(True)
        self.c_embedder_pre.train()

        self.x_embedder_lq = deepcopy(base_model.x_embedder)
        for p in self.x_embedder_lq.parameters():
            p.requires_grad_(True)
        self.x_embedder_lq.load_state_dict(base_model.x_embedder.state_dict())
        self.x_embedder_lq.train()

        self.x_embedder_pre = deepcopy(base_model.x_embedder)
        for p in self.x_embedder_pre.parameters():
            p.requires_grad_(True)
        self.x_embedder_pre.load_state_dict(base_model.x_embedder.state_dict())
        self.x_embedder_pre.train()
    
    def __getattr__(self, name: str) -> Tensor or Module:
        if name in ['forward', 'forward_with_dpmsolver', 'forward_with_cfg', 'forward_c_lq','forward_c_pre', 'load_state_dict']:
            return self.__dict__[name]
        elif name in ['base_model', 'controlnet_lq','controlnet_pre','c_embedder_lq','c_embedder_pre','x_embedder_lq','x_embedder_pre','cross_de','am_lq',"am_pre"]:
            return super().__getattr__(name)
        else:
            return getattr(self.base_model, name)

    def forward_c_lq(self, c):
        self.h, self.w = c.shape[-2]//self.patch_size, c.shape[-1]//self.patch_size
        pos_embed = torch.from_numpy(get_2d_sincos_pos_embed(self.pos_embed.shape[-1], (self.h, self.w), lewei_scale=self.lewei_scale, base_size=self.base_size)).unsqueeze(0).to(c.device).to(self.dtype)
        return self.c_embedder_lq(c) + pos_embed if c is not None else c

    def forward_c_pre(self, c):
        self.h, self.w = c.shape[-2]//self.patch_size, c.shape[-1]//self.patch_size
        pos_embed = torch.from_numpy(get_2d_sincos_pos_embed(self.pos_embed.shape[-1], (self.h, self.w), lewei_scale=self.lewei_scale, base_size=self.base_size)).unsqueeze(0).to(c.device).to(self.dtype)
        return self.c_embedder_pre(c) + pos_embed if c is not None else c

    def forward(self, x, timestep, y, mask=None, data_info=None, c_lq=None, c_pre=None, **kwargs):
        # modify the original PixArtMS forward function
        """
        Forward pass of PixArt.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N, 1, 120, C) tensor of class labels
        """
        if c_lq is not None:
            c_lq = c_lq.to(self.dtype)
            c_lq = self.c_embedder_lq(c_lq)

        if c_pre is not None:
            c_pre = c_pre.to(self.dtype)
            c_pre = self.c_embedder_pre(c_pre)

        bs = x.shape[0]
        x = x.to(self.dtype)
        timestep = timestep.to(self.dtype)
        y = y.to(self.dtype)
        c_size, ar = data_info['img_hw'].to(self.dtype), data_info['aspect_ratio'].to(self.dtype)
        self.h, self.w = x.shape[-2]//self.patch_size, x.shape[-1]//self.patch_size

        pos_embed = torch.from_numpy(get_2d_sincos_pos_embed(self.pos_embed.shape[-1], (self.h, self.w), lewei_scale=self.lewei_scale, base_size=self.base_size)).unsqueeze(0).to(x.device).to(self.dtype)
        x_lq = self.x_embedder_lq(x) + pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        x_pre = self.x_embedder_pre(x) + pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        x = self.x_embedder(x) + pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(timestep)  # (N, D)
        csize = self.csize_embedder(c_size, bs)  # (N, D)
        ar = self.ar_embedder(ar, bs)  # (N, D)
        t = t + torch.cat([csize, ar], dim=1)
        t0 = self.t_block(t)
        y = self.y_embedder(y, self.training)  # (N, D)
        if mask is not None:
            if mask.shape[0] != y.shape[0]:
                mask = mask.repeat(y.shape[0] // mask.shape[0], 1, 1, 1)
            mask = mask.squeeze(1).squeeze(1)
            y = y.squeeze(1).masked_select(mask.unsqueeze(-1) != 0).view(1, -1, x.shape[-1])
            y_lens = mask.sum(dim=1).tolist()
        else:
            y_lens = [y.shape[2]] * y.shape[0]
            y = y.squeeze(1).view(1, -1, x.shape[-1])
        if c_lq is not None and c_pre is not None:
            for index in range(0, self.copy_blocks_num):
                if index == 0:
                    c_lq = self.controlnet_lq[index]( x_lq, y, t0, y_lens, c_lq, **kwargs)
                    c_pre = self.controlnet_pre[index](x_pre, y, t0, y_lens, c_pre, **kwargs)
                    x = self.base_model.blocks[index]( x, y, t0, y_lens, **kwargs)
                    ref, de_map = self.cross_de[index](c_lq, c_pre)
                    x = x + ref
                    gamma1, beta1 =  self.am_pre[index](c_pre)
                    gamma2, beta2 = self.am_lq[index](c_lq, de_map)
                    x = (1+gamma1)*x+beta1
                    x = (1+gamma2)*x+beta2
                else:
                    c_lq = auto_grad_checkpoint(self.controlnet_lq[index], None, y, t0, y_lens, c_lq, **kwargs)
                    c_pre = auto_grad_checkpoint(self.controlnet_pre[index], None, y, t0, y_lens, c_pre, **kwargs)
                    x = auto_grad_checkpoint(self.base_model.blocks[index], x, y, t0, y_lens, **kwargs)
                    ref, de_map =self.cross_de[index](c_lq, c_pre)
                    x = x + ref
                    gamma1, beta1 = self.am_pre[index](c_pre)
                    gamma2, beta2 = self.am_lq[index](c_lq, de_map)
                    x = (1+gamma1)*x+beta1
                    x = (1+gamma2)*x+beta2

            for index in range(self.copy_blocks_num,self.total_blocks_num):
                x = auto_grad_checkpoint(self.base_model.blocks[index], x, y, t0, y_lens, **kwargs)
        else:
            for index in range(self.copy_blocks_num,self.total_blocks_num):
                x = auto_grad_checkpoint(self.base_model.blocks[index], x, y, t0, y_lens, **kwargs)

        x = self.final_layer(x, t)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        return x

    def forward_with_dpmsolver(self, x, t, y, data_info, c_lq, c_pre, **kwargs):
        model_out = self.forward(x, t, y, data_info=data_info, c_lq=c_lq, c_pre=c_pre,**kwargs)
        return model_out.chunk(2, dim=1)[0]

    
    def forward_with_cfg(self, x, timestep, y, cfg_scale, data_info, c_lq,c_pre, **kwargs):
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, timestep, y, data_info=data_info, c_lq=c_lq, c_pre=c_pre, **kwargs)
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

    def _gaussian_weights(self, tile_width, tile_height, nbatches, device):
        """Generates a gaussian mask of weights for tile contributions"""
        from numpy import pi, exp, sqrt
        import numpy as np

        latent_width = tile_width
        latent_height = tile_height

        var = 0.01
        midpoint = (latent_width - 1) / 2  # -1 because index goes from 0 to latent_width - 1
        x_probs = [exp(-(x-midpoint)*(x-midpoint)/(latent_width*latent_width)/(2*var)) / sqrt(2*pi*var) for x in range(latent_width)]
        midpoint = latent_height / 2
        y_probs = [exp(-(y-midpoint)*(y-midpoint)/(latent_height*latent_height)/(2*var)) / sqrt(2*pi*var) for y in range(latent_height)]

        weights = np.outer(y_probs, x_probs)
        return torch.tile(torch.tensor(weights, device=device), (nbatches, self.out_channels, 1, 1))

    def forward_with_cfg_tile(self, x, timestep, y, cfg_scale, data_info, c_lq,c_pre, latent_tiled_size, latent_tiled_overlap, **kwargs):
        batch_size = 1
        b, c, h, w = x.size()
        tile_size, tile_overlap = (latent_tiled_size, latent_tiled_overlap)
        assert h*w>tile_size*tile_size


        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)

        latent_model_input  = combined

        tile_weights = self._gaussian_weights(tile_size, tile_size, 1, latent_model_input.device)
        tile_size = min(tile_size, min(h, w))
        tile_weights = self._gaussian_weights(tile_size, tile_size, 1, latent_model_input.device)

        grid_rows = 0
        cur_x = 0
        while cur_x < latent_model_input.size(-1):
            cur_x = max(grid_rows * tile_size-tile_overlap * grid_rows, 0)+tile_size
            grid_rows += 1

        grid_cols = 0
        cur_y = 0
        while cur_y < latent_model_input.size(-2):
            cur_y = max(grid_cols * tile_size-tile_overlap * grid_cols, 0)+tile_size
            grid_cols += 1

        input_list = []
        cond_list_lq = []
        cond_list_pre = []
        noise_preds = []

        for row in range(grid_rows):
            noise_preds_row = []
            for col in range(grid_cols):
                if col < grid_cols-1 or row < grid_rows-1:
                    # extract tile from input image
                    ofs_x = max(row * tile_size-tile_overlap * row, 0)
                    ofs_y = max(col * tile_size-tile_overlap * col, 0)
                    # input tile area on total image
                if row == grid_rows-1:
                    ofs_x = w - tile_size
                if col == grid_cols-1:
                    ofs_y = h - tile_size

                input_start_x = ofs_x
                input_end_x = ofs_x + tile_size
                input_start_y = ofs_y
                input_end_y = ofs_y + tile_size

                # input tile dimensions
                input_tile = latent_model_input[:, :, input_start_y:input_end_y, input_start_x:input_end_x]
                input_list.append(input_tile)
                cond_tile_lq = c_lq[:, :, input_start_y:input_end_y, input_start_x:input_end_x]
                cond_list_lq.append(cond_tile_lq)
                cond_tile_pre = c_pre[:, :, input_start_y:input_end_y, input_start_x:input_end_x]
                cond_list_pre.append(cond_tile_pre)

                if len(input_list) == batch_size or col == grid_cols-1:
                    input_list_t = torch.cat(input_list, dim=0)
                    cond_list_lq_t = torch.cat(cond_list_lq, dim=0)
                    cond_list_pre_t = torch.cat(cond_list_pre, dim=0)

                    model_out = self.forward(input_list_t, timestep, y, data_info=data_info, c_lq=cond_list_lq_t, c_pre=cond_list_pre_t, **kwargs)

                    input_list = []
                    cond_list_lq = []
                    cond_list_pre = []

                noise_preds.append(model_out)

         # Stitch noise predictions for all tiles
        noise_pred = torch.zeros((b,self.out_channels,h,w), device=latent_model_input.device)
        contributors = torch.zeros((b,self.out_channels,h,w), device=latent_model_input.device)
        # Add each tile contribution to overall latents
        for row in range(grid_rows):
            for col in range(grid_cols):
                if col < grid_cols-1 or row < grid_rows-1:
                    # extract tile from input image
                    ofs_x = max(row * tile_size-tile_overlap * row, 0)
                    ofs_y = max(col * tile_size-tile_overlap * col, 0)
                    # input tile area on total image
                if row == grid_rows-1:
                    ofs_x = w - tile_size
                if col == grid_cols-1:
                    ofs_y = h - tile_size

                input_start_x = ofs_x
                input_end_x = ofs_x + tile_size
                input_start_y = ofs_y
                input_end_y = ofs_y + tile_size

                noise_pred[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += noise_preds[row*grid_cols + col] * tile_weights
                contributors[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += tile_weights
        # Average overlapping areas with more than 1 contributor
        noise_pred /= contributors

        model_out = noise_pred
        # model_out = self.forward(combined, timestep, y, data_info=data_info, c_lq=c_lq, c_pre=c_pre, **kwargs)
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)
    
    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        return super().load_state_dict(state_dict, strict)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        assert self.h * self.w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], self.h, self.w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, self.h * p, self.w * p))
        return imgs

    @property
    def dtype(self):
        # 返回模型参数的数据类型
        return next(self.parameters()).dtype