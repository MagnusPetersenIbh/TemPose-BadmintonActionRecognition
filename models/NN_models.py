import torch
import numpy as np
import os
from torch.nn.functional import elu
import math
from torch import nn, einsum
from torch.nn import MultiheadAttention
#import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# classes
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x,mask = None,y=None):
        return self.fn(self.norm(x),mask)
class PreNormCross(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x,mask = None,y=None):
        if y is not None:
            return self.fn(self.norm(x),self.norm(y),mask)
        else:
            return self.fn(self.norm(x),mask)
class PreNormMulti(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x,mask = None,need_weights=False,y=None):
        if y is not None:
            return self.fn(self.norm(x),self.norm(x),self.norm(y),mask,need_weights=need_weights)
        else:
            return self.fn(self.norm(x),self.norm(x),self.norm(x),mask,need_weights=need_weights)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x,mask = None):
        return self.net(x)

class AttentionV2(nn.Module): ### this is doublework 
    def __init__(self, dim, heads = 6, dim_head = 384, dropout = 0.):
        super().__init__()
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)


        self.to_out = nn.Sequential(
            nn.Linear(dim_head, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        self.to_q = nn.Linear(dim, dim_head, bias=False)
        self.to_k = nn.Linear(dim, dim_head, bias=False)
        self.to_v = nn.Linear(dim, dim_head, bias=False)
    def forward(self, x,mask = None):

        q = self.to_q(y)
        
        k = self.to_k(x)
        v = self.to_v(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        x = MultiheadAttention(dim, num_heads = heads, dropout = dropout,bias = False,batch_first=True)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 48, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x,mask = None):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if mask is not None:
            dots = dots.masked_fill(mask == 0, -1e9) 

        attn = self.attend(dots)        
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    
class CrossAttention(nn.Module):  
    def __init__(self, dim, heads=8, dim_head=48, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, y, mask=None):
        q = self.to_q(x)
        k, v = self.to_kv(y).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=self.heads), (q, k, v))

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if mask is not None:
            dots = dots.masked_fill(mask == 0, -1e9) 

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=self.heads)

        return self.to_out(out)
    
class TransformerV2(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.,get_score = False):
        super().__init__()
        #assert  dim % heads == 0
        self.get_att_score = get_score
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNormMulti(dim, MultiheadAttention(dim, num_heads = heads, dropout = dropout,bias = False,batch_first=True)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x,mask = None):
        att_weights = []
        for attn, ff in self.layers:
            if self.get_att_score:
                att_score,att_map = attn(x,mask=mask,need_weights=True)
                att_weights.append(att_map)
                x = att_score + x
            else:
                x = attn(x,mask=mask)[0] + x
                #x = attn(x,x,x,key_padding_mask=mask,need_weights=False) + x
            x = ff(x) + x
        return x,att_weights

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x,mask = None):
        for attn, ff in self.layers:
            x = attn(x,mask) + x
            x = ff(x) + x
        return x


