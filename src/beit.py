# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import collections.abc
from functools import partial
from itertools import repeat

import numpy as np
from mindspore import context
from mindspore import dtype as mstype
from mindspore import nn, Parameter, Tensor
from mindspore import ops as P


class DropPath(nn.Cell):
    def __init__(self, drop_prob):
        super(DropPath, self).__init__()
        self.keep_prob = 1 - drop_prob
        seed = 1
        self.rand = P.UniformReal(seed=seed)
        self.shape = P.Shape()
        self.floor = P.Floor()

    def construct(self, x):
        if self.training:
            x_shape = self.shape(x)
            random_tensor = self.rand((x_shape[0], 1, 1))
            random_tensor = random_tensor + self.keep_prob
            random_tensor = self.floor(random_tensor)
            x = x / self.keep_prob
            x = x * random_tensor
        return x


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


class Mlp(nn.Cell):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Dense(in_channels=in_features, out_channels=hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Dense(in_channels=hidden_features, out_channels=out_features)
        self.drop = nn.Dropout(keep_prob=1.0 - drop)

    def construct(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Cell):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
                 proj_drop=0., window_size=None, attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Dense(in_channels=dim, out_channels=all_head_dim, has_bias=qkv_bias)
        self.k = nn.Dense(in_channels=dim, out_channels=all_head_dim, has_bias=False)
        self.v = nn.Dense(in_channels=dim, out_channels=all_head_dim, has_bias=qkv_bias)

        if window_size:
            self.window_size = window_size
            self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
            self.relative_position_bias_table = Parameter(
                Tensor(np.zeros([self.num_relative_distance, num_heads]), dtype=mstype.float32))  # 2*Wh-1 * 2*Ww-1, nH
            # cls to token & token 2 cls & cls to cls

            # get pair-wise relative position index for each token inside the window
            coords_h = np.arange(self.window_size[0]).reshape(self.window_size[0], 1).repeat(self.window_size[0],
                                                                                             1).reshape(1, -1)
            coords_w = np.arange(self.window_size[1]).reshape(1, self.window_size[1]).repeat(self.window_size[1],
                                                                                             0).reshape(1, -1)
            coords_flatten = np.concatenate([coords_h, coords_w], axis=0)  # 2, Wh, Ww
            relative_coords = coords_flatten[:, :, np.newaxis] - coords_flatten[:, np.newaxis, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.transpose(1, 2, 0)  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * window_size[1] - 1
            relative_position_index = \
                np.zeros((window_size[0] * window_size[1] + 1,) * 2)
            relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            relative_position_index[0, 0:] = self.num_relative_distance - 3
            relative_position_index[0:, 0] = self.num_relative_distance - 2
            relative_position_index[0, 0] = self.num_relative_distance - 1
            self.relative_position_index = Parameter(Tensor(relative_position_index, dtype=mstype.int32),
                                                     requires_grad=False)
        else:
            self.window_size = None
            self.relative_position_bias_table = None
            self.relative_position_index = None

        self.attn_drop = nn.Dropout(keep_prob=1.0 - attn_drop)
        self.proj = nn.Dense(in_channels=all_head_dim, out_channels=dim)
        self.proj_drop = nn.Dropout(keep_prob=1.0 - proj_drop)
        self.batch_matmul = P.BatchMatMul(transpose_b=True)
        self.softmax = nn.Softmax(axis=-1)
        self.gather = P.Gather()

    def construct(self, x, rel_pos_bias=None):
        B, N, C = x.shape

        q = self.q(x).reshape(B, N, self.num_heads, -1).transpose(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, -1).transpose(0, 2, 1, 3)
        v = self.v(x).reshape(B, N, self.num_heads, -1).transpose(0, 2, 1, 3)
        q = q * self.scale
        attn = self.batch_matmul(q, k)

        if self.relative_position_bias_table is not None:
            relative_position_bias = \
                self.relative_position_bias_table[self.relative_position_index.reshape(-1)].reshape(
                    self.window_size[0] * self.window_size[1] + 1,
                    self.window_size[0] * self.window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH
            # relative_position_bias = self.gather(self.relative_position_bias_table,
            #                                      self.relative_position_index.reshape(-1), 0)
            # relative_position_bias = relative_position_bias.reshape(
            #     self.window_size[0] * self.window_size[1] + 1,
            #     self.window_size[0] * self.window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH

            relative_position_bias = P.Transpose()(relative_position_bias, (2, 0, 1,))  # nH, Wh*Ww, Wh*Ww
            attn = attn + P.ExpandDims()(relative_position_bias, 0)

        if rel_pos_bias is not None:
            attn = attn + rel_pos_bias

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = P.BatchMatMul()(attn, v).transpose(0, 2, 1, 3).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Cell):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 window_size=None, attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer((dim,))
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, window_size=window_size, attn_head_dim=attn_head_dim)

        self.drop_path = DropPath(drop_path)
        self.norm2 = norm_layer((dim,))
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values is not None and init_values > 0:
            self.gamma_1 = Parameter(Tensor(init_values * np.ones((dim)), dtype=mstype.float32), requires_grad=True)
            self.gamma_2 = Parameter(Tensor(init_values * np.ones((dim)), dtype=mstype.float32), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def construct(self, x, rel_pos_bias=None):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Cell):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_channels=in_chans, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size,
                              pad_mode='pad', has_bias=True)

    def construct(self, x):
        x = self.proj(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).transpose(0, 2, 1)
        return x


class RelativePositionBias(nn.Cell):

    def __init__(self, window_size, num_heads):
        super().__init__()
        self.window_size = window_size
        self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
        self.relative_position_bias_table = Parameter(
            np.zeros(self.num_relative_distance, num_heads))  # 2*Wh-1 * 2*Ww-1, nH
        # cls to token & token 2 cls & cls to cls

        # get pair-wise relative position index for each token inside the window
        coords_h = np.arange(self.window_size[0]).reshape(self.window_size[0], 1).repeat(self.window_size[0],
                                                                                         1).reshape(1, -1)
        coords_w = np.arange(self.window_size[1]).reshape(1, self.window_size[1]).repeat(self.window_size[1],
                                                                                         0).reshape(1, -1)
        coords_flatten = np.concatenate([coords_h, coords_w], axis=0)  # 2, Wh, Ww
        coords_flatten = coords_flatten.reshape(coords_flatten.shape[0], -1)
        relative_coords = coords_flatten[:, :, np.newaxis] - coords_flatten[:, np.newaxis, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.transpose(1, 2, 0)  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = \
            np.zeros((window_size[0] * window_size[1] + 1,) * 2)
        relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        relative_position_index[0, 0:] = self.num_relative_distance - 3
        relative_position_index[0:, 0] = self.num_relative_distance - 2
        relative_position_index[0, 0] = self.num_relative_distance - 1

        self.relative_position_index = Parameter(Tensor(relative_position_index, dtype=mstype.float32),
                                                 requires_grad=False)
        self.gather = P.Gather()

    def construct(self):
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.reshape(-1)]
        # relative_position_bias = self.gather(self.relative_position_bias_table,
        #                                      self.relative_position_index.reshape(-1), 0)
        relative_position_bias = relative_position_bias.reshape(
            self.window_size[0] * self.window_size[1] + 1,
            self.window_size[0] * self.window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH
        return P.Transpose()(relative_position_bias, (2, 0, 1,))  # nH, Wh*Ww, Wh*Ww


class VisionTransformer(nn.Cell):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=0.1,
                 use_abs_pos_emb=False, use_rel_pos_bias=True, use_shared_rel_pos_bias=False,
                 use_mean_pooling=True):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = Parameter(Tensor(np.zeros([1, 1, embed_dim]), dtype=mstype.float32))
        # self.mask_token = Parameter(torch.zeros(1, 1, embed_dim))
        if use_abs_pos_emb:
            self.pos_embed = Parameter(Tensor(np.zeros([1, num_patches + 1, embed_dim]), dtype=mstype.float32))
        else:
            self.pos_embed = None
        self.pos_drop = nn.Dropout(keep_prob=1.0 - drop_rate)

        if use_shared_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias(window_size=self.patch_embed.patch_shape, num_heads=num_heads)
        else:
            self.rel_pos_bias = None

        dpr = [x for x in np.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.use_rel_pos_bias = use_rel_pos_bias
        self.blocks = nn.CellList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, window_size=self.patch_embed.patch_shape if use_rel_pos_bias else None)
            for i in range(depth)])
        self.norm = P.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer((embed_dim,)) if use_mean_pooling else None
        self.head = nn.Dense(in_channels=embed_dim, out_channels=num_classes)

    def forward_features(self, x):
        x = self.patch_embed(x)
        batch_size, seq_len, _ = P.Shape()(x)

        cls_tokens = P.Tile()(self.cls_token, (batch_size, 1, 1))
        x = P.Concat(1)((cls_tokens, x))
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks:
            x = blk(x, rel_pos_bias=rel_pos_bias)

        x = self.norm(x)
        if self.fc_norm is not None:
            t = x[:, 1:, :]
            return self.fc_norm(P.ReduceMean()(t, 1))
        else:
            return x[:, 0]

    def construct(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def beit_base_patch16_224(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), **kwargs)
    return model


if __name__ == "__main__":
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    data = Tensor(np.ones([1, 3, 224, 224]), dtype=mstype.float32)
    model = beit_base_patch16_224()
    out = model(data)
    print(out.shape)
