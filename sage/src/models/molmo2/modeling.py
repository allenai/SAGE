import math
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, Dict, Any, Sequence, Callable

import torch
from torch import nn
from torch.nn import functional as F

from transformers.models.auto import AutoModelForCausalLM, AutoModelForImageTextToText
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.utils import GenerateOutput
from transformers.integrations import use_kernel_forward_from_hub
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_flash_attention_utils import (
    _flash_attention_forward,
    FlashAttentionKwargs,
    flash_attn_supports_top_left_mask,
)
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPast,
    BaseModelOutputWithPooling,
    CausalLMOutputWithPast,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import (
    ModelOutput,
    can_return_tuple,
    is_torch_flex_attn_available,
    logging,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)

from .config import Molmo2Config, Molmo2VitConfig, Molmo2AdapterConfig, Molmo2TextConfig


if is_torch_flex_attn_available():
    from torch.nn.attention.flex_attention import BlockMask

    from transformers.integrations.flex_attention import make_flex_block_causal_mask


logger = logging.get_logger(__name__)


MOLMO_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`Molmo2Config`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@dataclass
class Molmo2CausalLMOutputWithPast(ModelOutput):
    """
    Base class for Molmo2 causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        image_hidden_states (`torch.FloatTensor`, *optional*):
            A `torch.FloatTensor` of size `(batch_size, num_images, sequence_length, hidden_size)`.
            image_hidden_states of the model produced by the vision encoder and after projecting the last hidden state.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    image_hidden_states: Optional[torch.FloatTensor] = None


@dataclass
class Molmo2ModelOutputWithPast(BaseModelOutputWithPast):
    """
    Base class for Molmo2 outputs, with hidden states and attentions.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        image_hidden_states (`torch.FloatTensor`, *optional*):
            A `torch.FloatTensor` of size `(batch_num_patches, hidden_size)`.
            image_hidden_states of the model produced by the vision backbone
    """

    image_hidden_states: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None


class Molmo2PreTrainedModel(PreTrainedModel):
    config_class = Molmo2TextConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Molmo2DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = False
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_attention_backend = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear,)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, Molmo2Embedding):
            module.embedding.data.normal_(mean=0.0, std=std)
            module.new_embedding.data.normal_(mean=0.0, std=std)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, Molmo2RMSNorm):
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            if module.bias is not None:
                module.bias.data.zero_()


class ViTMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, hidden_act: str, device: Union[str, torch.device] = None):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=True, device=device)
        self.act = ACT2FN[hidden_act]
        self.w2 = nn.Linear(hidden_dim, dim, bias=True, device=device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.act(self.w1(x)))


class ViTMultiHeadDotProductAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        use_bias: bool = True,
        input_dim: Optional[int] = None,
        float32_attention: bool = True,
        attention_dropout: float = 0.0,
        residual_dropout: float = 0.0,
        device: Union[str, torch.device] = None,
        attn_implementation: str = "eager",
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.attn_implementation = attn_implementation
        self.is_causal = False

        input_dim = input_dim or hidden_size

        self.wq = nn.Linear(
            input_dim,
            self.num_heads * self.head_dim,
            bias=use_bias,
            device=device,
        )
        self.wk = nn.Linear(
            input_dim,
            self.num_key_value_heads * self.head_dim,
            bias=use_bias,
            device=device,
        )
        self.wv = nn.Linear(
            input_dim,
            self.num_key_value_heads * self.head_dim,
            bias=use_bias,
            device=device,
        )
        self.wo = nn.Linear(
            self.num_heads * self.head_dim,
            self.hidden_size,
        )
        self.float32_attention = float32_attention
        self.attention_dropout = attention_dropout
        self.residual_dropout = nn.Dropout(residual_dropout)

    def _split_heads(self, hidden_states, num_heads) -> torch.Tensor:
        return hidden_states.reshape(hidden_states.shape[:2] + (num_heads, self.head_dim))

    def _merge_heads(self, hidden_states) -> torch.Tensor:
        return hidden_states.reshape(hidden_states.shape[:2] + (self.hidden_size,))
    
    def forward(
        self,
        inputs_q: torch.Tensor,
        inputs_kv: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        if inputs_kv is not None:
            inputs_k = inputs_kv
            inputs_v = inputs_kv
        else:
            inputs_k = inputs_q
            inputs_v = inputs_q

        xq, xk, xv = self.wq(inputs_q), self.wk(inputs_k), self.wv(inputs_v)

        xq = self._split_heads(xq, self.num_heads)
        xk = self._split_heads(xk, self.num_key_value_heads)
        xv = self._split_heads(xv, self.num_key_value_heads)

        if self.num_heads != self.num_key_value_heads:
            xk = xk.repeat_interleave(self.num_key_value_groups, dim=2, output_size=self.num_heads)
            xv = xv.repeat_interleave(self.num_key_value_groups, dim=2, output_size=self.num_heads)
        
        og_dtype = xq.dtype

        if self.float32_attention:
            xq = xq.to(torch.float)
            xk = xk.to(torch.float)
        
        dropout_p = 0.0 if not self.training else self.attention_dropout
        
        if self.attn_implementation == "eager":
            attn_weights = torch.einsum("...qhd,...khd->...hqk", xq / math.sqrt(xq.size(-1)), xk)
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(xq.dtype)
            attn_weights = F.dropout(
                attn_weights,
                p=dropout_p,
                training=self.training
            )
            attn_output = torch.einsum("...hqk,...khd->...qhd", attn_weights.to(xv.dtype), xv)
        
        elif self.attn_implementation == "sdpa":
            if not torch.is_autocast_enabled():
                xv = xv.to(torch.float)
        
            attn_output = F.scaled_dot_product_attention(
                xq.transpose(1, 2).contiguous(),
                xk.transpose(1, 2).contiguous(),
                xv.transpose(1, 2).contiguous(),
                attn_mask=attn_mask,
                is_causal=False,
                dropout_p=dropout_p,
            ).transpose(1, 2)
        
        elif self.attn_implementation == "flash_attention_2":
            if xq.dtype == torch.float32:
                if torch.is_autocast_enabled():
                    target_dtype = torch.get_autocast_gpu_dtype()
                else:
                    target_dtype = self.wq.weight.dtype
            attn_output = _flash_attention_forward(
                xq,
                xk,
                xv,
                attention_mask=attn_mask,
                query_length=inputs_q.shape[1],
                is_causal=False,
                dropout=dropout_p,
                softmax_scale=xq.shape[-1] ** -0.5,
                use_top_left_mask=flash_attn_supports_top_left_mask(),
                target_dtype=target_dtype,
                implementation=self.attn_implementation,
            )
        else:
            raise ValueError(f"Attention implementation {self.attn_implementation} not supported")
        
        attn_output = attn_output.to(og_dtype)
        attn_output = self._merge_heads(attn_output)
        attn_output = self.wo(attn_output)
        attn_output = self.residual_dropout(attn_output)

        return attn_output


class Molmo2VisionBlock(nn.Module):

    def __init__(self, config: Molmo2VitConfig, device: Union[str, torch.device] = None):
        super().__init__()
        self.attention = ViTMultiHeadDotProductAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            float32_attention=config.float32_attention,
            attention_dropout=config.attention_dropout,
            residual_dropout=config.residual_dropout,
            device=device,
            attn_implementation=config._attn_implementation,
        )
        self.feed_forward = ViTMLP(config.hidden_size, config.intermediate_size, config.hidden_act, device=device)
        self.attention_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, device=device)
        self.ffn_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, device=device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x))
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class Molmo2VisionBlockCollection(nn.Module):
    
    def __init__(self, config: Molmo2VitConfig, device: Union[str, torch.device] = None):
        super().__init__()
        self.conifg = config
        self.resblocks = nn.ModuleList([
            Molmo2VisionBlock(config, device) for _ in range(config.num_hidden_layers)
        ])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        hidden_states = []
        for r in self.resblocks:
            x = r(x)
            hidden_states.append(x)
        return hidden_states


class Molmo2VisionTransformer(nn.Module):

    def __init__(self, config: Molmo2VitConfig, device: Union[str, torch.device] = None):
        super().__init__()
        self.config = config

        # positional embeddings
        self.scale = config.hidden_size ** -0.5
        self.num_prefix_tokens: int = 0 # no class embeddings
        self.positional_embedding = nn.Parameter(
            torch.zeros(config.image_num_pos, config.hidden_size, device=device),
        )

        image_patch_size = config.image_patch_size
        self.patch_embedding = nn.Linear(
            image_patch_size * image_patch_size * 3,
            config.hidden_size,
            bias=True,
            device=device,
        )

        self.transformer = Molmo2VisionBlockCollection(config, device)

    def add_pos_emb(self, x: torch.Tensor, patch_num: int) -> torch.Tensor:
        pos_emb = self.positional_embedding

        pos_emb = pos_emb.reshape(
            (int(math.sqrt(pos_emb.shape[0])), int(math.sqrt(pos_emb.shape[0])), pos_emb.shape[1])
        )

        (patch_num_0, patch_num_1) = patch_num

        if pos_emb.shape[0] != patch_num_0 or pos_emb.shape[1] != patch_num_1:
            # Dervied from https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
            # antialias: default True in jax.image.resize
            pos_emb = pos_emb.unsqueeze(0).permute(0, 3, 1, 2)
            pos_emb = F.interpolate(
                pos_emb, size=(patch_num_0, patch_num_1), mode="bicubic", align_corners=False, antialias=True,
            )
            pos_emb = pos_emb.permute(0, 2, 3, 1).squeeze(0)

        pos_emb = pos_emb.reshape(-1, pos_emb.shape[-1])
        x = x + pos_emb[None, :, :].to(x.dtype)
        return x

    def forward(self, x: torch.Tensor, patch_num: int = None) -> List[torch.Tensor]:
        """
        : param x: (batch_size, num_patch, n_pixels)
        """
        if patch_num is None:
            patch_num = self.config.image_num_patch

        B, N, D = x.shape

        x = self.patch_embedding(x)

        # class embeddings and positional embeddings
        x = self.add_pos_emb(x, patch_num)

        hidden_states = self.transformer(x)
        return hidden_states


class ImageProjectorMLP(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        hidden_act: str,
        device: Union[str, torch.device] = None,
    ):
        super().__init__()
        self.w1 = nn.Linear(input_dim, hidden_dim, bias=False, device=device)
        self.w2 = nn.Linear(hidden_dim, output_dim, bias=False, device=device)
        self.w3 = nn.Linear(input_dim, hidden_dim, bias=False, device=device)
        self.act = ACT2FN[hidden_act]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.act(self.w1(x)) * self.w3(x))


class Molmo2VisionBackbone(nn.Module):
    def __init__(self, vit_config: Molmo2VitConfig, adapter_config: Molmo2AdapterConfig):
        super().__init__()
        self.vit_config = vit_config
        self.adapter_config = adapter_config

        self.vit_layers = []
        for layer in adapter_config.vit_layers:
            if layer >= 0:
                self.vit_layers.append(layer)
            else:
                self.vit_layers.append(layer + vit_config.num_hidden_layers)
        
        last_layer_needed = max(self.vit_layers) + 1
        if last_layer_needed < vit_config.num_hidden_layers:
            new_vit_config = deepcopy(vit_config)
            new_vit_config.num_hidden_layers = last_layer_needed
            self.image_vit = Molmo2VisionTransformer(new_vit_config)
        else:
            self.image_vit = Molmo2VisionTransformer(vit_config)

        self.num_prefix_tokens: int = self.image_vit.num_prefix_tokens

        pool_dim = vit_config.hidden_size * len(adapter_config.vit_layers)
        self.image_pooling_2d = ViTMultiHeadDotProductAttention(
            hidden_size=adapter_config.hidden_size,
            num_heads=adapter_config.num_attention_heads,
            num_key_value_heads=adapter_config.num_key_value_heads,
            head_dim=adapter_config.head_dim,
            input_dim=pool_dim,
            float32_attention=adapter_config.float32_attention,
            attention_dropout=adapter_config.attention_dropout,
            residual_dropout=adapter_config.residual_dropout,
            attn_implementation=adapter_config._attn_implementation,
        )
        self.image_projector = ImageProjectorMLP(
            adapter_config.hidden_size,
            adapter_config.intermediate_size,
            adapter_config.text_hidden_size,
            adapter_config.hidden_act,
        )
        self.image_feature_dropout = nn.Dropout(adapter_config.image_feature_dropout)
    
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        : param images: (batch_size, num_crops, num_patch, n_pixels)
        """
        B, T, N, D = images.shape
        images = images.view(B * T, N, D)
        image_features = self.image_vit(images)

        features = []
        for layer in self.vit_layers:
            features.append(image_features[layer])
        image_features = torch.cat(features, dim=-1)

        if self.num_prefix_tokens > 0:
            image_features = image_features[:, 1:]
        image_features = image_features.view(B, T, N, -1)
        return image_features

    @property
    def dtype(self) -> torch.dtype:
        return self.image_vit.patch_embedding.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.image_vit.patch_embedding.weight.device
    
    def forward(
        self,
        images: torch.Tensor,
        pooled_patches_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        # image_features: (batch_size, num_crops(=num_image), num_patch, nximage_emb_dim)
        batch_size, num_image = images.shape[:2]
        images = images.to(device=self.device, dtype=self.dtype)
        image_features = self.encode_image(images)

        image_features = self.image_feature_dropout(image_features)
        dim = image_features.shape[-1]
        valid = pooled_patches_idx >= 0
        valid_token = torch.any(valid, -1)

        # Use `pooled_patches_idx` to arange the features for image pooling
        batch_idx = torch.arange(pooled_patches_idx.shape[0], dtype=torch.long, device=pooled_patches_idx.device)
        batch_idx = torch.tile(batch_idx.view(batch_size, 1, 1), [1, pooled_patches_idx.shape[1], pooled_patches_idx.shape[2]])

        # Now [batch, num_high_res_features, pool_dim, dim]
        to_pool = image_features.reshape(batch_size, -1, dim)[batch_idx, torch.clip(pooled_patches_idx, 0)]
        to_pool = to_pool * valid.to(self.dtype)[:, :, :, None]
        to_pool = to_pool.reshape([-1, pooled_patches_idx.shape[-1], dim])
        if self.adapter_config.pooling_attention_mask:
            attn_mask = valid.reshape([-1, 1, 1, valid.shape[-1]])
            denom = valid.view(-1, to_pool.shape[-2]).float().sum(-1)
            denom = torch.where(denom == 0, 1, denom)
            query = to_pool.sum(-2, keepdim=True) / denom[:, None, None].to(to_pool.dtype)
        else:
            attn_mask = None
            query = to_pool.mean(-2, keepdim=True)
        pooled_features = self.image_pooling_2d(query, to_pool, attn_mask=attn_mask)
        pooled_features = pooled_features.reshape([batch_size, -1, pooled_features.shape[-1]])

        # MLP layer to map the feature.
        pooled_features = self.image_projector(pooled_features)
        return pooled_features.view(-1, pooled_features.shape[-1])[valid_token.flatten()]


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# Copied from transformers.models.llama.modeling_llama.LlamaRotaryEmbedding
class Molmo2RotaryEmbedding(nn.Module):

    def __init__(self, config: Molmo2TextConfig, device: Union[str, torch.device] = None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq
    
    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


@use_kernel_forward_from_hub("RMSNorm")
class Molmo2RMSNorm(nn.Module):

    def __init__(
        self,
        size: int,
        eps: float = 1e-6,
        device: Union[str, torch.device] = None,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size, device=device))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(enabled=False, device_type=x.device.type):
            og_dtype = x.dtype
            x = x.to(torch.float32)
            variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.eps)
            x = x.to(og_dtype)
        
        return self.weight * x

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class Molmo2Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    # copied from transformers.models.llama.modeling_llama.LlamaAttention.__init__ with Llama->Molmo2
    def __init__(self, config: Molmo2TextConfig, layer_idx: Optional[int] = None) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scaling = self.head_dim**-0.5
        self.is_causal = True

        self.fused_dims = (
            config.num_attention_heads * config.head_dim,
            config.head_dim * config.num_key_value_heads,
            config.head_dim * config.num_key_value_heads,
        )
        self.att_proj = nn.Linear(
            config.hidden_size,
            sum(self.fused_dims),
            bias=config.qkv_bias,
        )

        # Layer norms.
        self.k_norm: Optional[Molmo2RMSNorm] = None
        self.q_norm: Optional[Molmo2RMSNorm] = None
        self.qk_norm_type: Optional[str] = None
        if config.use_qk_norm:
            k_norm_size = (
                config.head_dim
                if config.qk_norm_type == "qwen3" else
                config.num_key_value_heads * config.head_dim
            )
            self.k_norm = Molmo2RMSNorm(k_norm_size, eps=config.layer_norm_eps)
            q_norm_size = (
                config.head_dim
                if config.qk_norm_type == "qwen3" else
                config.num_attention_heads * config.head_dim
            )
            self.q_norm = Molmo2RMSNorm(q_norm_size, eps=config.layer_norm_eps)
            self.qk_norm_type = config.qk_norm_type

        self.attention_dropout = config.attention_dropout
        
        self.attn_out = nn.Linear(
            config.head_dim * config.num_attention_heads,
            config.hidden_size,
            bias=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        qkv = self.att_proj(hidden_states)
        query_states, key_states, value_states = qkv.split(self.fused_dims, dim=-1)
        value_states = value_states.view(hidden_shape)

        # Optionally apply layer norm to keys and queries.
        if self.q_norm is not None and self.k_norm is not None and self.qk_norm_type != "qwen3":
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        query_states = query_states.view(hidden_shape)
        key_states = key_states.view(hidden_shape)
        if self.q_norm is not None and self.k_norm is not None and self.qk_norm_type == "qwen3":
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )
    
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.attn_out(attn_output)

        return attn_output, attn_weights


class LanguageModelMLP(nn.Module):

    def __init__(
        self,
        input_dim: int,
        intermediate_size: int,
        hidden_act: str,
        device: Union[str, torch.device] = None,
    ):
        super().__init__()
        self.ff_proj = nn.Linear(input_dim, intermediate_size * 2, bias=False, device=device)
        self.ff_out = nn.Linear(intermediate_size, input_dim, bias=False, device=device)
        self.act = ACT2FN[hidden_act]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ff_proj(x)
        x, gate = x.chunk(2, dim=-1)
        x = self.act(gate) * x
        x = self.ff_out(x)
        return x


class Molmo2DecoderLayer(GradientCheckpointingLayer):

    def __init__(
        self,
        config: Molmo2TextConfig,
        layer_idx: Optional[int] = None,
        device: Union[str, torch.device] = None
    ):
        super().__init__()
        self.config = config

        self.self_attn = Molmo2Attention(config, layer_idx)
        self.attn_norm = Molmo2RMSNorm(
            config.hidden_size, eps=config.layer_norm_eps, device=device)
        self.dropout = nn.Dropout(config.residual_dropout)
        self.mlp = LanguageModelMLP(
            config.hidden_size, config.intermediate_size, config.hidden_act, device=device)
        self.ff_norm = Molmo2RMSNorm(
            config.hidden_size, eps=config.layer_norm_eps, device=device)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """

        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )

        hidden_states = residual + self.dropout(hidden_states)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.ff_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        hidden_states = residual + self.dropout(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


class Molmo2PostNormDecoderLayer(Molmo2DecoderLayer):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """

        residual = hidden_states

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_states = self.attn_norm(hidden_states)

        hidden_states = residual + self.dropout(hidden_states)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.ff_norm(hidden_states)

        hidden_states = residual + self.dropout(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


class Molmo2Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        num_new_embeddings: int,
        features: int,
        device: Union[str, torch.device] = None,
    ):
        super().__init__()
        self.embedding = nn.Parameter(
            torch.zeros(num_embeddings, features, device=device),
        )
        self.new_embedding = nn.Parameter(
            torch.zeros(num_new_embeddings, features, device=device),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.embedding(x, torch.cat([self.embedding, self.new_embedding], dim=0))


MOLMO2_TEXT_ONLY_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance, see our
            [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache);
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`CausalLMOutputWithPast`] instead of a plain tuple.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
"""


@add_start_docstrings(
    "The bare Molmo2 text-only model outputting raw hidden-states without any specific head on top.",
    MOLMO_START_DOCSTRING,
)
class Molmo2TextModel(Molmo2PreTrainedModel):
    def __init__(self, config: Molmo2TextConfig):
        super().__init__(config)
        self.config = config
        if config.additional_vocab_size is not None:
            self.wte = Molmo2Embedding(
                config.vocab_size,
                config.additional_vocab_size,
                config.hidden_size,
            )
        else:
            self.wte = nn.Embedding(config.vocab_size, config.hidden_size)
        self.emb_drop = nn.Dropout(config.embedding_dropout)
        decoder_layer = Molmo2PostNormDecoderLayer if config.norm_after else Molmo2DecoderLayer
        self.blocks = nn.ModuleList(
            [decoder_layer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.ln_f = Molmo2RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.rotary_emb = Molmo2RotaryEmbedding(config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> torch.nn.Module:
        return self.wte

    def set_input_embeddings(self, value: torch.nn.Module) -> None:
        self.wte = value

    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        image_attention_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        # TODO (joao): remove this exception in v4.56 -- it exists for users that try to pass a legacy cache
        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")
        
        if inputs_embeds is None:
            input_ids = input_ids * (input_ids != -1).to(input_ids.dtype)
            inputs_embeds = self.wte(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask,
            inputs_embeds,
            cache_position,
            past_key_values,
            output_attentions,
            image_attention_mask is not None,
        )

        if image_attention_mask is not None:
            bidir_mask = image_attention_mask[:, :, None] & image_attention_mask[:, None, :]
            bidir_mask = bidir_mask.unsqueeze(1).expand(-1, causal_mask.shape[1], -1, -1)
            assert causal_mask.shape == bidir_mask.shape
            if torch.is_floating_point(causal_mask):
                value = torch.tensor(0.0, device=causal_mask.device, dtype=causal_mask.dtype)
            else:
                assert causal_mask.dtype == torch.bool
                value = torch.tensor(True, device=causal_mask.device, dtype=causal_mask.dtype)
            causal_mask = torch.where(
                bidir_mask,
                value,
                causal_mask,
            )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_block in self.blocks[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_block(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **flash_attn_kwargs,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.ln_f(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(
        self,
        attention_mask: Union[torch.Tensor, "BlockMask"],
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool = False,
        use_image_attention_mask: bool = False,
    ):
        if self.config._attn_implementation == "flash_attention_2" and not use_image_attention_mask:
            if attention_mask is not None and (attention_mask == 0.0).any():
                return attention_mask
            return None
        if self.config._attn_implementation == "flex_attention" and not use_image_attention_mask:
            if isinstance(attention_mask, torch.Tensor):
                attention_mask = make_flex_block_causal_mask(attention_mask)
            return attention_mask

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_compilable_cache = past_key_values.is_compileable if past_key_values is not None else False

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_compilable_cache and not output_attentions and not use_image_attention_mask:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype = input_tensor.dtype
        sequence_length = input_tensor.shape[1]
        if using_compilable_cache:
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu", "npu"]
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            min_dtype = torch.finfo(dtype).min
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        cache_position: torch.Tensor,
        batch_size: int,
        **kwargs,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape
                `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache,
                to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=cache_position.device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=cache_position.device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                    causal_mask.device
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )

        return causal_mask


@add_start_docstrings(
    "The Molmo2 text-only model which consists of a language model + lm head.",
    MOLMO_START_DOCSTRING,
)
class Molmo2ForCausalLM(Molmo2PreTrainedModel, GenerationMixin):
    _tied_weights_keys = []  # Weights are not tied
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}
    base_model_prefix = "model"

    def __init__(self, config: Molmo2TextConfig):
        super().__init__(config)
        self.model = Molmo2TextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> torch.nn.Module:
        return self.model.wte

    def set_input_embeddings(self, value: torch.nn.Module) -> None:
        self.model.wte = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value: torch.nn.Module) -> None:
        self.lm_head = value

    def set_decoder(self, decoder: torch.nn.Module) -> None:
        self.model = decoder

    def get_decoder(self) -> torch.nn.Module:
        return self.model

    @can_return_tuple
    @add_start_docstrings_to_model_forward(MOLMO2_TEXT_ONLY_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        r"""
        ```python
        >>> from transformers import AutoTokenizer, Molmo2ForCausalLM

        >>> model = Molmo2ForCausalLM.from_pretrained("...")
        >>> tokenizer = AutoTokenizer.from_pretrained("...")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


MOLMO2_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        images (`torch.FloatTensor` of shape `(batch_size, n_crops, 27*27, 3*14*14)`, *optional*):
            The input crops in with pixel values between 0 and 1 and normalized with SigLIP2 mean/std

            Each crop contains 27x27 patches with 14*14*3 pixel values
        image_masks  (`torch.FloatTensor` of shape `(batch_size, n_crops, n_patches, n_features)`, *optional*):
            Image masks showing what percent of each patch is paddding
        token_pooling (`torch.LongTensor` of shape `(batch_size, n_image_tokens, n_pooled_patches)`):
            For each patch_id tokens in `input_ids`, the indices of the patches in `images`
            to pool for that token, masked with -1
            means ignore the patch.
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance, see our
            [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache);
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`Molmo2CausalLMOutputWithPast`] instead of a plain tuple.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
"""


@add_start_docstrings(
    "The bare Molmo2 model outputting raw hidden-states without any specific head on top.",
    MOLMO_START_DOCSTRING,
)
class Molmo2Model(Molmo2PreTrainedModel):
    _checkpoint_conversion_mapping = {}

    def __init__(self, config: Molmo2Config):
        super().__init__(config)
        self.transformer: Molmo2TextModel = Molmo2TextModel(config.text_config)
        self.vision_backbone: Optional[Molmo2VisionBackbone] = None
        if config.vit_config is not None and config.adapter_config is not None:
            self.vision_backbone = Molmo2VisionBackbone(config.vit_config, config.adapter_config)
        
        if config.bi_directional_attn:
            self.image_tokens = torch.as_tensor(
                [
                    config.image_patch_id,
                    config.image_col_id,
                    config.image_start_token_id,
                    config.image_end_token_id,
                    config.image_low_res_id,
                ],
                dtype=torch.long,
            )
        
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> torch.nn.Module:
        return self.transformer.wte

    def set_input_embeddings(self, value: torch.nn.Module) -> None:
        self.transformer.wte = value

    @property
    def device(self) -> torch.device:
        return self.transformer.ln_f.weight.device
    
    def build_batched_images(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.Tensor,
        image_token_pooling: torch.Tensor,
        image_grids: torch.Tensor,
        image_num_crops: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1) Count the number of images in each example
        raw_counts = (input_ids == self.config.image_end_token_id).sum(1)  # [N]
        # Each image is represented by global view and high-res view
        # so we divide by 2 to get the number of images
        counts = raw_counts // 2
        N = counts.size(0)
        device = input_ids.device

        # Total number of images in the batch
        num_images = int(counts.sum().item())

        # Sanity check
        assert image_grids.size(0) == num_images, \
            f"Expected {num_images} image grids, but got {image_grids.size(0)}"
        assert image_num_crops.size(0) == num_images, \
            f"Expected {num_images} image num crops, but got {image_num_crops.size(0)}"

        # 1-1) Compute per-image pooled patch count from image grids
        with torch.no_grad():
            first_prod = image_grids[:, :2].prod(dim=1)    # [num_images]
            second_prod = image_grids[:, 2:].prod(dim=1)   # [num_images]
            num_pooled_patches_per_image = (first_prod + second_prod).to(image_num_crops.dtype)  # [num_images]
        
        # pixel_values: [n_crops, n_patches, pixels_per_patch]
        n_crops, n_patches, pixels_per_patch = pixel_values.shape
        
        # 2) Map each image index → example index
        # Example: if counts = [2, 1, 3], then this becomes [0,0,1,2,2,2]
        example_ids_for_image = torch.arange(N, device=device).repeat_interleave(counts)  # [num_images]
        assert example_ids_for_image.numel() == num_images

        # 2-1) Compute crops_per_example by summing per-image crop counts
        crops_per_example = torch.zeros(
            N, dtype=image_num_crops.dtype, device=image_num_crops.device
        )
        crops_per_example.index_add_(0, example_ids_for_image, image_num_crops)  # [N]

        # 2-2) Per-image number of patches = (crops per image) * n_patches
        patches_per_image = image_num_crops * n_patches  # [num_images]

        # 2-3) Compute per-example per-image patch offsets
        counts_list = counts.tolist()
        index_offset_per_example_list = []
        offset_img = 0
        for c in counts_list:
            per_img_patches = patches_per_image[offset_img:offset_img + c]  # [c]
            # Offsets: [0, img0_total_patches, img0+img1_total_patches, ...]
            index_offset = [0] + per_img_patches.cumsum(0).tolist()[:-1]
            index_offset_per_example_list.append(index_offset)
            offset_img += c
        
        # 2-4) Compute num_pooled_patches_per_example
        num_pooled_patches_per_example = torch.zeros(
            N, dtype=num_pooled_patches_per_image.dtype, device=num_pooled_patches_per_image.device
        )
        num_pooled_patches_per_example.index_add_(
            0, example_ids_for_image, num_pooled_patches_per_image
        )

        # Sanity checks
        total_crops = int(crops_per_example.sum().item())
        assert total_crops == n_crops, \
            f"Expected {total_crops} crops, but got {n_crops}"

        total_num_pooled_patches = int(num_pooled_patches_per_example.sum().item())
        assert total_num_pooled_patches == image_token_pooling.size(0), \
            f"Expected {total_num_pooled_patches} pooled patches, but got {image_token_pooling.size(0)}"

        # 3) Build images tensor filled with -1
        M = int(crops_per_example.max().item())
        images = torch.full(
            (N, M, n_patches, pixels_per_patch),
            fill_value=-1,
            dtype=pixel_values.dtype,
            device=pixel_values.device,
        )

        # 4) Fill images with per-example slices from pixel_values
        offset_crop = 0
        for i in range(N):
            num = int(crops_per_example[i].item())
            cur = pixel_values[offset_crop:offset_crop + num]  # [num, n_patches, pixels_per_patch]
            images[i, :num] = cur
            offset_crop += num

        # Sanity check
        assert offset_crop == n_crops

        # 5) Build new_token_pooling tensor filled with -1
        P = int(num_pooled_patches_per_example.max().item())
        _, dim = image_token_pooling.shape
        new_token_pooling = torch.full(
            (N, P, dim),
            fill_value=-1,
            dtype=image_token_pooling.dtype,
            device=image_token_pooling.device,
        )

        # 6) Fill token_pooling with per-example slices, adding per-image patch offsets
        patch_offset = 0
        img_offset = 0

        for i, c in enumerate(counts_list):
            num_patches = int(num_pooled_patches_per_example[i].item())

            # Subsequence of pooled tokens belonging to this example
            cur = image_token_pooling[patch_offset:patch_offset + num_patches].clone()  # [num_patches, dim]

            index_offset_per_example = index_offset_per_example_list[i]  # length = c
            per_img_pooled = num_pooled_patches_per_image[img_offset:img_offset + c]   # [c]

            assert len(index_offset_per_example) == per_img_pooled.numel()

            # Apply per-image offsets to the (ragged) subsequence
            offset = 0
            for j in range(c):
                index_offset = int(index_offset_per_example[j])
                n = int(per_img_pooled[j].item())
                cur_slice = cur[offset:offset + n]

                # Apply offset across all columns
                cur[offset:offset + n] = torch.where(
                    cur_slice >= 0,
                    cur_slice + index_offset,
                    cur_slice,
                )
                offset += n

            new_token_pooling[i, :num_patches] = cur

            patch_offset += num_patches
            img_offset += c

        # Final sanity checks
        assert patch_offset == total_num_pooled_patches
        assert img_offset == num_images

        return images, new_token_pooling
    
    def build_batched_videos(
        self,
        input_ids: torch.LongTensor,
        pixel_values_videos: torch.Tensor,
        video_token_pooling: torch.Tensor,
        video_grids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # 1) Counter the number of videos in each example
        if self.config.use_frame_special_tokens:
            end_token_id = self.config.frame_end_token_id
        else:
            end_token_id = self.config.image_end_token_id
        counts = (input_ids == end_token_id).any(dim=1).long()  # [N]
        N = counts.size(0)
        device = input_ids.device

        # Total number of videos in the batch
        num_videos = int(counts.sum().item())

        # Sanity check
        assert video_grids.size(0) == num_videos, \
            f"Expected {num_videos} videos, but got {video_grids.size(0)}"
        
        video_num_frames = video_grids[:, 0]  # [num_videos]
        num_pooled_patches_per_video = video_grids.prod(dim=1)  # [num_videos]

        # pixel_values_videos: [n_frames, n_patches, pixels_per_patch]
        n_frames, n_patches, pixels_per_patch = pixel_values_videos.shape

        # 2) Map each video index -> example index
        # Example: if counts = [2, 1, 3], then this becomes [0,0,1,2,2,2]
        example_ids_for_video = torch.arange(N, device=device).repeat_interleave(counts)  # [num_videos]
        assert example_ids_for_video.numel() == num_videos

        # 2-1) Compute frames_per_example by summing per-video frame counts
        frames_per_example = torch.zeros(
            N, dtype=video_num_frames.dtype, device=device,
        )
        frames_per_example.index_add_(0, example_ids_for_video, video_num_frames)  # [N]

        # 2-2) Compute num_pooled_patches_per_example
        num_pooled_patches_per_example = torch.zeros(
            N, dtype=num_pooled_patches_per_video.dtype, device=num_pooled_patches_per_video.device,
        )
        num_pooled_patches_per_example.index_add_(
            0, example_ids_for_video, num_pooled_patches_per_video,
        )

        # Sanity checks
        total_frames = int(frames_per_example.sum().item())
        assert total_frames == n_frames, \
            f"Expected {total_frames} frames, but got {n_frames}"
        
        total_num_pooled_patches = int(num_pooled_patches_per_example.sum().item())
        assert total_num_pooled_patches == video_token_pooling.size(0), \
            f"Expected {total_num_pooled_patches} pooled patches, but got {video_token_pooling.size(0)}"
        
        # 3) Build videos tensor filled with -1
        M = int(frames_per_example.max().item())
        videos = torch.full(
            (N, M, n_patches, pixels_per_patch),
            fill_value=-1,
            dtype=pixel_values_videos.dtype,
            device=device,
        )

        # 4) Fill videos with per-examples slices from pixel_values_videos
        offset_frame = 0
        for i in range(N):
            num = int(frames_per_example[i].item())
            cur = pixel_values_videos[offset_frame:offset_frame + num]  # [num, n_patches, pixels_per_patch]
            videos[i, :num] = cur
            offset_frame += num
        
        # Sanity check
        assert offset_frame == n_frames

        # 5) Build new token_pooling tensor filled with -1
        P = int(num_pooled_patches_per_example.max().item())
        _, dim = video_token_pooling.shape
        new_token_pooling = torch.full(
            (N, P, dim),
            fill_value=-1,
            dtype=video_token_pooling.dtype,
            device=video_token_pooling.device,
        )

        # 6) Fill new token_pooling with per-examples slices from video_token_pooling
        patch_offset = 0
        for i in range(N):
            num_patches = int(num_pooled_patches_per_example[i].item())
            cur = video_token_pooling[patch_offset:patch_offset + num_patches]  # [num_patches, dim]
            new_token_pooling[i, :num_patches] = cur
            patch_offset += num_patches

        # Final sanity checks
        assert patch_offset == total_num_pooled_patches

        return videos, new_token_pooling
    
    def merge_visual_inputs(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_token_pooling: Optional[torch.Tensor] = None,
        image_grids: Optional[torch.Tensor] = None,
        image_num_crops: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        video_token_pooling: Optional[torch.Tensor] = None,
        video_grids: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if pixel_values is not None and pixel_values_videos is not None:
            raise ValueError("pixel_values and pixel_values_videos are provided at the same time")
        elif pixel_values is not None:
            assert input_ids is not None
            images, token_pooling = self.build_batched_images(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_token_pooling=image_token_pooling,
                image_grids=image_grids,
                image_num_crops=image_num_crops,
            )
        elif pixel_values_videos is not None:
            assert input_ids is not None
            images, token_pooling = self.build_batched_videos(
                input_ids=input_ids,
                pixel_values_videos=pixel_values_videos,
                video_token_pooling=video_token_pooling,
                video_grids=video_grids,
            )
        else:
            images, token_pooling = None, None
        return images, token_pooling

    def build_input_embeddings(
        self,
        input_ids: torch.LongTensor,
        images: Optional[torch.FloatTensor] = None,  # image inputs
        token_pooling: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        # Get embeddings of input.
        # shape: (batch_size, seq_len, d_model)
        input_ids = input_ids * (input_ids != -1).to(input_ids.dtype)
        x = self.transformer.wte(input_ids)

        image_features: Optional[torch.FloatTensor] = None    
        if images is not None:
            image_features = self.vision_backbone(images, token_pooling).to(x.device)
            is_image_patch = input_ids.view(-1) == self.config.image_patch_id
            assert is_image_patch.sum() == len(image_features)
            x.view(-1, x.shape[-1])[is_image_patch] += image_features

        # shape: (batch_size, seq_len, d_model)
        x = self.transformer.emb_drop(x)  # type: ignore

        return x, image_features

    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_token_pooling: Optional[torch.Tensor] = None,
        image_grids: Optional[torch.Tensor] = None,
        image_num_crops: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        video_token_pooling: Optional[torch.Tensor] = None,
        video_grids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, Molmo2ModelOutputWithPast]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        
        images, token_pooling = self.merge_visual_inputs(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_token_pooling=image_token_pooling,
            image_grids=image_grids,
            image_num_crops=image_num_crops,
            pixel_values_videos=pixel_values_videos,
            video_token_pooling=video_token_pooling,
            video_grids=video_grids,
        )

        if images is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both images and inputs_embeds at the same time."
            )

        if inputs_embeds is None:
            inputs_embeds, image_features = self.build_input_embeddings(
                input_ids, images, token_pooling)
        
        image_attention_mask: Optional[torch.Tensor] = None
        if images is not None and self.config.bi_directional_attn:
            if self.config.bi_directional_attn  == "image_tokens":
                image_tokens = self.image_tokens.to(inputs_embeds.device)
                image_attention_mask = torch.any(input_ids[:, :, None] == image_tokens[None, None, :], -1)
            else:
                raise NotImplementedError(self.config.bi_directional_attn)
        
        outputs = self.transformer(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            image_attention_mask=image_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
        )

        return Molmo2ModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features if images is not None else None,
        )

@add_start_docstrings(
    "The Molmo2 model which consists of a vision backbone and a language model + lm head.",
    MOLMO_START_DOCSTRING,
)
class Molmo2ForConditionalGeneration(Molmo2PreTrainedModel, GenerationMixin):
    _checkpoint_conversion_mapping = {}
    _tied_weights_keys = []  # Weights are not tied
    config_class = Molmo2Config

    def __init__(self, config: Molmo2Config):
        super().__init__(config)

        self.model = Molmo2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.vocab_size = config.vocab_size
        
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> torch.nn.Module:
        return self.model.transformer.wte

    def set_input_embeddings(self, value: torch.nn.Module) -> None:
        self.model.transformer.wte = value
    
    def get_output_embeddings(self):
        self.lm_head

    def set_output_embeddings(self, value: torch.nn.Module) -> None:
        self.lm_head = value
    
    # Make modules available throught conditional class for BC
    @property
    def language_model(self) -> torch.nn.Module:
        return self.model.transformer

    @property
    def vision_backbone(self) -> torch.nn.Module:
        return self.model.vision_backbone

    @can_return_tuple
    @add_start_docstrings_to_model_forward(MOLMO2_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_token_pooling: Optional[torch.Tensor] = None,
        image_grids: Optional[torch.Tensor] = None,
        image_num_crops: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        video_token_pooling: Optional[torch.Tensor] = None,
        video_grids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> Union[Tuple, Molmo2CausalLMOutputWithPast]:
        r"""
        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Molmo2ForConditionalGeneration

        >>> model = Molmo2ForConditionalGeneration.from_pretrained("...")
        >>> processor = AutoProcessor.from_pretrained("...")

        >>> prompt = "What's the content of the image?"
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> messages = [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image", "image": image}]}]

        >>> text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        >>> inputs = processor(text=text, images=image, return_tensors="pt")

        >>> # Generate
        >>> generated_ids = model.generate(**inputs, max_new_tokens=15)
        >>> generated_tokens = generated_ids[:, inputs['input_ids'].size(1):]
        >>> processor.post_process_image_text_to_text(generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "The image features a busy city street with a stop sign prominently displayed"
        ```"""
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_token_pooling=image_token_pooling,
            image_grids=image_grids,
            image_num_crops=image_num_crops,
            pixel_values_videos=pixel_values_videos,
            video_token_pooling=video_token_pooling,
            video_grids=video_grids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.vocab_size)

        return Molmo2CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_token_pooling: Optional[torch.Tensor] = None,
        image_grids: Optional[torch.Tensor] = None,
        image_num_crops: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        video_token_pooling: Optional[torch.Tensor] = None,
        video_grids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Optional[Union[int, torch.Tensor]] = None,
        **kwargs,
    ):

        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

        if cache_position[0] == 0:
            model_inputs["pixel_values"] = pixel_values
            model_inputs["image_token_pooling"] = image_token_pooling
            model_inputs["image_grids"] = image_grids
            model_inputs["image_num_crops"] = image_num_crops
            model_inputs["pixel_values_videos"] = pixel_values_videos
            model_inputs["video_token_pooling"] = video_token_pooling
            model_inputs["video_grids"] = video_grids

        return model_inputs

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ) -> Dict[str, Any]:
        if model_kwargs["use_cache"] and "pixel_values" in model_kwargs:
            # After the first step, no long pass the images into forward since the images tokens
            # are already cached
            for k in ["pixel_values", "image_token_pooling", "image_grids", "image_num_crops"]:
                del model_kwargs[k]
        if model_kwargs["use_cache"] and "pixel_values_videos" in model_kwargs:
            for k in ["pixel_values_videos", "video_token_pooling", "video_grids"]:
                del model_kwargs[k]
        return super()._update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder, num_new_tokens)

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        cache_position: torch.Tensor,
        batch_size: int,
        **kwargs,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape
                `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache,
                to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=cache_position.device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=cache_position.device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                    causal_mask.device
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )

        return causal_mask


# Always register for multi-modal features
AutoModelForImageTextToText.register(Molmo2Config, Molmo2ForConditionalGeneration)
AutoModelForCausalLM.register(Molmo2TextConfig, Molmo2ForCausalLM)