import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, fields
from functools import cached_property, partial
from itertools import islice
from typing import List, Optional, Set, Tuple, TypedDict, Union, Dict, Any, Annotated
from typing_extensions import TypeAlias, assert_never
from PIL.Image import Image

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BatchFeature, ProcessorMixin, TensorType, PreTrainedTokenizerBase
from transformers.image_utils import ImageInput
from transformers.tokenization_utils_base import TextInput

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.attention import Attention
from vllm.attention.layer import MultiHeadAttention
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    split_tensor_along_last_dim,
    tensor_model_parallel_all_gather,
)
from vllm.model_executor import SamplingMetadata
from vllm.model_executor.layers.activation import MulAndSilu, SiluAndMul, get_act_fn
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalKwargs,
    HfImageItem,
)
from vllm.multimodal.parse import (
    ImageProcessorItems,
    ProcessorBatchItems,
    ImageSize,
    ModalityDataItems,
    MultiModalDataItems,
    MultiModalDataParser,
)
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsLoRA,
    SupportsMultiModal,
    SupportsPP,
    SupportsQuant,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    flatten_bn,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
    merge_multimodal_embeddings,
)

logger = init_logger(__name__)


# Special tokens, these should be present in any tokenizer we use since the preprocessor uses them
IMAGE_PATCH_TOKEN = f"<im_patch>"  # Where to insert high-res tokens
IMAGE_LOW_RES_TOKEN = f"<im_low>"  # Where to insert low-res tokens
IM_START_TOKEN = f"<im_start>"
LOW_RES_IMAGE_START_TOKEN = f"<low_res_im_start>"
FRAME_START_TOKEN = f"<frame_start>"
IM_END_TOKEN = f"<im_end>"
FRAME_END_TOKEN= f"<frame_end>"
IM_COL_TOKEN = f"<im_col>"
IMAGE_PROMPT = "<|image|>"
VIDEO_PROMPT = "<|video|>"
_MAX_FRAMES_PER_VIDEO = 128


class Molmo2ImageInputs(TensorSchema):
    """
    Dimensions:
        - nc: The total number of crops (dynamic)
        - np: The total number of patches per crop
        - cps: Number of channels * patch_size * patch_size
        - npp: Number of pooled patches (dynamic)
        - pp: pooling_size * pooling_size
        - ni: Number of images
    """

    pixel_values: Annotated[torch.Tensor, TensorShape("nc", "np", "cps")]

    token_pooling: Annotated[torch.Tensor, TensorShape("npp", "pp")]
    """An index tensor that maps image features to their corresponding patch tokens before pooling."""

    num_pooled_patches: Annotated[torch.Tensor, TensorShape("ni")]


class Molmo2VideoInputs(TensorSchema):
    """
    Dimensions:
        - nc: The total number of frames (dynamic)
        - np: The total number of patches per frame
        - cps: Number of channels * patch_size * patch_size
        - npp: Number of pooled patches (dynamic)
        - pp: pooling_size * pooling_size
        - nv: Number of videos
    """

    pixel_values_videos: Annotated[torch.Tensor, TensorShape("nc", "np", "cps")]

    token_pooling: Annotated[torch.Tensor, TensorShape("npp", "pp")]
    """An index tensor that maps image features to their corresponding patch tokens before pooling."""

    num_pooled_patches: Annotated[torch.Tensor, TensorShape("nv")]


@dataclass
class VitConfig:
    """Config for a vision transformer"""

    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    head_dim: int = 72
    hidden_act: str = "gelu_pytorch_tanh"
    layer_norm_eps: float = 1e-6
    image_default_input_size: tuple[int, int] = (378, 378)
    image_patch_size: int = 14
    image_num_pos: int = 577

    def __post_init__(self):
        self.image_default_input_size = tuple(self.image_default_input_size)  # type: ignore[assignment]

    @property
    def image_num_patch(self):
        h, w = self.image_default_input_size
        return h // self.image_patch_size, w // self.image_patch_size


@dataclass
class AdapterConfig:
    """Config for a vit-llm adapter"""

    vit_layers: tuple[int, int] = (-3, -9)
    pooling_attention_mask: bool = False
    hidden_size: int = 1152
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    head_dim: int = 72
    hidden_act: str = "silu"
    intermediate_size: int = 18944
    text_hidden_size: int = 3584


@dataclass
class TextConfig:
    """Configuration for a text model transformer"""

    hidden_size: int = 3584
    """
    The hidden size of the model.
    """

    num_attention_heads: int = 28
    """
    The number of self-attention heads.
    """

    num_key_value_heads: int = 4
    """
    The number of heads to use for keys and values.
    """

    head_dim: int = 128
    """
    The head dimensionality for the attention mechanism.
    """

    vocab_size: int = 152064
    """Vocabulary size of the model."""

    additional_vocab_size: int = 128
    """Number of additional tokens to have the input embeddings for"""

    qkv_bias: bool = True
    """
    Do QKV projection a bias
    """

    num_hidden_layers: int = 48
    """
    The number of layers/blocks.
    """

    intermediate_size: int = 18944
    """
    The hidden size for the MLP.
    """

    hidden_act: str = "silu"
    """
    The activation function to use within the MLP layers.
    """

    max_position_embeddings: int = 4096
    """
    Max positional embeddings to use in RoPE cache
    """

    rope_theta: float = 1000000.0
    """
    RoPE theta parameter.
    """

    use_qk_norm: bool = False
    """
    Apply layer norm to the keys and queries within the attention mechanism.
    This can help stabilize training.
    """

    qk_norm_type: str = "olmo"
    """
    The type of layer norm to use for the keys and queries.
    Can be "olmo" or "qwen3".
    """

    layer_norm_eps: float = 1e-6
    """
    epsilon for layer norms
    """

    norm_after: bool = False
    """Apply layer norm before and after the attention and MLP blocks."""


class ViTMLP(nn.Module):
    """MLP used in Vision Transformer."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__()
        self.w1 = ColumnParallelLinear(
            dim,
            hidden_dim,
            bias=True,
            quant_config=quant_config,
        )
        # Activation function.
        self.act = get_act_fn(hidden_act)
        self.w2 = RowParallelLinear(
            hidden_dim,
            dim,
            bias=True,
            quant_config=quant_config,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.w1(x)
        x = self.act(x)
        x, _ = self.w2(x)
        return x


class ViTMultiHeadDotProductAttention(nn.Module):
    """Multi-head attention used in Vision Transformer."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        use_bias: bool = True,
        input_dim: int | None = None,
        use_pytorch_sdpa: bool = False,
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.total_num_heads = num_heads
        tp_size = get_tensor_model_parallel_world_size()

        assert self.hidden_size % self.total_num_heads == 0
        assert self.total_num_heads % tp_size == 0

        self.num_heads = self.total_num_heads // tp_size
        self.head_dim = head_dim

        assert self.head_dim == self.hidden_size // self.total_num_heads
        
        self.total_num_kv_heads = num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0

        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

        if input_dim is None:
            input_dim = self.hidden_size

        self.wq = ColumnParallelLinear(
            input_dim,
            self.total_num_heads * self.head_dim,
            bias=use_bias,
            quant_config=quant_config,
        )
        self.wk = ColumnParallelLinear(
            input_dim,
            self.total_num_kv_heads * self.head_dim,
            bias=use_bias,
            quant_config=quant_config,
        )
        self.wv = ColumnParallelLinear(
            input_dim,
            self.total_num_kv_heads * self.head_dim,
            bias=use_bias,
            quant_config=quant_config,
        )
        self.wo = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=use_bias,
            quant_config=quant_config,
        )
        self.scale = self.head_dim**-0.5
        self.use_pytorch_sdpa = use_pytorch_sdpa
        if use_pytorch_sdpa:
            self.attn = None
        else:
            self.attn = MultiHeadAttention(
                self.num_heads,
                self.head_dim,
                self.scale,
                num_kv_heads=self.num_kv_heads
            )
    
    def forward_sdpa(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, q_len, _ = query.size()
        kv_len = key.size(1)

        query = query.view(bsz, q_len, self.num_heads, self.head_dim)
        key = key.view(bsz, kv_len, self.num_kv_heads, self.head_dim)
        value = value.view(bsz, kv_len, self.num_kv_heads, self.head_dim)

        if self.num_heads != self.num_kv_heads:
            key = torch.repeat_interleave(
                key, self.num_heads // self.num_kv_heads, dim=2,
            )
            value = torch.repeat_interleave(
                value, self.num_heads // self.num_kv_heads, dim=2,
            )
        
        query, key, value = (
            x.transpose(1, 2)
            for x in (query, key, value)
        )

        out = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            is_causal=False,
        ).transpose(1, 2)

        return out.reshape(bsz, q_len, -1)

    def forward(
        self,
        inputs_q: torch.Tensor,
        inputs_kv: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:

        if inputs_kv is not None:
            inputs_k = inputs_kv
            inputs_v = inputs_kv
        else:
            inputs_k = inputs_q
            inputs_v = inputs_q

        xq, _ = self.wq(inputs_q)
        xk, _ = self.wk(inputs_k)
        xv, _ = self.wv(inputs_v)

        if self.use_pytorch_sdpa:
            output = self.forward_sdpa(xq, xk, xv, attn_mask)
        else:
            output = self.attn(xq, xk, xv)

        output, _ = self.wo(output)

        return output


class Molmo2VisionBlock(nn.Module):
    """Residual attention block used in Vision Transformer."""

    def __init__(
        self,
        config: VitConfig,
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__()
        self.attention = ViTMultiHeadDotProductAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            quant_config=quant_config,
        )
        self.feed_forward = ViTMLP(
            dim=config.hidden_size,
            hidden_dim=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
        )
        self.attention_norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
        )
        self.ffn_norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x))
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class Molmo2VisionBlockCollection(nn.Module):
    """Collection of residual attention blocks used in Vision Transformer."""

    def __init__(
        self,
        config: VitConfig,
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__()
        self.resblocks = nn.ModuleList(
            [
                Molmo2VisionBlock(config, quant_config)
                for _ in range(config.num_hidden_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        hidden_states = []
        for r in self.resblocks:
            x = r(x)
            hidden_states.append(x)
        return hidden_states


class Molmo2VisionTransformer(nn.Module):
    """Vision Transformer used in Vision Backbone."""

    def __init__(
        self,
        config: VitConfig,
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__()
        scale = config.hidden_size ** -0.5
        self.num_prefix_tokens: int = 0  # no class embeddings
        self.patch_num = config.image_num_patch
        self.positional_embedding = nn.Parameter(
            torch.randn(config.image_num_pos, config.hidden_size) * scale,
        )
        image_patch_size = config.image_patch_size
        self.patch_embedding = nn.Linear(
            image_patch_size * image_patch_size * 3,
            config.hidden_size,
            bias=True,
        )
        self.transformer = Molmo2VisionBlockCollection(config, quant_config)

    def add_pos_emb(self, x: torch.Tensor, patch_num: int) -> torch.Tensor:
        pos_emb = self.positional_embedding

        pos_emb = pos_emb.reshape(
            (
                int(math.sqrt(pos_emb.shape[0])),
                int(math.sqrt(pos_emb.shape[0])),
                pos_emb.shape[1]
            )
        )

        (patch_num_0, patch_num_1) = patch_num

        if pos_emb.shape[0] != patch_num_0 or pos_emb.shape[1] != patch_num_1:
            # from https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
            pos_emb = pos_emb.unsqueeze(0).permute(0, 3, 1, 2)
            pos_emb = F.interpolate(
                pos_emb,
                size=(patch_num_0, patch_num_1),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
            pos_emb = pos_emb.permute(0, 2, 3, 1).squeeze(0)
        
        pos_emb = pos_emb.reshape(-1, pos_emb.shape[-1])
        x = x + pos_emb[None, :, :].to(x.dtype)
        return x
    
    def forward(
        self,
        x: torch.Tensor,
        patch_num: int | None = None,
    ) -> list[torch.Tensor]:
        """
        : param x: (batch_size, num_patch, n_pixels)
        """
        if patch_num is None:
            patch_num = self.patch_num
        
        x = self.patch_embedding(x)

        x = self.add_pos_emb(x, patch_num)

        hidden_states = self.transformer(x)
        return hidden_states


class ImageProjectorMLP(nn.Module):
    """MLP used for the image projector"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__()

        self.merged_linear = MergedColumnParallelLinear(
            input_dim,
            [hidden_dim] * 2,
            bias=False,
            quant_config=quant_config,
        )
        # Activation function.
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

        # Feed-forward output projection.
        self.down_proj = RowParallelLinear(
            hidden_dim,
            output_dim,
            bias=False,
            quant_config=quant_config,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.merged_linear(x)
        x = self.act_fn(x)
        x, _ = self.down_proj(x)
        return x


class Molmo2VisionBackbone(nn.Module, SupportsQuant):
    packed_modules_mapping = {"merged_linear": ["gate_proj", "up_proj"]}

    def __init__(
        self,
        vit_config: VitConfig,
        adapter_config: AdapterConfig,
        quant_config: QuantizationConfig | None = None,
    ) -> None:
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
            vit_config.num_hidden_layers = last_layer_needed
        self.image_vit = Molmo2VisionTransformer(vit_config, quant_config)

        self.num_prefix_tokens: int = self.image_vit.num_prefix_tokens

        pool_dim = vit_config.hidden_size * len(adapter_config.vit_layers)
        self.image_pooling_2d = ViTMultiHeadDotProductAttention(
            hidden_size=adapter_config.hidden_size,
            num_heads=adapter_config.num_attention_heads,
            num_key_value_heads=adapter_config.num_key_value_heads,
            head_dim=adapter_config.head_dim,
            input_dim=pool_dim,
            use_pytorch_sdpa=adapter_config.pooling_attention_mask,
            quant_config=quant_config,
        )
        self.image_projector = ImageProjectorMLP(
            input_dim=adapter_config.hidden_size,
            hidden_dim=adapter_config.intermediate_size,
            output_dim=adapter_config.text_hidden_size,
            hidden_act=adapter_config.hidden_act,
            quant_config=quant_config,
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.image_vit.patch_embedding.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.image_vit.patch_embedding.weight.device
    
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

    def forward(
        self,
        images: torch.Tensor,
        token_pooling: torch.Tensor,
    ) -> torch.Tensor:

        # image_features: (batch_size, num_crops(=num_image), num_patch, nximage_emb_dim)
        batch_size, num_image = images.shape[:2]
        images = images.to(device=self.device, dtype=self.dtype)
        image_features = self.encode_image(images)

        dim = image_features.shape[-1]
        valid = token_pooling >= 0
        valid_token = torch.any(valid, -1)

        # Use `token_pooling` to arange the features for image pooling
        batch_idx = torch.arange(token_pooling.shape[0], dtype=torch.long, device=token_pooling.device)
        batch_idx = torch.tile(batch_idx.view(batch_size, 1, 1), [1, token_pooling.shape[1], token_pooling.shape[2]])

        # Now [batch, num_features, num_pooled_patches, dim]
        to_pool = image_features.reshape(batch_size, -1, dim)[batch_idx, torch.clip(token_pooling, 0)]
        to_pool = to_pool * valid.to(self.dtype)[:, :, :, None]
        to_pool = to_pool.reshape([-1, token_pooling.shape[-1], dim])
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

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("merged_linear", "gate_proj", 0),
            ("merged_linear", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class Molmo2Attention(nn.Module):
    """Molmo2's LLM Attention."""

    def __init__(
        self,
        config: TextConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads

        assert self.hidden_size % self.total_num_heads == 0
        assert self.total_num_heads % self.tp_size == 0

        self.num_heads = self.total_num_heads // self.tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= self.tp_size:
            assert self.total_num_kv_heads % self.tp_size == 0
        else:
            assert self.tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // self.tp_size)
        self.head_dim = config.head_dim

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        # Attention input projection. Projects x -> (q, k, v)
        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=config.qkv_bias,
            quant_config=quant_config,
        )

        self.tp_rank: int | None = None
        self.k_norm: nn.Module | None = None
        self.q_norm: nn.Module | None = None
        self.qk_norm_type: str | None = None
        if config.use_qk_norm:
            k_norm_size = (
                self.head_dim
                if config.qk_norm_type == "qwen3" else
                self.total_num_kv_heads * self.head_dim
            )
            self.tp_rank = get_tensor_model_parallel_rank()
            self.k_norm = RMSNorm(k_norm_size, eps=config.layer_norm_eps)
            q_norm_size = (
                self.head_dim
                if config.qk_norm_type == "qwen3" else
                self.total_num_heads * self.head_dim
            )
            self.q_norm = RMSNorm(q_norm_size, eps=config.layer_norm_eps)
            self.qk_norm_type = config.qk_norm_type
        # Rotary embeddings.
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=self.max_position_embeddings,
            base=self.rope_theta,
        )
        self.scaling = self.head_dim**-0.5
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

        # Attention output projection.
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
        )

    def _apply_qk_norm(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.tp_size > 1:
            q = tensor_model_parallel_all_gather(q.contiguous())
            k = tensor_model_parallel_all_gather(k.contiguous())
        q = self.q_norm(q)
        k = self.k_norm(k)
        if self.tp_size > 1:
            splitter = partial(split_tensor_along_last_dim, num_partitions=self.tp_size)
            q = splitter(q)[self.tp_rank]
            k = splitter(k)[self.tp_rank]
        return q, k

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        if self.q_norm is not None and self.k_norm is not None and self.qk_norm_type == "olmo":
            q, k = self._apply_qk_norm(q, k)
        elif self.q_norm is not None and self.k_norm is not None:
            q_by_head = q.view(
                *q.shape[:-1],
                q.shape[-1] // self.head_dim,
                self.head_dim,
            )
            q_by_head = self.q_norm(q_by_head)
            q = q_by_head.view(q.shape)
            k_by_head = k.view(
                *k.shape[:-1],
                k.shape[-1] // self.head_dim,
                self.head_dim,
            )
            k_by_head = self.k_norm(k_by_head)
            k = k_by_head.view(k.shape)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)

        if not kwargs.get("has_images", False):
            # Fast path for text-only inputs. The performance for the text-only
            # inputs are not affected by the naive attention below.
            output, _ = self.o_proj(attn_output)
            return output
        
        attn_output = self.naive_attn_with_masks(
            q,
            k,
            v,
            out=attn_output,
            **kwargs,
        )
        output, _ = self.o_proj(attn_output)
        return output
    
    def naive_attn_with_masks(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        q = q.view(-1, self.num_heads, self.head_dim)
        # Expand the key and value to handle GQA.
        num_queries_per_kv = self.num_heads // self.num_kv_heads
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        k = k.repeat_interleave(num_queries_per_kv, dim=-2)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        v = v.repeat_interleave(num_queries_per_kv, dim=-2)

        attn_masks = kwargs["attn_masks"]

        seq_lens = kwargs["seq_lens"]
        start_idx = 0
        for seq_len, attn_mask in zip(seq_lens, attn_masks):
            end_idx = start_idx + seq_len
            query = q[start_idx:end_idx].unsqueeze(0)
            key = k[start_idx:end_idx].unsqueeze(0)
            value = v[start_idx:end_idx].unsqueeze(0)

            # Transpose.
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)

            output = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask,
                self.scaling,
            )
            output = output.transpose(1, 2).flatten(-2, -1)
            out[start_idx:end_idx] = output
            start_idx = end_idx
        return out


class LanguageModelMLP(nn.Module):
    """Molmo2's LLM mlp."""

    def __init__(
        self,
        input_dim: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__()

        self.up_gate_proj = MergedColumnParallelLinear(
            input_dim,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
        )
        # Activation function.
        assert hidden_act == "silu"
        self.act_fn = MulAndSilu()
        # Feed-forward output projection.
        self.down_proj = RowParallelLinear(
            intermediate_size,
            input_dim,
            bias=False,
            quant_config=quant_config,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        up_gate, _ = self.up_gate_proj(x)
        x = self.act_fn(up_gate)
        x, _ = self.down_proj(x)
        return x


class Molmo2DecoderLayer(nn.Module):

    def __init__(
        self,
        config: TextConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        # Attention block.
        self.self_attn = Molmo2Attention(
            config,
            cache_config,
            quant_config,
            prefix=f"{prefix}.self_attn",
        )

        # MLP block.
        self.mlp = LanguageModelMLP(
            config.hidden_size,
            config.intermediate_size,
            config.hidden_act,
            quant_config,
        )

        # LayerNorm
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        **kwargs: object,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            **kwargs,
        )

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Molmo2DecoderNormAfterLayer(Molmo2DecoderLayer):

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        **kwargs: object,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        # Self Attention
        residual = hidden_states
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            **kwargs,
        )

        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = hidden_states + residual
        residual = hidden_states

        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = hidden_states + residual
        residual = None
        return hidden_states, residual


@support_torch_compile
class Molmo2TextModel(nn.Module, SupportsQuant):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.config = config

        if hasattr(config, "text_config"):
            hf_text_config = config.text_config
        else:
            hf_text_config = config.llm_config

        kwargs = {}
        for field in fields(TextConfig):
            kwargs[field.name] = getattr(hf_text_config, field.name)
        text_config = TextConfig(**kwargs)

        self.embedding_size = text_config.vocab_size
        self.embedding_size += text_config.additional_vocab_size or 0
        self.embed_tokens = VocabParallelEmbedding(
            self.embedding_size,
            text_config.hidden_size,
            quant_config=quant_config,
        )

        decoder_layer = (
            Molmo2DecoderNormAfterLayer if text_config.norm_after else Molmo2DecoderLayer
        )
        self.start_layer, self.end_layer, self.layers = make_layers(
            text_config.num_hidden_layers,
            lambda prefix: decoder_layer(
                text_config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
        )

        self.norm = RMSNorm(text_config.hidden_size, eps=text_config.layer_norm_eps)

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], text_config.hidden_size,
        )

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_tokens(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        # Apply blocks one-by-one.
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
                **kwargs,
            )
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })
        if residual is not None:
            hidden_states, _ = self.norm(hidden_states, residual)
        else:
            hidden_states = self.norm(hidden_states)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            if name.endswith(".bias") and name not in params_dict:
                continue
            if is_pp_missing_parameter(name, self):
                continue

            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


def _lowest_multiple(x: int, k: int) -> int:
    return (x // k) * k


def get_patches_grid_size(
    *,
    image_h: int,
    image_w: int,
    patch_size: int,
    pool_h: int,
    pool_w: int,
) -> tuple[int, int]:
    patch_h = image_h // patch_size
    patch_w = image_w // patch_size
    h_pad = _lowest_multiple(patch_h + pool_h - 1, pool_h) - patch_h
    w_pad = _lowest_multiple(patch_w + pool_w - 1, pool_w) - patch_w
    nrows = (patch_h + h_pad) // pool_h
    ncols = (patch_w + w_pad) // pool_w

    return nrows, ncols


def get_candidate_tilings(max_num: int) -> list[tuple[int, int]]:
    tilings = [
        (i, j)
        for i in range(1, max_num + 1)
        for j in range(1, max_num + 1)
        if i * j <= max_num
    ]
    return sorted(tilings, key=lambda x: (x[0] * x[1], x[0]))


def select_tiling(
    *,
    height: int,
    width: int,
    patch_size: int,
    max_num_patches: int,
):
    tilings = get_candidate_tilings(max_num_patches)
    candidate_tilings = np.array(tilings, dtype=np.int32)
    candidate_resolutions = candidate_tilings * patch_size

    original_size = np.array([height, width], dtype=np.float32)
    required_scale_d = candidate_resolutions.astype(np.float32) / original_size
    required_scale = required_scale_d.min(axis=-1, keepdims=True)

    if (required_scale < 1).all():
        ix = required_scale.argmax()
    else:
        ix = np.where(required_scale < 1.0, 10e9, required_scale).argmin()

    return candidate_tilings[ix]


def get_image_size(image: ImageInput) -> ImageSize:
    if isinstance(image, Image):
        return ImageSize(*image.size)
    elif isinstance(image, (np.ndarray, torch.Tensor)):
        assert image.ndim == 3
        h, w, c = image.shape
        assert c in [1, 3]
        return ImageSize(w, h)
    else:
        raise ValueError(f"Unknown image type: {type(image)}")


class Molmo2ProcessorWrapper:
    """
    Wraps :class:`Molmo2Processor` so that it can be called directly.
    """

    def __init__(self, processor: ProcessorMixin):
        super().__init__()

        self.processor = processor

    @cached_property
    def vocab(self) -> dict[str, int]:
        return self.processor.tokenizer.vocab  # type: ignore

    @cached_property
    def max_crops(self) -> int:
        image_processor = self.processor.image_processor  # type: ignore

        max_crops = image_processor.max_crops
        assert isinstance(max_crops, int)

        return max_crops

    @cached_property
    def image_pooling_h(self) -> int:
        image_processor = self.processor.image_processor  # type: ignore

        image_pooling_h = image_processor.pooling_size[0]
        assert isinstance(image_pooling_h, int)

        return image_pooling_h

    @cached_property
    def image_pooling_w(self) -> int:
        image_processor = self.processor.image_processor  # type: ignore

        image_pooling_w = image_processor.pooling_size[1]
        assert isinstance(image_pooling_w, int)

        return image_pooling_w

    @cached_property
    def video_pooling_h(self) -> int:
        video_processor = self.processor.video_processor  # type: ignore

        video_pooling_h = video_processor.pooling_size[0]
        assert isinstance(video_pooling_h, int)

        return video_pooling_h

    @cached_property
    def video_pooling_w(self) -> int:
        video_processor = self.processor.video_processor  # type: ignore

        video_pooling_w = video_processor.pooling_size[1]
        assert isinstance(video_pooling_w, int)

        return video_pooling_w

    @cached_property
    def base_image_input_size(self) -> tuple[int, int]:
        if getattr(self.processor, "image_processor", None) is not None:
            processor = self.processor.image_processor  # type: ignore
        else:
            processor = self.processor.video_processor  # type: ignore

        base_image_input_size = (processor.size["height"], processor.size["width"])

        return base_image_input_size

    @cached_property
    def image_patch_size(self) -> int:
        if getattr(self.processor, "image_processor", None) is not None:
            processor = self.processor.image_processor  # type: ignore
        else:
            processor = self.processor.video_processor  # type: ignore

        image_patch_size = processor.patch_size
        assert isinstance(image_patch_size, int)

        return image_patch_size

    @cached_property
    def overlap_margins(self) -> tuple[int, int]:
        image_processor = self.processor.image_processor  # type: ignore

        left_margin, right_margin = image_processor.overlap_margins
        assert isinstance(left_margin, int)
        assert isinstance(right_margin, int)

        return left_margin, right_margin

    @cached_property
    def bos_token(self) -> str:
        return self.processor.tokenizer.bos_token or self.processor.tokenizer.eos_token

    @cached_property
    def image_patch_id(self) -> int:
        return self.vocab[IMAGE_PATCH_TOKEN]

    @cached_property
    def im_col_id(self) -> int:
        return self.vocab[IM_COL_TOKEN]

    @cached_property
    def im_start_id(self) -> int:
        return self.vocab[IM_START_TOKEN]

    @cached_property
    def im_end_id(self) -> int:
        return self.vocab[IM_END_TOKEN]
    
    @cached_property
    def low_res_im_start_id(self) -> int:
        return self.vocab[LOW_RES_IMAGE_START_TOKEN]
    
    @cached_property
    def frame_start_id(self) -> int:
        return self.vocab[FRAME_START_TOKEN]
    
    @cached_property
    def frame_end_id(self) -> int:
        return self.vocab[FRAME_END_TOKEN]
    
    @cached_property
    def image_placeholder_id(self) -> int:
        return self.vocab[IMAGE_PROMPT]
    
    @cached_property
    def video_placeholder_id(self) -> int:
        return self.vocab[VIDEO_PROMPT]

    def select_tiling(
        self,
        *,
        image_height: int,
        image_width: int,
    ) -> tuple[int, int]:
        max_crops = self.max_crops
        left_margin, right_margin = self.overlap_margins
        base_image_input_size = self.base_image_input_size
        base_image_input_d = self.image_patch_size

        total_margin_pixels = base_image_input_d * (right_margin + left_margin)
        crop_patches = base_image_input_size[0] // base_image_input_d
        crop_window_patches = crop_patches - (right_margin + left_margin)
        crop_window_size = crop_window_patches * base_image_input_d
        tiling_h, tiling_w = select_tiling(
            height=image_height - total_margin_pixels,
            width=image_width - total_margin_pixels,
            patch_size=crop_window_size,
            max_num_patches=max_crops,
        )

        return tiling_h, tiling_w
    
    def get_base_grid_size(self, is_video: bool) -> tuple[int, int]:
        base_image_input_size = self.base_image_input_size

        return get_patches_grid_size(
            image_h=base_image_input_size[0],
            image_w=base_image_input_size[1],
            patch_size=self.image_patch_size,
            pool_h=self.video_pooling_h if is_video else self.image_pooling_h,
            pool_w=self.video_pooling_w if is_video else self.image_pooling_w,
        )

    def get_patches_grid_size(
        self,
        *,
        image_height: int,
        image_width: int,
    ) -> tuple[int, int]:
        left_margin, right_margin = self.overlap_margins
        base_image_input_size = self.base_image_input_size
        base_image_input_d = self.image_patch_size

        total_margin_pixels = base_image_input_d * (right_margin + left_margin)
        crop_patches = base_image_input_size[0] // base_image_input_d
        crop_window_patches = crop_patches - (right_margin + left_margin)
        crop_window_size = crop_window_patches * base_image_input_d

        tiling_h, tiling_w = self.select_tiling(
            image_height=image_height,
            image_width=image_width,
        )

        h, w = [
            tiling_h * crop_window_size + total_margin_pixels, 
            tiling_w * crop_window_size + total_margin_pixels,
        ]
        nrows, ncols = get_patches_grid_size(
            image_h=h,
            image_w=w,
            patch_size=base_image_input_d,
            pool_h=self.image_pooling_h,
            pool_w=self.image_pooling_w,
        )

        return nrows, ncols

    def __call__(
        self,
        text: TextInput | list[TextInput] | None = None,
        images: ImageInput | None = None,
        videos: dict[str, Any] | list[dict[str, Any]] | None = None,
        return_tensors: str | TensorType = None,
        **kwargs: object,
    ) -> BatchFeature:
        inputs = [text]
        if getattr(self.processor, "image_processor", None) is not None:
            inputs.append(images)
        if getattr(self.processor, "video_processor", None) is not None:
            inputs.append(videos)
        outputs = self.processor(  # type: ignore
            *inputs, return_tensors=return_tensors, **kwargs,
        )

        if images is None:
            images = []
        if not isinstance(images, list):
            images = [images]
        
        if videos is None:
            videos = []
        if not isinstance(videos, list):
            videos = [videos]
        
        assert len(videos) in {0, 1}, "At most one video is supported for Molmo2"
        
        # outputs = {k: v.squeeze(0) if k not in ["input_ids", "attention_mask"] else v for k, v in outputs.items()}
        attention_mask: torch.Tensor = outputs.pop("attention_mask")

        if len(images) > 0:
            # For each image: tiling_h * tiling_w + global view
            num_crops = []
            for image in images:
                image_size = get_image_size(image)
                tiling = self.select_tiling(
                    image_height=image_size.height,
                    image_width=image_size.width,
                )
                num_crops.append(np.prod(tiling) + 1)
                    
            assert sum(num_crops) == len(outputs["pixel_values"])
            assert sum(num_crops) == outputs["image_num_crops"].sum().item()
            image_grids: torch.Tensor = outputs.pop("image_grids")
            image_num_pooled_patches: torch.Tensor = (
                image_grids[:, :2].prod(dim=1) + image_grids[:, 2:].prod(dim=1)
            )
            outputs["image_num_pooled_patches"] = image_num_pooled_patches
            n_patches = outputs["pixel_values"].shape[1]
            outputs["image_num_patches"] = outputs["image_num_crops"] * n_patches
        
        if len(videos) > 0:
            video_grids: torch.Tensor = outputs.pop("video_grids")
            num_crops = []
            for video in videos:
                num_crops.append(len(video["frames"]))
            assert sum(num_crops) == len(outputs["pixel_values_videos"])
            outputs["video_num_crops"] = torch.tensor(num_crops)
            outputs["video_num_pooled_patches"] = video_grids.prod(dim=1)
            n_patches = outputs["pixel_values_videos"].shape[1]
            outputs["video_num_patches"] = outputs["video_num_crops"] * n_patches

        return BatchFeature(outputs)


MolmoVideoItem: TypeAlias = dict[str, Any]
MolmoImageItem: TypeAlias = HfImageItem | list[HfImageItem]

class MolmoVideoProcessorItems(ProcessorBatchItems[MolmoVideoItem]):

    def __init__(
        self,
        data: Sequence[MolmoVideoItem],
        metadata: dict[str, Any] | list[dict[str, Any] | None] | None = None,
    ) -> None:
        super().__init__(data, "video")
        self.metadata = metadata

    def get_num_frames(self, item_idx: int) -> int:
        return len(self.get(item_idx)["frames"])
    
    def get_timestamps(self, item_idx: int) -> np.ndarray:
        return self.get(item_idx)["timestamps"]

    def get_frame_size(self, item_idx: int) -> ImageSize:
        image = self.get(item_idx)["frames"][0]  # Assume that the video isn't empty

        if isinstance(image, Image):
            return ImageSize(*image.size)
        if isinstance(image, (np.ndarray, torch.Tensor)):
            h, w, _ = image.shape
            return ImageSize(w, h)

        assert_never(image)


class Molmo2MultiModalDataParser(MultiModalDataParser):

    def _parse_video_data(
        self,
        data: dict[str, Any] | list[dict[str, Any]],
    ) -> ModalityDataItems[Any, Any] | None:
        if self._is_empty(data):
            return None
        if not isinstance(data, (list, tuple)):
            data = [data]
        return MolmoVideoProcessorItems(data)


class Molmo2ProcessingInfo(BaseProcessingInfo):

    def get_hf_processor(self, **kwargs: object) -> Molmo2ProcessorWrapper:
        try:
            processor = self.ctx.get_hf_processor(**kwargs)
        except TypeError as e:
            # If ctx.get_hf_processor raises TypeError (e.g., returns tokenizer instead of processor),
            # we need to manually construct the processor
            # Try to get the tokenizer from the context
            tokenizer = None
            model_path = None
            
            # Try to get tokenizer using get_hf_tokenizer if available
            if hasattr(self.ctx, 'get_hf_tokenizer'):
                try:
                    tokenizer = self.ctx.get_hf_tokenizer(**kwargs)
                except Exception:
                    pass
            
            # If we don't have a tokenizer yet, try to get it from other context attributes
            if tokenizer is None:
                # Try to access tokenizer from context attributes
                if hasattr(self.ctx, 'tokenizer'):
                    tokenizer = self.ctx.tokenizer
                elif hasattr(self.ctx, '_tokenizer'):
                    tokenizer = self.ctx._tokenizer
            
            # Get model path from tokenizer or context
            if tokenizer is not None:
                model_path = getattr(tokenizer, "name_or_path", None)
                if model_path is None:
                    # Try to get path from tokenizer's init_kwargs or other attributes
                    model_path = getattr(tokenizer, "_tokenizer", {}).get("model_path", None) if hasattr(tokenizer, "_tokenizer") else None
            
            # If we still don't have a model path, try to get it from context
            if model_path is None:
                if hasattr(self.ctx, 'model_path'):
                    model_path = self.ctx.model_path
                elif hasattr(self.ctx, '_model_path'):
                    model_path = self.ctx._model_path
                elif hasattr(self.ctx, 'config'):
                    # Try to get path from config
                    config = self.ctx.config
                    if hasattr(config, '_name_or_path'):
                        model_path = config._name_or_path
                    elif hasattr(config, 'name_or_path'):
                        model_path = config.name_or_path
                    # Also check if config has hf_config with model path
                    elif hasattr(config, 'hf_config'):
                        hf_config = config.hf_config
                        if hasattr(hf_config, '_name_or_path'):
                            model_path = hf_config._name_or_path
                        elif hasattr(hf_config, 'name_or_path'):
                            model_path = hf_config.name_or_path
                    # Check model_config if available
                    elif hasattr(config, 'model_config') and hasattr(config.model_config, 'hf_config'):
                        hf_config = config.model_config.hf_config
                        if hasattr(hf_config, '_name_or_path'):
                            model_path = hf_config._name_or_path
                        elif hasattr(hf_config, 'name_or_path'):
                            model_path = hf_config.name_or_path
            
            # Now construct the processor
            from molmo_r1.src.models.molmo2.processing import VideoMolmo2Processor, Molmo2VideoProcessor
            
            if tokenizer is None:
                # If we still don't have a tokenizer, try to load it from model_path
                if model_path:
                    from transformers import AutoTokenizer
                    tokenizer = AutoTokenizer.from_pretrained(model_path, **kwargs)
                else:
                    raise RuntimeError(
                        "Cannot construct processor: failed to get tokenizer or model_path from context. "
                        f"Original error: {e}"
                    )
            
            if model_path:
                video_processor = Molmo2VideoProcessor.from_pretrained(model_path, **kwargs)
                processor = VideoMolmo2Processor(
                    video_processor=video_processor,
                    tokenizer=tokenizer,
                )
            else:
                # Fallback: create processor with default video processor
                video_processor = Molmo2VideoProcessor()
                processor = VideoMolmo2Processor(
                    video_processor=video_processor,
                    tokenizer=tokenizer,
                )
            return Molmo2ProcessorWrapper(processor)
        
        # If ctx.get_hf_processor returns a tokenizer instead of a processor,
        # load the processor from the tokenizer's path
        if isinstance(processor, PreTrainedTokenizerBase):
            from molmo_r1.src.models.molmo2.processing import VideoMolmo2Processor, Molmo2VideoProcessor
            tokenizer = processor
            model_path = getattr(tokenizer, "name_or_path", None)
            if model_path is None:
                # Try to get path from tokenizer's init_kwargs or other attributes
                model_path = getattr(tokenizer, "_tokenizer", {}).get("model_path", None) if hasattr(tokenizer, "_tokenizer") else None
            if model_path:
                video_processor = Molmo2VideoProcessor.from_pretrained(model_path, **kwargs)
                processor = VideoMolmo2Processor(
                    video_processor=video_processor,
                    tokenizer=tokenizer,
                )
            else:
                # Fallback: create processor with default video processor
                video_processor = Molmo2VideoProcessor()
                processor = VideoMolmo2Processor(
                    video_processor=video_processor,
                    tokenizer=tokenizer,
                )
        return Molmo2ProcessorWrapper(processor)
        
    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None, "video": None}

    def get_num_image_tokens(
        self,
        *,
        image_height: int,
        image_width: int,
        processor: Molmo2ProcessorWrapper | None = None,
    ) -> int:
        if processor is None:
            processor = self.get_hf_processor()
        
        hf_processor = processor.processor  # type: ignore

        resize_nrows, resize_cols = processor.get_base_grid_size(is_video=False)
        # start/end tokens + image patch token + col tokens
        if hf_processor.use_single_crop_col_tokens is not None:
            use_col_tokens = hf_processor.use_single_crop_col_tokens
        else:
            use_col_tokens = hf_processor.image_use_col_tokens
        extra = 2 + resize_nrows * (resize_cols + int(use_col_tokens))
        overlap_nrows, overlap_ncols = processor.get_patches_grid_size(
            image_height=image_height,
            image_width=image_width,
        )
        joint = 2 + overlap_nrows * (overlap_ncols + int(hf_processor.image_use_col_tokens))

        return extra + joint
    
    def get_num_video_tokens(
        self,
        *,
        num_frames: int,
        processor: Molmo2ProcessorWrapper | None = None,
    ) -> int:
        if processor is None:
            processor = self.get_hf_processor()

        resize_nrows, resize_cols = processor.get_base_grid_size(is_video=True)
        # start/end tokens
        extra = 2 + resize_nrows * (resize_cols + int(processor.processor.video_use_col_tokens))
        return num_frames * extra

    def get_image_size_with_most_features(self) -> ImageSize:
        processor = self.get_hf_processor()

        left_margin, right_margin = processor.overlap_margins
        base_image_input_size = processor.base_image_input_size
        base_image_input_d = processor.image_patch_size

        total_margin_pixels = base_image_input_d * (right_margin + left_margin)
        crop_patches = base_image_input_size[0] // base_image_input_d
        crop_window_patches = crop_patches - (right_margin + left_margin)
        crop_window_size = crop_window_patches * base_image_input_d

        tilings = get_candidate_tilings(processor.max_crops)
        largest_feature_size, largest_feature_pinpoint = 0, None

        for hr, wr in tilings:
            height = hr * crop_window_size + total_margin_pixels
            width = wr * crop_window_size + total_margin_pixels

            feat_size = self.get_num_image_tokens(
                image_height=height,
                image_width=width,
                processor=processor
            )
            if feat_size > largest_feature_size:
                largest_feature_size = feat_size
                largest_feature_pinpoint = ImageSize(width=width, height=height)

        if largest_feature_size == 0 or largest_feature_pinpoint is None:
            raise ValueError("Cannot have a largest feature size of 0!")

        return largest_feature_pinpoint
    
    def _get_max_video_frames(self, max_tokens: int) -> int:
        num_tokens_per_frame = self.get_num_video_tokens(num_frames=1)
        max_frames = max_tokens // num_tokens_per_frame
        return max(max_frames, 1)
    
    def get_num_frames_with_most_features(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> int:
        max_videos = mm_counts.get("video", 0)
        max_total_frames = self._get_max_video_frames(seq_len)
        max_frames_per_video = min(
            max_total_frames // max(max_videos, 1), _MAX_FRAMES_PER_VIDEO
        )
        return max(max_frames_per_video, 1)


class Molmo2DummyInputsBuilder(BaseDummyInputsBuilder[Molmo2ProcessingInfo]):

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)

        image_placeholder_token = IMAGE_PROMPT
        video_placeholder_token = VIDEO_PROMPT

        if num_images == 1:
            image_string = image_placeholder_token
        else:
            image_string = "".join(
                [f"Image {i + 1}" + image_placeholder_token for i in range(num_images)]
            )

        return image_string + video_placeholder_token * num_videos

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)
        
        dummy_images = []
        dummy_videos = []

        if num_images > 0:
            target_width, target_height = self.info.get_image_size_with_most_features()

            dummy_images = self._get_dummy_images(
                width=target_width,
                height=target_height,
                num_images=num_images,
            )

        if num_videos > 0:
            processor = self.info.get_hf_processor()
            base_image_input_size = processor.base_image_input_size
            target_num_frames = self.info.get_num_frames_with_most_features(seq_len, mm_counts)

            dummy_frames = np.full(
                (target_num_frames, base_image_input_size[0], base_image_input_size[1], 3),
                255,
                dtype=np.uint8,
            )
            dummy_videos = [
                dict(
                    frames=dummy_frames,
                    timestamps=np.array([0.5 * i for i in range(target_num_frames)]),
                    target_fps=-1,
                    sampling_augmentation="",
                )
                for _ in range(num_videos)
            ]

        return {
            "image": dummy_images,
            "video": dummy_videos,
        }


class Molmo2MultiModalProcessor(BaseMultiModalProcessor[Molmo2ProcessingInfo]):
    def _apply_hf_processor_tokens_only(
        self,
        prompt_tokens: list[int],
    ) -> list[int]:
        processor = self.info.get_hf_processor()
        tokenizer = processor.processor.tokenizer
        bos_token_id = tokenizer.bos_token_id or tokenizer.eos_token_id

        if prompt_tokens[0] != bos_token_id:
            # Prepend the bos token to the prompt tokens
            prompt_tokens = [bos_token_id] + prompt_tokens
        
        return prompt_tokens

    def _get_data_parser(self) -> MultiModalDataParser:
        return Molmo2MultiModalDataParser()

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        image_num_crops = hf_inputs.get("image_num_crops", torch.empty(0))
        image_num_pooled_patches = hf_inputs.get("image_num_pooled_patches", torch.empty(0))
        video_num_crops = hf_inputs.get("video_num_crops", torch.empty(0))
        video_num_pooled_patches = hf_inputs.get("video_num_pooled_patches", torch.empty(0))

        return dict(
            pixel_values=MultiModalFieldConfig.flat_from_sizes("image", image_num_crops),
            image_token_pooling=MultiModalFieldConfig.flat_from_sizes(
                "image", image_num_pooled_patches),
            image_num_crops=MultiModalFieldConfig.batched("image"),
            image_num_pooled_patches=MultiModalFieldConfig.batched("image"),
            image_num_patches=MultiModalFieldConfig.batched("image"),
            pixel_values_videos=MultiModalFieldConfig.flat_from_sizes("video", video_num_crops),
            video_token_pooling=MultiModalFieldConfig.flat_from_sizes(
                "video", video_num_pooled_patches),
            video_num_crops=MultiModalFieldConfig.batched("video"),
            video_num_pooled_patches=MultiModalFieldConfig.batched("video"),
            video_num_patches=MultiModalFieldConfig.batched("video"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        
        processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        img_patch_id = processor.image_patch_id
        img_col_id = processor.im_col_id
        img_start_id = processor.im_start_id
        img_end_id = processor.im_end_id
        image_use_col_tokens = processor.processor.image_use_col_tokens
        use_single_crop_col_tokens = processor.processor.use_single_crop_col_tokens
        use_single_crop_start_token = processor.processor.use_single_crop_start_token
        video_use_col_tokens = processor.processor.video_use_col_tokens
        use_frame_special_tokens = processor.processor.use_frame_special_tokens

        def get_image_replacement_molmo2(item_idx: int) -> list[int]:
            images = mm_items.get_items("image", ImageProcessorItems)
            image = images.get(item_idx)

            resize_nrows, resize_cols = processor.get_base_grid_size(is_video=False)
            if use_single_crop_col_tokens is not None:
                use_col_tokens = use_single_crop_col_tokens
            else:
                use_col_tokens = image_use_col_tokens
            if use_single_crop_start_token:
                start_id = processor.low_res_im_start_id
            else:
                start_id = img_start_id
            extra_row = [img_patch_id] * resize_cols + [img_col_id] * int(use_col_tokens)
            extra_joint = (
                [start_id] + extra_row * resize_nrows + [img_end_id]
            )

            image_size = get_image_size(image)

            nrows, ncols = processor.get_patches_grid_size(
                image_height=image_size.height,
                image_width=image_size.width,
            )

            joint_row = [img_patch_id] * ncols + [img_col_id] * int(image_use_col_tokens)
            joint = (
                [img_start_id] + joint_row * nrows + [img_end_id]
            )
            img_token_ids = extra_joint + joint
            
            return img_token_ids
        
        def get_video_replacement_molmo2(item_idx: int) -> list[int]:
            video = mm_items.get_items("video", MolmoVideoProcessorItems)
            if video is None:
                raise ValueError(f"Failed to apply prompt replacement for mm_items['video'][{item_idx}]: video items are None")
            if item_idx >= len(video):
                raise ValueError(f"Failed to apply prompt replacement for mm_items['video'][{item_idx}]: item index {item_idx} is out of range (video has {len(video)} items)")
            timestamps = video.get_timestamps(item_idx)
            if timestamps is None or len(timestamps) == 0:
                raise ValueError(f"Failed to apply prompt replacement for mm_items['video'][{item_idx}]: timestamps are None or empty")
            time_mode = processor.processor.time_mode
            nrows, ncols = processor.get_base_grid_size(is_video=True)

            if use_frame_special_tokens:
                start_id = processor.frame_start_id
                end_id = processor.frame_end_id
            else:
                start_id = img_start_id
                end_id = img_end_id

            img_token_ids = []

            for frame_idx, frame_time in enumerate(timestamps):
                if time_mode == "numbered-frames":
                    prev_space = " " if frame_idx > 0 else ""
                    frame_prefix = prev_space + f"{frame_idx+1}: " # explicit whitespace before/after image tokens
                elif time_mode == "per-frame":
                    prev_space = " " if frame_idx > 0 else ""
                    frame_prefix = prev_space + f"time {frame_time:.2f} " # explicit whitespace before/after image tokens
                elif time_mode == "per-frame-compact":
                    prev_space = " " if frame_idx > 0 else ""
                    frame_prefix = prev_space + f"{frame_time:.1f} " # explicit whitespace before/after image tokens
                else:
                    frame_prefix = None
                
                if frame_prefix is not None:
                    img_token_ids += processor.processor.tokenizer.encode(
                        frame_prefix, add_special_tokens=False,
                    )
                
                joint_row = [img_patch_id] * ncols + [img_col_id] * int(video_use_col_tokens)
                joint = [start_id] + nrows * joint_row + [end_id]
                img_token_ids += joint
            
            return img_token_ids

        return [
            PromptReplacement(
                modality=modality,
                target=[target],
                replacement=replacement_fn,
            )
            for modality, target, replacement_fn in zip(
                ["image", "video"],
                [processor.image_placeholder_id, processor.video_placeholder_id],
                [get_image_replacement_molmo2, get_video_replacement_molmo2]
            )
        ]


class ImageMolmo2ProcessingInfo(Molmo2ProcessingInfo):

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None}


class ImageMolmo2DummyInputsBuilder(BaseDummyInputsBuilder[ImageMolmo2ProcessingInfo]):

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)

        image_placeholder_token = IMAGE_PROMPT

        if num_images == 1:
            image_string = image_placeholder_token
        else:
            image_string = "".join(
                [f"Image {i + 1}" + image_placeholder_token for i in range(num_images)]
            )

        return image_string

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)
        if num_images == 0:
            return {"image": []}

        target_width, target_height = self.info.get_image_size_with_most_features()
        

        return {
            "image":
            self._get_dummy_images(
                width=target_width,
                height=target_height,
                num_images=num_images
            ),
        }


class ImageMolmo2MultiModalProcessor(BaseMultiModalProcessor[ImageMolmo2ProcessingInfo]):
    def _apply_hf_processor_tokens_only(
        self,
        prompt_tokens: list[int],
    ) -> list[int]:
        processor = self.info.get_hf_processor()
        tokenizer = processor.processor.tokenizer
        bos_token_id = tokenizer.bos_token_id or tokenizer.eos_token_id

        if prompt_tokens[0] != bos_token_id:
            # Prepend the bos token to the prompt tokens
            prompt_tokens = [bos_token_id] + prompt_tokens
        
        return prompt_tokens

    def _get_data_parser(self) -> MultiModalDataParser:
        return Molmo2MultiModalDataParser()

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        image_num_crops = hf_inputs.get("image_num_crops", torch.empty(0))
        image_num_pooled_patches = hf_inputs.get("image_num_pooled_patches", torch.empty(0))

        return dict(
            pixel_values=MultiModalFieldConfig.flat_from_sizes("image", image_num_crops),
            image_token_pooling=MultiModalFieldConfig.flat_from_sizes(
                "image", image_num_pooled_patches),
            image_num_crops=MultiModalFieldConfig.batched("image"),
            image_num_pooled_patches=MultiModalFieldConfig.batched("image"),
            image_num_patches=MultiModalFieldConfig.batched("image"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        
        processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        img_patch_id = processor.image_patch_id
        img_col_id = processor.im_col_id
        img_start_id = processor.im_start_id
        img_end_id = processor.im_end_id
        image_use_col_tokens = processor.processor.image_use_col_tokens
        use_single_crop_col_tokens = processor.processor.use_single_crop_col_tokens
        use_single_crop_start_token = processor.processor.use_single_crop_start_token

        def get_image_replacement_molmo2(item_idx: int) -> list[int]:
            images = mm_items.get_items("image", ImageProcessorItems)
            image = images.get(item_idx)

            resize_nrows, resize_cols = processor.get_base_grid_size(is_video=False)
            if use_single_crop_col_tokens is not None:
                use_col_tokens = use_single_crop_col_tokens
            else:
                use_col_tokens = image_use_col_tokens
            if use_single_crop_start_token:
                start_id = processor.low_res_im_start_id
            else:
                start_id = img_start_id
            extra_row = [img_patch_id] * resize_cols + [img_col_id] * int(use_col_tokens)
            extra_joint = (
                [start_id] + extra_row * resize_nrows + [img_end_id]
            )

            image_size = get_image_size(image)

            nrows, ncols = processor.get_patches_grid_size(
                image_height=image_size.height,
                image_width=image_size.width,
            )

            joint_row = [img_patch_id] * ncols + [img_col_id] * int(image_use_col_tokens)
            joint = (
                [img_start_id] + joint_row * nrows + [img_end_id]
            )
            img_token_ids = extra_joint + joint
            
            return img_token_ids
        return [
            PromptReplacement(
                modality="image",
                target=[processor.image_placeholder_id],
                replacement=get_image_replacement_molmo2,
            )
        ]


class VideoMolmo2ProcessingInfo(Molmo2ProcessingInfo):

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"video": None}


class VideoMolmo2DummyInputsBuilder(BaseDummyInputsBuilder[VideoMolmo2ProcessingInfo]):

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_videos = mm_counts.get("video", 0)

        video_placeholder_token = VIDEO_PROMPT

        return video_placeholder_token * num_videos

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> MultiModalDataDict:
        num_videos = mm_counts.get("video", 0)
        if num_videos == 0:
            return {"video": []}

        processor = self.info.get_hf_processor()
        base_image_input_size = processor.base_image_input_size
        
        target_num_frames = self.info.get_num_frames_with_most_features(seq_len, mm_counts)

        dummy_frames = np.full(
            (target_num_frames, base_image_input_size[0], base_image_input_size[1], 3),
            255,
            dtype=np.uint8,
        )
        dummy_videos = [
            dict(
                frames=dummy_frames,
                timestamps=np.array([0.5 * i for i in range(target_num_frames)]),
                target_fps=-1,
                sampling_augmentation="",
            )
            for _ in range(num_videos)
        ]

        return {
            "video": dummy_videos,
        }


class VideoMolmo2MultiModalProcessor(BaseMultiModalProcessor[VideoMolmo2ProcessingInfo]):
    def _apply_hf_processor_tokens_only(
        self,
        prompt_tokens: list[int],
    ) -> list[int]:
        processor = self.info.get_hf_processor()
        tokenizer = processor.processor.tokenizer
        bos_token_id = tokenizer.bos_token_id or tokenizer.eos_token_id

        if prompt_tokens[0] != bos_token_id:
            # Prepend the bos token to the prompt tokens
            prompt_tokens = [bos_token_id] + prompt_tokens
        
        return prompt_tokens

    def _get_data_parser(self) -> MultiModalDataParser:
        return Molmo2MultiModalDataParser()

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        video_num_crops = hf_inputs.get("video_num_crops", torch.empty(0))
        video_num_pooled_patches = hf_inputs.get("video_num_pooled_patches", torch.empty(0))

        return dict(
            pixel_values_videos=MultiModalFieldConfig.flat_from_sizes("video", video_num_crops),
            video_token_pooling=MultiModalFieldConfig.flat_from_sizes(
                "video", video_num_pooled_patches),
            video_num_crops=MultiModalFieldConfig.batched("video"),
            video_num_pooled_patches=MultiModalFieldConfig.batched("video"),
            video_num_patches=MultiModalFieldConfig.batched("video"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        
        processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        img_patch_id = processor.image_patch_id
        img_col_id = processor.im_col_id
        img_start_id = processor.im_start_id
        img_end_id = processor.im_end_id
        video_use_col_tokens = processor.processor.video_use_col_tokens
        use_frame_special_tokens = processor.processor.use_frame_special_tokens

        def get_video_replacement_molmo2(item_idx: int) -> list[int]:
            video = mm_items.get_items("video", MolmoVideoProcessorItems)
            if video is None:
                raise ValueError(f"Failed to apply prompt replacement for mm_items['video'][{item_idx}]: video items are None")
            if item_idx >= len(video):
                raise ValueError(f"Failed to apply prompt replacement for mm_items['video'][{item_idx}]: item index {item_idx} is out of range (video has {len(video)} items)")
            timestamps = video.get_timestamps(item_idx)
            if timestamps is None or len(timestamps) == 0:
                raise ValueError(f"Failed to apply prompt replacement for mm_items['video'][{item_idx}]: timestamps are None or empty")
            time_mode = processor.processor.time_mode
            nrows, ncols = processor.get_base_grid_size(is_video=True)

            if use_frame_special_tokens:
                start_id = processor.frame_start_id
                end_id = processor.frame_end_id
            else:
                start_id = img_start_id
                end_id = img_end_id

            img_token_ids = []

            for frame_idx, frame_time in enumerate(timestamps):
                if time_mode == "numbered-frames":
                    prev_space = " " if frame_idx > 0 else ""
                    frame_prefix = prev_space + f"{frame_idx+1}: " # explicit whitespace before/after image tokens
                elif time_mode == "per-frame":
                    prev_space = " " if frame_idx > 0 else ""
                    frame_prefix = prev_space + f"time {frame_time:.2f} " # explicit whitespace before/after image tokens
                elif time_mode == "per-frame-compact":
                    prev_space = " " if frame_idx > 0 else ""
                    frame_prefix = prev_space + f"{frame_time:.1f} " # explicit whitespace before/after image tokens
                else:
                    frame_prefix = None
                
                if frame_prefix is not None:
                    img_token_ids += processor.processor.tokenizer.encode(
                        frame_prefix, add_special_tokens=False,
                    )
                
                joint_row = [img_patch_id] * ncols + [img_col_id] * int(video_use_col_tokens)
                joint = [start_id] + nrows * joint_row + [end_id]
                img_token_ids += joint
            
            return img_token_ids

        return [
            PromptReplacement(
                modality="video",
                target=[processor.video_placeholder_id],
                replacement=get_video_replacement_molmo2,
            )
        ]


@MULTIMODAL_REGISTRY.register_processor(
    Molmo2MultiModalProcessor,
    info=Molmo2ProcessingInfo,
    dummy_inputs=Molmo2DummyInputsBuilder,
)
class Molmo2ForConditionalGeneration(
    nn.Module, SupportsMultiModal, SupportsPP, SupportsLoRA, SupportsQuant
):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_substr={
            # vision backbone mapping
            "image_projector.w1.": "image_projector.gate_proj.",
            "image_projector.w3.": "image_projector.up_proj.",
            "image_projector.w2.": "image_projector.down_proj.",
            # language backbone mapping
            "att_proj": "qkv_proj",
            "attn_out": "o_proj",
            "q_norm": "q_norm",
            "k_norm": "k_norm",
            "ff_proj": "up_gate_proj",
            "ff_out": "down_proj",
            "attn_norm": "input_layernorm",
            "ff_norm": "post_attention_layernorm",
        },
        orig_to_new_prefix={
            # vision backbone mapping
            "model.vision_backbone.": "vision_backbone.",
            # language backbone mapping
            "model.transformer.blocks.": "model.layers.",
            "model.transformer.ln_f.": "model.norm.",
        },
    )

    packed_modules_mapping = {
        "qkv_proj": ["qkv_proj"],
        "up_gate_proj": ["up_gate_proj"],  # language model
        "merged_linear": ["gate_proj", "up_proj"]  # image_projector
    }

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return IMAGE_PROMPT
        if modality.startswith("video"):
            return VIDEO_PROMPT

        raise ValueError("Only image or video modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        lora_config = vllm_config.lora_config
        self.config = config
        self.multimodal_config = multimodal_config
        self.lora_config = lora_config

        kwargs = {}
        for field in fields(VitConfig):
            kwargs[field.name] = getattr(config.vit_config, field.name)
        vit_config = VitConfig(**kwargs)

        kwargs = {}
        for field in fields(AdapterConfig):
            kwargs[field.name] = getattr(config.adapter_config, field.name)
        adapter_config = AdapterConfig(**kwargs)

        self.vision_backbone = Molmo2VisionBackbone(
            vit_config, adapter_config, quant_config,
        )
        self.model = Molmo2TextModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        
        self.img_patch_id = config.image_patch_id
        self.bi_directional_attn = config.bi_directional_attn
        if envs.VLLM_USE_V1 and self.bi_directional_attn:
            logger.warning_once(
                "Molmo2 has suboptimal results on V1 "
                "because of the simplified attention pattern being used."
            )

        if self.bi_directional_attn:
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

        if hasattr(config, "text_config"):
            hf_text_config = config.text_config
        else:
            hf_text_config = config.llm_config

        self.lm_head = ParallelLMHead(
            hf_text_config.vocab_size,
            hf_text_config.hidden_size,
            quant_config=quant_config,
        )
        self.logits_processor = LogitsProcessor(hf_text_config.vocab_size)

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def _parse_and_validate_image_input(
        self,
        **kwargs: object,
    ) -> Molmo2ImageInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)
        if pixel_values is None:
            return None

        if not isinstance(pixel_values, (torch.Tensor, list)):
            raise ValueError(
                "Incorrect type of pixel_values. "
                f"Got type: {type(pixel_values)}"
            )
        
        pixel_values = flatten_bn(pixel_values, concat=True)
        
        token_pooling = kwargs.pop("image_token_pooling", None)
        if not isinstance(token_pooling, (torch.Tensor, list)):
            raise ValueError(
                "Incorrect type of token_pooling. "
                f"Got type: {type(token_pooling)}"
            )
        
        token_pooling = flatten_bn(token_pooling, concat=True)
        
        num_pooled_patches = kwargs.pop("image_num_pooled_patches", None)
        if not isinstance(num_pooled_patches, (torch.Tensor, list)):
            raise ValueError(
                "Incorrect type of num_pooled_patches. "
                f"Got type: {type(num_pooled_patches)}"
            )
        
        num_patches = kwargs.pop("image_num_patches", None)
        if not isinstance(num_patches, (torch.Tensor, list)):
            raise ValueError(
                "Incorrect type of num_patches. "
                f"Got type: {type(num_patches)}"
            )

        num_pooled_patches = flatten_bn(num_pooled_patches, concat=True)
        num_patches = flatten_bn(num_patches, concat=True)
        accum_patches = [0] + num_patches.cumsum(dim=0)[:-1].tolist()
        patch_offset = 0
        new_token_pooling = token_pooling.clone()
        for i, n in enumerate(num_pooled_patches):
            cur_slice = token_pooling[patch_offset:patch_offset + n]
            index_offset = int(accum_patches[i])
            new_token_pooling[patch_offset:patch_offset + n] = torch.where(
                cur_slice >= 0,
                cur_slice + index_offset,
                cur_slice,
            )
            patch_offset += n

        return Molmo2ImageInputs(
            pixel_values=pixel_values,
            token_pooling=new_token_pooling,
            num_pooled_patches=num_pooled_patches,
        )
    
    def _parse_and_validate_video_input(
        self,
        **kwargs: object,
    ) -> Molmo2ImageInputs | None:
        pixel_values_videos = kwargs.pop("pixel_values_videos", None)
        if pixel_values_videos is None:
            return None

        if not isinstance(pixel_values_videos, (torch.Tensor, list)):
            raise ValueError(
                "Incorrect type of pixel_values_videos. "
                f"Got type: {type(pixel_values_videos)}"
            )
        
        pixel_values_videos = flatten_bn(pixel_values_videos, concat=True)
        
        token_pooling = kwargs.pop("video_token_pooling", None)
        if not isinstance(token_pooling, (torch.Tensor, list)):
            raise ValueError(
                "Incorrect type of token_pooling. "
                f"Got type: {type(token_pooling)}"
            )
        
        token_pooling = flatten_bn(token_pooling, concat=True)
        
        num_pooled_patches = kwargs.pop("video_num_pooled_patches", None)
        if not isinstance(num_pooled_patches, (torch.Tensor, list)):
            raise ValueError(
                "Incorrect type of num_pooled_patches. "
                f"Got type: {type(num_pooled_patches)}"
            )
        
        num_patches = kwargs.pop("video_num_patches", None)
        if not isinstance(num_patches, (torch.Tensor, list)):
            raise ValueError(
                "Incorrect type of num_patches. "
                f"Got type: {type(num_patches)}"
            )

        num_pooled_patches = flatten_bn(num_pooled_patches, concat=True)
        num_patches = flatten_bn(num_patches, concat=True)
        accum_patches = [0] + num_patches.cumsum(dim=0)[:-1].tolist()
        patch_offset = 0
        new_token_pooling = token_pooling.clone()
        for i, n in enumerate(num_pooled_patches):
            cur_slice = token_pooling[patch_offset:patch_offset + n]
            index_offset = int(accum_patches[i])
            new_token_pooling[patch_offset:patch_offset + n] = torch.where(
                cur_slice >= 0,
                cur_slice + index_offset,
                cur_slice,
            )
            patch_offset += n

        return Molmo2VideoInputs(
            pixel_values_videos=pixel_values_videos,
            token_pooling=new_token_pooling,
            num_pooled_patches=num_pooled_patches,
        )
    
    def _parse_and_validate_multimodal_inputs(self, **kwargs: object) -> dict:
        modalities = {}

        for input_key in kwargs:
            if input_key in ("pixel_values",) and "images" not in modalities:
                modalities["images"] = self._parse_and_validate_image_input(**kwargs)
            if input_key in ("pixel_values_videos",) and "videos" not in modalities:
                modalities["videos"] = self._parse_and_validate_video_input(**kwargs)
        return modalities

    def _process_image_input(
        self,
        image_input: Molmo2ImageInputs,
    ) -> list[torch.Tensor]:
        pixel_values = image_input["pixel_values"]
        token_pooling = image_input["token_pooling"]
        num_pooled_patches = image_input["num_pooled_patches"]

        image_features_flat = self.vision_backbone(
            images=pixel_values.unsqueeze(0),
            token_pooling=token_pooling.unsqueeze(0),
        )

        assert len(image_features_flat) == num_pooled_patches.sum()
        return image_features_flat.split(num_pooled_patches.tolist(), dim=0)

    def _process_video_input(
        self,
        video_input: Molmo2VideoInputs,
    ) -> list[torch.Tensor]:
        pixel_values_videos = video_input["pixel_values_videos"]
        token_pooling = video_input["token_pooling"]
        num_pooled_patches = video_input["num_pooled_patches"]

        image_features_flat = self.vision_backbone(
            images=pixel_values_videos.unsqueeze(0),
            token_pooling=token_pooling.unsqueeze(0),
        )

        assert len(image_features_flat) == num_pooled_patches.sum()
        return image_features_flat.split(num_pooled_patches.tolist(), dim=0)

    def get_language_model(self) -> torch.nn.Module:
        return self.model

    def get_multimodal_embeddings(
        self,
        **kwargs: object,
    ) -> MultiModalEmbeddings | None:
        modalities = self._parse_and_validate_multimodal_inputs(**kwargs)
        if not modalities:
            return []
        
        multimodal_embeddings: tuple[torch.Tensor, ...] = ()

        for modality in modalities:
            if modality == "images":
                image_input = modalities["images"]
                image_embeddings = self._process_image_input(image_input)
                multimodal_embeddings += image_embeddings
            if modality == "videos":
                video_input = modalities["videos"]
                video_embeddings = self._process_video_input(video_input)
                multimodal_embeddings += video_embeddings
        
        return multimodal_embeddings

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
    ) -> torch.Tensor:
        inputs_embeds = self.model.get_input_embeddings(input_ids)
        if multimodal_embeddings is not None and len(multimodal_embeddings) != 0:
            assert self.img_patch_id is not None

            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                multimodal_embeddings,
                self.img_patch_id,
            )
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.LongTensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:

        if intermediate_tensors is not None:
            inputs_embeds = None

        # NOTE: In v1, inputs_embeds is always generated at model runner, this
        # condition is for v0 compatibility.
        elif inputs_embeds is None:
            vision_embeddings = self.get_multimodal_embeddings(**kwargs)
            inputs_embeds = self.get_input_embeddings(input_ids, vision_embeddings)
            if (vision_embeddings is not None) and len(vision_embeddings) != 0 and self.bi_directional_attn:
                kwargs = self.prepare_attn_masks(
                    input_ids,
                    positions,
                    mask_dtype=self.dtype,
                    **kwargs,
                )
            input_ids = None

        hidden_states = self.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        return hidden_states
    
    def prepare_attn_masks(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        mask_dtype: torch.dtype,
        **kwargs: object,
    ) -> object:
        kwargs["has_images"] = True
        start_indices = (positions == 0).cpu().nonzero()
        num_seqs = len(start_indices)
        seq_lens = []
        for i in range(num_seqs):
            start_idx = start_indices[i].item()
            if i < num_seqs - 1:
                end_idx = start_indices[i + 1].item()
            else:
                end_idx = len(input_ids)
            seq_lens.append(end_idx - start_idx)
        kwargs["seq_lens"] = seq_lens

        image_tokens = self.image_tokens.to(input_ids.device)
        attn_masks = []
        start_idx = 0
        for seq_len in seq_lens:
            end_idx = start_idx + seq_len
            input_token_ids = input_ids[start_idx:end_idx]
            start_idx = end_idx
            # Create a causal mask.
            attn_mask = torch.empty(
                1,
                1,
                seq_len,
                seq_len,
                dtype=mask_dtype,
                device=input_ids.device,
            )
            attn_mask.fill_(float("-inf"))
            # Fill the lower triangle with 0.
            attn_mask = attn_mask.triu(diagonal=1)

            # Consider the bidirectional attention between image tokens.
            img_mask = torch.zeros_like(attn_mask, dtype=torch.long)
            img_pos = torch.any(input_token_ids[:, None] == image_tokens[None, :], -1)
            img_mask[:, :, :, img_pos] += 1
            img_mask[:, :, img_pos, :] += 1
            attn_mask = torch.where(img_mask == 2, 0, attn_mask)
            attn_masks.append(attn_mask)
        kwargs["attn_masks"] = attn_masks
        return kwargs

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        logits = self.logits_processor(
            self.lm_head,
            hidden_states,
            sampling_metadata,
        )
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):

        loader = AutoWeightsLoader(self)
        weights = _get_weights_with_merged_embedding(weights)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    def get_mm_mapping(self) -> MultiModelKeys:
        """
        Get the module prefix in multimodal models
        """
        return MultiModelKeys.from_string_field(
            language_model="model",
            connector="vision_backbone.image_projector",
            tower_model="vision_backbone",
        )


@MULTIMODAL_REGISTRY.register_processor(
    ImageMolmo2MultiModalProcessor,
    info=ImageMolmo2ProcessingInfo,
    dummy_inputs=ImageMolmo2DummyInputsBuilder,
)
class ImageMolmo2ForConditionalGeneration(
    nn.Module, SupportsMultiModal, SupportsPP, SupportsLoRA, SupportsQuant,
):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_substr={
            # vision backbone mapping
            "image_projector.w1.": "image_projector.gate_proj.",
            "image_projector.w3.": "image_projector.up_proj.",
            "image_projector.w2.": "image_projector.down_proj.",
            # language backbone mapping
            "att_proj": "qkv_proj",
            "attn_out": "o_proj",
            "q_norm": "q_norm",
            "k_norm": "k_norm",
            "ff_proj": "up_gate_proj",
            "ff_out": "down_proj",
            "attn_norm": "input_layernorm",
            "ff_norm": "post_attention_layernorm",
        },
        orig_to_new_prefix={
            # vision backbone mapping
            "model.vision_backbone.": "vision_backbone.",
            # language backbone mapping
            "model.transformer.blocks.": "model.layers.",
            "model.transformer.ln_f.": "model.norm.",
        },
    )

    packed_modules_mapping = {
        "qkv_proj": ["qkv_proj"],
        "up_gate_proj": ["up_gate_proj"],  # language model
        "merged_linear": ["gate_proj", "up_proj"]  # image_projector
    }

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return IMAGE_PROMPT

        raise ValueError("Only image modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        lora_config = vllm_config.lora_config
        self.config = config
        self.multimodal_config = multimodal_config
        self.lora_config = lora_config

        kwargs = {}
        for field in fields(VitConfig):
            kwargs[field.name] = getattr(config.vit_config, field.name)
        vit_config = VitConfig(**kwargs)

        kwargs = {}
        for field in fields(AdapterConfig):
            kwargs[field.name] = getattr(config.adapter_config, field.name)
        adapter_config = AdapterConfig(**kwargs)

        self.vision_backbone = Molmo2VisionBackbone(vit_config, adapter_config, quant_config)
        self.model = Molmo2TextModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"))
        
        self.img_patch_id = config.image_patch_id
        self.image_start_token_id = config.image_start_token_id
        self.image_end_token_id = config.image_end_token_id
        self.image_low_res_id = config.image_low_res_id
        self.image_col_id = config.image_col_id
        self.bi_directional_attn = config.bi_directional_attn
        if envs.VLLM_USE_V1 and self.bi_directional_attn:
            logger.warning_once(
                "Molmo2 has suboptimal results on V1 "
                "because of the simplified attention pattern being used."
            )

        if self.bi_directional_attn:
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

        if hasattr(config, "text_config"):
            hf_text_config = config.text_config
        else:
            hf_text_config = config.llm_config

        self.lm_head = ParallelLMHead(
            hf_text_config.vocab_size,
            hf_text_config.hidden_size,
            quant_config=quant_config,
        )
        self.logits_processor = LogitsProcessor(hf_text_config.vocab_size)

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def _parse_and_validate_image_input(
        self,
        **kwargs: object,
    ) -> Molmo2ImageInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)
        if pixel_values is None:
            return None

        if not isinstance(pixel_values, (torch.Tensor, list)):
            raise ValueError(
                "Incorrect type of pixel_values. "
                f"Got type: {type(pixel_values)}"
            )
        
        pixel_values = flatten_bn(pixel_values, concat=True)
        
        token_pooling = kwargs.pop("image_token_pooling", None)
        if not isinstance(token_pooling, (torch.Tensor, list)):
            raise ValueError(
                "Incorrect type of token_pooling. "
                f"Got type: {type(token_pooling)}"
            )
        
        token_pooling = flatten_bn(token_pooling, concat=True)
        
        num_pooled_patches = kwargs.pop("image_num_pooled_patches", None)
        if not isinstance(num_pooled_patches, (torch.Tensor, list)):
            raise ValueError(
                "Incorrect type of num_pooled_patches. "
                f"Got type: {type(num_pooled_patches)}"
            )
        
        num_patches = kwargs.pop("image_num_patches", None)
        if not isinstance(num_patches, (torch.Tensor, list)):
            raise ValueError(
                "Incorrect type of num_patches. "
                f"Got type: {type(num_patches)}"
            )

        num_pooled_patches = flatten_bn(num_pooled_patches, concat=True)
        num_patches = flatten_bn(num_patches, concat=True)
        accum_patches = [0] + num_patches.cumsum(dim=0)[:-1].tolist()
        patch_offset = 0
        new_token_pooling = token_pooling.clone()
        for i, n in enumerate(num_pooled_patches):
            cur_slice = token_pooling[patch_offset:patch_offset + n]
            index_offset = int(accum_patches[i])
            new_token_pooling[patch_offset:patch_offset + n] = torch.where(
                cur_slice >= 0,
                cur_slice + index_offset,
                cur_slice,
            )
            patch_offset += n

        return Molmo2ImageInputs(
            pixel_values=pixel_values,
            token_pooling=new_token_pooling,
            num_pooled_patches=num_pooled_patches,
        )
    
    def _parse_and_validate_multimodal_inputs(self, **kwargs: object) -> dict:
        modalities = {}

        for input_key in kwargs:
            if input_key in ("pixel_values",) and "images" not in modalities:
                modalities["images"] = self._parse_and_validate_image_input(**kwargs)
        
        return modalities

    def _process_image_input(
        self,
        image_input: Molmo2ImageInputs,
    ) -> list[torch.Tensor]:
        pixel_values = image_input["pixel_values"]
        token_pooling = image_input["token_pooling"]
        num_pooled_patches = image_input["num_pooled_patches"]

        image_features_flat = self.vision_backbone(
            images=pixel_values.unsqueeze(0),
            token_pooling=token_pooling.unsqueeze(0),
        )

        assert len(image_features_flat) == num_pooled_patches.sum()
        return image_features_flat.split(num_pooled_patches.tolist(), dim=0)

    def get_language_model(self) -> torch.nn.Module:
        return self.model

    def get_multimodal_embeddings(
        self,
        **kwargs: object,
    ) -> MultiModalEmbeddings | None:
        modalities = self._parse_and_validate_multimodal_inputs(**kwargs)
        if not modalities:
            return []
        
        multimodal_embeddings: tuple[torch.Tensor, ...] = ()

        for modality in modalities:
            if modality == "images":
                image_input = modalities["images"]
                image_embeddings = self._process_image_input(image_input)
                multimodal_embeddings += image_embeddings
        
        return multimodal_embeddings

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
    ) -> torch.Tensor:
        inputs_embeds = self.model.get_input_embeddings(input_ids)
        if multimodal_embeddings is not None and len(multimodal_embeddings) != 0:
            assert self.img_patch_id is not None

            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                multimodal_embeddings,
                self.img_patch_id,
            )
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.LongTensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:

        if intermediate_tensors is not None:
            inputs_embeds = None

        # NOTE: In v1, inputs_embeds is always generated at model runner, this
        # condition is for v0 compatibility.
        elif inputs_embeds is None:
            vision_embeddings = self.get_multimodal_embeddings(**kwargs)
            inputs_embeds = self.get_input_embeddings(input_ids, vision_embeddings)
            if (vision_embeddings is not None) and len(vision_embeddings) != 0 and self.bi_directional_attn:
                kwargs = self.prepare_attn_masks(
                    input_ids,
                    positions,
                    mask_dtype=self.dtype,
                    **kwargs,
                )
            input_ids = None

        hidden_states = self.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        return hidden_states

    def prepare_attn_masks(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        mask_dtype: torch.dtype,
        **kwargs: object,
    ) -> object:
        kwargs["has_images"] = True
        start_indices = (positions == 0).cpu().nonzero()
        num_seqs = len(start_indices)
        seq_lens = []
        for i in range(num_seqs):
            start_idx = start_indices[i].item()
            if i < num_seqs - 1:
                end_idx = start_indices[i + 1].item()
            else:
                end_idx = len(input_ids)
            seq_lens.append(end_idx - start_idx)
        kwargs["seq_lens"] = seq_lens

        image_tokens = self.image_tokens.to(input_ids.device)
        attn_masks = []
        start_idx = 0
        for seq_len in seq_lens:
            end_idx = start_idx + seq_len
            input_token_ids = input_ids[start_idx:end_idx]
            start_idx = end_idx
            # Create a causal mask.
            attn_mask = torch.empty(
                1,
                1,
                seq_len,
                seq_len,
                dtype=mask_dtype,
                device=input_ids.device,
            )
            attn_mask.fill_(float("-inf"))
            # Fill the lower triangle with 0.
            attn_mask = attn_mask.triu(diagonal=1)

            # Consider the bidirectional attention between image tokens.
            img_mask = torch.zeros_like(attn_mask, dtype=torch.long)
            img_pos = torch.any(input_token_ids[:, None] == image_tokens[None, :], -1)
            img_mask[:, :, :, img_pos] += 1
            img_mask[:, :, img_pos, :] += 1
            attn_mask = torch.where(img_mask == 2, 0, attn_mask)
            attn_masks.append(attn_mask)
        kwargs["attn_masks"] = attn_masks
        return kwargs

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        logits = self.logits_processor(
            self.lm_head,
            hidden_states,
            sampling_metadata,
        )
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):

        loader = AutoWeightsLoader(self)
        weights = _get_weights_with_merged_embedding(weights)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    def get_mm_mapping(self) -> MultiModelKeys:
        """
        Get the module prefix in multimodal models
        """
        return MultiModelKeys.from_string_field(
            language_model="model",
            connector="vision_backbone.image_projector",
            tower_model="vision_backbone",
        )


@MULTIMODAL_REGISTRY.register_processor(
    VideoMolmo2MultiModalProcessor,
    info=VideoMolmo2ProcessingInfo,
    dummy_inputs=VideoMolmo2DummyInputsBuilder,
)
class VideoMolmo2ForConditionalGeneration(
    nn.Module, SupportsMultiModal, SupportsPP, SupportsLoRA, SupportsQuant,
):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_substr={
            # vision backbone mapping
            "image_projector.w1.": "image_projector.gate_proj.",
            "image_projector.w3.": "image_projector.up_proj.",
            "image_projector.w2.": "image_projector.down_proj.",
            # language backbone mapping
            "att_proj": "qkv_proj",
            "attn_out": "o_proj",
            "q_norm": "q_norm",
            "k_norm": "k_norm",
            "ff_proj": "up_gate_proj",
            "ff_out": "down_proj",
            "attn_norm": "input_layernorm",
            "ff_norm": "post_attention_layernorm",
        },
        orig_to_new_prefix={
            # vision backbone mapping
            "model.vision_backbone.": "vision_backbone.",
            # language backbone mapping
            "model.transformer.blocks.": "model.layers.",
            "model.transformer.ln_f.": "model.norm.",
        },
    )

    packed_modules_mapping = {
        "qkv_proj": ["qkv_proj"],
        "up_gate_proj": ["up_gate_proj"],  # language model
        "merged_linear": ["gate_proj", "up_proj"]  # image_projector
    }

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("video"):
            return VIDEO_PROMPT

        raise ValueError("Only video modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        lora_config = vllm_config.lora_config
        self.config = config
        self.multimodal_config = multimodal_config
        self.lora_config = lora_config

        kwargs = {}
        for field in fields(VitConfig):
            kwargs[field.name] = getattr(config.vit_config, field.name)
        vit_config = VitConfig(**kwargs)

        kwargs = {}
        for field in fields(AdapterConfig):
            kwargs[field.name] = getattr(config.adapter_config, field.name)
        adapter_config = AdapterConfig(**kwargs)

        self.vision_backbone = Molmo2VisionBackbone(vit_config, adapter_config, quant_config)
        self.model = Molmo2TextModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"))
        
        self.img_patch_id = config.image_patch_id
        self.image_start_token_id = config.image_start_token_id
        self.image_end_token_id = config.image_end_token_id
        self.image_low_res_id = config.image_low_res_id
        self.image_col_id = config.image_col_id
        self.bi_directional_attn = config.bi_directional_attn
        if envs.VLLM_USE_V1 and self.bi_directional_attn:
            logger.warning_once(
                "Molmo2 has suboptimal results on V1 "
                "because of the simplified attention pattern being used."
            )

        if self.bi_directional_attn:
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

        if hasattr(config, "text_config"):
            hf_text_config = config.text_config
        else:
            hf_text_config = config.llm_config

        self.lm_head = ParallelLMHead(
            hf_text_config.vocab_size,
            hf_text_config.hidden_size,
            quant_config=quant_config,
        )
        self.logits_processor = LogitsProcessor(hf_text_config.vocab_size)

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def _parse_and_validate_video_input(
        self,
        **kwargs: object,
    ) -> Molmo2ImageInputs | None:
        pixel_values_videos = kwargs.pop("pixel_values_videos", None)
        if pixel_values_videos is None:
            return None

        if not isinstance(pixel_values_videos, (torch.Tensor, list)):
            raise ValueError(
                "Incorrect type of pixel_values_videos. "
                f"Got type: {type(pixel_values_videos)}"
            )
        
        pixel_values_videos = flatten_bn(pixel_values_videos, concat=True)
        
        token_pooling = kwargs.pop("video_token_pooling", None)
        if not isinstance(token_pooling, (torch.Tensor, list)):
            raise ValueError(
                "Incorrect type of token_pooling. "
                f"Got type: {type(token_pooling)}"
            )
        
        token_pooling = flatten_bn(token_pooling, concat=True)
        
        num_pooled_patches = kwargs.pop("video_num_pooled_patches", None)
        if not isinstance(num_pooled_patches, (torch.Tensor, list)):
            raise ValueError(
                "Incorrect type of num_pooled_patches. "
                f"Got type: {type(num_pooled_patches)}"
            )
        
        num_patches = kwargs.pop("video_num_patches", None)
        if not isinstance(num_patches, (torch.Tensor, list)):
            raise ValueError(
                "Incorrect type of num_patches. "
                f"Got type: {type(num_patches)}"
            )

        num_pooled_patches = flatten_bn(num_pooled_patches, concat=True)
        num_patches = flatten_bn(num_patches, concat=True)
        accum_patches = [0] + num_patches.cumsum(dim=0)[:-1].tolist()
        patch_offset = 0
        new_token_pooling = token_pooling.clone()
        for i, n in enumerate(num_pooled_patches):
            cur_slice = token_pooling[patch_offset:patch_offset + n]
            index_offset = int(accum_patches[i])
            new_token_pooling[patch_offset:patch_offset + n] = torch.where(
                cur_slice >= 0,
                cur_slice + index_offset,
                cur_slice,
            )
            patch_offset += n

        return Molmo2VideoInputs(
            pixel_values_videos=pixel_values_videos,
            token_pooling=new_token_pooling,
            num_pooled_patches=num_pooled_patches,
        )
    
    def _parse_and_validate_multimodal_inputs(self, **kwargs: object) -> dict:
        modalities = {}

        for input_key in kwargs:
            if input_key in ("pixel_values_videos",) and "videos" not in modalities:
                modalities["videos"] = self._parse_and_validate_video_input(**kwargs)
        return modalities

    def _process_video_input(
        self,
        video_input: Molmo2VideoInputs,
    ) -> list[torch.Tensor]:
        pixel_values_videos = video_input["pixel_values_videos"]
        token_pooling = video_input["token_pooling"]
        num_pooled_patches = video_input["num_pooled_patches"]

        image_features_flat = self.vision_backbone(
            images=pixel_values_videos.unsqueeze(0),
            token_pooling=token_pooling.unsqueeze(0),
        )

        assert len(image_features_flat) == num_pooled_patches.sum()
        return image_features_flat.split(num_pooled_patches.tolist(), dim=0)

    def get_language_model(self) -> torch.nn.Module:
        return self.model

    def get_multimodal_embeddings(
        self,
        **kwargs: object
    ) -> MultiModalEmbeddings | None:
        modalities = self._parse_and_validate_multimodal_inputs(**kwargs)
        if not modalities:
            return []
        
        multimodal_embeddings: tuple[torch.Tensor, ...] = ()

        for modality in modalities:
            if modality == "videos":
                video_input = modalities["videos"]
                video_embeddings = self._process_video_input(video_input)
                multimodal_embeddings += video_embeddings

        return multimodal_embeddings

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
    ) -> torch.Tensor:
        inputs_embeds = self.model.get_input_embeddings(input_ids)
        if multimodal_embeddings is not None and len(multimodal_embeddings) != 0:
            assert self.img_patch_id is not None

            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                multimodal_embeddings,
                self.img_patch_id,
            )
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.LongTensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:

        if intermediate_tensors is not None:
            inputs_embeds = None

        # NOTE: In v1, inputs_embeds is always generated at model runner, this
        # condition is for v0 compatibility.
        elif inputs_embeds is None:
            vision_embeddings = self.get_multimodal_embeddings(**kwargs)
            inputs_embeds = self.get_input_embeddings(input_ids, vision_embeddings)
            if (vision_embeddings is not None) and len(vision_embeddings) != 0 and self.bi_directional_attn:
                kwargs = self.prepare_attn_masks(
                    input_ids,
                    positions,
                    mask_dtype=self.dtype,
                    **kwargs,
                )                                     
            input_ids = None

        hidden_states = self.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        return hidden_states
    
    def prepare_attn_masks(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        mask_dtype: torch.dtype,
        **kwargs: object,
    ) -> object:
        kwargs["has_images"] = True
        start_indices = (positions == 0).cpu().nonzero()
        num_seqs = len(start_indices)
        seq_lens = []
        for i in range(num_seqs):
            start_idx = start_indices[i].item()
            if i < num_seqs - 1:
                end_idx = start_indices[i + 1].item()
            else:
                end_idx = len(input_ids)
            seq_lens.append(end_idx - start_idx)
        kwargs["seq_lens"] = seq_lens

        image_tokens = self.image_tokens.to(input_ids.device)
        attn_masks = []
        start_idx = 0
        for seq_len in seq_lens:
            end_idx = start_idx + seq_len
            input_token_ids = input_ids[start_idx:end_idx]
            start_idx = end_idx
            # Create a causal mask.
            attn_mask = torch.empty(
                1,
                1,
                seq_len,
                seq_len,
                dtype=mask_dtype,
                device=input_ids.device,
            )
            attn_mask.fill_(float("-inf"))
            # Fill the lower triangle with 0.
            attn_mask = attn_mask.triu(diagonal=1)

            # Consider the bidirectional attention between image tokens.
            img_mask = torch.zeros_like(attn_mask, dtype=torch.long)
            img_pos = torch.any(input_token_ids[:, None] == image_tokens[None, :], -1)
            img_mask[:, :, :, img_pos] += 1
            img_mask[:, :, img_pos, :] += 1
            attn_mask = torch.where(img_mask == 2, 0, attn_mask)
            attn_masks.append(attn_mask)
        kwargs["attn_masks"] = attn_masks
        return kwargs


    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        logits = self.logits_processor(
            self.lm_head,
            hidden_states,
            sampling_metadata,
        )
        return logits

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):

        loader = AutoWeightsLoader(self)
        weights = _get_weights_with_merged_embedding(weights)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    def get_mm_mapping(self) -> MultiModelKeys:
        """
        Get the module prefix in multimodal models
        """
        return MultiModelKeys.from_string_field(
            language_model="model",
            connector="vision_backbone.image_projector",
            tower_model="vision_backbone",
        )


def _get_weights_with_merged_embedding(
    weights: Iterable[Tuple[str, torch.Tensor]]
) -> Iterable[Tuple[str, torch.Tensor]]:
    embedding_weights = {}
    for name, weight in weights:
        if "wte.embedding" in name:
            embedding_weights["embedding"] = weight
        elif "wte.new_embedding" in name:
            embedding_weights["new_embedding"] = weight
        else:
            yield (name, weight)
    # this is compatible with most of quantization,
    # because they won't quantize embed_tokens
    embedding_weights = torch.cat(
        [embedding_weights["embedding"], embedding_weights["new_embedding"]],
        dim=0,
    )
    yield ("model.embed_tokens.weight", embedding_weights)