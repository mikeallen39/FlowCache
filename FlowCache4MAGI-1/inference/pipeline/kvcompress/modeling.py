import torch
from .kv_compressor import KVCompressor
from inference.model.dit.dit_module import CustomLayerNormLinear, FusedLayerNorm, PerChannelQuantizedFp8Linear, Attention
from inference.common import EngineConfig, InferenceParams, ModelConfig, ModelMetaArgs

def MagiAttention_init(
    self, model_config: ModelConfig, engine_config: EngineConfig, layer_number: int, compression_config: dict
):
    Attention.__init__(self, model_config, engine_config, layer_number)
    # super().__init__(model_config=model_config, engine_config=engine_config, layer_number=layer_number)

    # output 2x query, one for self-attn, one for cross-attn with condition
    self.linear_qkv = CustomLayerNormLinear(
        input_size=self.model_config.hidden_size,
        output_size_q=self.query_projection_size,
        output_size_kv=self.kv_projection_size,
        layer_number=self.layer_number,
        model_config=self.model_config,
        engine_config=self.engine_config,
    )

    # kv from condition, e.g., caption
    self.linear_kv_xattn = torch.nn.Linear(
        int(self.model_config.hidden_size * self.model_config.xattn_cond_hidden_ratio),  # 6144
        2 * self.kv_projection_size,  # 2048
        dtype=self.model_config.params_dtype,
        bias=False,
    )

    # Output.
    self.adapt_linear_quant = (
        self.engine_config.fp8_quant and self.layer_number != 0 and self.layer_number != model_config.num_layers - 1
    )
    submodules_linear_proj = PerChannelQuantizedFp8Linear if self.adapt_linear_quant else torch.nn.Linear
    self.linear_proj = submodules_linear_proj(
        2 * self.query_projection_size, self.model_config.hidden_size, dtype=self.model_config.params_dtype, bias=False
    )

    self.q_layernorm = FusedLayerNorm(model_config=self.model_config, hidden_size=self.hidden_size_per_attention_head)
    self.q_layernorm_xattn = FusedLayerNorm(
        model_config=self.model_config, hidden_size=self.hidden_size_per_attention_head
    )
    self.k_layernorm = FusedLayerNorm(model_config=self.model_config, hidden_size=self.hidden_size_per_attention_head)
    self.k_layernorm_xattn = FusedLayerNorm(
        model_config=self.model_config, hidden_size=self.hidden_size_per_attention_head
    )

    self.attn_weights_history = []

    # =============== New logic start ===============
    self.kv_cluster = KVCompressor(
        **compression_config["method_config"]
    )
    # =============== New logic end =================