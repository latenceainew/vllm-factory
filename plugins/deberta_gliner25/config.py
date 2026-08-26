"""GLiNER 2.5 vLLM config — DeBERTa encoder + inference boundary head."""

from transformers import PretrainedConfig


class GLiNER25Config(PretrainedConfig):
    """vLLM-facing config for the boundary plugin.

    Encoder fields match ``GLiNER2Config``. ``boundary_head`` is the inference
    subset of GLiNER2 ``BoundaryHeadSettings`` (no loss weights / warmup).
    """

    model_type = "gliner25"

    def __init__(
        self,
        num_hidden_layers: int = 0,
        num_attention_heads: int = 1,
        hidden_size: int = 768,
        encoder_model_name: str = "microsoft/mdeberta-v3-base",
        vocab_size: int = 250112,
        encoder_hidden_size: int = 768,
        encoder_num_hidden_layers: int = 12,
        encoder_num_attention_heads: int = 12,
        encoder_intermediate_size: int = 3072,
        encoder_hidden_act: str = "gelu",
        encoder_hidden_dropout_prob: float = 0.0,
        encoder_attention_probs_dropout_prob: float = 0.0,
        encoder_max_position_embeddings: int = 512,
        encoder_type_vocab_size: int = 0,
        encoder_layer_norm_eps: float = 1e-7,
        encoder_relative_attention: bool = True,
        encoder_max_relative_positions: int = -1,
        encoder_position_buckets: int = 256,
        encoder_pos_att_type: list | None = None,
        encoder_share_att_key: bool = True,
        encoder_norm_rel_ebd: str = "layer_norm",
        encoder_position_biased_input: bool = False,
        encoder_pad_token_id: int = 0,
        token_pooling: str = "first",
        boundary_head: dict | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_hidden_layers = 0
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.encoder_model_name = encoder_model_name
        self.vocab_size = vocab_size
        self.encoder_hidden_size = encoder_hidden_size
        self.encoder_num_hidden_layers = encoder_num_hidden_layers
        self.encoder_num_attention_heads = encoder_num_attention_heads
        self.encoder_intermediate_size = encoder_intermediate_size
        self.encoder_hidden_act = encoder_hidden_act
        self.encoder_hidden_dropout_prob = encoder_hidden_dropout_prob
        self.encoder_attention_probs_dropout_prob = encoder_attention_probs_dropout_prob
        self.encoder_max_position_embeddings = encoder_max_position_embeddings
        self.encoder_type_vocab_size = encoder_type_vocab_size
        self.encoder_layer_norm_eps = encoder_layer_norm_eps
        self.encoder_relative_attention = encoder_relative_attention
        self.encoder_max_relative_positions = encoder_max_relative_positions
        self.encoder_position_buckets = encoder_position_buckets
        self.encoder_pos_att_type = encoder_pos_att_type or ["p2c", "c2p"]
        self.encoder_share_att_key = encoder_share_att_key
        self.encoder_norm_rel_ebd = encoder_norm_rel_ebd
        self.encoder_position_biased_input = encoder_position_biased_input
        self.encoder_pad_token_id = encoder_pad_token_id
        self.token_pooling = token_pooling
        self.boundary_head = dict(boundary_head or {})
