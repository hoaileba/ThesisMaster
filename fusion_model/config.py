from transformers.configuration_utils import PretrainedConfig
class FusionModelConfig(PretrainedConfig):
    def __init__(self, config):
        super().__init__(config)
        self.decoder_model_name = config.get("decoder_model_name", "Qwen/Qwen3-Embedding-0.6B")
        self.encoder_model_name = config.get("encoder_model_name", "AITeamVN/Vietnamese_Embedding_v2")
        self.decoder_tokenizer_name = config.get("decoder_tokenizer_name", "Qwen/Qwen3-Embedding-0.6B")
        self.encoder_tokenizer_name = config.get("encoder_tokenizer_name", "AITeamVN/Vietnamese_Embedding_v2")
        self.device = config.get("device", "cuda")
        self.num_layers = config.get("num_layers", 12)
        self.num_heads = config.get("num_heads", 12)
        self.hidden_size = config.get("hidden_size", 768)
        self.num_attention_heads = config.get("num_attention_heads", 12)
        self.num_hidden_layers = config.get("num_hidden_layers", 12)