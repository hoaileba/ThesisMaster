from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass
import torch
import math
from typing import Optional, List
from transformers import AutoTokenizer, AutoModel
from torch import nn
from torch import Tensor
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.processing_utils import Unpack
import torch.nn.functional as F

from decoder_model import BaseModelOutputWithPastStack, DecoderModel

def get_embedding_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, 0].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, 0]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

class SelfAttentionFusion(nn.Module):
    """
    Self-Attention Fusion Layer.
    
    Concatenates encoder and decoder outputs, then applies self-attention
    to learn interactions between them.
    
    Input: [encoder_tokens; decoder_tokens] → Self-Attention → Mean Pooling
    """
    def __init__(self, hidden_dim: int, encoder_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        # Project encoder to same dimension as decoder
        self.encoder_proj = nn.Linear(encoder_dim, hidden_dim)
        
        # Self-attention projections (Q, K, V from same source)
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        self.W_o = nn.Linear(hidden_dim, hidden_dim)
        
        # Layer norm and dropout
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        decoder_hidden: torch.Tensor,  # (batch, num_layers, decoder_dim)
        encoder_hidden: torch.Tensor,  # (batch, seq_enc, encoder_dim)
    ) -> torch.Tensor:
        """
        Args:
            decoder_hidden: Decoder output (batch, num_dec_tokens, decoder_dim)
            encoder_hidden: Encoder output (batch, seq_enc, encoder_dim)
        Returns:
            fused: (batch, num_dec_tokens + seq_enc, hidden_dim)
        """
        batch_size = decoder_hidden.shape[0]
        
        # Project encoder to same dim as decoder
        encoder_proj = self.encoder_proj(encoder_hidden)  # (batch, seq_enc, hidden_dim)
        
        # Concatenate: [encoder; decoder]
        # Shape: (batch, seq_enc + num_dec_tokens, hidden_dim)
        combined = torch.cat([encoder_proj, decoder_hidden], dim=1)
        seq_len = combined.shape[1]
        
        # Self-Attention: Q, K, V all from combined
        Q = self.W_q(combined)
        K = self.W_k(combined)
        V = self.W_v(combined)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Attention output
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        attn_output = self.W_o(attn_output)
        
        # Residual connection + LayerNorm
        output = self.layer_norm(combined + self.dropout(attn_output))
        
        return output


@dataclass
class FusionOutput(ModelOutput):
    """Output of FusionModel with fused embeddings."""
    fused_hidden_states: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[torch.FloatTensor] = None
    decoder_hidden_states: Optional[torch.FloatTensor] = None


class FusionMLP(nn.Module):
    """
    MLP: Two linear transformations with GELU activation in between.
    MLP(x) = Linear2(GELU(Linear1(x)))
    """
    def __init__(self, hidden_dim: int, expansion_factor: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * expansion_factor)
        self.fc2 = nn.Linear(hidden_dim * expansion_factor, hidden_dim)
        self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, num_layers, hidden_dim)
        Returns:
            (batch, num_layers, hidden_dim)
        """
        return self.fc2(self.activation(self.fc1(x)))


class FusionModel(nn.Module):
    """
    Fusion Model combining Encoder (RoBERTa) and Decoder (Qwen3) outputs.
    
    Architecture:
    1. Encoder: Full sequence output (batch, seq_enc, encoder_dim)
    2. Decoder: Last N layers' last tokens (batch, num_layers, decoder_dim)
    3. Self-Attention: Attend over concatenated [encoder; decoder] tokens
    4. MLP: Transform fused representations
    5. Mean Pooling: Aggregate to single embedding
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Load pretrained models
        self.decoder_model = DecoderModel.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
        self.decoder_model.to("cuda")
        self.decoder_model.eval()
        print(self.decoder_model)
        
        self.encoder_model = AutoModel.from_pretrained("AITeamVN/Vietnamese_Embedding")
        self.encoder_model.to("cuda")
        self.encoder_model.eval()
        
        # Tokenizers
        self.decoder_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
        self.encoder_tokenizer = AutoTokenizer.from_pretrained("AITeamVN/Vietnamese_Embedding")
        self.device = "cuda"
        
        # Model dimensions
        self.decoder_dim = self.decoder_model.config.hidden_size  # Qwen3 hidden size
        self.encoder_dim = self.encoder_model.config.hidden_size  # RoBERTa hidden size
        self.num_decoder_layers = 6  # Number of decoder layers to use
        
        # Self-Attention Fusion: attend over both encoder and decoder outputs
        self.self_attention = SelfAttentionFusion(
            hidden_dim=self.decoder_dim,
            encoder_dim=self.encoder_dim,
            num_heads=16,
            dropout=0.1,
        )
        
        # MLP layer after self-attention
        self.fusion_mlp = FusionMLP(
            hidden_dim=self.decoder_dim,
            expansion_factor=4,
        )

    def forward(
        self,
        texts: List[str],
    ) -> FusionOutput:
        # Tokenize inputs
        input_tokens_encoder = self.encoder_tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        input_tokens_decoder = self.decoder_tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, padding_side='left'
        ).to(self.device)

        # Get model outputs
        encoder_output = self.encoder_model(**input_tokens_encoder)
        decoder_output = self.decoder_model(**input_tokens_decoder)
        
        # Encoder: full sequence output (batch, seq_enc, encoder_dim)
        encoder_hidden = encoder_output.last_hidden_state
        
        # Decoder: last N layers' last token
        # Stack last layers: (num_layers, batch, seq_dec, decoder_dim)
        hidden_decoder_states = torch.stack(decoder_output.stack_hidden_states[-self.num_decoder_layers:])
        # Get last token of each layer: (num_layers, batch, decoder_dim)
        last_hidden_decoder_states = hidden_decoder_states[:, :, -1, :]
        # Transpose to (batch, num_layers, decoder_dim)
        decoder_hidden = last_hidden_decoder_states.transpose(0, 1)
        
        # Self-Attention Fusion
        # Input: encoder (batch, seq_enc, encoder_dim), decoder (batch, num_layers, decoder_dim)
        # Output: (batch, seq_enc + num_layers, decoder_dim)
        fused_states = self.self_attention(
            decoder_hidden=decoder_hidden,
            encoder_hidden=encoder_hidden,
        )
        
        # MLP transformation
        fused_states = self.fusion_mlp(fused_states)  # (batch, seq_enc + num_layers, decoder_dim)
        
        # Mean Pooling over all tokens to get final embedding
        fused_embedding = fused_states.mean(dim=1)  # (batch, decoder_dim)
        
        # L2 Normalize
        fused_embedding = F.normalize(fused_embedding, p=2, dim=1)
        
        # Also compute encoder CLS embedding for reference
        encoder_cls = encoder_output.last_hidden_state[:, 0]
        
        return FusionOutput(
            fused_hidden_states=fused_embedding,
            encoder_hidden_states=encoder_cls,
            decoder_hidden_states=decoder_hidden,
        )
    
    def load_checkpoint(self, path: str):
        """Load fusion layers from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        # Support both old (cross_attention) and new (self_attention) checkpoints
        if "self_attention_state_dict" in checkpoint:
            self.self_attention.load_state_dict(checkpoint["self_attention_state_dict"])
        elif "cross_attention_state_dict" in checkpoint:
            print("⚠️ Loading old cross_attention checkpoint - architecture changed!")
            # Can't load directly, skip
        
        if "fusion_mlp_state_dict" in checkpoint:
            self.fusion_mlp.load_state_dict(checkpoint["fusion_mlp_state_dict"])
        
        print(f"✓ Loaded checkpoint from {path}")
        return checkpoint.get("epoch", 0), checkpoint.get("loss", 0.0)

if __name__ == "__main__":
    model = FusionModel(config=None)
    model.to("cuda")
    model.eval()
    texts = ["Hello, how are you?", "I am fine, thank you."]
    output = model(texts)
    print("Fused embedding:", output.fused_hidden_states.shape)
    print("Encoder embedding:", output.encoder_hidden_states.shape)
    print("Decoder hidden states:", output.decoder_hidden_states.shape)