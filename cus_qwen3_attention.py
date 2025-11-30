from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Model, check_model_inputs, DynamicCache, create_causal_mask, 
    create_sliding_window_causal_mask, TransformersKwargs, Unpack, Optional, Cache,
)
from transformers.modeling_outputs import ModelOutput
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List


@dataclass
class BaseModelOutputWithPast(ModelOutput):
    """Extended output with previous hidden state and attention mask."""
    last_hidden_state: Optional[torch.FloatTensor] = None
    previous_hidden_state: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None
    attention_mask: Optional[torch.Tensor] = None


class ScalarMixLayer(nn.Module):
    """
    Scalar mixing để tổng hợp các hidden states từ nhiều tầng.
    Hỗ trợ 3 modes:
    - 'learnable': Học trọng số cho từng tầng và chuẩn hóa bằng softmax
    - 'uniform': Trọng số đồng đều (simple averaging)
    - 'last_n': Chỉ lấy n tầng cuối với trọng số đồng đều
    """
    def __init__(
        self, 
        num_layers: int, 
        mode: str = 'learnable',
        initial_scalar: float = 0.0,
        last_n_layers: int = 4
    ):
        super().__init__()
        self.num_layers = num_layers
        self.mode = mode
        self.last_n_layers = last_n_layers
        
        if mode == 'learnable':
            # Khởi tạo trọng số cho mỗi tầng
            self.scalar_parameters = nn.Parameter(
                torch.FloatTensor([initial_scalar] * num_layers)
            )
            # Gamma để scale output
            self.gamma = nn.Parameter(torch.FloatTensor([1.0]))
        elif mode == 'uniform':
            # Trọng số đồng đều, không cần học
            self.register_buffer(
                'uniform_weights',
                torch.ones(num_layers) / num_layers
            )
        elif mode == 'last_n':
            # Chỉ lấy n tầng cuối
            weights = torch.zeros(num_layers)
            weights[-last_n_layers:] = 1.0 / last_n_layers
            self.register_buffer('last_n_weights', weights)
        else:
            raise ValueError(f"Unknown mode: {mode}. Choose from ['learnable', 'uniform', 'last_n']")
        
    def forward(self, hidden_states_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            hidden_states_list: List of tensors from each layer [h_0, h_1, ..., h_L]
                Each tensor shape: (batch_size, seq_len, hidden_size)
        Returns:
            Mixed hidden states: (batch_size, seq_len, hidden_size)
        """
        if self.mode == 'learnable':
            # Chuẩn hóa trọng số bằng softmax
            normed_weights = F.softmax(self.scalar_parameters, dim=0)
            # Tổng hợp weighted sum
            mixed_hidden_states = sum(
                weight * hidden_state 
                for weight, hidden_state in zip(normed_weights, hidden_states_list)
            )
            # Scale với gamma
            return self.gamma * mixed_hidden_states
            
        elif self.mode == 'uniform':
            # Simple averaging với trọng số đồng đều
            mixed_hidden_states = sum(
                weight * hidden_state 
                for weight, hidden_state in zip(self.uniform_weights, hidden_states_list)
            )
            return mixed_hidden_states
            
        elif self.mode == 'last_n':
            # Chỉ average n tầng cuối
            mixed_hidden_states = sum(
                weight * hidden_state 
                for weight, hidden_state in zip(self.last_n_weights, hidden_states_list)
            )
            return mixed_hidden_states


class CrossAttentionFusion(nn.Module):
    """
    Cross-Attention module sử dụng trực tiếp Q, K, V từ attention layers.
    Không thêm tham số học mới - chỉ tính toán attention scores.
    """
    def __init__(self, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        
    def forward(
        self,
        query: torch.Tensor,  # Q từ decoder attention: (batch, num_heads, query_len, head_dim)
        key: torch.Tensor,    # K từ RoBERTa: (batch, num_heads, kv_len, head_dim)
        value: torch.Tensor,  # V từ RoBERTa: (batch, num_heads, kv_len, head_dim)
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query: Q tensor đã được project từ decoder layer
            key: K tensor đã được project từ RoBERTa
            value: V tensor đã được project từ RoBERTa
            attention_mask: Optional mask
            
        Returns:
            Cross-attention output: (batch, query_len, hidden_size)
        """
        batch_size, num_heads, query_len, head_dim = query.shape
        kv_len = key.shape[2]
        
        # Scale factor
        scale = head_dim ** -0.5
        
        # Compute attention scores: Q @ K^T / sqrt(d_k)
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scale
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        # Softmax normalization
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # Weighted sum: Attention @ V
        attn_output = torch.matmul(attn_weights, value)
        
        # Reshape back: (batch, query_len, hidden_size)
        attn_output = attn_output.transpose(1, 2).contiguous()
        hidden_size = num_heads * head_dim
        attn_output = attn_output.view(batch_size, query_len, hidden_size)
        
        return attn_output


class Qwen3ModelWithFusion(Qwen3Model):
    """
    Custom Qwen3 Model với Layer-wise Fusion và Cross-Attention.
    Sử dụng Vietnamese Embedding model thay vì RoBERTa.
    """
    def __init__(
        self, 
        config, 
        embedding_model_name: str = "AITeamVN/Vietnamese_Embedding_v2",
        scalar_mix_mode: str = 'uniform',  # 'learnable', 'uniform', hoặc 'last_n'
        last_n_layers: int = 4,
    ):
        super().__init__(config)
        
        # Store config for lazy loading
        self._embedding_model_name = embedding_model_name
        self._scalar_mix_mode = scalar_mix_mode
        self._last_n_layers = last_n_layers
        self._embedding_initialized = False
        
        # Placeholder attributes - will be initialized lazily
        self.embedding_model = None
        self._embedding_base_model = None  # Renamed to avoid conflict with parent's base_model property
        self.embedding_hidden_size = None
        self.embedding_num_layers = None
        self.scalar_mix = None
        self.embedding_projection = None
        
        # Cross-Attention fusion (không có learnable parameters)
        self.cross_attention = CrossAttentionFusion(
            num_heads=config.num_attention_heads
        )
        
        # Layer norm sau cross-attention (giữ lại để ổn định)
        # self.fusion_layer_norm = nn.LayerNorm(config.hidden_size)
        
        # KHÔNG dùng gate mechanism - sử dụng simple residual connection
        # self.fusion_gate = nn.Linear(config.hidden_size * 2, config.hidden_size)
    
    def _initialize_embedding_model(self):
        """Lazy initialization of embedding model after from_pretrained completes."""
        if self._embedding_initialized:
            return
            
        # Determine the target device from existing parameters
        device = next(self.parameters()).device
        
        # Load Vietnamese Embedding model
        # SentenceTransformer wraps a transformer model, cần truy cập vào base model
        self.embedding_model = SentenceTransformer(
            self._embedding_model_name, 
            trust_remote_code=True,
            device=str(device)
        )
        
        # Lấy base transformer model từ SentenceTransformer
        # SentenceTransformer structure: [Transformer] -> [Pooling] -> [Normalize]
        self._embedding_base_model = self.embedding_model[0].auto_model
        
        self.embedding_hidden_size = self._embedding_base_model.config.hidden_size
        self.embedding_num_layers = self._embedding_base_model.config.num_hidden_layers
        
        # Scalar mixing cho embedding model layers
        self.scalar_mix = ScalarMixLayer(
            num_layers=self.embedding_num_layers + 1,  # +1 cho embedding layer
            mode=self._scalar_mix_mode,
            initial_scalar=0.0,
            last_n_layers=self._last_n_layers
        ).to(device)
        
        # Project embedding model hidden size về Qwen3 hidden size nếu khác nhau
        if self.embedding_hidden_size != self.config.hidden_size:
            self.embedding_projection = nn.Linear(
                self.embedding_hidden_size, 
                self.config.hidden_size
            ).to(device)
        else:
            self.embedding_projection = nn.Identity()
        
        self._embedding_initialized = True
        
    def encode_with_embedding_model(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode input với Vietnamese Embedding model và áp dụng layer-wise fusion.
        Trả về cả hidden states và Q, K, V từ attention layers.
        
        Returns:
            mixed_hidden_states: Weighted sum của các layers
            embedding_keys: K từ embedding model attention
            embedding_values: V từ embedding model attention
        """
        # Ensure embedding model is initialized
        self._initialize_embedding_model()
        
        # Get all hidden states từ base model
        outputs = self._embedding_base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=False,
            return_dict=True,
        )
        
        # List các hidden states từ tất cả các tầng
        all_hidden_states = outputs.hidden_states  # Tuple of (L+1) tensors
        
        # Áp dụng scalar mixing
        mixed_hidden_states = self.scalar_mix(list(all_hidden_states))
        
        # Project về Qwen3 hidden size
        projected_hidden_states = self.embedding_projection(mixed_hidden_states)
        
        # Để lấy K, V từ embedding model, ta sử dụng last hidden state
        last_hidden = outputs.last_hidden_state
        last_hidden = self.embedding_projection(last_hidden)
        
        batch_size, seq_len, hidden_size = last_hidden.shape
        num_heads = self.config.num_attention_heads
        head_dim = hidden_size // num_heads
        
        # Reshape để có format (batch, num_heads, seq_len, head_dim)
        embedding_kv = last_hidden.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        
        return projected_hidden_states, embedding_kv, embedding_kv

    @check_model_inputs
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        roberta_input_ids: Optional[torch.LongTensor] = None,
        roberta_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        
        # Standard embedding
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], 
                device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # Prepare causal mask
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Encode với Vietnamese Embedding model nếu có input
        embedding_features = None
        embedding_keys = None
        embedding_values = None
        if roberta_input_ids is not None:
            embedding_features, embedding_keys, embedding_values = self.encode_with_embedding_model(
                roberta_input_ids, 
                roberta_attention_mask
            )

        # Pass through decoder layers
        for idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            
            # Lấy hidden states từ decoder layer
            hidden_states = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs
            
            # Áp dụng cross-attention fusion sau một số layers nhất định
            # Ví dụ: sau layer thứ config.num_hidden_layers // 2
            if embedding_keys is not None and idx == self.config.num_hidden_layers // 2:
                # Lấy Q từ hidden_states của decoder
                batch_size, seq_len, hidden_size = hidden_states.shape
                num_heads = self.config.num_attention_heads
                head_dim = hidden_size // num_heads
                
                # Reshape hidden_states thành format multi-head
                query = hidden_states.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
                
                # Cross-attention: query từ decoder, key/value từ embedding model
                cross_attn_output = self.cross_attention(
                    query=query,
                    key=embedding_keys,
                    value=embedding_values,
                    attention_mask=None,
                )
                
                # Simple residual connection (không dùng gate)
                # Alpha = 0.5 để balance giữa decoder và cross-attention
                alpha = 0.5
                hidden_states = alpha * hidden_states + (1 - alpha) * cross_attn_output
                hidden_states = self.norm(hidden_states)

        # Final layer norm
        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            attention_mask=attention_mask,
        )


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """Pool last valid token from sequences."""
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device), 
            sequence_lengths
        ]


def get_detailed_instruct(task_description: str, query: str) -> str:
    """Format instruction with query."""
    return f'Instruct: {task_description}\nQuery: {query}'



def get_embedding(model, tokenizer, embedding_tokenizer, text):
    qwen_inputs = tokenizer(text, return_tensors="pt", padding=True)
    qwen_inputs = {k: v.to("cuda") for k, v in qwen_inputs.items()}
    
    # Tokenize cho Vietnamese Embedding model
    embedding_inputs = embedding_tokenizer(
        text, 
        return_tensors="pt", 
        padding=True,
        truncation=True,
        max_length=256
    )
    embedding_inputs = {k: v.to("cuda") for k, v in embedding_inputs.items()}
    
    # Forward pass với fusion
    print("\nRunning model with fusion...")
    with torch.no_grad():
        outputs = model(
            input_ids=qwen_inputs["input_ids"],
            attention_mask=qwen_inputs["attention_mask"],
            roberta_input_ids=embedding_inputs["input_ids"],  # Tên parameter giữ nguyên
            roberta_attention_mask=embedding_inputs["attention_mask"],
        )
    
    # Pool last token để có embedding
    embeddings = last_token_pool(
        outputs.last_hidden_state, 
        qwen_inputs["attention_mask"]
    )
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings
# ============================================================================
# EXAMPLE USAGE
# ============================================================================
if __name__ == "__main__":
    # Initialize models
    print("Loading models...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
    
    # Tokenizer cho Vietnamese Embedding model
    # SentenceTransformer có tokenizer riêng
    embedding_tokenizer = AutoTokenizer.from_pretrained("AITeamVN/Vietnamese_Embedding_v2")
    
    # Load custom model with fusion
    model = Qwen3ModelWithFusion.from_pretrained(
        "Qwen/Qwen3-Embedding-0.6B",
        embedding_model_name="AITeamVN/Vietnamese_Embedding_v2",
        scalar_mix_mode='uniform',  # Sử dụng uniform averaging
        last_n_layers=4  # Chỉ dùng khi mode='last_n'
    )
    model.to("cuda")
    model.eval()
    
    # Prepare inputs
    instruction = "Tìm kiếm thông tin trả lời câu hỏi sau:"
    question = "Pháp nhân có được hưởng di sản thừa kế từ chủ sở hữu đã chết hay không"
    ans = "bộ luật dân sự\nheader: Thừa kế theo pháp luật\nThừa kế theo pháp luật áp dụng khi không có di chúc hợp pháp, những người thừa kế theo di chúc chết trước hoặc cơ quan, tổ chức hưởng di chúc không còn tồn tại. Đồng thời, nó cũng áp dụng cho các phần di sản không được định đoạt trong di chúc, liên quan đến người không có quyền hưởng di sản, hoặc cơ quan, tổ chức không còn tồn tại.\n2. Thừa kế theo pháp luật cũng được áp dụng đối với các phần di sản sau đây:\nc) Phần di sản có liên quan đến người được thừa kế theo di chúc nhưng họ không có quyền hưởng di sản, từ chối nhận di sản, chết trước hoặc chết cùng thời điểm với người lập di chúc; liên quan đến cơ quan, tổ chức được hưởng di sản theo di chúc, nhưng không còn tồn tại vào thời điểm mở thừa kế.\n"
    true_ans =  "bộ luật dân sự\nheader: Cá nhân có quyền lập di chúc và hưởng di sản\nCá nhân có quyền lập di chúc để định đoạt tài sản của mình; để lại tài sản của mình cho người thừa kế theo pháp luật; hưởng di sản theo di chúc hoặc theo pháp luật.Người thừa kế không là cá nhân có quyền hưởng di sản theo di chúc."

    
    instruct_text = get_detailed_instruct(instruction, question)
    
    # Tokenize cho Qwen3
    question_embedding = get_embedding(model, tokenizer, embedding_tokenizer, instruct_text)


    # print(embeddings)
    ans_embedding = get_embedding(model, tokenizer, embedding_tokenizer, ans)
    true_ans_embedding = get_embedding(model, tokenizer, embedding_tokenizer, true_ans)
    similarity_true = F.cosine_similarity(question_embedding, true_ans_embedding, dim=1)
    similarity_false = F.cosine_similarity(question_embedding, ans_embedding, dim=1)
    print("similarity question and true answer", similarity_true)
    print("similarity question and false answer", similarity_false)
    print(ans_embedding.shape)