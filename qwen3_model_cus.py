from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Model, check_model_inputs, auto_docstring, DynamicCache, create_causal_mask, create_sliding_window_causal_mask,
    TransformersKwargs, Unpack, Optional, Cache, 
)
from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass
import torch
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F

from torch import Tensor

@dataclass
class BaseModelOutputWithPast(ModelOutput):
    """
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.

            If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
            hidden_size)` is output.
        past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

            Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
            `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
            input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: Optional[torch.FloatTensor] = None
    previous_hidden_state: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None
    attention_mask: Optional[torch.Tensor] = None

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery:{query}'
def add_new_token_mask(batch_input):
    new_length = batch_input.input_ids.shape[1] + 1
    new_attention_mask = []
    for element in batch_input.attention_mask:
        ones_count = element.sum()
        mask_new = torch.ones(ones_count + 1)
        padding_mask = torch.zeros(new_length - ones_count - 1)

        new_attention_mask.append(torch.cat([mask_new, padding_mask]))
    new_attention_mask = torch.stack(new_attention_mask)
    return new_attention_mask

def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

class Qwen3ModelCustom(Qwen3Model):
    def __init__(self, config):
        super().__init__(config)

    @check_model_inputs
    def forward(
        self,
        semantic_embeddings: Optional[torch.FloatTensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            # The sliding window alternating layers are not always activated depending on the config
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Step 1: Run all decoder layers like original Qwen3Model
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        # Step 2: If semantic_embeddings provided, add to end and run one more decoder layer
        if semantic_embeddings is not None:
            # Add semantic_embeddings at the end of hidden_states
            next_hidden_states = torch.cat([hidden_states, semantic_embeddings], dim=1)
            
            # Update position_ids and cache_position for the extended sequence
            extended_seq_len = next_hidden_states.shape[1]
            extended_cache_position = torch.arange(
                0, extended_seq_len, device=next_hidden_states.device
            )
            extended_position_ids = extended_cache_position.unsqueeze(0)
            
            # Extend attention_mask to include semantic_embeddings tokens
            semantic_token_count = semantic_embeddings.shape[1]
            if attention_mask is not None:
                # Add 1s for semantic_embeddings tokens (they should be attended to)
                extended_attention_mask = torch.cat([
                    attention_mask,
                    torch.ones(attention_mask.shape[0], semantic_token_count, device=attention_mask.device, dtype=attention_mask.dtype)
                ], dim=1)
            else:
                extended_attention_mask = None
            
            # Create new position embeddings for extended sequence
            extended_position_embeddings = self.rotary_emb(next_hidden_states, extended_position_ids)
            
            # Create new causal mask for extended sequence
            extended_mask_kwargs = {
                "config": self.config,
                "input_embeds": next_hidden_states,
                "attention_mask": extended_attention_mask,
                "cache_position": extended_cache_position,
                "past_key_values": None,
                "position_ids": extended_position_ids,
            }
            extended_causal_mask = create_causal_mask(**extended_mask_kwargs)
            
            # Run one more decoder layer (use the last layer)
            last_decoder_layer = self.layers[-1]
            next_hidden_states = last_decoder_layer(
                next_hidden_states,
                attention_mask=extended_causal_mask,
                position_ids=extended_position_ids,
                past_key_values=None,  # Don't use cache for the extra layer
                use_cache=False,
                cache_position=extended_cache_position,
                position_embeddings=extended_position_embeddings,
                **kwargs,
            )

        next_hidden_states = self.norm(next_hidden_states)
        print("next_hidden_states: ", next_hidden_states.shape)
        print("hidden_states: ", hidden_states.shape)
        print("extended_attention_mask: ", extended_attention_mask.shape)
        # print("past_key_values: ", past_key_values.get_seq_length())
        # print("use_cache: ", use_cache)
        # print("cache_position: ", cache_position.shape)
        # print("position_ids: ", position_ids.shape)
        # print("inputs_embeds: ", inputs_embeds.shape)
        # print("kwargs: ", kwargs)

        return BaseModelOutputWithPast(
            last_hidden_state=next_hidden_states,
            previous_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            attention_mask=extended_attention_mask,
        )

if __name__ == "__main__":
    embedding_model = SentenceTransformer("AITeamVN/Vietnamese_Embedding_v2", trust_remote_code=True)
    org_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
    # print(org_model)
    # print(embedding_model)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
    model = Qwen3ModelCustom.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
    # print(model)
    model.to("cuda")

    model.eval()
    instruction = "Tìm kiếm thông tin trả lời câu hỏi sau:"
    question = "Pháp nhân có được hưởng di sản thừa kế từ chủ sở hữu đã chết hay không"
    ans = "bộ luật dân sự\nheader: Thừa kế theo pháp luật\nThừa kế theo pháp luật áp dụng khi không có di chúc hợp pháp, những người thừa kế theo di chúc chết trước hoặc cơ quan, tổ chức hưởng di chúc không còn tồn tại. Đồng thời, nó cũng áp dụng cho các phần di sản không được định đoạt trong di chúc, liên quan đến người không có quyền hưởng di sản, hoặc cơ quan, tổ chức không còn tồn tại.\n2. Thừa kế theo pháp luật cũng được áp dụng đối với các phần di sản sau đây:\nc) Phần di sản có liên quan đến người được thừa kế theo di chúc nhưng họ không có quyền hưởng di sản, từ chối nhận di sản, chết trước hoặc chết cùng thời điểm với người lập di chúc; liên quan đến cơ quan, tổ chức được hưởng di sản theo di chúc, nhưng không còn tồn tại vào thời điểm mở thừa kế.\n"
    true_ans =  "bộ luật dân sự\nheader: Cá nhân có quyền lập di chúc và hưởng di sản\nCá nhân có quyền lập di chúc để định đoạt tài sản của mình; để lại tài sản của mình cho người thừa kế theo pháp luật; hưởng di sản theo di chúc hoặc theo pháp luật.Người thừa kế không là cá nhân có quyền hưởng di sản theo di chúc."

    instruct_text = get_detailed_instruct(instruction, question)
    # Embedding the question and the answer using qwen3 model
    org_embedding_question = org_model.encode([instruct_text])
    org_embedding_answer = org_model.encode([true_ans])

    # Embedding the question and the answer using qwen3 model custom
    test_input = tokenizer(instruct_text, return_tensors="pt")   
    test_input_ans = tokenizer(true_ans, return_tensors="pt")
    test_input_str = tokenizer.decode(test_input.input_ids[0])
    print(test_input_str)
    # test_input["attention_mask"] = add_new_token_mask(test_input)
    # test_input_ans["attention_mask"] = add_new_token_mask(test_input_ans)
    # print(test_input.input_ids.shape)
    # print(test_input.attention_mask.shape)
    semantic_embeddings_question = embedding_model.encode([question])
    semantic_embeddings_answer = embedding_model.encode([true_ans])

    semantic_embeddings_torch_question = torch.from_numpy(semantic_embeddings_question)
    semantic_embeddings_torch_answer = torch.from_numpy(semantic_embeddings_answer)

    semantic_embeddings_torch_question = semantic_embeddings_torch_question.to("cuda")
    semantic_embeddings_torch_answer = semantic_embeddings_torch_answer.to("cuda")
    test_input = test_input.to("cuda")
    test_input_ans = test_input_ans.to("cuda")

    output_question = model(**test_input, semantic_embeddings=semantic_embeddings_torch_question.unsqueeze(1))
    output_answer = model(**test_input_ans, semantic_embeddings=semantic_embeddings_torch_answer.unsqueeze(1))

    embedding_question = last_token_pool(output_question.last_hidden_state, output_question.attention_mask)
    embedding_answer = last_token_pool(output_answer.last_hidden_state, output_answer.attention_mask)

    embedding_question = F.normalize(embedding_question, p=2, dim=1)
    embedding_answer = F.normalize(embedding_answer, p=2, dim=1)

    # print(type(embedding_question))
    # print(embedding_question.shape)
    # embedding_question = torch.from_numpy(embedding_question).to("cuda")
    # embedding_answer = torch.from_numpy(embedding_answer).to("cuda")

    # print(embedding_question.tolist())
    # print(embedding_answer.tolist())

    # calculate the cosine similarity between the embedding and the org_embedding
    org_embedding_question = torch.from_numpy(org_embedding_question).to("cuda")
    org_embedding_answer = torch.from_numpy(org_embedding_answer).to("cuda")
    # print(type(org_embedding_question))
    # print(org_embedding_question.shape)

    semantic_embeddings_question = torch.from_numpy(semantic_embeddings_question).to("cuda")
    semantic_embeddings_answer = torch.from_numpy(semantic_embeddings_answer).to("cuda")

    cosine_similarity_question = F.cosine_similarity(org_embedding_question, org_embedding_answer, dim=1)
    print("cosine_similarity org model", cosine_similarity_question)

    cosine_encoder_only = F.cosine_similarity(semantic_embeddings_question, semantic_embeddings_answer, dim=1)
    print("cosine_similarity encoder only", cosine_encoder_only)


    # print(embedding_question.shape)
    # print(embedding_answer.shape)
    cosine_custom_only = F.cosine_similarity(embedding_question, embedding_answer, dim=1)
    print("cosine_similarity custom only", cosine_custom_only)