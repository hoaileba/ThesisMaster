from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
import torch
from torch import Tensor
tokenizer = AutoTokenizer.from_pretrained("AITeamVN/Vietnamese_Embedding")
model = AutoModel.from_pretrained("AITeamVN/Vietnamese_Embedding")
sentence_transformer = SentenceTransformer("AITeamVN/Vietnamese_Embedding")
def get_embedding_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, 0].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, 0]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

print(tokenizer)
print(model)

text = "Hello, how are you?"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
# print(outputs)
print(outputs.last_hidden_state.shape)
print(sentence_transformer.encode([text]))
print(F.normalize(outputs.last_hidden_state[:,0], p=2, dim=1))
print(get_embedding_token_pool(outputs.last_hidden_state, inputs.attention_mask))