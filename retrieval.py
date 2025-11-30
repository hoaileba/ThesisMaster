import json
import torch
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from DatabaseConnector.milvus.milvus_connector import MilvusConnector
from tqdm import tqdm
from qwen3_model_cus import Qwen3ModelCustom
from qwen3_model_cus import last_token_pool, add_new_token_mask
import os
import torch.nn.functional as F
os.environ["HF_HUB_OFFLINE"] = "1"
# cache_foler = "~/.cache/huggingface/hub"
os.environ['HF_HOME'] = "~/.cache/huggingface/"
class EmbeddingService:
    def __init__(self):
        # self.qwen_client = OpenAI(
        #     api_key="Oid4IIeJO75mKlbomJfrma8SpNwMAsMp",
        #     base_url="https://api.deepinfra.com/v1/openai"
        # )
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
        self.qwen3_model = Qwen3ModelCustom.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
        self.aiteam_model = SentenceTransformer("AITeamVN/Vietnamese_Embedding_v2", device="cuda")
    
    def get_qwen_embedding(self, text: str):
        response = self.qwen_client.embeddings.create(
            model="Qwen/Qwen3-Embedding-8B",
            input=text,
            encoding_format="float"
        )
        return response.data[0].embedding
    
    def get_aiteam_embedding(self, text: str):
        embeddings = self.aiteam_model.encode([text], convert_to_tensor=False)
        return embeddings.tolist()[0]
    
    def get_combined_embedding(self, text: str):
        with torch.no_grad():
            input = self.tokenizer(text, return_tensors="pt")
            input = input.to(self.qwen3_model.device)
            semantic_embeddings = self.aiteam_model.encode([text], convert_to_tensor=False)
            new_attention_mask = add_new_token_mask(input)
            input["attention_mask"] = new_attention_mask
            semantic_embeddings = torch.from_numpy(semantic_embeddings)
            semantic_embeddings = semantic_embeddings.to(self.qwen3_model.device)
            embedding = self.qwen3_model(**input, semantic_embeddings=semantic_embeddings.unsqueeze(1))
            embedding = last_token_pool(embedding.last_hidden_state, input["attention_mask"])
            embedding = F.normalize(embedding, p=2, dim=1)
            torch.cuda.empty_cache()
            return embedding.tolist()[0]


class RerankerService:
    def __init__(self, device="cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Reranker-4B", padding_side='left')
        self.model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-Reranker-4B").eval().to(device)
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.max_length = 8192
        
        prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(suffix, add_special_tokens=False)
        self.task = 'Given a web search query, retrieve relevant passages that answer the query'
    
    def format_instruction(self, query, doc):
        return f"<Instruct>: {self.task}\n<Query>: {query}\n<Document>: {doc}"
    
    def process_inputs(self, pairs):
        inputs = self.tokenizer(
            pairs, padding=False, truncation='longest_first',
            return_attention_mask=False, max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        )
        for i, ele in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = self.prefix_tokens + ele + self.suffix_tokens
        inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=self.max_length)
        for key in inputs:
            inputs[key] = inputs[key].to(self.model.device)
        return inputs
    
    @torch.no_grad()
    def compute_logits(self, inputs):
        batch_scores = self.model(**inputs).logits[:, -1, :]
        true_vector = batch_scores[:, self.token_true_id]
        false_vector = batch_scores[:, self.token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp().tolist()
        return scores
    
    def rerank(self, query, documents, batch_size=1):
        pairs = [self.format_instruction(query, doc) for doc in documents]
        all_scores = []
        
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i + batch_size]
            inputs = self.process_inputs(batch_pairs)
            scores = self.compute_logits(inputs)
            all_scores.extend(scores)
        
        return all_scores


class SearchService:
    def __init__(self):
        self.milvus = MilvusConnector(
            uri="http://localhost:19630",
            token="root:Milvus",
            db_name="Master"
        )
        self.collection_name = "vlsp_chunks_qwen3_cus"
    
    def search(self, embedding, limit=500):
        results = self.milvus.search_records(
            collection_name=self.collection_name,
            query_vector=embedding,
            vector_field="vector_content",
            limit=limit,
            output_fields=["aid","header","summary", "content", "law_id"],
            search_params={
                "M": 128,
                "efConstruction": 200
            }
        )
        return results
    
    def filter_unique_aids(self, results):
        seen_aids = set()
        unique_aids = []
        for result in results:
            aid = result["aid"]
            if aid not in seen_aids:
                seen_aids.add(aid)
                unique_aids.append(aid)
        return unique_aids


def main():
    embedding_service = EmbeddingService()
    search_service = SearchService()
    # reranker_service = RerankerService()
    
    with open("../tot_nghiep/vlsp/data/private_test.json", "r", encoding="utf-8") as f:
        questions = json.load(f)
    
    results = []
    for item in tqdm(questions, desc="Processing questions"):
        qid = item["qid"]
        question = item["question"]
        detail_results = []
        # embedding = embedding_service.get_aiteam_embedding(question)
        embedding = embedding_service.get_combined_embedding(question)
        search_results = search_service.search(embedding, limit=2000)
        for result in search_results:
            detail_results.append({
                "aid": result["aid"],
                "law_id": result["law_id"],
                "header": result["header"],
                "summary": result["summary"],
                "content": result["content"],
                "distance": result["distance"]
            })
        # documents = [result["text"] for result in search_results]
        # scores = reranker_service.rerank(question, documents)
        
        # ranked_results = sorted(zip(search_results, scores), key=lambda x: x[1], reverse=True)
        # top_50_results = [result for result, score in ranked_results[:50]]
        # unique_aids = search_service.filter_unique_aids(search_results)
        
        results.append({
            "qid": qid,
            # "relevant_laws": unique_aids
            "relevant_laws_detail": detail_results
        })
    
    with open("results_combined.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()

