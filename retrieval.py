import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from DatabaseConnector.milvus.milvus_connector import MilvusConnector
from tqdm import tqdm
from cus_qwen3_attention import Qwen3ModelWithFusion, get_embedding

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ['HF_HOME'] = "~/.cache/huggingface/"
class EmbeddingService:
    def __init__(self, device="cuda"):
        # Load Qwen3 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
        
        # Load Vietnamese Embedding tokenizer
        self.embedding_tokenizer = AutoTokenizer.from_pretrained("AITeamVN/Vietnamese_Embedding_v2")
        
        # Load custom Qwen3 model with fusion
        self.model = Qwen3ModelWithFusion.from_pretrained(
            "Qwen/Qwen3-Embedding-0.6B",
            embedding_model_name="AITeamVN/Vietnamese_Embedding_v2",
            scalar_mix_mode='uniform',
            last_n_layers=4
        )
        self.model.to(device)
        self.model.eval()
    
    def get_embedding(self, text: str):
        """Get embedding using Qwen3ModelWithFusion with cross-attention."""
        embedding = get_embedding(
            model=self.model,
            tokenizer=self.tokenizer,
            embedding_tokenizer=self.embedding_tokenizer,
            text=text
        )
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
            uri="http://103.253.20.30:19630",
            token="root:Milvus",
            db_name="Master"
        )
        self.collection_name = "vlsp_chunks_qwen3_cus_attention"
    
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
        embedding = embedding_service.get_embedding(question)
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

