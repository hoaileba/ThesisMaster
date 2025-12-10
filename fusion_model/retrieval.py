import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from DatabaseConnector.milvus.milvus_connector import MilvusConnector
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from fusion_model import FusionModel
# embedding_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
embedding_model = FusionModel(config=None)
embedding_model.load_checkpoint("checkpoints_bkai/best_model.pt")
embedding_model.to("cuda")
embedding_model.eval()

milvus_connector = MilvusConnector(
    uri="http://milvus-standalone-legalrag:19530",
    token="root:Milvus",
    db_name="Master",
)
# collection_name = "vlsp_chunks_qwen3"
collection_name = "vlsp_chunks_qwen3_fusion_mlp"

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery:{query}'
def filter_unique_aids(results,top_k: int = 10):
        seen_aids = set()
        unique_aids = []
        for result in results:
            aid = result["aid"]
            if aid not in seen_aids:
                seen_aids.add(aid)
                unique_aids.append(aid)
        return unique_aids[:top_k]
def search(query: str, limit: int = 1000):
    query = get_detailed_instruct("Tìm kiếm thông tin trả lời câu hỏi sau:", query)
    # embedding = embedding_model.encode([query]).tolist()[0]
    with torch.no_grad():
        embedding = embedding_model(texts=[query]).fused_hidden_states.tolist()[0]
    results = milvus_connector.search_records(
        collection_name=collection_name,
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

def calculate_metrics_from_json(data, k_values=[1, 3, 5, 10]):
    """
    Tính metrics từ format dữ liệu List of Dicts của user.
    Input format:
    [
        {"qid": ..., "pred_aids": [...], "label_aids": [...]},
        ...
    ]
    """
    # 1. Tách dữ liệu ra khỏi cấu trúc Dictionary
    predictions = [item['pred_aids'] for item in data]
    ground_truths = [item['label_aids'] for item in data]
    
    # Khởi tạo biến lưu kết quả
    metrics = {k: {'precision': 0.0, 'recall': 0.0, 'mrr': 0.0} for k in k_values}
    num_samples = len(data)
    
    print(f"Đang tính toán trên {num_samples} samples...")

    # 2. Loop tính toán
    for preds, truths in zip(predictions, ground_truths):
        truth_set = set(truths)
        num_relevant_total = len(truth_set)
        
        # Bỏ qua nếu không có label (dữ liệu lỗi)
        if num_relevant_total == 0:
            continue
            
        for k in k_values:
            # Lấy top k dự đoán
            top_k_preds = preds[:k]
            
            # Kiểm tra hit
            hits = [1 if p in truth_set else 0 for p in top_k_preds]
            num_hits = sum(hits)
            
            # --- Precision@K ---
            metrics[k]['precision'] += num_hits / k
            
            # --- Recall@K ---
            metrics[k]['recall'] += num_hits / num_relevant_total
            
            # --- MRR@K ---
            # Tìm vị trí đúng đầu tiên (1-based index)
            try:
                first_hit_index = hits.index(1) # trả về index đầu tiên tìm thấy số 1
                metrics[k]['mrr'] += 1.0 / (first_hit_index + 1)
            except ValueError:
                metrics[k]['mrr'] += 0.0

    # 3. Tính trung bình (Average)
    final_results = {}
    for k in k_values:
        final_results[f'Precision@{k}'] = metrics[k]['precision'] / num_samples
        final_results[f'Recall@{k}'] = metrics[k]['recall'] / num_samples
        final_results[f'MRR@{k}'] = metrics[k]['mrr'] / num_samples
        
    return final_results
if __name__ == "__main__":
    path_data = "../../data/vlsp/data/train.json"
    retrieval_results = []
    with open(path_data, "r", encoding="utf-8") as f:
        data = json.load(f)
    for item in tqdm(data, desc="Processing data"):
        query = item["question"]
        results = search(query)
        unique_aids = filter_unique_aids(results)
        retrieval_results.append({
            "qid": item["id"],
            "pred_aids": unique_aids,
            "label_aids": item["relevant_laws"]
        })
    metrics = calculate_metrics_from_json(retrieval_results)
    print(metrics)
    with open("retrieval_results_qwen3_fusion_mlp.json", "w", encoding="utf-8") as f:
        json.dump(retrieval_results, f, ensure_ascii=False, indent=4)
    with open("metrics_qwen3_fusion_mlp.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)