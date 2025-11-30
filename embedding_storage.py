from DatabaseConnector.milvus.milvus_connector import MilvusConnector
from tqdm import tqdm
from glob import glob
import json
from pymilvus import CollectionSchema, FieldSchema, DataType
from sentence_transformers import SentenceTransformer
import asyncio
from DatabaseConnector.milvus.milvus_connector import MilvusConnector
from pymilvus import CollectionSchema, FieldSchema, DataType
from tqdm import tqdm
from qwen3_model_cus import Qwen3ModelCustom, last_token_pool
from transformers import AutoTokenizer
import torch
import torch.nn.functional as F

milvus_connector = MilvusConnector(
    uri="http://localhost:19630",  # type: ignore
    token="root:Milvus",  # type: ignore
    db_name="Master",  # type: ignore
)
doc_collection = "documents"
chunks_collection = "vlsp_chunks_qwen3_cus"
qwen3_chunks_collection = "vlsp_chunks_qwen3"

document_schema = CollectionSchema(
    fields=[
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="aid", dtype=DataType.INT64, max_length=65535),
        FieldSchema(name="law_id", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535, enable_analyzer=True),
        FieldSchema(name="header", dtype=DataType.VARCHAR, max_length=65535, enable_analyzer=True),
        FieldSchema(name="summary", dtype=DataType.VARCHAR, max_length=65535, enable_analyzer=True),
        FieldSchema(name="augment", dtype=DataType.VARCHAR, max_length=65535, enable_analyzer=True, nullable=True),
        FieldSchema(name="vector_content", dtype=DataType.FLOAT_VECTOR, dim=1024),
        # FieldSchema(name="vector_summary", dtype=DataType.FLOAT_VECTOR, dim=1024),
        # FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=5120),
    ],
    enable_dynamic_field=True,
)


index_params = {
    "index_type": "HNSW",
    "metric_type": "COSINE",
    "params": {
        "M": 128,
        "efConstruction": 360
    }
}
milvus_connector.create_collection(
    collection_name=chunks_collection,
    schema=document_schema,
    index_params=index_params,
    vector_field=["vector_content"],  
)

model_embedding = SentenceTransformer("AITeamVN/Vietnamese_Embedding_v2", device="cuda")
# model_embedding_summary = SentenceTransformer("AITeamVN/Vietnamese_Embedding_v2", device="cuda:0")

qwen3_model = Qwen3ModelCustom.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
qwen3_model.to("cuda")
# qwen3_model_summary = Qwen3ModelCustom.from_pretrained("Qwen/Qwen3-Embedding-0.6B", device="cuda:0")

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
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

def create_embeddings_batch(texts, batch_size=64, model= SentenceTransformer("AITeamVN/Vietnamese_Embedding_v2", device="cuda")):
    """Create embeddings in batches for better performance using SentenceTransformers"""
    embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Creating embeddings"):
        with torch.no_grad():
            batch = texts[i:i + batch_size]
            # try:
                # Create embeddings for the batch using SentenceTransformers
            batch_input = tokenizer(batch, padding=True, return_tensors="pt")
            
            batch_embeddings = model.encode(batch)
            batch_embeddings = torch.from_numpy(batch_embeddings)
            batch_embeddings = batch_embeddings.to("cuda")

            new_attention_mask = add_new_token_mask(batch_input)
            batch_input["attention_mask"] = new_attention_mask
            batch_input = batch_input.to("cuda")
            embeddings_qwen3 = qwen3_model(**batch_input, semantic_embeddings=batch_embeddings.unsqueeze(1))
            embedding = last_token_pool(embeddings_qwen3.last_hidden_state, batch_input["attention_mask"])
            embedding = F.normalize(embedding, p=2, dim=1)
            embeddings.extend(embedding.tolist())
            torch.cuda.empty_cache()
        # except Exception as e:
        #     print(f"Error creating embeddings for batch {i//batch_size + 1}: {e}")
        #     # Fallback: create embeddings one by one for this batch
        #     for text in batch:
        #         try:
        #             embedding = model.encode([text], convert_to_tensor=False)
        #             embeddings.append(embedding[0].tolist())
        #         except Exception as fallback_e:
        #             print(f"Error creating embedding for individual text: {fallback_e}")
        #             # Use a zero vector as fallback
        #             embeddings.append([0.0] * 1024)
    
    return embeddings
async def async_create_embeddings_batch(texts, batch_size=64, model= SentenceTransformer("AITeamVN/Vietnamese_Embedding_v2", device="cuda")):
    # Chạy hàm sync trong thread riêng
    return await asyncio.to_thread(create_embeddings_batch, texts, batch_size, model)
def prepare_texts_for_embedding(data, mapping, mapping_docname):
    """Prepare all texts that need embedding"""
    content_texts = []
    summary_texts = []
    items_with_header_summary = []
    items_without_header_summary = []
    
    for item in data:
        law_id = item.get("law_id", "")
        aug = mapping.get(item["aid"], {})
        law_name = mapping_docname.get(law_id, {}).get("name", "")
        law_purpose = mapping_docname.get(law_id, {}).get("purpose", "")
        header = aug.get("header", "")
        summary = aug.get("summary", "")
        
        item['header'] = header
        item['summary'] = summary
        item["augment"] = ""
        
        if header and summary:
            content = law_name + "\n" + law_purpose + "\n" + header + "\n" + item["content"]
            summary_content = law_name + "\n" + law_purpose + "\n" + summary
            content_texts.append(content)
            summary_texts.append(summary_content)
            items_with_header_summary.append(item)
        else:
            content_texts.append(item["content"])
            summary_texts.append(item["content"])  # Use same text for both vectors
            item["header"] = ""
            item["summary"] = ""
            items_without_header_summary.append(item)
    
    return content_texts, summary_texts, items_with_header_summary, items_without_header_summary

with open("../tot_nghiep/vlsp/data/train.json", "r", encoding='utf-8') as f:
    train_data = json.load(f)
    
with open("../tot_nghiep/vlsp/data/mapping_docname_norm.json", "r", encoding='utf-8') as f:
    mapping_docname = json.load(f)
list_file_chunks = glob("../tot_nghiep/vlsp/leaf_chunks/*.json")
with open("../tot_nghiep/vlsp/data/mapping_docname_norm.json", "r", encoding='utf-8') as f:
    mapping_docname = json.load(f)
with open("../tot_nghiep/vlsp/aug_chunks/legal_corpus.json", "r") as f:
    raw_aid_header = json.load(f)
mapping = {}
for sample in raw_aid_header:
    augment = sample["augment"]
    if augment:
        try:
            augment = augment.replace("```json", "").replace("```", "")
            augment = json.loads(augment)
            header = augment.get("header", "")
            summary = augment.get("summary", "")
        except:
            augment = augment.replace("```json", "").replace("```", "")
            parts = augment.split("\n")
            header = ""
            summary = ""
            for part in parts:
                if part.startswith("header:"):
                    header = part.replace("\"header\":", "").strip()
                elif part.startswith("summary:"):
                    summary = part.replace("\"summary\":", "").strip()
    else:
        header = ""
        summary = ""
    mapping[sample["aid"]] = {
        "header": header,
        "summary": summary,
        "law_id": sample["law_id"],
        "content": sample["content"]
    }
async def process_files():
    """Process files with batched embedding generation"""
    for file in tqdm(list_file_chunks, desc="Processing files"):
        with open(file) as f:
            data = json.load(f)
        
        print(f"Processing {len(data)} items from {file}")
        
        # Prepare all texts for embedding
        content_texts, summary_texts, items_with_header_summary, items_without_header_summary = prepare_texts_for_embedding(
            data, mapping, mapping_docname
        )
        
        # tasks = [
        #     async_create_embeddings_batch(content_texts, batch_size=1, model=model_embedding),
        #     # async_create_embeddings_batch(summary_texts, batch_size=32, model=model_embedding_summary)
        # ]
        # content_embeddings = await asyncio.gather(*tasks)
        content_embeddings = create_embeddings_batch(content_texts, batch_size=1, model=model_embedding)
        
        # Assign embeddings back to items
        insert_data = []
        for i, item in enumerate(data):
            aid = item["aid"]
            law_id = item["law_id"]
            # results_qwen3 = milvus_connector.filter_records(qwen3_chunks_collection, f"aid == {aid} and law_id == '{law_id}'", ["id", "aid", "law_id", "vector"])
            # qwen3_vector = results_qwen3[0]["vector"]
            item["vector_content"] = content_embeddings[i] 
            # item["vector_summary"] = summary_embeddings[i]
            # item["vector"] = qwen3_vector + content_embeddings[i] 
            insert_data.append(item)
        
        # Insert data in batches
        batch_pushing = 1000
        for i in tqdm(range(0, len(insert_data), batch_pushing), desc="Inserting to Milvus"):
            if i + batch_pushing > len(insert_data):
                batch = insert_data[i:]
            else:
                batch = insert_data[i:i + batch_pushing]
            print(f"Inserting {len(batch)} records from {file} to {chunks_collection}")
            status = milvus_connector.insert_records(
                collection_name=chunks_collection,
                records=batch
            )
            print(status)

asyncio.run(process_files())