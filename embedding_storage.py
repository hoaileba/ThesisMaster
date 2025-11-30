import asyncio
import json
from glob import glob

import torch
from tqdm import tqdm
from pymilvus import CollectionSchema, FieldSchema, DataType
from transformers import AutoTokenizer

from DatabaseConnector.milvus.milvus_connector import MilvusConnector
from cus_qwen3_attention import Qwen3ModelWithFusion, get_embedding

milvus_connector = MilvusConnector(
    uri="http://103.253.20.30:19630",  # type: ignore
    token="root:Milvus",  # type: ignore
    db_name="Master",  # type: ignore
)
doc_collection = "documents"
chunks_collection = "vlsp_chunks_qwen3_cus_attention"
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

# Load Qwen3 tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B")

# Load Vietnamese Embedding tokenizer
embedding_tokenizer = AutoTokenizer.from_pretrained("AITeamVN/Vietnamese_Embedding_v2")

# Load custom Qwen3 model with fusion
qwen3_model = Qwen3ModelWithFusion.from_pretrained(
    "Qwen/Qwen3-Embedding-0.6B",
    embedding_model_name="AITeamVN/Vietnamese_Embedding_v2",
    scalar_mix_mode='uniform',
    last_n_layers=4
)
qwen3_model.to("cuda")
qwen3_model.eval()
def create_embeddings_batch(texts, batch_size=64):
    """Create embeddings in batches using Qwen3ModelWithFusion with cross-attention."""
    embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Creating embeddings"):
        batch = texts[i:i + batch_size]
        
        # Use get_embedding from cus_qwen3_attention
        batch_embeddings = get_embedding(
            model=qwen3_model,
            tokenizer=tokenizer,
            embedding_tokenizer=embedding_tokenizer,
            text=batch
        )
        
        embeddings.extend(batch_embeddings.tolist())
        torch.cuda.empty_cache()
    
    return embeddings
async def async_create_embeddings_batch(texts, batch_size=64):
    """Async wrapper for create_embeddings_batch."""
    return await asyncio.to_thread(create_embeddings_batch, texts, batch_size)
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

with open("./data/train.json", "r", encoding='utf-8') as f:
    train_data = json.load(f)
    
with open("./data/mapping_docname_norm.json", "r", encoding='utf-8') as f:
    mapping_docname = json.load(f)
list_file_chunks = glob("./data/leaf_chunks/*.json")
with open("./data/mapping_docname_norm.json", "r", encoding='utf-8') as f:
    mapping_docname = json.load(f)
with open("./data/legal_corpus.json", "r") as f:
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
            data[:10], mapping, mapping_docname
        )
        
        # Create embeddings using Qwen3ModelWithFusion
        content_embeddings = create_embeddings_batch(content_texts, batch_size=8)
        
        # Assign embeddings back to items
        insert_data = []
        for i, item in enumerate(data[:10]):
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