from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import json
from glob import glob
import torch
from tqdm import tqdm
from DatabaseConnector.milvus.milvus_connector import MilvusConnector
from pymilvus import CollectionSchema, FieldSchema, DataType
import asyncio
from fusion_model import FusionModel


class EmebeddingStore:
    def __init__(self, config: dict):
        self.milvus_connector = MilvusConnector(
            uri=config["uri"],  # type: ignore
            token=config["token"],  # type: ignore
            db_name=config["db_name"],  # type: ignore
        )
        self.collection_name = config["collection_name"]
        print(f"Creating collection {self.collection_name}")
        self.milvus_connector.create_collection(
            collection_name=self.collection_name,
            schema=config["schema"],
            index_params=config["index_params"],
            vector_field=config["vector_field"],  
        )
        print("Init Model")
        self.tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"])
        self.model = FusionModel(config=None)
        self.model.load_checkpoint("./checkpoints_bkai/best_model.pt")
        self.model.to("cuda")
        self.model.eval()

        print("Loading data")

        with open(config["train_data"], "r", encoding='utf-8') as f:
            self.train_data = json.load(f)
        with open(config["mapping_docname"], "r", encoding='utf-8') as f:
            self.mapping_docname = json.load(f)
        self.list_file_chunks = glob(config["list_file_chunks"])
        with open(config["legal_corpus"], "r") as f:
            self.raw_aid_header = json.load(f)


    def create_embeddings_batch(self, texts: list, batch_size: int = 64):
        """Create embeddings in batches using Qwen3ModelWithFusion with cross-attention."""
        embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Creating embeddings"):
            batch = texts[i:i + batch_size]
            with torch.no_grad():
                batch_embeddings = self.model(texts=batch)
                embedding = batch_embeddings.fused_hidden_states
                embeddings.extend(embedding.tolist())
            torch.cuda.empty_cache()
        return embeddings

    def prepare_texts_for_embedding(self, data: list, mapping: dict, mapping_docname: dict):
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

    async def process_files(self):
        """Process files with batched embedding generation"""
        for file in tqdm(self.list_file_chunks, desc="Processing files"):
            with open(file) as f:
                data = json.load(f)
            
            print(f"Processing {len(data)} items from {file}")
            
            # Prepare all texts for embedding
            content_texts, summary_texts, items_with_header_summary, items_without_header_summary = self.prepare_texts_for_embedding(
                data, self.mapping_docname, self.mapping_docname
            )
            
            # Create embeddings using Qwen3ModelWithFusion
            content_embeddings = self.create_embeddings_batch(content_texts, batch_size=32)
            
            # Assign embeddings back to items
            insert_data = []
            for i, item in enumerate(data):
                aid = item["aid"]
                law_id = item["law_id"]
                item["vector_content"] = content_embeddings[i] 
                insert_data.append(item)
            
            # Insert data in batches
            batch_pushing = 1000
            for i in tqdm(range(0, len(insert_data), batch_pushing), desc="Inserting to Milvus"):
                if i + batch_pushing > len(insert_data):
                    batch = insert_data[i:]
                else:
                    batch = insert_data[i:i + batch_pushing]
                print(f"Inserting {len(batch)} records from {file} to {self.collection_name}")
                status = self.milvus_connector.insert_records(
                    collection_name=self.collection_name,
                    records=batch
                )
                print(status)

if __name__ == "__main__":
    index_params = {
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "params": {
            "M": 128,
            "efConstruction": 360
        }
    }
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
    embedding_store = EmebeddingStore(config={
        "uri": "http://milvus-standalone-legalrag:19530",
        "token": "root:Milvus",
        "db_name": "Master",
        "collection_name": "vlsp_chunks_qwen3_fusion_mlp",
        "schema": document_schema,
        "index_params": index_params,   
        "vector_field": ["vector_content"],
        "tokenizer": "Qwen/Qwen3-Embedding-0.6B",
        "model": "Qwen/Qwen3-Embedding-0.6B",
        "train_data": "../../data/vlsp/data/train.json",
        "mapping_docname": "../../data/vlsp/data/mapping_docname_norm.json",
        "list_file_chunks": "../../data/vlsp/leaf_chunks/*.json",
        "legal_corpus": "../../data/vlsp/aug_chunks/legal_corpus.json",
    })
    asyncio.run(embedding_store.process_files())