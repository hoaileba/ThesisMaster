"""
Script để chuẩn bị dữ liệu từ Zalo Legal Dataset và BKAI Dataset cho training FusionModel.
Chuyển đổi dữ liệu sang format phù hợp với MultipleNegativesRankingLoss.

IMPORTANT: Khi dùng in-batch negatives, cần đảm bảo không có 2 sample với cùng positive
trong cùng batch. File này thêm `positive_id` để hỗ trợ UniquePositiveBatchSampler.
"""
import json
import random
import csv
import ast
import hashlib
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from dataclasses import dataclass
import os
import pandas as pd
from collections import defaultdict


@dataclass
class TrainingExample:
    """Một example cho training."""
    query: str
    positive: str
    negative: Optional[str] = None
    positive_id: Optional[int] = None  # ID để track duplicate positives


def compute_positive_id(text: str) -> int:
    """Compute a hash-based ID for positive text to track duplicates."""
    return int(hashlib.md5(text.encode()).hexdigest()[:8], 16)


def add_positive_ids(data: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Add positive_id to each sample based on the hash of positive text.
    This allows BatchSampler to ensure unique positives per batch.
    """
    for item in data:
        item['positive_id'] = compute_positive_id(item['positive'])
    return data


def get_duplicate_positive_stats(data: List[Dict[str, str]]) -> Dict:
    """Get statistics about duplicate positives in the dataset."""
    positive_counts = defaultdict(int)
    for item in data:
        positive_id = item.get('positive_id', compute_positive_id(item['positive']))
        positive_counts[positive_id] += 1
    
    total_positives = len(positive_counts)
    duplicates = {k: v for k, v in positive_counts.items() if v > 1}
    num_duplicated = len(duplicates)
    max_duplicates = max(positive_counts.values()) if positive_counts else 0
    
    return {
        'total_unique_positives': total_positives,
        'num_positives_with_duplicates': num_duplicated,
        'max_samples_per_positive': max_duplicates,
        'total_samples': len(data),
    }


def load_corpus(corpus_path: str) -> Dict[str, Dict[str, Dict]]:
    """
    Load legal corpus và tạo index theo law_id -> article_id -> article.
    
    Returns:
        Dict[law_id, Dict[article_id, article_dict]]
    """
    print(f"Loading corpus from {corpus_path}...")
    with open(corpus_path, 'r', encoding='utf-8') as f:
        corpus = json.load(f)
    
    # Tạo index
    corpus_index = {}
    total_articles = 0
    
    for law in corpus:
        law_id = law['law_id']
        corpus_index[law_id] = {}
        
        for article in law['articles']:
            article_id = article['article_id']
            corpus_index[law_id][article_id] = article
            total_articles += 1
    
    print(f"✓ Loaded {len(corpus_index)} laws with {total_articles} articles")
    return corpus_index


def load_questions(questions_path: str) -> List[Dict]:
    """Load questions từ train_question_answer.json."""
    print(f"Loading questions from {questions_path}...")
    with open(questions_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    questions = data['items']
    print(f"✓ Loaded {len(questions)} questions")
    return questions


def get_article_text(article: Dict) -> str:
    """Tạo text từ article (title + text)."""
    title = article.get('title', '')
    text = article.get('text', '')
    return f"{title}\n{text}".strip()


def prepare_training_data_mnrl(
    questions: List[Dict],
    corpus_index: Dict[str, Dict[str, Dict]],
    include_negatives: bool = False,
    num_hard_negatives: int = 1,
) -> List[Dict[str, str]]:
    """
    Chuẩn bị dữ liệu cho MultipleNegativesRankingLoss.
    
    IMPORTANT: Mỗi sample sẽ có `positive_id` để track duplicate positives.
    Khi dùng in-batch negatives, cần đảm bảo không có 2 sample với cùng positive_id
    trong cùng batch (dùng UniquePositiveBatchSampler).
    
    Args:
        questions: List các câu hỏi
        corpus_index: Index của corpus
        include_negatives: Có thêm hard negatives không
        num_hard_negatives: Số hard negatives mỗi example
    
    Returns:
        List[Dict] với keys: 'query', 'positive', 'positive_id', 'negative' (optional)
    """
    print("Preparing training data for MNRL...")
    training_data = []
    skipped = 0
    
    # Collect all articles for negative sampling
    all_articles = []
    for law_id, articles in corpus_index.items():
        for article_id, article in articles.items():
            all_articles.append({
                'law_id': law_id,
                'article_id': article_id,
                'text': get_article_text(article)
            })
    
    for item in tqdm(questions, desc="Processing questions"):
        question = item['question']
        relevant_articles = item['relevant_articles']
        
        # Lấy positive articles
        positive_texts = []
        relevant_keys = set()
        
        for ref in relevant_articles:
            law_id = ref['law_id']
            article_id = ref['article_id']
            relevant_keys.add((law_id, article_id))
            
            if law_id in corpus_index and article_id in corpus_index[law_id]:
                article = corpus_index[law_id][article_id]
                positive_texts.append(get_article_text(article))
            else:
                skipped += 1
        
        if not positive_texts:
            continue
        
        # Tạo training examples
        for positive_text in positive_texts:
            example = {
                'query': question,
                'positive': positive_text,
                'positive_id': compute_positive_id(positive_text),  # Track duplicates
            }
            
            # Thêm hard negative (random từ corpus, không trùng positive)
            if include_negatives:
                negatives = []
                attempts = 0
                while len(negatives) < num_hard_negatives and attempts < 100:
                    neg_article = random.choice(all_articles)
                    neg_key = (neg_article['law_id'], neg_article['article_id'])
                    if neg_key not in relevant_keys:
                        negatives.append(neg_article['text'])
                    attempts += 1
                
                if negatives:
                    example['negative'] = negatives[0] if num_hard_negatives == 1 else negatives
            
            training_data.append(example)
    
    print(f"✓ Created {len(training_data)} training examples")
    if skipped > 0:
        print(f"⚠ Skipped {skipped} articles not found in corpus")
    
    # Print duplicate stats for in-batch negative awareness
    stats = get_duplicate_positive_stats(training_data)
    print(f"📊 Duplicate positive stats:")
    print(f"   - Unique positives: {stats['total_unique_positives']}")
    print(f"   - Positives with duplicates: {stats['num_positives_with_duplicates']}")
    print(f"   - Max samples per positive: {stats['max_samples_per_positive']}")
    
    return training_data


def prepare_training_data_triplet(
    questions: List[Dict],
    corpus_index: Dict[str, Dict[str, Dict]],
) -> List[Dict[str, str]]:
    """
    Chuẩn bị dữ liệu cho TripletLoss (query, positive, negative).
    """
    return prepare_training_data_mnrl(
        questions, corpus_index, include_negatives=True, num_hard_negatives=1
    )


def prepare_corpus_for_retrieval(
    corpus_index: Dict[str, Dict[str, Dict]],
) -> Tuple[List[str], List[str]]:
    """
    Chuẩn bị corpus cho retrieval evaluation.
    
    Returns:
        texts: List các article texts
        ids: List các article ids (format: law_id__article_id)
    """
    texts = []
    ids = []
    
    for law_id, articles in corpus_index.items():
        for article_id, article in articles.items():
            text = get_article_text(article)
            doc_id = f"{law_id}__{article_id}"
            texts.append(text)
            ids.append(doc_id)
    
    return texts, ids


def split_train_val(
    data: List[Dict[str, str]],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Split data thành train và validation."""
    random.seed(seed)
    data_shuffled = data.copy()
    random.shuffle(data_shuffled)
    
    val_size = int(len(data_shuffled) * val_ratio)
    val_data = data_shuffled[:val_size]
    train_data = data_shuffled[val_size:]
    
    print(f"✓ Split: {len(train_data)} train, {len(val_data)} validation")
    return train_data, val_data


def save_data(data: List[Dict], output_path: str):
    """Save data to JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"✓ Saved to {output_path}")


def load_training_data(path: str) -> List[Dict[str, str]]:
    """Load training data from JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def prepare_zalo_dataset(
    corpus_path: str = "/home/htsc/dev/namdp/Master/FusionEmbedding/data/legal_dataset/zalo/legal_corpus.json",
    questions_path: str = "/home/htsc/dev/namdp/Master/FusionEmbedding/data/legal_dataset/zalo/train_question_answer.json",
    output_dir: str = "/home/htsc/dev/namdp/Master/FusionEmbedding/data/legal_dataset/zalo/processed",
    include_negatives: bool = True,
    val_ratio: float = 0.1,
):
    """
    Main function để chuẩn bị Zalo dataset.
    
    Args:
        corpus_path: Path to legal_corpus.json
        questions_path: Path to train_question_answer.json
        output_dir: Directory to save processed data
        include_negatives: Include hard negatives for triplet loss
        val_ratio: Validation split ratio
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    corpus_index = load_corpus(corpus_path)
    questions = load_questions(questions_path)
    
    # Prepare MNRL data (query, positive pairs)
    mnrl_data = prepare_training_data_mnrl(
        questions, corpus_index, include_negatives=False
    )
    
    # Prepare Triplet data (query, positive, negative)
    triplet_data = prepare_training_data_mnrl(
        questions, corpus_index, include_negatives=True
    )
    
    # Split train/val
    mnrl_train, mnrl_val = split_train_val(mnrl_data, val_ratio)
    triplet_train, triplet_val = split_train_val(triplet_data, val_ratio)
    
    # Save data
    save_data(mnrl_train, os.path.join(output_dir, "train_mnrl.json"))
    save_data(mnrl_val, os.path.join(output_dir, "val_mnrl.json"))
    save_data(triplet_train, os.path.join(output_dir, "train_triplet.json"))
    save_data(triplet_val, os.path.join(output_dir, "val_triplet.json"))
    
    # Save corpus for retrieval
    texts, ids = prepare_corpus_for_retrieval(corpus_index)
    corpus_data = [{"id": id, "text": text} for id, text in zip(ids, texts)]
    save_data(corpus_data, os.path.join(output_dir, "corpus.json"))
    
    # Print statistics
    print("\n" + "="*50)
    print("Dataset Statistics:")
    print("="*50)
    print(f"Total questions:        {len(questions)}")
    print(f"Total articles:         {len(texts)}")
    print(f"MNRL train examples:    {len(mnrl_train)}")
    print(f"MNRL val examples:      {len(mnrl_val)}")
    print(f"Triplet train examples: {len(triplet_train)}")
    print(f"Triplet val examples:   {len(triplet_val)}")
    print("="*50)
    
    # Sample data preview
    print("\n--- Sample MNRL training example ---")
    sample = mnrl_train[0]
    print(f"Query: {sample['query'][:100]}...")
    print(f"Positive: {sample['positive'][:100]}...")
    
    if triplet_train and 'negative' in triplet_train[0]:
        print("\n--- Sample Triplet training example ---")
        sample = triplet_train[0]
        print(f"Query: {sample['query'][:100]}...")
        print(f"Positive: {sample['positive'][:100]}...")
        print(f"Negative: {sample['negative'][:100]}...")
    
    return {
        'mnrl_train': mnrl_train,
        'mnrl_val': mnrl_val,
        'triplet_train': triplet_train,
        'triplet_val': triplet_val,
        'corpus': corpus_data,
    }


###############################################################################
# BKAI Dataset Processing
###############################################################################

def load_bkai_corpus(corpus_path: str) -> Dict[int, str]:
    """
    Load BKAI corpus từ CSV file.
    
    Returns:
        Dict[cid, text]
    """
    print(f"Loading BKAI corpus from {corpus_path}...")
    corpus_index = {}
    
    df = pd.read_csv(corpus_path)
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading corpus"):
        cid = int(row['cid'])
        text = str(row['text'])
        corpus_index[cid] = text
    
    print(f"✓ Loaded {len(corpus_index)} documents")
    return corpus_index


def load_bkai_questions(questions_path: str) -> List[Dict]:
    """
    Load BKAI questions từ CSV file.
    
    Returns:
        List of dicts with keys: question, context, cid, qid
    """
    print(f"Loading BKAI questions from {questions_path}...")
    questions = []
    
    df = pd.read_csv(questions_path)
    for _, row in df.iterrows():
        # Parse context (list of strings)
        context_str = row['context']
        try:
            context = ast.literal_eval(context_str) if isinstance(context_str, str) else context_str
        except:
            context = [context_str]
        
        # Parse cid (list of ints)
        cid_str = row['cid']
        try:
            if isinstance(cid_str, str):
                cid = ast.literal_eval(cid_str)
            else:
                cid = [int(cid_str)]
        except:
            cid = []
        
        questions.append({
            'question': str(row['question']),
            'context': context if isinstance(context, list) else [context],
            'cid': cid if isinstance(cid, list) else [cid],
            'qid': row['qid'],
        })
    
    print(f"✓ Loaded {len(questions)} questions")
    return questions


def prepare_bkai_training_data(
    questions: List[Dict],
    corpus_index: Dict[int, str] = None,
    include_negatives: bool = False,
    num_hard_negatives: int = 1,
) -> List[Dict[str, str]]:
    """
    Chuẩn bị dữ liệu BKAI cho training.
    
    BKAI dataset đã có context trong file train, không cần corpus.
    
    IMPORTANT: Mỗi sample sẽ có `positive_id` để track duplicate positives.
    Khi dùng in-batch negatives, cần đảm bảo không có 2 sample với cùng positive_id
    trong cùng batch (dùng UniquePositiveBatchSampler).
    
    Args:
        questions: List các câu hỏi từ BKAI
        corpus_index: Optional corpus index cho negative sampling
        include_negatives: Có thêm hard negatives không
        num_hard_negatives: Số hard negatives mỗi example
    
    Returns:
        List[Dict] với keys: 'query', 'positive', 'positive_id', 'negative' (optional)
    """
    print("Preparing BKAI training data...")
    training_data = []
    
    # Collect all contexts for negative sampling
    all_contexts = []
    all_cids = set()
    
    for item in questions:
        for ctx in item['context']:
            if ctx and len(ctx.strip()) > 0:
                all_contexts.append(ctx)
        for cid in item['cid']:
            all_cids.add(cid)
    
    for item in tqdm(questions, desc="Processing questions"):
        question = item['question']
        contexts = item['context']
        cids = set(item['cid'])
        
        # Mỗi context là một positive
        for ctx in contexts:
            if not ctx or len(ctx.strip()) == 0:
                continue
            
            positive_text = ctx.strip()
            example = {
                'query': question,
                'positive': positive_text,
                'positive_id': compute_positive_id(positive_text),  # Track duplicates
            }
            
            # Thêm hard negative
            if include_negatives and all_contexts:
                negatives = []
                attempts = 0
                while len(negatives) < num_hard_negatives and attempts < 100:
                    neg_ctx = random.choice(all_contexts)
                    # Đảm bảo không trùng với positive
                    if neg_ctx not in contexts and len(neg_ctx.strip()) > 0:
                        negatives.append(neg_ctx.strip())
                    attempts += 1
                
                if negatives:
                    example['negative'] = negatives[0] if num_hard_negatives == 1 else negatives
            
            training_data.append(example)
    
    print(f"✓ Created {len(training_data)} training examples")
    
    # Print duplicate stats for in-batch negative awareness
    stats = get_duplicate_positive_stats(training_data)
    print(f"📊 Duplicate positive stats:")
    print(f"   - Unique positives: {stats['total_unique_positives']}")
    print(f"   - Positives with duplicates: {stats['num_positives_with_duplicates']}")
    print(f"   - Max samples per positive: {stats['max_samples_per_positive']}")
    
    return training_data


def prepare_bkai_dataset(
    train_path: str = "/home/htsc/dev/namdp/Master/FusionEmbedding/data/legal_dataset/bkai/train_split.csv",
    val_path: str = "/home/htsc/dev/namdp/Master/FusionEmbedding/data/legal_dataset/bkai/val_split.csv",
    corpus_path: str = None,  # Optional, không cần nếu context đã có trong train
    output_dir: str = "/home/htsc/dev/namdp/Master/FusionEmbedding/data/legal_dataset/bkai/processed",
    include_negatives: bool = True,
):
    """
    Main function để chuẩn bị BKAI dataset.
    
    Args:
        train_path: Path to train_split.csv hoặc train.csv
        val_path: Path to val_split.csv (optional)
        corpus_path: Path to corpus.csv (optional, cho negative sampling)
        output_dir: Directory to save processed data
        include_negatives: Include hard negatives for triplet loss
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load corpus nếu có
    corpus_index = None
    if corpus_path and os.path.exists(corpus_path):
        corpus_index = load_bkai_corpus(corpus_path)
    
    # Load train questions
    train_questions = load_bkai_questions(train_path)
    
    # Load val questions nếu có
    val_questions = None
    if val_path and os.path.exists(val_path):
        val_questions = load_bkai_questions(val_path)
    
    # Prepare MNRL data (query, positive pairs)
    mnrl_train = prepare_bkai_training_data(
        train_questions, corpus_index, include_negatives=False
    )
    
    # Prepare Triplet data (query, positive, negative)
    triplet_train = prepare_bkai_training_data(
        train_questions, corpus_index, include_negatives=include_negatives
    )
    
    # Prepare validation data
    if val_questions:
        mnrl_val = prepare_bkai_training_data(
            val_questions, corpus_index, include_negatives=False
        )
        triplet_val = prepare_bkai_training_data(
            val_questions, corpus_index, include_negatives=include_negatives
        )
    else:
        # Split from train if no val file
        mnrl_train, mnrl_val = split_train_val(mnrl_train, val_ratio=0.1)
        triplet_train, triplet_val = split_train_val(triplet_train, val_ratio=0.1)
    
    # Save data
    save_data(mnrl_train, os.path.join(output_dir, "train_mnrl.json"))
    save_data(mnrl_val, os.path.join(output_dir, "val_mnrl.json"))
    save_data(triplet_train, os.path.join(output_dir, "train_triplet.json"))
    save_data(triplet_val, os.path.join(output_dir, "val_triplet.json"))
    
    # Print statistics
    print("\n" + "="*50)
    print("BKAI Dataset Statistics:")
    print("="*50)
    print(f"Total train questions:  {len(train_questions)}")
    if val_questions:
        print(f"Total val questions:    {len(val_questions)}")
    print(f"MNRL train examples:    {len(mnrl_train)}")
    print(f"MNRL val examples:      {len(mnrl_val)}")
    print(f"Triplet train examples: {len(triplet_train)}")
    print(f"Triplet val examples:   {len(triplet_val)}")
    print("="*50)
    
    # Sample data preview
    if mnrl_train:
        print("\n--- Sample MNRL training example ---")
        sample = mnrl_train[0]
        print(f"Query: {sample['query'][:100]}...")
        print(f"Positive: {sample['positive'][:100]}...")
    
    if triplet_train and 'negative' in triplet_train[0]:
        print("\n--- Sample Triplet training example ---")
        sample = triplet_train[0]
        print(f"Query: {sample['query'][:100]}...")
        print(f"Positive: {sample['positive'][:100]}...")
        print(f"Negative: {sample['negative'][:100]}...")
    
    return {
        'mnrl_train': mnrl_train,
        'mnrl_val': mnrl_val,
        'triplet_train': triplet_train,
        'triplet_val': triplet_val,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare data for FusionModel training")
    parser.add_argument("--dataset", type=str, default="zalo", choices=["zalo", "bkai"],
                        help="Dataset to prepare: 'zalo' or 'bkai'")
    
    args = parser.parse_args()
    
    if args.dataset == "zalo":
        # Prepare Zalo dataset
        print("\n" + "="*50)
        print("Preparing Zalo Legal Dataset")
        print("="*50 + "\n")
        
        data = prepare_zalo_dataset(
            corpus_path="../../data/legal_dataset/zalo/legal_corpus.json",
            questions_path="../../data/legal_dataset/zalo/train_question_answer.json",
            output_dir="../../data/legal_dataset/zalo/processed",
            include_negatives=True,
            val_ratio=0.1,
        )
        print("\n✓ Zalo data preparation completed!")
        print(f"Output files saved to: ../../data/legal_dataset/zalo/processed/")
        
    else:
        # Prepare BKAI dataset
        print("\n" + "="*50)
        print("Preparing BKAI Legal Dataset")
        print("="*50 + "\n")
        
        data = prepare_bkai_dataset(
            train_path="../../data/legal_dataset/bkai/train_split.csv",
            val_path="../../data/legal_dataset/bkai/val_split.csv",
            corpus_path=None,  # Context đã có trong train file
            output_dir="../../data/legal_dataset/bkai/processed",
            include_negatives=True,
        )
        print("\n✓ BKAI data preparation completed!")
        print(f"Output files saved to: ../../data/legal_dataset/bkai/processed/")

