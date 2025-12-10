"""
Training script using sentence_transformers library.
Supports MultipleNegativesRankingLoss, TripletLoss, CosineSimilarityLoss.

IMPORTANT: Khi dùng MNRL (in-batch negatives), cần đảm bảo không có 2 sample 
với cùng positive trong cùng batch để tránh false negatives.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Sampler
from typing import List, Dict, Optional, Union, Iterator
from datetime import datetime
from collections import defaultdict
import random
import hashlib
import os

from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.models import Transformer, Pooling
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

from fusion_model import FusionModel, CrossAttention


def compute_positive_id(text: str) -> int:
    """Compute a hash-based ID for positive text to track duplicates."""
    return int(hashlib.md5(text.encode()).hexdigest()[:8], 16)


class UniquePositiveBatchSampler(Sampler):
    """
    BatchSampler đảm bảo không có 2 sample với cùng positive trong cùng batch.
    
    QUAN TRỌNG cho in-batch negatives (MNRL): Nếu 2 sample có cùng positive text 
    trong cùng batch, khi tính in-batch negatives, positive của sample A sẽ bị 
    coi là negative của sample B (false negative).
    """
    
    def __init__(
        self,
        data: List[Dict[str, str]],
        batch_size: int,
        drop_last: bool = False,
        shuffle: bool = True,
    ):
        self.data = data
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        
        # Build positive_id -> indices mapping
        self.positive_to_indices = defaultdict(list)
        for idx, item in enumerate(data):
            pos_id = item.get('positive_id', compute_positive_id(item['positive']))
            self.positive_to_indices[pos_id].append(idx)
        
        self.num_positives = len(self.positive_to_indices)
        self._print_stats()
    
    def _print_stats(self):
        """Print sampler statistics."""
        total_samples = len(self.data)
        unique_positives = len(self.positive_to_indices)
        max_per_positive = max(len(v) for v in self.positive_to_indices.values())
        
        print(f"\n{'='*50}")
        print(f"UniquePositiveBatchSampler initialized:")
        print(f"  - Total samples: {total_samples}")
        print(f"  - Unique positives: {unique_positives}")
        print(f"  - Max samples per positive: {max_per_positive}")
        print(f"  - Batch size: {self.batch_size}")
        print(f"  - Effective batches per epoch: ~{unique_positives // self.batch_size}")
        print(f"{'='*50}\n")
    
    def __iter__(self) -> Iterator[List[int]]:
        """Yield batches where each batch has unique positives."""
        # Collect one index per positive group
        indices_pool = []
        positive_ids = list(self.positive_to_indices.keys())
        
        if self.shuffle:
            random.shuffle(positive_ids)
        
        for pos_id in positive_ids:
            group_indices = self.positive_to_indices[pos_id]
            chosen_idx = random.choice(group_indices)
            indices_pool.append(chosen_idx)
        
        # Yield batches
        batch = []
        for idx in indices_pool:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        
        if len(batch) > 0 and not self.drop_last:
            yield batch
    
    def __len__(self) -> int:
        if self.drop_last:
            return self.num_positives // self.batch_size
        return (self.num_positives + self.batch_size - 1) // self.batch_size


class FusionModelWrapper(nn.Module):
    """
    Wrapper để FusionModel tương thích với sentence_transformers.
    Freeze pretrained models, chỉ train CrossAttention layer.
    """
    
    def __init__(self, fusion_model: FusionModel):
        super().__init__()
        self.fusion_model = fusion_model
        self.device = fusion_model.device
        
        # Freeze pretrained models
        self._freeze_pretrained()
        
    def _freeze_pretrained(self):
        """Freeze encoder and decoder pretrained models."""
        for param in self.fusion_model.encoder_model.parameters():
            param.requires_grad = False
        for param in self.fusion_model.decoder_model.parameters():
            param.requires_grad = False
        print("✓ Frozen pretrained encoder and decoder models")
    
    def forward(self, features: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Forward pass compatible with sentence_transformers."""
        # Get texts from input_ids (decode back to text)
        # This is a workaround since sentence_transformers tokenizes first
        input_ids = features.get("input_ids")
        attention_mask = features.get("attention_mask")
        
        # Decode input_ids back to texts
        texts = self.fusion_model.decoder_tokenizer.batch_decode(
            input_ids, skip_special_tokens=True
        )
        
        # Forward through fusion model
        output = self._forward_fusion(texts)
        
        return {"sentence_embedding": output}
    
    def _forward_fusion(self, texts: List[str]) -> Tensor:
        """Forward pass with gradients only on fusion layer."""
        # Encoder forward (no grad)
        input_tokens_encoder = self.fusion_model.encoder_tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            encoder_output = self.fusion_model.encoder_model(**input_tokens_encoder)
            encoder_embedding = encoder_output.last_hidden_state[:, 0]
        
        # Decoder forward (no grad)
        input_tokens_decoder = self.fusion_model.decoder_tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, padding_side='left'
        ).to(self.device)
        
        with torch.no_grad():
            decoder_output = self.fusion_model.decoder_model(**input_tokens_decoder)
            hidden_decoder_states = torch.stack(decoder_output.stack_hidden_states[-5:])
            last_hidden_decoder_states = hidden_decoder_states[:, :, -1, :]
            last_hidden_decoder_states = last_hidden_decoder_states.transpose(0, 1)
        
        # Cross-Attention Fusion (with grad)
        encoder_embedding_seq = encoder_embedding.unsqueeze(1)
        fused_states = self.fusion_model.cross_attention(
            decoder_hidden=last_hidden_decoder_states,
            encoder_hidden=encoder_embedding_seq,
        )
        fused_embedding = fused_states.mean(dim=1)
        fused_embedding = F.normalize(fused_embedding, p=2, dim=1)
        
        return fused_embedding
    
    def tokenize(self, texts: List[str]) -> Dict[str, Tensor]:
        """Tokenize texts using decoder tokenizer."""
        return self.fusion_model.decoder_tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
    
    def get_sentence_embedding_dimension(self) -> int:
        """Return embedding dimension."""
        return self.fusion_model.decoder_dim


class FusionSentenceTransformer(SentenceTransformer):
    """Custom SentenceTransformer wrapper for FusionModel."""
    
    def __init__(self, fusion_model: FusionModel):
        # Initialize without calling parent __init__ fully
        nn.Module.__init__(self)
        
        self.fusion_wrapper = FusionModelWrapper(fusion_model)
        self._modules = nn.ModuleDict({"0": self.fusion_wrapper})
        # Store device without triggering SentenceTransformer's _target_device setter
        object.__setattr__(self, '_fusion_device', fusion_model.device)
        
    def encode(
        self,
        sentences: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        convert_to_tensor: bool = True,
        normalize_embeddings: bool = True,
        **kwargs,
    ) -> Union[Tensor, List[Tensor]]:
        """Encode sentences to embeddings."""
        if isinstance(sentences, str):
            sentences = [sentences]
        
        self.eval()
        all_embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(sentences), batch_size):
                batch = sentences[i:i + batch_size]
                embeddings = self.fusion_wrapper._forward_fusion(batch)
                if normalize_embeddings:
                    embeddings = F.normalize(embeddings, p=2, dim=1)
                all_embeddings.append(embeddings)
        
        all_embeddings = torch.cat(all_embeddings, dim=0)
        
        if not convert_to_tensor:
            return all_embeddings.cpu().numpy()
        return all_embeddings
    
    def forward(self, features: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Forward pass."""
        return self.fusion_wrapper(features)
    
    def tokenize(self, texts: List[str]) -> Dict[str, Tensor]:
        """Tokenize texts."""
        return self.fusion_wrapper.tokenize(texts)
    
    def get_sentence_embedding_dimension(self) -> int:
        """Return embedding dimension."""
        return self.fusion_wrapper.get_sentence_embedding_dimension()


class FusionTrainerST:
    """Trainer using sentence_transformers style."""
    
    def __init__(
        self,
        model: FusionSentenceTransformer,
        loss_type: str = "mnrl",  # "mnrl", "triplet", "cosine"
        learning_rate: float = 1e-4,
        device: str = "cuda",
        log_dir: str = None,
    ):
        self.model = model
        self.device = device
        self.learning_rate = learning_rate
        self.loss_type = loss_type
        self.global_step = 0
        
        # TensorBoard writer
        if log_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = f"./runs/fusion_st_{loss_type}_{timestamp}"
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=log_dir)
        print(f"✓ TensorBoard logging to: {log_dir}")
        
        # Setup loss function
        self.loss_fn = self._get_loss_function()
        
        self._print_trainable_params()
    
    def _get_loss_function(self) -> nn.Module:
        """Get loss function based on type."""
        if self.loss_type == "mnrl":
            # MultipleNegativesRankingLoss - best for in-batch negatives
            return losses.MultipleNegativesRankingLoss(self.model)
        elif self.loss_type == "triplet":
            return losses.TripletLoss(self.model)
        elif self.loss_type == "cosine":
            return losses.CosineSimilarityLoss(self.model)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
    
    def _print_trainable_params(self):
        """Print trainable parameters info."""
        fusion_model = self.model.fusion_wrapper.fusion_model
        total_params = sum(p.numel() for p in fusion_model.parameters())
        trainable_params = sum(p.numel() for p in fusion_model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        print(f"\n{'='*50}")
        print(f"Total parameters:     {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Frozen parameters:    {frozen_params:,}")
        print(f"Trainable ratio:      {trainable_params/total_params*100:.2f}%")
        print(f"Loss function:        {self.loss_type.upper()}")
        print(f"{'='*50}\n")
    
    def prepare_data_mnrl(
        self,
        data: List[Dict[str, str]],
    ) -> List[InputExample]:
        """Prepare data for MultipleNegativesRankingLoss."""
        examples = []
        for item in data:
            examples.append(InputExample(
                texts=[item["query"], item["positive"]]
            ))
        return examples
    
    def prepare_data_triplet(
        self,
        data: List[Dict[str, str]],
    ) -> List[InputExample]:
        """Prepare data for TripletLoss."""
        examples = []
        for item in data:
            examples.append(InputExample(
                texts=[item["query"], item["positive"], item["negative"]]
            ))
        return examples
    
    def prepare_data_cosine(
        self,
        data: List[Dict[str, str]],
    ) -> List[InputExample]:
        """Prepare data for CosineSimilarityLoss."""
        examples = []
        for item in data:
            # Positive pair
            examples.append(InputExample(
                texts=[item["query"], item["positive"]],
                label=1.0
            ))
            # Negative pair
            examples.append(InputExample(
                texts=[item["query"], item["negative"]],
                label=0.0
            ))
        return examples
    
    def _create_mnrl_dataloader(
        self,
        train_data: List[Dict[str, str]],
        batch_sampler: UniquePositiveBatchSampler,
    ):
        """
        Create custom dataloader for MNRL that uses UniquePositiveBatchSampler.
        Returns batches of raw data dicts instead of InputExample.
        """
        class MNRLBatchIterator:
            def __init__(self, data, sampler):
                self.data = data
                self.sampler = sampler
            
            def __iter__(self):
                for batch_indices in self.sampler:
                    batch = [self.data[i] for i in batch_indices]
                    yield batch
            
            def __len__(self):
                return len(self.sampler)
        
        return MNRLBatchIterator(train_data, batch_sampler)
    
    def train(
        self,
        train_data: List[Dict[str, str]],
        num_epochs: int = 10,
        batch_size: int = 8,
        warmup_steps: int = 100,
        save_dir: str = "./checkpoints_st",
        eval_data: List[Dict[str, str]] = None,
    ):
        """Train the model."""
        os.makedirs(save_dir, exist_ok=True)
        
        # Log hyperparameters
        self.writer.add_hparams(
            {
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "learning_rate": self.learning_rate,
                "loss_type": self.loss_type,
                "num_samples": len(train_data),
            },
            {"hparam/placeholder": 0},
        )
        
        # Prepare training data
        if self.loss_type == "mnrl":
            train_examples = self.prepare_data_mnrl(train_data)
            # IMPORTANT: Use UniquePositiveBatchSampler for MNRL to avoid false negatives
            batch_sampler = UniquePositiveBatchSampler(
                data=train_data,
                batch_size=batch_size,
                drop_last=False,
                shuffle=True,
            )
            # Custom dataloader using batch indices
            train_dataloader = self._create_mnrl_dataloader(train_data, batch_sampler)
        elif self.loss_type == "triplet":
            train_examples = self.prepare_data_triplet(train_data)
            train_dataloader = DataLoader(
                train_examples,
                shuffle=True,
                batch_size=batch_size,
            )
        else:
            train_examples = self.prepare_data_cosine(train_data)
            train_dataloader = DataLoader(
                train_examples,
                shuffle=True,
                batch_size=batch_size,
            )
        
        # Setup optimizer (only for cross_attention parameters)
        fusion_model = self.model.fusion_wrapper.fusion_model
        optimizer = torch.optim.AdamW(
            fusion_model.cross_attention.parameters(),
            lr=self.learning_rate,
        )
        
        # Training loop
        self.model.train()
        total_steps = len(train_dataloader) * num_epochs
        
        print(f"Starting training...")
        print(f"  Num examples: {len(train_examples)}")
        print(f"  Batch size: {batch_size}")
        print(f"  Num epochs: {num_epochs}")
        print(f"  Total steps: {total_steps}")
        
        best_loss = float("inf")
        
        for epoch in range(1, num_epochs + 1):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch in train_dataloader:
                optimizer.zero_grad()
                
                # Get texts from batch
                if self.loss_type == "mnrl":
                    # Custom dataloader returns dicts directly
                    texts_a = [item["query"] for item in batch]
                    texts_b = [item["positive"] for item in batch]
                    
                    # Forward
                    emb_a = self.model.fusion_wrapper._forward_fusion(texts_a)
                    emb_b = self.model.fusion_wrapper._forward_fusion(texts_b)
                    
                    # InfoNCE loss
                    loss = self._compute_mnrl_loss(emb_a, emb_b)
                    
                elif self.loss_type == "triplet":
                    anchors = [ex.texts[0] for ex in batch]
                    positives = [ex.texts[1] for ex in batch]
                    negatives = [ex.texts[2] for ex in batch]
                    
                    emb_a = self.model.fusion_wrapper._forward_fusion(anchors)
                    emb_p = self.model.fusion_wrapper._forward_fusion(positives)
                    emb_n = self.model.fusion_wrapper._forward_fusion(negatives)
                    
                    loss = self._compute_triplet_loss(emb_a, emb_p, emb_n)
                    
                else:  # cosine
                    texts_a = [ex.texts[0] for ex in batch]
                    texts_b = [ex.texts[1] for ex in batch]
                    labels = torch.tensor([ex.label for ex in batch], device=self.device)
                    
                    emb_a = self.model.fusion_wrapper._forward_fusion(texts_a)
                    emb_b = self.model.fusion_wrapper._forward_fusion(texts_b)
                    
                    loss = self._compute_cosine_loss(emb_a, emb_b, labels)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                self.global_step += 1
                
                # Log step loss to TensorBoard
                self.writer.add_scalar("Loss/step", loss.item(), self.global_step)
            
            avg_loss = epoch_loss / num_batches
            print(f"Epoch {epoch}/{num_epochs} - Loss: {avg_loss:.4f}")
            
            # Log epoch metrics to TensorBoard
            self.writer.add_scalar("Loss/epoch", avg_loss, epoch)
            self.writer.add_scalar("Learning_Rate", self.learning_rate, epoch)
            
            # Save checkpoint
            if avg_loss < best_loss:
                best_loss = avg_loss
                self.save_checkpoint(os.path.join(save_dir, "best_model.pt"), epoch, avg_loss)
                self.writer.add_scalar("Loss/best", best_loss, epoch)
            
            self.save_checkpoint(
                os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pt"),
                epoch,
                avg_loss,
            )
        
        # Close TensorBoard writer
        self.writer.close()
        
        print(f"\n✓ Training completed! Best loss: {best_loss:.4f}")
        print(f"✓ TensorBoard logs saved to: {self.log_dir}")
        print(f"  Run: tensorboard --logdir {self.log_dir}")
    
    def _compute_mnrl_loss(self, emb_a: Tensor, emb_b: Tensor, temperature: float = 0.05) -> Tensor:
        """Compute MultipleNegativesRankingLoss (InfoNCE)."""
        scores = torch.matmul(emb_a, emb_b.T) / temperature
        labels = torch.arange(len(emb_a), device=self.device)
        return F.cross_entropy(scores, labels)
    
    def _compute_triplet_loss(self, anchor: Tensor, positive: Tensor, negative: Tensor, margin: float = 0.5) -> Tensor:
        """Compute TripletLoss."""
        pos_sim = F.cosine_similarity(anchor, positive)
        neg_sim = F.cosine_similarity(anchor, negative)
        loss = F.relu(margin - pos_sim + neg_sim)
        return loss.mean()
    
    def _compute_cosine_loss(self, emb_a: Tensor, emb_b: Tensor, labels: Tensor) -> Tensor:
        """Compute CosineSimilarityLoss."""
        cos_sim = F.cosine_similarity(emb_a, emb_b)
        loss = F.mse_loss(cos_sim, labels)
        return loss
    
    def save_checkpoint(self, path: str, epoch: int, loss: float):
        """Save model checkpoint (only fusion layer)."""
        fusion_model = self.model.fusion_wrapper.fusion_model
        checkpoint = {
            "epoch": epoch,
            "loss": loss,
            "cross_attention_state_dict": fusion_model.cross_attention.state_dict(),
        }
        torch.save(checkpoint, path)
        print(f"✓ Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        fusion_model = self.model.fusion_wrapper.fusion_model
        checkpoint = torch.load(path, map_location=self.device)
        fusion_model.cross_attention.load_state_dict(checkpoint["cross_attention_state_dict"])
        print(f"✓ Loaded checkpoint from {path}")
        return checkpoint["epoch"], checkpoint["loss"]


def train_with_sentence_transformers(
    train_data: List[Dict[str, str]],
    num_epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    loss_type: str = "mnrl",
    save_dir: str = "./checkpoints_st",
):
    """Main training function using sentence_transformers style."""
    # Initialize FusionModel
    print("Initializing FusionModel...")
    fusion_model = FusionModel(config=None)
    fusion_model.to("cuda")
    
    # Wrap with SentenceTransformer interface
    st_model = FusionSentenceTransformer(fusion_model)
    
    # Initialize trainer
    trainer = FusionTrainerST(
        model=st_model,
        loss_type=loss_type,
        learning_rate=learning_rate,
    )
    
    # Train
    trainer.train(
        train_data=train_data,
        num_epochs=num_epochs,
        batch_size=batch_size,
        save_dir=save_dir,
    )
    
    return st_model


def train_legal_dataset(
    train_data_path: str,
    val_data_path: str = None,
    num_epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    loss_type: str = "mnrl",
    save_dir: str = "./checkpoints",
):
    """
    Train FusionModel với Legal Dataset (Zalo hoặc BKAI).
    
    Cần chạy prepare_data.py trước để tạo dữ liệu processed.
    
    Args:
        train_data_path: Path to train_mnrl.json hoặc train_triplet.json
        val_data_path: Path to val_mnrl.json hoặc val_triplet.json
        num_epochs: Số epochs
        batch_size: Batch size
        learning_rate: Learning rate
        loss_type: "mnrl" hoặc "triplet"
        save_dir: Directory để save checkpoints
    """
    import json
    
    # Load training data
    print(f"Loading training data from {train_data_path}...")
    with open(train_data_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    print(f"✓ Loaded {len(train_data)} training examples")
    
    # Load validation data
    if val_data_path and os.path.exists(val_data_path):
        print(f"Loading validation data from {val_data_path}...")
        with open(val_data_path, 'r', encoding='utf-8') as f:
            val_data = json.load(f)
        print(f"✓ Loaded {len(val_data)} validation examples")
    else:
        val_data = None
    
    # Train
    trained_model = train_with_sentence_transformers(
        train_data=train_data,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        loss_type=loss_type,
        save_dir=save_dir,
    )
    
    return trained_model


# Alias for backward compatibility
def train_zalo_legal_dataset(**kwargs):
    """Alias for train_legal_dataset (backward compatibility)."""
    return train_legal_dataset(**kwargs)


def train_bkai_legal_dataset(
    train_data_path: str = "../../data/legal_dataset/bkai/processed/train_mnrl.json",
    val_data_path: str = "../../data/legal_dataset/bkai/processed/val_mnrl.json",
    **kwargs
):
    """Train FusionModel với BKAI Legal Dataset."""
    return train_legal_dataset(
        train_data_path=train_data_path,
        val_data_path=val_data_path,
        save_dir=kwargs.pop("save_dir", "./checkpoints_bkai"),
        **kwargs
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train FusionModel")
    parser.add_argument("--dataset", type=str, default="sample", choices=["sample", "zalo", "bkai"],
                        help="Dataset to use: 'sample' for demo, 'zalo' for Zalo, 'bkai' for BKAI")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--loss", type=str, default="mnrl", choices=["mnrl", "triplet", "cosine"],
                        help="Loss type")
    parser.add_argument("--save_dir", type=str, default="./checkpoints_st", help="Save directory")
    
    args = parser.parse_args()
    
    if args.dataset == "zalo":
        # Train với Zalo Legal Dataset
        print("\n" + "="*50)
        print("Training with Zalo Legal Dataset")
        print("="*50 + "\n")
        
        trained_model = train_legal_dataset(
            train_data_path="../../data/legal_dataset/zalo/processed/train_mnrl.json",
            val_data_path="../../data/legal_dataset/zalo/processed/val_mnrl.json",
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            loss_type=args.loss,
            save_dir=args.save_dir if args.save_dir != "./checkpoints_st" else "./checkpoints_zalo",
        )
    elif args.dataset == "bkai":
        # Train với BKAI Legal Dataset
        print("\n" + "="*50)
        print("Training with BKAI Legal Dataset")
        print("="*50 + "\n")
        
        trained_model = train_legal_dataset(
            train_data_path="../../data/legal_dataset/bkai/processed/train_mnrl.json",
            val_data_path="../../data/legal_dataset/bkai/processed/val_mnrl.json",
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            loss_type=args.loss,
            save_dir=args.save_dir if args.save_dir != "./checkpoints_st" else "./checkpoints_bkai",
        )
    else:
        # Train với sample data
        sample_data = [
            {
                "query": "Pháp nhân có được hưởng di sản thừa kế không?",
                "positive": "Người thừa kế không là cá nhân có quyền hưởng di sản theo di chúc.",
                "negative": "Hợp đồng mua bán nhà đất phải được công chứng.",
            },
            {
                "query": "Quy định về thừa kế theo pháp luật?",
                "positive": "Thừa kế theo pháp luật áp dụng khi không có di chúc hợp pháp.",
                "negative": "Công ty cổ phần phải có ít nhất 3 cổ đông.",
            },
            {
                "query": "Điều kiện kết hôn theo luật Việt Nam?",
                "positive": "Nam từ đủ 20 tuổi, nữ từ đủ 18 tuổi mới được kết hôn.",
                "negative": "Thuế thu nhập cá nhân được tính theo biểu thuế lũy tiến.",
            },
            {
                "query": "Quyền sở hữu tài sản là gì?",
                "positive": "Quyền sở hữu bao gồm quyền chiếm hữu, quyền sử dụng và quyền định đoạt tài sản.",
                "negative": "Bảo hiểm xã hội là chế độ bắt buộc đối với người lao động.",
            },
        ]
        
        trained_model = train_with_sentence_transformers(
            train_data=sample_data,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            loss_type=args.loss,
            save_dir=args.save_dir,
        )
    
    # Test encoding
    print("\n--- Testing encoding ---")
    test_sentences = [
        "Pháp nhân có quyền thừa kế không?",
        "Quy định về kết hôn ở Việt Nam?",
    ]
    embeddings = trained_model.encode(test_sentences)
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Compute similarity
    similarity = F.cosine_similarity(embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0))
    print(f"Similarity between sentences: {similarity.item():.4f}")

