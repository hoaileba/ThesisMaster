import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from typing import List, Dict, Tuple, Optional, Iterator
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict
import random
import json
import os

from fusion_model import FusionModel


###############################################################################
# CachedMultipleNegativesRankingLoss - Gradient Caching for Large Batches
###############################################################################

class CachedMultipleNegativesRankingLoss:
    """
    Cached Multiple Negatives Ranking Loss for large batch training.
    
    Implements gradient caching technique from sentence-transformers:
    1. Forward pass in mini-batches WITHOUT gradients → cache embeddings
    2. Re-forward with gradients and compute loss on full batch
    3. Backprop through fusion layers only
    
    This allows training with batch sizes far beyond GPU memory.
    
    Reference: https://www.sbert.net/docs/package_reference/sentence_transformer/losses.html#cachedmultiplenegativesrankingloss
    """
    
    def __init__(
        self,
        model: FusionModel,
        temperature: float = 0.05,
        mini_batch_size: int = 16,
        use_amp: bool = True,
    ):
        self.model = model
        self.temperature = temperature
        self.mini_batch_size = mini_batch_size
        self.use_amp = use_amp
        self.cross_entropy = nn.CrossEntropyLoss()
    
    @torch.no_grad()
    def _encode_no_grad(self, texts: List[str]) -> torch.Tensor:
        """Encode texts WITHOUT gradients (for caching)."""
        all_embeddings = []
        
        for i in range(0, len(texts), self.mini_batch_size):
            batch_texts = texts[i:i + self.mini_batch_size]
            with autocast(enabled=self.use_amp):
                output = self.model(batch_texts)
                embeddings = output.fused_hidden_states
            all_embeddings.append(embeddings)
        
        return torch.cat(all_embeddings, dim=0)
    
    def _encode_with_grad(self, texts: List[str], cached_emb: torch.Tensor) -> torch.Tensor:
        """
        Re-encode texts WITH gradients.
        
        Uses cached embeddings as reference to ensure same results,
        but creates computation graph for backpropagation.
        """
        all_embeddings = []
        
        for i in range(0, len(texts), self.mini_batch_size):
            batch_texts = texts[i:i + self.mini_batch_size]
            with autocast(enabled=self.use_amp):
                output = self.model(batch_texts)
                embeddings = output.fused_hidden_states
            all_embeddings.append(embeddings)
        
        return torch.cat(all_embeddings, dim=0)
    
    def __call__(
        self,
        queries: List[str],
        positives: List[str],
        optimizer: torch.optim.Optimizer,
        scaler: Optional[GradScaler] = None,
    ) -> float:
        """
        Compute cached MNRL loss and perform optimization step.
        
        Steps:
        1. Cache embeddings (no grad) for full batch
        2. Re-encode with grad and compute loss
        3. Backward pass through fusion layers
        """
        device = self.model.device
        
        # Step 1: Cache embeddings without gradients
        query_cache = self._encode_no_grad(queries)
        pos_cache = self._encode_no_grad(positives)
        
        # Step 2: Re-encode with gradients and compute loss
        optimizer.zero_grad()
        
        # Forward with gradients
        self.model.self_attention.train()
        self.model.fusion_mlp.train()
        
        query_emb = self._encode_with_grad(queries, query_cache)
        pos_emb = self._encode_with_grad(positives, pos_cache)
        
        # Compute InfoNCE loss
        with autocast(enabled=self.use_amp):
            sim_matrix = torch.matmul(query_emb, pos_emb.T) / self.temperature
            labels = torch.arange(query_emb.size(0), device=device)
            loss = self.cross_entropy(sim_matrix, labels)
        
        # Step 3: Backward
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        return loss.item()


class MNRLDataset(Dataset):
    """Dataset for MNRL training: (query, positive) pairs with positive_id for deduplication."""
    
    def __init__(self, data: List[Dict[str, str]]):
        """
        Args:
            data: List of dicts with keys: 'query', 'positive', 'positive_id' (optional)
        """
        self.data = data
        # Build positive_id -> indices mapping for UniquePositiveBatchSampler
        self.positive_to_indices = defaultdict(list)
        for idx, item in enumerate(data):
            pos_id = item.get('positive_id', hash(item['positive']))
            self.positive_to_indices[pos_id].append(idx)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[str, str]:
        item = self.data[idx]
        return item['query'], item['positive']
    
    def get_positive_id(self, idx: int) -> int:
        """Get positive_id for a sample index."""
        item = self.data[idx]
        return item.get('positive_id', hash(item['positive']))


class TripletDataset(Dataset):
    """Dataset for triplet training: (query, positive, negative)."""
    
    def __init__(self, data: List[Dict[str, str]]):
        """
        Args:
            data: List of dicts with keys: 'query', 'positive', 'negative'
        """
        self.data = data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[str, str, str]:
        item = self.data[idx]
        return item['query'], item['positive'], item.get('negative', '')


class UniquePositiveBatchSampler(Sampler):
    """
    BatchSampler đảm bảo không có 2 sample với cùng positive trong cùng batch.
    
    QUAN TRỌNG cho in-batch negatives: Nếu 2 sample có cùng positive text trong 
    cùng batch, khi tính in-batch negatives, positive của sample A sẽ bị coi là 
    negative của sample B (false negative), làm giảm chất lượng training.
    
    Thuật toán:
    1. Group các sample theo positive_id
    2. Với mỗi batch, chỉ lấy tối đa 1 sample từ mỗi positive group
    3. Shuffle để đảm bảo randomness
    """
    
    def __init__(
        self,
        dataset: MNRLDataset,
        batch_size: int,
        drop_last: bool = False,
        shuffle: bool = True,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        
        # Get positive_id -> indices mapping from dataset
        self.positive_to_indices = dataset.positive_to_indices
        self.num_positives = len(self.positive_to_indices)
        
        # Stats
        self._print_stats()
    
    def _print_stats(self):
        """Print sampler statistics."""
        total_samples = len(self.dataset)
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
        """
        Yield batches where each batch has unique positives.
        
        Strategy: 
        - Shuffle all positive groups
        - For each positive group, pick one random sample index
        - Fill batches ensuring no duplicate positives
        """
        # Collect one index per positive group (shuffle within group)
        indices_pool = []
        positive_ids = list(self.positive_to_indices.keys())
        
        if self.shuffle:
            random.shuffle(positive_ids)
        
        for pos_id in positive_ids:
            group_indices = self.positive_to_indices[pos_id]
            # Pick one random sample from this positive group
            chosen_idx = random.choice(group_indices)
            indices_pool.append(chosen_idx)
        
        # Yield batches
        batch = []
        for idx in indices_pool:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        
        # Handle remaining samples
        if len(batch) > 0 and not self.drop_last:
            yield batch
    
    def __len__(self) -> int:
        """Return number of batches per epoch."""
        if self.drop_last:
            return self.num_positives // self.batch_size
        else:
            return (self.num_positives + self.batch_size - 1) // self.batch_size


###############################################################################
# Loss Functions - Using nn.CrossEntropyLoss for simplicity
###############################################################################

class ContrastiveLoss(nn.Module):
    """Triplet contrastive loss using cosine similarity."""
    
    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin
    
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        pos_sim = F.cosine_similarity(anchor, positive, dim=1)
        neg_sim = F.cosine_similarity(anchor, negative, dim=1)
        loss = F.relu(self.margin - pos_sim + neg_sim)
        return loss.mean()


class FusionTrainer:
    """
    Trainer for FusionModel with CachedMultipleNegativesRankingLoss.
    
    Features:
    - Gradient caching for large batch training
    - Mixed precision (AMP) support
    - Frozen pretrained models (only train fusion layers)
    """
    
    def __init__(
        self,
        model: FusionModel,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        loss_type: str = "infonce",
        temperature: float = 0.05,
        device: str = "cuda",
        log_dir: str = None,
        mini_batch_size: int = 16,
        use_amp: bool = True,
    ):
        self.model = model
        self.device = device
        self.loss_type = loss_type
        self.learning_rate = learning_rate
        self.temperature = temperature
        self.global_step = 0
        self.mini_batch_size = mini_batch_size
        self.use_amp = use_amp
        
        # Mixed precision scaler
        self.scaler = GradScaler() if use_amp else None
        
        # TensorBoard writer
        if log_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = f"./runs/fusion_{loss_type}_{timestamp}"
        self.writer = SummaryWriter(log_dir=log_dir)
        print(f"✓ TensorBoard logging to: {log_dir}")
        
        # Freeze pretrained models
        self._freeze_pretrained()
        
        # Optimize fusion layers (self_attention + fusion_mlp)
        trainable_params = (
            list(self.model.self_attention.parameters()) +
            list(self.model.fusion_mlp.parameters())
        )
        self.optimizer = AdamW(
            trainable_params,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        
        # Loss functions
        if loss_type == "triplet":
            self.criterion = ContrastiveLoss(margin=0.5)
            self.cached_loss = None
        else:
            self.criterion = nn.CrossEntropyLoss()
            # CachedMNRL for large batch InfoNCE
            self.cached_loss = CachedMultipleNegativesRankingLoss(
                model=model,
                temperature=temperature,
                mini_batch_size=mini_batch_size,
                use_amp=use_amp,
            )
        
        self._print_trainable_params()
        
        # Log hyperparameters
        self.writer.add_text("hyperparameters", f"""
        - learning_rate: {learning_rate}
        - weight_decay: {weight_decay}
        - loss_type: {loss_type}
        - temperature: {temperature}
        - mini_batch_size: {mini_batch_size} (gradient cache)
        - use_amp: {use_amp}
        """)
    
    def _freeze_pretrained(self):
        """Freeze encoder and decoder pretrained models."""
        # Freeze encoder
        for param in self.model.encoder_model.parameters():
            param.requires_grad = False
        
        # Freeze decoder
        for param in self.model.decoder_model.parameters():
            param.requires_grad = False
        
        print("✓ Frozen pretrained encoder and decoder models")
    
    def _print_trainable_params(self):
        """Print trainable parameters info."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        print(f"\n{'='*50}")
        print(f"Total parameters:     {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Frozen parameters:    {frozen_params:,}")
        print(f"Trainable ratio:      {trainable_params/total_params*100:.2f}%")
        print(f"{'='*50}\n")
    
    def train_step(
        self,
        queries: List[str],
        positives: List[str],
        negatives: List[str] = None,
    ) -> float:
        """
        Training step with CachedMultipleNegativesRankingLoss.
        
        For InfoNCE: Uses gradient caching for large batch training
        For Triplet: Direct forward pass (no caching needed)
        """
        if self.loss_type == "triplet" and negatives is not None:
            # Triplet loss: direct forward (smaller batches typically)
            self.model.self_attention.train()
            self.model.fusion_mlp.train()
            self.optimizer.zero_grad()
            
            with autocast(enabled=self.use_amp):
                query_output = self.model(queries)
                pos_output = self.model(positives)
                neg_output = self.model(negatives)
                
                query_emb = query_output.fused_hidden_states
                pos_emb = pos_output.fused_hidden_states
                neg_emb = neg_output.fused_hidden_states
                
                loss = self.criterion(query_emb, pos_emb, neg_emb)
            
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            return loss.item()
        
        # InfoNCE with CachedMNRL for large batch training
        return self.cached_loss(
            queries=queries,
            positives=positives,
            optimizer=self.optimizer,
            scaler=self.scaler,
        )
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
    ) -> float:
        """Train one epoch."""
        total_loss = 0.0
        num_batches = len(dataloader)
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            if self.loss_type == "triplet":
                queries, positives, negatives = batch
                loss = self.train_step(list(queries), list(positives), list(negatives))
            else:
                # MNRL: only query and positive
                if len(batch) == 2:
                    queries, positives = batch
                else:
                    queries, positives, _ = batch
                loss = self.train_step(list(queries), list(positives))
            
            total_loss += loss
            self.global_step += 1
            
            # Log step loss to TensorBoard
            self.writer.add_scalar("Loss/step", loss, self.global_step)
            
            pbar.set_postfix({"loss": f"{loss:.4f}"})
        
        avg_loss = total_loss / num_batches
        
        # Log epoch metrics to TensorBoard
        self.writer.add_scalar("Loss/epoch", avg_loss, epoch)
        self.writer.add_scalar("Learning_Rate", self.learning_rate, epoch)
        
        return avg_loss
    
    def close(self):
        """Close TensorBoard writer."""
        self.writer.close()
    
    def save_checkpoint(self, path: str, epoch: int, loss: float, best_loss: float = None):
        """Save model checkpoint (fusion layers only)."""
        checkpoint = {
            "epoch": epoch,
            "loss": loss,
            "best_loss": best_loss if best_loss is not None else loss,
            "global_step": self.global_step,
            "self_attention_state_dict": self.model.self_attention.state_dict(),
            "fusion_mlp_state_dict": self.model.fusion_mlp.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        # Save scaler state if using AMP
        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        
        torch.save(checkpoint, path)
        print(f"✓ Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str) -> Tuple[int, float, float]:
        """
        Load model checkpoint for resuming training.
        
        Returns:
            Tuple of (epoch, loss, best_loss)
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load model states
        # Support both old (cross_attention) and new (self_attention) checkpoints
        if "self_attention_state_dict" in checkpoint:
            self.model.self_attention.load_state_dict(checkpoint["self_attention_state_dict"])
        elif "cross_attention_state_dict" in checkpoint:
            print("⚠️ Old checkpoint format (cross_attention) - cannot load into new self_attention architecture")
        if "fusion_mlp_state_dict" in checkpoint:
            self.model.fusion_mlp.load_state_dict(checkpoint["fusion_mlp_state_dict"])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # Load scaler state if available
        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        # Load global step
        if "global_step" in checkpoint:
            self.global_step = checkpoint["global_step"]
        
        best_loss = checkpoint.get("best_loss", checkpoint["loss"])
        
        print(f"✓ Loaded checkpoint from {path}")
        print(f"  - Resuming from epoch {checkpoint['epoch']}")
        print(f"  - Last loss: {checkpoint['loss']:.4f}")
        print(f"  - Best loss: {best_loss:.4f}")
        print(f"  - Global step: {self.global_step}")
        
        return checkpoint["epoch"], checkpoint["loss"], best_loss


def train(
    train_data: List[Dict[str, str]],
    num_epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 1e-4,
    save_dir: str = "./checkpoints",
    loss_type: str = "infonce",
    log_dir: str = None,
    mini_batch_size: int = 16,
    use_amp: bool = True,
    resume_from: str = None,
):
    """
    Main training function with CachedMultipleNegativesRankingLoss.
    
    Args:
        train_data: Training data
        num_epochs: Number of epochs
        batch_size: Effective batch size (can be large with gradient caching)
        learning_rate: Learning rate
        save_dir: Checkpoint save directory
        loss_type: "infonce" or "triplet"
        log_dir: TensorBoard log directory
        mini_batch_size: Mini-batch size for gradient caching (fits in GPU)
        use_amp: Use mixed precision training
        resume_from: Path to checkpoint for resuming training (None for fresh start)
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Determine log directory
    if log_dir is None:
        if resume_from:
            # Try to use existing log dir from checkpoint directory
            checkpoint_dir = os.path.dirname(resume_from)
            runs_dir = os.path.join(checkpoint_dir, "runs")
            if os.path.exists(runs_dir):
                # Find latest run directory
                existing_runs = sorted([d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))])
                if existing_runs:
                    log_dir = os.path.join(runs_dir, existing_runs[-1])
        
        if log_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = os.path.join(save_dir, f"runs/fusion_{loss_type}_{timestamp}")
    
    # Print training config
    print(f"\n{'='*60}")
    if resume_from:
        print(f"🔄 Resuming Training with CachedMNRL")
        print(f"   - Resume from: {resume_from}")
    else:
        print(f"🚀 Training with CachedMultipleNegativesRankingLoss")
    print(f"   - Effective batch size: {batch_size}")
    print(f"   - Mini-batch size (GPU): {mini_batch_size}")
    print(f"   - Mixed Precision (AMP): {use_amp}")
    print(f"   - Loss: {loss_type}")
    print(f"   - Total epochs: {num_epochs}")
    print(f"{'='*60}\n")
    
    # Initialize model
    print("Initializing FusionModel...")
    model = FusionModel(config=None)
    model.to("cuda")
    
    # Initialize trainer
    trainer = FusionTrainer(
        model=model,
        learning_rate=learning_rate,
        loss_type=loss_type,
        log_dir=log_dir,
        mini_batch_size=mini_batch_size,
        use_amp=use_amp,
    )
    
    # Resume from checkpoint if provided
    start_epoch = 1
    best_loss = float("inf")
    
    if resume_from and os.path.exists(resume_from):
        print(f"\n📂 Loading checkpoint: {resume_from}")
        last_epoch, last_loss, best_loss = trainer.load_checkpoint(resume_from)
        start_epoch = last_epoch + 1
        print(f"   → Starting from epoch {start_epoch}")
    elif resume_from:
        print(f"⚠️ Checkpoint not found: {resume_from}")
        print("   → Starting fresh training")
    
    # Log training config
    trainer.writer.add_hparams(
        {
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "mini_batch_size": mini_batch_size,
            "learning_rate": learning_rate,
            "loss_type": loss_type,
            "use_amp": use_amp,
            "num_samples": len(train_data),
            "resume_from_epoch": start_epoch - 1,
        },
        {"hparam/placeholder": 0},
    )
    
    # Create dataset and dataloader based on loss type
    if loss_type == "triplet":
        dataset = TripletDataset(train_data)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )
    else:
        dataset = MNRLDataset(train_data)
        batch_sampler = UniquePositiveBatchSampler(
            dataset=dataset,
            batch_size=batch_size,
            drop_last=False,
            shuffle=True,
        )
        dataloader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=0,
        )
    
    # Training loop (continue from start_epoch)
    for epoch in range(start_epoch, num_epochs + 1):
        avg_loss = trainer.train_epoch(dataloader, epoch)
        print(f"Epoch {epoch}/{num_epochs} - Average Loss: {avg_loss:.4f}")
        
        # Save best checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            trainer.save_checkpoint(
                os.path.join(save_dir, "best_model.pt"),
                epoch,
                avg_loss,
                best_loss=best_loss,
            )
            trainer.writer.add_scalar("Loss/best", best_loss, epoch)
        
        # Save latest checkpoint (for resuming)
        trainer.save_checkpoint(
            os.path.join(save_dir, "latest_checkpoint.pt"),
            epoch,
            avg_loss,
            best_loss=best_loss,
        )
        
        # Also save epoch-specific checkpoint
        trainer.save_checkpoint(
            os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pt"),
            epoch,
            avg_loss,
            best_loss=best_loss,
        )
    
    # Close TensorBoard writer
    trainer.close()
    
    print(f"\n✓ Training completed! Best loss: {best_loss:.4f}")
    print(f"✓ TensorBoard logs saved to: {log_dir}")
    print(f"  Run: tensorboard --logdir {log_dir}")
    return model


def load_training_data(path: str) -> List[Dict[str, str]]:
    """Load training data from JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def train_legal_dataset(
    train_data_path: str,
    val_data_path: Optional[str] = None,
    num_epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 1e-4,
    loss_type: str = "infonce",
    save_dir: str = "./checkpoints",
    log_dir: str = None,
    mini_batch_size: int = 16,
    use_amp: bool = True,
    resume_from: str = None,
):
    """
    Train FusionModel với Legal Dataset (Zalo hoặc BKAI).
    
    Args:
        train_data_path: Path to training data JSON
        val_data_path: Path to validation data JSON (optional)
        num_epochs: Number of training epochs
        batch_size: Effective batch size (with gradient caching)
        learning_rate: Learning rate
        loss_type: "infonce" or "triplet"
        save_dir: Directory to save checkpoints
        log_dir: TensorBoard log directory
        mini_batch_size: Mini-batch size for gradient caching
        use_amp: Use mixed precision
        resume_from: Path to checkpoint for resuming training
    """
    print(f"Loading training data from {train_data_path}...")
    train_data = load_training_data(train_data_path)
    print(f"✓ Loaded {len(train_data)} training examples")
    
    if val_data_path and os.path.exists(val_data_path):
        print(f"Loading validation data from {val_data_path}...")
        val_data = load_training_data(val_data_path)
        print(f"✓ Loaded {len(val_data)} validation examples")
    
    trained_model = train(
        train_data=train_data,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        loss_type=loss_type,
        save_dir=save_dir,
        log_dir=log_dir,
        mini_batch_size=mini_batch_size,
        use_amp=use_amp,
        resume_from=resume_from,
    )
    
    return trained_model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train FusionModel with CachedMNRL")
    parser.add_argument("--dataset", type=str, default="sample", choices=["sample", "zalo", "bkai"],
                        help="Dataset to use")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Effective batch size (with caching)")
    parser.add_argument("--mini_batch_size", type=int, default=16, help="Mini-batch size for GPU")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--loss", type=str, default="infonce", choices=["infonce", "triplet"],
                        help="Loss type")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Save directory")
    parser.add_argument("--no_amp", action="store_true", help="Disable mixed precision")
    parser.add_argument("--resume", type=str, default=None, 
                        help="Path to checkpoint for resuming training (e.g., ./checkpoints/latest_checkpoint.pt)")
    
    args = parser.parse_args()
    use_amp = not args.no_amp
    
    # Determine save_dir and resume_from
    if args.dataset == "zalo":
        save_dir = args.save_dir if args.save_dir != "./checkpoints" else "./checkpoints_zalo"
    elif args.dataset == "bkai":
        save_dir = args.save_dir if args.save_dir != "./checkpoints" else "./checkpoints_bkai"
    else:
        save_dir = args.save_dir
    
    # Auto-detect latest checkpoint if --resume is "auto" or "latest"
    resume_from = args.resume
    if resume_from in ["auto", "latest"]:
        latest_path = os.path.join(save_dir, "latest_checkpoint.pt")
        if os.path.exists(latest_path):
            resume_from = latest_path
            print(f"📂 Auto-detected checkpoint: {resume_from}")
        else:
            print(f"⚠️ No checkpoint found at {latest_path}, starting fresh")
            resume_from = None
    
    if args.dataset == "zalo":
        print("\n" + "="*50)
        print("Training with Zalo Legal Dataset")
        print("="*50 + "\n")
        
        data_file = "train_mnrl.json" if args.loss == "infonce" else "train_triplet.json"
        val_file = "val_mnrl.json" if args.loss == "infonce" else "val_triplet.json"
        
        trained_model = train_legal_dataset(
            train_data_path=f"../../data/legal_dataset/zalo/processed/{data_file}",
            val_data_path=f"../../data/legal_dataset/zalo/processed/{val_file}",
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            loss_type=args.loss,
            save_dir=save_dir,
            mini_batch_size=args.mini_batch_size,
            use_amp=use_amp,
            resume_from=resume_from,
        )
        
    elif args.dataset == "bkai":
        print("\n" + "="*50)
        print("Training with BKAI Legal Dataset")
        print("="*50 + "\n")
        
        data_file = "train_mnrl.json" if args.loss == "infonce" else "train_triplet.json"
        val_file = "val_mnrl.json" if args.loss == "infonce" else "val_triplet.json"
        
        trained_model = train_legal_dataset(
            train_data_path=f"../../data/legal_dataset/bkai/processed/{data_file}",
            val_data_path=f"../../data/legal_dataset/bkai/processed/{val_file}",
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            loss_type=args.loss,
            save_dir=save_dir,
            mini_batch_size=args.mini_batch_size,
            use_amp=use_amp,
            resume_from=resume_from,
        )
        
    else:
        print("\n" + "="*50)
        print("Training with Sample Data")
        print("="*50 + "\n")
        
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
        
        trained_model = train(
            train_data=sample_data,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            loss_type=args.loss,
            save_dir=save_dir,
            mini_batch_size=args.mini_batch_size,
            use_amp=use_amp,
            resume_from=resume_from,
        )
    
    print("\n✓ Training completed!")

