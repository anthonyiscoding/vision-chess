import torch
import torch.nn as nn
import lightning as L
from torchtune.modules import FeedForward
from vision.model.tokenizer import special_tokens_to_embeddings


class PreNormTransformerLayer(nn.Module):
    def __init__(self, attn, mlp, dim):
        super().__init__()
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=attn.num_heads,
            dim_feedforward=mlp.w1.out_features,
            dropout=attn.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

    def forward(self, x):
        return self.transformer_layer(x)


class ChessModel(L.LightningModule):
    def __init__(
        self,
        emb_dim: int,
        hidden_dim: int,
        num_heads: int,
        transformer_layers: int,
        vocabulary_size: int,
        qkv_bias: bool = True,
        max_seq_len: int = 110,
        attn_dropout: float = 0.0,
        learning_rate: float = 1e-5,
        scheduler_patience: int = 4,
        reduce_lr_by: float = 0.5,
    ):
        """Initialize the Chess Transformer Model.

        Args:
            emb_dim: Embedding dimension for tokens and positions
            hidden_dim: Hidden dimension for the feedforward layers
            num_heads: Number of attention heads in multi-head attention
            transformer_layers: Number of transformer blocks
            vocabulary_size: Size of the token vocabulary
            qkv_bias: Whether to use bias in query, key, value projections
            max_seq_len: Maximum sequence length for positional embeddings (number_of_moves - 2; ex. 102 == 50 turns per player)
            attn_dropout: Dropout rate for attention layers
            learning_rate: Learning rate for the optimizer
            scheduler_patience: How many epochs to wait for improvement before reducing learning rate
            reduce_lr_by: What how much to reduce lr by when scheduler_patience is reached (lr * reduce_lr_by)
        """
        super().__init__()
        self.save_hyperparameters()

        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.transformer_layers = transformer_layers
        self.vocabulary_size = vocabulary_size
        self.qkv_bias = qkv_bias
        self.max_seq_len = max_seq_len
        self.attn_dropout = attn_dropout
        self.learning_rate = learning_rate
        self.scheduler_patience = scheduler_patience
        self.reduce_lr_by = reduce_lr_by

        ff_config = {
            "gate_proj": nn.Linear(self.emb_dim, self.hidden_dim),
            "up_proj": nn.Linear(self.emb_dim, self.hidden_dim),
            "down_proj": nn.Linear(self.hidden_dim, self.emb_dim),
        }

        self.token_embedding = nn.Embedding(
            self.vocabulary_size,
            self.emb_dim,
            padding_idx=special_tokens_to_embeddings["<|pad|>"],
        )
        self.positional_embedding = nn.Embedding(
            self.max_seq_len,
            self.emb_dim,
        )

        self.transformer_blocks = nn.ModuleList(
            [
                PreNormTransformerLayer(
                    attn=nn.MultiheadAttention(
                        embed_dim=self.emb_dim,
                        num_heads=self.num_heads,
                        dropout=self.attn_dropout,
                        bias=self.qkv_bias,
                        batch_first=True,
                    ),
                    mlp=FeedForward(**ff_config),
                    dim=self.emb_dim,
                )
                for _ in range(self.transformer_layers)
            ]
        )

        self.final_norm = nn.RMSNorm(self.emb_dim)
        self.out_head = nn.Linear(self.emb_dim, self.vocabulary_size, bias=False)

    def forward(self, idx):
        _, seq_len = idx.shape
        token_embeds = self.token_embedding(idx)
        positional_embeds = self.positional_embedding(
            torch.arange(seq_len, device=idx.device)
        )

        x = token_embeds + positional_embeds
        for block in self.transformer_blocks:
            x = block(x)
        x = self.final_norm(x)
        return self.out_head(x)

    def _shared_step(self, batch, batch_idx, stage):
        input_ids, target_ids = batch

        # Ensure tensors are on the correct device and contiguous
        input_ids: torch.Tensor = input_ids.to(self.device).contiguous()
        target_ids: torch.Tensor = target_ids.to(self.device).contiguous()

        # # Should be no unknown inputs in the data
        # # TODO: Figure out how to do this more efficiently in a single check
        # if special_tokens_to_embeddings["<|unk|>"] in input_ids:
        #     return None
        # if special_tokens_to_embeddings["<|unk|>"] in target_ids:
        #     return None

        output: torch.Tensor = self(input_ids)
        output = output.view(-1, output.size(-1))  # (batch*seq_len, vocab_size)
        target: torch.Tensor = target_ids.view(-1)  # (batch*seq_len,)

        mask: torch.Tensor = target != special_tokens_to_embeddings["<|pad|>"]
        total = mask.sum().item()
        if not mask.any() or total == 0:
            # All tokens are padding, skip this batch
            return None

        if mask.shape[0] != output.shape[0]:
            raise ValueError(
                f"Shape mismatch: mask.shape[0] ({mask.shape[0]}) != output.shape[0] ({output.shape[0]}). "
                f"input_ids.shape: {input_ids.shape}, target_ids.shape: {target_ids.shape}, "
                f"output.shape: {output.shape}, mask.shape: {mask.shape}"
            )

        output = output[mask]
        target = target[mask]

        loss = nn.functional.cross_entropy(output, target, label_smoothing=0.05)

        if torch.isnan(loss) or torch.isinf(loss):
            self.log(f"{stage}_loss_exploded", loss)
            return None

        pred = output.argmax(dim=1)
        correct = pred.eq(target).sum().item()  # TODO: target[mask].view_as(pred)?
        accuracy = correct / total if total > 0 else 0.0
        perplexity = torch.exp(loss)

        prog_bar_display = {
            "train": {"loss": True, "perplexity": False, "accuracy": False},
            "val": {"loss": True, "perplexity": False, "accuracy": False},
        }

        self.log(
            f"{stage}_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=prog_bar_display[stage]["loss"],
            logger=True,
        )
        self.log(
            f"{stage}_perplexity",
            perplexity,
            on_step=True,
            on_epoch=True,
            prog_bar=prog_bar_display[stage]["perplexity"],
            logger=True,
        )
        self.log(
            f"{stage}_accuracy",
            accuracy,
            on_step=True,
            on_epoch=True,
            prog_bar=prog_bar_display[stage]["accuracy"],
            logger=True,
        )

        return {"loss": loss, "accuracy": accuracy, "perplexity": perplexity}

    def training_step(self, batch, batch_idx):
        result = self._shared_step(batch, batch_idx, "train")
        if result is None:
            return None
        return result["loss"]

    def validation_step(self, batch, batch_idx):
        result = self._shared_step(batch, batch_idx, "val")
        if result is None:
            return None
        return result

    def configure_optimizers(self):
        print(
            f"Optimizer Config: learning_rate: {self.learning_rate}, scheduler_patience: {self.scheduler_patience}, reduce_by_lr: {self.reduce_lr_by}"
        )
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=self.scheduler_patience,
            factor=self.reduce_lr_by,
            mode="min",
            threshold_mode="abs",
            threshold=1e-3,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }

    def on_before_optimizer_step(self, optimizer):
        self.clip_gradients(
            optimizer, gradient_clip_val=1.0, gradient_clip_algorithm="norm"
        )
