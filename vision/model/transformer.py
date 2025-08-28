import torch
import torch.nn as nn
import lightning as L
from torchtune.modules import FeedForward
from vision.model.tokenizer import special_tokens_to_embeddings


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.scale * normed


class PreNormTransformerLayer(nn.Module):
    def __init__(self, attn, mlp, dim):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = attn
        self.norm2 = RMSNorm(dim)
        self.mlp = mlp

    def forward(self, x):
        # residual scaling to avoid variance blowup
        normed_x = self.norm1(x)
        # PyTorch MultiheadAttention expects (query, key, value) and returns (output, weights)
        attn_output, _ = self.attn(normed_x, normed_x, normed_x)
        x = x + (attn_output / (2**0.5))
        x = x + (self.mlp(self.norm2(x)) / (2**0.5))
        return x


class ChessModel(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters(config)

        input_dim = config.emb_dim
        hidden_dim = config.hidden_dim

        ff_config = {
            "gate_proj": nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU()),
            "down_proj": nn.Linear(hidden_dim, input_dim),
        }

        self.token_embedding = nn.Embedding(
            config.vocabulary_size, config.emb_dim, padding_idx=0
        )
        self.positional_embedding = nn.Embedding(config.max_seq_len, config.emb_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                PreNormTransformerLayer(
                    attn=nn.MultiheadAttention(
                        embed_dim=config.emb_dim,
                        num_heads=config.num_heads,
                        dropout=config.attn_dropout,
                        bias=config.qkv_bias,
                        batch_first=True,
                    ),
                    mlp=FeedForward(**ff_config),
                    dim=config.emb_dim,
                )
                for _ in range(config.transformer_layers)
            ]
        )

        self.final_norm = RMSNorm(config.emb_dim)
        self.out_head = nn.Linear(config.emb_dim, config.vocabulary_size, bias=False)

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

        output = self(input_ids)
        output = output.view(-1, output.size(-1))  # (batch*seq_len, vocab_size)
        target = target_ids.view(-1)  # (batch*seq_len,)

        mask = target != special_tokens_to_embeddings["<|pad|>"]
        total = mask.sum().item()

        if not mask.any() or total == 0:
            # All tokens are padding, skip this batch
            return None

        if mask.shape[0] != output.shape[0]:
            # TODO: Determine if still needed and log this if so
            return None

        loss = nn.functional.cross_entropy(output[mask], target[mask])

        if torch.isnan(loss) or torch.isinf(loss):
            self.log(f"{stage}_loss_exploded", loss)
            return None

        pred = output[mask].argmax(dim=1)
        correct = (
            pred.eq(target[mask]).sum().item()
        )  # TODO: target[mask].view_as(pred)?
        accuracy = correct / total if total > 0 else 0.0
        perplexity = torch.exp(loss)

        self.log(
            f"{stage}_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            f"{stage}_perplexity",
            perplexity,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            f"{stage}_accuracy",
            accuracy,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return {"loss": loss, "accuracy": accuracy, "perplexity": perplexity}

    def training_step(self, batch, batch_idx):
        result = self._shared_step(batch, batch_idx, "train")
        if result is None:
            return None
        return result["loss"]

    def validation_step(self, batch, batch_idx):
        result = self._shared_step(batch, batch_idx, "validate")
        if result is None:
            return None
        return result

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.01)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

    # TODO: Probably not needed
    # def on_save_checkpoint(self, checkpoint):
    #     if hasattr(self, "config"):
    #         checkpoint["config"] = self.config.to_dict()
    #         checkpoint["model_metadata"] = {
    #             "timestamp": datetime.now().isoformat(),
    #             "epoch": self.current_epoch,
    #             "global_step": self.global_step,
    #         }
