from dataclasses import dataclass
import torch.nn as nn
import torch.nn.functional as F
import torch
import einops #type: ignore

@dataclass
class Config:
    batch_size = 8
    context_length = 256
    max_iters = 5000
    vocab_size = 65
    embedding_dim = 384
    num_heads = 6
    head_size = 64
    num_layers = 6
    dropout = 0.5


def assert_shapes_BTC(x:torch.Tensor, config: Config) -> None:
    assert len(x.shape) == 3, f"Expected 3-d input, but got {x.shape}"
    assert x.shape[2] == config.embedding_dim, f"Expected last dimension to be {config.embedding_dim} but got {x.shape[2]}"


class SingleAttentionBlock(nn.Module):
    def __init__(self, config:Config):
        super().__init__()
        self.config = config
        self.q_weight = nn.Linear(config.embedding_dim, config.head_size, bias=False)
        self.k_weight = nn.Linear(config.embedding_dim, config.head_size, bias=False)
        self.v_weight = nn.Linear(config.embedding_dim, config.head_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert_shapes_BTC(x, self.config)
        B, T, C = x.shape
        Q = self.q_weight(x) # b, t, h
        K = self.k_weight(x)
        V = self.v_weight(x)
        K_transpose = einops.rearrange(K, 'b t h -> b h t')
        attn_unscaled = Q @ K_transpose
        mask = torch.triu(torch.ones(T, T), diagonal=1).bool().to(x.device)
        attn_unscaled.masked_fill_(mask, float('-inf'))
        attn_scaled = attn_unscaled * (self.config.embedding_dim ** -0.5) # b, t, t
        attn_probs = F.softmax(attn_scaled, dim=-1)
        return attn_probs @ V #b, t, h





class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, config:Config):
        super().__init__()
        self.config = config
        self.blocks = nn.ModuleList(SingleAttentionBlock(config) for _ in range(config.num_heads))
        self.projection = nn.Linear(config.head_size * config.num_heads, config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        assert_shapes_BTC(x, self.config)
        output = torch.cat([block(x) for block in self.blocks], dim=-1)
        output = self.projection(output)
        output = self.dropout(output)
        assert_shapes_BTC(output, self.config)
        return output


class MLP(nn.Module):
    def __init__(self, config:Config):
        super().__init__()
        self.config=config
        self.linear1 = nn.Linear(config.embedding_dim, 4*config.embedding_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(4*config.embedding_dim, config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert_shapes_BTC(x, self.config)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class CombinedBlock(nn.Module):
    def __init__(self, config:Config):
        super().__init__()
        self.config = config
        self.ln1 = nn.LayerNorm(config.embedding_dim)
        self.attention = MultiHeadAttentionBlock(config)
        self.ln2 = nn.LayerNorm(config.embedding_dim)
        self.mlp = MLP(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert_shapes_BTC(x, self.config)
        x = x + self.dropout(self.attention(self.ln1(x)))
        x = x + self.dropout(self.mlp(self.ln2(x)))
        return x




class GPT(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.embedding_table = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.positional_embedding = nn.Embedding(config.context_length, config.embedding_dim)
        self.blocks = nn.Sequential(*[CombinedBlock(config) for _ in range(config.num_layers)])
        self.ln_final = nn.LayerNorm(self.config.embedding_dim)
        self.unembed_table = nn.Linear(config.embedding_dim, config.vocab_size)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
       assert len(x.shape) == 2, f"Expected shape of dimension 2 but got {x.shape}"
       assert x.shape[1] <= self.config.context_length, f"Input sequence length {x.shape[1]} exceeds maximum context length {self.config.context_length}"
       B, T = x.shape
       embedding = self.embedding_table(x)
       positional_embedding = self.positional_embedding(torch.arange(T, device=x.device))
       x = embedding + positional_embedding
       assert_shapes_BTC(x, self.config)
       x = self.blocks(x)
       assert_shapes_BTC(x, self.config)
       x = self.ln_final(x)
       assert_shapes_BTC(x, self.config)
       x = self.unembed_table(x)
       return x
