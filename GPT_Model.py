
import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyperparameters
batch_size = 64   # how many independent sequences will we process in parallel?
block_size = 256  # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
# ------------

torch.manual_seed(1337)

# Load dataset
with open('input.txt', 'r', encoding = 'utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

# Encoder / decoder mappings
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]           # encoder: take a string, output list of integers
decode = lambda l: ''.join([itos[i] for i in l])  # decoder: integers back to string

# Train / val splits
data  = torch.tensor(encode(text), dtype = torch.long)
n     = int(0.9 * len(data))
train_data  = data[:n]
val_data    = data[n:]

# --------------------------------------------------
# get_batch(): returns (x,y) pairs of shape (B, block_size)
# --------------------------------------------------
def get_batch(split):
    data_src = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_src) - block_size, (batch_size,))

    x = torch.stack([data_src[i:i+block_size] for i in ix])
    y = torch.stack([data_src[i+1:i+block_size+1] for i in ix])

    x, y = x.to(device), y.to(device)
    return x, y

# --------------------------------------------------
# estimate_loss(): evaluates train + val loss WITHOUT 
# gradient computation (no_grad + model.eval)
# --------------------------------------------------
@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train','val']:
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            X,Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

# --------------------------------------------------
# Attention Head
# --------------------------------------------------
class Head(nn.Module):
    """ One head of self-attention """
    def __init__(self, n_embd, head_size):
        super().__init__()
        # Linear layers reduce dimension from n_embd → head_size
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # Dropout for attention weights
        self.dropout = nn.Dropout(dropout)

        # Causal mask: lower triangular matrix
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B,T,C = x.shape

        k = self.key(x)                      # (B,T,head_size)
        q = self.query(x)                    # (B,T,head_size)

        # Scaled dot‑product attention
        wei = q @ k.transpose(-2,-1)         # (B,T,hs) @ (B,hs,T) → (B,T,T)
        wei = wei * (k.shape[-1] ** -0.5)    # scale by 1/sqrt(head_size)

        # Apply causal mask
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

        # Softmax over last dimension (distribution over time)
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        # Weighted aggregation of values
        v = self.value(x)     # (B,T,hs)
        out = wei @ v         # (B,T,hs)
        return out

# --------------------------------------------------
# Multi-Head Attention
# --------------------------------------------------
class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, n_embd, num_heads, head_size):
        super().__init__()

        # Create independent heads
        self.heads = nn.ModuleList([Head(n_embd, head_size) for _ in range(num_heads)])

        # Projection: concat_heads_dim → n_embd
        self.proj = nn.Linear(head_size * num_heads, n_embd)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Run all heads in parallel, concatenate results
        out = torch.cat([h(x) for h in self.heads], dim=-1)

        # Project back to model embedding space + dropout
        out = self.dropout(self.proj(out))
        return out

# --------------------------------------------------
# FeedForward Network (position-wise MLP)
# --------------------------------------------------
class FeedForward(nn.Module):
    """ A simple linear layer followed by ReLU and projection """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),   # expand
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),   # project back
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

# --------------------------------------------------
# Transformer Block: Attention + MLP + Residual + LayerNorm
# --------------------------------------------------
class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head

        self.sa   = MultiHeadAttention(n_embd, n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1  = nn.LayerNorm(n_embd)
        self.ln2  = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Pre-LN architecture (using layer norm first then proceeding for each computation)
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# --------------------------------------------------
# Full GPT Model
# --------------------------------------------------
class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Token → embedding vectors
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)

        # Position → embedding vectors
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # Stack of Transformer Blocks
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])

        # Final Layer Norm
        self.ln_f = nn.LayerNorm(n_embd)

        # Final linear layer: project embeddings → logits over vocab
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # Initialize weights
        self.apply(self._init_weights)

    # Custom weight initialization
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # Token + position embeddings
        tok_emb = self.token_embedding_table(idx)               # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T,C)
        x = tok_emb + pos_emb                                   # (B,T,C)

        # Pass through Transformer blocks
        x = self.blocks(x)

        # Final normalization + logits
        x = self.ln_f(x)
        logits = self.lm_head(x)                                # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B2, T2, C = logits.shape
            logits = logits.view(B2*T2, C)
            targets = targets.view(B2*T2)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    # --------------------------------------------------
    # Text generation loop
    # --------------------------------------------------
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # Limit context to last block_size tokens
            idx_cond = idx[:, -block_size:]

            # Forward pass
            logits, _ = self(idx_cond)

            # Focus only on last timestep
            logits = logits[:, -1, :]  # (B, vocab_size)

            # Convert to probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample next token index
            idx_next = torch.multinomial(probs, num_samples=1)  # (B,1)

            # Append new token
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

# --------------------------------------------------
# Create model
# --------------------------------------------------
model = GPTLanguageModel().to(device)
print(f"Model params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# --------------------------------------------------
# Training Loop
# --------------------------------------------------
for iter in range(max_iters):

    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss(model)
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# --------------------------------------------------
# Text Generation
# --------------------------------------------------
context = torch.zeros((1,1), dtype=torch.long, device=device)
generated_tokens = model.generate(context, max_new_tokens=500)[0].tolist()
print(decode(generated_tokens))
#open('generated_novel.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))