from typing import Tuple
import torch
from generate import generate_text
from model import Config, GPT
import torch.nn.functional as F
import einops
import warnings

from utils import encode, load_text
warnings.filterwarnings("ignore", message="MPS: nonzero op is supported natively starting from macOS 14.0")

def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    # For MPS
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

set_seed()




if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not "
              "built with MPS enabled.")
        device = torch.device('cpu')
    else:
        print("MPS not available because the current MacOS version is not 12.3+ "
              "and/or you do not have an MPS-enabled device on this machine.")

else:
    device = torch.device("mps")

config = Config()
text = load_text()
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}




tokens = torch.tensor(encode(text), device=device)
train_num = round(0.9*len(tokens))
train_set = tokens[:train_num]
val_set = tokens[train_num:]

def get_batch(is_train:bool) -> Tuple[torch.Tensor, torch.Tensor]:
    data = train_set if is_train else val_set
    indices = torch.randint(high=(len(data)-config.context_length-1), size=(config.batch_size,))
    x = torch.stack([data[i:i+config.context_length] for i in indices])
    y = torch.stack([data[i+1:i+config.context_length+1] for i in indices])
    x = x.to(device)
    y = y.to(device)

    return x,y

model = GPT(config)
model.to(device)

def eval_model(model: GPT) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(10):  # run 10 batches
            val_indices, targets = get_batch(False)
            logits = model(val_indices)
            reshaped_logits = einops.rearrange(logits, 'b t c -> (b t) c')
            reshaped_targets = einops.rearrange(targets, 'b t -> (b t)')
            loss = F.cross_entropy(reshaped_logits, reshaped_targets)
            losses.append(float(loss.item()))
    model.train()
    return sum(losses) / len(losses)  # average loss

def training_loop():
   optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
   best_val_loss = float('inf')

   for i in range(config.max_iters):
       optimizer.zero_grad()
       indices, targets = get_batch(True)
       logits = model(indices)
       reshaped_logits = einops.rearrange(logits, 'b t c -> (b t) c')
       reshaped_targets = einops.rearrange(targets, 'b t -> (b t)')
       smoothing = 0.1
       n_classes = reshaped_logits.size(-1)  # Use reshaped_logits instead of logits
       targets_one_hot = F.one_hot(reshaped_targets, n_classes)  # Use reshaped_targets
       targets_smooth = (1 - smoothing) * targets_one_hot + smoothing / n_classes
       loss = -(targets_smooth * F.log_softmax(reshaped_logits, dim=-1)).sum(dim=-1).mean()
       loss.backward()
       optimizer.step()

       if i % 100 == 0 or i == config.max_iters - 1:
           val_loss = eval_model(model)
           print(f"iter {i}: train loss {loss.item():.4f}, val loss {val_loss:.4f}")

           if val_loss < best_val_loss:
               print(f"Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}. Saving model...")
               best_val_loss = val_loss
               torch.save(model.state_dict(), 'best_model.pt')

           with torch.no_grad():
               prompt = "First Citizen:\n"
               print("\nSample text:")
               print((generate_text(model,prompt=prompt,max_tokens=10,temperature=0.8,device=device)))
               model.train()

if __name__ == "__main__":
    training_loop()

