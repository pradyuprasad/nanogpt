from model import GPT, Config
import torch
from utils import encode, decode

model = GPT(Config())


def load_model(model_path: str, config: Config):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # First load to CPU
    model = GPT(config).to(device)
    state_dict = torch.load(model_path, weights_only=True, map_location='cpu')
    model.load_state_dict(state_dict)


    model.eval()
    return model, device

def sample_next_token(logits: torch.Tensor, temperature: float = 0.8) -> int:
    """Take the most likely non-space token"""
    # Get sorted indices and values
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)

    # Find first non-space token (token != 1)
    for idx, token_id in enumerate(sorted_indices):
        if token_id.item() != 1:  # If not a space
            return token_id.item()

    return int(sorted_indices[0].item())  # Fallback to most likely if somehow all are spaces


def generate_text(
    model: GPT,
    prompt: str,
    max_tokens: int,
    temperature: float = 0.8,  # Not used anymore but kept for compatibility
    device: torch.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu'),
    debug: bool = False
) -> str:
    print("Prompt:", prompt, end="", flush=True)
    print("model generated below:")
    context = torch.tensor(encode(prompt), dtype=torch.long).to(device).unsqueeze(0)

    generated = []
    for k in range(max_tokens):
        logits = model(context)
        next_token_logits = logits[0, -1, :]
        next_token = sample_next_token(next_token_logits)
        generated.append(next_token)

        if debug:
            new_text = decode([next_token])
            print(new_text, end="", flush=True)

        next_token_tensor = torch.tensor([[next_token]], device=device)
        context = torch.cat([context, next_token_tensor], dim=1)

        if context.size(1) >= model.config.context_length:
            context = context[:, -model.config.context_length:]

    print("\n")
    return prompt + decode(generated)



if __name__ == "__main__":
    config = Config()
    model, device = load_model('best_model.pt', config)

    # Generate text
    prompt = '''
    First Citizen:
    Before we proceed any further, hear me speak.
    '''



    generated_text = generate_text(
        model,
        prompt=prompt,
        max_tokens=200,  # How many tokens to generate
        temperature=0,
        device=device,  # Adjust for more/less randomness,
        debug=True
    )
    print(generated_text)
