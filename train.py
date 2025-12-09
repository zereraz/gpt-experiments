import torch.nn.functional as F
import torch


def train(model, dataloader, epochs, device="mps"):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[train] Starting: {total_params:,} params, {epochs} epochs, {len(dataloader)} batches")

    for epoch in range(epochs):
        total_loss = 0

        for i, batch in enumerate(dataloader):
            batch = batch.to(device)
            x, y = batch[:, :-1], batch[:, 1:]

            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

            if (i + 1) % 1000 == 0:
                print(
                    f"[train] Epoch {epoch + 1}/{epochs}, Batch {i + 1}/{len(dataloader)}, Loss: {loss.item():.4f}"
                )

        avg_loss = total_loss / len(dataloader)
        print(f"[train] Epoch {epoch + 1}/{epochs} done, Avg Loss: {avg_loss:.4f}")


def save_model(model, path: str):
    torch.save(model.state_dict(), path)
    print(f"[model] Saved to {path}")


def load_model(model, path: str, device: str = "mps"):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    print(f"[model] Loaded from {path}")
    return model


def generate(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 50,
    temperature: float = 1.0,
    device: str = "mps",
):
    model.eval()
    tokens = tokenizer.encode(prompt)

    with torch.no_grad():
        for _ in range(max_tokens):
            x = torch.tensor([tokens[-128:]], device=device)
            logits = model(x)[0, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            tokens.append(next_token)

    return tokenizer.decode(tokens)
