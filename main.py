import os
from tokenizer import BPETokenizer
from dataset import TextDataset
from gpt import GPT
from train import train, save_model, load_model, generate
from torch.utils.data import DataLoader

TOKENIZER_PATH = "tokenizer.json"
MODEL_PATH = "model.pt"

with open("/usr/share/dict/words", "r") as f:
    text = f.read()

# Load or train tokenizer
tokenizer = BPETokenizer(vocab_size=1000)
if os.path.exists(TOKENIZER_PATH):
    tokenizer.load(TOKENIZER_PATH)
else:
    tokenizer.train(text)
    tokenizer.save(TOKENIZER_PATH)

# Create model
model = GPT(vocab_size=1000, d_model=64, n_heads=4, n_layers=4, max_seq_len=128, d_ff=256)

# Load or train model
if os.path.exists(MODEL_PATH):
    load_model(model, MODEL_PATH)
else:
    dataset = TextDataset(text, tokenizer, seq_len=128)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    train(model, dataloader, epochs=10)
    save_model(model, MODEL_PATH)

# Generate text
print("\n[generate] Sampling from model:")
output = generate(model, tokenizer, "hello", max_tokens=50)
print(output)
