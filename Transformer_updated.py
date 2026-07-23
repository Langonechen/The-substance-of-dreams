
import os
import json
import time
import textwrap
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import functional as F

from tqdm import tqdm 

# Viz and statistical work
from sklearn.decomposition import PCA

# Retrieval & Vector Math
import faiss

# NLP & ML
from sentence_transformers import SentenceTransformer, util, CrossEncoder
from transformers import pipeline, AutoTokenizer

# Chunking
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Setting random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
# create a torch.device object for tensor/model placement
device = torch.device(DEVICE)

# Extract only the text part from dataset and save it in a txt file
DATA_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
dreams_path = DATA_DIR / "dreams_emotions1.csv"
dream_text_path = DATA_DIR / "dream_text.txt"

dream_emotions_full = pd.read_csv(dreams_path)
if "dream_text" not in dream_emotions_full.columns:
    raise ValueError("dreams_emotions1.csv must contain a dream_text column")

dream_text = "\n\n".join(
    dream_emotions_full["dream_text"].dropna().astype(str).tolist()
)
dream_text_path.write_text(dream_text, encoding="utf-8")
target_word_count = int(round(dream_emotions_full["dream_text"].dropna().astype(str).str.split().str.len().mean()))
target_word_count = max(target_word_count, 1)


# RAG part -------------------------------------------------------
embedder = SentenceTransformer("BAAI/bge-small-en-v1.5", device = DEVICE) 

with dream_text_path.open('r', encoding='utf-8') as f:          
    full_text = f.read()

# Configure the LangChain text splitter
# 2000 chars is roughly 350 words
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 150,
    chunk_overlap = 30,
    length_function = len,
    separators = ["\n\n", "\n", " ", "", "."]
)

# create the corpus of semantic chunks
corpus = text_splitter.split_text(full_text)

# Embed the chunks
embeddings = embedder.encode(corpus, normalize_embeddings = True)

# FAISS --> Dense Retrieval
dimension = embeddings.shape[1]
# Inner product (requires normalized vectors)
faiss_index = faiss.IndexFlatIP(dimension)
# FAISS requires numpy.float(32) arrays
faiss_vectors = np.array(embeddings).astype("float32")
faiss_index.add(faiss_vectors)

print(f"FAISS Index Total vectors: {faiss_index.ntotal}")

# Search for emotions built from unique macro-emotion labels
query = "anger fear_anxiety sadness_loss joy_amusement love_trust"
emotion_col = None
if "dominant_macro_emotions" in dream_emotions_full.columns:
    emotion_col = "dominant_macro_emotions"
elif "dominant_macro_emotion" in dream_emotions_full.columns:
    emotion_col = "dominant_macro_emotion"

if emotion_col is not None:
    unique_emotions = (
        dream_emotions_full[emotion_col]
        .dropna()
        .astype(str)
        .str.strip()
        .replace("", np.nan)
        .dropna()
        .unique()
    )
    query = " ".join(unique_emotions.tolist()) or query
    
print(f"\nQuery: {query}")

query_emb = np.array(embedder.encode([query], normalize_embeddings = True)).astype("float32")

# Top k results to fetch
k = 3
distances, indices = faiss_index.search(query_emb, k)

for rank, (idx, score) in enumerate(zip(indices[0], distances[0])):
    print(f"Rank {rank + 1} (Dense Score: {score:.4f}) : {corpus[idx]}")


# LLM part ----------------------------------------------------------
# hyperparameters
batch_size = 16  # how many independent sequences will we process in parallel? 
block_size = 512   # what is the max content length for prediction 
max_iters = 5000
eval_interval = 500
learning_rate = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
# Specific for GPT-2 small structural dimensions
n_embd = 512
n_head = 8
n_layer = 6
dropout = 0.1

#------------------------------------------------------
torch.manual_seed(1337)

tokenizer = AutoTokenizer.from_pretrained("gpt2")
# GPT-2 does not have a pad token by default 
tokenizer.pad_token = tokenizer.eos_token

# Update vocab_size
vocab_size = tokenizer.vocab_size

# read it in to inspect it
with open('dream_text.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Train and test splits
data = torch.tensor(tokenizer.encode(text), dtype = torch.long)
n = int(0.9*len(data)) # first 90% of the data will be used as train data
train_data = data[:n]
validation_data = data[n:]

# We have to work with chunks of the dataset 
train_data[:block_size +1]  

# data loading
def get_batch(split):
    """generate a small batch of data of inputs x and targets y"""
    data = train_data if split == 'train' else validation_data
    # handle short datasets by clamping the sampling range
    max_start = max(1, len(data) - block_size)
    ix = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):

    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias = False)
        self.query = nn.Linear(n_embd, head_size, bias = False)
        self.value = nn.Linear(n_embd, head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

        # Store head_size explicitly for scaling
        self.head_size = head_size
    
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * (self.head_size ** -0.5)
        #mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim =-1)
        wei = self.dropout(wei)
        # perform the weigthed aggregation of the values
        v = self.value(x)
        out = wei @ v  # (B, T, T) @ (B, T, C) --> (B, T, C)
        return out

# Multi-head attention:
class MultiHeadAttention(nn.Module):
    """multiple heads of self attention in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim = -1)
        out = self.proj(out)
        return out
    
# To give more time to the logits: 
class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    """ Transformer block: communication followed by computation"""

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        x = x + self.sa(self.ln1(x))       
        x = x + self.ffwd(self.ln2(x))
        return x

# The simplest one is the Bigram Language Model 
class BigramLanguageModel(nn.Module):
    
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(1024, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head = n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)  # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)
    
    def forward(self, idx, targets = None):
        B, T = idx.shape
        
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C) = (Batch, Time, Channel) Tensor
        pos_emb = self.position_embedding_table(torch.arange(T, device = device))  # (T, C)
        x = tok_emb + pos_emb  # x holds not only the tokens identity but the position at which they occur
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets) # -log-likelihood loss

        return logits, loss
    
    def generate(self, idx, max_new_tokens, temperature = 0.4, top_k = 5):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)

            # Apply temperature
            logits = logits / temperature

            # Apply top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            # apply softmax to get probabilities
            probs = F.softmax(logits, dim = -1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples = 1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat( (idx, idx_next), dim = 1 )  # (B, T+1)
        return idx

model = BigramLanguageModel(vocab_size)
m = model.to(device)

# the number of parameters
print(sum(p.numel() for p in m.parameters())/1e6, "M parameters")

# create a Pytorch optimizer
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample of a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none = True)
    loss.backward()
    optimizer.step()


# BRIDGING RAG WITH LLM -------------------------------------------------------------------------
# Gather the retrieved evidence chunks into a single string
retrieved_texts = [corpus[idx] for idx in indices[0]]  #grab only the first row
evidence_string = "\n".join(retrieved_texts)

# Construct the prompt
prompt = f"Context:\n{evidence_string}\n\nQuestion: {query}\n\nAnswer:"

# Encode the prompt (char-level encoder)
encoded_prompt = tokenizer.encode(prompt, return_tensors = 'pt').to(device)

# Generate the answer: the output will be the original context + new generated tokens
generated_indices = m.generate(encoded_prompt, max_new_tokens = 50)[0].tolist()

# Decode the integers back to text and print
new_tokens_only = generated_indices[encoded_prompt.shape[1]:]
final_output = tokenizer.decode(new_tokens_only, skip_special_tokens = True)
print("----LLM OUTPUT: ----")
print(final_output)


# -----------------------------------------------------------------------------------
test_df = pd.read_csv("truncated_dreams_reports.csv", sep = ";")

predictions = []

# Evaluation mode
m.eval()

for index, row in tqdm(test_df.iterrows(), total=len(test_df)):
    query_id = row['Emotion Label']
    query_text = row['Dream Text']
    
    # --- RAG RETRIEVAL ---
    query_emb = np.array(embedder.encode([query_text], normalize_embeddings=True)).astype("float32")
    distances, indices = faiss_index.search(query_emb, k=3)
    
    # Gather the retrieved evidence chunks
    retrieved_texts = [corpus[idx] for idx in indices[0]]
    evidence_string = "\n".join(retrieved_texts)
    
    # --- LLM GENERATION ---
    prompt = f"Similar dreams:\n{evidence_string}\n\nIncomplete dream: {query_text}\n\nContinuation:"
    input_ids = tokenizer.encode(prompt, return_tensors = "pt").to(device)
    input_ids = input_ids[:, -1024:]
    
    # Generate the answer 
    with torch.no_grad(): # Disable gradients for faster inference
        generated_indices = m.generate(input_ids, max_new_tokens=50)[0].tolist()
    
    # Keep only the newly generated tokens
    new_tokens_only = generated_indices[input_ids.shape[1]:]
    final_answer = tokenizer.decode(new_tokens_only, skip_special_tokens= True).strip()
    
    # Save the id and the generated answer
    predictions.append({
        'id': query_id,
        'answer': final_answer
    })

# Create the prediction DataFrame and save it to CSV
results_df = pd.DataFrame(predictions)
results_df.to_csv('predictions.csv', index=False)

print(results_df.head())

# model.gradient_checkpointing_enable() --> to enable gradient checkpointing for faster training
# lr_scheduler_type = "cosine"

