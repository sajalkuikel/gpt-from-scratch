with open ('input.txt', 'r', encoding='utf-8' ) as f:
    text = f.read()
print('dataset character length:', len(text) )
print(text[:500])
# Getting only unique characters in the dataset with set()
chars = sorted(list(set(text)))
vocab_size  = len(chars)
print(''.join(chars))

print(vocab_size)
# creating a map of string to integers
stoi =  {ch:i for i, ch in enumerate(chars)}
stoi

# creating a map of integer to string
itos =  {i:ch for i, ch in enumerate(chars)}
itos

# Encoder: output the mapped integer given the stirng
encode =  lambda s: [stoi[c] for c in s ]
encode('Hello') 
# Decoder : output the mapped string, given the list of integer
decode = lambda l: ''.join( [itos[i] for i in l ] )
decode( [20, 43, 50, 50, 53]   )
# Sentence piece

# tiktoken - by openAI  
# -  BPE tokenizer , sub word tokenizer
# Used by gpts 
import torch
data =  torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:1000])
# Splitting into train, and validation sets


n = int(0.9 * len(data)) # n = 90 %

train_data  = data[:n] # First 90% 
val_data  = data[n:] # Remaining 10 %
# defining the context length (chunk size)
block_size = 8
train_data[:block_size + 1 ]
decode(train_data[:block_size + 1 ].tolist())
x = train_data[:block_size]
y =  train_data[1: block_size + 1]

decode ( x.tolist() ) , decode (y.tolist()) 
for t in range(block_size):
    context =  x[: t + 1]
    target = y[t]
    print(f"when input is {context} the target is {target}")
# introducing a batch dimension 

torch.manual_seed(1337)
batch_size =  4
block_size = 8

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size, ) )
    x = torch.stack([data[ i : i + block_size] for i in ix])
    y = torch.stack([data[i+1: i + block_size + 1] for i in ix])
    return x,y

xb, yb = get_batch('train')

print('inputs:')
print(xb.shape)
print(xb)

##############

print('targets:')
print(yb.shape)
print(yb)

print('-----')

for b in range(batch_size):
    for t in range(block_size):
        context = xb[b, :t+1]
        target  = yb[b,t]
        print(f"when input is {context.tolist()} the target is: {target}")

print(xb)
# Implementing Bigram language model

import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)
n_embd = 32
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        
        tok_emb = self.token_embedding_table(idx) # arranged in  (batch, time, channel) tensor form
        logits = self.lm_head(tok_emb)
        
        # loss = F.cross_entropy(logits, targets) # does not work because the function expects Channel to be the second argument
        #https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss 
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view( B * T)
            loss = F.cross_entropy(logits, targets)
        return logits , loss
        # logit:scores for the next character in the sequence
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
    
m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb)
print(logits.shape)
print(loss)

# batch=1 , time=1
# filled with zeros, zero is an embedding for  new line 
idx = torch.zeros((1,1), dtype=torch.long)


print(decode(m.generate(idx, max_new_tokens = 100)[0].tolist() ))
# training
# creating pytorch optimizer
optimizer  = torch.optim.AdamW(m.parameters(), lr= 1e-3)
#creating a training loop
batch_size = 32
for steps in range(1000):
    xb, yb = get_batch('train')

    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
print(loss.item())
print(decode(m.generate(idx, max_new_tokens = 500)[0].tolist() ))
### Self Attention
#example:

torch.manual_seed(1337)

B, T, C = 4,8,2
x = torch.randn(B,T,C)
x.shape 
xbow =  torch.zeros((B, T, C))
for b in range(B):
    for t in range(T):
        xprev = x[b, :t+1]
        xbow[b,t] = torch.mean(xprev, 0)
# torch.tril(torch.ones(T,T))
# version 2: using matrix multiply for a weighted aggregation
wei = torch.tril(torch.ones(T,T))
wei = wei / wei.sum(1, keepdim=True) # averaging every row 
xbow2 = wei @ x 
torch.allclose(xbow, xbow2, rtol= 1e-04) #  adjusting the relative tolerance for less strict comparison. the default value is 1e-05 in PyTorch 2.2
xbow[0] , xbow2[0] 
# v3 : using softmax
tril = torch.tril(torch.ones(T,T))
wei = torch.zeros((T,T))
wei = wei.masked_fill(tril ==0, float('-inf'))
wei
# expontiate every single element and divide by the sum
# that means get zero inplace of -inf
# by setting them to -inf, we are enforcing it to not add that token
wei  = F.softmax(wei, dim=1)
wei