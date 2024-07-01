import json
import numpy as np
import os, random
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm, wandb

api_key = "4556651ce2176ed46d63906314d5c7703d3ff741"
wandb.login(key=api_key)
wandb.init(project="ontologyReasoning")

MAX_LENGTH = 512
PAD_TOKEN = 0

def pad_and_truncate(sequence, max_length):
    if len(sequence) > max_length:
        return sequence[:max_length]
    else:
        # Ensure sequence is a list of integers
        sequence = list(sequence)
        padding = [PAD_TOKEN] * (max_length - len(sequence))
        return sequence + padding
    
def collate_fn(batch):
    premises = [torch.tensor(item['premise'], dtype=torch.long) for item in batch]
    proofs = [torch.tensor(item['proof'], dtype=torch.long) for item in batch]

    # for idx, (premise, proof) in enumerate(zip(premises, proofs)):
    #     print(f"Premise length [{idx}]: {len(premise)}, Proof length [{idx}]: {len(proof)}")
    
    premises_padded = torch.stack(premises)
    proofs_padded = torch.stack(proofs)
    premises_masks = (premises_padded != PAD_TOKEN).long()
    proofs_masks = (proofs_padded != PAD_TOKEN).long()
    return premises_padded, proofs_padded, premises_masks, proofs_masks

def json_to_string(input_files, proof_files, ontology_path):
    premises, proofs = [], []
    for input_f, proof_f in zip(input_files, proof_files):
        # Attempt to process the input file
        input_file_path = os.path.join(ontology_path, input_f)
        try:
            with open(input_file_path, 'r') as f:
                premise_string = json.dumps(json.load(f)).encode('utf-8')
                encoded_premise = np.frombuffer(premise_string, dtype=np.uint8)
            premises.append(encoded_premise)
        except json.JSONDecodeError as e:
            print(f"Error loading JSON from {input_file_path}: {str(e)}")
            # Optionally, raise the error to stop the process or handle it appropriately
            # raise

        # Attempt to process the proof file
        proof_file_path = os.path.join(ontology_path, proof_f)
        try:
            with open(proof_file_path, 'r') as f:
                proof_string = json.dumps(json.load(f)).encode('utf-8')
                encoded_proof = np.frombuffer(proof_string, dtype=np.uint8)
            proofs.append(encoded_proof)
        except json.JSONDecodeError as e:
            print(f"Error loading JSON from {proof_file_path}: {str(e)}")
            # Optionally, raise the error to stop the process or handle it appropriately
            # raise

    return premises, proofs


def load_ontology(path, train_ratio=0.7, val_ratio=0.1):
    premises, proofs = [], []
    
    print(f"Path: {path}")
    for dir in os.listdir(path):
        ontology_path = os.path.join(path, dir)
        input_files = sorted([f for f in os.listdir(ontology_path) if f.startswith("Input")])
        proof_files = sorted([f for f in os.listdir(ontology_path) if f.startswith('Proof')])
        cur_dir_premises, cur_dir_proofs = json_to_string(input_files, proof_files, ontology_path)
        
        premises.extend(cur_dir_premises)
        proofs.extend(cur_dir_proofs)
        
        print(f"Current dir: {dir}: premises: {len(cur_dir_premises)}, proofs: {len(cur_dir_proofs)}")
        assert len(premises) == len(proofs)

    # Pad and truncate sequences to ensure they fit within MAX_LENGTH
    padded_premises = []
    padded_proofs = []
    
    for i, (premise, proof) in enumerate(zip(premises, proofs)):
        padded_premises.append(pad_and_truncate(premise, MAX_LENGTH))
        padded_proofs.append(pad_and_truncate(proof, MAX_LENGTH))

    data_length = len(premises)
    train_len = int(data_length * train_ratio)
    val_len = int(train_len * val_ratio)

    indices = np.arange(data_length)
    np.random.shuffle(indices)

    train_indices = indices[:train_len]
    val_indices = indices[train_len:train_len + val_len]
    test_indices = indices[train_len + val_len:]

    train_data = [{"premise": padded_premises[i], "proof": padded_proofs[i]} for i in train_indices]
    val_data = [{"premise": padded_premises[i], "proof": padded_proofs[i]} for i in val_indices]
    test_data = [{"premise": padded_premises[i], "proof": padded_proofs[i]} for i in test_indices]

    return train_data, val_data, test_data

train_data, val_data, test_data = load_ontology("data_direct_true_100_ontologies")

train_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False, collate_fn=collate_fn)

def lower_triangular_mask(seq_length):
    """
    Create a lower triangular mask
    """

    lt_mask = np.triu(np.ones((1, seq_length, seq_length)), k=1)
    lt_mask = torch.from_numpy(lt_mask) == 0

    return lt_mask

def create_mask(seq_length, pad_mask):
    """
    Creates the transformer decode mask
    """

    # create pad mask
    pad_mask = pad_mask.unsqueeze(-2)
    # print("Padding Mask;: ", pad_mask.shape, pad_mask[0, 0])

    # create lower triangular mask
    lt_mask = lower_triangular_mask(seq_length)
    # print("Casual Mask;: ", lt_mask.shape, lt_mask[0])

    # return the bitwise AND of the two masks
    return pad_mask & lt_mask

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

class InputEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_seq_len):

        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.embedding = nn.Embedding(vocab_size, embed_dim).to(device)

    @staticmethod
    def positional_encoding(max_seq_len, embed_dim, n):

        # Generate an empty matrix
        position = torch.arange(max_seq_len, device=device).unsqueeze(1)
        division_term = torch.exp(torch.arange(0, embed_dim, 2, device=device) * (-np.log(n) / embed_dim))
        pos_enc = torch.zeros(max_seq_len, embed_dim, device=device)

        pos_enc[:, 0::2] = torch.sin(position * division_term)
        pos_enc[:, 1::2] = torch.cos(position * division_term)

        return pos_enc

    def forward(self, x):
        # (batch, seq_length) ---> (batch, seq_length, embed_size)
        _, seq_len = x.size()
        token_embed = self.embedding(x)
        # print("Token embedshape: ", token_embed.shape)
        
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum sequence length {self.max_seq_len}")

        pos_encodings = InputEmbedding.positional_encoding(self.max_seq_len, self.embed_dim, 10000).to(device)
        self.register_buffer('pos_enc', pos_encodings)
        
        # print("Positional encoding shape: ", pos_encodings.shape)
        return token_embed + pos_encodings[:seq_len]

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, heads = 4):
        super().__init__()

        assert embed_dim % heads == 0

        self.embed_dim = embed_dim
        self.heads = heads
        self.d_k = embed_dim // heads # Dimenstions of vector seen by each head

        self.w_q = nn.Linear(embed_dim, embed_dim, bias = False)
        self.w_k = nn.Linear(embed_dim, embed_dim, bias = False)
        self.w_v = nn.Linear(embed_dim, embed_dim, bias = False)

        self.w_o = nn.Linear(embed_dim, embed_dim, bias = False)
        self.dropout = nn.Dropout(0.1)

        self.attention_scores = None

    def attention(self, query, key, value, mask=None):
        d_k = query.shape[-1]

        #(batch, head, seq_len, d_k) ---> (batch, head, seq_length, seq_length)
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(d_k)
        # print(f"Attention scores shape: {attention_scores.shape}")
        
        if mask is not None:
        #   mask = mask.unsqueeze(1).unsqueeze(2)
            # print("Mask shape: ", mask.shape)
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))

        attention_scores = F.softmax(attention_scores, dim=-1)
        self.attention_scores = attention_scores

        #(batch, head, seq_length, seq_length) --> (batch, head, seq_len, d_k)
        return torch.matmul(attention_scores, value)

    def forward(self, x, mask=None):
        # print(f"Multihead input shape: {x.shape}")
        batch_size, sequence_length, _ = x.size()

        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)
        # print(f"Shapes Q: {Q.shape}, K: {K.shape}, V: {V.shape}")

        # Split each tensor into heads, where each head has size d_k

        # (batch, seq_length, embed_dim) --> (batch, seq_len, head, embed_dim) --> (batch, head, seq_len, embed_dim)
        Q = Q.view(batch_size, sequence_length, self.heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, sequence_length, self.heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, sequence_length, self.heads, self.d_k).transpose(1, 2)

        mask = create_mask(sequence_length, mask).unsqueeze(1)
        # print("Combined mask example: ", mask.shape, mask[0])
        
        attention_output = self.attention(Q, K, V, mask=mask)
            
        attention_output = self.dropout(attention_output)
        
        # (batch, head, seq_len, d_k) --> (batch, seq_len, head, d_k) --> (batch, seq_len, head * d_k)
        combined_heads = attention_output.transpose(1, 2).contiguous().view(batch_size, sequence_length, self.heads * self.d_k)
        return self.w_o(combined_heads)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, heads, d_ff):
        super().__init__()
        self.attention = MultiHeadSelfAttention(embed_dim, heads).to(device)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, d_ff),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_ff, embed_dim)
        )

    def forward(self, x, mask=None):
        attention = self.attention(x, mask=mask)
        x = self.norm1(attention + x)

        fforward = self.feed_forward(x)
        return self.norm2(fforward + x)

class TransformerModel(nn.Module):
    def __init__(self, embed_dim, heads, d_ff, seq_len, N, num_tokens):
        super().__init__()

        self.num_tokens = num_tokens

        self.embedding_layer = InputEmbedding(num_tokens, embed_dim, seq_len)
        self.transformer_layers = nn.ModuleList(
            [TransformerBlock(embed_dim, heads, d_ff) for _ in range(N)]
        )

        # self.transformer_block = TransformerBlock(embed_dim, heads, d_ff)
        self.dropout = nn.Dropout(0.1)

        self.to_probs = nn.Linear(embed_dim, num_tokens)

    def forward(self, x, padding_mask=None):
        x = self.embedding_layer(x)
        batch_size, seq_length, embed_dim = x.size()

        x = self.dropout(x)
        
        for layer in self.transformer_layers:
            x = layer(x, mask=padding_mask)

        # print(f"Transformer output shape: {x.shape}")
        x = self.to_probs(x.view(batch_size*seq_length, embed_dim)).view(batch_size, seq_length, self.num_tokens)

        x = F.log_softmax(x, dim=2)
        
        # print(f"Output shape: {x.shape}")
        return x

config = {
    'pad_index': 0,
    'learning_rate': 0.0001,
    'batch_size': 32,
    'num_epochs': 1000,
    'lr_warmup': 10000,
    'seed': 42,
    'embed_dim': 256,
    'heads': 4,
    'd_ff': 256,
    'seq_len': 512,
    'N': 8,
    'num_tokens': 256,
    'context': 512,
    'char_to_gen': 512,
    'log_interval': 100,
    'save_interval': 1000,
    'compression_interval': 1000,
    'log_dir': 'logs/',
    'model_dir': 'saved_models/',
    'data_path': 'data_direct_true_100_ontologies',
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = TransformerModel(embed_dim=config['embed_dim'], heads=config['heads'], d_ff=config['d_ff'],
                         seq_len=config['seq_len'], N=config['N'], num_tokens=config['num_tokens']).to(device)

optimizer = optim.Adam(model.parameters(), config['learning_rate'])
criterion = nn.NLLLoss(ignore_index=config['pad_index'])

def decode_sequence(sequence):
    """Decode a sequence of token indices to a UTF-8 string."""
    return ''.join(chr(token) for token in sequence.tolist())

def print_example_output(model, premises_padded, padding_mask):
    """Print the premise and the generated proof for a random example."""
    model.eval()
    with torch.no_grad():
        random_idx = random.randint(0, premises_padded.size(0) - 1)
        random_premise = premises_padded[random_idx].unsqueeze(0)  # Add batch dimension
        random_premise_mask = padding_mask[random_idx].unsqueeze(0)  # Add batch dimension

        # Generate predictions for the random premise
        predictions = model(random_premise, padding_mask=random_premise_mask)
        predictions = predictions.argmax(dim=-1).squeeze(0)  # Get the index of the highest probability and remove batch dim

        # Decode the premise and predictions to UTF-8
        premise_text = decode_sequence(random_premise.squeeze().cpu())
        predicted_text = decode_sequence(predictions.cpu())

        print(f"SEED (Premise): {premise_text}")
        print(f"Generated Proof: {predicted_text}")
    model.train()
    
num_epochs = 10

for epoch in tqdm.trange(num_epochs):
    model.train()
    running_loss = 0.0

    for i, batch in enumerate(train_loader):
        premises_padded, proofs_padded, premises_masks, proofs_masks = batch
        premises_padded = premises_padded.to(device)
        proofs_padded = proofs_padded.to(device)
        premises_masks = premises_masks.to(device)
        proofs_masks = proofs_masks.to(device)

        if premises_padded.size(1) > config['seq_len']:
            premises_padded = premises_padded[:, :config['seq_len']]
            premises_masks = premises_masks[:, :config['seq_len']]
            
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        # print("Proofs padded shape: ", proofs_masks.shape)
        outputs = model(premises_padded, padding_mask=premises_masks)  # (batch_size, seq_length, num_tokens)

        
        # Reshape for NLLLoss
        outputs = outputs.view(-1, outputs.size(-1))  # (batch_size * seq_length, num_tokens)
        # print("Outputs shape after view: ", outputs.shape)
        proofs_padded = proofs_padded.view(-1)  # (batch_size * seq_length)
        # print("Proofs padded shape: ", proofs_padded.shape)
        
        loss = criterion(outputs, proofs_padded)
        
        # Backward pass and optimization
        loss.backward()
        total_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        wandb.log({"Loss/train": loss.item(),
            "Gradient norm": total_norm})
        
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
        if i % 10 == 3:    # Print every 100 batches
            print(f"[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 100:.3f}")
            running_loss = 0.0
            
        # Print example output every 10 iterations
        if i % 10 == 3:
            print_example_output(model, premises_padded, premises_masks)

wandb.finish()
print("Finished Training")