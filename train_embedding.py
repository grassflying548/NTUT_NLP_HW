import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
import numpy as np

# ---------------------------
# 1. Prepare a small corpus
# ---------------------------
sentences = [
    "the quick brown fox jumps over the lazy dog",
    "i love natural language processing",
    "the fox is quick and the dog is lazy"
]

# ---------------------------
# 2. Build the vocabulary
# ---------------------------
def build_vocab(sentences):
    words = []
    for sentence in sentences:
        words.extend(sentence.lower().split())
    word_counts = Counter(words)
    # Create a mapping from word to index
    vocab = {word: idx for idx, (word, count) in enumerate(word_counts.items())}
    return vocab

vocab = build_vocab(sentences)
vocab_size = len(vocab)
print("Vocabulary:", vocab)

# ---------------------------
# 3. Generate training data for Skip-Gram
# ---------------------------
def generate_training_data(sentences, vocab, window_size=2):
    training_data = []
    for sentence in sentences:
        tokens = sentence.lower().split()
        token_ids = [vocab[word] for word in tokens]
        for i, center in enumerate(token_ids):
            # Define the context window boundaries
            for j in range(max(0, i - window_size), min(len(token_ids), i + window_size + 1)):
                if i != j:
                    context = token_ids[j]
                    training_data.append((center, context))
    return training_data

training_data = generate_training_data(sentences, vocab, window_size=2)
print("Number of training pairs:", len(training_data))
# Print a few sample pairs (center, context)
print("Sample training pairs:", training_data[:5])

# ---------------------------
# 4. Define the Skip-Gram Model
# ---------------------------
class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGramModel, self).__init__()
        # Embedding layer maps each word to an embedding vector
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        # A linear layer to predict the probability distribution over the vocabulary
        self.linear = nn.Linear(embedding_dim, vocab_size)
    
    def forward(self, center_word):
        # center_word: tensor of shape (batch_size,)
        emb = self.embeddings(center_word)  # shape: (batch_size, embedding_dim)
        out = self.linear(emb)               # shape: (batch_size, vocab_size)
        return out

# Hyperparameters
embedding_dim = 10   # Dimension of the embedding vectors
learning_rate = 0.01
num_epochs = 100

# Instantiate the model, loss function, and optimizer
model = SkipGramModel(vocab_size, embedding_dim)
criterion = nn.CrossEntropyLoss()  # Combines log-softmax and negative log likelihood loss
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ---------------------------
# 5. Training Loop
# ---------------------------
print("\nStarting training...")
for epoch in range(num_epochs):
    total_loss = 0
    # Shuffle training data for each epoch
    np.random.shuffle(training_data)
    
    # Process each training pair (in a production setting, batching is recommended)
    for center, context in training_data:
        # Convert indices to tensors (batch size 1)
        center_tensor = torch.tensor([center], dtype=torch.long)
        context_tensor = torch.tensor([context], dtype=torch.long)
        
        # Forward pass
        output = model(center_tensor)  # output shape: (1, vocab_size)
        loss = criterion(output, context_tensor)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}")

print("Training complete.")

# ---------------------------
# 6. Inspect the learned embeddings
# ---------------------------
embeddings = model.embeddings.weight.data.numpy()

print("\nLearned Embeddings:")
for word, idx in vocab.items():
    print(f"{word}: {embeddings[idx]}")
