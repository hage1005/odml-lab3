
# ### SST-2 trial 2

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import time
import torch
from torch.quantization.quantize_fx import prepare_fx, convert_fx, prepare_qat_fx
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
class SST2Dataset(Dataset):
    def __init__(self, file_path, vocab=None, max_vocab_size=10000, test_mode=False):
        self.data = pd.read_csv(file_path, sep='\t', header
        ='infer')
        self.sentences = self.data['sentence'].tolist()
        
        self.labels = self.data['label'].tolist()

        # Build vocabulary if none is provided
        if vocab is None:
            self.vocab = self.build_vocab(self.sentences, max_vocab_size)
        else:
            self.vocab = vocab

    def build_vocab(self, sentences, max_vocab_size):
        counter = Counter()
        for sentence in sentences:
            counter.update(sentence.split())
        most_common = counter.most_common(max_vocab_size - 1)  # Reserve one for <UNK>
        vocab = {word: idx + 1 for idx, (word, _) in enumerate(most_common)}
        vocab['<UNK>'] = 0  # <UNK> token for unknown words
        return vocab

    def encode_sentence(self, sentence):
        return [self.vocab.get(word, self.vocab['<UNK>']) for word in sentence.split()]

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.encode_sentence(self.sentences[idx])
        sentence = torch.tensor(sentence, dtype=torch.long)  # Input should be LongTensor
        if self.labels is not None:
            label = int(self.labels[idx])
            return sentence, torch.tensor(label)
        else:
            return sentence


def train_model(model, train_loader, criterion, optimizer, num_epochs=2, calibrate=False):
    model = model.to(device)
    model.train()
    for epoch in range(num_epochs):
        epoch_start = time.time()
        running_loss = 0.0
        for sentences, labels in train_loader:
            sentences, labels = sentences.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(sentences.float())
            if calibrate:
                continue
            loss = criterion(outputs.squeeze(), labels.float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        epoch_end = time.time()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Time: {epoch_end - epoch_start:.2f}s')

import os

def print_size_of_model(model, label=""):
    torch.save(model.state_dict(), "temp.p")
    size=os.path.getsize("temp.p")
    print("model: ",label,' \t','Size (MB):', size/1e6)
    os.remove('temp.p')
    return size

from torch.nn.utils.rnn import pad_sequence
import torch

def collate_fn_fixed_length(batch, max_length=50):
    sentences, labels = zip(*batch)
    
    # Convert each sentence to a tensor and pad/truncate to max_length
    sentences = [torch.tensor(sentence) for sentence in sentences]
    
    # Pad sequences to the max_length
    sentences_padded = pad_sequence(sentences, batch_first=True, padding_value=0)
    
    # Crop sequences to the max_length (necessary if any sequences are longer than max_length)
    sentences_padded = sentences_padded[:, :max_length]
    
    # Ensure that all sequences are padded to max_length
    if sentences_padded.size(1) < max_length:
        padding = torch.zeros((sentences_padded.size(0), max_length - sentences_padded.size(1)), dtype=sentences_padded.dtype)
        sentences_padded = torch.cat([sentences_padded, padding], dim=1)
    
    labels = torch.tensor(labels, dtype=torch.float32)
    
    return sentences_padded, labels


class FeedForwardNN(nn.Module):
    def __init__(self, vocab_size, hidden_size=256):
        super(FeedForwardNN, self).__init__()
        self.fc1 = nn.Linear(vocab_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)  # Binary classification
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def evaluate_model(model, data_loader):
    model = model.to(device)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for sentences, labels in data_loader:
            sentences, labels = sentences.to(device), labels.to(device)
            outputs = model(sentences.float())
            predicted = torch.round(torch.sigmoid(outputs.squeeze()))
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')

def measure_latency(model, test_loader):
    model = model.to(device)
    model.eval()
    start_time = time.time()
    
    with torch.no_grad():
        for i, (sentences, _) in enumerate(test_loader):
            sentences = sentences.to(device)
            if i == 0:  # First inference is often slower due to caching, so skip it
                continue
            model(sentences.float())
    
    end_time = time.time()
    avg_inference_time = (end_time - start_time) / len(test_loader)
    print(f'Average Inference Latency: {avg_inference_time:.6f} seconds')


def calculate_flops(model, input_dim):
    # Assume feed-forward layers perform matmul ops
    flops = 0
    flops += input_dim * model.fc1.out_features  # Input layer to hidden
    flops += model.fc1.out_features * model.fc2.out_features  # Hidden to output
    print(f'Total FLOPs for one forward pass: {flops}')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(model, calibrate=False):
    # Define hyperparameters
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 5

    # Load dataset and dataloaders
    train_dataset = SST2Dataset('data/SST-2/train.tsv', max_vocab_size=10000)
    test_dataset = SST2Dataset('data/SST-2/dev.tsv', vocab=train_dataset.vocab)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_fixed_length)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_fixed_length)

    # Initialize model, loss function, and optimizer
    vocab_size = len(train_dataset.vocab)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train_model(model, train_loader, criterion, optimizer, num_epochs, calibrate)

    # Evaluate the model
    evaluate_model(model, test_loader)

    # Measure efficiency
    measure_latency(model, test_loader)
    # calculate_flops(model, vocab_size)
    print(f'Total Parameters: {count_parameters(model)}')

train_dataset = SST2Dataset('data/SST-2/train.tsv', max_vocab_size=10000)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn_fixed_length)

hidden_size = 256
model = FeedForwardNN(50, hidden_size)
# train(model)
print_size_of_model(model, "float32")
dynamic_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
# train(model)
print_size_of_model(dynamic_model, "int8 dynamic_model")
static_model = prepare_fx(model, {"": torch.quantization.default_qconfig}, example_inputs=next(iter(train_loader))[0])
train(static_model, calibrate=True)
model = convert_fx(static_model)
print_size_of_model(static_model, "int8 (FX)")
# train(model)
# model = torch.quantization.quantize_fx.prepare_qat_fx




