from collections import Counter
from torch.utils.data import Dataset, DataLoader
import numpy as np
class SST2BoWDataset(Dataset):
    def __init__(self, file_path, vocab=None, vocab_size=5000):
        self.data = []
        self.labels = []
        self.vocab = vocab
        self.vocab_size = vocab_size

        with open(file_path, 'r') as f:
            next(f)
            for line in f:
                tokens, label = line.strip().split('\t')
                tokens = tokens.split()
                label = int(label)
                self.data.append(tokens)
                self.labels.append(label)

        if self.vocab is None:
            self.vocab = self.build_vocab(self.data)
        
        self.bow_data = [self.text_to_bow(tokens) for tokens in self.data]

    def build_vocab(self, data):
        all_words = [word for sentence in data for word in sentence]
        word_counts = Counter(all_words).most_common(self.vocab_size)
        vocab = {word: idx for idx, (word, _) in enumerate(word_counts)}
        return vocab

    def text_to_bow(self, tokens):
        bow_vector = np.zeros(len(self.vocab), dtype=np.float32)
        token_counts = Counter(tokens)
        for token, count in token_counts.items():
            if token in self.vocab:
                bow_vector[self.vocab[token]] = count
        return bow_vector

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        bow_vector = self.bow_data[idx]
        label = self.labels[idx]
        return torch.tensor(bow_vector, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

train_dataset = SST2BoWDataset('data/SST-2/train.tsv', vocab_size=5000)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

dev_dataset = SST2BoWDataset('data/SST-2/dev.tsv', vocab=train_dataset.vocab, vocab_size=5000)
dev_loader = DataLoader(dev_dataset, batch_size=64, shuffle=False)
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# Set up TensorRT logger and load engine
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def load_engine(engine_path):
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

engine = load_engine("model_int8.trt")
context = engine.create_execution_context()

# Allocate device memory for inputs and outputs
input_shape = (1, len(dataset.vocab))  # Shape based on BoW vector length
output_shape = (1, 2)  # Assuming binary classification
d_input = cuda.mem_alloc(np.prod(input_shape) * np.float32().itemsize)
d_output = cuda.mem_alloc(np.prod(output_shape) * np.float32().itemsize)
bindings = [int(d_input), int(d_output)]
stream = cuda.Stream()

def infer(bow_vector):
    # Ensure the vector is in the correct shape and data type
    input_data = bow_vector.cpu().numpy().astype(np.float32).reshape(1, -1)  # Adjust shape if needed
    output_data = np.empty((1, 2), dtype=np.float32)  # Adjust shape if needed

    # Copy input to device, run inference, and copy output back to host
    cuda.memcpy_htod_async(d_input, input_data, stream)
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(output_data, d_output, stream)
    stream.synchronize()
    
    return output_data

correct = 0
total = 0

for bow_vectors, labels in dev_loader:
    # Run inference for each sample in the batch
    for bow_vector, label in zip(bow_vectors, labels):
        output = infer(bow_vector)
        predicted_label = np.argmax(output)  # Get the predicted class
        if predicted_label == label.item():
            correct += 1
        total += 1

accuracy = correct / total
print(f"Accuracy: {accuracy * 100:.2f}%")