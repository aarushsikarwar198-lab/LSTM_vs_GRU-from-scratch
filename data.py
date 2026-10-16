import torch 
import datasets
ds = datasets.load_dataset("stanfordnlp/imdb")
import re
import collections
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
# print(ds['train'][0])

ds1 = ds["train"].train_test_split(test_size=0.1 , seed = 42)
val = ds1["test"]
train = ds1["train"]
test = ds['test']


def tokenize(text):
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)           
    return re.findall(r"[a-z0-9']+|[!?]", text) 

train = train.map(lambda x: {"tokens": tokenize(x["text"])})
val   = val.map(lambda x: {"tokens": tokenize(x["text"])})
test  = test.map(lambda x: {"tokens": tokenize(x["text"])})

token_counts = collections.Counter([token for example in train['tokens'] for token in example])

# print(token_counts.most_common(10))

lengths = [len(example) for example in train['tokens']]
x = np.mean(lengths)
p90 = np.percentile(lengths, 90)
p95 = np.percentile(lengths, 95)
p100 = np.percentile(lengths, 100)

# print(f"Mean: {x}, 90th percentile: {p90}, 95th percentile: {p95}, 100th percentile: {p100}")
max_len = 256
ltc = len(token_counts)

count = 0
for token in token_counts.items():
    if token[1] < 5:
        count += 1

vocab = dict({"<pad>": 0, "<unk>": 1})
for i, (token, count) in enumerate(token_counts.most_common(20000)):
    if count < 5:
        break
    vocab[token] = len(vocab)


def encode(tokens, vocab, max_len):
    unk, pad = vocab["<unk>"], vocab["<pad>"]
    ids = [vocab.get(t, unk) for t in tokens]  
    ids = ids[-max_len:]                        
    return [pad] * (max_len - len(ids)) + ids  

train_ids = torch.tensor([encode(t, vocab, max_len) for t in train["tokens"]] , dtype = torch.float32)
val_ids   = torch.tensor([encode(t, vocab, max_len) for t in val["tokens"]] , dtype = torch.float32)
test_ids  = torch.tensor([encode(t, vocab, max_len) for t in test["tokens"]] , dtype = torch.float32)

val_labels = torch.tensor(val['label'], dtype = torch.float32)
train_labels = torch.tensor(train['label'], dtype = torch.float32)
test_labels = torch.tensor(test['label'], dtype = torch.float32)

train_ds = TensorDataset(train_ids, train_labels)
val_ds = TensorDataset(val_ids, val_labels)
test_ds = TensorDataset(test_ids, test_labels)

val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)
train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)
def make_train_loader(seed = 42 , batch_size = 512):
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True ,generator = torch.Generator().manual_seed(seed))
    return train_loader

train_loader = make_train_loader(seed = 42 , batch_size =512)


data = {
    "train_loader": train_loader,
    "val_loader": val_loader,
    "test_loader": test_loader,
    "vocab_size": len(vocab),
    "pad_idx": vocab["<pad>"],
    "max_len" : max_len
}


torch.save(data , "imdb_preprocessed.pt")
