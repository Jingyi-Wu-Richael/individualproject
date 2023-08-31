from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch
import string
import re
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import os
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.set_device(device)


def setup_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


setup_seed(66)

df = pd.read_csv('emotion-emotion_69k.csv')
df['emotion'] = df['emotion'].fillna('')
df_full = df[df['emotion'].str.split().apply(len) == 1]
df_full = df_full[df_full['emotion'].ne('(')]
balanced_df = pd.DataFrame()
unique_emotions = df_full['emotion'].unique()
for emotion in unique_emotions:
    subset = df_full[df_full['emotion'] == emotion]
    samples = subset.sample(n=1200, replace=True)  # 'replace=True' for oversampling if n > len(subset)
    balanced_df = pd.concat([balanced_df, samples])
balanced_df = balanced_df.sample(frac=1).reset_index(drop=True)  # Shuffling the data
df_train, df_test = train_test_split(balanced_df, test_size=0.1, shuffle=True, random_state=0,
                                     stratify=balanced_df['emotion'])
df_train, df_val = train_test_split(df_train, test_size=0.1, shuffle=True, random_state=0, stratify=df_train['emotion'])
str_punc = string.punctuation.replace(',', '').replace("'", "")


def clean(text):
    global str_punc
    text = re.sub(r'[^a-zA-Z ]', '', text)
    text = text.lower()
    return text


X_train = df_train['empathetic_dialogues'].apply(clean).tolist()
y_train = df_train['emotion']
X_test = df_test['empathetic_dialogues'].apply(clean).tolist()
y_test = df_test['emotion']
X_val = df_val['empathetic_dialogues'].apply(clean).tolist()
y_val = df_val['emotion']

le = LabelEncoder()
y_train = torch.LongTensor(le.fit_transform(y_train))
y_test = torch.LongTensor(le.transform(y_test))
y_val = torch.LongTensor(le.transform(y_val))


# Convert labels to one-hot encoding
def one_hot(labels, num_classes):
    # Create a tensor of zeros with size (len(labels), num_classes)
    one_hot_labels = torch.zeros(len(labels), num_classes)
    # Use scatter_ to put a 1 at the index of the correct label
    one_hot_labels = one_hot_labels.scatter_(1, labels.unsqueeze(1), 1.)
    return one_hot_labels


num_classes = len(y_train.unique())
y_train_onehot = one_hot(y_train, num_classes)
y_val_onehot = one_hot(y_val, num_classes)

from transformers import AutoTokenizer, AutoModelForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
base_model = AutoModelForSequenceClassification.from_pretrained("facebook/opt-125m", num_labels=num_classes).to(device)
tokenizer.pad_token = tokenizer.eos_token
base_model.config.pad_token_id = tokenizer.pad_token_id
base_model = base_model.to(device)

# Move y_train_onehot and y_val_onehot to the GPU
y_train_onehot = y_train_onehot.to(device)
y_val_onehot = y_val_onehot.to(device)

class AutoMemoryModule(nn.Module):
    def __init__(self, embedding_layer, max_sentence_length, max_memory_context, embedding_size, padding_token,
                 device='cpu'):
        super().__init__()
        self.max_sentence_length = max_sentence_length
        self.max_memory_context = max_memory_context
        self.embedding_size = embedding_size  # This should be 768 for "microsoft/DialoGPT-small"
        self.device = device
        self.padding_token = padding_token
        self.embedding = embedding_layer
        self.score_net_input_tokens = nn.Linear(self.embedding_size, 1).to(device=device)
        self.score_net_memory_context = nn.Linear(self.embedding_size, 1).to(device=device)

    def forward(self, input_tokens, memory_context):
        if memory_context is None:
            memory_context = torch.zeros(self.max_memory_context, dtype=torch.long).to(device=self.device)
            memory_context.fill_(self.padding_token)
        batch_size, seq_len = input_tokens.shape
        input_tokens = input_tokens.to(device=self.device)
        padded_input_tokens = nn.functional.pad(input_tokens, (0, self.max_sentence_length - seq_len),
                                                value=self.padding_token)
        # score the padded input tokens
        #print("1:", padded_input_tokens.shape)
        input_tokens_embedding = self.embedding(padded_input_tokens).view(-1, self.embedding_size)
        #print("2:", input_tokens_embedding.shape)
        input_tokens_scoring = self.score_net_input_tokens(input_tokens_embedding).squeeze().view(batch_size, seq_len)
        # score the memory context
        #print("3:", input_tokens_scoring.shape)
        #print(memory_context.shape)
        memory_context_embedding = self.embedding(memory_context).view(-1, self.embedding_size)
        #print("4:", memory_context_embedding.shape)
        memory_context_scoring = self.score_net_memory_context(memory_context_embedding).squeeze().view(-1, self.max_memory_context)
        # filter out the padding tokens from the padded input tokens and their scores
        padding_token_idx = torch.nonzero(padded_input_tokens.view(-1) != self.padding_token).squeeze(dim=1)
        filtered_input_tokens = padded_input_tokens.view(-1)[padding_token_idx]
        filtered_input_tokens_scoring = input_tokens_scoring.view(-1)[padding_token_idx]
        ctx_padding_token_idx = torch.nonzero(memory_context.view(-1) != self.padding_token).squeeze(dim=1)
        filtered_memory_context = memory_context.view(-1)[ctx_padding_token_idx]
        filtered_memory_context_scoring = memory_context_scoring.view(-1)[ctx_padding_token_idx]
        combined_tokens = torch.cat((filtered_input_tokens, filtered_memory_context), dim=0)
        scores = torch.cat((filtered_input_tokens_scoring, filtered_memory_context_scoring), dim=0)
        # remove duplicate tokens and their scores
        unique_tokens, indices = torch.unique(combined_tokens, return_inverse=True)
        unique_scores = torch.full_like(unique_tokens, -1e20, dtype=scores.dtype)
        unique_scores = unique_scores.scatter(0, indices, scores)
        # sort the combined tokens and their scores by the scores
        sorted_scores, sorted_indices = torch.sort(unique_scores, descending=True)
        sorted_combined_tokens = unique_tokens[sorted_indices]
        # trim the combined tokens and their scores to the max memory context size
        trimmed_combined_tokens = sorted_combined_tokens[:self.max_memory_context]
        trimmed_scores = sorted_scores[:self.max_memory_context]
        # pad the trimmed tokens and their scores with padding tokens and -1e20 respectively
        trimmed_combined_tokens = nn.functional.pad(trimmed_combined_tokens,
                                                    (0, self.max_memory_context - trimmed_combined_tokens.shape[-1]),
                                                    value=self.padding_token)
        trimmed_scores = nn.functional.pad(trimmed_scores, (0, self.max_memory_context - trimmed_scores.shape[-1]),
                                           value=-1e20)
        #print("trimmed_combined_tokens:",  trimmed_combined_tokens.shape)
        #print("trimmed_scores:",  trimmed_scores.shape)
        return trimmed_combined_tokens, trimmed_scores

class DialoGPTWithMemory(nn.Module):
    def __init__(self, base_model, auto_memory_module):
        super().__init__()
        self.base_model = base_model
        self.auto_memory_module = auto_memory_module

    def forward(self, input_ids, attention_mask, memory_context):
        new_memory_context, _ = self.auto_memory_module(input_ids, memory_context)
        new_memory_context = new_memory_context.reshape(batch_size, -1)
        #print("hello:", new_memory_context.shape, attention_mask.shape)
        outputs = self.base_model(input_ids=new_memory_context, attention_mask=attention_mask)
        return outputs, new_memory_context


# Define your hyperparameters
batch_size = 4
# Create data loaders
X_train_enc = tokenizer(X_train, padding='max_length', truncation=True, max_length=512, return_tensors="pt")
X_val_enc = tokenizer(X_val, padding='max_length', truncation=True, max_length=512, return_tensors="pt")
train_dataset = TensorDataset(X_train_enc['input_ids'], X_train_enc['attention_mask'], y_train_onehot)
val_dataset = TensorDataset(X_val_enc['input_ids'], X_val_enc['attention_mask'], y_val_onehot)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)
token_dim = 100
max_memory_size = batch_size * 512
max_sentence_length = 512
num_classes = len(y_train.unique())
# Decide on number of epochs
n_epochs = 100
vocab_size = tokenizer.vocab_size
# Use the embedding layer from the DialoGPT model
embedding_layer = base_model.get_input_embeddings()
# # Then, pass this embedding layer to the AutoMemoryModule
auto_memory_module = AutoMemoryModule(embedding_layer, max_sentence_length, max_memory_size, embedding_size=768,
                                      padding_token=0, device=device)
# Initialize the DialoGPTWithMemory model
model = DialoGPTWithMemory(base_model, auto_memory_module).to(device)
unfreeze_layers = [f'layers.{i}.' for i in range(0, 8)]
for name, param in model.named_parameters():
    for t in unfreeze_layers:
        if t in name:
            param.requires_grad = False
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.size())
embedding_size = model.base_model.get_input_embeddings().weight.size(0)
print("Vocabulary size:", vocab_size)
# print("Embedding layer size:", embedding_size)
# assert vocab_size == embedding_size, "Vocabulary size does not match the embedding layer size!"
# Define loss function (criterion) and optimizer
# criterion = nn.CrossEntropyLoss()
criterion = nn.BCEWithLogitsLoss()
lr = 0.00005
from optimization import BertAdam
optimizer = BertAdam(model.parameters(), lr=lr)
# Check if the tokenizer has a padding token
# print("Tokenizer padding token:", tokenizer.pad_token)
# If it's None, you can set it
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
# Now, make sure the model's configuration has the same padding token ID
base_model.config.pad_token_id = tokenizer.pad_token_id
# print("Model padding token ID:", base_model.config.pad_token_id)
# Training loop
for epoch in range(n_epochs):
    # Set model to training mode
    model.train()
    total_loss = 0
    total_batches = 0
    # Initialize memory context for each new epoch
    memory_context = torch.zeros((batch_size, max_memory_size), dtype=torch.long, device=device)
    # Iterate over batches in the training data loader
    for input_ids, attention_mask, labels in train_loader:
        optimizer.zero_grad()
        new_memory_context, _ = auto_memory_module(input_ids, memory_context)
        outputs, _ = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device),
                           memory_context=new_memory_context)
        loss = criterion(outputs.logits, labels.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_batches += 1
        memory_context = new_memory_context
        if total_batches % 100 == 0:
            print(epoch, total_batches, loss.item())
    print(f"Epoch {epoch + 1}/{n_epochs}, Training Loss: {total_loss / total_batches}")
    model.eval()
    total_val_loss = 0
    total_val_batches = 0
    labels_val_indices = []
    preds = []
    memory_context_val = torch.zeros((batch_size, max_memory_size), dtype=torch.long, device=device)
    with torch.no_grad():
        for input_ids_val, attention_mask_val, labels_val in val_loader:
            new_memory_context_val, _ = auto_memory_module(input_ids_val, memory_context_val)
            outputs_val, _ = model(input_ids=input_ids_val.to(device),
                                   attention_mask=attention_mask_val.to(device),
                                   memory_context=new_memory_context_val)
            loss_val = criterion(outputs_val.logits, labels_val.to(device))
            total_val_loss += loss_val.item()
            total_val_batches += 1
            memory_context_val = new_memory_context_val
            preds.extend(torch.argmax(outputs_val.logits, dim=1).cpu().numpy())
            labels_val_indices.extend(torch.argmax(labels_val, dim=1).cpu().numpy())
    precision, recall, f1, _ = precision_recall_fscore_support(labels_val_indices, preds,
                                                               average='macro', zero_division=1)
    result = [epoch + 1, total_loss / total_batches, total_val_loss / total_val_batches, precision, recall, f1]
    print("Eval:", result)
    result = [str(i) for i in result]
    with open(f"{n_epochs}_{lr}_opt_memory_8.txt", mode="a", encoding="utf8") as f:
        f.write("\t".join(result) + "\n")