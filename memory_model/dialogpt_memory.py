# data preparation: Empathetic Dialogues (Facebook AI) 25k
import pandas as pd
from sklearn.model_selection import train_test_split
df= pd.read_csv('emotion-emotion_69k.csv')
# df= df.loc[(df.labels!=1) & (df.labels!=3)]
print(df.columns)
# print(df['emotion'].value_counts())
df['emotion'] = df['emotion'].fillna('')
df_full = df[df['emotion'].str.split().apply(len) == 1]
df_full = df_full[df_full['emotion'].ne('(')]
print(df_full['emotion'].value_counts())

balanced_df = pd.DataFrame()
unique_emotions = df_full['emotion'].unique()
for emotion in unique_emotions:
    subset = df_full[df_full['emotion'] == emotion]
    samples = subset.sample(n=1200, replace=True)  # 'replace=True' for oversampling if n > len(subset)
    balanced_df = pd.concat([balanced_df, samples])
balanced_df = balanced_df.sample(frac=1).reset_index(drop=True)  # Shuffling the data

# split the dataset into training and test sets
df_train, df_test = train_test_split(balanced_df, test_size=0.2, shuffle=True, random_state=0, stratify=balanced_df['emotion'])
# Further split the training data into training and validation sets
df_train, df_val = train_test_split(df_train, test_size=0.1, shuffle=True, random_state=0, stratify=df_train['emotion'])
# check
print(df_train['emotion'].value_counts())
print(df_val['emotion'].value_counts())
print(df_test['emotion'].value_counts())

from sklearn.preprocessing import LabelEncoder
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch
import string
import re
import pandas as pd

str_punc = string.punctuation.replace(',', '').replace("'","")

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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.set_device(device)

tokenizer = AutoTokenizer.from_pretrained("../microsoft")
base_model = AutoModelForSequenceClassification.from_pretrained("../microsoft", num_labels=num_classes).to(device)
tokenizer.pad_token = tokenizer.eos_token
base_model.config.pad_token_id = tokenizer.pad_token_id
base_model = base_model.to(device)
# Move y_train_onehot and y_val_onehot to the GPU
y_train_onehot = y_train_onehot.to(device)
y_val_onehot = y_val_onehot.to(device)
# Decide on number of epochs

class AutoMemoryModule(nn.Module):
    def __init__(self, embedding_layer, max_sentence_length, max_memory_context, embedding_size, padding_token, device='cpu'):
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
        # ... (rest of the code remains the same)
        if memory_context is None:
            memory_context = torch.zeros(self.max_memory_context, dtype=torch.long).to(device=self.device)
            # fill memory context with padding tokens
            memory_context.fill_(self.padding_token)

        batch_size, seq_len = input_tokens.shape
        input_tokens = input_tokens.to(device=self.device)
        padded_input_tokens = nn.functional.pad(input_tokens, (0, self.max_sentence_length - seq_len),  value=self.padding_token)
        # score the padded input tokens
        input_tokens_embedding = self.embedding(padded_input_tokens).view(batch_size, seq_len, self.embedding_size).view(-1, self.embedding_size)
        input_tokens_scoring = self.score_net_input_tokens(input_tokens_embedding).squeeze().view(batch_size, seq_len)

        # score the memory context
        memory_context_embedding = self.embedding(memory_context).view(-1, self.embedding_size)
        memory_context_scoring = self.score_net_memory_context(memory_context_embedding).squeeze().view(-1, self.max_memory_context)

        # filter out the padding tokens from the padded input tokens and their scores
        padding_token_idx = torch.nonzero(padded_input_tokens.view(-1) != self.padding_token).squeeze(dim=1)
        filtered_input_tokens = padded_input_tokens.view(-1)[padding_token_idx]
        filtered_input_tokens_scoring = input_tokens_scoring.view(-1)[padding_token_idx]

        ctx_padding_token_idx = torch.nonzero(memory_context.view(-1) != self.padding_token).squeeze(dim=1)
        filtered_memory_context = memory_context.view(-1)[ctx_padding_token_idx]
        filtered_memory_context_scoring = memory_context_scoring.view(-1)[ctx_padding_token_idx]


        # combine the filtered input tokens and their scores with the memory context
        # and their scores
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
        trimmed_combined_tokens = nn.functional.pad(trimmed_combined_tokens, (0, self.max_memory_context - trimmed_combined_tokens.shape[-1]), value=self.padding_token)
        trimmed_scores = nn.functional.pad(trimmed_scores, (0, self.max_memory_context - trimmed_scores.shape[-1]), value=-1e20)
        # print("Padded input tokens shape:", padded_input_tokens.shape)
        # print("Max index in padded input tokens:", padded_input_tokens.max().item())
        # print("Memory context shape:", memory_context.shape)
        # print("Max index in memory context:", memory_context.max().item())
        # print("Trimmed combined tokens shape:", trimmed_combined_tokens.shape)
        # print("Max index in trimmed combined tokens:", trimmed_combined_tokens.max().item())

        return trimmed_combined_tokens, trimmed_scores
    
# Import necessary modules
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
from optimization import BertAdam
# Define the classification model with memory
class DialoGPTWithMemory(nn.Module):
    def __init__(self, base_model, auto_memory_module):
        super().__init__()
        self.base_model = base_model
        self.auto_memory_module = auto_memory_module

    def forward(self, input_ids, attention_mask, memory_context):
      # print(f"Shapes - input_ids: {input_ids.shape}, attention_mask: {attention_mask.shape}, memory_context: {memory_context.shape}")
      # Update the memory context with the input tokens
      new_memory_context, _ = self.auto_memory_module(input_ids, memory_context)
      # print(f"Shape of new_memory_context: {new_memory_context.shape}")
      # # Reshape new_memory_context to have two dimensions
      # new_memory_context = new_memory_context.unsqueeze(0)
      # # Use the updated memory context as input to the base mode
      # outputs = self.base_model(input_ids=new_memory_context, attention_mask=attention_mask)
      # # Assuming new_memory_context is the combined tokens for the entire batch, you can reshape it as:
      new_memory_context = new_memory_context.reshape(batch_size, -1)
      # Now new_memory_context should have shape: torch.Size([4, 2048])
      outputs = self.base_model(input_ids=new_memory_context, attention_mask=attention_mask)
      # print(f"Shape of outputs: {outputs.logits.shape}")
      return outputs, new_memory_context
for lr in [0.00005]:
    for batch_size in [4]:
        # Convert the training and validation data to tensors
        X_train_enc = tokenizer(X_train, padding='max_length', truncation=True, max_length=512, return_tensors="pt")
        X_val_enc = tokenizer(X_val, padding='max_length', truncation=True, max_length=512, return_tensors="pt")
        # Create data loaders
        train_dataset = TensorDataset(X_train_enc['input_ids'], X_train_enc['attention_mask'], y_train_onehot)
        val_dataset = TensorDataset(X_val_enc['input_ids'], X_val_enc['attention_mask'], y_val_onehot)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        # Define your hyperparameters
        token_dim = 100
        max_memory_size = 2048
        max_sentence_length = 512
        num_classes = len(y_train.unique())
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Decide on number of epochs
        n_epochs = 20
        vocab_size = tokenizer.vocab_size
        # Use the embedding layer from the DialoGPT model
        embedding_layer = base_model.get_input_embeddings()
        # # Then, pass this embedding layer to the AutoMemoryModule
        auto_memory_module = AutoMemoryModule(embedding_layer, max_sentence_length, max_memory_size, embedding_size=768, padding_token=0, device=device)
        # Initialize the base model
        base_model = AutoModelForSequenceClassification.from_pretrained("../microsoft", num_labels=num_classes).to(device)
        # Initialize the DialoGPTWithMemory model
        model = DialoGPTWithMemory(base_model, auto_memory_module).to(device)
        freeze_layers = [f'h.{i}.' for i in range(0, 11)]

        for name, param in model.named_parameters():
            for t in freeze_layers:
                if t in name:
                    param.requires_grad = False
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name, param.size())
        embedding_size = model.base_model.get_input_embeddings().weight.size(0)
        # print("Vocabulary size:", vocab_size)
        # print("Embedding layer size:", embedding_size)
        assert vocab_size == embedding_size, "Vocabulary size does not match the embedding layer size!"
        # Define loss function (criterion) and optimizer
        # criterion = nn.CrossEntropyLoss()
        criterion = nn.BCEWithLogitsLoss()
        # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.00001)
        optimizer = BertAdam(model.parameters(), lr=lr)
        # Check if the tokenizer has a padding token
        # print("Tokenizer padding token:", tokenizer.pad_token)

        # If it's None, you can set it
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Now, make sure the model's configuration has the same padding token ID
        base_model.config.pad_token_id = tokenizer.pad_token_id

        # print("Model padding token ID:", base_model.config.pad_token_id)

        # Create an empty dataframe to store the metrics for each epoch
        metrics_df = pd.DataFrame(columns=['epoch', 'train_loss', 'val_loss', 'precision', 'recall', 'f1_score'])
        count_time = 0
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
                # Zero the gradients
                optimizer.zero_grad()
                new_memory_context, _ = auto_memory_module(input_ids, memory_context)
                # Forward pass
                outputs, _ = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device), memory_context=new_memory_context)
                # Compute loss
                loss = criterion(outputs.logits, labels.to(device))
                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                #print(loss)
                # print(loss)
                total_loss += loss.item()
                total_batches += 1
                if count_time % 20 ==0:
                    print(epoch, count_time, total_loss/total_batches)
                count_time += 1
                # print(memory_context)
                memory_context = new_memory_context
            print(f"Epoch {epoch+1}/{n_epochs}, Training Loss: {total_loss / total_batches}")
            # Validation loop
            model.eval()
            precision_scores = []
            recall_scores = []
            f1_scores = []
            total_val_loss = 0
            total_val_batches = 0
            # Initialize memory context for each new validation epoch
            memory_context_val = torch.zeros((batch_size, max_memory_size), dtype=torch.long, device=device)
            with torch.no_grad():
                for input_ids_val, attention_mask_val, labels_val in val_loader:
                    # Forward pass
                    new_memory_context_val, _ = auto_memory_module(input_ids_val, memory_context_val)
                    outputs_val, _ = model(input_ids=input_ids_val.to(device), attention_mask=attention_mask_val.to(device), memory_context=new_memory_context_val)
                    # Compute loss
                    loss_val = criterion(outputs_val.logits, labels_val.to(device))
                    total_val_loss += loss_val.item()
                    total_val_batches += 1
                    memory_context_val = new_memory_context_val
                    # Apply sigmoid to the outputs and get the predicted classes
                    preds = torch.argmax(torch.sigmoid(outputs_val.logits), dim=1)
                    # Convert one-hot encoded labels to class indices
                    labels_val_indices = torch.argmax(labels_val, dim=1)
                    precision, recall, f1, _ = precision_recall_fscore_support(labels_val_indices.cpu().numpy(), preds.cpu().numpy(), average='macro', zero_division=1)  # Modify here
                    precision_scores.append(precision)
                    recall_scores.append(recall)
                    f1_scores.append(f1)
            avg_precision = np.mean(precision_scores)
            avg_recall = np.mean(recall_scores)
            avg_f1 = np.mean(f1_scores)
            print(f'Avg Precision: {avg_precision}, Avg Recall: {avg_recall}, Avg F1: {avg_f1}')
            print(f"Epoch {epoch+1}/{n_epochs}, Validation Loss: {total_val_loss / total_val_batches}")
            # Append the metrics for this epoch to the dataframe
            new_row = pd.DataFrame({
            'epoch': [epoch+1],
            'train_loss': [total_loss / total_batches],
            'val_loss': [total_val_loss / total_val_batches],
            'precision': [avg_precision],
            'recall': [avg_recall],
            'f1_score': [avg_f1]
            })
        metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)
        metrics_df.to_csv(f"lr_{lr}_bs_{batch_size}_jieguo.csv", index=False)
