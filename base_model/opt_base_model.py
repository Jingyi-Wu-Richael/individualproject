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

from transformers import AutoTokenizer, AutoModelForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
base_model = AutoModelForSequenceClassification.from_pretrained("facebook/opt-125m", num_labels=num_classes).to(device)
tokenizer.pad_token = tokenizer.eos_token
base_model.config.pad_token_id = tokenizer.pad_token_id
base_model = base_model.to(device)
# Move y_train_onehot and y_val_onehot to the GPU
y_train_onehot = y_train_onehot.to(device)
y_val_onehot = y_val_onehot.to(device)
# Decide on number of epochs

# Import necessary modules
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
# Define the classification model with memory

class ClassificationModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, **inputs):
        output = self.base_model(**inputs)
        return output.logits

# Initialize the model
model = ClassificationModel(base_model)
# Define loss function (criterion) and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters())

# Decide on number of epochs
n_epochs = 100

# Set the batch size
batch_size = 4

# Convert the training and validation data to tensors
X_train_enc = tokenizer(X_train, padding='max_length', truncation=True, max_length=512, return_tensors="pt")
X_val_enc = tokenizer(X_val, padding='max_length', truncation=True, max_length=512, return_tensors="pt")

# Move encoded inputs to the GPU
X_train_enc['input_ids'], X_train_enc['attention_mask'] = X_train_enc['input_ids'].to(device), X_train_enc['attention_mask'].to(device)
X_val_enc['input_ids'], X_val_enc['attention_mask'] = X_val_enc['input_ids'].to(device), X_val_enc['attention_mask'].to(device)

# Create data loaders
train_dataset = TensorDataset(X_train_enc['input_ids'], X_train_enc['attention_mask'], y_train_onehot)
val_dataset = TensorDataset(X_val_enc['input_ids'], X_val_enc['attention_mask'], y_val_onehot)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# Create an empty dataframe to store the metrics for each epoch
# Training loop
for epoch in range(n_epochs):
    # Set model to training mode
    model.train()

    total_loss = 0
    total_batches = 0

    # Iterate over batches in the training data loader
    for input_ids, attention_mask, labels in train_loader:
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device))#.to(device)
        # outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        # Compute loss
        #loss = criterion(outputs.logits, labels)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        #print(loss)
        total_batches += 1
        if total_batches % 100 == 0:
            print(epoch, total_batches, loss.item())

    print(f"Epoch {epoch+1}/{n_epochs}, Training Loss: {total_loss / total_batches}")

    # Validation loop
    model.eval()
    precision_scores = []
    recall_scores = []
    f1_scores = []
    total_val_loss = 0
    total_val_batches = 0
    labels_val_indices = []
    preds = []
    with torch.no_grad():
        for input_ids_val, attention_mask_val, labels_val in val_loader:
            # Forward pass
            outputs_val = model(input_ids=input_ids_val, attention_mask=attention_mask_val.to(device))#.to(device)

            # Compute loss
            loss_val = criterion(outputs_val, labels_val)

            total_val_loss += loss_val.item()
            total_val_batches += 1

            # Apply sigmoid to the outputs and get the predicted classes
            #preds = torch.argmax(torch.sigmoid(outputs_val), dim=1)

            ## Convert one-hot encoded labels to class indices
            #labels_val_indices = torch.argmax(labels_val, dim=1)

            preds.extend(torch.argmax(outputs_val, dim=1).cpu().numpy())
            labels_val_indices.extend(torch.argmax(labels_val, dim=1).cpu().numpy())
    precision, recall, f1, _ = precision_recall_fscore_support(labels_val_indices, preds, average='macro', zero_division=1)
    result = [epoch + 1, total_loss / total_batches, total_val_loss / total_val_batches, precision, recall, f1]
    print("Eval:", result)
    result = [str(i) for i in result]
    with open(f"{n_epochs}_125opt_result.txt", mode="a", encoding="utf8") as f:
        f.write("\t".join(result) + "\n")
        