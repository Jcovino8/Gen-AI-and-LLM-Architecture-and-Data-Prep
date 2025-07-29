# Lab 3 - Creating an NLP Data Loader

import torchtext
print(torchtext.__version__)
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import multi30k, Multi30k
from typing import Iterable, List
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchdata.datapipes.iter import IterableWrapper, Mapper
import torchtext

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import random

sentences = [
    "If you want to know what a man's like, take a good look at how he treats his inferiors, not his equals.",
    "Fame's a fickle friend, Harry.",
    "It is our choices, Harry, that show what we truly are, far more than our abilities.",
    "Soon we must all face the choice between what is right and what is easy.",
    "Youth can not know how age thinks and feels. But old men are guilty if they forget what it was to be young.",
    "You are awesome!"
]

# Define a custom dataset
class CustomDataset(Dataset):
    def __init__(self, sentences):
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]

# Create an instance of your custom dataset
custom_dataset = CustomDataset(sentences)

# Define batch size
batch_size = 2

# Create a DataLoader
dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)

# Iterate through the DataLoader
for batch in dataloader:
    print(batch)


sentences = [
    "If you want to know what a man's like, take a good look at how he treats his inferiors, not his equals.",
    "Fame's a fickle friend, Harry.",
    "It is our choices, Harry, that show what we truly are, far more than our abilities.",
    "Soon we must all face the choice between what is right and what is easy.",
    "Youth can not know how age thinks and feels. But old men are guilty if they forget what it was to be young.",
    "You are awesome!"
]

# Define a custom data set
class CustomDataset(Dataset):
    def __init__(self, sentences, tokenizer, vocab):
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.vocab = vocab

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        tokens = self.tokenizer(self.sentences[idx])
        # Convert tokens to tensor indices using vocab
        tensor_indices = [self.vocab[token] for token in tokens]
        return torch.tensor(tensor_indices)

# Tokenizer
tokenizer = get_tokenizer("basic_english")

# Build vocabulary
vocab = build_vocab_from_iterator(map(tokenizer, sentences))

# Create an instance of your custom data set
custom_dataset = CustomDataset(sentences, tokenizer, vocab)

print("Custom Dataset Length:", len(custom_dataset))
print("Sample Items:")
for i in range(6):
    sample_item = custom_dataset[i]
    print(f"Item {i + 1}: {sample_item}")

# Create an instance of your custom data set
custom_dataset = CustomDataset(sentences, tokenizer, vocab)

# Define batch size
batch_size = 2

# Create a data loader
#dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)

# Iterate through the data loader
for batch in dataloader:
    print(batch)

# Create a custom collate function
def collate_fn(batch):
    # Pad sequences within the batch to have equal lengths
    padded_batch = pad_sequence(batch, batch_first=True, padding_value=0)
    return padded_batch

# Create a data loader with the custom collate function with batch_first=True,
dataloader = DataLoader(custom_dataset, batch_size=batch_size, collate_fn=collate_fn)

# Iterate through the data loader
for batch in dataloader: 
    for row in batch:
        for idx in row:
            words = [vocab.get_itos()[idx] for idx in row]
        print(words)

# Create a custom collate function
def collate_fn_bfFALSE(batch):
    # Pad sequences within the batch to have equal lengths
    padded_batch = pad_sequence(batch, padding_value=0)
    return padded_batch



# Create a data loader with the custom collate function with batch_first=True,
dataloader_bfFALSE = DataLoader(custom_dataset, batch_size=batch_size, collate_fn=collate_fn_bfFALSE)

# Iterate through the data loader
for seq in dataloader_bfFALSE:
    for row in seq:
        #print(row)
        words = [vocab.get_itos()[idx] for idx in row]
        print(words)

# Iterate through the data loader with batch_first = TRUE
for batch in dataloader:    
    print(batch)
    print("Length of sequences in the batch:",batch.shape[1])

# Define a custom data set
class CustomDataset(Dataset):
    def __init__(self, sentences):
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]

custom_dataset=CustomDataset(sentences)

custom_dataset[0]

def collate_fn(batch):
    # Tokenize each sample in the batch using the specified tokenizer
    tensor_batch = []
    for sample in batch:
        tokens = tokenizer(sample)
        # Convert tokens to vocabulary indices and create a tensor for each sample
        tensor_batch.append(torch.tensor([vocab[token] for token in tokens]))

    # Pad sequences within the batch to have equal lengths using pad_sequence
    # batch_first=True ensures that the tensors have shape (batch_size, max_sequence_length)
    padded_batch = pad_sequence(tensor_batch, batch_first=True)
    
    # Return the padded batch
    return padded_batch

# Create a data loader for the custom dataset
dataloader = DataLoader(
    dataset=custom_dataset,   # Custom PyTorch Dataset containing your data
    batch_size=batch_size,     # Number of samples in each mini-batch
    shuffle=True,              # Shuffle the data at the beginning of each epoch
    collate_fn=collate_fn      # Custom collate function for processing batches
)

for batch in dataloader:
    print(batch)
    print("shape of sample",len(batch))


# Exercise 1: Create a data loader with a collate function that processes batches of French text 
corpus = [
    "Ceci est une phrase.",
    "C'est un autre exemple de phrase.",
    "Voici une troisième phrase.",
    "Il fait beau aujourd'hui.",
    "J'aime beaucoup la cuisine française.",
    "Quel est ton plat préféré ?",
    "Je t'adore.",
    "Bon appétit !",
    "Je suis en train d'apprendre le français.",
    "Nous devons partir tôt demain matin.",
    "Je suis heureux.",
    "Le film était vraiment captivant !",
    "Je suis là.",
    "Je ne sais pas.",
    "Je suis fatigué après une longue journée de travail.",
    "Est-ce que tu as des projets pour le week-end ?",
    "Je vais chez le médecin cet après-midi.",
    "La musique adoucit les mœurs.",
    "Je dois acheter du pain et du lait.",
    "Il y a beaucoup de monde dans cette ville.",
    "Merci beaucoup !",
    "Au revoir !",
    "Je suis ravi de vous rencontrer enfin !",
    "Les vacances sont toujours trop courtes.",
    "Je suis en retard.",
    "Félicitations pour ton nouveau travail !",
    "Je suis désolé, je ne peux pas venir à la réunion.",
    "À quelle heure est le prochain train ?",
    "Bonjour !",
    "C'est génial !"
]
for batch in dataloader:
    print(batch)

def collate_fn_fr(batch):
    # Pad sequences within the batch to have equal lengths
    tensor_batch=[]
    for sample in batch:
        tokens = tokenizer(sample)
        tensor_batch.append(torch.tensor([vocab[token] for token in tokens]))
         
    padded_batch = pad_sequence(tensor_batch,batch_first=True)
    return padded_batch

# Build tokenizer
tokenizer = get_tokenizer('spacy', language='fr_core_news_sm')

# Build vocabulary
vocab = build_vocab_from_iterator(map(tokenizer, corpus))

# Sort sentences based on their length
sorted_data = sorted(corpus, key=lambda x: len(tokenizer(x)))
#print(sorted_data)
dataloader = DataLoader(sorted_data, batch_size=4, shuffle=False, collate_fn=collate_fn_fr)

for batch in dataloader:
    print(batch)

# [Optional] Data loader for German - English translation task

# You would modify the URLs for the data set since the links to the original data set are broken

multi30k.URL["train"] = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMSkillsNetwork-AI0205EN-SkillsNetwork/training.tar.gz"
multi30k.URL["valid"] = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMSkillsNetwork-AI0205EN-SkillsNetwork/validation.tar.gz"

SRC_LANGUAGE = 'de'
TGT_LANGUAGE = 'en'

train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
data_set = iter(train_iter)

for n in range(5):
    # Getting the next pair of source and target sentences from the training data set
    src, tgt = next(data_set)

    # Printing the source (German) and target (English) sentences
    print(f"sample {str(n+1)}")
    print(f"Source ({SRC_LANGUAGE}): {src}\nTarget ({TGT_LANGUAGE}): {tgt}")

german, english = next(data_set)
print(f"Source German ({SRC_LANGUAGE}): {german}\nTarget English  ({TGT_LANGUAGE}): { english }")

from torchtext.data.utils import get_tokenizer

# Making a placeholder dict to store both tokenizers
token_transform = {}

token_transform[SRC_LANGUAGE] = get_tokenizer('spacy', language='de_core_news_sm')
token_transform[TGT_LANGUAGE] = get_tokenizer('spacy', language='en_core_web_sm')

token_transform['de'](german)
token_transform['en'](english)

# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

#place holder dict for 'en' and 'de' vocab transforms
vocab_transform = {}

def yield_tokens(data_iter: Iterable, language: str) -> List[str]:
    # Define a mapping to associate the source and target languages
    # with their respective positions in the data samples.
    language_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}

    # Iterate over each data sample in the provided dataset iterator
    for data_sample in data_iter:
        # Tokenize the data sample corresponding to the specified language
        # and yield the resulting tokens.
        yield token_transform[language](data_sample[language_index[language]])

for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    # Training data iterator
    train_iterator = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    #To decrease the number of padding tokens, you sort data on the source length to batch similar-length sequences together
    sorted_dataset = sorted(train_iterator, key=lambda x: len(x[0].split()))
    # Create torchtext's Vocab object
    vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(sorted_dataset, ln),
                                                    min_freq=1,
                                                    specials=special_symbols,
                                                    special_first=True)

# If not set, it throws ``RuntimeError`` when the queried token is not found in the Vocabulary.
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
  vocab_transform[ln].set_default_index(UNK_IDX)

seq_en=vocab_transform['en'](token_transform['en'](english))
print(f"English text string: {english}\n English sequence: {seq_en}")

seq_de=vocab_transform['de'](token_transform['de'](german))
print(f"German text string: {german}\n German sequence: {seq_de}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# function to add BOS/EOS, flip source sentence and create tensor for input sequence indices
def tensor_transform_s(token_ids: List[int]):
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.flip(torch.tensor(token_ids), dims=(0,)),
                      torch.tensor([EOS_IDX])))

# function to add BOS/EOS and create tensor for input sequence indices
def tensor_transform_t(token_ids: List[int]):
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))

seq_en=tensor_transform_s(seq_en)
seq_en
seq_de=tensor_transform_t(seq_de)
seq_de
# helper function to club together sequential operations
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

# ``src`` and ``tgt`` language text transforms to convert raw strings into tensors indices
text_transform = {}

text_transform[SRC_LANGUAGE] = sequential_transforms(token_transform[SRC_LANGUAGE], #Tokenization
                                            vocab_transform[SRC_LANGUAGE], #Numericalization
                                            tensor_transform_s) # Add BOS/EOS and create tensor

text_transform[TGT_LANGUAGE] = sequential_transforms(token_transform[TGT_LANGUAGE], #Tokenization
                                            vocab_transform[TGT_LANGUAGE], #Numericalization
                                            tensor_transform_t) # Add BOS/EOS and create tensor


# function to collate data samples into batch tensors
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_sequences = text_transform[SRC_LANGUAGE](src_sample.rstrip("\n"))
        src_sequences = torch.tensor(src_sequences, dtype=torch.int64)
        tgt_sequences = text_transform[TGT_LANGUAGE](tgt_sample.rstrip("\n"))
        tgt_sequences = torch.tensor(tgt_sequences, dtype=torch.int64)
        src_batch.append(src_sequences)
        tgt_batch.append(tgt_sequences)

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX,batch_first=True)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX,batch_first=True)
    
    return src_batch.to(device), tgt_batch.to(device)

BATCH_SIZE = 4

train_iterator = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
sorted_train_iterator = sorted(train_iterator, key=lambda x: len(x[0].split()))
train_dataloader = DataLoader(sorted_train_iterator, batch_size=BATCH_SIZE, collate_fn=collate_fn,drop_last=True)

valid_iterator = Multi30k(split='valid', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
sorted_valid_dataloader = sorted(valid_iterator, key=lambda x: len(x[0].split()))
valid_dataloader = DataLoader(sorted_valid_dataloader, batch_size=BATCH_SIZE, collate_fn=collate_fn,drop_last=True)


src, trg = next(iter(train_dataloader))
src,trg












