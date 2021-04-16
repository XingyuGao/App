import streamlit as st
import torch
import spacy
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import random
from torchtext import data
import pandas as pd
from torchtext.legacy import data
import torch.nn as nn
import en_core_web_sm

#Set a seed
SEED = 1234

#Initialize the set seed with some parameters, make sure the output is the same each time
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

#Initialize the TEXT and LABEL, determine how emotions and comments will be processed
#As you can see when doing tokenize, I set spacy as my tokenize method.
#At the same time, I used en_core_web_sm as the tokenize language.
#Since this is a small data set, medium and large scale tokenize languages are not required.
TEXT = data.Field(tokenize = 'spacy',
                  tokenizer_language = 'en_core_web_sm',
                  batch_first = True)
LABEL = data.LabelField(dtype = torch.float)

# A huge and corpulent function, which is used to load and process the data I processed earlier
# In addition, I set MAX_VOCAB_SIZE because the vocabulary is so full
# that I can only select the first 25,000 words that appear most frequently
def load_file(filepath, device, MAX_VOCAB_SIZE=25_000):
    # First, set fields, the parameters to process the dataset
    tv_datafields = [("text", TEXT), ("label", LABEL)]

    # Read in data sets and apply parameters to process data sets, generating training, validation, and test sets
    train = data.TabularDataset.splits(path=filepath,train="Train.csv", format="csv",
                                                    skip_header=True, fields=tv_datafields)[0]

    # Generate vocabulary (that is, indexes)
    TEXT.build_vocab(train,
                 max_size = MAX_VOCAB_SIZE,
                 vectors = "glove.6B.100d",
                 unk_init = torch.Tensor.normal_)

    LABEL.build_vocab(train)

    return TEXT, LABEL, train

#Set an appropriate BATCH_SIZE, that is, the amount of training each time
BATCH_SIZE = 64

#Add device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Processing the dataset by using function load_file
TEXT, LABEL, train= load_file('data',device)

# Establishing a 2-D CNN Model
class CNN(nn.Module):

    # First, define a initilize function
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim,
                 dropout, pad_idx):
        super().__init__()

        # Defining the embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        # Put all the convl into a nn.ModuleList make it convient to be use by other modules
        # Define the parameters of the convolution, the input channel is one because there's only text input
        # Number of output channels is number of filters
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=n_filters,
                      kernel_size=(fs, embedding_dim))
            for fs in filter_sizes
        ])

        # Define the full connection layer
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)

        self.dropout = nn.Dropout(dropout)

    # Then, a forward function
    def forward(self, text):
        # Text
        embedded = self.embedding(text)

        # Embedded layer
        embedded = embedded.unsqueeze(1)

        # Convolution layer
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]

        # Pooling layer
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        cat = self.dropout(torch.cat(pooled, dim=1))

        return self.fc(cat)

#Set some parameters related to CNN, such as input dimension, embedding layer number, and so on
INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
N_FILTERS = 100
FILTER_SIZES = [3,4,5]
OUTPUT_DIM = 1
DROPOUT = 0.5
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

#Build the model!
model = CNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, PAD_IDX)

#Then load the pre-trained embeddings
pretrained_embeddings = TEXT.vocab.vectors
model.embedding.weight.data.copy_(pretrained_embeddings)

#Zero the initial weights of the unknown and padding tokens.
UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)


#Load the model I have alread traind
model = torch.load('Amazon-food3-model.pt', map_location='cpu')

#Load spacy
nlp = spacy.load('en_core_web_sm')

#Predict funciton
def predict_sentiment(model, sentence, min_len=5):
    # Set in Evaluate mode
    model.eval()

    # Tokenizes the sentence
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]

    # Complete its length
    if len(tokenized) < min_len:
        tokenized += ['<pad>'] * (min_len - len(tokenized))

    # Indexes the tokens
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]

    # Converts into a tensor
    tensor = torch.LongTensor(indexed).to(device)

    # Add a batch dimension
    tensor = tensor.unsqueeze(0)

    # Output prediction
    prediction = torch.sigmoid(model(tensor))

    return prediction.item()

#Text analyz
def text_analyzer(my_test):
    result = predict_sentiment(model, my_test)
    return result

#Finally, I create streamlit web page
st.title("NLP with Amazon fine food")
message = st.text_area("Enter Your Text")
if st.button("Analyze"):
    nlp_result = text_analyzer(message)
    if nlp_result >= 0.5:
        st.success("This sentence is more likely to be a negative review, the score is:" + str(nlp_result))
    else:
        st.success("This sentence is more likely to be a positive review, the score is:" + str(nlp_result))