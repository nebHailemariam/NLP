import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

class Classifier(nn.Module):
    def __init__(
        self,
        word_vocab_size,
        word_emb_size,
        pos_vocab_size,
        pos_emb_size,
        depl_vocab_size,
        depl_emb_size,
        out_size,
        layer_sizes,
        dropout,
    ):
        super(Classifier, self).__init__()

        # TODO: Define the embedding layers according to the relevant vocab_size and embedding size
        self.word_emb = 
        self.pos_emb = 
        self.depl_emb = 

        # TODO: Determine the input size to the hidden layers using the formula provided in the handout
        in_size = 

        input_sizes = [in_size] + layer_sizes[:-1]
        output_sizes = layer_sizes
        layers = []

        # Adding the hidden layers
        for s1, s2 in zip(input_sizes, output_sizes):

            # TODO: Append a linear layer to layers with the input and output size
            

            # TODO: Append a LeakyReLu layer, feel free to experiment here
            

            # TODO: Append a dropout layer with our input dropout value
            

        # TODO: Append the output layer using the layer size of the last year 
        # as the input and out_size as the output
        

        # TODO: Initialize the layers of the networks by passing the 
        # unpacked list through a PyTorch Sequential container
        self.layers = 

    def forward(self, inputs):
        # TODO: Define the inputs to the each embedding layer
        # Hint: Each element in input has length 52, to be separated into three parts
        word_emb_input = 
        pos_emb_input = 
        depl_emb_input = 

        # Here we pass the inputs through the embedding layers and concatenate their outputs.
        embs = [
            *self.word_emb(word_emb_input).split(1, dim=1),
            *self.pos_emb(pos_emb_input).split(1, dim=1),
            *self.depl_emb(depl_emb_input).split(1, dim=1),
        ]
        embs = [torch.squeeze(tens) for tens in embs]
        embs = torch.cat(embs, dim=-1)

        # TODO: Pass the output of the embeddings layer through the rest of the layers.
        output = 
        return output