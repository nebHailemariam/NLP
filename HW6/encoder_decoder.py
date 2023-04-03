import operator
import os
import pickle
import random
import re
import string
import unicodedata

import numpy as np
import sentencepiece as spm
import torch
import torch.nn.functional as F
from sacrebleu.metrics import BLEU, CHRF
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)
MAX_LENGTH = 20


"""## Model Definitions

Now we need to define our models. In general, we need to inherit the `torch.nn.Module` class, define an `__init__` function where we create the layers in our model, and a `forward` function where we compute the outputs from the inputs.

You can check [here](https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html) for a more detailed tutorial on building neural networks in Pytorch.

For the encoder, we use a bidirectional LSTM.
"""

class EncoderRNN(torch.nn.Module):
    def __init__(self, input_size, emb_size, hidden_size, num_layers=2, dropout=0.3):
        super().__init__()

        # TO-DO: Create nn.Embedding layer with (input_size, emb_size)
        self.embedding = None

        # TO-DO: Create nn.LSTM layer with (emb_size, hidden_size)
        self.lstm = torch.nn.LSTM(
            input_size=# TO-DO,
            hidden_size=# TO-DO,
            num_layers=# TO-DO,
            batch_first=True,
            bidirectional=True,
            dropout=# TO-DO,
        )

    def forward(self, inputs):
        # TO-DO: Run the input through the embedding layer
        embedded = None

        # TO-DO: Run both the embedded and hidden through LSTM
        output, hidden = None

        # Return both output and hidden
        return output, hidden


"""We use an LSTM decoder. The model definitions will be very similar except that we add a linear layer to project the hidden representations to the output tokens."""

class DecoderRNN(torch.nn.Module):
    def __init__(self, output_size, emb_size, hidden_size, num_layers=4, dropout=0.3):
        super().__init__()

        # TO-DO: Create nn.Embedding layer with (output_size, emb_size)
        self.embedding = None

        # TO-DO: Create nn.LSTM layer with (emb_size, hidden_size)
        # Make sure to set batch_first=True
        self.lstm = torch.nn.LSTM(
            input_size=# TO-DO,
            hidden_size=# TO-DO,
            num_layers=# TO-DO,
            batch_first=True,
            bidirectional=False,
            dropout=# TO-DO,
        )

        # TO-DO: Create a nn.Linear layer with (hidden_size, output_size)
        self.proj = None

    def forward(self, inputs, hidden):
        # TO-DO: Run the input through the embedding layer
        inputs = None

        # TO-DO: Run both the input and hidden through LSTM
        outputs, hidden = None

        # TO-DO: Run the output through the linear layer
        outputs = None

        # Return both output and hidden
        return outputs, hidden


"""Helper functions to convert the sentences into vector. Given a list of sentence, we use a Sentencepiece model to tokenize it and convert it into a list of word IDs.

Check [Sentencepiece documentations](https://github.com/google/sentencepiece/blob/master/python/README.md#usage) to see how to use the models to convert text into ids.
"""

def preprocess(sp, data):
    # TO-DO: use the Sentencepiece model to tokenize and turn the sentences into ids
    # remember to add bos and eos tokens (from the Sentencepiece model) to the beginning and end of each sentence
    raise NotImplemented
    return None

"""### Training Loop

Implement the main training loop here. This function takes in one batch of input and target tensors and does a forward pass, backward pass, and weight updates. 
"""

def train(
    input_tensor,
    target_tensor,
    encoder,
    decoder,
    optimizer,
    criterion,
    use_teacher_forcing,
    training=True,
):
    if training:
        # TO-DO: Reset/Zero parameter gradients of the optimizer. Hint: https://pytorch.org/tutorials/recipes/recipes/zeroing_out_gradients.html#zero-the-gradients-while-training-the-network
        raise NotImplemented

    loss = 0

    encoder_output, encoder_hidden = encoder(input_tensor)
    dec_length = target_tensor.size()[-1]

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        # Remember that we are giving target[:-1] as the input, and matching targe[1:] with the output
        
        # TO-DO: Run decoder by providing target and encoder_hidden as input
        decoder_output, _ = None

        # TO-DO: Calculate loss
        loss = None

    else:
        dec_length = 0
        # Without teacher forcing: use its own predictions as the next input
        decoder_input = torch.tensor(
            [[SOS_token]] * target_tensor.size()[0], device=device
        )
        last_hidden = encoder_hidden
        for di in range(target_tensor.size()[-1] - 1):
            dec_length += 1
            # TO-DO: Run decoder by providing decoder_input and decoder_hidden as input
            decoder_output, last_hidden = None

            # Take the top output of current timestep of decoder. This will be input to next timestep
            topv, topi = decoder_output.topk(1, dim=-1)
            decoder_input = topi.squeeze(-1).detach()  # detach from history as input

            loss += criterion(decoder_output.squeeze(-2), target_tensor[:,di+1])
            if not training and torch.sum(decoder_input) == 0:
                break

    if training:
        # TO-DO: Backprop by calling backward() function on loss
        raise NotImplemented

        # TO-DO: Update weights using step() on both encoder_optimizer and decoder_optimizer
        raise NotImplemented

    return loss.item() / dec_length


def predict(encoder, decoder, sentence, src_sp, tgt_sp):
    with torch.no_grad():
        input_tensor = [[src_sp.bos_id()] + src_sp.encode_as_ids(sentence) + [src_sp.eos_id()]]
        input_tensor = torch.tensor(input_tensor, device=device)

        encoder_output, encoder_hidden = encoder(input_tensor)
        decoder_input = torch.tensor([[tgt_sp.bos_id()]], device=device)
        decoder_hidden = encoder_hidden

        decoded_ids = []

        for di in range(MAX_LENGTH):
            # TO-DO: generate next output and hidden state from decoder_input and decoder_hidden
            decoder_output, decoder_hidden = None

            # TO-DO: get the id of the most likely item from decoder_output
            _, topi = None
            if topi.item() == tgt_sp.eos_id():
                break
            else:
                decoded_ids.append(topi.item())

            decoder_input = topi.squeeze(-1).detach()
        # TO-DO: use the Sentencepiece model to convert the ids back to a string
        decoded_words = None

        return decoded_words

"""Bonus points (20%): implement beam search. This is optional.

To make autograding work, your beam_search should return a list of predictions from the most likely to the least likely, and each prediction should be a tuple of (score, text). For example:

```
[(0,2, "Thank you"), (0.1, "Thanks"), (0.01, "Thanks you")]
```
"""

def beam_search(encoder, decoder, sentence, src_sp, tgt_sp, beam_size=5):
    assert beam_size > 1, "if beam_size = 1, then that's greedy search"
    with torch.no_grad():
        # TO-DO: Just like predict() but instead of the topk with 1, take topk with beam_size and rank sort the beam for the more likely probability
        raise NotImplemented

