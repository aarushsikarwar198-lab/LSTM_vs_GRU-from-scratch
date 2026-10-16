# =============================================================================
# models.py
# What this file does:
#   Contains the three neural network pieces of the project:
#     1. LSTM       - an LSTM layer written by hand (no nn.LSTM)
#     2. GRU        - a GRU layer written by hand (no nn.GRU)
#     3. Classifier - the full model: words -> embedding -> LSTM/GRU -> yes/no answer
# =============================================================================

import torch
import torch.nn as nn

class LSTM(nn.Module):
  def __init__(self , input_features , hidden_size):
      super(LSTM , self).__init__()
      # hidden_size = how many numbers the LSTM uses to remember things
      self.hidden_size =hidden_size
      # input_features = size of the vector for each word coming in (embedding size)
      self.input_features = input_features
      # One linear layer that makes ALL 4 gate values from the input word at once
      # (4 gates x hidden_size numbers).
      self.x_lin = nn.Linear(input_features, 4 * hidden_size)
      # One linear layer that makes the 4 gate values from the previous hidden state.
      # (no bias here, because x_lin already has one)
      self.h_lin = nn.Linear(hidden_size, 4 * hidden_size, bias=False)
      # Gate activations: sigmoid gives numbers between 0 and 1 ("how much to let through")
      self.i_gate = nn.Sigmoid()   # input gate: how much new info to write
      self.f_gate = nn.Sigmoid()   # forget gate: how much old memory to keep
      self.o_gate = nn.Sigmoid()   # output gate: how much memory to show as output
      # Tanh gives numbers between -1 and 1
      self.c_gate = nn.Tanh()      # squashes the memory before it becomes the output
      self.p_gate = nn.Tanh()      # makes the new candidate memory
  def forward(self, x):
      # x has shape: (batch, time steps, embedding size)
      # Start with an empty hidden state (h_p) and empty memory (c_p), all zeros.
      h_p = x.new_zeros(x.size(0), self.hidden_size)
      c_p = x.new_zeros(x.size(0), self.hidden_size)
      # Do the input part of the math for ALL time steps in one go (faster),
      # then split it into one piece per time step.
      x_ta = self.x_lin(x)
      xs = torch.unbind(x_ta , dim = 1)
      # We will keep the hidden state of every time step in this list.
      outputs = []
      # Go through the review one word (time step) at a time.
      for t in range(x.size(1)):
        # Combine the current word's part with the previous hidden state's part.
        u_t = xs[t] + self.h_lin(h_p)
        # Cut the result into 4 equal pieces, one for each gate.
        i_t , f_t , o_t , p_t = torch.split(u_t , self.hidden_size , dim = 1)
        i_t = self.i_gate(i_t)    # input gate values (0 to 1)
        f_t = self.f_gate(f_t)    # forget gate values (0 to 1)
        o_t = self.o_gate(o_t)    # output gate values (0 to 1)
        p_t = self.p_gate(p_t)    # candidate new memory (-1 to 1)
        # Forget part of the old memory.
        c_p = torch.mul(f_t , c_p)
        # Decide how much of the new candidate memory to write.
        i_t = torch.mul(i_t ,p_t)
        # New memory = (kept old memory) + (new written memory).
        c_p = torch.add(c_p , i_t)
        # New hidden state = output gate x tanh(memory).
        h_p = torch.mul(self.c_gate(c_p) , o_t)
        outputs.append(h_p)
      # Stack the hidden states of all time steps into one tensor.
      outputs = torch.stack(outputs, dim=1)
      # Return all outputs, plus the final hidden state and final memory.
      return outputs, (h_p, c_p)


class GRU(nn.Module):
    def __init__(self, input_features, hidden_size):
        super(GRU, self).__init__()
        # hidden_size = how many numbers the GRU uses to remember things
        self.hidden_size = hidden_size
        # size of the vector for each incoming word (embedding size)
        self.input_features = input_features
        # One linear layer that makes the 3 pieces (reset, update, candidate) from the input word.
        self.x_lin = nn.Linear(input_features , 3*hidden_size)
        # Same for the previous hidden state (this one keeps its own bias).
        self.h_lin = nn.Linear(hidden_size , 3*hidden_size)
        # Reset gate: how much of the old memory to use when making the new candidate
        self.r_gate = nn.Sigmoid()
        # Update gate: how much to replace old memory with the new candidate
        self.z_gate = nn.Sigmoid()
        # Tanh for the candidate memory
        self.c_gate = nn.Tanh()

    def forward(self, x):
        # x has shape: (batch, time steps, embedding size)
        # Start with an empty hidden state (all zeros).
        h_p = x.new_zeros(x.size(0), self.hidden_size)
        outputs = []
        # Do the input part of the math for all time steps at once, then split per time step.
        x_ta = self.x_lin(x)
        xs = torch.unbind(x_ta , dim = 1)
        # Go through the review one word (time step) at a time.
        for t in range(x.size(1)):
            x_t = xs[t]
            # The hidden-state part of the math for this step.
            h_t = self.h_lin(h_p)
            # Cut both into 3 equal pieces: (reset, update, candidate).
            x_1 , x_2 , x_3 = torch.split(x_t , self.hidden_size , dim = 1)
            h_1 , h_2 , h_3 = torch.split(h_t , self.hidden_size , dim = 1) 
            # Reset gate (0 to 1)
            r_t = self.r_gate(x_1 + h_1)
            # Update gate (0 to 1)
            z_t = self.z_gate(x_2 + h_2)
            # Candidate new memory; the reset gate decides how much old memory (h_3) it can see
            c_t = self.c_gate(x_3 + torch.mul(r_t , h_3))
            # New hidden state = mix of old state and candidate.
            # z_t near 0 -> keep old memory, z_t near 1 -> take the candidate.
            h_p = (1 - z_t) * h_p + z_t * c_t
            outputs.append(h_p)
        # Stack the hidden states of all time steps into one tensor.
        outputs = torch.stack(outputs, dim=1)
        # Return all outputs and the final hidden state.
        return outputs, h_p


# The full model: turns a review (numbers) into ONE score (> 0 means positive, < 0 means negative).
class Classifier(nn.Module):
    def __init__(self , vocab_size , embed_dim, hidden_size , rnn_type , dropout , pad_idx):
        super(Classifier , self).__init__()
        self.vocab_size = vocab_size      # number of words in the vocabulary
        self.embed_dim = embed_dim        # size of the vector for each word
        self.hidden_size = hidden_size    # memory size of the LSTM/GRU
        self.rnn_type = rnn_type          # "LSTM" or "GRU"
        self.dropout = dropout            # chance of switching off a number during training (0.5 = 50%)
        self.pad_idx = pad_idx            # the number that means "padding"
        # Embedding: gives every word number its own learnable vector.
        # pad_idx tells it that the padding vector should stay zero and not be learned.
        self.embed = nn.Embedding(vocab_size , embed_dim , pad_idx )
        # Choose which recurrent layer to use. Anything that is not "gru" uses the LSTM.
        if(rnn_type.lower() == "gru"):
            self.model = GRU(embed_dim , hidden_size)
        else:
            self.model = LSTM(embed_dim , hidden_size)
        # Dropout helps stop the model from just memorizing the training data.
        self.drop = nn.Dropout(dropout)
        # Final layer: hidden state -> 1 number (the score).
        self.linear = nn.Linear(hidden_size , 1)

    def forward(self, tokens):
        # words (numbers) -> word vectors
        i_p = self.embed(tokens)
        # run the LSTM/GRU over the whole review
        out , state = self.model(i_p)
        # The GRU returns just the final hidden state;
        # the LSTM returns (hidden state, memory), so we take the first one.
        if(self.rnn_type.lower() == 'gru'):
            h = state
        else:
            h = state[0]
        # We only use the FINAL hidden state (it has "read" the whole review,
        # and because of left padding the last position is a real word).
        h = self.drop(h)
        # turn it into one score per review
        h = self.linear(h)
        return h