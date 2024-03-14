# Importing pytorch
import torch
import torch.nn as nn
from torch.nn import functional as F

import dataclasses 
from typing import Optional
from tokenizer import *

### BASE 
@dataclasses.dataclass # a class meant specifically to just hold data
class BaseConfig:
    """ 
    The default configuration & hyperparameters for FractalFormer
    """
    # The number of tokens in the vocabulary.
    vocab_size: int = tokenizer.vocab_len
    
    # The maximum sequence length that this model might ever be used with.
    max_position_embeddings: int = 256
    
    # The number of layers in the model.
    num_hidden_layers: int = 4
    
    # The number of attention heads used in the attention layers of the model.
    num_attention_heads: int = 4
    
    # The number of key-value heads for implementing multi-query attention.
    num_key_value_heads: int = 1
    # Ensures that the number of query heads is evenly divisible by the number of KV heads.
    assert num_attention_heads % num_key_value_heads == 0
    
    # The hidden size of the model, AKA the embedding dimension
    hidden_size: int = 128
    # the attention heads need to cleanly divide up the hidden_size of the model for MQA
    assert hidden_size % num_attention_heads == 0

    # how much larger the inner dimension of the MLP should be than the hidden size of the model
    intermediate_multiplier = 4
    # The inner dimension of the MLP part of the decoder layer
    @property
    def intermediate_size(self):
        return self.intermediate_multiplier * self.hidden_size
    
    # The number of head dimensions
    head_dim: int = 32
    
    # The epsilon used by the rms normalization layers.
    rms_norm_eps: float = 1e-6 # this is to promote numerical stability & prevent dividing by 0
    
    # the scaling factor that determines the frequencies for the rotary positional encodings
    rope_theta = 100.0
    # smaller models should use a smaller theta, but I'm just guessing here. 1000 might work too. 10,000 is the usual

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # the % of neurons to dropout in the MLP
    dropout = 0.1

    ####### for debugging & visualization
    verbose = {
    'RMSNorm': False,
    'MLP': False,
    'MQA': False,
    'Layer': False,
    'OutputLayer': False,
    'FractalLoss': False,
    'FractalFormer': False,
    'Model': False,
    'Sampler': False,
    'Generate': False
    }

    ####### FractalFormer-specific hyperparameters

    # the number of levels for sub-models to exist on
    levels = 3
    
    # the number of splits to make at a given level
    split = 2 # i don't recommend choosing any value other than 2
    # needs to be divisible by 2 in order to splice cleanly
    assert split % 2 == 0
    # RoPE requires a head dimension of length larger than 1 in order to work
    assert head_dim // (split * (levels-1)) > 1
    # really though you shouldn't be getting anywhere near that small of a head dimension even at the lowest level, that'd be useless

    @property
    def model_count(self):
        return [self.split**i for i in range(self.levels)]

    @property
    def model_dim_list(self):
        return [self.hidden_size // (self.split**i) for i in range(self.levels)]

    @property
    def head_dim_list(self):
        return [self.head_dim // (self.split**i) for i in range(self.levels)]