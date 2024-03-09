# Importing pytorch
import torch
import torch.nn as nn
from torch.nn import functional as F

# the tokenizer
from tokenizer import *

# the config
from configs import BaseConfig as Config
config = Config()

# Imports used for the model
import re
from typing import Any, List, Sequence, Tuple, Union
import numpy as np

def apply_rotary_emb(x: torch.Tensor, dim: int, theta: float = 10000.0) -> torch.Tensor:
    """Applies the rotary embedding to the inputted query or key tensor"""
    # Get sequence length
    seq_len = x.size(1)
    device = x.device
    
    # Dynamically compute frequency cis based on the input sequence length
    # dynamic is less efficient but pre-computed was giving me trouble so whatever
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(seq_len, device=device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64

    # Apply rotary embeddings to the input tensor
    x_ = torch.view_as_complex(torch.stack(torch.chunk(x.transpose(1, 2).float(), 2, dim=-1), dim=-1))
    x_out = torch.view_as_real(x_ * freqs_cis.unsqueeze(0)).type_as(x)  # Ensure batch dimension is handled
    x_out = torch.cat(torch.chunk(x_out, 2, dim=-1), dim=-2)
    x_out = x_out.reshape(x_out.shape[0], x_out.shape[1], x_out.shape[2], -1).transpose(1, 2)

    return x_out

class RMSNorm(torch.nn.Module):
    """
    Implements the RMS Normalization (Root Mean Square Normalization) layer.
    RMSNorm is a variant of layer normalization that normalizes the activations
    of the previous layer based on their root mean square value.

    Parameters:
    - dim (int): The dimension of the input features the normalization is applied to.
    - eps (float): A small value added to the denominator for numerical stability. Default is 1e-6.
    - add_unit_offset (bool): If True, adds a unit (1) to the learned scaling coefficient, effectively
      starting with no scaling. If False, the scaling coefficient starts from zero. Default is True.
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        #add_unit_offset: bool = True,
    ):
        super().__init__() 
        self.eps = eps  # Small epsilon value for numerical stability since you can't divide by 0
        #self.add_unit_offset = add_unit_offset  # Flag to determine if a unit should be added to the weight
        
        # Initialize the weight parameter with zeros, which will be learned during training.
        # The shape of the weight is [dim], meaning one weight per feature dimension.
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        """
        Private helper function to normalize the input tensor.

        Parameters:
        - x (Tensor): The input tensor to normalize.

        Returns:
        - Tensor: The normalized tensor.
        """
        # Calculate the root mean square value for each feature (across the last dimension),
        # then use reciprocal square root (rsqrt) for normalization.
        # Add self.eps to the denominator for numerical stability.
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor, model: int = 0) -> torch.Tensor:
        """
        Forward pass of the RMSNorm layer

        Parameters:
        - x (Tensor): The input tensor to normalize.
        - model (int): the index indicating the model being used in this layer. used for splicing self.weight

        Returns:
        - output: The normalized and scaled tensor.
        """
            
        # Normalize the input tensor using the _norm function and ensure the data type matches the input.
        x = self._norm(x.float()).type_as(x)
        
        # grabbing x's dimension to use for splicing
        dim = x.shape[-1]
        
        # calculating skip for our splice
        skip = model * dim
        
        # scale the normalized tensor by (1 + self.weight), which effectively starts with no scaling
        spliced_scale = self.weight[skip:skip + dim]
        output = x * (1 + spliced_scale)
                          
        return output

class MLP(nn.Module):
    """
    This class implements a multi-layer perceptron with a GeGLU gating mechanism. The GeGLU
    activation combines a standard GeLU activation with a learned gating mechanism, enabling
    the network to control the flow of information more dynamically.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dropout: float = 0.1,
    ):
        """
        Initializes the GemmaMLP module.

        Parameters:
            hidden_size (int): The size of the input and output tensors.
            intermediate_size (int): The size of the tensor after the initial transformation
                                     and before the gating and final projection. This is typically
                                     larger than the hidden size to allow for a richer representation.
            dropout (float): the dropout rate to use during training in forwardTuple()
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        assert intermediate_size % hidden_size == 0
        self.intermediate_multiplier = intermediate_size // hidden_size

        # Linear transformation for the gating mechanism, projecting input to an intermediate size.
        self.Wgate = nn.Parameter(torch.Tensor(hidden_size, intermediate_size))
        self.Bgate = nn.Parameter(torch.Tensor(intermediate_size))

        # Linear transformation for the input tensor, also projecting to the intermediate size but
        # intended for element-wise multiplication with the gated output.
        self.Wup = nn.Parameter(torch.Tensor(hidden_size, intermediate_size))
        self.Bup = nn.Parameter(torch.Tensor(intermediate_size))

        # Linear transformation to project the gated and combined tensor back to the original
        # hidden size, completing the MLP structure.
        self.Wdown = nn.Parameter(torch.Tensor(intermediate_size, hidden_size))
        self.Bdown = nn.Parameter(torch.Tensor(hidden_size))

        # Initialize weights with uniform distribution
        # For gate & up, where in_features is hidden_size
        limit_gateup = 1 / np.sqrt(hidden_size)
        nn.init.uniform_(self.Wgate, -limit_gateup, limit_gateup)
        nn.init.uniform_(self.Bgate, -limit_gateup, limit_gateup)
        nn.init.uniform_(self.Wup, -limit_gateup, limit_gateup)
        nn.init.uniform_(self.Bup, -limit_gateup, limit_gateup)
        
        # For down, where in_features is intermediate_size
        limit_down = 1 / np.sqrt(intermediate_size)
        nn.init.uniform_(self.Wdown, -limit_down, limit_down)
        nn.init.uniform_(self.Bdown, -limit_down, limit_down)
        
        # defining our dropout for training in forwardTuple()
        self.drop = nn.Dropout(dropout)

    def forwardTensor(self, x, model:int=0):
        """
        Defines the forward pass of the MLP module during inference.

        Parameters:
            x (Tensor): The input tensor to the MLP. 
                        shape (batch size, sequence length, hidden dimension) where hidden dimension changes by which model was used
            model (int): the indicator of which model we're using. 
                        used in calculating our skip length for splicing. 
                        defaults to the equivalent of what's used in MatFormer+, meaning no skip, aka we use the top-left-most splice

        Returns:
            Tensor: The output tensor after applying the GeGLU gating mechanism and the MLP transformations.
        """
            
        # figuring out how we should do our splicing
        d_dim = x.shape[-1]
        d_skip = model * d_dim
        i_dim = d_dim * self.intermediate_multiplier
        i_skip = model * i_dim
        
        # Applies linear transformation for gating.
        Wgate = self.Wgate[d_skip:d_skip + d_dim, i_skip:i_skip + i_dim]
        Bgate = self.Bgate[i_skip:i_skip + i_dim]
        Xgate = x @ Wgate + Bgate

        # Applies GeLU activation to the gate, introducing non-linearity and enabling the gating mechanism.
        Xgate = F.gelu(Xgate)

        # Applies another linear transformation to the input tensor for subsequent combination with the gate.
        Wup = self.Wup[d_skip:d_skip + d_dim, i_skip:i_skip + i_dim]
        Bup = self.Bup[i_skip:i_skip + i_dim]
        Xup = x @ Wup + Bup

        # Element-wise multiplication of the gated tensor with the transformed input tensor, modulating
        # the input based on the gate's activation.
        Xfuse = Xgate * Xup

        # Applies the final linear transformation to project the modulated tensor back to the hidden size.
        Wdown = self.Wdown[i_skip:i_skip + i_dim, d_skip:d_skip + d_dim]
        Bdown = self.Bdown[d_skip:d_skip + d_dim]
        outputs = Xfuse @ Wdown + Bdown

        # Returns the final output tensor of the MLP, after gating and modulation.
        return outputs

    def forwardTuple(self, x, drop_bool: bool = True):
        """
        Defines the forward pass of the MLP module during training.

        Parameters:
            x (Tuple[Tuple[Tensor]]): 
                The input tuple of tuples of tensors to the MLP. 
                first tuple is of length config.levels and second layer of tuples have lengths of config.model_count
                tensors are shape (batch size, sequence length, hidden dimension) where hidden dimension changes by which model was used

        Returns:
            Tuple[Tuple[Tensor]]: 
                The output tuple of tuples of tensors after applying the GeGLU gating mechanism and the MLP transformations.
        """

        # if we had sent through the config we could've just grabbed these values from there but too late now
        num_levels = len(x)
        models_per_level = [len(x[i]) for i in range(num_levels)]
        
        out = ()
        for i in range(num_levels):
            
            out_lvl = ()
            for j in range(models_per_level[i]):

                output = self.forwardTensor(x[i][j], model=j)
                    
                out_lvl += (self.drop(output),) if drop_bool else (output,)

            # pretty sure i have to save & store everything without overwriting to prevent in-place arguments. so annoying
            out += (out_lvl,)
        return out
        
    def forward(self, x, model=0, drop_bool = True):
        train = True if type(x) == tuple else False
        return self.forwardTuple(x, drop_bool) if train else self.forwardTensor(x, model)

class MultiQueryAttention(nn.Module):
    """
    Implements Multi-Query Attention which supports a distinct number of attention heads for queries and key-values (KV).
    In the case where the same number of queries and key-values are used, this implemenation is equivalent to regular Multi-Head Attention.  
    """
    
    def __init__(self, config: Config):
        super().__init__()

        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        
        # Determines the number of query heads associated with each KV head.
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.theta = config.rope_theta

        # Calculates the total size for all query projections.
        self.q_size = self.num_heads * self.head_dim
        # Calculates the total size for all key and value projections.
        self.kv_size = self.num_kv_heads * self.head_dim
        
        # Initialize our learnable matrices
        # the linear projection layer for queries, keys, and values
        # no real reason why we're creating one matrix instead of separate ones. cleaner model summary view?
        self.Wqkv = nn.Parameter(torch.Tensor(self.hidden_size,
                                              (self.num_heads + 2 * self.num_kv_heads) * self.head_dim))
        # the output projection layer, mapping the concatenated attention outputs back to the hidden size.
        self.Wo = nn.Parameter(torch.Tensor(self.num_heads * self.head_dim, self.hidden_size))
        
        # Initialize weights with uniform distribution
        # For qkv_proj, where in_features is hidden_size
        limit_Wqkv = 1 / np.sqrt(self.hidden_size)
        nn.init.uniform_(self.Wqkv, -limit_Wqkv, limit_Wqkv)
        # for o_proj, where in_features is self.num_heads * self.head_dim
        limit_Wo = 1 / np.sqrt(self.num_heads * self.head_dim)
        nn.init.uniform_(self.Wo, -limit_Wo, limit_Wo)
        
        # for our attention mask we'll use very large negative values to prevent attending to certain tokens
        mask_negatives = torch.full((1, 1, config.max_position_embeddings, config.max_position_embeddings),
                                 -2.3819763e38).to(torch.float)
        # then we'll replace the lower triangular ones with 0's to allow attention to see past tokens
        mask = torch.triu(mask_negatives, diagonal=1).to(config.device)
        # to define self.mask as a tensor that shouldn't undergo gradient descent
        self.register_buffer('mask', mask)
        
        # defining our dropout
        self.drop = nn.Dropout(config.dropout)

    def forwardTensor(self,
                      x: torch.Tensor,
                      model: int = 0,
                     ) -> torch.Tensor:
        """
        Inputs:
            x (torch.Tensor): Te input tensor to the attention mechanism.
                        shape (batch_size, input_len, hidden_size)
            model (int): the indicator of which model we're using. 
                        used in calculating our skip length for splicing. 
                        defaults to the equivalent of what's used in MatFormer+, meaning no skip, aka we use the top-left-most splice
        
        Returns:
            Tensor: The output tensor after applying the attention mechanism
        """
        global verbose
        if verbose: print("----------------- MultiQueryAttention.forwardTensor() --------------------")
        
        # Ensures the input tensor is 3-dimensional (batch_size, input_len, hidden_size).
        x_shape = x.shape
        assert len(x_shape) == 3

        # Extracts input sequence length and embedding dimension length from the hidden states tensor.
        batch_size, input_len, d_dim = x_shape
        
        # figuring out how we should do our splicing
        # first along the embedding dimension
        d_skip = model * d_dim  # the size of our skip along the model's embedding dimension
        
        # then for splicing along the head sizes dimension
        index = config.model_dim_list.index(d_dim)
        models_in_this_level = config.model_count[index] # how many models are in this level
        h_dim = config.head_dim_list[index] # the head dimension size of this model in this level
        h_skip = model * h_dim # the size of our skip along the head dimension

        # Splits the Wqkv tensor into separate tensors for queries, keys, and values based on their respective sizes.
        Wq, Wk, Wv = self.Wqkv.split([self.q_size,
                                      self.kv_size,
                                      self.kv_size],dim=-1)
        
        # splicing to get our correct weight matrices for each respective head
        # d_dim is relatively self-explanatory
        # i*self.head_dim is bc we initialized one single q, k, and v matrix for all heads so we have to
        # iterate through said matrix to get to the correct head
        Wq = torch.cat([Wq[d_skip:d_skip + d_dim,\
                               i*self.head_dim + h_skip:i*self.head_dim + h_skip + h_dim] \
                               for i in range(self.num_heads)], dim=1)
        Wk = torch.cat([Wk[d_skip:d_skip + d_dim,\
                               i*self.head_dim + h_skip:i*self.head_dim + h_skip + h_dim] \
                               for i in range(self.num_kv_heads)], dim=1)
        Wv = torch.cat([Wv[d_skip:d_skip + d_dim,\
                               i*self.head_dim + h_skip:i*self.head_dim + h_skip + h_dim] \
                               for i in range(self.num_kv_heads)], dim=1)
        
        # this needs to be size (d_dim, (self.num_heads + 2 * self.num_kv_heads) * h_dim) aka (32,24)
        # recombine the spliced Wq Wk and Wv. Now they're the right size for matmul against x
        Wqkv_spliced = torch.cat((Wq, Wk, Wv), dim=-1)

        # finally we can project x to get our queries, keys and values
        xqkv = x @ Wqkv_spliced
            
        # Splits the combined Xqkv tensor into separate tensors for queries (xq), keys (xk), and values (xv) based on their respective sizes.
        xq, xk, xv = xqkv.split([self.q_size // models_in_this_level,
                                 self.kv_size // models_in_this_level,
                                 self.kv_size // models_in_this_level],dim=-1)

        # Reshapes each of the Q, K, and V tensors to separate the heads and align the dimensions for attention operations.
        xq = xq.view(batch_size, input_len, self.num_heads, h_dim)#, self.head_dim)
        xk = xk.view(batch_size, input_len, self.num_kv_heads, h_dim)#, self.head_dim)
        xv = xv.view(batch_size, input_len, self.num_kv_heads, h_dim)#, self.head_dim)

        # Applies rotary positional embeddings to queries and keys to incorporate positional information.
        xq = apply_rotary_emb(xq, h_dim, self.theta)
        xk = apply_rotary_emb(xk, h_dim, self.theta)

        # If the number of KV heads is different from the number of query heads, adjusts keys and values to match the query heads count.
        if self.num_kv_heads != self.num_heads:
            # [batch_size, input_len, n_local_heads, head_dim]
            xk = torch.repeat_interleave(xk, self.num_queries_per_kv, dim=2)
            xv = torch.repeat_interleave(xv, self.num_queries_per_kv, dim=2)

        # Transposes Q, K, and V tensors to align them for the batch matrix multiplication in attention calculation.
        # [batch_size, n_local_heads, input_len, head_dim]
        q = xq.transpose(1, 2)
        k = xk.transpose(1, 2)
        v = xv.transpose(1, 2)

        # Calculates attention scores by performing a batch matrix multiplication between queries and keys, followed by scaling.
        # [batch_size, n_local_heads, input_len, input_len]
        scores = torch.matmul(q, k.transpose(2, 3)) * h_dim**-0.5
        
        # Applies the lower-triangular mask to the attention scores
        scores = scores + self.mask[...,:input_len, :input_len] # make sure mask is the correct size. input_len <= max_seq_len

        # Applies softmax to the scores to obtain attention probabilities
        scores = F.softmax(scores, dim=-1)
        
        # Computes the weighted sum of values based on the attention scores to obtain the output of the attention mechanism.
        # [batch_size, n_local_heads, input_len, head_dim]
        attention = torch.matmul(scores, v)

        # Reshapes the attention output to match the expected output dimensions, combining the heads back into the hidden dimension.
        # [batch_size, input_len, hidden_dim]
        attention = attention.transpose(1, 2).contiguous().view(batch_size, input_len, -1)

        # Splice the output projection
        Wo = torch.cat([self.Wo[i*self.head_dim + h_skip:i*self.head_dim + h_skip + h_dim,\
                                d_skip:d_skip + d_dim,\
                               ] for i in range(self.num_heads)], dim=0)
            
        # Applies the final linear projection to the attention output, mapping it back to the hidden size dimension.
        output = attention @ Wo
            
        return output

    def forwardTuple(self,
                     x: Tuple[Tuple[torch.Tensor]],
                     drop_bool: bool = True
                    ) -> torch.Tensor:
        """
        Defines the forward pass of the Attention module during training.

        Parameters:
            x (Tuple[Tuple[Tensor]]): 
                The input tuple of tuples of tensors 
                first tuple is of length config.levels and second layer of tuples have lengths of config.model_count
                tensors are shape (batch size, sequence length, hidden dimension) where hidden dimension changes by which model was used

        Returns:
            Tuple[Tuple[Tensor]]: 
                The output tuple of tuples of tensors after applying the MQA mechanism
        """
            
        # forwardTuple() should only be used during training, so we assert input_len == max_position_embeddings
        input_len = x[0][0].shape[1]
        assert input_len == config.max_position_embeddings

        # we could define these from the config but this way the method is more flexible to testing
        num_levels = len(x)
        models_per_level = [len(x[i]) for i in range(num_levels)]

        # the loop that iterates over levels, aka the different potential sizes of models
        out = ()
        for i in range(num_levels):

            # now for the loop that iterates over models in this level
            out_lvl = ()
            for j in range(models_per_level[i]):

                output = self.forwardTensor(x[i][j], model=j)
                
                out_lvl += (self.drop(output),) if drop_bool else (output,)
            
            out += (out_lvl,)

        return out
        
    def forward(self, x, model=0, drop_bool = True):
        train = True if type(x) == tuple else False
        return self.forwardTuple(x, drop_bool) if train else self.forwardTensor(x, model)

class Layer(nn.Module):
    """
    A decoder layer that integrates the MultiQueryAttention and MLP. It includes
    normalization steps both before and after the attention mechanism to stabilize and accelerate training.
    """

    def __init__(self, config: Config):
        super().__init__()

        # Initializes the GemmaAttention mechanism with parameters from the config, enabling self-attention within the decoder layer.
        self.self_attn = MultiQueryAttention(config)
        
        # Initializes the GemmaMLP module, providing a non-linear transformation after the attention mechanism.
        self.mlp = MLP(
            # the hidden dimension of the model
            hidden_size = config.hidden_size,
            # the number of nodes in the center of the two feedforward layers
            intermediate_size = config.intermediate_size,
            # the % of neurons to set to 0 during training
            dropout = config.dropout,
        )
        
        # Applies RMSNorm normalization to the input of the decoder layer for stable training dynamics.
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps = config.rms_norm_eps)
        
        # Applies RMSNorm after the attention mechanism and before the MLP to ensure the output is well-conditioned for further processing.
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps = config.rms_norm_eps)

    def forwardTensor(self,
                # The input tensor to the decoder layer. shape (batch_size, input_len, hidden_size)
                x: torch.Tensor,
                model: int = 0,
                drop_bool: bool = False
                ) -> torch.Tensor:
        
        # Self Attention Block
        # Stores the original input for use as a residual connection, aiding in mitigating the vanishing gradient problem
        residual_connection = x
        # Normalizes the input before processing by the attention mechanism.
        x = self.input_layernorm(x, model)
        # Processes the normalized input through the GemmaAttention mechanism
        x = self.self_attn(x, model, drop_bool)
        # The aforementioned residual connection
        x = residual_connection + x

        # MLP Block
        # Again, stores the output of the attention block for use as a residual connection before processing by the MLP.
        residual_connection = x
        # Normalizes the output of the attention block before passing it to the MLP, ensuring a stable input distribution.
        x = self.post_attention_layernorm(x, model)
        # Transforms the normalized attention output through the MLP, introducing additional non-linearity and capacity to the model.
        x = self.mlp(x, model, drop_bool)
        # Another residual connection
        x = residual_connection + x

        return x

    def forwardTuple(self,
                     x: Tuple[Tuple[torch.Tensor]],
                    ) -> torch.Tensor:
        """
        Defines the forward pass of a decoder layer during training.

        Parameters:
            x (Tuple[Tuple[Tensor]]): 
                The input tuple of tuples of tensors 
                first tuple is of length config.levels and second layer of tuples have lengths of config.model_count
                tensors are shape (batch size, sequence length, hidden dimension) where hidden dimension changes by which model was used

        Returns:
            Tuple[Tuple[Tensor]]: 
                The output tuple of tuples of tensors after applying the decoder layer
        """
            
        # forwardTuple() should only be used during training, so we assert input_len == max_position_embeddings
        input_len = x[0][0].shape[1]
        assert input_len == config.max_position_embeddings

        # we could define these from the config but this way the method is more flexible to testing
        num_levels = len(x)
        models_per_level = [len(x[i]) for i in range(num_levels)]

        # the loop that iterates over levels, aka the different potential sizes of models
        out = ()
        for i in range(num_levels):

            # now for the loop that iterates over models in this level
            out_lvl = ()
            for j in range(models_per_level[i]):

                output = self.forwardTensor(x[i][j], model = j, drop_bool = True)
                
                out_lvl += (output,)
            
            out += (out_lvl,)

        return out
        
    def forward(self, x, model=0):
        train = True if type(x) == tuple else False
        return self.forwardTuple(x) if train else self.forwardTensor(x, model)

class OutputLayer(nn.Module):
    def __init__(self, embedding: torch.Tensor, config: Config):
        super().__init__()
        self.embedding = embedding
        self.v = config.vocab_size
        self.model_dim_list = config.model_dim_list

        # applies RMSNorm to the embedding matrix
        self.embedding_norm = RMSNorm(config.hidden_size,
                                      eps = config.rms_norm_eps)
        
        # Applies RMSNorm to the model's final residual state before we use the embedding matrix to get logits
        self.final_norm = RMSNorm(config.hidden_size,
                                  eps = config.rms_norm_eps)

    def forwardTensor(self, x, model=0):

        # setting up our splicing logic
        d_i = x.shape[-1]
        skip = model * d_i

        # splice out our embedding matrix according to what model we're using
        sliced_embed = self.embedding[:,skip:skip + d_i]

        # normalize our sliced embedding matrix
        normed_sliced_embed = self.embedding_norm(sliced_embed)

        # normalize the residual state before the final linear layer
        x = self.final_norm(x, model)

        # calculating the final output logits of the model
        logits = x @ normed_sliced_embed.t()

        return logits

    def forwardTuple(self, x):
        """
        Defines the forward pass of the final embedding classification layer during training.

        Parameters:
            x (Tuple[Tuple[Tensor]]): 
                The input tuple of tuples of tensors 
                first tuple is of length config.levels and second layer of tuples have lengths of config.model_count
                tensors are shape (batch size, sequence length, hidden dimension) where hidden dimension changes by which model was used

        Returns:
            output (Tuple[Tuple[Tensor]]): 
                The output tuple of tuples of tensors after applying the final embedding classification
        """
            
        # forwardTuple() should only be used during training, so we assert input_len == max_position_embeddings
        assert type(x) == tuple
        input_len = x[0][0].shape[1]
        assert input_len == config.max_position_embeddings

        # we could define these from the config but this way the method is more flexible to testing
        num_levels = len(x)
        models_per_level = [len(x[i]) for i in range(num_levels)]

        # the loop that iterates over levels, aka the different potential sizes of models
        out = ()
        for i in range(num_levels):

            # now for the loop that iterates over models in this level
            out_lvl = ()
            for j in range(models_per_level[i]):

                output = self.forwardTensor(x[i][j], model = j)
                
                out_lvl += (output,)
            
            out += (out_lvl,)
        
        return out
        
    def forward(self, x, model=0):
        train = True if type(x) == tuple else False
        return self.forwardTuple(x) if train else self.forwardTensor(x, model)

class FractalLoss(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, logits, target):
        """
        input: 
            - logits are a tuple of tuples of tensors each of shape [batch_size, max_seq_len, vocab_size]
            - target is a shape [batch_size, max_seq_len] tensor of the integer indices of the correct tokens
        output: a tensor containing a single float of the loss value
        """
            
        assert type(logits) == tuple # since this function should only be used during training
            
        # should only be used during training, so we assert input_len == max_position_embeddings
        b,t,v = logits[0][0].shape
        assert t == config.max_position_embeddings
        
        # Calculate losses for each output and stack them. 
        # i apologize for the weird format instead of regular for loops, but it feels better in my head
        loss = torch.stack([ # stacks across levels
                            torch.stack( # stacks across models in level
                                        [self.criterion(logits_ij.view(b*t, v), # reshapes for CELoss
                                                        target.view(b*t)) 
                                         for logits_ij in logits[i]] # iterates across models in level
                            ).sum() # sums across models in level
                            for i in range(len(logits))] # iterates across levels
                          ).sum() # sums across levels

        return loss

class FractalFormer_base(nn.Module):
    def __init__(self, config: Config, tokenizer: tokenizer):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer

        # hyperparameters
        self.hidden_size = config.hidden_size
        self.max_seq_len = config.max_position_embeddings
        self.head_dim = config.head_dim
        self.vocab_size = config.vocab_size

        ### FractalFormer-specific hyperparameters
        self.num_levels = config.levels # the number of levels for sub-models to exist on
        self.split = config.split # the number of splits to make at a given level
        self.model_count = config.model_count # list of number of models at a given level
        self.model_dim_list = config.model_dim_list # list of hidden dimensions corresponding to each given level
        self.head_dim_list = config.head_dim_list # list of attention head dimensions corresponding to each given level    

        # the embedding matrix. for converting tokens to the first residual state, and the last residual state to logits
        self.embedder = nn.Embedding(config.vocab_size, config.hidden_size)

        # for normalizing the initial embeddings
        self.embedder_norm = RMSNorm(config.hidden_size)

        # Initialize a sequence of DecoderLayer instances as specified by the number of hidden layers in the config
        self.layers = nn.ModuleList(Layer(config) for _ in range(config.num_hidden_layers))

        # initializing output layer
        self.output_layer = OutputLayer(self.embedder.weight, config)
        # i think i need to do this bc in the above version you can't use `self.` inside the init
        #@property 
        #def output_layer(self):
            #return OutputLayer(self.embedder.weight, config)

        # the loss function
        self.criterion = FractalLoss(config)

    def forwardTensor(self,
                      input_token_ids: torch.Tensor,
                      level: int = 0, # integer designating the level of model to use. 0 is largest model, -1 is smallest
                      model: int = 0, # integer designating the model in that level to use. 0 is top-left, -1 is bottom right
                     ) -> torch.Tensor:
        """
        inputs: 
            - input_token_ids (torch.Tensor): a tensor of integers size (batch_size, sequence_length)
            - level: integer designating the level of model to use. 0 is largest model, -1 is smallest
            - model: integer designating the model in that level to use. 0 is top-left, -1 is bottom right
        output: a torch.Tensor shape (batch_size, sequence_length, vocab_size)
        """
        
        # adjusting everything to the specified level & model
        d_dim = self.hidden_size // (2**level)
        d_skip = model * d_dim
        
        # turn the input tokens into the first residual state using the embedding matrix
        # (batch_size, input_len) & (vocab_size, hidden_size) -> (batch_size, input_len, hidden_size) -> (batch_size, input_len, d_dim)
        x = self.embedder(input_token_ids)

        x = x[:,:, d_skip:d_skip + d_dim]
        
        # Gemma normalizes the embedding by sqrt(hidden_size)
        # the question is, should I do this with the full sized hidden_size or do it at the splice size????
        # imma do it at the splice size and change it later if i think the models aren't learning well
        #x = x * (d_dim**0.5)
        # alternatively i could just switch to doing a regular RMSNorm which would be more like me
        # if i figure out this different sizes of hyperspheres thing it'd be more in line with that
        x = self.embedder_norm(x, model)

        # Iteratively process the input through each Layer
        for i, layer in enumerate(self.layers):
            x = layer(x, model)

        logits = self.output_layer(x, model)

        return logits

    def forwardTuple(self,
                     input_token_ids: torch.Tensor,
                     target_token_ids: torch.Tensor,
                    ) -> torch.Tensor:
        
        # use the embedding matrix to turn the input tokens into the first residual state of the largest model
        # (batch_size, input_len) & (vocab_size, hidden_size) -> (batch_size, input_len, hidden_size)
        x0 = self.embedder(input_token_ids)

        # create the first fractal tuple of residual states
        x = ()
        for i, models_in_level in enumerate(config.model_count):
            
            x_lvl = ()
            for j, d_dim in enumerate(config.model_dim_list):

                skip = j * d_dim
                
                x_ij_spliced = x0[:,:,skip:skip + d_dim]
                    
                x_ij_spliced_normed = self.embedder_norm(x_ij_spliced, model=j) # * (d_dim**0.5) # if i want to do Gemma normalization instead
                
                x_lvl += (x_ij_spliced_normed,)  
            x += (x_lvl,)

        # Iteratively process the input through each Layer
        for i, layer in enumerate(self.layers):
            
            x = layer(x)

        logits = self.output_layer(x)

        return logits

    def forward(self,
                input_token_ids: torch.Tensor, # a shape (batch_size, input_seq_len OR max_seq_len)list of integer token ids
                target_token_ids: torch.Tensor = None, # a shape (batch_size, max_seq_len) list of token ids to train on
                level: int = 0, # integer designating the level of model to use. 0 is largest model
                model: int = 0, # integer designating the model in that level to use. 0 is top-left model in level
                ):
        
        if target_token_ids is None: # if we're not training, then we don't need to calculate loss
            logits = self.forwardTensor(input_token_ids, level, model)
            loss = None
        else:
            # if we are training
            # training uses a tuple of tuples of tensors
            logits = self.forwardTuple(input_token_ids, target_token_ids) # -> Tuple[Tuple[Tensor shape (batch_size, max_seq_len, vocab_size)]]
            
            # custom Fractal CELoss function
            loss = self.criterion(logits, target_token_ids) 
        
        return logits, loss

    @torch.no_grad() # no need to keep track of gradients during inference
    def Sampler(
        self,
        logits: torch.Tensor, # shape (batch_size, input_len, vocab_size)
        temperature: float, # controls how boring vs random the outputs should be
        top_p: float, # the maximum cumulative probability of output options we're willing to consider
        top_k: int, # the maximum number of output options we're willing to consider
    ) -> torch.Tensor:
        """
        The Sampler function is responsible for generating token predictions from Gemma's output.
        It supports temperature scaling, top-p (nucleus) sampling, and top-k sampling 
        The class operates as follows:
    
        1. Selects the last hidden state for each sequence in the batch
    
        2. Computes logits by multiplying the selected hidden states with the transposed embedding matrix. 
    
        3. Temperature is used to scale the logits, making the distribution over tokens sharper (lower temperature) 
        or flatter (higher temperature), which affects the randomness of the sampling (flatter -> more random)
    
        4. The softmax function is applied to the scaled logits to obtain a probability distribution over the vocabulary.
    
        5. For top-p sampling, the function computes the cumulative sum of the sorted probabilities and masks out tokens until the 
        cumulative probability exceeds the threshold defined by `top_ps`. This allows the model to focus on a subset of the most 
        probable tokens while ignoring the long tail of less likely tokens. 
        We to ignore long tail probabilities to avoid nonsensical output
    
        7. For top-k sampling, the function masks out all tokens except the `k` most likely ones, as specified by `top_ks`. 
        This ensures that the model only considers a fixed number of the most probable tokens for the next token prediction.
    
        8. After applying both the top-p and top-k masks, the probabilities are re-normalized so that they sum up to 1
    
        9. The function then samples from the re-normalized probability distribution to select the next token. 
        """
        if config.verbose['Sampler']:
            print("----------------- FractalFormer.Sampler() --------------")
            print(f"temperature: {temperature}, top_p: {top_p}, top_k: {top_k}")
            
        # Select the last element for each sequence.
        # (batch_size, input_len, vocab_size) -> (batch_size, vocab_size)
        logits = logits[:,-1,:]
        if config.verbose['Sampler']: print(f"logits: {logits.shape}\n{logits}")
        
        # Apply temperature scaling
        # (batch_size, vocab_size) / float -> (batch_size, vocab_size)
        logits.clone().div_(temperature) # the clone() is because i didn't properly prevent gradient tracking and i'm too lazy to fix the issue at its cause
        if config.verbose['Sampler']: print(f"logits w temperature: {logits.shape}\n{logits}")

        # Calculate probabilities with softmax.
        probs = torch.softmax(logits, dim=-1, dtype=torch.float) # dim=-1 is the vocab_size dimension that we calculate along
        if config.verbose['Sampler']: print(f"probs: {probs.shape}\n{probs}")

        # sort the probabilities to for use in top-p & top-k
        # both are (batch_size, vocab_size)
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        # probs_sort contains float probabilities while probs_idx contains integer indices
        if config.verbose['Sampler']: 
            print(f"probs_sort: {probs_sort.shape}\n{probs_sort}")
            print(f"probs_idx: {probs_idx.shape}\n{probs_idx}")

        # calculating top-p
        # creates same-size tensor of cumulatve probabilities instead of indivdiual probs
        probs_sum = torch.cumsum(probs_sort, dim=-1) 
        if config.verbose['Sampler']: print(f"probs_sum: {probs_sum.shape}\n{probs_sum}")
        # mask where 0's are top-p selections & 1's are to be excluded
        top_ps_mask = (probs_sum - probs_sort) > top_p
        if config.verbose['Sampler']: print(f"top_ps_mask: {top_ps_mask.shape}\n{top_ps_mask}")
        # the original probabilities with excluded tokens changed to 0.0
        probs_sort = torch.where(top_ps_mask, 0, probs_sort) 
        if config.verbose['Sampler']: print(f"probs_sort: {probs_sort.shape}\n{probs_sort}")

        # calculating top_k
        # create a shape (vocab_size) tensor that just iterates up by 1's
        top_ks_mask = torch.arange(probs_idx.shape[-1], device=probs_idx.device) 
        if config.verbose['Sampler']: print(f"top_ks_mask: {top_ks_mask.shape}\n{top_ks_mask}")
        # expand our mask along the batch_size dimension to become size (batch_size, vocab_size)
        # "expand" means copy the original into this new size, so each length vocab_size row is the same
        top_ks_mask = top_ks_mask.expand(probs_idx.shape[0], -1)
        if config.verbose['Sampler']: print(f"top_ks_mask: {top_ks_mask.shape}\n{top_ks_mask}")
        # top_ks is a list of integers. we keep whichever entries in top_ks_mask are greater than their corresponding entries in top_ks
        top_ks_mask = top_ks_mask >= top_k
        if config.verbose['Sampler']: print(f"top_ks_mask: {top_ks_mask.shape}\n{top_ks_mask}")

        # we'll be combining top-p with top-k and using whichever gives us fewer tokens. a very conservative approach
        # this trims probs_sort to also fit within our top_k requirement
        probs_sort = torch.where(top_ks_mask, 0, probs_sort)
        if config.verbose['Sampler']: print(f"probs_sort: {probs_sort.shape}\n{probs_sort}")

        # Re-normalization so that total probabilities add up to 1
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        if config.verbose['Sampler']: print(f"probs_sort: {probs_sort.shape}\n{probs_sort}")
        
        # now we rearrange the modified probabilities in probs_sort back to their original order according to probs_idx
        probs = torch.gather(probs_sort,
                             dim=-1,
                             index=torch.argsort(probs_idx, dim=-1))
        if config.verbose['Sampler']: print(f"probs: {probs.shape}\n{probs}")
        
        # samples from the distribution
        next_token_id = torch.multinomial(probs, num_samples=1)
        if config.verbose['Sampler']: print(f"next_token_id: {next_token_id.shape}\n{next_token_id}")
        
        return next_token_id # returns the predicted token
        
    def generate(
        self,
        prompt: str,
        output_len: int = 100, # the model will output 100 tokens
        temperature: float = 0.7, # 0.95 is pretty close to not even using temperature at all (1.0 would be no effect)
        top_p: float = 1.0, # defaulting to 1 means we essentially don't use top-p
        top_k: int = config.vocab_size, # setting top_k = vocab_size means we're effectively not using top_k at all
        level: int = 0, # which size model we want to perform inference with
        model: int = 0, # which model in that level we want to perform inference with
    ) -> str: 
        
        # encoding the prompt into token indices
        tokens = self.tokenizer.encode(prompt)

        # turning it into the right tensor shape
        tokens = torch.tensor(tokens, device=config.device).unsqueeze(0)
        
        # we wouldn't want to go past the maximum context length we trained on
        assert len(tokens) + output_len <= self.config.max_position_embeddings

        for i in range(output_len):
            # get the model's output logits and ignore the loss, which would be a NoneType object
            logits, _ = self(tokens[:,:self.max_seq_len], level=level, model=model)
            
            next_token = self.Sampler(
                logits = logits, # the actual output of the model
                temperature = temperature,
                top_p = top_p,
                top_k = top_k
            )
            #print(next_token)

            # add our new token to the sequence
            tokens = torch.cat((tokens, next_token), dim=1)

        # decode our list of tokens to an actual string
        output = self.tokenizer.decode(tokens.squeeze(0).tolist())

        return output

# data loading for training which generates a small batch of data of inputs x and targets y
def get_batch(split, batch_size):
    # whether we grab from our training or validation dataset
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - config.max_position_embeddings, (batch_size,))
    x = torch.stack([data[i:i+config.max_position_embeddings] for i in ix])
    y = torch.stack([data[i+1:i+config.max_position_embeddings+1] for i in ix])
    x, y = x.to(config.device), y.to(config.device)
    return x, y

@torch.no_grad()
def estimate_loss(model, batch_size, eval_iters = 10): # to estimate loss during the training loop
    out = {}
    model.eval() # sets model to eval mode
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, batch_size)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() # just resets to training mode
    return out