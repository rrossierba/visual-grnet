"""
Hierarchical Attention Network (HAN) components.
Reference: Yang, Z., Yang, D., Dyer, C., He, X., Smola, A. J., & Hovy, E. H. (2016).
   Hierarchical Attention Networks for Document Classification. In Proceedings 
   of the 2016 Conference on Empirical Methods in Natural Language Processing 
   (pp. 1480-1489).

Original implementation: 
- https://github.com/mattia93/GRNet/blob/main/code/GRNet_approach.ipynb
- https://github.com/philipperemy/keras-attention-mechanism
"""

import tensorflow as tf
from keras import ops
from keras.backend import epsilon, floatx
from typing import Union
from keras import Layer
from keras import initializers, regularizers, constraints

class AttentionWeights(Layer):
    """
    Custom Keras layer to compute token or sentence-level attention weights.

    This layer implements the scoring and alignment mechanism from Hierarchical 
    Attention Networks (HAN). It learns a parameter matrix $W$ and bias vector $b$ 
    to score incoming sequence representations through a tanh non-linearity, 
    outputting a normalized probability distribution across the sequence length.

    Parameters
    ----------
    step_dim : int
        The time step or sequence dimension size (number of tokens/sentences).
    W_regularizer : Optional[keras.regularizers.Regularizer], default None
        Optional regularizer instance applied to the attention weight matrix $W$.
    b_regularizer : Optional[keras.regularizers.Regularizer], default None
        Optional regularizer instance applied to the attention bias vector $b$.
    W_constraint : Optional[keras.constraints.Constraint], default None
        Optional constraint instance applied to the attention weight matrix $W$.
    b_constraint : Optional[keras.constraints.Constraint], default None
        Optional constraint instance applied to the attention bias vector $b$.
    bias : bool, default True
        If True, includes a learnable bias vector during score calculation.
    **kwargs : dict
        Additional keyword arguments passed to the base `keras.layers.Layer`.

    Attributes
    ----------
    supports_masking : bool
        Indicates that the layer can safely propagate downstream mask tensors.
    init : keras.initializers.Initializer
        The weight matrix initializer configuration.
    W : keras.Variable
        The learnable weight vector projecting hidden states.
    b : Optional[keras.Variable]
        The learnable bias vector added to alignment scores.
    features_dim : int
        The dimensionality of the incoming feature representations (e.g., LSTM state size).
    """

    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super().__init__(**kwargs)

    def build(self, input_shape):
        """
        Create the internal trainable weights for the attention scoring mechanism.

        Parameters
        ----------
        input_shape : tuple of int
            The structural shape of the input tensor, expected to be 3D 
            as (batch_size, sequence_length, feature_dimension).
        """
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name=f'{self.name}_W',
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight(shape=(input_shape[1],),
                                     initializer='zero',
                                     name=f'{self.name}_b',
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def call(self, x, mask=None):
        """
        Compute normalized attention weights across the sequence axis.

        Parameters
        ----------
        x : tf.Tensor
            A 3D tensor sequence formatted as (batch_size, step_dim, features_dim).
        mask : Optional[tf.Tensor], default None
            A boolean mask tensor indicating valid sequence elements.

        Returns
        -------
        tf.Tensor
            A 2D probability distribution tensor formatted as (batch_size, step_dim)
            where elements along the sequence axis sum up to 1.0.
        """
        x_dtype = x.dtype
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = ops.reshape(
            ops.dot(
                ops.reshape(x, (-1, features_dim)),
                ops.reshape(self.W, (features_dim, 1))
            ),
            (-1, step_dim)
        )

        if self.bias:
            eij += self.b 

        eij = ops.tanh(eij)
        a = ops.exp(eij)

        if mask is not None:
            a *= ops.cast(mask, x_dtype)

        a /= ops.cast(ops.sum(a, axis=1, keepdims=True) + epsilon(), x_dtype)
        return a

    def compute_output_shape(self, input_shape):
        """
        Calculate the output shape coordinates generated by the forward pass logic.

        Parameters
        ----------
        input_shape : tuple of int
            The raw dimensions of the input layer states.

        Returns
        -------
        tuple of int
            The structural 2D shape configuration mapping (batch_size, step_dim).
        """
        return input_shape[0], self.step_dim

    def get_config(self):
        """
        Serialize the explicit structural layer parameters into a metadata dictionary.

        Returns
        -------
        dict
            A unified configuration dictionary containing internal layer attributes.
        """
        config = {'step_dim': self.step_dim}
        base_config = super().get_config()
        return {**base_config, **config}

class ContextVector(Layer):
    """
    Custom Keras layer to compute the weighted sum of sequence states.

    This layer maps the input sequence tensors against an engineered 
    attention weight probability distribution vector, performing a tensor dot 
    product reduction to collapse the sequence time axis into a dense context vector.

    Attributes
    ----------
    features_dim : int
        The dimensionality of the sequence representation features extracted from build.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.features_dim = 0

    def build(self, input_shape):
        """
        Initialize the structural parameters and validate incoming list shapes.

        Parameters
        ----------
        input_shape : list of tuples
            A structural pair of shape tuples: `[hidden_states_shape, attention_weights_shape]`.
        """
        assert len(input_shape) == 2
        self.features_dim = input_shape[0][-1]
        self.built = True

    def call(self, x, **kwargs):
        """
        Execute the weighted reduction aggregation on the input sequence.

        Parameters
        ----------
        x : list or tuple of tf.Tensor
            A collection containing two tensors:
            - `x[0]`: 3D hidden states tensor (batch_size, sequence_dim, features_dim).
            - `x[1]`: 2D attention probability matrix (batch_size, sequence_dim).

        Returns
        -------
        tf.Tensor
            A 2D summarized context vector tensor formatted as (batch_size, features_dim).
        """
        assert len(x) == 2
        h = x[0]
        a = x[1]
        
        a = ops.expand_dims(a, axis=-1)
        weighted_input = h * a
        return ops.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        """
        Calculate the output shape dimensions generated by the forward pass aggregation.

        Parameters
        ----------
        input_shape : list of tuples
            The layout shapes of incoming input sequences.

        Returns
        -------
        tuple of int
            A 2D tuple mapping the batch scale size to the extracted feature dimension.
        """
        return input_shape[0][0], self.features_dim

    def get_config(self):
        """
        Serialize the structural layer parameters into a metadata dictionary.

        Returns
        -------
        dict
            The base configuration state mapping dictionary.
        """
        return super().get_config()