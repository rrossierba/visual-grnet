import keras
from keras import layers, ops
import tensorflow as tf

def build_encoder(
    image_shape: tuple,
    filter_sizes: list,
    latent_dim: int,
    activation: str,
    kernel_initializer: str,
    kernel_size: int,
    variational: bool = False,
    use_batch_norm: bool = True
):
    """
    Build the encoder convolutional network for an autoencoder.

    Supports standard, denoising, or variational autoencoder architectures
    by stacking convolutional blocks followed by a flattening and dense projection.

    Parameters
    ----------
    image_shape : tuple of int
        Shape of the input images formatted as (height, width, channels).
    filter_sizes : list of int
        Number of filters for each convolutional layer in the encoder sequence.
    latent_dim : int
        Dimensionality of the latent space bottleneck.
    activation : str
        Activation function to apply after each convolution or batch normalization
        layer (e.g., 'relu', 'selu').
    kernel_initializer : str
        Initializer for the convolutional kernel weights matrices (e.g., 'he_normal').
    kernel_size : int
        Width and height of the 2D convolution window.
    variational : bool, default False
        If True, builds a Variational Autoencoder (VAE) encoder that outputs both
        the mean (z_mean) and log-variance (z_log_var) tensors. If False, builds
        a deterministic encoder returning a single latent vector.
    use_batch_norm : bool, default True
        Whether to include BatchNormalization layers before the activation function
        and disable bias in the preceding convolution.

    Returns
    -------
    keras.Model
        The constructed Keras functional Model representing the encoder.
        Outputs a list of two tensors `[z_mean, z_log_var]` if `variational`
        is True, or a single `latent_representation` tensor otherwise.
    """
    image_inputs = keras.Input(shape=image_shape)
    x = image_inputs

    for idx, filters in enumerate(filter_sizes):
        sequential_block = []

        if not use_batch_norm:
            sequential_block.append(layers.Conv2D(filters, kernel_size=kernel_size, strides=2, padding='same', kernel_initializer=kernel_initializer))
        else:
            sequential_block.append(layers.Conv2D(filters, kernel_size=kernel_size, strides=2, padding='same', use_bias=False, kernel_initializer=kernel_initializer))
            sequential_block.append(layers.BatchNormalization())

        sequential_block.append(layers.Activation(activation))
        x = keras.Sequential(sequential_block, name=f'encoder_block_{idx}')(x)

    x = layers.Flatten()(x)

    if variational:
        z_mean = layers.Dense(latent_dim, name='z_mean')(x)
        z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)
        return keras.Model(image_inputs, [z_mean, z_log_var], name='encoder')
    else:
        latent_representation = layers.Dense(latent_dim, name='latent_representation')(x)
        return keras.Model(image_inputs, latent_representation, name='encoder')

def build_decoder(
    image_shape: tuple,
    filter_sizes: list,
    latent_dim: int,
    activation: str,
    kernel_initializer: str,
    kernel_size: int,
    use_batch_norm: bool = True
):
    """
    Build the decoder transposed convolutional network for an autoencoder.

    Reconstructs the original image shape from the latent space representation
    using a Dense projection followed by standard transposed convolutions.

    Parameters
    ----------
    image_shape : tuple of int
        Shape of the target output images formatted as (height, width, channels).
    filter_sizes : list of int
        Number of filters for each convolutional layer (should match the encoder
        filter sequence configuration).
    latent_dim : int
        Dimensionality of the latent space bottleneck.
    activation : str
        Activation function to apply after each convolution or batch normalization
        layer (e.g., 'relu', 'selu').
    kernel_initializer : str
        Initializer for the convolutional kernel weights matrices (e.g., 'he_normal').
    kernel_size : int
        Width and height of the 2D convolution window.
    use_batch_norm : bool, default True
        Whether to include BatchNormalization layers before the activation function
        and disable bias in the preceding dense/convolutional layers.

    Returns
    -------
    keras.Model
        The constructed Keras functional Model representing the decoder, mapping
        the latent input vector back to the reconstructed image dimensions.
    """
    latent_inputs = keras.Input(shape=(latent_dim,))
    internal_img_size = image_shape[0] // (2 ** len(filter_sizes))

    decoder_preprocessing = []
    if not use_batch_norm:
        decoder_preprocessing.append(layers.Dense(internal_img_size * internal_img_size * filter_sizes[-1], kernel_initializer=kernel_initializer))
    else:
        decoder_preprocessing.append(layers.Dense(internal_img_size * internal_img_size * filter_sizes[-1], use_bias=False, kernel_initializer=kernel_initializer))
        decoder_preprocessing.append(layers.BatchNormalization())

    decoder_preprocessing.append(layers.Activation(activation))
    decoder_preprocessing.append(layers.Reshape((internal_img_size, internal_img_size, filter_sizes[-1])))
    x = keras.Sequential(decoder_preprocessing, name='decoder_preprocessing')(latent_inputs)

    for idx, filters in enumerate(reversed(filter_sizes[:-1])):
        sequential_block = []
        if not use_batch_norm:
            sequential_block.append(layers.Conv2DTranspose(filters, kernel_size=kernel_size, strides=2, padding='same', kernel_initializer=kernel_initializer))
        else:
            sequential_block.append(layers.Conv2DTranspose(filters, kernel_size=kernel_size, strides=2, padding='same', use_bias=False, kernel_initializer=kernel_initializer))
            sequential_block.append(layers.BatchNormalization())

        sequential_block.append(layers.Activation(activation))
        x = keras.Sequential(sequential_block, name=f'decoder_block_{idx}')(x)

    decoder_outputs = layers.Conv2DTranspose(image_shape[2], kernel_size=kernel_size, strides=2, padding='same', activation='sigmoid', name='decoder_output')(x)
    return keras.Model(latent_inputs, decoder_outputs, name='decoder')

@keras.saving.register_keras_serializable()
class SparseAutoencoder(keras.Model):
    """
    Standard autoencoder with joint reconstruction and latent L1 regularization.

    This model implements a custom training and testing loop under the Keras 3
    framework. It optimizes a joint loss function consisting of the Mean Squared
    Error (MSE) for image reconstruction and an optional L1 regularization penalty
    applied to the latent space representations to enforce sparsity.

    Parameters
    ----------
    encoder : keras.Model
        The encoder subsystem mapping the input space to the latent bottleneck.
    decoder : keras.Model
        The decoder subsystem mapping the latent bottleneck back to the reconstruction space.
    l1_lambda : float, default 0.0
        The scalar multiplier for the L1 regularization penalty applied to the latent space.
    **kwargs : dict
        Additional keyword arguments passed to the base `keras.Model` constructor.

    Attributes
    ----------
    encoder : keras.Model
        The functional or sequential Keras model representing the encoder.
    decoder : keras.Model
        The functional or sequential Keras model representing the decoder.
    l1_lambda : float
        Regularization coefficient weight.
    loss_tracker : keras.metrics.Mean
        Metric tracker for the total combined loss.
    reconstruction_loss_tracker : keras.metrics.Mean
        Metric tracker for the Mean Squared Error reconstruction loss component.
    regularization_loss_tracker : keras.metrics.Mean
        Metric tracker for the L1 latent penalty component.
    """

    def __init__(self, encoder: keras.Model, decoder: keras.Model, l1_lambda: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.l1_lambda = l1_lambda

        self.loss_tracker = keras.metrics.Mean(name='loss')
        self.reconstruction_loss_tracker = keras.metrics.Mean(name='reconstruction_loss')
        self.regularization_loss_tracker = keras.metrics.Mean(name='regularization_loss')

    def call(self, inputs):
        """
        Perform the forward pass of the autoencoder.

        Parameters
        ----------
        inputs : keras.KerasTensor
            Input tensor batches matching the encoder's input shape configuration.

        Returns
        -------
        keras.KerasTensor
            The reconstructed tensor outputted by the decoder network.
        """
        encoded = self.encoder(inputs)
        return self.decoder(encoded)

    @property
    def metrics(self):
        """
        List the performance metrics monitored by the model.

        Returns
        -------
        list of keras.metrics.Metric
            The stateful metric tracking objects for loss components.
        """
        return [self.loss_tracker, self.reconstruction_loss_tracker, self.regularization_loss_tracker]

    def get_build_config(self):
        """
        Generate the configuration dictionary required to build the model's states.

        Returns
        -------
        dict
            A dictionary tracking the explicit expected input shapes based on the encoder.
        """
        return {'input_shape': self.encoder.input_shape}

    def build_from_config(self, config):
        """
        Build the model's structural state using a provided configuration schema.

        Parameters
        ----------
        config : dict
            The configuration mapping containing building parameters.
        """
        self.build(config['input_shape'])

    def get_config(self):
        """
        Serialize the object instances and properties into a configuration dictionary.

        Returns
        -------
        dict
            A dictionary containing the metadata and serialized sub-models.
        """
        config = super().get_config()
        config.update({
            'encoder': keras.saving.serialize_keras_object(self.encoder),
            'decoder': keras.saving.serialize_keras_object(self.decoder),
            'l1_lambda': self.l1_lambda
        })
        return config

    @classmethod
    def from_config(cls, config):
        """
        Instantiate the Autoencoder class from its serialized configuration state.

        Parameters
        ----------
        config : dict
            The serialized configuration schema.

        Returns
        -------
        Autoencoder
            An initialized instance of the Autoencoder class.
        """
        config['encoder'] = keras.saving.deserialize_keras_object(config.pop('encoder'))
        config['decoder'] = keras.saving.deserialize_keras_object(config.pop('decoder'))
        return cls(**config)

    def calculate_loss(self, data, latent_representation, reconstruction):
        """
        Compute the compound loss function including pixel-averaged MSE and L1 penalties.

        Parameters
        ----------
        data : tf.Tensor
            Ground truth original input image tensor.
        latent_representation : tf.Tensor
            The intermediate latent vectors produced by the encoder.
        reconstruction : tf.Tensor
            The output reconstructed image tensor generated by the decoder.

        Returns
        -------
        loss : tf.Tensor
            Scalar tensor representing the total combined optimization objective.
        reconstruction_loss : tf.Tensor
            Scalar tensor representing the spatial sample-averaged reconstruction MSE.
        regularization_term : tf.Tensor
            Scalar tensor representing the latent L1 penalty term.
        """
        reconstruction_loss = ops.mean(
            ops.mean(
                keras.losses.mean_squared_error(data, reconstruction), 
                axis=(1, 2)
            )
        )

        regularization_term = ops.mean(
            ops.sum(
                ops.abs(latent_representation), 
                axis=-1
            )
        )

        loss = reconstruction_loss + self.l1_lambda * regularization_term
        return loss, reconstruction_loss, regularization_term

    def train_step(self, data):
        """
        Execute a single gradient update step on a batch of training data.

        Parameters
        ----------
        data : tf.Tensor
            A tensor batch representing the input images.

        Returns
        -------
        dict
            A dictionary mapping metric tracking names to their current scalar values.
        """
        with tf.GradientTape() as tape:
            latent_representation = self.encoder(data, training=True)
            reconstruction = self.decoder(latent_representation, training=True)
            loss, reconstruction_loss, regularization_term = self.calculate_loss(data, latent_representation, reconstruction)

        grads = tape.gradient(loss, self.trainable_weights) # type: ignore
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.loss_tracker.update_state(loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.regularization_loss_tracker.update_state(regularization_term)
        
        return {
            'loss': self.loss_tracker.result(),
            'reconstruction_loss': self.reconstruction_loss_tracker.result(),
            'regularization_loss': self.regularization_loss_tracker.result()
        }

    def test_step(self, data):
        """
        Evaluate the model metrics over a single validation or testing batch.

        Parameters
        ----------
        data : tf.Tensor
            A tensor batch representing the input images.

        Returns
        -------
        dict
            A dictionary mapping evaluation metric names to their current scalar values.
        """
        latent_representation = self.encoder(data, training=False)
        reconstruction = self.decoder(latent_representation, training=False)

        loss, reconstruction_loss, regularization_term = self.calculate_loss(data, latent_representation, reconstruction)

        self.loss_tracker.update_state(loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.regularization_loss_tracker.update_state(regularization_term)
        
        return {
            'loss': self.loss_tracker.result(),
            'reconstruction_loss': self.reconstruction_loss_tracker.result(),
            'regularization_loss': self.regularization_loss_tracker.result()
        }


@keras.saving.register_keras_serializable()
class DenoisingAutoencoder(keras.Model):
    """
    Denoising Autoencoder (DAE) for robust feature learning.

    This model trains the network to reconstruct clean target images from inputs
    corrupted by additive Gaussian noise, forcing the encoder to extract noise-invariant
    latent representations.

    Parameters
    ----------
    encoder : keras.Model
        The encoder subsystem mapping the corrupted input space to the latent bottleneck.
    decoder : keras.Model
        The decoder subsystem mapping the latent bottleneck back to the reconstruction space.
    noise_factor : float, default 0.2
        The standard deviation scalar coefficient for the Gaussian noise distribution.
    **kwargs : dict
        Additional keyword arguments passed to the base `keras.Model` constructor.

    Attributes
    ----------
    encoder : keras.Model
        The functional or sequential Keras model representing the encoder.
    decoder : keras.Model
        The functional or sequential Keras model representing the decoder.
    noise_factor : float
        The scale factor applied to the random normal noise.
    loss_tracker : keras.metrics.Mean
        Metric tracker for the Mean Squared Error reconstruction loss.
    """

    def __init__(self, encoder: keras.Model, decoder: keras.Model, noise_factor=0.2, **kwargs):
        """
        Initialize the Denoising Autoencoder.
        """
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.noise_factor = noise_factor

        self.loss_tracker = keras.metrics.Mean(name='loss')

    def call(self, inputs):
        """
        Perform the forward pass of the DAE by corrupting the inputs with noise.

        Parameters
        ----------
        inputs : keras.KerasTensor
            Clean input tensor batches matching the encoder's expected input dimensions.

        Returns
        -------
        keras.KerasTensor
            The reconstructed tensor outputted by the decoder network.
        """
        noisy_inputs = self.add_gaussian_noise(inputs)
        encoded = self.encoder(noisy_inputs)
        return self.decoder(encoded)

    @property
    def metrics(self):
        """
        List the performance metrics monitored by the model.

        Returns
        -------
        list of keras.metrics.Metric
            The stateful metric tracking objects for loss components.
        """
        return [self.loss_tracker]

    def get_build_config(self):
        """
        Generate the configuration dictionary required to build the model's states.

        Returns
        -------
        dict
            A dictionary tracking the explicit expected input shapes based on the encoder.
        """
        return {'input_shape': self.encoder.input_shape}

    def build_from_config(self, config):
        """
        Build the model's structural state using a provided configuration schema.

        Parameters
        ----------
        config : dict
            The configuration mapping containing building parameters.
        """
        self.build(config['input_shape'])

    def get_config(self):
        """
        Serialize the object instances and properties into a configuration dictionary.

        Returns
        -------
        dict
            A dictionary containing the metadata and serialized sub-models.
        """
        config = super().get_config()
        config.update({
            'encoder': keras.saving.serialize_keras_object(self.encoder),
            'decoder': keras.saving.serialize_keras_object(self.decoder),
            'noise_factor': self.noise_factor,
        })
        return config

    @classmethod
    def from_config(cls, config):
        """
        Instantiate the DAE class from its serialized configuration state.

        Parameters
        ----------
        config : dict
            The serialized configuration schema.

        Returns
        -------
        DAE
            An initialized instance of the DAE class.
        """
        config['encoder'] = keras.saving.deserialize_keras_object(config.pop('encoder'))
        config['decoder'] = keras.saving.deserialize_keras_object(config.pop('decoder'))
        return cls(**config)

    def add_gaussian_noise(self, images):
        """
        Corrupt input images with additive zero-mean Gaussian noise.

        The resulting pixel values are clipped to enforce the valid normalized 
        intensity range [0.0, 1.0].

        Parameters
        ----------
        images : tf.Tensor
            The clean baseline input image tensor.

        Returns
        -------
        tf.Tensor
            The corrupted image tensor clipped between 0.0 and 1.0.
        """
        noisy = images + self.noise_factor * tf.random.normal(shape=tf.shape(images))
        return tf.clip_by_value(noisy, 0.0, 1.0)

    def calculate_loss(self, data, reconstruction):
        """
        Compute the spatial sample-averaged Mean Squared Error reconstruction loss.

        Parameters
        ----------
        data : tf.Tensor
            Ground truth original (clean) input image tensor.
        reconstruction : tf.Tensor
            The output reconstructed image tensor generated by the decoder.

        Returns
        -------
        tf.Tensor
            Scalar tensor representing the mean reconstruction loss across the batch.
        """
        return ops.mean(
            ops.mean(
                keras.losses.mean_squared_error(data, reconstruction), 
                axis=(1, 2)
            )
        )

    def train_step(self, data):
        """
        Execute a single gradient update step on a batch of training data.

        Parameters
        ----------
        data : tf.Tensor
            A tensor batch representing the clean input images.

        Returns
        -------
        dict
            A dictionary mapping metric tracking names to their current scalar values.
        """
        noisy_data = self.add_gaussian_noise(data)
        with tf.GradientTape() as tape:
            latent_representation = self.encoder(noisy_data, training=True)
            reconstruction = self.decoder(latent_representation, training=True)
            reconstruction_loss = self.calculate_loss(data, reconstruction)

        grads = tape.gradient(reconstruction_loss, self.trainable_weights) # type: ignore
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.loss_tracker.update_state(reconstruction_loss)
        return {'loss': self.loss_tracker.result()}

    def test_step(self, data):
        """
        Evaluate the model metrics over a single validation or testing batch.

        Parameters
        ----------
        data : tf.Tensor
            A tensor batch representing the clean input images.

        Returns
        -------
        dict
            A dictionary mapping evaluation metric names to their current scalar values.
        """
        noisy_data = self.add_gaussian_noise(data)
        latent_representation = self.encoder(noisy_data, training=False)
        reconstruction = self.decoder(latent_representation, training=False)

        reconstruction_loss = self.calculate_loss(data, reconstruction)

        self.loss_tracker.update_state(reconstruction_loss)
        return {'loss': self.loss_tracker.result()}
    
@keras.saving.register_keras_serializable()
class Sampler(keras.Layer):
    """
    Reparameterization trick layer for Variational Autoencoders.

    This layer samples a latent vector $z$ from a isotropic Gaussian distribution
    parameterized by a learned mean (`z_mean`) and log-variance (`z_log_var`),
    ensuring backpropagation compatibility via the reparameterization trick.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, z_mean, z_log_var):
        """
        Apply the reparameterization trick to sample from the latent distribution.

        Parameters
        ----------
        z_mean : tf.Tensor
            Mean tensor of the latent distribution.
        z_log_var : tf.Tensor
            Log-variance tensor of the latent distribution.

        Returns
        -------
        tf.Tensor
            The sampled latent space representation tensor $z$.
        """
        batch_size = tf.shape(z_mean)[0]
        z_size = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch_size, z_size))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
    def compute_output_shape(self, input_shape):
        """
        Compute the output shape of the layer based on the input configurations.

        Parameters
        ----------
        input_shape : tuple or list of tuples
            The shape structural schema of the incoming tensors.

        Returns
        -------
        tuple
            The expected shape configuration of the output sampled tensor.
        """
        return input_shape[0]


@keras.saving.register_keras_serializable()
class VariationalAutoencoder(keras.Model):
    """
    Variational Autoencoder (VAE) architecture with adjustable KL penalty.

    This class provides a complete Keras Model integration for a VAE. It handles
    the mapping from input space to a parameterized latent Gaussian distribution,
    stochastic sampling, and reconstruction. The objective optimization utilizes a
    reconstruction loss coupled with a Kullback-Leibler (KL) divergence penalty
    scaled by a custom coefficient factor.

    Parameters
    ----------
    encoder : keras.Model
        The encoder network outputting both `z_mean` and `z_log_var` states.
    decoder : keras.Model
        The decoder network mapping the sampled latent vectors back to the target space.
    beta : float, default 0.01
        The regularization scale factor applied to the KL divergence loss term.
    **kwargs : dict
        Additional keyword arguments passed to the base `keras.Model` constructor.

    Attributes
    ----------
    encoder : keras.Model
        The structural Keras model acting as the distribution encoder.
    sampler : Sampler
        Stochastic reparameterization sampling layer.
    decoder : keras.Model
        The structural Keras model acting as the generative decoder.
    beta : float
        Regularization multiplier coefficient for the KL divergence term.
    reconstruction_loss_tracker : keras.metrics.Mean
        Metric tracker for the batch-averaged Sum of Squared Errors (SSE).
    kl_loss_tracker : keras.metrics.Mean
        Metric tracker for the batch-averaged Kullback-Leibler divergence.
    total_loss_tracker : keras.metrics.Mean
        Metric tracker for the combined compound loss function state.
    """

    def __init__(self, encoder, decoder, beta=0.01, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.sampler = Sampler()
        self.decoder = decoder
        self.beta = beta

        self.reconstruction_loss_tracker = keras.metrics.Mean(name='reconstruction_loss')
        self.kl_loss_tracker = keras.metrics.Mean(name='kl_loss')
        self.total_loss_tracker = keras.metrics.Mean(name='total_loss')
    
    def build(self, input_shape):
        """
        Build the foundational sub-layers and functional models.

        Parameters
        ----------
        input_shape : tuple
            The definitive structural input dimensions expected by the encoder.
        """
        self.encoder.build(input_shape)
        if not self.sampler.built:
            self.sampler.build(self.encoder.output_shape)
            
        super().build(input_shape)

    @property
    def metrics(self):
        """
        List the performance metrics monitored by the model during training and evaluation.

        Returns
        -------
        list of keras.metrics.Metric
            The stateful tracking objects monitoring VAE loss components.
        """
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]
    
    def get_build_config(self):
        """
        Generate the configuration dictionary required to build the model's structural states.

        Returns
        -------
        dict
            A dictionary capturing the input shapes derived from the active encoder.
        """
        return {'input_shape': self.encoder.input_shape}

    def build_from_config(self, config):
        """
        Build the inner structural states from a given configuration mapping schema.

        Parameters
        ----------
        config : dict
            The configuration dictionary holding input dimension configurations.
        """
        self.build(config['input_shape'])

    def call(self, inputs):
        """
        Perform the full stochastic forward pass execution of the VAE.

        Parameters
        ----------
        inputs : keras.KerasTensor
            Input batches matching the internal encoder execution bounds.

        Returns
        -------
        keras.KerasTensor
            The reconstructed spatial output tensor generated via the decoder network.
        """
        z_mean, z_log_var = self.encoder(inputs)
        z = self.sampler(z_mean, z_log_var)
        return self.decoder(z)

    def get_config(self):
        """
        Serialize the class property attributes and sub-models into a configuration schema.

        Returns
        -------
        dict
            A dictionary containing serialized sub-networks and hyperparameter configurations.
        """
        config = super().get_config()
        config.update({
            'encoder': keras.saving.serialize_keras_object(self.encoder),
            'decoder': keras.saving.serialize_keras_object(self.decoder),
            'beta': self.beta,
        })
        return config

    @classmethod
    def from_config(cls, config):
        """
        Instantiate the explicit VAE class from a serialized configuration state dictionary.

        Parameters
        ----------
        config : dict
            The serialized configuration schema.

        Returns
        -------
        VAE
            An initialized instance of the VAE class.
        """
        config['encoder'] = keras.saving.deserialize_keras_object(config.pop('encoder'))
        config['decoder'] = keras.saving.deserialize_keras_object(config.pop('decoder'))
        return cls(**config)

    def calculate_loss(self, data, z_mean, z_log_var, reconstruction):    
        """
        Compute the total compound VAE loss combining SSE reconstruction and KL divergence.

        Parameters
        ----------
        data : tf.Tensor
            Ground truth original baseline training image tensor.
        z_mean : tf.Tensor
            Latent mean matrix generated by the encoder network.
        z_log_var : tf.Tensor
            Latent log-variance matrix generated by the encoder network.
        reconstruction : tf.Tensor
            Reconstructed spatial tensor produced by the decoder network.

        Returns
        -------
        total_loss : tf.Tensor
            Scalar tensor defining the joint variational optimization target objective.
        reconstruction_loss : tf.Tensor
            Scalar tensor representing the batch-averaged Sum of Squared Errors (SSE).
        kl_loss : tf.Tensor
            Scalar tensor tracking the evaluated Kullback-Leibler distribution divergence.
        """
        reconstruction_loss = ops.mean(
            ops.mean(
                keras.losses.mean_squared_error(data, reconstruction), 
                axis=(1, 2)
            )
        )

        kl_loss = ops.mean(
            ops.sum(
                -0.5 * (1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var)), 
                axis=1 
            ) 
        ) 

        total_loss = reconstruction_loss + self.beta * kl_loss
        return total_loss, reconstruction_loss, kl_loss

    def train_step(self, data):
        """
        Execute a single optimization update gradient step over a training data batch.

        Parameters
        ----------
        data : tf.Tensor
            A tensor batch representing the clean baseline input images.

        Returns
        -------
        dict
            A dictionary mapping tracking metrics names to their current scalar results.
        """
        with tf.GradientTape() as tape:
            z_mean, z_log_var = self.encoder(data, training=True)
            z = self.sampler(z_mean, z_log_var)
            reconstruction = self.decoder(z, training=True)

            total_loss, reconstruction_loss, kl_loss = self.calculate_loss(data, z_mean, z_log_var, reconstruction)
        
        grads = tape.gradient(total_loss, self.trainable_weights) # type: ignore
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            'loss': self.total_loss_tracker.result(),
            'reconstruction_loss': self.reconstruction_loss_tracker.result(),
            'kl_loss': self.kl_loss_tracker.result(),
        }
    
    def test_step(self, data):
        """
        Evaluate the monitored loss states over a validation or testing input batch.

        Parameters
        ----------
        data : tf.Tensor or tuple
            A validation batch tensor, or a data structure containing evaluation inputs.

        Returns
        -------
        dict
            A dictionary mapping metric tracking names to evaluated scalar results.
        """
        if isinstance(data, tuple):
            data = data[0]

        z_mean, z_log_var = self.encoder(data, training=False)
        z = self.sampler(z_mean, z_log_var)
        reconstruction = self.decoder(z, training=False)

        total_loss, reconstruction_loss, kl_loss = self.calculate_loss(data, z_mean, z_log_var, reconstruction)

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {
            'loss': self.total_loss_tracker.result(),
            'reconstruction_loss': self.reconstruction_loss_tracker.result(),
            'kl_loss': self.kl_loss_tracker.result(),
        }