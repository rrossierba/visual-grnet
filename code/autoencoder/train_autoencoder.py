"""
Autoencoder training execution pipeline.

This script orchestrates the end-to-end training pipeline for the convolutional
autoencoder. It configures architectural hyperparameters, handles dataset batching
and prefetching, instantiates the Keras sub-models, and executes the training
loop utilizing performance-driven callbacks.
"""

import keras
import os
import json
from autoencoder import *
from dataset_loader import *
import tensorflow as tf

def load_config(config_path: str) -> dict:
    """
    Load and parse the JSON configuration file.

    Parameters
    ----------
    config_path : str
        The path pointing to the JSON configuration metadata file.

    Returns
    -------
    dict
        A nested dictionary containing configuration states.
    """
    with open(config_path, 'r') as file:
        return json.load(file)

config = load_config('../files/configuration/autoencoder_configuration.json')

dataset_cfg = config['dataset']
training_config = config['training']
model_cfg = config['model']

keras.utils.set_random_seed(training_config['seed'])

train_ds, val_ds = load_images(
    dataset_cfg['img_directory'], 
    dataset_cfg['image_dimension'], 
    dataset_cfg['channels']
)

batch_size = training_config['batch_size']
train_ds = train_ds.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
print(f"Train dataset: {len(train_ds)} batches\nValidation dataset: {len(val_ds)} batches.")

image_shape = (dataset_cfg['image_dimension'], dataset_cfg['image_dimension'], dataset_cfg['channels'])
model_type = model_cfg['type'].lower()

is_vae = (model_type == 'vae')
encoder = build_encoder(
    image_shape=image_shape,
    filter_sizes=model_cfg['filter_sizes'],
    latent_dim=model_cfg['latent_dim'],
    activation=model_cfg['activation'],
    kernel_size=model_cfg['kernel_size'],
    kernel_initializer=model_cfg['kernel_initializer'],
    use_batch_norm=model_cfg['use_batch_norm'],
    variational=is_vae
)

decoder = build_decoder(
    image_shape=image_shape,
    filter_sizes=model_cfg['filter_sizes'],
    latent_dim=model_cfg['latent_dim'],
    activation=model_cfg['activation'],
    kernel_size=model_cfg['kernel_size'],
    kernel_initializer=model_cfg['kernel_initializer'],
    use_batch_norm=model_cfg['use_batch_norm']
)

if model_type == 'sae':
    autoencoder = SparseAutoencoder(
        encoder=encoder,
        decoder=decoder,
        l1_lambda=model_cfg.get('l1_lambda', 0.0),
        name=model_type
    )
elif model_type == 'dae':
    autoencoder = DenoisingAutoencoder(
        encoder=encoder,
        decoder=decoder,
        noise_factor=model_cfg.get('noise_factor', 0.2),
        name=model_type
    )
elif model_type == 'vae':
    autoencoder = VariationalAutoencoder(
        encoder=encoder,
        decoder=decoder,
        beta=model_cfg.get('beta', 0.01),
        name=model_type
    )
else:
    raise ValueError(f"Unsupported model type requested: {model_type}")

print(f'Model used: {model_type.upper()}')
print(autoencoder.summary())
print(encoder.summary())
print(decoder.summary())

optimizer = keras.optimizers.Adam(learning_rate=training_config['learning_rate'])
autoencoder.compile(optimizer=optimizer)
autoencoder.build(input_shape=(None,) + image_shape)

save_folder = f'{training_config['model_folder']}/{model_type}'
version = training_config['version']
save_path = f'{save_folder}/{model_type}_{version}.keras'

if not os.path.exists(save_folder):
    os.makedirs(save_folder, exist_ok=True)

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath=save_path,
        save_weights_only=False,
        monitor='val_loss',
        mode='min',
        save_best_only=True
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=10,               
        mode='min'
    )
]

history = autoencoder.fit(
    x=train_ds,
    validation_data=val_ds,
    epochs=training_config['epochs'],
    callbacks=callbacks,
)

with open(f'{save_folder}/{model_type}_v{version}_history.json', 'w') as f:
    json.dump(history.history, f)

autoencoder.save(f'{save_folder}/{model_type}_v{version}_final.keras')
print(f'Training complete. Model with best val_loss is saved at: {save_path}')