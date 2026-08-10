"""
Configuration-driven hyperparameter optimization pipeline using Optuna.

This script loads parameters and search space definitions from an external JSON
configuration file to dynamically build, train, and prune autoencoder models
during optimization trials.
"""

import optuna
import os
import sys
import logging
import json
import tensorflow as tf
from keras import optimizers, callbacks
from autoencoder import *
from dataset_loader import *

optuna.logging.disable_default_handler()
optuna_logger = logging.getLogger('optuna')
optuna_logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(levelname)1.1s %(asctime)s] %(message)s')

class InfoDebugFilter(logging.Filter):
    """
    Logging filter to restrict stream output to records below WARNING level.
    """
    def filter(self, record):
        """
        Determine if the specified log record should be processed.

        Parameters
        ----------
        record : logging.LogRecord
            The stateful log record instance evaluated by the logging framework.

        Returns
        -------
        bool
            True if the log level is strictly lower than WARNING, False otherwise.
        """
        return record.levelno < logging.WARNING

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.addFilter(InfoDebugFilter())
stdout_handler.setFormatter(formatter)

stderr_handler = logging.StreamHandler(sys.stderr)
stderr_handler.setLevel(logging.WARNING)
stderr_handler.setFormatter(formatter)

optuna_logger.addHandler(stdout_handler)
optuna_logger.addHandler(stderr_handler)

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

def create_model(trial, dataset_cfg: dict, tuning_cfg: dict, search_cfg: dict):
    """
    Instantiate and compile an Autoencoder model using hyperparameter suggestions.

    Parameters
    ----------
    trial : optuna.trial.Trial
        The active hyperparameter optimization trial tracking parameter suggestions.
    dataset_cfg : dict
        Configuration sub-dictionary specifying target image attributes.
    tuning_cfg : dict
        Configuration sub-dictionary specifying batch sizing bounds.
    search_cfg : dict
        Configuration sub-dictionary defining search space distributions and bounds.

    Returns
    -------
    Autoencoder
        A compiled instance of the Autoencoder class configured with the sampled parameters.
    """
    keras.backend.clear_session()

    latent_dim_bounds = search_cfg['latent_dim_bounds']
    latent_dim_exp = trial.suggest_int('latent_dim', latent_dim_bounds[0], latent_dim_bounds[1]) 
    latent_dim = 2 ** latent_dim_exp 
    
    filter_configs = {int(k): v for k, v in search_cfg['filter_dim_configs'].items()}
    filter_dim_idx = trial.suggest_categorical('filter_dim_index', list(filter_configs.keys()))
    filter_dim = filter_configs[filter_dim_idx]

    hidden_activation = trial.suggest_categorical('activation', search_cfg['activations'])
    kernel_initializer = {
        'relu': 'he_uniform',
        'leaky_relu': 'he_uniform',
        'silu': 'he_uniform',
        'selu': 'lecun_normal'
    }.get(hidden_activation, 'glorot_uniform')
    
    kernel_size_bounds = search_cfg['kernel_size_bounds']
    kernel_size = trial.suggest_int('kernel_size', kernel_size_bounds[0], kernel_size_bounds[1])
    
    if hidden_activation == 'selu':
        use_batch_norm = False
    else:
        use_batch_norm = trial.suggest_categorical('use_batch_norm', [True, False])

    l1_lambda_bounds = search_cfg['l1_lambda_bounds']
    l1_lambda = trial.suggest_float('l1_lambda', l1_lambda_bounds[0], l1_lambda_bounds[1], log=True)
    learning_rate = search_cfg['learning_rate']

    encoder_decoder_config = {
        'image_shape': (dataset_cfg['image_dimension'], dataset_cfg['image_dimension'], dataset_cfg['channels']),
        'filter_sizes': filter_dim,
        'latent_dim': latent_dim,
        'activation': hidden_activation,
        'kernel_initializer': kernel_initializer,
        'kernel_size': kernel_size,
        'use_batch_norm': use_batch_norm
    }
    encoder = build_encoder(**encoder_decoder_config)
    decoder = build_decoder(**encoder_decoder_config)

    autoencoder = SparseAutoencoder(
        encoder=encoder,
        decoder=decoder,
        l1_lambda=l1_lambda
    )

    autoencoder.compile(optimizer=optimizers.Adam(learning_rate=learning_rate))

    print(f"Trial {trial.number}: latent_dim={latent_dim}, filter_dim={filter_dim}, activation={hidden_activation}, kernel_size={kernel_size}, learning_rate={learning_rate:.1e}, batch_size={tuning_cfg['batch_size']}")
    return autoencoder

def objective(trial, train_ds, val_ds, dataset_cfg: dict, tuning_cfg: dict, search_cfg: dict):
    """
    Objective function evaluated by the Optuna study to guide optimization.

    Parameters
    ----------
    trial : optuna.trial.Trial
        The active hyperparameter optimization trial tracking parameter suggestions.
    train_ds : tf.data.Dataset
        The batched and prefetched training dataset stream matching model dimensions.
    val_ds : tf.data.Dataset
        The batched and prefetched validation dataset stream matching model dimensions.
    dataset_cfg : dict
        Configuration sub-dictionary specifying target image attributes.
    tuning_cfg : dict
        Configuration sub-dictionary specifying execution patience and epochs bounds.
    search_cfg : dict
        Configuration sub-dictionary defining search space distributions and bounds.

    Returns
    -------
    float
        The minimum validation loss metric scored across the evaluation epochs.
    """
    model = create_model(trial, dataset_cfg, tuning_cfg, search_cfg)
    
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=tuning_cfg['patience'],
        mode='min'
    )

    pruner = optuna.integration.TFKerasPruningCallback(trial, 'val_loss')
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=tuning_cfg['epochs'],
        callbacks=[early_stopping, pruner],
        verbose=2 
    )

    best_val_loss = min(history.history['val_loss'])
    return best_val_loss

if __name__ == '__main__':
    config = load_config('../files/configuration/autoencoder_tune_config.json')
    dataset_cfg = config['dataset']
    tuning_cfg = config['tuning']
    search_cfg = config['search_space']

    study_name = tuning_cfg['study_name']
    actual_folder = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(actual_folder, f'{study_name}.db')
    storage_name = f'sqlite:///{db_path}'
    
    study = optuna.create_study(
        direction='minimize',
        study_name=study_name,
        storage=storage_name,
        sampler=optuna.samplers.TPESampler(seed=tuning_cfg['seed']),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=tuning_cfg['n_startup_trials'],
            n_warmup_steps=tuning_cfg['n_warmup_steps']
        )
    )

    train_base_ds, val_base_ds = load_images(
        dataset_cfg['img_directory'], 
        dataset_cfg['image_dimension'], 
        dataset_cfg['channels']
    )
    
    batch_size = tuning_cfg['batch_size']
    train_ds = train_base_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = val_base_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    print(len(train_ds))
    print(len(val_ds))
    
    study.optimize(
        lambda trial: objective(trial, train_ds, val_ds, dataset_cfg, tuning_cfg, search_cfg), 
        n_trials=tuning_cfg['n_trials']
    )