"""
Pipeline module for precomputing state embeddings from autoencoder checkpoints.

This script parses an external configuration dictionary to resolve path hierarchies,
loads pre-trained custom Keras autoencoder blocks, filters planning sequences 
across structural data splits, and streams image sets into compressed feature archives.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
os.environ['EXPERIMENT_SEED'] = '42'

import json
import random
from embedding_dataset import (
    get_mean_std,
    precompute_embeddings_split
)
from visual_grnet import (
    build_network,
    train_network,
    NetworkParams,
    CallbackParams,
    get_train_validation_split_embeddings,
    RecurrentParams,
    DatasetParams,
    OptimizerParams,
    AdapterParams,
    EarlyStoppingParams,
    ModelCheckpointParams,
    ReduceLROnPlateauParams,
    LRSchedulerParams,
    StaticLoggerParams,
    get_bias_initialization,
    WeightedBCE,
    calculate_frequencies
)
from get_predictions_and_results import run_evaluation
import numpy as np
import pandas as pd
from report import create_report, _get_embeddings_stats
from pathlib import Path
from autoencoder.autoencoder import DAE, Autoencoder, VAE
from keras.losses import BinaryFocalCrossentropy, BinaryCrossentropy, Loss
from keras.models import load_model
from dataclasses import asdict
from typing import Union

VERBOSE = 2
base_dir = Path(__file__).resolve().parent.parent

def load_config(config_path: str) -> dict:
    """
    Load and parse the JSON embedding extraction configuration file.

    Parameters
    ----------
    config_path : str
        The path pointing to the JSON configuration metadata file.

    Returns
    -------
    dict
        A nested dictionary containing path configurations and pipeline parameters.
    """
    with open(config_path, 'r') as file:
        return json.load(file)

def extract_embeddings(config: dict) -> None:
    """
    Execute configuration-driven automated image state feature embedding extraction.

    Initializes problem-goal maps, constructs complete network layouts from storage 
    checkpoints, organizes sequence directories extracted from partition arrays, 
    and pipes batch tensors into compressed file segments.

    Parameters
    ----------
    config : dict
        A nested dictionary mapping application run paths, validation bounds, 
        and structural network specifications.
    """
    paths_cfg = config['paths']
    pipeline_cfg = config['pipeline_params']
    extract_cfg = config['extraction']

    autoencoder_name = extract_cfg['autoencoder_name']
    save_str = extract_cfg['save_str']

    with open(base_dir / paths_cfg['goals_dictionary'], 'r') as f:
        dizionario_goal = json.load(f)
    with open(base_dir / paths_cfg['problem_goals'], 'r') as f:
        problem_goals = json.load(f)
    with open(base_dir / paths_cfg['problem_goals_pergen'], 'r') as f:
        problem_goals_pergen = json.load(f)
    
    path_ae = base_dir / paths_cfg['encoder_models_directory'] / f'{autoencoder_name}.keras'
    ae: Union[VAE, Autoencoder, DAE] = load_model(
        path_ae, 
        custom_objects={'VAE': VAE, 'Autoencoder': Autoencoder, 'DAE': DAE}, 
        compile=False
    ) # type: ignore
    encoder = ae.encoder 
    
    print(encoder.summary())

    all_sequence_path = base_dir / paths_cfg['blocksworld_plans_directory']
    pergen_sequence_path = base_dir / paths_cfg['pergen_blocksworld_directory']
    print(f"Base plans path: {all_sequence_path}")
    print(f"Pergen plans path: {pergen_sequence_path}")

    with open(base_dir / paths_cfg['splits_json_file'], 'r') as f:
        splits = json.load(f)

    train_sequences = [all_sequence_path / s for s in splits['train']]
    validation_sequences = [all_sequence_path / s for s in splits['val']]
    test_sequences = [all_sequence_path / s for s in splits['test']]
    test_pergen_sequences = list(pergen_sequence_path.glob('p*'))

    split_npz_paths = precompute_embeddings_split(
        encoder=encoder,
        splits={
            f'train_{save_str}': train_sequences, 
            f'validation_{save_str}': validation_sequences, 
            f'test_{save_str}': test_sequences,
            f'test_pergen_{save_str}': test_pergen_sequences
        },
        dizionario_goal=dizionario_goal,
        goals={
            f'train_{save_str}': problem_goals, 
            f'validation_{save_str}': problem_goals, 
            f'test_{save_str}': problem_goals,
            f'test_pergen_{save_str}': problem_goals_pergen
        },
        output_dir=base_dir / paths_cfg['output_embeddings_directory'],
        image_height=pipeline_cfg['image_height'],
        image_width=pipeline_cfg['image_width'],
        channels=pipeline_cfg['channels'],
        batch_size=pipeline_cfg['batch_size'],
        verbose=pipeline_cfg['verbose']
    )

    if isinstance(split_npz_paths, dict):
        for split, path in split_npz_paths.items():
            print(f"\nAnalyzing distribution stats for partition split: {split}")
            _get_embeddings_stats(path, print_results=True)

def _build_train_paths(base_dir: Path, version: int, dataset_str: str) -> dict:
    """Funzione che definisce i percorsi dei diversi elementi necessari ad una run

    Parameters
    ----------
    base_dir : Path
        Directory padre della cartella 'code' e 'files'
    version : int
        Versione dell'esperimento
    dataset_str : str
        Identificatore testuale del dataset per la run

    Returns
    -------
    dict
        Dizionario contenente i diversi percorsi per una run
    """    

    files_dir = base_dir / 'files'/'bw'
    return {
        'train':        files_dir / 'bw_embeddings' / f'train_{dataset_str}.npz',
        'validation':   files_dir / 'bw_embeddings' / f'validation_{dataset_str}.npz',
        'version_dir':  files_dir / 'experiments' / f'v{version}',
        'params':       files_dir / 'experiments' / f'v{version}' / f'visual-grnet-bw_params-v{version}.json',
        'checkpoint':   files_dir / 'experiments' / f'v{version}' / f'visual-grnet-bw-v{version}.keras',
        'final_model':  files_dir / 'experiments' / f'v{version}' / f'visual-grnet-bw-v{version}-final.keras',
        'history':      files_dir / 'experiments' / f'v{version}' / f'visual-grnet-bw_history-v{version}.json',
        'train_debug_results': files_dir / 'experiments' / f'v{version}' / f'train-debug-v{version}.pkl',
    }

def train_model(version: int, params: NetworkParams, callback_params: CallbackParams | None = None, epochs: int = 50):
    """
    Execute the end-to-end training pipeline for a sequence classification network.

    This function sets up experimental file architectures, initializes training 
    and validation embedding sequences, dynamically configures standard or 
    weighted binary cross-entropy loss based on training fluent frequencies, logs 
    hyperparameters, and executes the optimization loop.

    Parameters
    ----------
    version : int
        The unique version identifier for the current experimental trial.
    params : NetworkParams
        Dataclass instance tracking architectural choices, hyperparameters, 
        and dataset parameters.
    callback_params : Optional[CallbackParams], default None
        Dataclass tracking tracking parameter configurations for early stopping, 
        checkpointing, or custom metrics loggers.
    epochs : int, default 50
        The maximum number of recurrent training epochs to execute.

    Raises
    ------
    ValueError
        If the `dataset_version` property within `params.dataset` is not specified.
    """    
    dataset_version = params.dataset.dataset_version
    if dataset_version is None:
        raise ValueError('Dataset not specified')

    paths = _build_train_paths(base_dir=base_dir, version=version, dataset_str=dataset_version)
    train_dataset_path = paths.get('train', Path('train'))
    validation_dataset_path = paths.get('validation', Path('validation'))
    checkpoint_model_path = paths.get('checkpoint', Path('checkpoint'))
    final_model_path = paths.get('final_model', Path('final'))
    history_path = paths.get('history', Path('history'))
    params_path = paths.get('params', Path('params'))
    
    if params.loss_function == 'bce':
        params.loss_function = BinaryCrossentropy()
        print('Loss function: bce')
    elif params.loss_function == 'weighted_bce':
        pos_frequence, neg_frequence = calculate_frequencies(paths['train'])
        params.loss_function = WeightedBCE(
            pos_weights=neg_frequence,
            neg_weights=pos_frequence
        )
        print('Loss function: weighted bce')

    version_dir = paths.get('version_dir', Path('version_dir'))
    version_dir.mkdir(parents=True, exist_ok=True)

    mean, std = get_mean_std(train_dataset_path) if params.dataset.normalize else (None, None)
    train_dataset, validation_dataset = get_train_validation_split_embeddings(
        train_path=train_dataset_path,
        validation_path=validation_dataset_path,
        max_dim=params.dataset.max_sequence_dim,
        min_percentage=params.dataset.min_percentage,
        max_percentage=params.dataset.max_percentage,
        batch_size=params.dataset.batch_size,
        embedding_dim=params.embedding_dim,
        num_classes=params.dataset.num_classes,
        ignore_last_n_states=params.dataset.ignore_last_n_states,
        norm_mean=mean,
        norm_std=std
    )
    
    if train_dataset is not None:
        if params.optimizer.lr_scheduler.use_warmup:
            params.optimizer.lr_scheduler.steps_per_epoch = len(train_dataset)

        y_true = np.concatenate([
            y_batch.numpy() if not isinstance(y_batch, np.ndarray) else y_batch 
            for _, y_batch in train_dataset
        ], axis=0).astype(int)
        
        bias_initializer = get_bias_initialization(y_true)

        visual_grnet = build_network(params, bias_initializer)
        visual_grnet.build(input_shape=(None, params.dataset.max_sequence_dim, params.embedding_dim))
        print(visual_grnet.summary())

        with open(params_path, 'w') as params_file:
            dict_to_save = asdict(params)
            if not isinstance(params.loss_function, str):
                dict_to_save['loss_function'] = params.loss_function.name
                dict_to_save['loss_function_params'] = params.loss_function.get_config()
            json.dump(dict_to_save, params_file, indent=4)

        history = train_network(
            model=visual_grnet,
            train_data=train_dataset,
            validation_data=validation_dataset,
            checkpoint_path=checkpoint_model_path,
            callback_params=callback_params,
            epochs=epochs,
            verbose=VERBOSE
        )
        
        visual_grnet.save(final_model_path)
        with open(history_path, 'w') as history_file:
            json.dump(history, history_file, indent=4)

if __name__ == '__main__':
    with open('train_config.json', 'r') as f:
        config = json.load(f)

    opt_params = config['optuna_best_params']
    net_cfg = config['network']
    cb_cfg = config['callbacks']
    exp_cfg = config['experiment']
    ds_bounds = config['dataset_bounds']

    VERBOSE = exp_cfg.get('verbose', 2)
    version_number = exp_cfg['version']

    params = NetworkParams(
        model_name=net_cfg['model_name'],
        embedding_dim=net_cfg['embedding_dim'],
        use_bidirectional=net_cfg['use_bidirectional'],
        adapter=AdapterParams(**net_cfg['adapter']),
        loss_function=net_cfg['loss_function'],
        recurrent=RecurrentParams(
            units=opt_params['recurrent_dim'],
            dropout=opt_params['dropout'],
            recurrent_dropout=opt_params['recurrent_dropout']
        ),
        dataset=DatasetParams(
            num_classes=ds_bounds['num_classes'],
            max_sequence_dim=ds_bounds['max_sequence_dim'],
            min_percentage=ds_bounds['min_train_percentage'],
            max_percentage=ds_bounds['max_train_percentage'],
            ignore_last_n_states=ds_bounds['ignore_last_n_states'],
            normalize=opt_params['normalize'],
            dataset_version=opt_params['embedding_model']
        ),
        optimizer=OptimizerParams(
            learning_rate=opt_params['learning_rate'],
            lr_scheduler=LRSchedulerParams(
                use_warmup=True, 
                warmup_epochs=opt_params['warmup_epochs'], 
                min_lr=opt_params['learning_rate_min']
            ),
            clipnorm=1.0
        )
    )

    callback_params = CallbackParams(
        early_stopping=EarlyStoppingParams(
            start_from_epoch=opt_params['warmup_epochs'], 
            **cb_cfg['early_stopping']
        ),
        checkpoint=ModelCheckpointParams(**cb_cfg['checkpoint']),
        reduce_lr=ReduceLROnPlateauParams(**cb_cfg['reduce_lr']),
        static_logger=StaticLoggerParams(**cb_cfg['static_logger']),
        use_weight_monitor=cb_cfg['use_weight_monitor']
    )

    train_model(
        version=version_number, 
        params=params, 
        callback_params=callback_params, 
        epochs=exp_cfg['epochs']
    )
    
    run_evaluation(version_number)
    create_report(version_number)