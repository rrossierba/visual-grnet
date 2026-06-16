"""
Hyperparameter optimization framework and dataset caching pipeline via Optuna.

This module initializes cluster-worker CPU core affinity layouts, overrides 
TensorFlow operational thread topologies to guarantee process isolation, and provides 
pre-calculation caching sub-routines for dataset Z-score scaling matrices.
"""

import os
import psutil
import sys
import argparse
import gc
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--num_processes', type=int, default=1)
parser.add_argument('--worker_id', type=int, default=0)
args = parser.parse_args()

def load_config(config_path: str) -> dict:
    """
    Load and parse the JSON configuration metadata file.

    Parameters
    ----------
    config_path : str
        The string path targeting the JSON configuration schema.

    Returns
    -------
    dict
        A nested dictionary containing framework properties and dataset bounds.
    """
    with open(config_path, 'r') as file:
        return json.load(file)

config = load_config("../files/configuration/visual_grnet_tuning_configuration.json")
env_cfg = config['environment']
res_cfg = config['resources']
ds_bounds = config['dataset_bounds']
paths_cfg = config['paths']

os.environ['TF_CPP_MIN_LOG_LEVEL'] = env_cfg['tf_cpp_min_log_level']
os.environ['TF_ENABLE_ONEDNN_OPTS'] = env_cfg['tf_enable_onednn_opts']
os.environ['EXPERIMENT_SEED'] = env_cfg['experiment_seed']

n_cores = os.cpu_count()
if n_cores is None:
    n_cores = res_cfg['default_n_cores_fallback']

reserved = res_cfg['reserved_cores']
num_processes = args.num_processes
worker_id = args.worker_id

cores_per_worker = (n_cores - reserved) // num_processes
start_core = worker_id * cores_per_worker
end_core = start_core + cores_per_worker

process = psutil.Process(os.getpid())
process.cpu_affinity(list(range(start_core, end_core)))

import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(cores_per_worker)
tf.config.threading.set_inter_op_parallelism_threads(cores_per_worker)

print(f'\nworker id: {worker_id}, start core: {start_core}, end core: {end_core}', file=sys.stderr)
print(f"intra_op: {tf.config.threading.get_intra_op_parallelism_threads()}", file=sys.stderr)
print(f"inter_op: {tf.config.threading.get_inter_op_parallelism_threads()}", file=sys.stderr)

import optuna
import math
import logging
from embedding_dataset import get_mean_std
from visual_grnet import (
    build_network,
    NetworkParams,
    CallbackParams,
    get_train_validation_split_embeddings,
    RecurrentParams,
    DatasetParams,
    OptimizerParams,
    EarlyStoppingParams,
    ModelCheckpointParams,
    ReduceLROnPlateauParams,
    StaticLoggerParams,
    LRSchedulerParams,
    get_bias_initialization,
    _build_callbacks,
)
import numpy as np
from keras.losses import BinaryCrossentropy
from keras.regularizers import L1, L2, L1L2
from keras import backend as K

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
            The log record instance evaluated by the logging framework.

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

base_dir = Path(__file__).resolve().parent.parent
MIN_TRAIN_PERCENTAGE = ds_bounds['min_train_percentage']
MAX_TRAIN_PERCENTAGE = ds_bounds['max_train_percentage']
IGNORE_LAST_N_STATES = ds_bounds['ignore_last_n_states']
MAX_SEQUENCE_DIM = ds_bounds['max_sequence_dim']
NUM_CLASSES = ds_bounds['num_classes']
VERBOSE = ds_bounds['verbose']
TUNING_PERCENTAGE = ds_bounds['tuning_percentage']
MONITOR_METRIC = ds_bounds['monitor_metric']

FILES_DIR = base_dir / paths_cfg['files_sub_directory']

def precalculate_and_save(embedding_models: list[str]) -> None:
    """
    Precompute and cache empirical dataset distributions and model bias priors.

    Iterates through a collection of target embedding models, extracts global 
    feature-wise mean and standard deviation matrices from associated training splits, 
    and extracts multi-hot goal vectors to derive the optimal base frequency bias initialization.

    Parameters
    ----------
    embedding_models : list of str
        A list of sub-network string identifiers mapping the training archives 
        to be evaluated and cached.
    """    
    caches_dir = FILES_DIR / 'optuna' / 'cache'
    
    for model_name in embedding_models:
        train_path = FILES_DIR / 'bw_embeddings' / f'train_{model_name}.npz'
        norm_path = caches_dir / f'norm_{model_name}.npz'
        bias_path = caches_dir / f'bias_{model_name}.npy'

        if not train_path.exists():
            print(f"[{model_name}] train file not found, skipping")
            continue

        print(f"[{model_name}] calculating...")

        mean, std = get_mean_std(train_path)
        norm_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(norm_path, mean=mean, std=std)
        print(f"[{model_name}] norm saved to {norm_path}")

        data = np.load(train_path, allow_pickle=False)
        seq_names = data['seq_names'].tolist()
        y_true = np.stack([data[f'goal_{name}'] for name in seq_names]).astype(np.float32)

        bias = get_bias_initialization(y_true)
        bias_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(bias_path, bias)
        print(f"[{model_name}] bias saved to {bias_path}")

        print(f"[{model_name}] done")

def load_precalculated(
    model_name: str,
    normalize: bool
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray]:
    """
    Load precomputed normalization parameters and neural network bias initializers.

    Retrieves cached Z-score empirical scaling properties (mean and standard 
    deviation) alongside multi-hot frequency-derived bias initialization arrays 
    from the file system cache directory.

    Parameters
    ----------
    model_name : str
        The name identifier tracking the active embedding model architecture.
    normalize : bool
        Flag controlling whether to read and unpack the empirical Z-score tracking arrays.

    Returns
    -------
    mean : np.ndarray or None
        The precalculated feature-wise empirical mean array of shape (embedding_dim,), 
        or None if `normalize` is configured to False.
    std : np.ndarray or None
        The precalculated feature-wise empirical standard deviation array of shape 
        (embedding_dim,), or None if `normalize` is configured to False.
    bias : np.ndarray
        The parsed multi-hot structural target frequency bias vector.

    Raises
    ------
    FileNotFoundError
        If the targeted cache binaries matching `model_name` are missing from the disk space.
    """    
    caches_dir = FILES_DIR / 'optuna' / 'cache'
    bias_path = caches_dir / f'bias_{model_name}.npy'
    norm_path = caches_dir / f'norm_{model_name}.npz'
    
    if not bias_path.exists() or not norm_path.exists():
        raise FileNotFoundError(f"Precalculated files for '{model_name}' not found in {caches_dir}.")

    if normalize:
        norm = np.load(norm_path)
        mean = norm['mean']
        std = norm['std']
    else:
        mean = std = None
        
    bias = np.load(bias_path)
    return mean, std, bias

def create_model_and_dataset(trial: optuna.trial.Trial, search_cfg:dict, static_cfg:dict):
    """
    Sample hyperparameters via Optuna to assemble a compiled network and data loaders.

    This factory function extracts suggested architectural boundaries (LSTM hidden 
    units, dropout coefficients) and learning scheduler policies (learning rate bounds) 
    from a stateful optimization trial. It then maps structural arguments to construct 
    the training and validation sequence pipelines.

    Parameters
    ----------
    trial : optuna.trial.Trial
        The active tuning trial tracking hyperparameter exploration checkpoints.

    Returns
    -------
    visual_grnet : keras.models.Model
        The instantiated and parameter-initialized structural sequence neural network.
    train_dataset : EmbeddingSequence or None
        The training data provider subsampled using the active tuning percentage limit.
    validation_dataset : EmbeddingSequence or None
        The validation data provider tracking complete out-of-sample datasets.
    """
    space_rd = search_cfg['recurrent_dim']
    recurrent_dim = trial.suggest_int('recurrent_dim', space_rd['low'], space_rd['high'], step=space_rd['step'])

    space_do = search_cfg['dropout']
    dropout = trial.suggest_float('dropout', space_do['low'], space_do['high'], step=space_do['step'])

    space_rdo = search_cfg['recurrent_dropout']
    recurrent_dropout = trial.suggest_float('recurrent_dropout', space_rdo['low'], space_rdo['high'], step=space_rdo['step'])

    space_lr = search_cfg['learning_rate']
    lr_max = trial.suggest_float('learning_rate', space_lr['low'], space_lr['high'], log=space_lr['log'])

    space_lrm = search_cfg['learning_rate_min']
    lr_min = trial.suggest_float('learning_rate_min', space_lrm['low'], space_lrm['high'], log=space_lrm['log'])

    warmup_epochs = static_cfg['warmup_epochs']

    normalize = trial.suggest_categorical('normalize', search_cfg['normalize'])
    embedding_model = trial.suggest_categorical('embedding_model', search_cfg['embedding_model'])

    train_dataset_path = FILES_DIR / 'bw_embeddings' / f'train_{embedding_model}.npz'
    validation_dataset_path = FILES_DIR / 'bw_embeddings' / f'validation_{embedding_model}.npz'

    params = NetworkParams(
        model_name='visual-grnet-bw', 
        embedding_dim=256, 
        output_regularizer=None,
        loss_function=BinaryCrossentropy(),
        recurrent=RecurrentParams(
            units=recurrent_dim,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout
        ),
        dataset=DatasetParams( 
            num_classes=NUM_CLASSES,
            max_sequence_dim=math.ceil(MAX_SEQUENCE_DIM * MAX_TRAIN_PERCENTAGE),
            min_percentage=MIN_TRAIN_PERCENTAGE,
            max_percentage=MAX_TRAIN_PERCENTAGE,
            ignore_last_n_states=IGNORE_LAST_N_STATES,
        ),
        optimizer=OptimizerParams(
            learning_rate=lr_max,
            lr_scheduler=LRSchedulerParams(
                use_warmup=True,
                warmup_epochs=warmup_epochs, 
                min_lr=lr_min
            ),
            clipnorm=1.0
        )
    )

    mean, std, bias_initializer = load_precalculated(
        model_name=embedding_model,
        normalize=normalize
    )
        
    dataset_common_args = {
        'max_dim': params.dataset.max_sequence_dim,
        'min_percentage': params.dataset.min_percentage,
        'max_percentage': params.dataset.max_percentage,
        'batch_size': params.dataset.batch_size,
        'embedding_dim': params.embedding_dim,
        'num_classes': params.dataset.num_classes,
        'ignore_last_n_states': params.dataset.ignore_last_n_states,
        'norm_mean': mean,
        'norm_std': std,
    }

    train_dataset, _ = get_train_validation_split_embeddings(
        train_path=train_dataset_path, 
        validation_path=None,
        **dataset_common_args,
        limit=TUNING_PERCENTAGE
    )

    _, validation_dataset = get_train_validation_split_embeddings(
        train_path=None,
        validation_path=validation_dataset_path,
        **dataset_common_args
    )

    if params.optimizer.lr_scheduler.use_warmup and (train_dataset is not None):
        params.optimizer.lr_scheduler.steps_per_epoch = len(train_dataset) 

    visual_grnet = build_network(params, bias_initializer)
    visual_grnet.build(input_shape=(None, params.dataset.max_sequence_dim, params.embedding_dim))

    return visual_grnet, train_dataset, validation_dataset

def objective(trial: optuna.trial.Trial) -> float:
    """
    Evaluate a single hyperparameter configuration trial suggested by Optuna.

    This objective function isolates execution contexts for individual trials.
    It builds a specific model architecture and its data loaders based on trial
    suggestions, injects early stopping and automatic pruning callbacks, and monitors
    validation checkpoints. At completion, it ensures deterministic resource 
    deallocation to prevent memory accumulation.

    Parameters
    ----------
    trial : optuna.trial.Trial
        The active tuning trial tracking hyperparameter exploration checkpoints.

    Returns
    -------
    float
        The optimal validation metric value achieved during evaluation (minimum 
        validation loss or maximum validation metric depending on configuration).
    """
    model = None
    train_dataset = None
    val_dataset = None

    try:
        model, train_dataset, val_dataset = create_model_and_dataset(
            trial, 
            search_cfg=config['search_space'], 
            static_cfg=config['static_hyperparams']
        )
        callback_params = CallbackParams(
            early_stopping=EarlyStoppingParams(
                use_callback=True, 
                verbose=0, 
                monitor=MONITOR_METRIC, 
                mode='min' if 'loss' in MONITOR_METRIC else 'max',
                patience=5, 
                start_from_epoch=5
            ),
            checkpoint=ModelCheckpointParams(use_callback=False),
            reduce_lr=ReduceLROnPlateauParams(use_callback=False),
            static_logger=StaticLoggerParams(use_callback=False),
            use_weight_monitor=False
        )

        callbacks = _build_callbacks(train_dataset, params=callback_params)
        callbacks.append(optuna.integration.KerasPruningCallback(trial, MONITOR_METRIC))

        history = model.fit(
            x=train_dataset,
            validation_data=val_dataset,
            epochs=30,
            verbose=VERBOSE, 
            callbacks=callbacks
        )

        if 'loss' in MONITOR_METRIC:
            best_result = min(history.history[MONITOR_METRIC])
        else:
            best_result = max(history.history[MONITOR_METRIC])
        return best_result

    finally:
        if model is not None:
            del model
        if train_dataset is not None:
            del train_dataset
        if val_dataset is not None:
            del val_dataset
        gc.collect()
        K.clear_session()

def run_study() -> None:
    """
    Initialize and execute the parallelized Optuna hyperparameter optimization study.

    Spawns or hooks into an SQLite-backed study repository to manage cluster-wide trials. 
    It configures a Tree-structured Parzen Estimator (TPE) sampler initialized via unique
    worker-seeded sequences, configures intermediate median step pruners to terminate 
    unpromising trials early, and assigns specific execution trial budgets.
    """
    study_name = 'visual-grnet-tuning'
    db_path = FILES_DIR / f'{study_name}.db'
    storage_name = f'sqlite:///{db_path}'

    study = optuna.create_study(
        direction='minimize' if 'loss' in MONITOR_METRIC else 'maximize',
        load_if_exists=True,
        study_name=study_name,
        storage=storage_name,
        sampler=optuna.samplers.TPESampler(seed=int(os.getenv('EXPERIMENT_SEED', 42)) + worker_id),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=15,
            n_warmup_steps=10
        )
    )

    study.optimize(objective, n_trials=70 // num_processes)

if __name__ == '__main__':
    #precalculate_and_save()
    run_study()