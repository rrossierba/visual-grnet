"""
Report generation utilities for out-of-sample test datasets.

This module provides structural layout parsing, automated file path mapping,
and execution performance plotting components used to compile comprehensive 
Markdown and PDF analytical reports for trained models.
"""

from pathlib import Path
from time import time
from datetime import timedelta
import json
import numpy as np
import pandas as pd
import pypandoc
from typing import Union
from visual_grnet import (
    evaluate,
    get_train_validation_split_embeddings,
    get_test_dataset,
    get_trained_model,
    split_dataset_sequences,
    get_model_predictions,
)
from embedding_dataset import get_mean_std
from constants import REPORT, THRESHOLDS
from keras import Model, Layer
import matplotlib.pyplot as plt

base_dir = Path(__file__).resolve().parent.parent

def _create_params_report(params: dict, indent: int = 0) -> str:
    """
    Recursively construct a Markdown formatted string of nested configuration parameters.

    Iterates through a flat or nested parameters dictionary to output an aligned 
    Markdown list tracking hyperparameters hierarchies via tab indentations.

    Parameters
    ----------
    params : dict
        A key-value dictionary tracking configuration or model hyperparameters.
    indent : int, default 0
        The current structural nesting depth coefficient used to pre-allocate tabs.

    Returns
    -------
    str
        A multi-line formatted Markdown tracking the internal parameters configuration.
    """
    if indent == 0:
        to_return = '\n### Parameters'
    else:
        to_return = ''

    for param, value in params.items():
        if isinstance(value, dict):
            to_return += '\n\n' + '\t' * indent + f'* `{param}`:'
            to_return += f'{_create_params_report(value, indent + 1)}'
        else:
            to_return += '\n\n' + '\t' * indent + f'* `{param}`: `{value}`'
    
    return to_return

def _build_paths(base_dir: Path, version: int, final: bool) -> dict:
    """
    Map and resolve the unified absolute directory layout for a given experiment version.

    Parses the target model configuration parameters from disk to locate 
    associated datasets, model checkpoints, training history charts, and 
    destination reporting structures.

    Parameters
    ----------
    base_dir : pathlib.Path
        The absolute root path directory pointing to the active project workspace.
    version : int
        The unique integer tracking code of the specific experiment trial.
    final : bool
        Flag indicating whether to target the fully evaluated production model 
        or an intermediate epoch checkpoint.

    Returns
    -------
    dict of (str, pathlib.Path)
        A structural dictionary mapping specific data and file targets to their 
        resolved filesystem path locations.
    """
    final_str = '-final' if final else ''
    model_file = f'visual-grnet-bw-v{version}{final_str}.keras'
    model_name = model_file.split('.')[0]

    bw_dir = base_dir / 'files' / 'bw'

    paths = {
        'model_file': model_file,
        'model_name': model_name,
        'model_checkpoint':         bw_dir / 'experiments' / f'v{version}' / model_file,
        'params':                   bw_dir / 'experiments' / f'v{version}' / f'visual-grnet-bw_params-v{version}.json',
        'history':                  bw_dir / 'experiments' / f'v{version}' / f'visual-grnet-bw_history-v{version}.json',
        'predictions_folder':       bw_dir / 'experiments' / f'v{version}' / 'predictions',
        'image_sequences':          bw_dir / 'bw_sequences',
        'dizionario_goal':          bw_dir / 'dizionario_goal_bw.json',
        'problem_goals':            bw_dir / 'goals_bw.json',
        'report_dir':               bw_dir / 'experiments' / f'v{version}' / 'report',
        'results_md':               bw_dir / 'experiments' / f'v{version}' / 'report' / f'test_results_{model_name}.md',
        'results_pdf':              bw_dir / 'reports' / f'test_results_{model_name}.pdf'
    }

    with open(paths['params']) as f:
        params = json.load(f)

    dataset_version = params['dataset']['dataset_version']
    paths['train'] = bw_dir / 'bw_embeddings' / f'train_{dataset_version}.npz'
    paths['test'] = bw_dir / 'bw_embeddings' / f'test_{dataset_version}.npz'

    return paths

def _save_history(history_path: Union[Path, str], report_dir: Path) -> Path:
    """
    Generate and export multi-metric training trajectory charts to disk.

    Parses a training metrics log file, dynamically builds individual aligned 
    subplots tracking training versus validation optimization histories, 
    and outputs a high-resolution PNG artifact.

    Parameters
    ----------
    history_path : str or pathlib.Path
        The path targeting the serialized JSON history file generated during training.
    report_dir : pathlib.Path
        The destination directory path where the exported PNG plot will be generated.

    Returns
    -------
    pathlib.Path
        The specific absolute path where the plotted trajectory image was saved.
    """
    history_path = Path(history_path)
    
    with open(history_path, 'r') as history_file:
        history = json.load(history_file)

    to_plot = [h for h in history.keys() if not h.startswith('val')]

    plt.figure(figsize=(12, 12))

    for idx, metric in enumerate(to_plot, 1):
        x = list(range(1, len(history[metric]) + 1))
        plt.subplot(len(to_plot), 1, idx)
        plt.plot(x, history[metric], label=metric)
        if f'val_{metric}' in history:
            plt.plot(x, history[f'val_{metric}'], label=f'val_{metric}')
        plt.title(metric)
        plt.xlabel('Epoch')
        plt.xticks(x)
        plt.ylabel(metric)
        plt.legend()
        plt.tight_layout()
        plt.grid()

    save_path = history_path.with_suffix('.png').name

    if not (report_dir / save_path).exists():
        report_dir.mkdir(parents=True, exist_ok=True)

    final_destination = report_dir / save_path
    plt.savefig(final_destination)
    plt.close()

    return final_destination

def _get_embeddings_stats(path: Union[str, Path], print_results: bool = True, clip_bound: Union[int, float] = 3.0) -> str:
    """
    Compute and analyze structural statistical properties of stored sequence embeddings.

    Loads a compressed target dataset partition, flattens sequential configurations, 
    and extracts key distributions including global limits, dimension-wise variations, 
    and the saturation density under a Z-score clipping threshold.

    Parameters
    ----------
    path : str or pathlib.Path
        The destination target path pointing to the precomputed NPZ embeddings archive.
    print_results : bool, default True
        If True, outputs the compiled empirical summary log to standard output.
    clip_bound : int or float, default 3.0
        The symmetric threshold boundaries enforced to evaluate distribution saturation.

    Returns
    -------
    str
        A formatted Markdown string block containing the evaluated statistical results.
    """
    path = Path(path)
    data = np.load(path, allow_pickle=False)

    seq_names = data["seq_names"].tolist()
    all_embeddings = np.concatenate([data[f"embeddings_{name}"] for name in seq_names], axis=0)

    buffer = f'\n\n**Embedding analysis for dataset {path.stem}**\n'
    buffer += f'\n* all embeddings min: {all_embeddings.min():.6f}'
    buffer += f'\n* all embeddings max: {all_embeddings.max():.6f}'

    mean_per_dim = all_embeddings.mean(axis=0)
    buffer += f"\n* mean min: {mean_per_dim.min():.4f}"
    buffer += f"\n* mean max: {mean_per_dim.max():.4f}"
    buffer += f"\n* mean average: {mean_per_dim.mean():.4f}"

    std_per_dim = all_embeddings.std(axis=0)
    buffer += f"\n* std min: {std_per_dim.min():.4f}"
    buffer += f"\n* std max: {std_per_dim.max():.4f}"
    buffer += f"\n* std average: {std_per_dim.mean():.4f}"

    mean, std = get_mean_std(path)
    normalized = (all_embeddings - mean) / std
    clipped = np.clip(normalized, -clip_bound, clip_bound)

    frac_clipped = (normalized != clipped).mean()
    buffer += f"\n* clipped values ([{-clip_bound}, {clip_bound}]): {frac_clipped * 100:.2f}%"

    if print_results:
        print(buffer)

    return buffer

def _load_model_assets(paths: dict) -> tuple:
    """
    Extract and initialize validation assets, metrics parameters, and compiled networks.

    Attempts to render model optimization history curves, processes structural hyperparameter
    JSON definitions into Markdown tables, and instantiates the target trained weights matrix.

    Parameters
    ----------
    paths : dict
        A lookup dictionary tracking absolute experiment file path boundaries.

    Returns
    -------
    history_md : str
        Markdown image block link targeting the generated history chart, or a fallback notice.
    params_md : str
        Markdown tabular configuration mapping of model hyperparameters, or a fallback notice.
    params : dict
        The raw extracted dictionary containing structured experiment parameters.
    model : keras.models.Model
        The instantiated trained Keras network recovered from the checkpoint layout.
    """
    try:
        history_image_path = _save_history(paths['history'], paths['report_dir'])
        history_md = f'\n\n![History {paths["model_name"]}]({history_image_path})\n\n'
        print('History image provided')
    except Exception as e:
        history_md = 'History image not provided\n\n'
        print(f'History image not provided: {e}')

    try:
        with open(paths['params'], 'r') as f:
            params = json.load(f)
        params_md = _create_params_report(params)
        print('Params report provided')
    except FileNotFoundError:
        params = {}
        params_md = 'No params report provided\n\n'
        print('Params report not provided')

    model = get_trained_model(paths['model_checkpoint'])
    return history_md, params_md, params, model

def _build_dataset(paths: dict, common_args: dict):
    """
    Assemble an operational test dataset stream configuration mapping evaluation bounds.

    Parameters
    ----------
    paths : dict
        A lookup dictionary tracking absolute experiment file path boundaries.
    common_args : dict
        A dictionary containing dataset arguments (batch sizing, horizons, dimensions).

    Returns
    -------
    EmbeddingSequence
        An operational streaming interface instance mapping out-of-sample data points.
    """
    test_dataset = get_test_dataset(
        test_path=paths['test'],
        **common_args,
    )   
    return test_dataset

def _get_or_load_predictions(
    model, paths: dict, params: dict, test_percentage: float,
    mean: np.ndarray|None, std: np.ndarray|None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Retrieve from cache or compute raw prediction distributions and multi-threshold metrics.

    Checks the target directory for pre-computed prediction tensors matching the current 
    observation horizon percentage. If missing, it dynamically compiles the underlying 
    dataset stream, runs model inference, and caches the results to avoid redundant passes.

    Parameters
    ----------
    model : keras.models.Model
        The trained neural network used to execute the forward pass operations.
    paths : dict
        A lookup dictionary tracking absolute experiment file path boundaries.
    params : dict
        The operational hyperparameters block containing structural network profiles.
    test_percentage : float
        The fractional observation step threshold applied to slice the evaluation states.
    mean : np.ndarray
        Empirical Z-score centering shift array matching input dimension shapes.
    std : np.ndarray
        Empirical Z-score scaling variance normalization array matching input dimension shapes.

    Returns
    -------
    y_pred : np.ndarray
        A 2D array of predicted confidence distributions across categorical goal sets.
    y_true : np.ndarray
        A 2D multi-hot array tracking true target goal intersections.
    metrics_df : pd.DataFrame
        A structured pandas DataFrame summarizing model accuracy configurations across 
        pre-allocated decision thresholds.
    """
    model_name = paths['model_name']
    test_stem = paths['test'].name.split('.')[0] if paths['test'] is not None else 'ct'
    pct_label = test_percentage * 100

    preds_path = paths['predictions_folder'] / f'predictions_{model_name}_{pct_label}_{test_stem}.npz'

    if preds_path.exists():
        print(f'Loading cached predictions for {pct_label}%')
        saved = np.load(preds_path)
        y_pred = saved['y_pred']
        y_true = saved['y_true']
    else:
        print(f'Computing predictions for {pct_label}% of sequences')

        ignore_last_n = params.get('dataset', {}).get('ignore_last_n_states', 0)
        common_args = {
            'max_dim': params['dataset']['max_sequence_dim'],
            'batch_size': params['dataset']['batch_size'],
            'percentage': test_percentage,
            'embedding_dim': params['embedding_dim'],
            'num_classes': params['dataset']['num_classes'],
            'ignore_last_n_states': 0 if test_percentage == 1.0 else ignore_last_n,
            'norm_mean': mean,
            'norm_std': std,
        }

        test_dataset = _build_dataset(paths, common_args)
        y_pred, y_true = get_model_predictions(model, test_dataset=test_dataset)

        if not preds_path.exists():
            preds_path.parent.mkdir(parents=True, exist_ok=True)

        np.savez(preds_path, y_true=y_true, y_pred=y_pred, allow_pickle=False)

    thresholds = THRESHOLDS.THRESHOLDS
    metrics = {
        t: evaluate(y_pred, y_true, threshold=t, print_results=False)
        for t in thresholds
    }
    metrics_df = pd.DataFrame(metrics).rename(index=REPORT.METRICS_DICT)

    return y_pred, y_true, metrics_df

def _build_report_header(paths: dict, history_md: str, params_md: str, notes_md: str, embedding_stats_md: str|None) -> str:
    """
    Assemble the top-level Markdown layout structure and metadata for the report.

    Parameters
    ----------
    paths : dict
        A lookup dictionary tracking absolute experiment file path boundaries.
    history_md : str
        Markdown syntax string tracking the model's history charts link.
    params_md : str
        Markdown serialized block containing hyperparameters summaries.
    notes_md : str
        Markdown string containing custom notes or evaluation context descriptions.
    embedding_stats_md : str
        Markdown serialized block containing statistical analysis of the dataset.

    Returns
    -------
    str
        The compiled header string block formatted in Markdown.
    """
    model_name = paths['model_name']
    buffer = f'# Report for {model_name}\n\n'
    
    if history_md:
        buffer += history_md
    if notes_md:
        buffer += notes_md
        
    if paths.get("train") is not None and paths.get("test") is not None:
        buffer += f'\n### Used datasets\n\n* **Train partition**: `{paths["train"]}`\n* **Test partition**: `{paths["test"]}`\n\n'

    if params_md:
        buffer += params_md
    if embedding_stats_md:
        buffer += embedding_stats_md

    return buffer

def _format_percentage_section(
    test_percentage: float, y_pred: np.ndarray, y_true: np.ndarray, metrics_df: pd.DataFrame
) -> str:
    """
    Generate a structural Markdown evaluation block for a specific sequence horizon.

    Formats prediction shape coordinates, value distribution limits (min, max, mean), 
    and appends a tabular summary of classification metrics.

    Parameters
    ----------
    test_percentage : float
        The operational observation window sequence fraction (e.g., 0.1, 0.5).
    y_pred : np.ndarray
        The 2D output probability array predicted by the model forward pass.
    y_true : np.ndarray
        The 2D ground truth multi-hot targets array.
    metrics_df : pd.DataFrame
        A structured DataFrame containing performance metrics across thresholds.

    Returns
    -------
    str
        A compiled block string tracking performance criteria in Markdown layout.
    """
    buffer = [
        f'\n\n## {test_percentage * 100:.0f}% of the states',
        '\n\n### Prediction stats\n',
        f'\n* `y_pred` shape: {y_pred.shape}',
        f'\n* `y_true` shape: {y_true.shape}',
        f'\n* Prediction distribution boundaries:\n\t* `min`: {y_pred.min():.4f}\n\t* `max`: {y_pred.max():.4f}\n\t* `mean`: {y_pred.mean():.4f}',
        '\n\n**Threshold evaluation**\n\n',
        metrics_df.to_markdown(floatfmt='.4f'),
    ]
    return ''.join(buffer)

def _save_report(report: str, paths: dict) -> None:
    """
    Export the final compiled report string into Markdown and production PDF formats.

    Writes the raw Markdown artifact after local directory cleanup transformations 
    and calls `pypandoc` to invoke a system-level XeLaTeX compiler.

    Parameters
    ----------
    report : str
        The complete unrolled document data formatted in Markdown syntax.
    paths : dict
        A lookup dictionary tracking absolute destination path configurations.
    """
    paths['results_md'].parent.mkdir(parents=True, exist_ok=True)
    paths["results_pdf"].parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(paths['results_md'], 'w') as f:
            report_md = report.replace(str(paths['report_dir']), '')
            f.write(report_md)
        
        pypandoc.convert_text(
            report,
            to="pdf",
            format="md",
            outputfile=paths["results_pdf"],
            extra_args=[
                "--pdf-engine=xelatex",
                "-V", "geometry:margin=2cm",
                "-V", r"header-includes=\usepackage{float}\floatplacement{figure}{H}",
                "--variable", r"newcommand=\newcommand{\sectionbreak}{\clearpage}",
            ],
        )
        print(f"Report saved to {paths['results_pdf']}")
    except Exception as e:
        print(f"Report not saved: {e}")

def _run_percentages(model, paths: dict, params: dict, mean: np.ndarray|None, std: np.ndarray|None) -> str:
    """
    Iterate over sequential tracking horizons to generate aggregated evaluation text chunks.

    Parameters
    ----------
    model : keras.models.Model
        The trained structural network being profiled.
    paths : dict
        A lookup dictionary tracking absolute experiment file path boundaries.
    params : dict
        The global hyperparameter configuration dictionary.
    mean : np.ndarray
        Empirical Z-score centering shift array matching input dimension shapes.
    std : np.ndarray
        Empirical Z-score scaling variance normalization array matching input dimension shapes.

    Returns
    -------
    str
        A contiguous Markdown string text block collecting calculations for all horizons.
    """
    test_percentages = [0.1, 0.3, 0.5, 0.7, 1.0]
    report = ''
    for pct in test_percentages:
        y_pred, y_true, metrics_df = _get_or_load_predictions(model, paths, params, pct, mean, std)
        report += _format_percentage_section(pct, y_pred, y_true, metrics_df)
    return report

def describe_layer(layer: Layer, indent: int = 0) -> str:
    """
    Inspect and introspect a structural Keras Layer instance to extract architectural metadata.

    Parses naming conventions, tensor shapes for input/output interfaces, computes dense 
    trainable vs non-trainable parameter scaling dimensions, and isolates operational keys 
    from the internal layer configuration block.

    Parameters
    ----------
    layer : keras.layers.Layer
        The target instantiated neural network sub-component or activation block.
    indent : int, default 0
        The structural indentation scaling factor mapping layout visualization tiers.

    Returns
    -------
    str
        A compiled Markdown snippet profiling the dimensional and functional states of the layer.
    """
    prefix = '\t' * indent + '* '
    buffer = []

    name = layer.name
    class_name = layer.__class__.__name__

    inputs = layer.input
    outputs = layer.output
    dot = '\n\n' + '\t' * (indent + 1) + ' * '
    
    if isinstance(inputs, (list, tuple)) and len(inputs) > 0:
        inputs_str = dot + dot.join([str(i) for i in inputs])
    else:
        inputs_str = str(inputs)

    if isinstance(outputs, (list, tuple)) and len(outputs) > 0:
        outputs_str = dot + dot.join([str(i) for i in outputs])
    else:
        outputs_str = str(outputs)

    try:
        total_params = layer.count_params()
    except Exception:
        total_params = "N/A"

    trainable_params = sum(p.numpy().size for p in layer.trainable_weights) if layer.trainable_weights else 0
    non_trainable_params = sum(p.numpy().size for p in layer.non_trainable_weights) if layer.non_trainable_weights else 0

    try:
        config = layer.get_config()
        relevant_keys = [
            "units", "filters", "kernel_size", "strides", "padding",
            "activation", "rate", "axis", "pool_size", "use_bias",
            "return_sequences", "merge_mode", "num_heads", "key_dim",
            'dropout', 'recurrent_dropout', 'loss', 'losses'
        ]
        config_str = ", ".join(
            f"{k}={config[k]}" for k in relevant_keys if k in config
        )
    except Exception:
        config_str = ""

    buffer.append(f"\nLayer: `{name}`  [{class_name}]")
    buffer.append(f"\n{prefix}Inputs : {inputs_str}")
    buffer.append(f"\n{prefix}Outputs : {outputs_str}")
    buffer.append(f"\n{prefix}Parameters : {total_params} ({trainable_params} trainable, {non_trainable_params} non trainable)")
    if config_str:
        buffer.append(f"\n{prefix}Config : {config_str}")
    buffer.append(f"\n{prefix}Trainable : {layer.trainable}\n")

    return "".join(buffer)

def describe_model(model: Model, indent: int = 0) -> str:
    """
    Recursively generate a structural Markdown summary profile of a Keras Model.

    Traverses the network topology layer by layer. If a nested sub-model is 
    encountered, the function recurses deeper to map the internal structural 
    hierarchy, maintaining clear visual alignment via indentation padding.

    Parameters
    ----------
    model : keras.models.Model
        The instantiated Keras model or functional sub-network to inspect.
    indent : int, default 0
        The structural nesting depth coefficient governing line indentation prefixes.

    Returns
    -------
    str
        A compiled multi-line Markdown string detailing the structural layer properties.
    """
    prefix = "  " * indent
    buffer = []

    buffer.append(f"\n{prefix}**Model: {model.name}  [{model.__class__.__name__}]**\n")
    try:
        buffer.append(f"\n{prefix}Total parameters: {model.count_params():,}\n")
    except Exception:
        pass

    for layer in model.layers:
        if isinstance(layer, Model):
            buffer.append(f"{prefix} Submodel: {layer.name}")
            buffer.append(describe_model(layer, indent=indent + 1))
        else:
            buffer.append(describe_layer(layer, indent=indent))

    return "\n".join(buffer)

def create_report(
    version: int,
    final: bool = False,
    both: bool = False,
    save: bool = True,
    return_str: bool = False,
) -> Union[str, None]:
    """
    Orchestrate the automated analytical report generation pipeline.

    Loads validation history logs, runs sequential embedding statistics, parses 
    the underlying network architecture, and evaluates out-of-sample tracking 
    performance. The results are formatted as clean Markdown text chunks and 
    exported to production PDF structures.

    Parameters
    ----------
    version : int
        The unique experiment tracking version identifier matching directory targets.
    final : bool, default False
        If True, targets the fully evaluated production model layout checkpoint.
    both : bool, default False
        If True, appends a secondary full evaluation cycle isolating the production model.
    save : bool, default True
        If True, exports the compiled report text to physical file storages on disk.
    return_str : bool, default False
        If True, returns the final compiled Markdown string content.

    Returns
    -------
    str or None
        The complete generated Markdown document text if `return_str` is True, 
        otherwise None.
    """
    paths = _build_paths(Path(base_dir), version, final)
    print('Compiling the report...')
    start_time = time()

    history_md, params_md, params, model = _load_model_assets(paths)
    notes_md = '' 
    
    if paths.get('train') is not None and paths.get('test') is not None:
        print('Extracting empirical dataset partition distributions...')
        embedding_stats_md = _get_embeddings_stats(paths['train'], print_results=False)
        mean, std = get_mean_std(paths['train'])
        embedding_stats_md += _get_embeddings_stats(paths['test'], print_results=False)
    else:
        embedding_stats_md = None
        mean = std = None
    
    report = _build_report_header(paths, history_md, params_md, notes_md, embedding_stats_md)
    report += '\n\n### Model\n\n' + describe_model(model) + '\n\n'
    report += _run_percentages(model, paths, params, mean, std)

    if both:
        final_paths = _build_paths(Path(base_dir), version, final=True)
        if final_paths.get('train') is not None and final_paths.get('test') is not None:
            final_mean, final_std = get_mean_std(final_paths['train'])
        else:
            final_mean = final_std = None
        _, _, final_params, final_model = _load_model_assets(final_paths)

        report += f'\n\n# Report for {final_paths["model_name"]}\n\n'
        report += _run_percentages(final_model, final_paths, final_params, final_mean, final_std)

    if not return_str:
        elapsed = timedelta(seconds=time() - start_time)
        report += f'\n\nElapsed time: {elapsed}'
        print(f'Elapsed time: {elapsed}')

    if return_str and not save:
        return report

    if save:
        _save_report(report, paths)

    return report if return_str else None

def convert_markdown_to_pdf(version: int, final: bool = False) -> None:
    """
    Compile an existing standalone Markdown report file into a production PDF document.

    Invokes system-level XeLaTeX rendering hooks via pypandoc to apply geometric 
    margins and strict object placement constraints.

    Parameters
    ----------
    version : int
        The specific integer experimental trial code to target.
    final : bool, default False
        If True, paths are directed to parse the final validated production file layout.
    """
    paths = _build_paths(Path(base_dir), version, final)

    try:
        pypandoc.convert_file(
            paths["results_md"],
            to="pdf",
            outputfile=paths["results_pdf"],
            extra_args=[
                "--pdf-engine=xelatex",
                "-V", "geometry:margin=2cm",
                "-V", r"header-includes=\usepackage{float}\floatplacement{figure}{H}",
                "--variable", r"newcommand=\newcommand{\sectionbreak}{\clearpage}",
            ],
        )
        print(f"Report saved to {paths['results_pdf']}")
    except Exception as e:
        print(f"Report conversion aborted: {e}")
