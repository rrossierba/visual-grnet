import json
from embedding_dataset import precompute_embeddings_split
from report import _get_embeddings_stats
from pathlib import Path
from autoencoder.autoencoder import DAE, Autoencoder, VAE
from keras.models import load_model
from typing import Union

VERBOSE = 2
base_dir = Path(__file__).resolve().parent.parent

def extract_embeddings(configuration_dictionary: dict) -> None:
    """
    Execute configuration-driven automated image state feature embedding extraction.

    Initializes problem-goal maps, constructs complete network layouts from storage 
    checkpoints, organizes sequence directories extracted from partition arrays, 
    and pipes batch tensors into compressed file segments.

    Parameters
    ----------
    configuration_dictionary : dict
        A nested dictionary mapping application run paths, validation bounds, 
        and structural network specifications.
    """
    paths_cfg = configuration_dictionary.get('paths',{})
    pipeline_cfg = configuration_dictionary.get('pipeline_params', {})
    extract_cfg = configuration_dictionary.get('extraction', {})

    autoencoder_name = extract_cfg['autoencoder_name']
    save_str = extract_cfg['save_str']

    with open(base_dir / paths_cfg['goals_dictionary'], 'r') as goal_indexes_file:
        goal_indexes = json.load(goal_indexes_file)
    with open(base_dir / paths_cfg['problem_goals'], 'r') as problem_goals_file:
        problem_goals = json.load(problem_goals_file)
    with open(base_dir / paths_cfg['problem_goals_pergen'], 'r') as problem_goals_pergen_file:
        problem_goals_pergen = json.load(problem_goals_pergen_file)

    path_ae = base_dir / paths_cfg['encoder_models_directory'] / f'{autoencoder_name}.keras'
    ae: Union[VAE, Autoencoder, DAE] = load_model(
        path_ae,
        custom_objects={'VAE': VAE, 'Autoencoder': Autoencoder, 'DAE': DAE},
        compile=False
    )
    encoder = ae.encoder

    print(encoder.summary())

    all_sequence_path = base_dir / paths_cfg['blocksworld_plans_directory']
    pergen_sequence_path = base_dir / paths_cfg['pergen_blocksworld_directory']
    print(f"Base plans path: {all_sequence_path}")
    print(f"Pergen plans path: {pergen_sequence_path}")

    with open(base_dir / paths_cfg['splits_json_file'], 'r') as splits_json_file:
        splits = json.load(splits_json_file)

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
        goal_indexes=goal_indexes,
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


if __name__ == '__main__':
    with open(base_dir / 'files' / 'configuration' / 'embeddings_extraction.json', 'r') as f:
        config = json.load(f)
    extract_embeddings(config)
