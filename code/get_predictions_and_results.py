import json
import time
import os
import numpy as np
import pandas as pd
from pathlib import Path
from embedding_dataset import EmbeddingSequence, get_mean_std
from keras.models import load_model
from attention_extraction_layers import AttentionWeights, ContextVector
from typing import Union

SEED = int(os.getenv('EXPERIMENT_SEED', 43))

def get_score(prediction: np.ndarray, possible_goal: list) -> float:
    """
    Calculate the cumulative validation score for a specific goal configuration.

    Sums the model's output probabilities or activation values corresponding to 
    the indices of the fluents present within the targeted candidate goal.

    Parameters
    ----------
    prediction : np.ndarray
        A 1D array representing the raw distribution predicted by the model.
    possible_goal : list of int
        A collection of indices tracking the specific fluents that characterize 
        the candidate goal condition.

    Returns
    -------
    float
        The calculated cumulative scalar score for the given goal configuration.
    """
    return float(np.sum(prediction[possible_goal]))

def get_max(scores: np.ndarray) -> list:
    """
    Retrieve all positional array indices that achieve the maximum recorded score.

    Identifies the absolute maximum score value within the input array and 
    extracts all corresponding indices, supporting scenarios with multiple 
    equally optimal candidate solutions.

    Parameters
    ----------
    scores : np.ndarray
        A 1D array of floats containing computed recognition scores for each candidate.

    Returns
    -------
    list of int
        A list containing the positional indices of the maximum recorded values.
    """
    max_val = np.max(scores)
    return np.argwhere(scores == max_val).flatten().tolist()

def get_scores(prediction: np.ndarray, possible_goals: dict) -> np.ndarray | None:
    """
    Compute cumulative recognition scores for all candidate goal configurations.

    Maps over a structured dictionary of possible goals, evaluating the scalar 
    score of each item via its constituent active fluents indices.

    Parameters
    ----------
    prediction : np.ndarray
        A 1D array tracking the raw output distribution predicted by the model.
    possible_goals : dict of (int, list of int)
        A dictionary mapping candidate goal identifiers (keys) to their respective 
        lists of constituent active fluent indices (values).

    Returns
    -------
    np.ndarray or None
        A 1D float array tracking evaluated scores indexed by candidate IDs. 
        Returns None if an IndexError is encountered due to dimensional mismatch.
    """
    try:
        scores = np.zeros((max(possible_goals) + 1), dtype=float)
        for index, goal_fluents in possible_goals.items():
            scores[index] = get_score(prediction, goal_fluents)
        return scores
    except IndexError as e:
        print('Index error:', e)
        return None

def get_result(scores: np.ndarray, correct_goal: int) -> bool:
    """
    Evaluate the success of the goal recognition task against the ground truth.

    Identifies the maximum score index. If a tie occurs between multiple optimal 
    candidates, a single candidate index is selected uniformly at random. The chosen 
    index is then validated against the ground truth goal.

    Parameters
    ----------
    scores : np.ndarray
        A 1D float array tracking calculated verification scores for each candidate goal.
    correct_goal : int
        The ground truth integer index representing the correct target goal configuration.

    Returns
    -------
    bool
        True if the maximum scoring index (or the randomly selected tie-breaker) 
        matches the correct target goal, False otherwise.
    """
    idx_max_list = get_max(scores)
    
    if len(idx_max_list) == 1:
        idx_max = idx_max_list[0]
    else:
        print(f"Algorithm chose randomly one of {len(idx_max_list)} equal candidates.")
        idx_max = idx_max_list[np.random.randint(0, len(idx_max_list))]
    
    return int(idx_max) == int(correct_goal)

def get_results_json(results, percentage, time, goals, problem_name):
    """
    Format evaluation performance evaluation metrics and targets into a JSON-compatible schema.

    Processes prediction outcomes, formatted floating-point array scores, and multi-hot
    fluent goals to compile an experimental tracking summary dictionary.

    Parameters
    ----------
    results : tuple or list
        A structured containing evaluation states where `results[0]` maps the predicted
        goal index, `results[1]` maps the actual baseline target index, and `results[2]` 
        holds the raw numerical score arrays.
    percentage : float
        The sequential frame observation threshold used during the current evaluation trial.
    time : float
        The total computational processing execution runtime elapsed during testing steps.
    goals : Optional[dict of (int, list)]
        A dictionary mapping validation goal indexes to their specific active fluent collections.
    problem_name : str
        The alpha-numeric operational identifier tracking the specific validation instance.

    Returns
    -------
    dict
        A standardized serialized dictionary structure ready for JSON parsing exports.
    """
    scores = results[2]
    
    if scores is not None:
        scores_dict = {i: f"{s:.5f}" for i, s in enumerate(scores)}
    else:
        scores_dict = None

    if goals is not None:
        goals_dict = {k: " ".join(str(e) for e in goals[k]) for k in goals}
    else:
        goals_dict = None

    json_dict = {
        "INSTANCE": problem_name,
        "PERCENTAGE": percentage,
        "PREDICTED": results[0],
        "ACTUAL": results[1],
        "SCORES": scores_dict,
        "TOTALRUNTIME": time,
        "GOALS": goals_dict,
    }
    return json_dict

def get_goal(seq_name: str, all_goals: dict, goals_dict: dict) -> tuple | None:
    """
    Extract and resolve the ordered fluent indices composing a sequence's target goal.

    Parses the target baseline problem ID from the sequence string layout, retrieves 
    its categorical ground truth string subgoals, and maps them to their corresponding 
    integer matrix locations.

    Parameters
    ----------
    seq_name : str
        The alpha-numeric naming string tracking the target execution path (e.g., "p001_seq2").
    all_goals : dict of (str, list of str)
        The global dictionary mapping base problem IDs to their respective active 
        lowercase goal fluent configurations.
    goals_dict : dict of (str, int)
        The mapping dictionary translating uppercase fluent identifiers to discrete 
        feature array index positions.

    Returns
    -------
    tuple of int or None
        A sorted tuple tracking the target fluent indices that uniquely represent 
        the goal condition. Returns None if the problem ID is not found.
    """    
    problem_name = seq_name.split('-')[0].split('_')[0]
    subgoals = all_goals.get(problem_name, None)
    if subgoals is None:
        print(f"problem {problem_name} does not have a defined goal")
        return None

    new_goal = [goals_dict[g.upper()] for g in subgoals]
    new_goal = np.sort(new_goal)
    return tuple(new_goal)

def get_test_sequences(
    test_npz_file: Union[Path, str], 
    problem_names: list, 
    percentage: float,
    embedding_dim: int,
    max_sequence_len: int,
    num_classes: int, 
    mean: Union[np.ndarray, None] = None, 
    std: Union[np.ndarray, None] = None, 
    batch_size: int = 1
) -> EmbeddingSequence:
    """
    Filter and compile an isolated EmbeddingSequence dataset for targeted validation.

    Loads a compressed master archive, samples sequence configurations matching the requested 
    problems, selects the optimal path when duplicates emerge based on minimum path length constraints, 
    and returns a structured Keras sequence data streaming provider.

    Parameters
    ----------
    test_npz_file : str or pathlib.Path
        The destination file path pointing to the archive holding all precomputed test state arrays.
    problem_names : list of dict
        A collection of dictionary elements identifying the targeted subset instances. Each 
        element is structured as `{'name': sequence_prefix_string}`.
    percentage : float
        The frame extraction ratio applied to sequence lengths during testing phases.
    embedding_dim : int
        The functional dimensional features sizing constraint characterizing target vectors.
    max_sequence_len : int
        The fixed spatial horizontal layout boundary managing padding truncations.
    num_classes : int
        The operational feature size representing the categorical multi-hot goals target space.
    mean : Optional[np.ndarray], default None
        Empirical Z-score scaling matrix used to center the input distributions.
    std : Optional[np.ndarray], default None
        Empirical Z-score scaling matrix used to stabilize input variances.
    batch_size : int, default 1
        The structural chunking scale tracking the packed arrays generated per step.

    Returns
    -------
    EmbeddingSequence
        An operational instance of the custom `EmbeddingSequence` pipeline ready for inference loops.
    """    
    all_test_embeddings = np.load(test_npz_file)
    all_test_seq_names = all_test_embeddings.get('seq_names')
    
    random_shuffler = np.random.default_rng(seed=SEED)
    random_shuffler.shuffle(all_test_seq_names)

    test_sequences_names = []
    test_sequences = {}
    avg_len = []

    for problem in problem_names:
        problem_name = problem['name']
        mask = np.char.startswith(all_test_seq_names, problem_name)
        sequences = all_test_seq_names[mask]

        if len(sequences) > 1:
            lengths = [all_test_embeddings[f'embeddings_{s}'].shape[0] for s in sequences]
            min_index = np.argmin(lengths)
            avg_len.append(np.min(lengths))
            sequence = sequences[min_index]
        else:
            sequence = sequences[0]

        test_sequences_names.append(sequence)
        test_sequences[f'embeddings_{sequence}'] = all_test_embeddings[f'embeddings_{sequence}']
        test_sequences[f'goal_{sequence}'] = all_test_embeddings[f'goal_{sequence}']
        
    test_sequences['seq_names'] = np.array(test_sequences_names, dtype=str)
    return EmbeddingSequence(
        npz=test_sequences,
        max_dim=max_sequence_len,
        batch_size=batch_size,
        embedding_dim=embedding_dim,
        num_classes=num_classes,
        perc=percentage,
        norm_mean=mean,
        norm_std=std
    )

def compute_results_plans_category(
    model,
    problems: list,
    possible_goals: dict,
    dizionario_goal: dict,
    train_npz_file: Union[Path, str],
    test_npz_file: Union[Path, str],
    results_dir: Union[Path, str],
    embedding_dim: int,
    max_sequence_len: int,
    num_classes: int,
    norm: bool = False
) -> list:
    """
    Evaluate model recognition performance across multiple state observation horizons.

    This function chunks sequence state data into fixed incremental steps (10%, 
    30%, 50%, 70%, 100%), invokes batched model inference to compute structural 
    predictions, parses candidate goal scores, and tracks aggregated execution metrics.

    Parameters
    ----------
    model : keras.models.Model
        The compiled sequence classification model or network architecture.
    problems : list of dict
        A collection of dictionaries tracking problem identifiers, structured 
        as `[{'name': problem_name, 'goal_number': ground_truth_idx}]`.
    possible_goals : dict of (str, list of str)
        A dictionary mapping categorical goal numbers to lists of their 
        constituent literal string fluents.
    dizionario_goal : dict of (str, int)
        A lookup mapping translating uppercase string fluents to discrete index integers.
    train_npz_file : str or pathlib.Path
        The filepath pointing to the training archive used to derive normalization statistics.
    test_npz_file : str or pathlib.Path
        The filepath pointing to the testing archive holding evaluation sequence arrays.
    results_dir : str or pathlib.Path
        The target directory path where tracking records and data outputs are generated.
    embedding_dim : int
        The functional structural dimension size characterizing target vectors.
    max_sequence_len : int
        The fixed spatial horizontal layout boundary managing padding truncations.
    num_classes : int
        The operational feature size representing the categorical multi-hot goals target space.
    norm : bool, default False
        If True, applies empirical Z-score standardization to the input data streams.

    Returns
    -------
    list of dict
        A compiled list containing serialized JSON-compatible result metadata mappings.
    """    
    if isinstance(results_dir, (str, Path)):
        results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    mean, std = get_mean_std(train_npz_file) if norm else (None, None)
    
    possible_goals_indexes = {
        int(goal_number): [dizionario_goal.get(f.upper()) for f in fluents]
        for goal_number, fluents in possible_goals.items()
    }
    
    results = [] 
    for perc_action in [0.1, 0.3, 0.5, 0.7, 1.0]:
        print(f"Working with {perc_action * 100}% of states")
        test_sequence = get_test_sequences(
            test_npz_file=test_npz_file,
            problem_names=problems,
            percentage=perc_action,
            mean=mean,
            std=std,
            batch_size=64,
            embedding_dim=embedding_dim,
            max_sequence_len=max_sequence_len,
            num_classes=num_classes,
        )

        start_time = time.time()
        predictions = model.predict(test_sequence, verbose=0) 
        elapsed_time = time.time() - start_time

        assert len(predictions) == len(problems), f"Sequence mismatch: {len(test_sequence.seq_names)} vs {len(problems)}"

        for y_pred, problem, dataset_seq_name in zip(predictions, problems, test_sequence.seq_names):
            problem_name = problem['name']
            goal_number = problem['goal_number']

            assert dataset_seq_name == problem_name, f"Sequence metadata naming conflict. Expected: {dataset_seq_name}, got: {problem_name}"

            scores = get_scores(y_pred, possible_goals_indexes) 
            if scores is not None:
                out = get_result(scores, goal_number)
            else:
                out = False
            res = [out, goal_number, scores]

            results.append(
                get_results_json(
                    results=res,
                    percentage=perc_action,
                    time=elapsed_time,
                    goals=possible_goals,
                    problem_name=problem_name
                )
            )

    return results

def load_config(config_path: str) -> dict:
    """
    Load and parse the JSON evaluation configuration file.

    Parameters
    ----------
    config_path : str
        The path pointing to the JSON configuration metadata file.

    Returns
    -------
    dict
        A nested dictionary containing configuration paths and parameters.
    """
    with open(config_path, 'r') as file:
        return json.load(file)

def evaluate(config: dict) -> dict:
    """
    Orchestrate the holistic validation pipeline driven by external configurations.

    Parses structural parameters from the configuration dictionary, initializes
    compiled model artifacts from storage containing custom attention layers, and
    builds accuracy statistics grouped across categorical planning splits.

    Parameters
    ----------
    config : dict
        The configuration dictionary holding path setups and evaluation parameters.

    Returns
    -------
    dict of (str, dict of (str, str))
        A nested dictionary tracking multi-level recognition accuracy scores
        formatted as percentages for each planning category and observation scale.
    """
    paths_cfg = config['paths']
    eval_cfg = config['evaluation']
    
    version = eval_cfg['version']
    suffix = eval_cfg.get('suffix', '')
    base_dir = Path(paths_cfg['base_directory'])

    with open(base_dir / paths_cfg['plans_groups_file'], 'r') as f:
        plan_categories = json.load(f)
    with open(base_dir / paths_cfg['possible_goals_file'], 'r') as f:
        possible_goals = json.load(f)
    with open(base_dir / paths_cfg['goals_dictionary_file'], 'r') as f:
        dizionario_goal = json.load(f)
    
    model_path = base_dir / 'experiments' / f'v{version}' / f'visual-grnet-bw-v{version}{suffix}.keras'
    
    with open(model_path.parent / f'visual-grnet-bw_params-v{version}.json', 'r') as f:
        params = json.load(f)
        
    dataset_str = params['dataset']['dataset_version']
    normalize = params['dataset']['normalize']
    max_sequence_dim = params['dataset']['max_sequence_dim']
    num_classes = params['dataset']['num_classes']
    embedding_dim = params['embedding_dim']

    emb_dir = paths_cfg['embeddings_directory']
    test_npz_file = base_dir / emb_dir / f'test_pergen_{dataset_str}.npz'
    train_npz_file = base_dir / emb_dir / f'train_{dataset_str}.npz'
    
    results_dir = base_dir / paths_cfg['results_directory']
    result_file_path = results_dir / f'result_{version}{suffix}.pkl'

    all_res = []
    if result_file_path.exists(): 
        res_df: pd.DataFrame = pd.read_pickle(result_file_path)
        print(len(res_df))
    else: 
        try:
            model = load_model(
                model_path,
                custom_objects={
                    "AttentionWeights": AttentionWeights,
                    "ContextVector": ContextVector
                },
                compile=False
            )
            print("Model loaded")
            print(model.summary())
        except OSError as e:
            print(e)
            print(
                "Error while loading the model.\n"
                "Please check the configuration parameters are correct."
            )
            return {}

        for category, plans in plan_categories.items(): 
            print('Processing category:', category)
            category_result = compute_results_plans_category(
                model=model,
                problems=plans['plans'],
                possible_goals=possible_goals[category],
                dizionario_goal=dizionario_goal,
                test_npz_file=test_npz_file,
                train_npz_file=train_npz_file,
                results_dir=results_dir,
                norm=normalize,
                embedding_dim=embedding_dim,
                max_sequence_len=max_sequence_dim,
                num_classes=num_classes,
            )
            
            for r in category_result:
                r['p'] = category

            all_res.extend(category_result)

        res_df = pd.DataFrame(all_res)
        res_df.to_pickle(result_file_path)

    percentages = list(res_df["PERCENTAGE"].unique())
    ps = list(res_df['p'].unique())
    
    results_dict = {}
    for p in ps:
        p_str = p
        results_dict[p_str] = {}
        for perc in percentages:
            results = res_df[(res_df['PERCENTAGE'] == perc) & (res_df['p'] == p)]
            correct = results[results['PREDICTED'] == True]
            acc = len(correct) / len(results)
            results_dict[p_str][f'{perc * 100}%'] = f'{acc * 100:.2f}'
    
    return results_dict

def run_evaluation(config_path: str = "eval_config.json") -> None:
    """
    Execute the goal recognition evaluation pipeline and format outputs as a DataFrame.

    Loads runtime configurations from a JSON file, invokes the performance 
    evaluation script, filters results based on configured display categories, 
    and prints a structured evaluation summary table to standard output.

    Parameters
    ----------
    config_path : str, default "eval_config.json"
        The file path targeting the external JSON configuration schema.
    """
    config = load_config(config_path)
    res_dict = evaluate(config)

    if not res_dict:
        print("[!] Evaluation returned no results.")
        return

    res_df = pd.DataFrame(res_dict)
    
    display_cols = config['evaluation'].get('display_categories', [])
    valid_cols = [col for col in display_cols if col in res_df.columns]
    
    if valid_cols:
        res_df = res_df.loc[:, valid_cols]
        
    print(res_df)
    
if __name__ == "__main__":
    run_evaluation("../files/configuration/visual_grnet_evaluation_configuration.json")