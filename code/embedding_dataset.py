"""
Utilities for converting planning states and goals into embeddings and sequences.

This module handles the extraction and numerical ordering of sequential state images
from disk and parses PDDL problem goals into multi-hot vector representations
suitable for sequence-to-sequence model inputs.
"""

import math
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Dict, List, Tuple, Union
from keras.models import Model
from keras.utils import Sequence

def _compute_goal_vector(
    sequence_name: str,
    goals: Dict[str, List[str]],
    dizionario_goal: Dict[str, int],
    num_classes: int,
) -> np.ndarray:
    """
    Compute the multi-hot goal vector for a PDDL planning problem.

    The goal vector is constructed as a multi-hot encoded array where each active 
    index (set to 1.0) represents a specific fluent that must be satisfied in the 
    problem's target goal state condition.

    Parameters
    ----------
    sequence_name : str
        The identifier name of the sequence, used to extract the base problem ID.
    goals : dict of (str, list of str)
        A dictionary mapping problem identifiers to their respective list of
        goal fluents formatted in lowercase.
    dizionario_goal : dict of (str, int)
        A lookup dictionary mapping uppercase fluent string configurations to their
        corresponding unique integer vector indices.
    num_classes : int
        The total dimensionality of the target goal vector space.

    Returns
    -------
    np.ndarray
        A 1D float32 array representing the multi-hot encoded goal state.
    """    
    problem_id = sequence_name.split('_')[0]
    goal_vector = np.zeros(num_classes, dtype=np.float32)
    for subgoal in goals.get(problem_id, []):
        idx = dizionario_goal.get(subgoal.upper())
        if idx is not None:
            goal_vector[idx] = 1.0
    return goal_vector

def _get_sorted_states(
    seq_path: Path,
) -> List[Path]:
    """
    Retrieve and sort all state image paths belonging to a specific sequence directory.

    Scans the targeted filesystem path for PNG images and sorts them numerically 
    based on the step index extracted from the second token of their filenames.

    Parameters
    ----------
    seq_path : pathlib.Path
        The filesystem directory path tracking the target sequence states.

    Returns
    -------
    list of pathlib.Path
        A numerically sorted list of image file paths representing sequential states.
    """
    states = sorted(
        seq_path.glob('*.png'),
        key=lambda x: int(x.stem.split('_')[1])
    )
    return states

def precompute_embeddings_split(
    encoder: Model,
    splits: Dict[str, Union[List[Path], List[str]]],
    dizionario_goal: Dict[str, int],
    goals: Dict[str, Dict[str, List[str]]],
    output_dir: Union[Path, str],
    image_height: int,
    image_width: int,
    channels: int,
    batch_size: int = 64,
    verbose: int = 0
) -> dict | None:
    """
    Precompute and store compressed state embeddings and multi-hot goal vectors.

    This function iterates through dataset splits (e.g., 'train', 'val', 'test'),
    extracts sorted image state frames for each planning sequence, runs a batched
    forward pass through the provided encoder, and maps the associated PDDL goals.
    The resulting arrays are packed and exported into individual compressed NPZ archives.

    Parameters
    ----------
    encoder : keras.models.Model
        The trained or initialized structural model used to project image states 
        into the latent embedding space.
    splits : dict of (str, list of pathlib.Path or list of str)
        A dictionary mapping split identifiers to sequences of filesystem directories 
        containing target image frames.
    dizionario_goal : dict of (str, int)
        A lookup mapping tracking uppercase goal fluents to their unique integer 
        positional array indices.
    goals : dict of (str, dict of (str, list of str))
        Nested dictionary tracking split names to a secondary mapping of sequence 
        identifiers and their respective active goal fluents list.
    output_dir : str or pathlib.Path
        The destination root path directory where the compiled NPZ archives will be saved.
    image_height : int
        The fixed spatial target vertical pixel height configuration for decoding inputs.
    image_width : int
        The fixed spatial target horizontal pixel width configuration for decoding inputs.
    channels : int
        The color channel count dimension of the sequence state images.
    batch_size : int, default 64
        The slice scaling size passed to the sub-routine network inference pipeline.
    verbose : int, default 0
        The verbosity level setting passed to the internal Keras prediction call.

    Returns
    -------
    dict of (str, pathlib.Path) or None
        A dictionary mapping the parsed split labels to their validated output 
        NPZ archive paths on disk.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    num_classes = len(set(dizionario_goal.values()))
    res = {}

    for split_name, sequence_paths in splits.items():
        out_file = output_dir / f'{split_name}.npz'
        res[split_name] = out_file

        if out_file.exists():
            print(f'[SKIP] {split_name}.npz already exists')
            continue

        print(f'\n[{split_name.upper()}] Getting frames for {len(sequence_paths)} sequences...')

        sequences: List[Tuple[str, List[Path]]] = []
        skipped = 0
        for seq_path in sequence_paths:
            seq_path = Path(seq_path)
            states = _get_sorted_states(seq_path)
            if not states:
                print(f'[!] {seq_path.name}: there are no valid states, skip')
                skipped += 1
                continue
            sequences.append((seq_path.name, states))

        if not sequences:
            print(f'[!] No valid sequence for "{split_name}", skip split')
            continue

        print(f'  ok: {len(sequences)} valid sequences ({skipped} skipped)')

        embeddings_map = _encode_all_sequences(
            encoder=encoder,
            sequences=sequences,
            image_height=image_height,
            image_width=image_width,
            channels=channels,
            batch_size=batch_size,
            verbose=verbose
        )

        npz_data: Dict[str, np.ndarray] = {}
        seq_names: List[str] = []

        for seq_name, embeddings in embeddings_map.items():
            goal_vector = _compute_goal_vector(seq_name, goals[split_name], dizionario_goal, num_classes)
            npz_data[f'embeddings_{seq_name}'] = embeddings
            npz_data[f'goal_{seq_name}'] = goal_vector
            seq_names.append(seq_name)

        npz_data['seq_names'] = np.array(seq_names, dtype=str)
        np.savez_compressed(out_file, **npz_data)
        print(f'Saved {out_file} ({len(seq_names)} sequences)')
    
    return res

def get_mean_std(npz_path: Union[Path, str]) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate global feature-wise mean and standard deviation from an NPZ dataset.

    Flattens and extracts all computed state embeddings stored across multiple 
    sequences inside an archive file to compute tracking statistics. Elements exhibiting 
    zero variance have their standard deviation overridden to 1.0 to prevent division-by-zero
    faults during Z-score standardization.

    Parameters
    ----------
    npz_path : str or pathlib.Path
        The target path pointing to the compressed NPZ archive containing the 
        precomputed sequence embeddings map.

    Returns
    -------
    mean : np.ndarray
        A 1D array of shape (embedding_dim,) containing the calculated empirical 
        feature-wise mean.
    std : np.ndarray
        A 1D array of shape (embedding_dim,) containing the corrected empirical 
        standard deviation.
    """ 
    data = np.load(npz_path, allow_pickle=False)
    seq_names = data['seq_names'].tolist()

    all_embeddings = np.concatenate([data[f'embeddings_{name}'] for name in seq_names], axis=0)

    mean = all_embeddings.mean(axis=0)
    std  = all_embeddings.std(axis=0)
    std  = np.where(std == 0, 1.0, std)

    return mean, std

def _encode_all_sequences(
    encoder: Model,
    sequences: List[Tuple[str, List[Path]]],
    image_height: int,
    image_width: int,
    channels: int,
    batch_size: int,
    verbose: int
) -> Dict[str, np.ndarray]:
    """
    Extract state embeddings for multiple image sequences using an optimized TF pipeline.

    Flattens all incoming sequence file paths into a single contiguous stream to maximize 
    GPU pipeline utilization via a `tf.data.Dataset`. After running batched model inference, 
    the resulting flat predictions matrix is partitioned back into sequence-specific 
    arrays using tracked sequence lengths.

    Parameters
    ----------
    encoder : keras.models.Model
        The sub-network model used to encode raw images into dense latent representations.
    sequences : list of tuple of (str, list of pathlib.Path)
        A list mapping each unique sequence name to its corresponding ordered collection
        of filesystem image state paths.
    image_height : int
        The target vertical pixel resolution for resizing the input images.
    image_width : int
        The target horizontal pixel resolution for resizing the input images.
    channels : int
        The expected number of color channels for decoding the image streams.
    batch_size : int
        The slice processing size for chunking elements within the dataset stream.
    verbose : int
        The verbosity display configuration passed directly to the model's prediction runner.

    Returns
    -------
    dict of (str, np.ndarray)
        A dictionary mapping sequence string names to a 2D float32 NumPy array of shape 
        (num_states, embedding_dim), where `num_states` matches the respective sequence length.
    """    
    all_paths: List[str] = []
    seq_lengths: List[Tuple[str, int]] = []
    for seq_name, states in sequences:
        all_paths.extend([str(s) for s in states])
        seq_lengths.append((seq_name, len(states)))

    print(f'Total images to process: {len(all_paths)}')

    images_ds = (
        tf.data.Dataset.from_tensor_slices(all_paths)
        .map(
            lambda p: _read_image_tf(p, image_height, image_width, channels),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    all_embeddings = encoder.predict(images_ds, verbose=verbose)
    if isinstance(all_embeddings, (list, tuple)):
        all_embeddings = all_embeddings[0]
    
    res: Dict[str, np.ndarray] = {}
    cursor = 0
    for seq_name, length in seq_lengths:
        res[seq_name] = all_embeddings[cursor : cursor + length].astype(np.float32)
        cursor += length

    return res

def _read_image_tf(
    path: tf.Tensor,
    image_height: int,
    image_width: int,
    channels: int,
) -> tf.Tensor:
    """
    Read, decode, resize, and scale a single image tensor using native TF ops.

    This function acts as a standalone mapping target for parallel `tf.data` pipelines,
    ensuring graph execution compatibility by avoiding non-serializable Python/PIL code.

    Parameters
    ----------
    path : tf.Tensor
        A scalar string tensor tracking the target file path on disk.
    image_height : int
        The targeted target spatial vertical pixel constraint.
    image_width : int
        The targeted target spatial horizontal pixel constraint.
    channels : int
        The color channel layout dimension configuration passed to the PNG decoder.

    Returns
    -------
    tf.Tensor
        A 3D float32 tensor of shape (image_height, image_width, channels) normalized 
        within the range [0.0, 1.0].
    """
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=channels)
    img = tf.image.resize(img, [image_height, image_width])
    img = tf.cast(img, tf.float32) / 255.0
    return img

class EmbeddingSequence(Sequence):
    """
    Custom Keras Sequence dataset partitioner for processing sequential embeddings.

    This data generator pipeline handles the batching, subset sampling, Z-score 
    normalization, and fixed-length padding of precomputed planning state/action 
    embeddings and their corresponding multi-hot encoded target goal vectors.

    Parameters
    ----------
    npz : str, pathlib.Path or dict
        Path to the compressed NPZ file or a pre-loaded dictionary containing 
        the keys 'seq_names', 'embeddings_{name}', and 'goal_{name}'.
    max_dim : int
        The fixed maximum sequence length constraint enforced via truncation or padding.
    batch_size : int
        The number of sequence sample matrices packed within a single mini-batch.
    embedding_dim : int
        The operational feature dimensionality of the input embedding vectors.
    num_classes : int
        The total dimensionality of the target goal vector space.
    ignore_last_n_states : int, default 2
        The explicit count of terminal sequence frames to discard before sampling.
    norm_mean : Optional[np.ndarray], default None
        Empirical feature-wise mean vector of shape (embedding_dim,) used for stabilization.
    norm_std : Optional[np.ndarray], default None
        Empirical feature-wise standard deviation vector of shape (embedding_dim,) used for stabilization.
    perc : Optional[float], default None
        The constant fraction of elements to sample from the active sequence lengths.
        Defaults to 1.0 if not specified.
    limit : Optional[float], default None
        The fraction of total dataset sequences to load (used to subsample large datasets).

    Attributes
    ----------
    max_dim : int
        Maximum permitted sequence size constraint.
    batch_size : int
        The mini-batch partition factor.
    embedding_dim : int
        Feature dimensionality size.
    num_classes : int
        Goal target feature space dimension.
    ignore_last_n_states : int
        Count of truncated terminal frames.
    norm_mean : Optional[np.ndarray]
        Z-score centering shift array.
    norm_std : Optional[np.ndarray]
        Z-score scaling normalization array.
    perc : float
        The sequential frame sampling coefficient.
    seq_names : list of str
        The collection of identified sequence names processed within the partition split.
    """

    def __init__(
        self,
        npz: Union[Path, str, dict],
        max_dim: int,
        batch_size: int,
        embedding_dim: int,
        num_classes: int,
        ignore_last_n_states: int = 2,
        norm_mean: Union[np.ndarray, None] = None,
        norm_std: Union[np.ndarray, None] = None,
        perc: Union[float, None] = None,
        limit: Union[float, None] = None,
    ):
        super().__init__()

        self.max_dim = max_dim
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.ignore_last_n_states = ignore_last_n_states
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.perc = perc if perc is not None else 1.0

        if isinstance(npz, (Path, str)):
            data = np.load(npz, allow_pickle=False)
        elif isinstance(npz, dict):
            data = npz
        else:
            raise TypeError("Unsupported data storage configuration type provided.")

        self.seq_names: List[str] = data["seq_names"].tolist()

        if limit is not None:
            np.random.shuffle(self.seq_names)
            self.seq_names = self.seq_names[:int(limit * len(self.seq_names))]

        self._embeddings: List[np.ndarray] = []
        self._goals: List[np.ndarray] = []
        for seq_name in self.seq_names:
            self._embeddings.append(data[f"embeddings_{seq_name}"].astype(np.float32))
            self._goals.append(data[f"goal_{seq_name}"].astype(np.float32))

        self._indices = np.arange(len(self._embeddings))
        if isinstance(npz, (Path, str)):
            print(f"[EmbeddingSequence] {npz} — {len(self._embeddings)} loaded sequences")

    def __len__(self) -> int:
        """
        Calculate the total number of mini-batches generated per training epoch.

        Returns
        -------
        int
            The total batch count ceiling mapping the dataset size against batch scale bounds.
        """
        return math.ceil(len(self._embeddings) / self.batch_size)

    def __getitem__(self, batch_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Construct a structured mini-batch tracking sequence matrices and target vectors.

        Parameters
        ----------
        batch_idx : int
            The unique positional batch lookup pointer evaluated during training execution.

        Returns
        -------
        X : np.ndarray
            A 3D tensor batch of shape (actual_batch, max_dim, embedding_dim) 
            containing processed transition action embeddings.
        Y : np.ndarray
            A 2D tensor batch of shape (actual_batch, num_classes) mapping the target goal states.
        """     
        start = batch_idx * self.batch_size
        end = min(start + self.batch_size, len(self._embeddings))
        batch_indices = self._indices[start:end]
        actual_batch = len(batch_indices)

        X = np.zeros((actual_batch, self.max_dim, self.embedding_dim), dtype=np.float32)
        Y = np.zeros((actual_batch, self.num_classes), dtype=np.float32)

        for i, seq_idx in enumerate(batch_indices):
            seq_name = self.seq_names[seq_idx]
            action_embeddings = self._get_action_embeddings(self._embeddings[seq_idx])

            X[i] = self._sample_and_pad(embeddings=action_embeddings, seed=self._get_seed(seq_name))
            Y[i] = self._goals[seq_idx]

        return X, Y

    def _get_seed(self, sequence_name: str) -> int:
        """
        Extract a unique consistent integer seed identifier from a sequence name string.

        Guarantees that a specific sequence path variant samples identical components 
        and structural splits consistently across disparate epochs.

        Parameters
        ----------
        sequence_name : str
            The raw alpha-numeric identifier name of the plan layout (e.g., "p000931_2").

        Returns
        -------
        int
            A parsed integer seed derived from the sequence naming schema.
        """        
        sequence_number = int(sequence_name.replace('_', '').replace('p', ''))
        return sequence_number
    
    def _get_perc(self, seed: int) -> float:
        """
        Retrieve the target sequence frame sampling percentage coefficient.

        Parameters
        ----------
        seed : int
            The sampling sequence seed constraint.

        Returns
        -------
        float
            The static target sampling percentage coefficient tracking the state.
        """        
        return self.perc
    
    def _get_action_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compute transition action representations from contiguous sequential state states.

        An action embedding representation is defined mathematically as the vector 
        difference between the resulting state vector $s_{t+1}$ and the preceding state $s_t$.

        Parameters
        ----------
        embeddings : np.ndarray
            A 2D matrix of shape (num_states, embedding_dim) tracking raw state features.

        Returns
        -------
        np.ndarray
            A vectorized 2D matrix of shape (num_states - 1, embedding_dim) representing 
            the transition differences.
        """        
        action_embeddings = np.diff(embeddings, axis=0)
        assert len(action_embeddings) == len(embeddings) - 1, "Mismatched dimensions during action delta extraction."
        return action_embeddings

    def _sample_and_pad(self, embeddings: np.ndarray, seed: int) -> np.ndarray:
        """
        Execute randomized deterministic subsampling, standard scaling, and truncation padding.

        Extracts a specific percentage of frames using a fixed seed, handles 
        optional terminal state truncation constraints, applies optional 
        Z-score standardization, and formats the output into a uniform array dimension.

        Parameters
        ----------
        embeddings : np.ndarray
            The full baseline 2D matrix tracking transition action embeddings.
        seed : int
            The operational seed governing the repeatable random distribution masking.

        Returns
        -------
        np.ndarray
            The processed 2D matrix formatted to uniform dimensions of shape 
            (max_dim, embedding_dim), with unused spaces padded with -100.0.
        """        
        num_states = len(embeddings)
        
        perc = self._get_perc(seed)
        size = max(1, min(int(math.ceil(num_states * perc)), self.max_dim))

        if self.ignore_last_n_states > 0 and perc < 1.0 and num_states - self.ignore_last_n_states > 0:
            num_states = num_states - self.ignore_last_n_states
            embeddings = embeddings[:num_states]
        
        if num_states < size:
            size = num_states

        rng_seq = np.random.default_rng(seed)
        fixed_permutation = rng_seq.permutation(num_states)

        indices = np.sort(fixed_permutation[:size])
        sampled = embeddings[indices]

        if self.norm_mean is not None and self.norm_std is not None:
            sampled = (sampled - self.norm_mean) / self.norm_std

        padded = np.full((self.max_dim, self.embedding_dim), -100.0, dtype=np.float32)
        padded[:size] = sampled

        return padded

    def on_epoch_end(self) -> None:
        """
        Shuffle the internal sequential tracking indices at the completion of an epoch.

        Alters mini-batch composition dynamics across steps to prevent model overfitting.
        """
        np.random.shuffle(self._indices)

class EmbeddingSequenceMultiPerc(EmbeddingSequence):
    """
    Specialized Keras Sequence generator for dynamic bounded-random sampling.

    This class extends `EmbeddingSequence` by introducing a variable sampling strategy. 
    Instead of applying a fixed frame selection percentage to all sequences, it 
    samples a sequence-specific pseudo-random percentage from a uniform continuous 
    distribution bounded by a minimum and maximum threshold.

    Parameters
    ----------
    npz_path : str or pathlib.Path
        Path to the compressed NPZ file or a pre-loaded dictionary tracking sequence 
        embeddings and target goal vectors.
    max_dim : int
        The fixed maximum sequence length constraint enforced via truncation or padding.
    min_perc : float
        The lower bound fraction constraint for the uniform random sampling distribution.
    max_perc : float
        The upper bound fraction constraint for the uniform random sampling distribution.
    batch_size : int
        The number of sequence sample matrices packed within a single mini-batch.
    embedding_dim : int
        The operational feature dimensionality of the input embedding vectors.
    num_classes : int
        The total dimensionality of the target goal vector space.
    ignore_last_n_states : int, default 2
        The explicit count of terminal sequence frames to discard before sampling.
    norm_mean : Optional[np.ndarray], default None
        Empirical feature-wise mean vector of shape (embedding_dim,) used for stabilization.
    norm_std : Optional[np.ndarray], default None
        Empirical feature-wise standard deviation vector of shape (embedding_dim,) used for stabilization.
    limit : Optional[float], default None
        The fraction of total dataset sequences to load for dataset subsetting.

    Attributes
    ----------
    min_perc : float
        The lower threshold coefficient for uniform distribution interval mapping.
    max_perc : float
        The upper threshold coefficient for uniform distribution interval mapping.
    """

    def __init__(
        self,
        npz_path: Union[Path, str],
        max_dim: int,
        min_perc: float,
        max_perc: float,
        batch_size: int,
        embedding_dim: int,
        num_classes: int,
        ignore_last_n_states: int = 2,
        norm_mean: Union[np.ndarray, None] = None,
        norm_std: Union[np.ndarray, None] = None,
        limit: Union[float, None] = None,
    ):
        self.min_perc = min_perc
        self.max_perc = max_perc

        super().__init__(
            npz=npz_path,
            max_dim=max_dim,
            batch_size=batch_size,
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            ignore_last_n_states=ignore_last_n_states,
            norm_mean=norm_mean,
            norm_std=norm_std,
            limit=limit
        )
    
    def _get_perc(self, seed: int) -> float:
        """
        Sample a pseudo-random percentage from a bounded uniform distribution.

        Overrides the parent method to provide randomized scaling behavior.
        By utilizing the sequence-specific integer seed, the calculated percentage 
        remains deterministic for any given unique sequence name across disparate epochs, 
        maintaining computational consistency.

        Parameters
        ----------
        seed : int
            The sequence-derived seed value initializing the local generator state.

        Returns
        -------
        float
            A randomly sampled float value bounded bounded within the range 
            [`min_perc`, `max_perc`].
        """        
        rng_seq = np.random.default_rng(seed)
        perc = rng_seq.uniform(self.min_perc, self.max_perc)
        return perc