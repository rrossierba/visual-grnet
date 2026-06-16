from keras.utils import Sequence
import tensorflow as tf
from PIL import Image
import numpy as np
import random
import math
from pathlib import Path
from typing import Union, List, Dict, Tuple, Generator

class SequenceGeneratorMultiPerc(Sequence):
    """
    Legacy image-based Keras Sequence generator with dynamic sequence length sampling.

    This generator reads raw PNG state images directly from disk, executes spatial 
    resizing and normalization, and processes them into tensor sequences. 
    It applies a variable continuous uniform percentage mapping to dynamically 
    subsample each sequence execution path on the fly.

    .. note::
       This component has been deprecated in favor of `EmbeddingSequence` to eliminate 
       the runtime CPU bottleneck caused by synchronous disk I/O and image decoding.

    Parameters
    ----------
    sequences : list of (str or pathlib.Path)
        A collection of directory paths containing the sequence images to be processed.
    image_width : int
        The target horizontal pixel resolution configuration for resizing operations.
    image_height : int
        The target vertical pixel resolution configuration for resizing operations.
    channels : int
        The expected number of color channels (e.g., 1 for grayscale, 3 for RGB).
    dizionario_goal : dict of (str, int)
        A lookup mapping translating uppercase goal fluent strings to unique vector indices.
    goals : dict of (str, list of str)
        A dictionary mapping base problem identifiers to their corresponding list of active subgoals.
    batch_size : int
        The size configuration specifying how many sequences to pack per mini-batch.
    max_dim : int
        The maximum allowed sequence length constraint acting as the padding/truncation boundary.
    min_perc : float
        The lower bound percentage threshold governing the continuous uniform sampling window.
    max_perc : float
        The upper bound percentage threshold governing the continuous uniform sampling window.
    shuffle : bool, default True
        If True, shuffles the sequence dataset paths at initialization and at each epoch completion.
    ignore_last_n_states : int, default 2
        The count of terminal sequence frames to discard before executing the sampling routines.
    """

    def __init__(
        self,
        sequences: List[Union[Path, str]],
        image_width: int,
        image_height: int,
        channels: int,
        dizionario_goal: Dict[str, int],
        goals: Dict[str, List[str]],
        batch_size: int,
        max_dim: int,
        min_perc: float,
        max_perc: float,
        shuffle: bool = True,
        ignore_last_n_states: int = 2
    ) -> None:
        self.sequences = sequences
        self.dizionario_goal = dizionario_goal
        self.goals = goals
        self.image_width = image_width
        self.image_height = image_height
        self.channels = channels
        self.batch_size = batch_size
        self.max_dim = max_dim
        self.min_perc = min_perc
        self.max_perc = max_perc
        self.shuffle = shuffle
        self.ignore_last_n_states = ignore_last_n_states
        
        self.num_classes = len(set(self.dizionario_goal.values()))
        self.sequence_paths = {}

        if self.shuffle:
            random.shuffle(self.sequences)

        for seq_path in self.sequences:
            if isinstance(seq_path, str):
                seq_path = Path(seq_path)
            states = list(seq_path.glob('*.png')) 
            states.sort(key=lambda x: int(x.stem.split('_')[1]))
            
            if self.ignore_last_n_states > 0:
                states = states[:-self.ignore_last_n_states]
                
            self.sequence_paths[seq_path.name] = states

        super().__init__()

    def __len__(self) -> int:
        """
        Calculate the total number of mini-batches generated per training epoch.

        Returns
        -------
        int
            The total batch count ceiling matching dataset dimensions against batch scale steps.
        """
        return math.ceil(len(self.sequences) / self.batch_size)

    def on_epoch_end(self) -> None:
        """
        Shuffle the tracking sequence paths at the completion of an epoch.

        Alters mini-batch structural arrangements to prevent model overfitting conditions.
        """
        if self.shuffle:
            random.shuffle(self.sequences)

    def _get_seed(self, sequence_name: str) -> int:
        """
        Generate a reproducible integer seed derived from the alpha-numeric sequence name.

        Parameters
        ----------
        sequence_name : str
            The unique identifier string of the sequence (e.g., "p000931_2").

        Returns
        -------
        int
            A deterministic integer seed enforcing repeatable random generation boundaries.
        """
        sequence_splits = sequence_name.split('_')
        return int(sequence_splits[0].replace('p', '')) + int(sequence_splits[1])

    def _process_image(self, image_path: Path) -> np.ndarray:
        """
        Load, decode, transform, and normalize a raw image file from disk.

        Handles target channel conversions, executes high-fidelity Lanczos 
        interpolation resizing operations, and scales pixel coordinates to a [0.0, 1.0] float array.

        Parameters
        ----------
        image_path : pathlib.Path
            The absolute or relative filesystem target path pointing to the source image.

        Returns
        -------
        np.ndarray
            A normalized float32 array tracking spatial dimensions as (height, width, channels).
        """
        with Image.open(str(image_path)) as img:
            if self.channels == 3:
                img = img.convert('RGB')
            elif self.channels == 1:
                img = img.convert('L')
            
            if img.size != (self.image_width, self.image_height):
                img = img.resize((self.image_width, self.image_height), Image.Resampling.LANCZOS)
            
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        if self.channels == 1 and img_array.ndim == 2:
            img_array = np.expand_dims(img_array, axis=-1)
            
        return img_array

    def _get_processed_sequence(self, seq_name: str, perc: float) -> np.ndarray:
        """
        Retrieve and assemble the sampled image frame sequence array for a target layout.

        Extracts the baseline sorted path listing, computes the fraction scaling size bounds, 
        and performs a repeatable random selection before invoking pixel matrix transformations.

        Parameters
        ----------
        seq_name : str
            The identity string tracking the target active sequence.
        perc : float
            The continuous fractional coefficient dictating the sample extraction ratio.

        Returns
        -------
        np.ndarray
            A 4D float32 sequence tensor formatted as (max_dim, height, width, channels).
        """
        X = np.zeros((self.max_dim, self.image_height, self.image_width, self.channels), dtype=np.float32)
        
        states = self.sequence_paths.get(seq_name, [])
        if not states or len(states) == 0:
            return X
            
        size = max(1, int(np.ceil(len(states) * perc)))
        
        seed = self._get_seed(seq_name)
        random_generator = random.Random(seed)
        selected_states = random_generator.sample(states, size)
        
        selected_states = selected_states[:self.max_dim]
        
        for j, state_path in enumerate(selected_states):
            X[j] = self._process_image(state_path)
            
        return X

    def __getitem__(self, index: int):
        """
        Construct a batch tracking multi-dimensional image tensors and target vectors.

        Parameters
        ----------
        index : int
            The positional index pointer tracking the active batch block requested.

        Returns
        -------
        X : np.ndarray
            A 5D array structured as (current_batch_size, max_dim, height, width, channels).
        Y : np.ndarray
            A 2D multi-hot encoded goal matrix structured as (current_batch_size, num_classes).
        """
        batches = self.sequences[index * self.batch_size : (index + 1) * self.batch_size]
        current_batch_size = len(batches)
        
        X = np.zeros((current_batch_size, self.max_dim, self.image_height, self.image_width, self.channels), dtype=np.float32)
        Y = np.zeros((current_batch_size, self.num_classes), dtype=np.float32)
        
        for i, seq_path in enumerate(batches):
            seq_name = Path(seq_path).name
            perc = np.random.uniform(self.min_perc, self.max_perc)
            
            X[i] = self._get_processed_sequence(seq_name, perc)
            Y[i] = self._get_goal_vector(seq_name)
            
        return X, Y

    def _get_goal_vector(self, sequence_name: str) -> np.ndarray:
        """
        Compute the multi-hot target goal vector encoding active problem subgoals.

        Parameters
        ----------
        sequence_name : str
            The structural sequence file string name tracking the underlying problem context.

        Returns
        -------
        np.ndarray
            A 1D float32 array tracking active conditions via high activation states.
        """
        problem_id = sequence_name.split('_')[0]
        goals = self.goals.get(problem_id, [])
        
        goal_vector = np.zeros(self.num_classes, dtype=np.float32)
        for subgoal in goals:
            label_idx = self.dizionario_goal.get(subgoal.upper())
            if label_idx is not None:
                goal_vector[label_idx] = 1.0
                
        return goal_vector

class SequenceDatasetMultiPerc:
    """
    Optimized TensorFlow image sequence generator with dynamic bounded-random sampling.

    This class provides an automated wrapper around the `tf.data` pipeline API.
    It builds a data streaming pipeline from a lightweight Python generator, moving 
    heavy processing workloads (decoding, resizing, sorting, and padding) directly 
    into the TensorFlow graph to maximize host multi-core CPU execution parallelism.

    Parameters
    ----------
    image_width : int
        The target horizontal pixel resolution configuration for resizing operations.
    image_height : int
        The target vertical pixel resolution configuration for resizing operations.
    channels : int
        The expected number of color channels (e.g., 1 for grayscale, 3 for RGB).
    sequences : list of (str or pathlib.Path)
        A collection of absolute directory paths containing the planning sequences to load.
    dizionario_goal : dict of (str, int)
        A lookup mapping translating uppercase goal fluent strings to unique vector indices.
    goals : dict of (str, list of str)
        A dictionary mapping base problem identifiers to their corresponding list of active subgoals.
    batch_size : int
        The size configuration specifying how many sequences to pack per mini-batch.
    max_dim : int
        The maximum allowed sequence length constraint acting as the padding boundary.
    min_perc : float
        The lower bound percentage threshold governing the continuous uniform sampling window.
    max_perc : float
        The upper bound percentage threshold governing the continuous uniform sampling window.
    ignore_last_n_states : int, default 2
        The count of terminal sequence frames to discard before executing the sampling routines.

    Attributes
    ----------
    num_classes : int
        The total dimensionality of the target goal vector space.
    sequence_data : list of tuple
        Internal metadata cache repository tracking calculated structural tuples of 
        `(sequence_number, list_of_image_paths, multi_hot_goal_vector)`.
    """

    def __init__(
        self,
        image_width: int,
        image_height: int,
        channels: int,
        sequences: list,
        dizionario_goal: dict,
        goals: dict,
        batch_size: int,
        max_dim: int,
        min_perc: float,
        max_perc: float,
        ignore_last_n_states: int = 2
    ):
        self.sequences = sequences
        self.image_width = image_width
        self.image_height = image_height
        self.channels = channels
        self.batch_size = batch_size
        self.max_dim = max_dim
        self.min_perc = min_perc
        self.max_perc = max_perc
        self.ignore_last_n_states = ignore_last_n_states
        
        self.num_classes = len(set(dizionario_goal.values()))
        self.sequence_data: List[Tuple[int, List[str], List[float]]] = []
        
        for sequence_path in sequences:
            sequence_path = Path(sequence_path)
            states = list(sequence_path.glob('*.png'))

            if not states or len(states) == 0:
                continue
            
            states.sort(key=lambda x: int(x.stem.split('_')[1]))
            if ignore_last_n_states > 0 and len(states) > ignore_last_n_states:
                states = states[:-ignore_last_n_states]
            
            problem_id = sequence_path.name.split('_')[0]
            sequence_number = int(problem_id.replace('p', ''))
            goal_vector = [0.0] * self.num_classes
            for subgoal in goals.get(problem_id, []):
                label_idx = dizionario_goal.get(subgoal.upper())
                if label_idx is not None:
                    goal_vector[label_idx] = 1.0

            self.sequence_data.append((sequence_number, [str(p) for p in states], goal_vector))

    def _sample_generator(self) -> Generator[Tuple[int, List[str], List[float]], None, None]:
        """
        Yield single sequence meta-records sequentially from the precalculated tracking cache.

        Returns
        -------
        Generator
            A native Python generator streaming structural tuples containing unique identification
            indexes, file paths lists, and multi-hot goal vectors.
        """
        for sequence_number, paths, goal in self.sequence_data:
            yield sequence_number, paths, goal

    def _read_and_decode(self, file_path: tf.Tensor) -> tf.Tensor:
        """
        Load, parse, and scale a standalone target image path using native graph execution operations.

        Parameters
        ----------
        file_path : tf.Tensor
            A scalar string tensor mapping the source file path locations on disk.

        Returns
        -------
        tf.Tensor
            A 3D float32 image tensor formatted as (image_height, image_width, channels)
            normalized within the range [0.0, 1.0].
        """
        img_bytes = tf.io.read_file(file_path)
        img = tf.image.decode_png(img_bytes, channels=self.channels)
        if img.shape[0] != self.image_height or img.shape[1] != self.image_width:
            img = tf.image.resize(img, [self.image_height, self.image_width])
        img = tf.cast(img, tf.float32) / 255.0
        return img

    def _process_sequence_tf(self, sequence_number: tf.Tensor, paths: tf.Tensor, goal: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Execute deterministic stateless subsampling, image decoding map routines, and sequence padding.

        This method operates entirely within the TensorFlow runtime graph. It samples a continuous
        random percentage threshold, handles stateless multi-core data shuffles using unique sequence 
        identifiers as operational seeds, maps file-read tasks, and forces constant tensor shapes.

        Parameters
        ----------
        sequence_number : tf.Tensor
            A scalar integer tensor uniquely tracking the numerical index of the source planning task.
        paths : tf.Tensor
            A 1D string tensor tracking the structural ordered file paths list matching the sequence.
        goal : tf.Tensor
            A 1D float32 tensor representing the multi-hot encoded target goal state configurations.

        Returns
        -------
        images_padded : tf.Tensor
            A 4D sequence tensor of static dimensions (max_dim, image_height, image_width, channels) 
            padded with -1.0 constants.
        goal : tf.Tensor
            A 1D tensor of static dimension (num_classes,) tracking the target outputs.
        """
        num_states = tf.shape(paths)[0]
        
        perc = tf.random.uniform([], self.min_perc, self.max_perc)
        size = tf.cast(tf.math.ceil(tf.cast(num_states, tf.float32) * perc), tf.int32)
        size = tf.minimum(tf.maximum(1, size), self.max_dim)

        seed = tf.stack([sequence_number, 0])
        shuffled = tf.random.experimental.stateless_shuffle( # type: ignore
            tf.range(num_states),
            seed=seed
        )
        indices = tf.sort(shuffled[:size])
        selected_paths = tf.gather(paths, indices)

        images = tf.map_fn(self._read_and_decode, selected_paths, fn_output_signature=tf.float32)

        padding_size = self.max_dim - tf.shape(images)[0]
        paddings = [[0, padding_size], [0, 0], [0, 0], [0, 0]]
        images_padded = tf.pad(images, paddings, constant_values=-1.0)

        images_padded.set_shape([self.max_dim, self.image_height, self.image_width, self.channels])
        goal.set_shape([self.num_classes])

        return images_padded, goal

    def build_dataset(self) -> tf.data.Dataset:
        """
        Assemble the multi-threaded optimized tf.data.Dataset processing pipeline.

        Instantiates the pipeline from the raw cache generator generator, injects multi-epoch
        shuffling operations, maps file processing operations across execution CPU threads, 
        packs individual entries into uniform batch clusters, and enforces tracking cardinalities.

        Returns
        -------
        tf.data.Dataset
            An optimized training-ready dataset instance configured with parallel pipeline execution 
            and automated prefetching elements.
        """
        dataset = tf.data.Dataset.from_generator(
            self._sample_generator,
            output_signature=(
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(None,), dtype=tf.string), 
                tf.TensorSpec(shape=(self.num_classes,), dtype=tf.float32)
            )
        )
        
        dataset = dataset.shuffle(buffer_size=len(self.sequence_data), reshuffle_each_iteration=True)
        dataset = dataset.map(self._process_sequence_tf, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        total_steps = math.ceil(len(self.sequences) / self.batch_size)
        dataset = dataset.assert_cardinality(total_steps)
        print(f"[SequenceDataset] {total_steps} step caricati")
        return dataset