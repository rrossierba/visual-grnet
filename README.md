# Visual GRNet

**Visual Goal Recognition Network** — A deep learning framework for goal recognition in planning domains using visual state representations.

Visual GRNet addresses the problem of *goal recognition* from sequences of visual states.
Given a partial observation of state images produced during the execution of a plan, the system predicts which goal the agent is pursuing.
The approach relies on a two-stage learning pipeline: a convolutional autoencoder first learns compressed latent representations of state images, and a recurrent attention-based network then operates on sequences of these embeddings to predict multi-hot goal vectors.

>**Note:** The dataset is not included in this repository due to its size. Only the source code and configuration files are provided.
>However, the used dataset is composed of images from the Blocksworld Planning domain, obtained using [this tool](https://github.com/rrossierba/plan-image-conversion) from already solved plans.


---

## Pipeline

The system is organized into three sequential stages. The evaluation of the autoencoder and Visual GRNet model (e.g., reconstruction quality, multi-label classification metrics) and the evaluation of the goal recognition task (i.e., identifying the correct goal among candidates) are treated as **distinct evaluation processes**.

### 1. Train the Autoencoder

A convolutional autoencoder is trained on state images (256×256 RGB) to learn a compressed latent representation.
The encoder maps each state image to a fixed-size embedding vector (default: 256 dimensions), which serves as the input representation for the downstream goal recognition model.
Three architectural variants are supported:

| Variant | Description |
|---|---|
| **Autoencoder** | Standard autoencoder with optional L1 latent regularization for sparsity |
| **DAE** | Denoising Autoencoder — reconstructs clean images from Gaussian-corrupted inputs, encouraging noise-invariant representations |
| **VAE** | Variational Autoencoder — learns a parameterized latent Gaussian distribution, regularized via a scaled KL divergence penalty |

All variants share the same convolutional encoder–decoder backbone with configurable filter sizes, kernel dimensions, activation functions, and optional batch normalization. Training uses MSE reconstruction loss and is managed through JSON configuration files.

Once trained, the encoder is frozen and used to extract state embeddings for the next stage.

### 2. Train the Visual GRNet Model

This stage encompasses two sub-steps: **embedding extraction** and **model training**.

#### Embedding Extraction

The trained encoder is used to project all state images across the dataset splits (train, validation, test) into the latent space. The resulting embeddings are stored as compressed `.npz` archives, each containing per-sequence embedding matrices and corresponding multi-hot goal vectors. This precomputation step eliminates redundant image I/O during model training.

#### Model Training

The Visual GRNet model is a recurrent attention-based sequence classifier that processes *action embeddings* — defined as the vector difference between consecutive state embeddings ($a_t = s_{t+1} - s_t$) — and outputs a multi-hot prediction over the goal fluent space (506 classes in the Blocksworld domain).

The architecture is composed of:

- **Adapter Layer** *(optional)* — A two-layer feed-forward projection with a residual skip connection and layer normalization, used to align the embedding dimensionality with the recurrent hidden size.
- **Masked LSTM** — A recurrent layer (optionally wrapped in a Bidirectional layer) that processes variable-length padded sequences. Padding positions are identified by a sentinel value (`-100.0`) and excluded via masking.
- **Attention Mechanism** — An implementation of Hierarchical Attention Networks (HAN) that scores each time step through a learned weight vector and tanh non-linearity, producing a normalized attention distribution. A context vector is computed as the weighted sum of LSTM hidden states.
- **Dense Output Head** — A sigmoid-activated dense layer producing multi-label predictions. Output biases are initialized using log-odds derived from empirical label frequencies to stabilize early training in the presence of severe class imbalance.

Key training features include:

- **Dynamic Observation Window** — During training, each sequence is observed at a random percentage (uniformly sampled between configurable bounds, e.g., 30%–70%) to improve generalization across varying levels of partial observability.
- **Linear Learning Rate Warmup** — The learning rate is linearly ramped from a minimum value to a peak over a configurable number of warmup epochs, preventing early gradient instability in recurrent architectures with linear activations.
- **Hyperparameter Tuning** — An Optuna-based optimization pipeline with Tree-structured Parzen Estimator (TPE) sampling and median pruning supports parallelized multi-worker hyperparameter search over LSTM dimensions, dropout rates, learning rates, normalization strategies, and embedding model selection.

### 3. Evaluate Goal Recognition

Goal recognition evaluation measures the model's ability to identify the *correct goal* from a set of candidate goals, given a partial observation of a plan execution.
This is distinct from the multi-label classification metrics computed during training — here, the task is framed as a selection problem.

The evaluation procedure works as follows:

1. A partial sequence of states is observed at a fixed observation percentage (10%, 30%, 50%, 70%, or 100%).
2. The Visual GRNet model produces a predicted activation vector over the goal fluent space.
3. Each candidate goal is scored by summing the predicted activations at the indices corresponding to its constituent fluents.
4. The candidate with the highest cumulative score is selected as the recognized goal.
5. If multiple candidates achieve the same maximum score, one is selected uniformly at random.

Results are reported as recognition accuracy per planning category and observation percentage. The evaluation spans multiple plan complexity categories (e.g., p01 through p07), and results are serialized as DataFrames for further analysis.

---

## Project Structure

```
visual-grnet/
├── README.md
├── code/
│   ├── autoencoder/
│   │   ├── autoencoder.py              # Autoencoder architectures (AE, DAE, VAE)
│   │   ├── dataset_loader.py           # Image loading pipeline for autoencoder training
│   │   ├── train_autoencoder.py        # Autoencoder training entry point
│   │   └── optuna_tuning.py            # Hyperparameter tuning for the autoencoder
│   ├── main.py                         # Embedding extraction & Visual GRNet training pipeline
│   ├── visual_grnet.py                 # Model definition, training loop, and evaluation utilities
│   ├── attention_extraction_layers.py  # HAN attention layers (AttentionWeights, ContextVector)
│   ├── embedding_dataset.py            # Embedding extraction and sequence data loaders
│   ├── sequence_generator.py           # Legacy image-based sequence generators
│   ├── get_predictions_and_results.py  # Goal recognition evaluation pipeline
│   ├── optuna_tuning.py                # Hyperparameter optimization for Visual GRNet
│   ├── report.py                       # Automated Markdown/PDF report generation
│   └── constants.py                    # Metric names and threshold definitions
├── files/
│   └── configuration/
│       ├── autoencoder_configuration.json
│       ├── autoencoder_tune_config.json
│       ├── embeddings_extraction.json
│       ├── train_visual_grnet_config.json
│       ├── visual_grnet_tuning_configuration.json
│       └── visual_grnet_evaluation_configuration.json
```

---

## Technical Details

### Dependencies

The framework is built on **TensorFlow / Keras 3** and relies on the following main libraries:

- **NumPy**, **Pandas** — Data manipulation and storage
- **scikit-learn** — Classification metrics (precision, recall, F1, AUC)
- **Optuna** — Bayesian hyperparameter optimization
- **Matplotlib** — Training history visualization
- **pypandoc** + **XeLaTeX** — Automated PDF report generation (optional)
- **Pillow** — Image processing (legacy pipeline)

### Configuration

All pipeline stages are driven by JSON configuration files located in `files/configuration/`. These define dataset paths, architectural hyperparameters, training schedules, search spaces, and evaluation parameters. This design allows full experimental reproducibility without modifying source code.

### Data Format

The input data consists of **PNG images** representing individual states. Each planning problem produces a sequence of state images, where each image captures the spatial configuration of blocks at a specific step of the plan execution. These images are organized on disk as directories (one per sequence), each containing numerically ordered PNG files (e.g., `state_0.png`, `state_1.png`, ...).

During the embedding extraction phase, each image is loaded, resized to 256×256 pixels, normalized to the [0, 1] range, and passed through the trained autoencoder's encoder to produce a dense embedding vector. The resulting per-sequence embedding matrices are then **cached into compressed `.npz` archives** (one per dataset split: train, validation, test). This caching step decouples the autoencoder from the Visual GRNet training process, avoiding repeated image I/O and encoder forward passes.

Each `.npz` archive contains:
- `seq_names` — An array of sequence identifiers
- `embeddings_{name}` — A 2D float32 matrix of shape `(num_states, embedding_dim)` for each sequence, where each row is the latent representation of a single state image
- `goal_{name}` — A 1D float32 multi-hot vector of shape `(num_classes,)` encoding the target goal fluents for that sequence

The Visual GRNet model is trained exclusively on these cached embeddings. At training time, the data loaders read the `.npz` archives, compute action embeddings as the element-wise difference between consecutive state vectors, apply dynamic subsampling and padding, and feed the resulting fixed-length sequences to the model.
