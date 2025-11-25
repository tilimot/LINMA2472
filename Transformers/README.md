# Single-Head Transformer (Custom AD Implementation)

This project implements a **Transformer** from scratch in Julia. It features a **Single-Head Attention** mechanism and is trained using a **Handmade Autodifferentiation (AD) system** (`VectReverse`) developed for this project.

---

## I. Project Structure

The project organization is the following:

| Directory/File | Purpose |
| :--- | :--- |
| `corpus/` | Contains the raw training corpus (`input.txt`). |
| `BSON_files/` | Stores serialized results (preprocessed data and trained weights). |
| `Project.toml`, `Manifest.toml` | Handles project dependencies and version locking. |
| `run.jl` | The main script (Command Line Interface entry point). |
| `utils/config.jl` | Centralized location for all hyperparameters and file paths. |
| `utils/preprocess.jl` | Tokenization, vocabulary building, and data encoding. |
| `utils/transformers.jl` | **Core Model Logic:** Attention mechanism, Forward pass, and utilities. |
| `utils/models.jl` | Generation structures and training function wrappers. |
| `utils/reverse_vectorized.jl` | **Custom AD System (VectReverse).** |

---

## II. Architecture and Key Mechanisms

### 1. Word-Level Encoding and Embedding
The model uses **Word Tokenization** with a minimum frequency threshold (`MIN_FREQ`) to limit the vocabulary size ($V$).
* **Mechanism:** Token indices are mapped to dense vectors (dimension $d_{model}$) via an embedding lookup table ($\mathbf{E}$). This matrix ($\mathbf{E}$) is the first set of weights learned by the model.
* **Tokens:** Special tokens like `[BOS]`, `[EOS]`, and `[UNK]` are used to manage sequence starts, ends, and rare words, respectively.

### 2. Single-Head Causal Attention
The core computational layer uses the Scaled Dot-Product Attention:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$

* **Causal Masking ($M$):** A strict upper-triangular mask (set to $-\infty$) is applied to the attention scores. This is crucial for generation tasks as it ensures that the prediction at time $t$ depends only on information available at or before time $t$, maintaining the auto-regressive property.

### 3. Custom Autodifferentiation (AD) and Training
* **Gradient Calculation:** The entire backward pass is computed using the custom AD system, accessed via **`VectReverse.gradient!`**.
* **Loss Function:** Training minimizes the **Mean Squared Error (MSE)** between the model's logits and the one-hot encoded target tokens.
* **Optimizer:** Weights are updated using the **Adam** optimization rule from the `Optimisers` package.

---

## III. Configuration and Usage Guide

All model parameters and file paths are centralized in **`utils/config.jl`**.

### 1. Prerequisites and Setup
1.  **Navigate** to the project's root directory (`Transformers/`).
2.  **Instantiate Dependencies**: Since `Project.toml` is present, use the Julia package manager to set up the environment:
    ```bash
    julia
    julia> ]
    (@v1.x) pkg> activate .
    (Transformers) pkg> instantiate
    ```

### 2. Command Line Interface (`run.jl`)
The script must be executed using the `--project=.` flag to ensure the local environment is loaded.

| Command | Action | Description |
| :--- | :--- | :--- |
| `preprocess` | Data preparation | Tokenizes the corpus and saves encoded data to `BSON_files/`. |
| `train` | Model Training | Initializes, trains the model, and saves weights to `BSON_files/`. Uses parameters from `config.jl`. |
| `generate` | Text Generation | Loads trained weights and generates a text sequence. |
| `all` | Full Pipeline | Executes `preprocess`, `train`, and `generate` sequentially. |

### 3. Execution Examples

#### A. Preprocessing
(Must be run first to create the data files.)

```bash
julia --project=. run.jl preprocess
```


#### B. Training

(Uses default parameters: dmodel​=32, Niters​=100, LR=0.01.)

```bash
julia --project=. run.jl train
```

#### C. Generation

Load the saved weights and generate text based on a prompt:

```bash
# Generate with a context:
julia --project=. run.jl generate "The queen said that"

# Generate from the start (using [BOS]):
julia --project=. run.jl generate
```

#### D. All

Launch point A, B, C at the same time: 

```bash
julia --project=. run.jl all

# Generate with a context
julia --project=. run.jl generate "The queen said that"
```
