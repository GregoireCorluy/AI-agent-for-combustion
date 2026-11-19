QLoRA fine-tuning
=================

This folder contains an SFT/QLoRA pipeline for the local Qwen3-8B model. The task maps a combustion-system description to the InputParameters model used by CombustionAgent.

Files
-----

- configs/qlora.yaml: model, dataset, quantization, LoRA, training, and output settings.
- src/config.py: YAML loading and project-relative path handling.
- src/data.py: reads the chat JSONL and builds the training target.
- src/model.py: 4-bit NF4 loading, k-bit preparation, and PEFT LoRA setup.
- src/validate_dataset.py: checks JSON/schema validity, split group isolation, and token lengths.
- src/train.py: TRL SFTTrainer training entry point.
- src/metrics.py and src/evaluate_model.py: greedy generation and task metrics.
- src/inference.py: one-description generation with the trained adapter.
- src/merge_adapter.py: optional merge for deployment.
- scripts/: shell launchers.

Dataset target and metadata
---------------------------

The source records contain assistant JSON with input_parameters, provenance, explicit_anchors, and other notes. Training keeps only this object as the completion:

    {"input_parameters": { ... }}

The model is not trained to generate provenance or dataset metadata. Those source fields remain available to validation and evaluation. The target completion ends with the tokenizer's EOS token, which is Qwen3's assistant-turn terminator. The prompt is formatted with thinking disabled and the same setting is used by evaluation and inference. Fuel composition uses the field name fraction, whose basis is mole fraction. The schema is imported from src/CombustionAgent/parameters.py.

The dataset labels for inferred condition ranges are documented as provisional application-conditioned sweeps, not validated operating limits or engineering ground truth. Evaluation scores how well the model reproduces those labels; it cannot establish that the inferred ranges are scientifically correct.

The explicit-anchor metric checks that a stated single value remains inside the predicted swept range and that its corresponding unit is preserved. Since the output schema contains a range, not a separate anchor field, the metric does not require a range endpoint to equal the anchor.

Environment
-----------

Use Linux or WSL with all three NVIDIA GPUs visible to CUDA and a CUDA-enabled PyTorch build. Install the repository's CUDA-matched PyTorch first, then the extra packages:

    python -m pip install -r requirements_finetuning.txt

The configured model path is ../models/Qwen3-8B relative to this folder. The loader resolves existing local model directories relative to the config and otherwise passes the value through as a Hub ID. Change model.name_or_path in configs/qlora.yaml to another local snapshot or Hub model ID if needed. The scripts add the repository src/ directory to Python's import path for the InputParameters schema. The default config path is resolved from the script package, so the commands can be launched from another working directory.

Run
---

From fine-tuning/:

    bash scripts/run_train.sh

This runs dataset validation, creates one shared timestamped run name, and starts three-GPU training. To run training directly instead, first validate the data, then supply a unique run name:

    python src/validate_dataset.py --config configs/qlora.yaml
    accelerate launch --multi_gpu --num_processes 3 src/train.py --config configs/qlora.yaml --run-name trial-001

Each process loads a 4-bit model replica on its assigned GPU and trains on a different portion of each batch. DDP synchronizes LoRA gradients across the three replicas; GPU memory is not pooled. The configured per-device batch size is 1 and gradient accumulation is 8, giving an effective global batch size of 24 across three GPUs.

Each training launch gets its own run directory under outputs/qwen3-8b-qlora/runs/. The run name is timestamped by scripts/run_train.sh, or can be supplied as --run-name when launching src/train.py directly. Each run directory keeps its own checkpoint history, final adapter, qlora.yaml snapshot, run.json manifest, and TensorBoard event files. The adapter directory also contains system_prompt.txt: the prepared system prompt used during training. Inference loads this prompt from the selected adapter, so each run remains paired with its own prompt. Existing runs are never reused. A latest_run.json pointer selects the newest completed adapter by default; pass --run-name to evaluate, infer with, or merge a specific run.

Training metrics are written locally as TensorBoard event files under each run's tensorboard/ subdirectory. You do not need to launch TensorBoard before training; the Trainer writes the event files during training. To browse every run, launch it from fine-tuning/ with:

    tensorboard --logdir outputs/qwen3-8b-qlora/runs

Then open the local URL printed by TensorBoard in a browser. Console logging remains enabled as well. You can launch TensorBoard after training and the saved curves remain available.

The training launcher creates a shared run name before starting the three distributed processes. For direct multi-GPU launches, pass the same explicit run name to every process, or use scripts/run_train.sh.

Evaluate the adapter and base model on the test split:

    python src/evaluate_model.py --config configs/qlora.yaml --split test
    python src/evaluate_model.py --config configs/qlora.yaml --split test --base-only

Adapter reports are saved under outputs/evaluation/<run-name>/, with the run name selected from the latest completed training by default. Base-model-only reports are saved under outputs/evaluation/base/. Use --run-name to select an older adapter run. To generate one result with the latest adapter:

    python src/inference.py "I want to simulate an ammonia flame in a stationary gas turbine."

To select a particular run, add --run-name followed by its directory name.

Optional adapter merge:

    python src/merge_adapter.py --config configs/qlora.yaml

The merge command uses the latest completed run by default; --run-name selects an older run.

The merged model is loaded in BF16 on CPU during merging, so allow sufficient system RAM and disk space. Merging is optional; inference can load the base model and adapter separately.

Tuning
------

Defaults use 2,048 tokens, batch size 1 per GPU for training, evaluation batches of 4, gradient accumulation 8, 3 epochs, NF4 double quantization, BF16 compute, LoRA rank 16, and all-linear targets. Evaluation uses left padding so examples of different prompt lengths can be generated together. If evaluation runs out of GPU memory, lower evaluation.batch_size; if training runs out of memory, reduce max_seq_length first.

The validator checks that metadata.split matches its JSONL file, and that scenario groups and whitespace-normalized user prompts do not cross split boundaries. It decodes one formatted sample and checks that its completion contains exactly one EOS token at the end. It reports token-length percentiles for both the full prompt-plus-completion sequence and the completion alone. It fails if a full sequence exceeds training.max_seq_length or if a completion exceeds evaluation.max_new_tokens. The test split is for final evaluation. Use validation loss to select checkpoints and avoid tuning repeatedly against test metrics.
