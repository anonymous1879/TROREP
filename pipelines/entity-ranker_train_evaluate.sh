#!/usr/bin/env bash
set -euo pipefail

# NOTE: This script requires an adjusted version of the 'create_rerank_data.py' script, one which takes a 'folds' argument
# With this, we can create the training and test data per fold, instead of having to construct all examples and then split

SKIP_TRAINING=false
SKIP_EVALUATION=false

# The name of the current variation
VARIATION="entity_ranker"
VARIATION_SUFFIX="-default"
QDER_DATA_DIR="${QDER_DATA_DIR:-data}"
QDER_MODEL_DIR="${QDER_MODEL_DIR:-models}"

# Prepared Data
CORPUS="${QDER_DATA_DIR}/entity-linked.robust04.jsonl"
QUERIES="${QDER_DATA_DIR}/title.queries.tsv"

FOLDS="${QDER_DATA_DIR}/folds.json"
QRELS="${QDER_DATA_DIR}/qrels.robust04.txt"
ENTITY_QRELS="${QDER_DATA_DIR}/entity-qrels.robust04.txt"
ENTITY_DESCRIPTIONS="${QDER_DATA_DIR}/entity_descriptions.jsonl"

FOLD_START=0
FOLD_COUNT=5
TRAIN_BATCH_SIZE=16
TEST_BATCH_SIZE=400

# Standard QDER Hyperparameters
USE_CUDA=true
TEXT_ENCODER="bert"
SCORE_METHOD="bilinear"
EPOCHS=10


# Set up the optional flags
run_flags=()

if [[ "$USE_CUDA" == true ]]; then
  run_flags+=(--use-cuda)
fi

# Check input files
required_files=(
  "${CORPUS}"
  "${QUERIES}"
  "${ENTITY_DESCRIPTIONS}"
  "${FOLDS}"
  "${QRELS}"
)

for f in "${required_files[@]}"; do
  if [[ ! -f "$f" ]]; then
    echo "Missing file: $f"
    exit 1
  fi
done

# Prepare fold directories
for i in $(seq $FOLD_START $((FOLD_COUNT - 1))); do
  mkdir -p "${QDER_DATA_DIR}/fold-${i}"
  mkdir -p "${QDER_MODEL_DIR}/fold-${i}/${VARIATION}"
  mkdir -p "${QDER_DATA_DIR}/results/${VARIATION}"
done

# Starting venv
python -m venv .venv --system-site-packages
source .venv/bin/activate
pip install -r requirements.txt

if [[ ! -f "${ENTITY_QRELS}" ]]; then
  echo "Creating entity QRELs"
  python scripts/data_preparation/make_entity_qrels.py --qrels "${QRELS}" --docs "${CORPUS}" --save "${ENTITY_QRELS}"
fi

# Data-creation, training, and testing fold-loop
for i in $(seq $FOLD_START $((FOLD_COUNT - 1))); do
  echo "+--- Started Fold ${i} --+"

  if [[ ! -f "${QDER_DATA_DIR}/fold-${i}/entity-qrels.train.txt" ]]; then
    echo "Splitting entity QRELs for training and testing..."
    python -m scripts.data_preparation.split_run_by_fold --folds "$FOLDS" --save "${QDER_DATA_DIR}" --file "${ENTITY_QRELS}" --train "entity-qrels.train.txt" --test "entity-qrels.test.txt"
  fi

  if [[ "$SKIP_TRAINING" == false ]]; then
    echo "Constructing training data... (Fold ${i})"
    python -m scripts.data_preparation.make_entity_data --qrels "${ENTITY_QRELS}" --queries "${QUERIES}" --save "${QDER_DATA_DIR}/fold-${i}/train.${VARIATION}.jsonl" --desc "${ENTITY_DESCRIPTIONS}" \
            --folds "$FOLDS" --fold-index "$i" --train

    echo "Training model... (Fold ${i})"
    python -m scripts.train --output-dir "${QDER_MODEL_DIR}/fold-${i}/${VARIATION}" --text-enc "${TEXT_ENCODER}" --use-scores --use-entities --score-method "${SCORE_METHOD}" --epochs "${EPOCHS}" \
            --train-data "${QDER_DATA_DIR}/fold-${i}/train.${VARIATION}.jsonl" --qrels "${QDER_DATA_DIR}/fold-${i}/entity-qrels.train.txt" --batch-size "$TRAIN_BATCH_SIZE" \
            "${run_flags[@]}" --save-every 5

    # Clean the training data to save storage on cluster
    rm "${QDER_DATA_DIR}/fold-${i}/train.${VARIATION}.jsonl"
  fi

  if [[ "$SKIP_EVALUATION" == false ]]; then
    echo "Constructing testing data... (Fold ${i})"
    python -m scripts.data_preparation.make_entity_data --qrels "${ENTITY_QRELS}" --queries "${QUERIES}" --save "${QDER_DATA_DIR}/fold-${i}/test.${VARIATION}.jsonl" --desc "${ENTITY_DESCRIPTIONS}" \
            --folds "$FOLDS" --fold-index "$i"

    echo "Evaluating model... (Fold ${i})"
    # We pass the test data also using --train-data (this is because QDER's interface expects both, although only the test data is needed, it's just there to fill the requirements without changing the script provided)
    python -m scripts.test --checkpoint "${QDER_MODEL_DIR}/fold-${i}/${VARIATION}/final_model.pt" --test-data "${QDER_DATA_DIR}/fold-${i}/test.${VARIATION}.jsonl" --train-data "${QDER_DATA_DIR}/fold-${i}/test.${VARIATION}.jsonl" \
          --qrels "${QDER_DATA_DIR}/fold-${i}/entity-qrels.test.txt" --output-dir "${QDER_DATA_DIR}/results/${VARIATION}/" --save-run "${QDER_DATA_DIR}/results/${VARIATION}${VARIATION_SUFFIX}/qder.${VARIATION}${VARIATION_SUFFIX}.fold-${i}.run" \
          --metric map --use-entities --use-scores --score-method "${SCORE_METHOD}" --eval-batch-size "$TEST_BATCH_SIZE" \
          "${run_flags[@]}"

    # Clean the testing data to save storage on cluster
    rm "${QDER_DATA_DIR}/fold-${i}/test.${VARIATION}.jsonl"
  fi

  echo "+--- Finished Fold ${i} ---+"
done
