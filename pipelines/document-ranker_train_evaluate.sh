#!/usr/bin/env bash
set -euo pipefail

# NOTE: This script requires an adjusted version of the 'create_rerank_data.py' script, one which takes a 'folds' argument
# With this, we can create the training and test data per fold, instead of having to construct all examples and then split

SKIP_TRAINING=false
SKIP_EVALUATION=false

# The name of the current variation
VARIATION="eranker_dranker"
QDER_DATA_DIR="${QDER_DATA_DIR:-ff_data}"
QDER_MODEL_DIR="${QDER_MODEL_DIR:-models}"

# Prepared Data
CORPUS="${QDER_DATA_DIR}/entity-linked.robust04.jsonl"
QUERIES="${QDER_DATA_DIR}/title.queries.tsv"
INITIAL_RANKING="${QDER_DATA_DIR}/title.bm25-rm3.run"
ENTITY_RANKING="${QDER_DATA_DIR}/entity_ranker-default.run"
ENTITY_EMBEDDINGS="${QDER_DATA_DIR}/mmead_embeddings.jsonl"

FOLDS="${QDER_DATA_DIR}/folds.json"
QRELS="${QDER_DATA_DIR}/qrels.robust04.txt"

FOLD_START=0
FOLD_COUNT=5
TRAIN_BATCH_SIZE=16
TEST_BATCH_SIZE=400

# Standard QDER Hyperparameters
USE_CUDA=true
BALANCE_TESTING=false
TEXT_ENCODER="bert"
SCORE_METHOD="bilinear"
EPOCHS=10


# Set up the optional flags
test_creation_flags=()
run_flags=()

if [[ "$BALANCE_TESTING" == true ]]; then
  test_creation_flags+=(--balance)
fi

if [[ "$USE_CUDA" == true ]]; then
  run_flags+=(--use-cuda)
fi

# Check input files
required_files=(
  "${CORPUS}"
  "${QUERIES}"
  "${INITIAL_RANKING}"
  "${ENTITY_RANKING}"
  "${ENTITY_EMBEDDINGS}"
  "${FOLDS}"
  "${QRELS}"
)

for f in "${required_files[@]}"; do
  [[ -f "$f" ]] || error "Missing file: $FILE"
done

# Prepare fold directories
for i in $(seq $FOLD_START $((FOLD_COUNT - 1))); do
  mkdir -p "${QDER_DATA_DIR}/fold-${i}"
  mkdir -p "${QDER_MODEL_DIR}/fold_${i}/${VARIATION}"
  mkdir -p "${QDER_DATA_DIR}/results/${VARIATION}"
done

# Data-creation, training, and testing fold-loop
for i in $(seq 0 $((FOLD_COUNT - 1))); do
  echo "+--- Started Fold ${i} --+"

  if [[ ! -f "${QDER_DATA_DIR}/fold-${i}/qrels.train.txt" ]]; then
    echo "Splitting QRELs for training and testing..."
    python scripts/data_preparation/split_run_by_fold.py --folds "$FOLDS" --save "${QDER_DATA_DIR}" --file "${QRELS}" --train "qrels.train.txt" --test "qrels.test.txt"
  fi

  if [[ "$SKIP_TRAINING" == false ]]; then
    echo "Constructing training data... (Fold ${i})"
    python scripts/data_preparation/create_rerank_data.py --queries "$QUERIES" --docs "$CORPUS" --qrels "${QDER_DATA_DIR}/fold-${i}/qrels.train.txt" --doc-run "$INITIAL_RANKING" --entity-run "$ENTITY_RANKING" --embeddings "$ENTITY_EMBEDDINGS" \
            --save "${QDER_DATA_DIR}/fold-${i}/train.${VARIATION}.jsonl" --folds "$FOLDS" --fold-index "$i" --train --balance

    echo "Training model... (Fold ${i})"
    python -m scripts.train --output-dir "${QDER_MODEL_DIR}/fold_${i}/${VARIATION}" --text-enc "${TEXT_ENCODER}" --use-scores --use-entities --score-method "${SCORE_METHOD}" --epochs "${EPOCHS}" \
            --train-data "${QDER_DATA_DIR}/fold-${i}/train.${VARIATION}.jsonl" --qrels "${QDER_DATA_DIR}/fold-${i}/qrels.train.txt" --batch-size "$TRAIN_BATCH_SIZE" \
            "${run_flags[@]}"
  fi

  if [[ "$SKIP_EVALUATION" == false ]]; then
    echo "Constructing test data... (Fold ${i})"
    python scripts/data_preparation/create_rerank_data.py --queries "$QUERIES" --docs "$CORPUS" --qrels "${QDER_DATA_DIR}/fold-${i}/qrels.test.txt" --doc-run "$INITIAL_RANKING" --entity-run "$ENTITY_RANKING" --embeddings "$ENTITY_EMBEDDINGS" \
            --save "${QDER_DATA_DIR}/fold-${i}/test.${VARIATION}.jsonl" --folds "$FOLDS" --fold-index "$i" \
            "${test_creation_flags[@]}"

    echo "Evaluating model... (Fold ${i})"
    # We pass the test data also using --train-data (this is because QDER's interface expects both, although only the test data is needed)
    python -m scripts.test --checkpoint "${QDER_MODEL_DIR}/fold_${i}/${VARIATION}/final_model.pt" --test-data "${QDER_DATA_DIR}/fold-${i}/test.${VARIATION}.jsonl" --train-data "${QDER_DATA_DIR}/fold-${i}/test.${VARIATION}.jsonl" \
          --qrels "${QDER_DATA_DIR}/fold-${i}/qrels.test.txt" --output-dir "${QDER_DATA_DIR}/results/${VARIATION}/" --save-run "${QDER_DATA_DIR}/results/${VARIATION}/qder.${VARIATION}.fold-${i}.run" \
          --metric map --use-entities --use-scores --score-method "${SCORE_METHOD}" --eval-batch-size "$TEST_BATCH_SIZE" \
          "${run_flags[@]}"
  fi

  echo "+--- Finished Fold ${i} ---+"
done
