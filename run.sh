#!/bin/bash

module add gcc
module add cmake

chmod +x ./scripts/run-production.sh
chmod +x ./scripts/build-production.sh

nice -n 19 ./scripts/run-production.sh

echo "Confirmation by the provided python evaluator:"

python3 evaluator/evaluate.py test_predictions.csv ./data/fashion_mnist_test_labels.csv
