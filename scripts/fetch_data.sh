#!/usr/bin/env bash
# Downloads the human-annotated corpora used for training and evaluation into data/.
# Licences: UNER (CC BY-SA 4.0), GermEval 2014 (CC BY 4.0), WNUT 2017 (CC BY 4.0), Broad Twitter Corpus (CC BY 4.0), Presidio synth (MIT).
set -euo pipefail
mkdir -p data && cd data
[ -d UNER_English-EWT ] || git clone --depth 1 https://github.com/UniversalNER/UNER_English-EWT
[ -d UNER_German-PUD ]  || git clone --depth 1 https://github.com/UniversalNER/UNER_German-PUD
if [ ! -d NER-datasets ]; then
  git clone --depth 1 --filter=blob:none --sparse https://github.com/davidsbatista/NER-datasets
  (cd NER-datasets && git sparse-checkout set GermEval2014 WNUT17)
fi
[ -d broad_twitter_corpus ] || git clone --depth 1 https://github.com/GateNLP/broad_twitter_corpus
mkdir -p presidio-research
[ -f presidio-research/synth_dataset_v2.json ] || curl -sSL -o presidio-research/synth_dataset_v2.json \
  https://raw.githubusercontent.com/microsoft/presidio-research/master/data/synth_dataset_v2.json
[ -f presidio-research/templates.txt ] || curl -sSL -o presidio-research/templates.txt \
  https://raw.githubusercontent.com/microsoft/presidio-research/master/presidio_evaluator/data_generator/raw_data/templates.txt
