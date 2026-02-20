INPUT ?= data/patients_sample_50.jsonl
OUTPUT ?= data/baseline_outputs.jsonl
REPORT ?= data/eval_report.json
DETERMINISM_REPORT ?= data/determinism_report.json
MODEL ?= gpt-4.1-mini
DATASET ?= iammustafatz/diabetes-prediction-dataset

.PHONY: prepare baseline evals determinism score all clean

prepare:
	uv run prepare_dataset.py \
		--dataset-slug $(DATASET) \
		--output data/patients.jsonl \
		--sample-output $(INPUT)

baseline:
	uv run run_baseline.py \
		--input $(INPUT) \
		--output $(OUTPUT) \
		--model $(MODEL)

evals:
	uv run run_evals.py \
		--input $(INPUT) \
		--outputs $(OUTPUT) \
		--report $(REPORT)

determinism:
	uv run run_evals.py \
		--determinism \
		--input $(INPUT) \
		--model $(MODEL) \
		--report $(DETERMINISM_REPORT)

score:
	@python3 -c 'import json; r=json.load(open("$(REPORT)")); s=(r.get("primary_score",{}) or {}).get("value_pct"); print(s if s is not None else r.get("local_metrics_summary",{}).get("aggregate_local_score_pct", 0.0))'

all: baseline evals determinism score

clean:
	rm -f data/patients.jsonl \
		data/patients_sample_50.jsonl \
		data/baseline_outputs.jsonl \
		data/eval_report.json \
		data/determinism_report.json
