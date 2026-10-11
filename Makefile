# Machine-specific settings (interpreter, model cache) live in local.mk, e.g.
#   PYTHON = D:/venvs/gst-copilot/Scripts/python.exe
#   export HF_HOME = D:/hf-cache
-include local.mk
PYTHON ?= python
SPLIT ?= dev

.PHONY: ingest amendments version-rule36 version-rules core-provisions eval eval-retrieval \
	templated-items split tune-abstention eval-generation eval-generation-dry test

ingest:
	$(PYTHON) -m ingest.act_snapshot

amendments:
	$(PYTHON) -m ingest.notifications

version-rule36: amendments
	$(PYTHON) -m ingest.rule_versions

version-rules: amendments
	$(PYTHON) -m ingest.all_rule_versions

core-provisions:
	$(PYTHON) -m ingest.core_provisions

eval:
	$(PYTHON) -m eval.harness

eval-retrieval:
	$(PYTHON) -m eval.retrieval

# Templated eval items from verified data, then the dev/test split (hand-written items keep
# their assignment).
templated-items:
	$(PYTHON) -m eval.templated

split: templated-items
	$(PYTHON) -m eval.split

tune-abstention:
	$(PYTHON) -m eval.tune_abstention

# Needs GEMINI_API_KEY and GROQ_API_KEY in .env. Dev by default; `make eval-generation SPLIT=test`
# runs the test split (once).
eval-generation:
	$(PYTHON) -m eval.generation --split $(SPLIT)

eval-generation-dry:
	$(PYTHON) -m eval.generation --split $(SPLIT) --dry-run

test:
	$(PYTHON) -m pytest