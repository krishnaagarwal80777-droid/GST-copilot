# Machine-specific settings (interpreter, model cache) live in local.mk, e.g.
#   PYTHON = D:/venvs/gst-copilot/Scripts/python.exe
#   export HF_HOME = D:/hf-cache
-include local.mk
PYTHON ?= python

.PHONY: ingest amendments version-rule36 version-rules core-provisions eval eval-retrieval test

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

test:
	$(PYTHON) -m pytest