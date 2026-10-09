# ===================================================================
# MIND Training Makefile
# ===================================================================
# Usage:
#   make train GPU=6 CONFIG=m                  # multidomain, default log
#   make train GPU=6 CONFIG=f                  # fast test (~10 min)
#   make train GPU=6 CONFIG=m LOG=my_run       # custom log name
#   make train GPU=auto CONFIG=f               # auto-pick least used GPU
#
# CONFIG shortcuts:
#   m  = pretraining_config_multidomain.yaml
#   f  = pretraining_config_fast.yaml
#   (or pass full filename without .yaml)
#
# Defaults: GPU=auto, CONFIG=m, LOG=train_<config>
# ===================================================================

SHELL := /bin/bash
CONDA_ROOT ?= /opt/miniconda3
MIND_CONDA_ENV ?= /home/yusuf/.conda/envs/mind_esa3d
MIND_HF_HOME ?= /home/yusuf/.cache/huggingface
CONDA_RUN := source "$(CONDA_ROOT)/etc/profile.d/conda.sh" && conda activate "$(MIND_CONDA_ENV)" && export PATH="$(MIND_CONDA_ENV)/bin:$$PATH" HF_HOME="$(MIND_HF_HOME)" HUGGINGFACE_HUB_CACHE="$(MIND_HF_HOME)/hub" TRANSFORMERS_CACHE="$(MIND_HF_HOME)/hub" &&

GPU ?= auto
CONFIG ?= m
LOG ?=
EPOCHS ?=
SEED ?=
CKPT ?=
MAX_ATOMS ?=
PYTEST_ARGS ?= tests --ignore=tests/test_multigpu_training.py

ifneq ($(strip $(EPOCHS)),)
  TRAIN_EPOCH_ARGS := --max-epochs $(EPOCHS)
endif
ifneq ($(strip $(SEED)),)
  TRAIN_SEED_ARGS := --seed $(SEED)
endif
ifneq ($(strip $(CKPT)),)
  TRAIN_CKPT_ARGS := --ckpt-path "$(CKPT)"
endif
ifneq ($(strip $(MAX_ATOMS)),)
  TRAIN_MAX_ATOMS_ARGS := --max-atoms-per-batch $(MAX_ATOMS)
endif

# Map single-letter shortcuts to full config names
ifeq ($(CONFIG),m)
  CONFIG_FILE := core/pretraining_config_multidomain.yaml
  CONFIG_TAG := multidomain
  CONFIG_IS_FAST := 0
else ifeq ($(CONFIG),f)
  CONFIG_FILE := core/pretraining_config_fast.yaml
  CONFIG_TAG := fast
  CONFIG_IS_FAST := 1
else
  CONFIG_FILE := core/$(CONFIG).yaml
  CONFIG_TAG := $(CONFIG)
  CONFIG_IS_FAST := 0
endif

# Log directory: logs/<MonthYear>/  e.g. logs/June2026/
LOG_MONTH_DIR := logs/$(shell date +%B%Y)

# Log filename: <name><datetime>.log or <name>fast<datetime>.log
_LOG_DATETIME := $(shell date +%Y%m%d_%H%M%S)
ifeq ($(LOG),)
  _LOG_BASE := $(CONFIG_TAG)
else
  _LOG_BASE := $(LOG)
endif
ifeq ($(CONFIG_IS_FAST),1)
  LOG_FILE := $(LOG_MONTH_DIR)/$(_LOG_BASE)fast_$(_LOG_DATETIME).log
else
  LOG_FILE := $(LOG_MONTH_DIR)/$(_LOG_BASE)_$(_LOG_DATETIME).log
endif

# A unique checkpoint directory is essential when independent validation runs
# share the machine. The old single default directory allowed concurrent jobs
# to overwrite each other's last.ckpt.
OUTPUT_DIR ?= outputs/$(_LOG_BASE)_$(_LOG_DATETIME)

MIN_VRAM_MB ?= 15000

# Resolve GPU: if "auto", pick GPU with most free memory (must have >= MIN_VRAM_MB free)
RESOLVED_GPU = $(shell if [ "$(GPU)" = "auto" ]; then \
	best=$$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null | \
		awk -F', ' '$$2 >= $(MIN_VRAM_MB) {print $$1, $$2}' | sort -k2 -rn | head -1 | cut -d' ' -f1); \
	if [ -z "$$best" ]; then echo "NONE"; else echo "$$best"; fi; \
	else echo "$(GPU)"; fi)

# ===================================================================
# Data Processing
# ===================================================================
# Raw protein/RNA structures -> edge-free fixed-slot chunks
# (data_loading/build_edge_free_chunks.py). No defaults and no overwrite: the
# destination must be new and every argument is explicit. Runs detached.
#   make build-edge-free DOMAIN=protein MANIFEST=<csv> RAW_DIR=<dir> \
#        DEST_PREFIX=<new prefix> CHUNK_SIZE=4000 WORKERS=32
#   Optional: SPLIT_GROUP=pdb-entry (RNA), BUILD_ARGS="--no-split --chunks 0"
# UniMol and COCONUT still use data_loading/process_chunked_dataset.py.
# ===================================================================

WORKERS ?= 32
SPLIT_GROUP ?= id

.PHONY: train test-unit stop status logs help build-edge-free

test-unit:
	$(CONDA_RUN) python -m pytest -q $(PYTEST_ARGS)

train:
	@if [ "$(RESOLVED_GPU)" = "NONE" ]; then \
		echo "ERROR: No GPU with >= $(MIN_VRAM_MB) MB free VRAM found."; \
		echo "Available GPUs:"; \
		nvidia-smi --query-gpu=index,memory.free,memory.total --format=csv,noheader 2>/dev/null; \
		echo ""; \
		echo "Use GPU=<number> to force a specific GPU, or wait for one to free up."; \
		exit 1; \
	fi
	@echo "=== MIND Training ==="
	@echo "  GPU:    $(RESOLVED_GPU) $(if $(filter auto,$(GPU)),(auto-selected))"
	@echo "  Config: $(CONFIG_FILE)"
	@echo "  Epochs: $(if $(strip $(EPOCHS)),$(EPOCHS),config default)"
	@echo "  Seed:   $(if $(strip $(SEED)),$(SEED),config default)"
	@echo "  Resume: $(if $(strip $(CKPT)),$(CKPT),no)"
	@echo "  Atoms:  $(if $(strip $(MAX_ATOMS)),$(MAX_ATOMS),config default)"
	@echo "  Log:    $(LOG_FILE)"
	@echo "  Output: $(OUTPUT_DIR)"
	@echo "====================="
	@mkdir -p $(LOG_MONTH_DIR)
	$(CONDA_RUN) CUDA_VISIBLE_DEVICES=$(RESOLVED_GPU) nohup python -m core.train_pretrain \
		--config-yaml-path $(CONFIG_FILE) \
		$(TRAIN_EPOCH_ARGS) \
		$(TRAIN_SEED_ARGS) \
		$(TRAIN_CKPT_ARGS) \
		$(TRAIN_MAX_ATOMS_ARGS) \
		--output-dir $(OUTPUT_DIR) \
		> $(LOG_FILE) 2>&1 &
	@echo "PID: $$!"
	@echo "Follow: tail -f $(LOG_FILE)"

stop:
	@echo "Killing training processes on GPU $(GPU)..."
	@ps aux | grep "CUDA_VISIBLE_DEVICES=$(GPU).*train_pretrain" | grep -v grep | awk '{print $$2}' | xargs -r kill
	@echo "Done."

status:
	@echo "=== MIND Training (all users) ==="
	@ps -eo user:12,pid,etime,args | grep '[t]rain_pretrain\|core[./]train' | grep -v grep || echo "  (no MIND training running)"
	@echo ""
	@echo "=== MIND Data Processing (all users) ==="
	@ps -eo user:12,pid,args | grep '[c]ache_to_pyg\|create_chunk' | grep -v grep | \
		awk '{print $$1}' | sort | uniq -c | \
		awk '{printf "  %s: %d worker processes\n", $$2, $$1}' || echo "  (none)"
	@echo ""
	@echo "=== GPU Usage ==="
	@nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || true
	@echo ""
	@echo "=== GPU Processes ==="
	@nvidia-smi --query-compute-apps=gpu_name,pid,used_gpu_memory,process_name --format=csv,noheader 2>/dev/null | \
		while IFS=, read gpu pid mem name; do \
		usr=$$(ps -o user= -p $$(echo $$pid | tr -d ' ') 2>/dev/null | tr -d ' '); \
		idx=$$(nvidia-smi --query-compute-apps=gpu_bus_id,pid --format=csv,noheader 2>/dev/null | \
			grep "$$(echo $$pid | tr -d ' ')" | head -1 | cut -d, -f1 | tr -d ' '); \
		gpuidx=$$(nvidia-smi --query-gpu=index,gpu_bus_id --format=csv,noheader 2>/dev/null | \
			grep "$$idx" | head -1 | cut -d, -f1 | tr -d ' '); \
		printf "  GPU %-2s | %-14s | PID %-8s |%s\n" "$${gpuidx:-?}" "$${usr:-?}" "$$pid" "$$mem"; \
	done 2>/dev/null || true

build-edge-free:
	@test -n "$(DOMAIN)" -a -n "$(MANIFEST)" -a -n "$(RAW_DIR)" -a -n "$(DEST_PREFIX)" -a -n "$(CHUNK_SIZE)" || { \
		echo "ERROR: set DOMAIN (protein|rna), MANIFEST, RAW_DIR, DEST_PREFIX and CHUNK_SIZE"; exit 1; }
	@echo "=== MIND edge-free chunk build ==="
	@echo "  Domain: $(DOMAIN)   Manifest: $(MANIFEST)"
	@echo "  Raw:    $(RAW_DIR)"
	@echo "  Dest:   $(DEST_PREFIX)_*   Chunk size: $(CHUNK_SIZE)   Workers: $(WORKERS)"
	@echo "  Log:    $(LOG_MONTH_DIR)/build_$(DOMAIN)_$(_LOG_DATETIME).log"
	@mkdir -p $(LOG_MONTH_DIR)
	$(CONDA_RUN) setsid nohup python -u -m data_loading.build_edge_free_chunks \
		--domain $(DOMAIN) --manifest "$(MANIFEST)" --raw-dir "$(RAW_DIR)" \
		--dest-prefix "$(DEST_PREFIX)" --chunk-size $(CHUNK_SIZE) --workers $(WORKERS) \
		--split-group $(SPLIT_GROUP) $(BUILD_ARGS) \
		> $(LOG_MONTH_DIR)/build_$(DOMAIN)_$(_LOG_DATETIME).log 2>&1 < /dev/null &
	@echo "Follow: tail -f $(LOG_MONTH_DIR)/build_$(DOMAIN)_$(_LOG_DATETIME).log"

logs:
	@find logs/ -name "*.log" -printf "%T@ %p\n" 2>/dev/null | sort -rn | head -10 | awk '{print $$2}' || echo "No log files found."

help:
	@echo "MIND Commands:"
	@echo ""
	@echo "  Training:"
	@echo "    make train CONFIG=m LOG=run1         Full multidomain training"
	@echo "    make train CONFIG=f                  Fast sanity check"
	@echo "    make train CONFIG=f EPOCHS=1         Override config epoch count"
	@echo "    make train CONFIG=f SEED=43          Independent seeded run"
	@echo "    make train CONFIG=m CKPT=path EPOCHS=101  Resume full checkpoint state"
	@echo "    make train CONFIG=m MAX_ATOMS=18000      Override dynamic atom budget"
	@echo "    make test-unit                       Run the complete unit-test suite"
	@echo "    make stop GPU=6                      Kill training on GPU 6"
	@echo "    make status                          Show all processes + GPUs"
	@echo "    make logs                            List recent log files"
	@echo ""
	@echo "  Data Processing:"
	@echo "    make build-edge-free DOMAIN=protein MANIFEST=.. RAW_DIR=.. DEST_PREFIX=.. CHUNK_SIZE=4000"
	@echo "                                         Raw -> edge-free chunks (detached, no defaults)"
	@echo ""
	@echo "  CONFIG shortcuts: m=multidomain, f=fast"
