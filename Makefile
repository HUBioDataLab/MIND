# ===================================================================
# MIND Training Makefile
# ===================================================================
# Usage:
#   make train GPU=6 CONFIG=m                  # four-domain run, default log
#   make train GPU=6 CONFIG=f                  # fast test
#   make train GPU=6 CONFIG=m LOG=my_run       # custom log name
#   make train GPU=auto CONFIG=f               # auto-pick least used GPU
#
# CONFIG shortcuts:
#   m  = pretraining_config_multidomain_1m_coconut.yaml   (four domains, latest)
#   f  = pretraining_config_fast.yaml
#   (or pass full filename without .yaml)
#
# Defaults: GPU=auto, CONFIG=m, LOG=train_<config>
# ===================================================================

SHELL := /bin/bash
CONDA_ROOT ?= /opt/miniconda3
MIND_CONDA_ENV ?= /home/yusuf/.conda/envs/mind_esa3d
MIND_HF_HOME ?= /home/yusuf/models/huggingface
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
  CONFIG_FILE := core/pretraining_config_multidomain_1m_coconut.yaml
  CONFIG_TAG := multidomain_1m_coconut
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
# Single-domain configs (pretraining_config_<domain>_only) write under outputs/<domain>_only/.
RUN_GROUP := $(if $(filter pretraining_config_%_only,$(CONFIG)),$(patsubst pretraining_config_%,%,$(CONFIG))/,)
OUTPUT_DIR ?= outputs/$(RUN_GROUP)$(_LOG_BASE)_$(_LOG_DATETIME)

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

.PHONY: train test-unit stop status logs help build-edge-free check-epoch-plan eval-pretrain eval-counterfactual

PLAN_EPOCHS ?= 5

test-unit:
	$(CONDA_RUN) python -m pytest -q $(PYTEST_ARGS)

# Natural batch plan of the first PLAN_EPOCHS epochs from chunk metadata (CPU, reads only);
# fails when a fixed epoch_num_batches would be exceeded. Run it before a long run.
# The full output goes to logs/<MonthYear>/check_epoch_plan_*.log; the terminal gets the
# PLAN_SUMMARY line and the log path (plus the log tail on failure).
check-epoch-plan:
	@mkdir -p $(LOG_MONTH_DIR); \
	plan_log=$(LOG_MONTH_DIR)/check_epoch_plan_$(CONFIG_TAG)_$(_LOG_DATETIME).log; \
	$(CONDA_RUN) CUDA_VISIBLE_DEVICES="" WANDB_MODE=disabled nice -n 19 python -m core.check_epoch_plan \
		--config-yaml-path $(CONFIG_FILE) --epochs $(PLAN_EPOCHS) $(TRAIN_SEED_ARGS) > $$plan_log 2>&1; \
	rc=$$?; \
	grep -E '^PLAN_SUMMARY: ' $$plan_log; \
	if [ $$rc -ne 0 ]; then tail -n 20 $$plan_log >&2; fi; \
	echo "log: $$plan_log"; \
	exit $$rc

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
	@echo "=== MIND Downstream / Evaluation (all users) ==="
	@out=$$(ps -eo user:12,pid,etime,args | grep '[d]ownstream\.\(embed\|probe\|finetune\|report\)\|[c]ore\.evaluate_pretrain' | \
		sed -E 's/--threads [0-9]+ //' | cut -c1-150); \
		[ -n "$$out" ] && echo "$$out" || echo "  (none)"
	@echo ""
	@echo "=== Chain scripts (run_*_chain.sh, run_*_finetune.sh) ==="
	@out=$$(ps -eo user:12,pid,etime,args | grep '[r]un_[a-z_]*\.sh' | cut -c1-120); \
		[ -n "$$out" ] && echo "$$out" || echo "  (none)"
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
	@nvidia-smi --query-compute-apps=gpu_bus_id,pid,used_gpu_memory --format=csv,noheader 2>/dev/null | \
		while IFS=, read bus pid mem; do \
		bus=$$(echo $$bus | tr -d ' '); pid=$$(echo $$pid | tr -d ' '); \
		usr=$$(ps -o user= -p $$pid 2>/dev/null | tr -d ' '); \
		gpuidx=$$(nvidia-smi --query-gpu=index,gpu_bus_id --format=csv,noheader 2>/dev/null | \
			grep -F "$$bus" | head -1 | cut -d, -f1 | tr -d ' '); \
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

# Offline evaluation of a frozen checkpoint (GPU workload, user-started). Masks and the batch
# plan depend only on the data selection, split and seed, so checkpoints evaluated with the same
# EVAL_CONFIG see identical masks. OUT must be a new directory.
#   make eval-pretrain CKPT=<ckpt> SPLIT=test SEED=0 GPU=<id> OUT=<new dir> \
#        [EVAL_CONFIG=pretraining_config_multidomain_1m_coconut] [DOMAINS=protein,rna] [MAX_BATCHES=N]
#        [COUNTERFACTUAL=<mode> [CF_SIGMA=<Angstrom>]]
# COUNTERFACTUAL (G1 step 0, core/counterfactual_eval.py) corrupts what the encoder/ESA see of the
# masked UniMol atoms, evaluation only: masked_pos_noise (CF_SIGMA, default 1.0), masked_pos_random,
# masked_dist_shuffle, esa_edges_remove, esa_edges_shuffle, masked_edges_remove. Unset/none = unchanged.
eval-pretrain:
	@test -n "$(CKPT)" -a -n "$(SPLIT)" -a -n "$(OUT)" || { echo "ERROR: set CKPT, SPLIT and OUT"; exit 1; }
	@if [ "$(RESOLVED_GPU)" = "NONE" ]; then echo "ERROR: No GPU with >= $(MIN_VRAM_MB) MB free VRAM found."; exit 1; fi
	@mkdir -p $(LOG_MONTH_DIR)
	$(CONDA_RUN) $(DS_ENV) CUDA_VISIBLE_DEVICES=$(RESOLVED_GPU) nohup python -u -m core.evaluate_pretrain --threads $(DS_THREADS) \
		--ckpt "$(CKPT)" --split $(SPLIT) --seed $(or $(SEED),0) --output-dir "$(OUT)" --device cuda:0 \
		$(if $(strip $(EVAL_CONFIG)),--config-yaml-path core/$(EVAL_CONFIG).yaml,) \
		$(if $(strip $(DOMAINS)),--domains $(DOMAINS),) $(if $(strip $(MAX_BATCHES)),--max-batches $(MAX_BATCHES),) \
		$(if $(strip $(COUNTERFACTUAL)),--counterfactual $(COUNTERFACTUAL),) \
		$(if $(strip $(CF_SIGMA)),--counterfactual-sigma $(CF_SIGMA),) \
		> $(LOG_MONTH_DIR)/eval_pretrain_$(SPLIT)_$(_LOG_DATETIME).log 2>&1 &
	@echo "GPU: $(RESOLVED_GPU)   Log: $(LOG_MONTH_DIR)/eval_pretrain_$(SPLIT)_$(_LOG_DATETIME).log"

# G1 step 0: UniMol counterfactual sweep on one frozen checkpoint (GPU workload, user-started).
# Runs the modes one after another in one detached shell on one GPU; each writes OUT_ROOT/<mode>/
# (OUT_ROOT/<mode> must not exist). The loop stops at the first failing mode.
#   make eval-counterfactual CKPT=<ckpt> SPLIT=test SEED=0 GPU=<id> OUT_ROOT=<new dir> \
#        [CF_MODES="none masked_pos_noise ..."] [DOMAINS=unimol] [MAX_BATCHES=N]
CF_MODES ?= none masked_pos_noise masked_pos_random masked_dist_shuffle esa_edges_remove esa_edges_shuffle masked_edges_remove
eval-counterfactual:
	@test -n "$(CKPT)" -a -n "$(SPLIT)" -a -n "$(OUT_ROOT)" || { echo "ERROR: set CKPT, SPLIT and OUT_ROOT"; exit 1; }
	@if [ "$(RESOLVED_GPU)" = "NONE" ]; then echo "ERROR: No GPU with >= $(MIN_VRAM_MB) MB free VRAM found."; exit 1; fi
	@mkdir -p $(LOG_MONTH_DIR)
	$(CONDA_RUN) $(DS_ENV) CUDA_VISIBLE_DEVICES=$(RESOLVED_GPU) nohup bash -c 'set -e; for mode in $(CF_MODES); do \
		echo "=== counterfactual $$mode ==="; \
		python -u -m core.evaluate_pretrain --threads $(DS_THREADS) --ckpt "$(CKPT)" --split $(SPLIT) \
			--seed $(or $(SEED),0) --output-dir "$(OUT_ROOT)/$$mode" --device cuda:0 \
			--domains $(or $(DOMAINS),unimol) --counterfactual $$mode \
			$(if $(strip $(EVAL_CONFIG)),--config-yaml-path core/$(EVAL_CONFIG).yaml,) \
			$(if $(strip $(MAX_BATCHES)),--max-batches $(MAX_BATCHES),); \
		done' > $(LOG_MONTH_DIR)/eval_counterfactual_$(SPLIT)_$(_LOG_DATETIME).log 2>&1 &
	@echo "GPU: $(RESOLVED_GPU)   Modes: $(CF_MODES)   Log: $(LOG_MONTH_DIR)/eval_counterfactual_$(SPLIT)_$(_LOG_DATETIME).log"

# ===================================================================
# Downstream evaluation (downstream/README.md)
# ===================================================================
# Data steps write under /home/yusuf/data/downstream/<task>/; run them only on an explicit
# decision. embed and finetune are GPU workloads, probe trains small heads on CPU; all three
# are user-started. Snapshots, embeddings and results go under outputs/ (MIND_OUTPUTS_ROOT
# overrides it).
#   make ds-snapshot CKPT=<ckpt> NAME=<snapshot> [ALIAS=current_best]
#   make ds-download TASK=ec            make ds-process TASK=ec WORKERS=16
#   make ds-download TASK=lba           make ds-process TASK=lba [TAGS="identity30 identity60"]
#   make ds-download TASK=go_mf         (the three GO branches share one download and one processed file;
#   make ds-validate TASK=go_mf          ds-process TASK=go_mf WORKERS=16 builds all of them; validate is read-only)
#   make ds-download TASK=fold         (GearNet Fold3D, Zenodo 7593591, 670,898,225 B; md5 and size are pinned)
#   make ds-validate TASK=fold          ds-process TASK=fold WORKERS=16     (probe: one accuracy per test split)
#   make ds-download TASK=reaction      (IEConv ProtFunct; Google Drive archive, size/md5 not published: the first
#   make ds-validate TASK=reaction       download stops until archive_sha256/archive_bytes are pinned in task.yaml)
#   make ds-leakage LEAK_TASK=go REFERENCE=protein_1m NAMESPACE=pdb_chain PROVENANCE=<task dir>/provenance.json \
#        QUERIES="valid=<dir>/splits/valid.txt test=<dir>/splits/test.txt" [ID_MAP=<pdb_chain->uniprot tsv>]
#   make ds-process TASK=mn_esol        (MoleculeNet mn_bbbp|bace|clintox|tox21|toxcast|sider|hiv|pcba|muv|esol|freesolv|lipo|qm7|qm8;
#   make ds-validate TASK=mn_esol        the Uni-Mol LMDB archive is already extracted, so there is no ds-download; validate
#                                        reads and checks without writing; probe/report work as for the other tasks)
#   make ds-embed TASK=ec MODEL=<snapshot|alias|esm2_t6_8M_UR50D|saprot_650m_pdb> GPU=<id> [RANDOM_INIT=1]
#   make ds-probe TASK=ec MODEL_ID=<id> HEAD=linear|mlp [POOLING=mean_residues] [SEEDS="0 1 2"]
#   make ds-finetune TASK=ec MODEL=<snapshot> GPU=<id> [RANDOM_INIT=1]
#   make ds-report TASK=ec
# LBA adds SPLIT_TAG=identity30|identity60 and REPRESENTATION=joint|separate to
# ds-embed and ds-probe (ds-finetune takes SPLIT_TAG only; it is joint-only).
# Results ledger (append-only JSONL + generated reports under MIND_RESULTS_ROOT, outside Git; CPU only;
# reads outputs, never writes them; ds-report is separate and unchanged):
#   make ds-ledger-add OUTPUTS_ROOT=<outputs root> [EVAL_DIRS="<eval dir> ..."]   (re-running adds nothing new)
#   make ds-ledger-annotate SELECT="task=ec data_revision=<rev>" STATUS=superseded REASON="..." [REPLACED_BY=<rev>]
#   make ds-ledger-report [MODELS="<a> <b>"]   make ds-summary   make ds-missing   make ds-ledger-verify
#   make ds-compare A=<model> B=<model> [TASKS="ec lba"] [BOOTSTRAP=1000]
#   make ds-ledger-recompute MODEL=<model> CONTROL=<model with macro AUPR> DATA_REVISION=<rev> [TASK=ec] [DRY_RUN=1]
#        (macro AUPR a probe did not record, from its saved predictions.npz; four checks, nothing written on failure)
# MODELS defaults to summary_models in $(MIND_RESULTS_ROOT)/comparisons.yaml, else the list in ledger_report.py.
# One command for one MIND model (downstream/run_all.py; GPU embeds + CPU probes, user-started, no fine-tune):
#   make ds-run MODEL=<snapshot | path/to.ckpt> [NAME=<snapshot name, required with a ckpt>] [TASKS=all|ec,go_mf,...]
#        GPU=2,3 [PER_GPU=1] [HEADS=linear,mlp] [PROBE_JOBS=1] [RANDOM_INIT=1] [OUTPUTS_ROOT=<dir>] [PLAN=1]
#        [COMPARE_TO=<snapshot[.random_init],...> [BOOTSTRAP=1000]] [COUNTERFACTUAL=1 [CF_MODES=...] [EVAL_CONFIG=...]]
#   snapshot (ckpt) -> embeds on the GPUs (every model) -> [counterfactual sweep] -> probes (SEEDS) -> ledger add ->
#   final metrics + missing cells per model -> ds-compare A=<each COMPARE_TO> B=MODEL -> ledger report.
#   PLAN=1 prints what would run and writes nothing. ds-run and ds-probe run detached by default (setsid + nohup:
#   closing the terminal does not stop them; the log ends with EXIT_CODE=<n>; make ds-status lists the logs);
#   FOREGROUND=1 runs them in the terminal as before.
# ===================================================================

TASK ?= ec
HEAD ?= linear
SEEDS ?= 0 1 2
POOLING ?=
# CPU threads per downstream/eval process (unbounded torch threads slowed a live run on 25 Oct)
DS_THREADS ?= 8
DS_ENV := OMP_NUM_THREADS=$(DS_THREADS) OPENBLAS_NUM_THREADS=$(DS_THREADS) MKL_NUM_THREADS=$(DS_THREADS)
DS_LOG = $(LOG_MONTH_DIR)/ds_$(TASK)_$(1)_$(_LOG_DATETIME).log
# Long ds-* jobs (ds-probe, ds-run) run in their own session (setsid + nohup, no terminal): closing the SSH/terminal
# connection does not stop them. $(1) = log, $(2) = command; the log ends with EXIT_CODE=<n> of the command
# (non-zero when the Python step fails, as pipefail gave in the foreground). FOREGROUND=1 = the old foreground run.
# The terminal prints only the log path; follow the job with `make ds-status`.
DS_DETACH = log=$(1); echo "Log: $$log"; \
	setsid nohup bash -c '$(2); code=$$?; echo "EXIT_CODE=$$code"; exit $$code' > $$log 2>&1 < /dev/null &
DS_RANDOM_INIT_ARGS := $(if $(filter 1,$(RANDOM_INIT)),--random-init,)
# Fine-tune overrides: a random-init encoder needs a far larger LR than a pretrained one.
FT_ARGS ?=
DS_HF_ARGS := $(if $(filter esm2_% saprot_%,$(MODEL)),--hf-cache-dir $(MIND_HF_HOME)/hub,)
# LBA-only selectors; empty for every other task so their CLIs never see them.
SPLIT_TAG ?=
REPRESENTATION ?=
TAGS ?=
DS_SPLIT_TAG_ARGS := $(if $(strip $(SPLIT_TAG)),--split-tag $(SPLIT_TAG),)
DS_LBA_ARGS := $(DS_SPLIT_TAG_ARGS) $(if $(strip $(REPRESENTATION)),--representation $(REPRESENTATION),)
# ec prepare parallelizes structure parsing with --workers; lba reads one LMDB
# sequentially and only bounds its torch threads.
# GO branches (go_mf, go_bp, go_cc) share one prepare module.
# MoleculeNet tasks (mn_*) share the molnet prepare module, which takes --task and reads the LMDB files serially.
DS_PREPARE_MODULE = $(if $(filter go_%,$(TASK)),go,$(if $(filter mn_%,$(TASK)),molnet,$(TASK)))
DS_MOLNET_ARGS = $(if $(filter mn_%,$(TASK)),--task $(TASK),)
DS_PROCESS_ARGS = $(if $(filter mn_%,$(TASK)),$(DS_MOLNET_ARGS),$(if $(filter lba,$(TASK)),--threads $(DS_THREADS) $(if $(strip $(TAGS)),--tags $(TAGS),),--workers $(WORKERS)))
DS_DOWNLOAD_ARGS = $(if $(filter mn_%,$(TASK)),$(DS_MOLNET_ARGS),$(if $(filter lba,$(TASK)),--threads $(DS_THREADS) $(if $(strip $(TAGS)),--tags $(TAGS),),))

.PHONY: ds-snapshot ds-download ds-process ds-validate ds-leakage ds-embed ds-probe ds-finetune ds-report test-downstream \
	ds-ledger-add ds-ledger-annotate ds-ledger-report ds-summary ds-missing ds-compare ds-ledger-verify ds-ledger-recompute \
	ds-run ds-status

ds-snapshot:
	@test -n "$(CKPT)" -a -n "$(NAME)" || { echo "ERROR: set CKPT and NAME"; exit 1; }
	$(CONDA_RUN) python -m downstream.common.registry snapshot "$(CKPT)" "$(NAME)"
	$(if $(strip $(ALIAS)),$(CONDA_RUN) python -m downstream.common.registry alias "$(ALIAS)" "$(NAME)" --repoint,)

# `set -o pipefail`: without it the pipeline exit status is tee's (always 0) and a failed prepare/probe looks like success.
ds-download ds-process:
	@mkdir -p $(LOG_MONTH_DIR)
	set -o pipefail; $(CONDA_RUN) $(DS_ENV) CUDA_VISIBLE_DEVICES="" python -u -m downstream.tasks.$(DS_PREPARE_MODULE).prepare $(patsubst ds-%,%,$@) \
		$(if $(filter ds-process,$@),$(DS_PROCESS_ARGS),$(DS_DOWNLOAD_ARGS)) 2>&1 | tee $(call DS_LOG,$(patsubst ds-%,%,$@))

# Read-only: re-checks labels and split lists against task.yaml (go, fold, reaction).
ds-validate:
	$(CONDA_RUN) $(DS_ENV) CUDA_VISIBLE_DEVICES="" python -u -m downstream.tasks.$(DS_PREPARE_MODULE).prepare validate $(DS_MOLNET_ARGS)

# Leakage registry: compares task split ids with a pretraining reference (CPU, read-only on the references;
# appends a record to the task's provenance.json and writes a per-id TSV next to it).
ds-leakage:
	@test -n "$(LEAK_TASK)" -a -n "$(REFERENCE)" -a -n "$(NAMESPACE)" -a -n "$(PROVENANCE)" -a -n "$(QUERIES)" || \
		{ echo "ERROR: set LEAK_TASK, REFERENCE, NAMESPACE, PROVENANCE and QUERIES"; exit 1; }
	$(CONDA_RUN) $(DS_ENV) CUDA_VISIBLE_DEVICES="" python -u -m downstream.common.leakage --task "$(LEAK_TASK)" \
		--reference "$(REFERENCE)" --query-namespace "$(NAMESPACE)" --provenance "$(PROVENANCE)" \
		$(foreach q,$(QUERIES),--query $(q)) $(if $(strip $(ID_MAP)),--id-map $(ID_MAP),)

ds-embed ds-finetune:
	@test -n "$(MODEL)" || { echo "ERROR: set MODEL (snapshot name, alias or ESM-2 name)"; exit 1; }
	@if [ "$(RESOLVED_GPU)" = "NONE" ]; then echo "ERROR: No GPU with >= $(MIN_VRAM_MB) MB free VRAM found."; exit 1; fi
	@mkdir -p $(LOG_MONTH_DIR)
	$(CONDA_RUN) $(DS_ENV) CUDA_VISIBLE_DEVICES=$(RESOLVED_GPU) nohup python -u -m downstream.$(patsubst ds-%,%,$@) \
		--threads $(DS_THREADS) --task $(TASK) --model "$(MODEL)" --device cuda:0 $(DS_RANDOM_INIT_ARGS) \
		$(if $(filter ds-embed,$@),$(DS_HF_ARGS) $(DS_LBA_ARGS),--seeds $(SEEDS) $(DS_SPLIT_TAG_ARGS) $(FT_ARGS)) \
		> $(call DS_LOG,$(patsubst ds-%,%,$@)) 2>&1 &
	@echo "GPU: $(RESOLVED_GPU)   Log: $(call DS_LOG,$(patsubst ds-%,%,$@))"

DS_PROBE_CMD = $(CONDA_RUN) $(DS_ENV) CUDA_VISIBLE_DEVICES="" python -u -m downstream.probe --threads $(DS_THREADS) \
	--task $(TASK) --model-id "$(MODEL_ID)" --head $(HEAD) --seeds $(SEEDS) $(if $(strip $(POOLING)),--pooling $(POOLING),) \
	$(DS_LBA_ARGS)

ds-probe:
	@test -n "$(MODEL_ID)" || { echo "ERROR: set MODEL_ID (snapshot, <snapshot>.random_init or ESM-2 name)"; exit 1; }
	@mkdir -p $(LOG_MONTH_DIR)
ifeq ($(FOREGROUND),1)
	set -o pipefail; $(DS_PROBE_CMD) 2>&1 | tee $(call DS_LOG,probe_$(HEAD))
else
	$(call DS_DETACH,$(call DS_LOG,probe_$(HEAD)),$(DS_PROBE_CMD))
endif

ds-report:
	$(CONDA_RUN) python -m downstream.report --task $(TASK)

MIND_RESULTS_ROOT ?= /home/yusuf/mind_results
BOOTSTRAP ?= 1000
LEDGER_ENV = MIND_RESULTS_ROOT="$(MIND_RESULTS_ROOT)" CUDA_VISIBLE_DEVICES="" $(DS_ENV)
LEDGER_MODELS_ARGS = $(if $(strip $(MODELS)),--models $(MODELS),)

ds-ledger-add:
	@test -n "$(strip $(OUTPUTS_ROOT)$(EVAL_DIRS))" || { echo "ERROR: set OUTPUTS_ROOT=<outputs root> and/or EVAL_DIRS=\"<eval dir> ...\""; exit 1; }
	$(CONDA_RUN) $(LEDGER_ENV) python -m downstream.ledger add $(if $(strip $(OUTPUTS_ROOT)),--outputs-root "$(OUTPUTS_ROOT)",) \
		$(foreach d,$(EVAL_DIRS),--eval-dir "$(d)")

ds-ledger-annotate:
	@test -n "$(SELECT)" -a -n "$(STATUS)" -a -n "$(REASON)" || { echo "ERROR: set SELECT, STATUS and REASON"; exit 1; }
	$(CONDA_RUN) $(LEDGER_ENV) python -m downstream.ledger annotate --select "$(SELECT)" --status $(STATUS) --reason "$(REASON)" \
		$(if $(strip $(REPLACED_BY)),--replaced-by "$(REPLACED_BY)",)

ds-ledger-report:
	$(CONDA_RUN) $(LEDGER_ENV) python -m downstream.ledger_report report $(LEDGER_MODELS_ARGS)

ds-summary ds-missing:
	$(CONDA_RUN) $(LEDGER_ENV) python -m downstream.ledger_report $(patsubst ds-%,%,$@) $(LEDGER_MODELS_ARGS)

ds-compare:
	@test -n "$(A)" -a -n "$(B)" || { echo "ERROR: set A and B (model keys as in the ledger)"; exit 1; }
	$(CONDA_RUN) $(LEDGER_ENV) python -m downstream.ledger_compare --a "$(A)" --b "$(B)" --bootstrap $(BOOTSTRAP) \
		$(if $(strip $(TASKS)),--tasks $(TASKS),)

ds-ledger-verify:
	$(CONDA_RUN) $(LEDGER_ENV) python -m downstream.ledger verify

ds-ledger-recompute:
	@test -n "$(MODEL)" -a -n "$(CONTROL)" -a -n "$(DATA_REVISION)" || { echo "ERROR: set MODEL, CONTROL and DATA_REVISION"; exit 1; }
	$(CONDA_RUN) $(LEDGER_ENV) python -m downstream.ledger recompute-macro-aupr --task $(TASK) --model "$(MODEL)" \
		--control "$(CONTROL)" --data-revision "$(DATA_REVISION)" $(if $(strip $(POOLING)),--pooling $(POOLING),) \
		$(if $(filter 1,$(DRY_RUN)),--dry-run,)

HEADS ?= linear,mlp
PER_GPU ?= 1
PROBE_JOBS ?= 1
DS_RUN_CMD = python -u -m downstream.run_all --model "$(MODEL)" $(if $(strip $(NAME)),--name "$(NAME)",) \
	--tasks $(or $(strip $(TASKS)),all) --gpus "$(GPU)" --per-gpu $(PER_GPU) --heads $(HEADS) --seeds $(SEEDS) \
	--probe-jobs $(PROBE_JOBS) --threads $(DS_THREADS) --log-root $(LOG_MONTH_DIR) $(DS_RANDOM_INIT_ARGS) \
	$(if $(strip $(OUTPUTS_ROOT)),--outputs-root "$(OUTPUTS_ROOT)",) \
	$(if $(strip $(COMPARE_TO)),--compare-to "$(COMPARE_TO)" --bootstrap $(BOOTSTRAP),) \
	$(if $(filter 1,$(COUNTERFACTUAL)),--counterfactual --cf-split $(or $(CF_SPLIT),test) --cf-seed $(or $(SEED),0) \
		--cf-domains $(or $(DOMAINS),unimol) --cf-modes $(CF_MODES) \
		$(if $(strip $(EVAL_CONFIG)),--cf-eval-config core/$(EVAL_CONFIG).yaml,),) \
	$(if $(filter 1,$(PLAN)),--plan,)

# Detached by default (DETACH=1 is still accepted); PLAN=1 and FOREGROUND=1 run in the terminal.
ds-run:
	@test -n "$(MODEL)" || { echo "ERROR: set MODEL (snapshot name or .ckpt path; a path also needs NAME)"; exit 1; }
ifeq ($(filter 1,$(PLAN) $(FOREGROUND)),)
	@mkdir -p $(LOG_MONTH_DIR)
	$(call DS_DETACH,$(LOG_MONTH_DIR)/ds_run_$(_LOG_DATETIME)_$$$$.log,$(CONDA_RUN) $(LEDGER_ENV) $(DS_RUN_CMD))
else
	$(CONDA_RUN) $(LEDGER_ENV) $(DS_RUN_CMD)
endif

# Newest ds_*.log files of this month with their EXIT_CODE line. No EXIT_CODE: still running, killed, or a
# foreground log; check the process (make status) before concluding anything.
ds-status:
	@ls -t $(LOG_MONTH_DIR)/ds_*.log 2>/dev/null | head -$(or $(N),15) | while read -r f; do \
		code=$$(grep -a '^EXIT_CODE=' "$$f" | tail -1); \
		echo "$${code:-EXIT_CODE yok (koşuyor olabilir; make status ile doğrula)}  $$f  (son yazma $$(date -r "$$f" '+%F %T'))"; \
	done

test-downstream:
	$(CONDA_RUN) CUDA_VISIBLE_DEVICES="" python -m pytest -q tests/test_downstream_*.py

logs:
	@find logs/ -name "*.log" -printf "%T@ %p\n" 2>/dev/null | sort -rn | head -10 | awk '{print $$2}' || echo "No log files found."

help:
	@echo "MIND Commands:"
	@echo ""
	@echo "  Training:"
	@echo "    make train CONFIG=m LOG=run1         Four-domain training (multidomain_1m_coconut)"
	@echo "    make train CONFIG=f                  Fast sanity check"
	@echo "    make train CONFIG=f EPOCHS=1         Override config epoch count"
	@echo "    make train CONFIG=f SEED=43          Independent seeded run"
	@echo "    make train CONFIG=m CKPT=path EPOCHS=101  Resume full checkpoint state"
	@echo "    make train CONFIG=m MAX_ATOMS=18000      Override dynamic atom budget"
	@echo "    make train CONFIG=pretraining_config_protein_only EPOCHS=500 SEED=42  Protein-only (outputs/protein_only/)"
	@echo "    make check-epoch-plan CONFIG=pretraining_config_protein_only  Epoch plan fits epoch_num_batches?"
	@echo "    make test-unit                       Run the complete unit-test suite"
	@echo "    make eval-pretrain CKPT=<ckpt> SPLIT=test GPU=<id> OUT=<new dir>  Offline checkpoint eval (GPU)"
	@echo "    make eval-counterfactual CKPT=<ckpt> SPLIT=test GPU=<id> OUT_ROOT=<new dir>  UniMol counterfactual sweep (GPU)"
	@echo ""
	@echo "  Downstream (see downstream/README.md):"
	@echo "    make ds-snapshot CKPT=<ckpt> NAME=<snapshot> ALIAS=current_best"
	@echo "    make ds-download TASK=ec / ds-process TASK=ec WORKERS=16   (data, explicit decision)"
	@echo "    make ds-embed TASK=ec MODEL=<snapshot> GPU=<id> [RANDOM_INIT=1]   (GPU)"
	@echo "    make ds-probe TASK=ec MODEL_ID=<id> HEAD=linear|mlp"
	@echo "    make ds-finetune TASK=ec MODEL=<snapshot> GPU=<id>   (GPU, long)"
	@echo "    make ds-report TASK=ec / make test-downstream"
	@echo "    make ds-ledger-add OUTPUTS_ROOT=<dir> / ds-ledger-report / ds-compare A=<m> B=<m>   (results ledger, CPU)"
	@echo "    LBA: make ds-download TASK=lba / ds-process TASK=lba"
	@echo "         make ds-embed TASK=lba MODEL=<snapshot> GPU=<id> [SPLIT_TAG=identity30] [REPRESENTATION=joint]"
	@echo "         make ds-probe TASK=lba MODEL_ID=<id> HEAD=mlp [SPLIT_TAG=identity30] [REPRESENTATION=joint]"
	@echo "    make stop GPU=6                      Kill training on GPU 6"
	@echo "    make status                          Show all processes + GPUs"
	@echo "    make logs                            List recent log files"
	@echo ""
	@echo "  Data Processing:"
	@echo "    make build-edge-free DOMAIN=protein MANIFEST=.. RAW_DIR=.. DEST_PREFIX=.. CHUNK_SIZE=4000"
	@echo "                                         Raw -> edge-free chunks (detached, no defaults)"
	@echo ""
	@echo "  CONFIG shortcuts: m=multidomain_1m_coconut, f=fast"
