# ===================================================================
# MIND Training Makefile
# ===================================================================
# Usage:
#   make train GPU=6 CONFIG=m                  # multidomain, default log
#   make train GPU=6 CONFIG=f                  # fast test (~10 min)
#   make train GPU=6 CONFIG=m LOG=my_run       # custom log name
#   make train GPU=1 CONFIG=f LOG=fast_check   # fast on GPU 1
#
# CONFIG shortcuts:
#   m  = pretraining_config_multidomain.yaml
#   f  = pretraining_config_fast.yaml
#   (or pass full filename without .yaml)
#
# Defaults: GPU=6, CONFIG=m, LOG=train_<config>
# ===================================================================

GPU ?= 6
CONFIG ?= m
LOG ?=

# Map single-letter shortcuts to full config names
ifeq ($(CONFIG),m)
  CONFIG_FILE := core/pretraining_config_multidomain.yaml
  CONFIG_TAG := multidomain
else ifeq ($(CONFIG),f)
  CONFIG_FILE := core/pretraining_config_fast.yaml
  CONFIG_TAG := fast
else
  CONFIG_FILE := core/$(CONFIG).yaml
  CONFIG_TAG := $(CONFIG)
endif

# Default log name if not provided
ifeq ($(LOG),)
  LOG_FILE := train_$(CONFIG_TAG).log
else
  LOG_FILE := $(LOG).log
endif

.PHONY: train stop status logs help

train:
	@echo "=== MIND Training ==="
	@echo "  GPU:    $(GPU)"
	@echo "  Config: $(CONFIG_FILE)"
	@echo "  Log:    $(LOG_FILE)"
	@echo "====================="
	CUDA_VISIBLE_DEVICES=$(GPU) nohup python -m core.train_pretrain \
		--config-yaml-path $(CONFIG_FILE) \
		> $(LOG_FILE) 2>&1 &
	@echo "PID: $$!"
	@echo "Follow: tail -f $(LOG_FILE)"

stop:
	@echo "Killing training processes on GPU $(GPU)..."
	@ps aux | grep "CUDA_VISIBLE_DEVICES=$(GPU).*train_pretrain" | grep -v grep | awk '{print $$2}' | xargs -r kill
	@echo "Done."

status:
	@echo "=== Running MIND processes ==="
	@ps aux | grep train_pretrain | grep -v grep || echo "No training processes found."
	@echo ""
	@nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || true

logs:
	@ls -lt *.log 2>/dev/null | head -5 || echo "No log files found."

help:
	@echo "MIND Training Commands:"
	@echo "  make train GPU=6 CONFIG=m LOG=run1   Full multidomain training"
	@echo "  make train GPU=6 CONFIG=f             Fast sanity check (~10 min)"
	@echo "  make stop GPU=6                       Kill training on GPU 6"
	@echo "  make status                           Show running processes + GPU usage"
	@echo "  make logs                             List recent log files"
	@echo ""
	@echo "CONFIG shortcuts: m=multidomain, f=fast"
