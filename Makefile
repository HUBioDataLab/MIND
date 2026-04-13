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
