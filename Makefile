# SAM3 Segmentation Studio - Service Management
# Mac (MLX): make install / make restart / make logs
# Linux GPU: make install-gpu / make restart-gpu / make logs-gpu

.PHONY: install uninstall restart status logs \
        install-gpu uninstall-gpu restart-gpu status-gpu logs-gpu help

PROJECT_DIR := $(shell pwd)
SERVICE_DIR := app

# ============================================================
# Mac (MLX / launchd)
# ============================================================
PLIST_SRC := $(SERVICE_DIR)/sam3.plist
PLIST_DEST := $(HOME)/Library/LaunchAgents/com.mlxsam3.app.plist
SERVICE_LABEL := com.mlxsam3.app
LOGS_DIR := $(SERVICE_DIR)/logs

install:
	@mkdir -p $(LOGS_DIR)
	@sed "s|/Users/clean1master|$(HOME)|g" $(PLIST_SRC) > $(PLIST_DEST)
	@launchctl load $(PLIST_DEST)
	@echo "✅ MLX 服务已安装并启动"

uninstall:
	@launchctl unload $(PLIST_DEST) 2>/dev/null || true
	@rm -f $(PLIST_DEST)
	@echo "🛑 MLX 服务已卸载"

restart:
	@launchctl unload $(PLIST_DEST) 2>/dev/null || true
	@sleep 1
	@launchctl load $(PLIST_DEST)
	@echo "🔄 MLX 服务已重启"

status:
	@launchctl list $(SERVICE_LABEL) 2>/dev/null && echo "✅ MLX 服务运行中" || echo "❌ MLX 服务未运行"

logs:
	@echo "=== stdout.log ===" && tail -50 $(LOGS_DIR)/stdout.log 2>/dev/null || true
	@echo "" && echo "=== stderr.log ===" && tail -50 $(LOGS_DIR)/stderr.log 2>/dev/null || true

# ============================================================
# Linux GPU (systemd)
# ============================================================
GPU_SERVICE_SRC := $(SERVICE_DIR)/sam3-gpu.service
GPU_SERVICE_NAME := sam3-gpu

install-gpu:
	@sed "s|/home/biofuture|$(HOME)|g" $(GPU_SERVICE_SRC) > /tmp/$(GPU_SERVICE_NAME).service
	@sudo cp /tmp/$(GPU_SERVICE_NAME).service /etc/systemd/system/$(GPU_SERVICE_NAME).service
	@sudo systemctl daemon-reload
	@sudo systemctl enable $(GPU_SERVICE_NAME)
	@sudo systemctl start $(GPU_SERVICE_NAME)
	@echo "✅ GPU 服务已安装并启动"
	@echo "   查看日志: make logs-gpu"

uninstall-gpu:
	@sudo systemctl stop $(GPU_SERVICE_NAME) 2>/dev/null || true
	@sudo systemctl disable $(GPU_SERVICE_NAME) 2>/dev/null || true
	@sudo rm -f /etc/systemd/system/$(GPU_SERVICE_NAME).service
	@sudo systemctl daemon-reload
	@echo "🛑 GPU 服务已卸载"

restart-gpu:
	@sudo systemctl restart $(GPU_SERVICE_NAME)
	@echo "🔄 GPU 服务已重启"

status-gpu:
	@sudo systemctl status $(GPU_SERVICE_NAME) --no-pager -l 2>/dev/null || echo "❌ GPU 服务未安装"

logs-gpu:
	@journalctl -u $(GPU_SERVICE_NAME) -n 100 --no-pager

help:
	@echo "使用方法: make [命令]"
	@echo ""
	@echo "  === Mac (MLX) ==="
	@echo "  install     - 安装并启动 MLX 服务（开机自启）"
	@echo "  uninstall   - 停止并卸载 MLX 服务"
	@echo "  restart     - 重启 MLX 服务"
	@echo "  status      - 查看 MLX 服务状态"
	@echo "  logs        - 查看 MLX 最近日志"
	@echo ""
	@echo "  === Linux GPU ==="
	@echo "  install-gpu   - 安装并启动 GPU 服务（开机自启）"
	@echo "  uninstall-gpu - 停止并卸载 GPU 服务"
	@echo "  restart-gpu   - 重启 GPU 服务"
	@echo "  status-gpu    - 查看 GPU 服务状态"
	@echo "  logs-gpu      - 查看 GPU 最近日志"
