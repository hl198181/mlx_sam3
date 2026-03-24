# SAM3 Segmentation Studio - Mac Service Management

.PHONY: install uninstall restart status logs

PROJECT_DIR := $(shell pwd)
SERVICE_DIR := app
PLIST_SRC := $(SERVICE_DIR)/sam3.plist
PLIST_DEST := $(HOME)/Library/LaunchAgents/com.mlxsam3.app.plist
SERVICE_LABEL := com.mlxsam3.app
LOGS_DIR := $(SERVICE_DIR)/logs

install: ## 安装并启动服务（开机自启）
	@mkdir -p $(LOGS_DIR)
	@sed "s|/Users/clean1master|$(HOME)|g" $(PLIST_SRC) > $(PLIST_DEST)
	@launchctl load $(PLIST_DEST)
	@echo "✅ 服务已安装并启动"
	@echo "   Plist: $(PLIST_DEST)"
	@echo "   日志: $(LOGS_DIR)/"

uninstall: ## 停止并卸载服务
	@launchctl unload $(PLIST_DEST) 2>/dev/null || true
	@rm -f $(PLIST_DEST)
	@echo "🛑 服务已卸载"

restart: ## 重启服务
	@launchctl unload $(PLIST_DEST) 2>/dev/null || true
	@sleep 1
	@launchctl load $(PLIST_DEST)
	@echo "🔄 服务已重启"

status: ## 查看服务状态
	@launchctl list $(SERVICE_LABEL) 2>/dev/null && echo "✅ 服务运行中" || echo "❌ 服务未运行"

logs: ## 查看最近日志
	@echo "=== stdout.log ===" && tail -50 $(LOGS_DIR)/stdout.log 2>/dev/null || true
	@echo "" && echo "=== stderr.log ===" && tail -50 $(LOGS_DIR)/stderr.log 2>/dev/null || true

help: ## 显示帮助
	@echo "使用方法: make [命令]"
	@echo ""
	@echo "  install   - 安装并启动服务（开机自启）"
	@echo "  uninstall - 停止并卸载服务"
	@echo "  restart   - 重启服务"
	@echo "  status    - 查看服务状态"
	@echo "  logs      - 查看最近日志"
