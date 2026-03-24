#!/bin/bash

# SAM3 Segmentation Studio - LaunchAgent Service Manager
# Usage: ./install-service.sh [install|uninstall|status|logs|restart]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLIST_SRC="$SCRIPT_DIR/sam3.plist"
PLIST_DEST="$HOME/Library/LaunchAgents/com.mlxsam3.app.plist"
SERVICE_LABEL="com.mlxsam3.app"
LOGS_DIR="$SCRIPT_DIR/logs"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

mkdir -p "$LOGS_DIR"

case "$1" in
    install)
        # Update plist with current user's home directory
        sed "s|/Users/clean1master|$HOME|g" "$PLIST_SRC" > "$PLIST_DEST"

        launchctl load "$PLIST_DEST"
        echo -e "${GREEN}Service installed and started.${NC}"
        echo -e "  Label: $SERVICE_LABEL"
        echo -e "  Plist: $PLIST_DEST"
        echo -e "  Logs: $LOGS_DIR/"
        ;;

    uninstall)
        launchctl unload "$PLIST_DEST" 2>/dev/null
        rm -f "$PLIST_DEST"
        echo -e "${YELLOW}Service uninstalled.${NC}"
        ;;

    restart)
        launchctl unload "$PLIST_DEST" 2>/dev/null
        sleep 1
        launchctl load "$PLIST_DEST"
        echo -e "${GREEN}Service restarted.${NC}"
        ;;

    status)
        if launchctl list "$SERVICE_LABEL" &>/dev/null; then
            echo -e "${GREEN}Service is running.${NC}"
            launchctl list "$SERVICE_LABEL"
        else
            echo -e "${RED}Service is not running.${NC}"
        fi
        ;;

    logs)
        if [ -f "$LOGS_DIR/stdout.log" ]; then
            echo -e "${GREEN}=== stdout.log (last 50 lines) ===${NC}"
            tail -50 "$LOGS_DIR/stdout.log"
        fi
        if [ -f "$LOGS_DIR/stderr.log" ]; then
            echo -e "${RED}=== stderr.log (last 50 lines) ===${NC}"
            tail -50 "$LOGS_DIR/stderr.log"
        fi
        ;;

    *)
        echo "Usage: $0 [install|uninstall|restart|status|logs]"
        echo ""
        echo "  install   - Install and start the service (auto-start on boot)"
        echo "  uninstall - Stop and remove the service"
        echo "  restart   - Restart the service"
        echo "  status    - Check service status"
        echo "  logs      - View recent logs"
        exit 1
        ;;
esac
