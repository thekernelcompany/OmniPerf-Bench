#!/bin/bash
# Cleanup old baseline images when disk usage exceeds threshold
THRESHOLD=80  # percent

usage=$(df /ephemeral | tail -1 | awk '{print $5}' | tr -d '%')
if [ "$usage" -gt "$THRESHOLD" ]; then
    echo "$(date): Disk at ${usage}%, cleaning up..."
    # Keep only the 10 most recent baseline images
    docker images --format "{{.ID}} {{.CreatedAt}}" | grep vllm-baseline-built | sort -k2 -r | tail -n +11 | awk '{print $1}' | xargs -r docker rmi
    docker system prune -f
else
    echo "$(date): Disk at ${usage}%, no cleanup needed"
fi
