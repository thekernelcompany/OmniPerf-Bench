#!/usr/bin/env python3
"""
Push locally cached baseline images to Docker Hub.

Runs as a separate process - monitors for new baseline images and pushes them.
"""

import subprocess
import time
import sys

# Unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

DOCKER_HUB_REPO = "shikhar481/vllm_fixed_human_images"
LOCAL_PREFIX = "vllm-baseline-built"

def get_local_baseline_images():
    """Get list of locally cached baseline images."""
    result = subprocess.run(
        ['docker', 'images', '--format', '{{.Repository}}:{{.Tag}}', LOCAL_PREFIX],
        capture_output=True, text=True
    )
    images = [img.strip() for img in result.stdout.strip().split('\n') if img.strip()]
    return images


def get_pushed_tags():
    """Get list of tags already pushed to Docker Hub."""
    # Check what's already on Docker Hub
    result = subprocess.run(
        ['docker', 'images', '--format', '{{.Tag}}', DOCKER_HUB_REPO],
        capture_output=True, text=True
    )
    tags = set(tag.strip() for tag in result.stdout.strip().split('\n') if tag.strip())
    return tags


def push_image(local_image: str) -> bool:
    """Tag and push a baseline image to Docker Hub."""
    # Extract parent commit hash from local tag
    # vllm-baseline-built:54600709b6d4 -> baseline-54600709b6d4
    parent_hash = local_image.split(':')[1]
    remote_tag = f"baseline-{parent_hash}"
    remote_image = f"{DOCKER_HUB_REPO}:{remote_tag}"

    print(f"Pushing {local_image} -> {remote_image}")

    # Tag for remote
    result = subprocess.run(
        ['docker', 'tag', local_image, remote_image],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  Failed to tag: {result.stderr}")
        return False

    # Push to Docker Hub
    result = subprocess.run(
        ['docker', 'push', remote_image],
        capture_output=True, text=True,
        timeout=1800  # 30 min timeout per image
    )
    if result.returncode != 0:
        print(f"  Failed to push: {result.stderr}")
        return False

    print(f"  Successfully pushed {remote_image}")
    return True


def push_all_once():
    """Push all unpushed baseline images once."""
    local_images = get_local_baseline_images()
    print(f"Found {len(local_images)} local baseline images")

    pushed = 0
    failed = 0
    skipped = 0

    for local_image in local_images:
        parent_hash = local_image.split(':')[1]
        remote_tag = f"baseline-{parent_hash}"

        # Check if already pushed by trying to inspect remote
        check = subprocess.run(
            ['docker', 'manifest', 'inspect', f"{DOCKER_HUB_REPO}:{remote_tag}"],
            capture_output=True, text=True
        )
        if check.returncode == 0:
            print(f"  SKIP: {remote_tag} already exists on Docker Hub")
            skipped += 1
            continue

        if push_image(local_image):
            pushed += 1
        else:
            failed += 1

    print(f"\nSummary: {pushed} pushed, {failed} failed, {skipped} skipped")
    return pushed, failed


def watch_and_push(interval: int = 60):
    """Watch for new baseline images and push them."""
    print(f"Watching for new baseline images (checking every {interval}s)...")
    pushed_set = set()

    while True:
        local_images = get_local_baseline_images()

        for local_image in local_images:
            if local_image in pushed_set:
                continue

            parent_hash = local_image.split(':')[1]
            remote_tag = f"baseline-{parent_hash}"

            # Check if already on Docker Hub
            check = subprocess.run(
                ['docker', 'manifest', 'inspect', f"{DOCKER_HUB_REPO}:{remote_tag}"],
                capture_output=True, text=True
            )
            if check.returncode == 0:
                pushed_set.add(local_image)
                continue

            # Push it
            if push_image(local_image):
                pushed_set.add(local_image)

        time.sleep(interval)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--watch':
        interval = int(sys.argv[2]) if len(sys.argv) > 2 else 60
        watch_and_push(interval)
    else:
        # One-time push of all images
        push_all_once()
