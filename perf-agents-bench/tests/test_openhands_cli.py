#!/usr/bin/env python3
"""
Test script for OpenHands CLI integration.

This script tests the basic functionality of the OpenHands CLI integration
without requiring API keys or complex setup.
"""

import tempfile
import shutil
from pathlib import Path
from bench.agents.openhands_cli import OpenHandsCLI, create_opensource_task

def test_basic_functionality():
    """Test basic OpenHands CLI functionality."""
    print("Testing OpenHands CLI integration...")

    # Test CLI initialization
    try:
        cli = OpenHandsCLI()
        print("✓ OpenHands CLI initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize OpenHands CLI: {e}")
        return False

    # Test workspace validation
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace = Path(temp_dir)
        (workspace / "test.py").write_text("print('hello world')")

        if cli.validate_workspace(workspace):
            print("✓ Workspace validation passed")
        else:
            print("✗ Workspace validation failed")
            return False

    return True

def test_task_creation():
    """Test task creation functionality."""
    print("\nTesting task creation...")

    task = create_opensource_task(
        task_name="Test Optimization",
        description="Simple test task",
        target_files=["test.py"],
        constraints=["No breaking changes"],
        reference_commit="abc123",
        primary_metric="performance"
    )

    if "# Performance Optimization Task" in task:
        print("✓ Task creation successful")
        print(f"Task length: {len(task)} characters")
        return True
    else:
        print("✗ Task creation failed")
        return False

def test_git_workspace():
    """Test with a git workspace."""
    print("\nTesting git workspace...")

    with tempfile.TemporaryDirectory() as temp_dir:
        workspace = Path(temp_dir)

        # Initialize git repo
        import subprocess
        subprocess.run(["git", "init"], cwd=workspace, check=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=workspace, check=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=workspace, check=True)

        # Create test file and commit
        (workspace / "test.py").write_text("print('hello world')")
        subprocess.run(["git", "add", "test.py"], cwd=workspace, check=True)
        subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=workspace, check=True)

        # Test workspace validation
        cli = OpenHandsCLI()
        if cli.validate_workspace(workspace):
            print("✓ Git workspace validation passed")
            return True
        else:
            print("✗ Git workspace validation failed")
            return False

def main():
    """Run all tests."""
    print("OpenHands CLI Integration Test Suite")
    print("=" * 40)

    tests = [
        test_basic_functionality,
        test_task_creation,
        test_git_workspace
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")

    print("\n" + "=" * 40)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1

if __name__ == "__main__":
    exit(main())
