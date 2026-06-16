#!/usr/bin/env python3
"""
Test script to verify enhanced training features work correctly
"""

import subprocess
import sys
from pathlib import Path

def test_argument_parsing():
    """Test that new arguments are accepted by the training script"""
    script_path = Path("scripts/train_yolo_head_staff.py")
    if not script_path.exists():
        print("Training script not found")
        return False
        
    # Test basic argument parsing
    try:
        result = subprocess.run([
            sys.executable, str(script_path), "--help"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            print("Help command failed")
            return False
            
        # Check for new arguments in help output
        help_text = result.stdout
        required_args = [
            "--checkpoint-period",
            "--min-delta",
            "--hyperparameter-opt",
            "--opt-iters"
        ]
        
        for arg in required_args:
            if arg not in help_text:
                print(f"Missing argument in help: {arg}")
                return False
                
        print("✓ Argument parsing test passed")
        return True
    except Exception as e:
        print(f"Argument parsing test failed: {e}")
        return False

def test_evaluation_arguments():
    """Test that new arguments are accepted by the evaluation script"""
    script_path = Path("scripts/eval_yolo_head_staff.py")
    if not script_path.exists():
        print("Evaluation script not found")
        return False
        
    # Test basic argument parsing
    try:
        result = subprocess.run([
            sys.executable, str(script_path), "--help"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            print("Help command failed")
            return False
            
        # Check for new arguments in help output
        help_text = result.stdout
        required_args = [
            "--verbose",
            "--per-class"
        ]
        
        for arg in required_args:
            if arg not in help_text:
                print(f"Missing argument in help: {arg}")
                return False
                
        print("✓ Evaluation argument parsing test passed")
        return True
    except Exception as e:
        print(f"Evaluation argument parsing test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Testing Enhanced Training Features...")
    print("=" * 50)
    
    tests = [
        test_argument_parsing,
        test_evaluation_arguments
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"Tests passed: {passed}/{len(tests)}")
    
    if passed == len(tests):
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())