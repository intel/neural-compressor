#!/usr/bin/env python3

# Copyright (c) 2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
LM-Eval launcher with Auto-Round extension support.
This script patches vLLM to support Auto-Round quantization before running lm_eval.
"""

import os
import sys
import argparse
import subprocess

def setup_auto_round_extension():
    """Setup Auto-Round extension for vLLM."""
    # Check if extension should be enabled
    VLLM_ENABLE_AR_EXT = os.environ.get("VLLM_ENABLE_AR_EXT", "") in [
        "1",
        "true", 
        "True",
    ]
    
    if VLLM_ENABLE_AR_EXT:
        try:
            print("Loading Auto-Round extension...")
            import vllm.model_executor.layers.quantization.auto_round as auto_round_module

            from auto_round_extension.vllm_ext.auto_round_ext import AutoRoundExtensionConfig

            auto_round_module.AutoRoundConfig = AutoRoundExtensionConfig
            from auto_round_extension.vllm_ext.envs_ext import extra_environment_variables
            
            print("✓ Auto-Round extension loaded successfully")
            return True
        except ImportError as e:
            print(f"Warning: Failed to load Auto-Round extension: {e}")
            return False
    else:
        print("Auto-Round extension disabled (VLLM_ENABLE_AR_EXT not set)")
        return False

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="LM-Eval launcher with Auto-Round extension support",
        add_help=False  # We'll pass through to lm_eval
    )
    
    # Add our own arguments
    parser.add_argument(
        "--enable-ar-ext", 
        action="store_true",
        help="Enable Auto-Round extension (can also use VLLM_ENABLE_AR_EXT=1)"
    )

    # Parse known args so we can handle our flags and pass the rest to lm_eval
    args, remaining_args = parser.parse_known_args()
    
    return args, remaining_args

def main():
    """Main function to setup extension and run lm_eval."""
    # Parse arguments
    args, remaining_args = parse_args()
    
    # Set up environment for Auto-Round extension
    if args.enable_ar_ext or os.environ.get("VLLM_ENABLE_AR_EXT"):
        os.environ["VLLM_ENABLE_AR_EXT"] = "1"
    
    # Setup the extension
    extension_loaded = setup_auto_round_extension()
    
    if extension_loaded:
        print("Auto-Round extension is ready for vLLM models")
    
    # Prepare lm_eval command
    lm_eval_cmd = ["lm_eval"] + remaining_args
    
    print(f"Running: {' '.join(lm_eval_cmd)}")
    print("-" * 50)
    
    try:
        # Run lm_eval with the remaining arguments
        result = subprocess.run(lm_eval_cmd, check=True)
        sys.exit(result.returncode)
    except subprocess.CalledProcessError as e:
        print(f"Error running lm_eval: {e}")
        sys.exit(e.returncode)
    except FileNotFoundError:
        print("Error: lm_eval not found. Please install lm_eval package:")
        print("pip install lm_eval")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(130)

if __name__ == "__main__":
    main()