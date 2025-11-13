#!/usr/bin/env python3
"""
Warning script to inform users about build time when installing from source.
This is executed early in the build process to warn about long build times.
"""
import os
import sys

def show_warning():
    """Show warning about source installation."""
    
    # Check if we're in a CI environment (skip warning)
    ci_vars = ['CI', 'GITHUB_ACTIONS', 'GITLAB_CI', 'CIRCLECI', 'TRAVIS']
    if any(os.getenv(var) for var in ci_vars):
        return
    
    # Check if we're building a wheel (not editable install)
    if '--build' in sys.argv or 'bdist_wheel' in sys.argv:
        return
    
    warning = """
╔══════════════════════════════════════════════════════════════════════════╗
║                          ⚠️  BUILD WARNING ⚠️                            ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  You are building PyOpenSim from source, which will:                    ║
║                                                                          ║
║  ⏱️  Take 30-60 minutes (building OpenSim C++ libraries)                ║
║  💾 Require 5+ GB of disk space                                         ║
║  🔧 Need: CMake, SWIG, C++ compiler, and build tools                    ║
║                                                                          ║
║  RECOMMENDED ALTERNATIVES:                                               ║
║                                                                          ║
║  1️⃣  Install from PyPI (pre-built wheels):                             ║
║     pip install pyopensim                                                ║
║                                                                          ║
║  2️⃣  Install from GitHub Release:                                       ║
║     pip install "https://github.com/hudsonburke/pyopensim/             ║
║                  releases/download/v4.5.2/pyopensim-...-....whl"        ║
║                                                                          ║
║  Press Ctrl+C now to cancel, or wait 10 seconds to continue...          ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
"""
    
    print(warning, file=sys.stderr, flush=True)
    
    # Give user time to cancel
    import time
    for i in range(10, 0, -1):
        print(f"\rContinuing in {i} seconds... ", end='', file=sys.stderr, flush=True)
        time.sleep(1)
    print("\n\nProceeding with build...\n", file=sys.stderr, flush=True)

if __name__ == '__main__':
    show_warning()
