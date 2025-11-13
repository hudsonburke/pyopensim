#!/usr/bin/env python3
"""Apply the overload fix to existing stub files."""

import sys
from pathlib import Path

# Add the scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))

from generate_stubs import (fix_missing_overload_decorators, fix_missing_type_imports, 
                            fix_undefined_simbody_types, remove_duplicate_overloads, 
                            remove_orphaned_overload_decorators, add_missing_final_overload_decorators)

def fix_stub_file(file_path: Path):
    """Apply overload fixes to a single stub file."""
    print(f"Fixing {file_path.name}...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original = content
    content = fix_missing_overload_decorators(content)
    content = fix_missing_type_imports(content)
    content = fix_undefined_simbody_types(content)
    content = remove_duplicate_overloads(content)
    content = remove_orphaned_overload_decorators(content)
    content = add_missing_final_overload_decorators(content)
    
    if content != original:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  ✓ Fixed {file_path.name}")
        return True
    else:
        print(f"  - No changes needed for {file_path.name}")
        return False

if __name__ == "__main__":
    stub_dir = Path("/home/hudson/pyopensim/src/pyopensim")
    
    if not stub_dir.exists():
        print(f"Error: {stub_dir} does not exist")
        sys.exit(1)
    
    stub_files = list(stub_dir.glob("*.pyi"))
    if not stub_files:
        print(f"No .pyi files found in {stub_dir}")
        sys.exit(1)
    
    print(f"Found {len(stub_files)} stub files to process\n")
    
    fixed_count = 0
    for stub_file in stub_files:
        if stub_file.name != "__init__.pyi":  # Skip __init__.pyi
            if fix_stub_file(stub_file):
                fixed_count += 1
    
    print(f"\n{'='*50}")
    print(f"Fixed {fixed_count} out of {len(stub_files)-1} stub files")
    print(f"{'='*50}")
