#!/usr/bin/env python3
"""
Improved stub generation for PyOpenSim using mypy's stubgen with post-processing.
Fixes common SWIG-related issues in generated stubs.
"""

import re
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


def ensure_mypy_available() -> bool:
    """Check if mypy is available, install if needed."""
    try:
        import mypy.stubgen  # noqa: F401
        print("[OK] mypy is available")
        return True
    except ImportError:
        print("Installing mypy for stub generation...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "mypy"], check=True)
            print("[OK] mypy installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Failed to install mypy: {e}")
            return False


def fix_malformed_self_parameters(content: str) -> str:
    """Fix malformed self parameters in stub content."""
    # Pattern to match malformed self parameters like selfProperty, selfos, selfX_BD, etc.
    # This matches 'self' followed immediately by a capital letter or lowercase letter(s)
    patterns = [
        # Fix selfCamelCase -> self, CamelCase
        (r'\bself([A-Z][a-zA-Z_0-9]*)', r'self, \1'),
        # Fix selflowercase -> self, lowercase  
        (r'\bself([a-z][a-zA-Z_0-9]*)', r'self, \1'),
        # Fix selfX_Y style parameters -> self, X_Y
        (r'\bself([A-Z_][A-Z_0-9]*)', r'self, \1'),
    ]
    
    fixed_content = content
    for pattern, replacement in patterns:
        fixed_content = re.sub(pattern, replacement, fixed_content)
    
    return fixed_content


def fix_duplicate_self_parameters(content: str) -> str:
    """Fix cases where self appears twice like 'self, self, param' or 'self, self: Type'."""
    # Fix patterns like "self, self, param" -> "self, param"
    content = re.sub(r'\bself,\s*self,\s*', 'self, ', content)
    
    # Fix patterns like "(self, self: Type)" -> "(self)" in __init__ and other methods
    # This handles SWIG Director classes that sometimes generate duplicate self
    # Use more flexible matching to handle edge cases
    content = re.sub(r'\(self,\s*self:\s*[A-Za-z_][\w]*\s*\)', r'(self)', content)
    
    return content


def fix_init_return_types(content: str) -> str:
    """Fix __init__ methods that return Any instead of None.
    
    SWIG sometimes generates __init__ methods with return type Any,
    but __init__ must always return None in Python.
    """
    # Fix __init__ methods with -> Any return type
    content = re.sub(
        r'(\bdef __init__\([^)]*\))\s*->\s*Any\s*:',
        r'\1 -> None:',
        content
    )
    
    return content


def fix_duplicate_parameters(content: str) -> str:
    """Fix methods with duplicate parameter names.
    
    Sometimes SWIG generates methods with duplicate parameter names
    from overloaded C++ methods with different const qualifiers.
    """
    lines = content.split('\n')
    result = []
    
    for line in lines:
        # Check if this is a function definition
        if 'def ' in line and '(' in line:
            # Extract the parameter list
            match = re.search(r'def\s+\w+\((.*?)\)\s*->', line)
            if match:
                params_str = match.group(1)
                params = [p.strip() for p in params_str.split(',') if p.strip()]
                
                # Check for duplicates
                seen_names = set()
                new_params = []
                has_duplicates = False
                
                for param in params:
                    # Get parameter name (before : if typed)
                    param_name = param.split(':')[0].strip()
                    
                    if param_name in seen_names:
                        # Duplicate found - rename it
                        counter = 2
                        new_name = f"{param_name}{counter}"
                        while new_name in seen_names:
                            counter += 1
                            new_name = f"{param_name}{counter}"
                        
                        # Reconstruct the parameter with new name
                        if ':' in param:
                            type_part = param.split(':', 1)[1]
                            param = f"{new_name}: {type_part}"
                        else:
                            param = new_name
                        has_duplicates = True
                    
                    seen_names.add(param.split(':')[0].strip())
                    new_params.append(param)
                
                # Reconstruct the line if we found duplicates
                if has_duplicates:
                    new_params_str = ', '.join(new_params)
                    line = re.sub(
                        r'(def\s+\w+\().*?(\)\s*->)',
                        rf'\1{new_params_str}\2',
                        line
                    )
        
        result.append(line)
    
    return '\n'.join(result)


def fix_missing_type_imports(content: str) -> str:
    """Add missing type imports that are commonly needed."""
    lines = content.split('\n')
    
    # Check if we need to add imports
    has_typing_import = any('from typing import' in line for line in lines[:10])
    has_any_import = any('Any' in line for line in lines[:10])
    needs_overload = '@overload' in content
    has_overload_import = any('overload' in line and 'from typing import' in line for line in lines[:10])
    
    imports_to_add = []
    
    # If we have type annotations but no typing imports, add them
    if 'Any' in content and not has_any_import:
        imports_to_add.append('Any')
    
    if needs_overload and not has_overload_import:
        imports_to_add.append('overload')
    
    if imports_to_add:
        if has_typing_import:
            # Find the typing import line and add missing imports to it
            for i, line in enumerate(lines):
                if line.startswith('from typing import'):
                    for import_name in imports_to_add:
                        if import_name not in line:
                            # Add import to existing import line
                            if line.endswith('import'):
                                lines[i] = line + ' ' + import_name
                            else:
                                lines[i] = line + ', ' + import_name
                    break
        else:
            # Add new typing import with all needed imports
            import_line = 'from typing import ' + ', '.join(imports_to_add)
            lines.insert(0, import_line)
    
    return '\n'.join(lines)


def fix_undefined_simbody_types(content: str) -> str:
    """Fix undefined SimTK/Simbody types by adding type aliases.
    
    SimTK C++ types like Real, Array_, ArrayIndexTraits are exposed through
    SWIG but may not have proper type definitions in stubs. This function
    adds appropriate type aliases, preferring to reference actual Python
    classes where they exist.
    """
    lines = content.split('\n')
    
    # Check if file has undefined types that need fixing
    # Common SimTK types that appear in parameters but may not be defined
    # For templated C++ types (ending in _), we map to the concrete Python class if it exists
    undefined_types = {
        # Fundamental types
        'Real': 'float',  # SimTK::Real is double in C++
        'Array_': 'int',  # Array index/size type
        'ArrayIndexTraits': 'int',  # Array index traits type
        'BodyOrSpaceType': 'int',  # Enum for body or space reference frame
        
        # Template types that map to concrete classes (typedef Name_<Real> Name)
        'Vec': 'Vec3',  # Vec<3> is the typical instantiation
        'UnitVec': 'UnitVec3',  # UnitVec<Real,1> typedef
        'Vector_': 'Vector',  # Vector_<Real> typedef
        'VectorBase': 'VectorBaseDouble',  # VectorBase<Real> typedef
        'VectorView_': 'VectorView',  # VectorView_<Real> typedef
        'RowVector_': 'RowVector',  # RowVector_<Real> typedef  
        'RowVectorBase': 'RowVectorBaseDouble',  # RowVectorBase<Real> typedef
        'RowVectorView_': 'RowVectorView',  # RowVectorView_<Real> typedef
        'Rotation_': 'Rotation',  # Rotation_<Real> typedef
        'InverseRotation_': 'InverseRotation',  # InverseRotation_<Real> typedef
        'Transform_': 'Transform',  # Transform_<Real> typedef
        'InverseTransform_': 'Any',  # InverseTransform_<Real> (may not be exposed)
        'Quaternion_': 'Quaternion',  # Quaternion_<Real> typedef
        'Mat': 'Mat33',  # Mat<M,N> - typically Mat33 (3x3)
        'Matrix_': 'Matrix',  # Matrix_<Real> typedef
        'MatrixView_': 'MatrixView',  # MatrixView_<Real> typedef
        
        # Types that may not be fully exposed
        'MultibodySystem': 'Any',  # Simbody multibody system (C++ only)
        'Visualizer': 'Any',  # Simbody visualizer (C++ only)
        'DecorationGenerator': 'Any',  # Decoration generator (C++ only)
    }
    
    # Check which types are actually used but not defined
    types_needed = {}
    for type_name, type_value in undefined_types.items():
        # Check if type is used in content
        if type_name in content:
            # Check if it's already defined (as class or alias)
            # Use word boundaries to avoid matching substrings like Mat in Matrix
            class_pattern = rf'^class {re.escape(type_name)}[:\(]'
            alias_pattern = rf'^{re.escape(type_name)} ='
            has_class = bool(re.search(class_pattern, content, re.MULTILINE))
            has_alias = bool(re.search(alias_pattern, content, re.MULTILINE))
            
            if not (has_class or has_alias):
                types_needed[type_name] = type_value
    
    if not types_needed:
        return content
    
    # Find where to insert type aliases (after imports, before first class)
    insert_pos = 0
    for i, line in enumerate(lines):
        if line.startswith('from ') or line.startswith('import ') or line.startswith('#'):
            insert_pos = i + 1
        elif line.startswith('class ') or line.startswith('def '):
            break
    
    # Build type alias declarations with informative comments
    type_aliases = ['# SimTK/Simbody type aliases for C++ template instantiations']
    for type_name in sorted(types_needed.keys()):
        type_value = types_needed[type_name]
        comment = ''
        
        # Add helpful comments explaining the C++ to Python mapping
        if type_name == 'Real':
            comment = '  # SimTK::Real (double precision)'
        elif type_name in ('Array_', 'ArrayIndexTraits'):
            comment = '  # C++ size_t for array indexing'
        elif type_name == 'BodyOrSpaceType':
            comment = '  # Enum for reference frame'
        elif type_name.endswith('_') and type_value != 'Any':
            # Template type that maps to concrete class
            comment = f'  # C++ template type, typically {type_value}'
        elif type_name.endswith('_'):
            comment = '  # C++ template type'
        elif type_value == 'Any' and type_name not in ('InverseTransform_',):
            comment = '  # C++ type not fully exposed to Python'
        elif type_value.startswith('Vec') or type_value.startswith('Mat'):
            comment = f'  # Typically instantiated as {type_value}'
            
        type_aliases.append(f'{type_name} = {type_value}{comment}')
    
    type_aliases.append('')  # Add blank line after aliases
    
    # Insert after imports
    for alias in reversed(type_aliases):
        lines.insert(insert_pos, alias)
    
    return '\n'.join(lines)


def fix_missing_overload_decorators(content: str) -> str:
    """Add @overload decorator to first overloaded function when missing.
    
    When stubgen generates overloaded functions, the first one often lacks
    the @overload decorator. This function finds such cases and adds the
    missing decorator.
    """
    lines = content.split('\n')
    result = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Check if this is a method/function definition
        if re.match(r'^\s+def \w+\(', line):
            # Extract the function name and indentation
            match = re.match(r'^(\s+)def (\w+)\(', line)
            if match:
                indent = match.group(1)
                func_name = match.group(2)
                
                # Look ahead to see if there's an overloaded version
                # (same function name with @overload decorator)
                j = i + 1
                found_overload = False
                while j < len(lines):
                    next_line = lines[j]
                    
                    # Check for @overload decorator followed by same function
                    if re.match(r'^\s+@overload\s*$', next_line):
                        # Check if next non-empty line has same function name
                        k = j + 1
                        while k < len(lines) and lines[k].strip() == '':
                            k += 1
                        if k < len(lines):
                            func_match = re.match(rf'^{indent}def ({func_name})\(', lines[k])
                            if func_match:
                                found_overload = True
                                break
                    
                    # Stop looking if we hit another method/function or class
                    if re.match(r'^\s+(def \w+\(|class \w+)', next_line):
                        if not re.match(rf'^\s+@overload', lines[j-1] if j > 0 else ''):
                            break
                    
                    j += 1
                
                # If we found an overload but current line doesn't have @overload,
                # add it
                if found_overload:
                    prev_line = lines[i-1] if i > 0 else ''
                    if not re.match(r'^\s+@overload\s*$', prev_line):
                        result.append(f'{indent}@overload')
        
        result.append(line)
        i += 1
    
    return '\n'.join(result)


def remove_duplicate_overloads(content: str) -> str:
    """Remove duplicate overload signatures that are identical.
    
    SWIG sometimes generates multiple overload signatures that are identical
    in Python (differ only in C++ const-ness). This removes exact duplicates.
    """
    lines = content.split('\n')
    result = []
    seen_signatures = {}  # Maps (class_name, func_signature) -> line_index
    i = 0
    
    current_class = None
    
    while i < len(lines):
        line = lines[i]
        
        # Track current class
        class_match = re.match(r'^class (\w+)', line)
        if class_match:
            current_class = class_match.group(1)
            seen_signatures = {}  # Reset for new class
            result.append(line)
            i += 1
            continue
        
        # Check if this is an @overload decorator
        if re.match(r'^\s+@overload\s*$', line):
            # Get the function signature on next line
            if i + 1 < len(lines):
                func_line = lines[i + 1]
                func_match = re.match(r'^(\s+)(def \w+\([^)]*\)\s*->\s*.*:.*)', func_line)
                if func_match:
                    signature = func_match.group(2).strip()
                    key = (current_class, signature)
                    
                    # Check if we've seen this exact signature before
                    if key in seen_signatures:
                        # Skip this @overload and the function line (duplicate)
                        i += 2
                        continue
                    else:
                        seen_signatures[key] = i
        
        result.append(line)
        i += 1
    
    return '\n'.join(result)


def remove_orphaned_overload_decorators(content: str) -> str:
    """Remove @overload decorators from functions that aren't actually overloaded.
    
    After removing duplicate overloads, some functions may be left with an @overload
    decorator but no actual overloads. This removes those decorators.
    """
    lines = content.split('\n')
    
    # First pass: identify which functions are truly overloaded
    overloaded_functions = set()  # Set of (class_name, func_name) that have multiple versions
    current_class = None
    func_counts = {}  # Count occurrences of each function
    
    for i, line in enumerate(lines):
        class_match = re.match(r'^class (\w+)', line)
        if class_match:
            current_class = class_match.group(1)
            continue
        
        func_match = re.match(r'^\s+def (\w+)\(', line)
        if func_match:
            func_name = func_match.group(1)
            key = (current_class, func_name)
            func_counts[key] = func_counts.get(key, 0) + 1
    
    # Mark functions that appear multiple times as truly overloaded
    for key, count in func_counts.items():
        if count > 1:
            overloaded_functions.add(key)
    
    # Second pass: remove @overload from functions that aren't overloaded
    result = []
    current_class = None
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        class_match = re.match(r'^class (\w+)', line)
        if class_match:
            current_class = class_match.group(1)
            result.append(line)
            i += 1
            continue
        
        # Check if this is an @overload decorator
        if re.match(r'^\s+@overload\s*$', line) and i + 1 < len(lines):
            # Get the function on the next line
            func_line = lines[i + 1]
            func_match = re.match(r'^\s+def (\w+)\(', func_line)
            
            if func_match:
                func_name = func_match.group(1)
                key = (current_class, func_name)
                
                # If this function is NOT truly overloaded, skip the @overload decorator
                if key not in overloaded_functions:
                    i += 1  # Skip the @overload line
                    result.append(func_line)  # Add the function line
                    i += 1
                    continue
        
        result.append(line)
        i += 1
    
    return '\n'.join(result)


def add_missing_final_overload_decorators(content: str) -> str:
    """Add @overload decorator to functions that are missing it but should have it.
    
    When there are multiple overloads of a function, ALL of them should have
    @overload, including the last one. This function finds cases where earlier
    overloads have @overload but the final one doesn't.
    """
    lines = content.split('\n')
    result = []
    i = 0
    
    current_class = None
    
    while i < len(lines):
        line = lines[i]
        
        # Track current class to avoid cross-class pollution
        class_match = re.match(r'^class (\w+)', line)
        if class_match:
            current_class = class_match.group(1)
            result.append(line)
            i += 1
            continue
        
        # Check if this is a function definition WITHOUT @overload
        func_match = re.match(r'^(\s+)def (\w+)\(', line)
        if func_match:
            prev_line = lines[i-1] if i > 0 else ''
            has_overload = re.match(r'^\s+@overload\s*$', prev_line)
            
            if not has_overload:
                indent = func_match.group(1)
                func_name = func_match.group(2)
                
                # Look backward to see if there's another overload of this function WITH @overload
                # BUT stay within the current class
                found_overload_before = False
                for j in range(max(0, i - 20), i):
                    # Stop if we hit a class definition (went out of current class)
                    if re.match(r'^class \w+', lines[j]):
                        break
                    
                    if re.match(rf'^{re.escape(indent)}def {re.escape(func_name)}\(', lines[j]):
                        # Check if this earlier occurrence has @overload
                        if j > 0 and re.match(r'^\s+@overload\s*$', lines[j-1]):
                            found_overload_before = True
                            break
                
                # If we found an earlier overload with @overload, add @overload to this one too
                if found_overload_before:
                    result.append(f'{indent}@overload')
        
        result.append(line)
        i += 1
    
    return '\n'.join(result)


def fix_common_swig_issues(content: str) -> str:
    """Fix common SWIG-generated stub issues."""
    # Fix empty parameter lists that should have self
    content = re.sub(r'def (\w+)\(\)', r'def \1(self)', content)
    
    # Fix malformed overload decorators
    content = re.sub(r'@overload\s*def (\w+)\(self([^)]*)\)([^:]*):(.*)$', 
                     r'@overload\n    def \1(self\2)\3:\4', content, flags=re.MULTILINE)
    
    # Fix trailing commas in parameter lists
    content = re.sub(r'def (\w+)\([^)]*,\s*\)', lambda m: m.group(0).replace(', )', ')'), content)
    
    return content


def post_process_stub_file(file_path: Path) -> None:
    """Post-process a generated stub file to fix common issues."""
    print(f"  Post-processing: {file_path.name}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Apply all fixes
        original_content = content
        content = fix_malformed_self_parameters(content)
        content = fix_duplicate_self_parameters(content)
        content = fix_init_return_types(content)
        content = fix_duplicate_parameters(content)
        content = fix_missing_type_imports(content)
        content = fix_undefined_simbody_types(content)
        content = fix_common_swig_issues(content)
        content = fix_missing_overload_decorators(content)
        content = remove_duplicate_overloads(content)
        content = remove_orphaned_overload_decorators(content)
        content = add_missing_final_overload_decorators(content)
        
        # Only write back if content changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"    [OK] Fixed issues in {file_path.name}")
        else:
            print(f"    [OK] No issues found in {file_path.name}")

    except Exception as e:
        print(f"    [ERROR] Error processing {file_path.name}: {e}")


def generate_stubs_with_stubgen(package_path: Path, output_dir: Path) -> bool:
    """Generate stub files using mypy's stubgen."""
    print(f"Generating stubs for package at: {package_path}")
    
    # Add the package directory to Python path
    if package_path and package_path.exists():
        sys.path.insert(0, str(package_path.parent))
    
    # PyOpenSim modules to generate stubs for
    modules = ['simbody', 'common', 'simulation', 'actuators', 'analyses', 'tools']
    
    success_count = 0
    
    for module in modules:
        module_name = f"pyopensim.{module}"
        print(f"Generating stubs for {module_name}...")
        
        try:
            # Run stubgen for this module
            result = subprocess.run([
                sys.executable, "-m", "mypy.stubgen",
                "-m", module_name,
                "-o", str(output_dir),
                "--ignore-errors"
            ], capture_output=True, text=True, check=False)
            
            if result.returncode == 0:
                print(f"  [OK] Generated stubs for {module_name}")
                success_count += 1
            else:
                print(f"  [WARN] Warning: stubgen had issues with {module_name}")
                if result.stderr:
                    print(f"    stderr: {result.stderr}")
                # Still count as success since stubs are usually generated despite warnings
                success_count += 1

        except Exception as e:
            print(f"  [ERROR] Error generating stubs for {module_name}: {e}")
    
    return success_count > 0


def post_process_all_stubs(output_dir: Path) -> None:
    """Post-process all generated stub files."""
    print("Post-processing generated stub files...")
    
    # Find all .pyi files in the pyopensim directory
    pyopensim_dir = output_dir / "pyopensim"
    if pyopensim_dir.exists():
        stub_files = list(pyopensim_dir.glob("*.pyi"))
        for stub_file in stub_files:
            if stub_file.name != "__init__.pyi":  # Skip our custom __init__.pyi
                post_process_stub_file(stub_file)
    else:
        print(f"  [WARN] Warning: Expected stub directory not found: {pyopensim_dir}")


def create_init_stub(output_dir: Path) -> None:
    """Create the main __init__.pyi file with proper imports and exports.

    This creates a comprehensive stub that matches the runtime behavior of __init__.py,
    enabling IDE autocomplete for both structured imports (pyopensim.simulation.Model)
    and flat imports (pyopensim.Model).
    """
    init_stub_content = '''from typing import Any
from . import actuators, analyses, common, simbody, simulation, tools

# Version is imported from package metadata
__version__: str
__opensim_version__: str

# Optional modules - mirror the runtime try/except behavior
try:
    from . import examplecomponents
except ImportError:
    examplecomponents = None

try:
    from . import moco
except ImportError:
    moco = None

try:
    from . import report
except ImportError:
    report = None

# Re-exported classes from simbody
from .simbody import Vec3, Rotation, Transform, Inertia, Gray, SimTK_PI

# Re-exported classes from common
from .common import Component, Property, Storage, Array, StepFunction, ConsoleReporter

# Re-exported classes from simulation
from .simulation import (
    Model,
    Manager,
    State,
    Body,
    PinJoint,
    PhysicalOffsetFrame,
    Ellipsoid,
    Millard2012EquilibriumMuscle,
    PrescribedController,
    InverseKinematicsSolver,
    InverseDynamicsSolver
)

# Re-exported classes from actuators
from .actuators import Muscle, CoordinateActuator, PointActuator

# Re-exported classes from tools
from .tools import InverseKinematicsTool, InverseDynamicsTool, ForwardTool, AnalyzeTool

__all__ = [
    # Core modules
    'simbody', 'common', 'simulation', 'actuators', 'analyses', 'tools',
    # Optional modules (if available)
    'examplecomponents', 'moco', 'report',
    # Common classes at top level for convenience
    'Model', 'Manager', 'State', 'Body',
    'Component', 'Property',
    'Vec3', 'Rotation', 'Transform', 'Inertia',
    'PinJoint', 'PhysicalOffsetFrame', 'Ellipsoid',
    'Millard2012EquilibriumMuscle', 'PrescribedController',
    'StepFunction', 'ConsoleReporter',
    'Gray', 'SimTK_PI',
    'Storage', 'Array',
    'InverseKinematicsSolver', 'InverseDynamicsSolver',
    'Muscle', 'CoordinateActuator', 'PointActuator',
    'InverseKinematicsTool', 'InverseDynamicsTool',
    'ForwardTool', 'AnalyzeTool',
    '__version__', '__opensim_version__'
]
'''

    init_file = output_dir / "pyopensim" / "__init__.pyi"
    init_file.parent.mkdir(parents=True, exist_ok=True)

    with open(init_file, 'w') as f:
        f.write(init_stub_content)

    print("[OK] Generated main __init__.pyi")


def main():
    """Main stub generation function."""
    if len(sys.argv) < 2:
        print("Usage: generate_stubs.py <output_dir> [package_path]")
        print("  output_dir: Directory where .pyi files will be created")
        print("  package_path: Optional path to built pyopensim package")
        sys.exit(1)
    
    output_dir = Path(sys.argv[1])
    package_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Ensure mypy is available
    if not ensure_mypy_available():
        sys.exit(1)
    
    print("\n" + "="*60)
    print("PYOPENSIM STUB GENERATION")
    print("="*60)
    
    # Generate stubs using stubgen
    if generate_stubs_with_stubgen(package_path, output_dir):
        print("\n" + "-"*40)
        # Post-process the generated stubs to fix issues
        post_process_all_stubs(output_dir)
        
        print("\n" + "-"*40)
        # Create main __init__.pyi file
        create_init_stub(output_dir)

        print("\n" + "="*60)
        print(f"[OK] Stub generation completed successfully!")
        print(f"  Files written to: {output_dir}")
        print(f"  Post-processing applied to fix SWIG-related issues")
        print("="*60)
    else:
        print("[ERROR] Stub generation failed")
        sys.exit(1)


if __name__ == "__main__":
    main()