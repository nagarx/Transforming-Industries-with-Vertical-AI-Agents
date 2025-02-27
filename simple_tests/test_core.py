"""
Simple tests for the core functionalities of agentic_systems.
These tests don't require external dependencies and can run in CI environments.
"""

import os

def test_package_version():
    """Test that the package version is properly defined."""
    from agentic_systems import __version__
    assert isinstance(__version__, str)
    assert len(__version__.split('.')) >= 2  # Verify it has at least major.minor format

def test_package_exists():
    """Test that the package directory structure exists."""
    import agentic_systems
    # Just verify the package can be imported
    assert agentic_systems is not None
    
def test_directory_structure():
    """Test that the basic directory structure is in place."""
    # Get the agentic_systems package location
    import agentic_systems
    base_dir = os.path.dirname(agentic_systems.__file__)
    
    # Check for key directories only (not modules that might trigger imports)
    dirs_to_check = [
        "core",
        "agents",
        os.path.join("core", "memory"),
        os.path.join("core", "tools"),
    ]
    
    for d in dirs_to_check:
        dir_path = os.path.join(base_dir, d)
        assert os.path.isdir(dir_path), f"Directory {d} should exist"
        # Check for __init__.py in each directory
        init_path = os.path.join(dir_path, "__init__.py")
        assert os.path.isfile(init_path), f"__init__.py should exist in {d}"

def test_module_imports():
    """Test that core modules can be imported without loading dependencies."""
    # Check if modules exist without actually importing them
    # This avoids importing modules that have external dependencies
    for module_path in [
        'agentic_systems.core',
        'agentic_systems.agents',
        'agentic_systems.core.memory',
        'agentic_systems.core.tools',
    ]:
        spec = importlib.util.find_spec(module_path)
        assert spec is not None, f"Module {module_path} should exist"
    
    # Only import the base package which shouldn't have external dependencies
    import agentic_systems
    assert hasattr(agentic_systems, 'core')
    assert hasattr(agentic_systems, 'agents')
    
def test_package_structure():
    """Test that the package structure is as expected."""
    import agentic_systems
    
    # Check for expected attributes/modules
    assert hasattr(agentic_systems, '__version__')
    
    # Check for expected module files
    for module_path in [
        'agentic_systems.core.memory.base_memory',
        'agentic_systems.agents.base_agent',
        'agentic_systems.core.tools.base_tool',
    ]:
        spec = importlib.util.find_spec(module_path)
        assert spec is not None, f"Module {module_path} should exist" 