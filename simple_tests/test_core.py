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