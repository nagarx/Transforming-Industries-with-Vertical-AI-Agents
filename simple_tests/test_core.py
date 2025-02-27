"""
Simple tests for the core functionalities of agentic_systems.
These tests don't require external dependencies and can run in CI environments.
"""

def test_package_version():
    """Test that the package version is properly defined."""
    from agentic_systems import __version__
    assert isinstance(__version__, str)
    assert len(__version__.split('.')) >= 2  # Verify it has at least major.minor format
    
def test_module_imports():
    """Test that core modules can be imported."""
    # Import core modules to ensure they exist
    import agentic_systems.core
    import agentic_systems.agents
    
    # Check that the modules exist
    assert hasattr(agentic_systems, 'core')
    assert hasattr(agentic_systems, 'agents')
    
def test_package_structure():
    """Test that the package structure is as expected."""
    import agentic_systems
    
    # Check for expected attributes/modules
    assert hasattr(agentic_systems, '__version__')
    
    # These assertions verify that the main package structure exists
    # without actually importing potentially problematic modules
    import importlib.util
    
    # Check that key files exist
    for module_path in [
        'agentic_systems.core',
        'agentic_systems.agents',
        'agentic_systems.core.memory',
        'agentic_systems.core.tools',
    ]:
        spec = importlib.util.find_spec(module_path)
        assert spec is not None, f"Module {module_path} should exist" 