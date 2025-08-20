import os
import pytest

# --- configuration before importing modin ---
os.environ["MODIN_ENGINE"] = "ray"   # force Modin to use Ray backend
os.environ["MODIN_CPUS"] = "4"       # limit number of CPUs to 4
os.environ["MODIN_MEMORY"] = str(4 * 1024**3)  # 4 GB

import ray
import modin.pandas as pd
import modin.config as cfg

@pytest.fixture(scope="module", autouse=True)
def ray_shutdown_after_tests():
    """Ensure Ray is shut down after all tests."""
    yield
    if ray.is_initialized():
        ray.shutdown()

def test_modin_engine_is_ray():
    # Check if Modin engine is Ray
    assert cfg.Engine.get().lower() == "ray", "Modin is not using Ray backend"

def test_ray_initialized():
    # Force Modin to start Ray by creating a minimal DataFrame
    _ = pd.DataFrame({"x": []})

    # Check if Ray has been initialized by Modin
    assert ray.is_initialized(), "Ray is not initialized by Modin"

def test_ray_cpu_limit():
    # Check if Ray reports correct CPU resources
    resources = ray.available_resources()
    assert "CPU" in resources, "Ray does not report CPU resource"
    assert int(resources["CPU"]) == 4, f"Expected 4 CPUs, got: {resources['CPU']}"

def test_modin_dataframe_operations():
    # Create a simple DataFrame
    df = pd.DataFrame({"x": range(10), "y": range(10, 20)})

    # Apply a simple function in parallel
    result = df["x"].apply(lambda v: v * 2)

    # Verify correctness of computation
    expected = [i * 2 for i in range(10)]
    assert result.tolist() == expected, "Apply operation returned unexpected results"