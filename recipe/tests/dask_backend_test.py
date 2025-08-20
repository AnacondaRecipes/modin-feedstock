import os
import pytest

# --- configuration before importing modin ---
os.environ["MODIN_ENGINE"] = "dask"   # force Modin to use Dask backend

import modin.pandas as pd
import modin.config as cfg
from distributed import Client, default_client, LocalCluster

@pytest.fixture(scope="module", autouse=True)
def dask_shutdown_after_tests():
    """Ensure Dask cluster is closed after all tests."""
    yield
    # If no client was started, this will raise ValueError
    client = default_client()
    client.close()

def test_modin_engine_is_dask():
    # Check if Modin engine is Dask
    assert cfg.Engine.get().lower() == "dask", "Modin is not using Dask backend"

def test_dask_initialized():
    # Force Modin to start Dask by creating a minimal DataFrame
    _ = pd.DataFrame({"x": []})

    # Check if a Dask client has been started by Modin
    client = default_client()
    assert isinstance(client, Client), "Dask Client is not initialized by Modin"
    # Check number of workers/threads
    assert client.nthreads()  # should return dict with at least 1 worker

def test_modin_dataframe_operations():
    # Create a simple DataFrame
    df = pd.DataFrame({"x": range(10), "y": range(10, 20)})

    # Apply a simple function in parallel
    result = df["x"].apply(lambda v: v * 2)

    # Verify correctness of computation
    expected = [i * 2 for i in range(10)]
    assert result.tolist() == expected, "Apply operation returned unexpected results"