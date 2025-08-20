# Refactor one line test into test suite with manual ray.init() based on following instructions:
# https://github.com/modin-project/modin/blob/1551d01e7ec9ba140b9bf0dbd88ebc15bdcf4e27/docs/getting_started/using_modin/using_modin_locally.rst?plain=1#L59-L66

import os
import pytest

# --- Configuration before imports ---
os.environ["MODIN_ENGINE"] = "ray"
os.environ["RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE"] = "1"
os.environ["RAY_DISABLE_IMPORT_WARNING"] = "1"
os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["RAY_ENABLE_MAC_LARGE_OBJECT_STORE"] = "1"

import ray
import modin.pandas as pd
import modin.config as cfg

@pytest.fixture(scope="module", autouse=True)
def setup_ray_cluster():
    """
    Explicitly initializes Ray with specified resources,
    then configures Modin to use this cluster.
    """
    # Make sure Ray is not already initialized
    if ray.is_initialized():
        ray.shutdown()
    
    # Initialize Ray with specified resources
    ray.init(
        num_cpus=4,
        object_store_memory=8 * 1024**3,  # 8 GB
        ignore_reinit_error=True,
        include_dashboard=False,
        log_to_driver=False,
        _system_config={
            "object_store_full_delay_ms": 100,
        }
    )
    
    # Check if Ray is properly initialized
    assert ray.is_initialized(), "Ray initialization failed"
    
    # Now Modin should use the already initialized Ray cluster
    print(f"Ray cluster resources: {ray.cluster_resources()}")
    print(f"Ray available resources: {ray.available_resources()}")
    
    yield
    
    # Cleanup after tests
    if ray.is_initialized():
        ray.shutdown()

def test_ray_cluster_status():
    """Checks Ray cluster status"""
    assert ray.is_initialized(), "Ray should be initialized"
    
    cluster_resources = ray.cluster_resources()
    available_resources = ray.available_resources()
    
    print(f"Cluster resources: {cluster_resources}")
    print(f"Available resources: {available_resources}")
    
    # Check if cluster has CPU resources
    assert "CPU" in cluster_resources, "Ray cluster should have CPU resources"
    assert cluster_resources["CPU"] == 4.0, f"Expected 4 CPUs, got: {cluster_resources['CPU']}"

def test_basic_ray_operations():
    """Tests basic Ray operations"""
    
    @ray.remote
    def simple_task(x):
        return x * 2
    
    # Execute task
    future = simple_task.remote(5)
    result = ray.get(future)
    
    assert result == 10, f"Expected 10, got: {result}"

def test_modin_engine_is_ray():
    """Checks if Modin uses Ray backend"""
    assert cfg.Engine.get().lower() == "ray", "Modin is not using Ray backend"

def test_modin_dataframe_creation():
    """Tests Modin DataFrame creation"""
    try:
        # Create simple DataFrame
        df = pd.DataFrame({"x": range(5), "y": range(5, 10)})
        
        # Check basic properties
        assert len(df) == 5, f"Expected 5 rows, got: {len(df)}"
        assert list(df.columns) == ["x", "y"], f"Unexpected columns: {df.columns}"
        
        # Check if data is correct
        assert df["x"].tolist() == list(range(5)), "Column 'x' has unexpected values"
        assert df["y"].tolist() == list(range(5, 10)), "Column 'y' has unexpected values"
        
    except Exception as e:
        pytest.fail(f"DataFrame creation failed: {e}")

def test_modin_dataframe_operations():
    """Tests Modin DataFrame operations"""
    try:
        # Create DataFrame
        df = pd.DataFrame({"x": range(10), "y": range(10, 20)})
        
        # Execute operations
        result_apply = df["x"].apply(lambda v: v * 2)
        result_sum = df["x"].sum()
        result_mean = df["y"].mean()
        
        # Check results
        expected_apply = [i * 2 for i in range(10)]
        assert result_apply.tolist() == expected_apply, "Apply operation failed"
        
        assert result_sum == sum(range(10)), f"Sum operation failed: {result_sum}"
        assert result_mean == sum(range(10, 20)) / 10, f"Mean operation failed: {result_mean}"
        
    except Exception as e:
        pytest.fail(f"DataFrame operations failed: {e}")

def test_modin_groupby_operations():
    """Tests Modin groupby operations"""
    try:
        # Create DataFrame with groups
        df = pd.DataFrame({
            "group": ["A", "B", "A", "B", "A", "B"],
            "value": [1, 2, 3, 4, 5, 6]
        })
        
        # Execute grouping
        grouped = df.groupby("group")["value"].sum()
        
        # Check results (A: 1+3+5=9, B: 2+4+6=12)
        result_dict = grouped.to_dict()
        expected = {"A": 9, "B": 12}
        
        assert result_dict == expected, f"GroupBy failed. Expected: {expected}, Got: {result_dict}"
        
    except Exception as e:
        pytest.fail(f"GroupBy operations failed: {e}")