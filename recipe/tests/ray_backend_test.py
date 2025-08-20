import os
import pytest
import time

# --- Configuration before imports ---
os.environ["MODIN_ENGINE"] = "ray"
os.environ["RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE"] = "1"
os.environ["RAY_DISABLE_IMPORT_WARNING"] = "1"
os.environ["RAY_DEDUP_LOGS"] = "0"
# Linux-specific optimizations
os.environ["OMP_NUM_THREADS"] = "1"  # Prevent thread oversubscription

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
    
    # Initialize Ray with minimal configuration suitable for low-memory environments
    ray.init(
        num_cpus=1,  # Further reduced for stability
        object_store_memory=100 * 1024**2,  # Only 100MB to fit in small /dev/shm
        ignore_reinit_error=True,
        include_dashboard=False,
        log_to_driver=False,
        # Force using /tmp for object store due to small /dev/shm
        _plasma_directory="/tmp",
    )
    
    # Check if Ray is properly initialized with retry
    max_retries = 3
    for attempt in range(max_retries):
        if ray.is_initialized():
            break
        time.sleep(1)
        if attempt == max_retries - 1:
            pytest.fail("Ray initialization failed after retries")
    
    # Wait for Ray cluster to stabilize
    time.sleep(2)
    
    # Verify Ray cluster is healthy with longer wait
    try:
        # Wait longer for cluster to initialize
        time.sleep(5)
        
        cluster_resources = ray.cluster_resources()
        if not cluster_resources:
            # Try one more time with additional wait
            time.sleep(5)
            cluster_resources = ray.cluster_resources()
            
        if not cluster_resources:
            pytest.skip("Ray cluster failed to start properly - likely /dev/shm size issue")
            
    except Exception as e:
        pytest.skip(f"Ray cluster is unhealthy: {e}")
    
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
    
    # Check if cluster has CPU resources (more flexible)
    assert "CPU" in cluster_resources, "Ray cluster should have CPU resources"
    cpu_count = cluster_resources.get("CPU", 0)
    assert cpu_count >= 0.5, f"Expected at least 0.5 CPU, got: {cpu_count}"

def test_modin_engine_is_ray():
    """Checks if Modin uses Ray backend"""
    assert cfg.Engine.get().lower() == "ray", "Modin is not using Ray backend"

def test_basic_ray_operations():
    """Tests basic Ray operations with error handling"""
    
    @ray.remote
    def simple_task(x):
        return x * 2
    
    try:
        # Execute task with timeout
        future = simple_task.remote(5)
        result = ray.get(future, timeout=30)  # 30 second timeout
        
        assert result == 10, f"Expected 10, got: {result}"
        
    except ray.exceptions.RayTimeoutError:
        pytest.fail("Ray task timed out - cluster may be unstable")
    except ray.exceptions.LocalRayletDiedError:
        pytest.fail("Local raylet died during task execution")
    except Exception as e:
        pytest.fail(f"Ray task failed: {e}")

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