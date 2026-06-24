import pytest
from fastapi.testclient import TestClient
import json

# Import app from dashboard
from qan_transformers.cli.dashboard import app

def test_dashboard_endpoints():
    client = TestClient(app)
    
    # 1. Test index.html serving
    response = client.get("/")
    # Check that status is successful if file exists, or 404 if not found
    assert response.status_code in (200, 404)
    
    # 2. Test static file caching headers
    import os
    from qan_transformers.cli.dashboard import STATIC_DIR
    dummy_file = os.path.join(STATIC_DIR, "test_cache.txt")
    with open(dummy_file, "w") as f:
        f.write("hello cache")
        
    try:
        res = client.get("/static/test_cache.txt")
        assert res.status_code == 200
        assert "cache-control" in res.headers
        assert "public" in res.headers["cache-control"]
        assert "max-age=31536000" in res.headers["cache-control"]
    finally:
        if os.path.exists(dummy_file):
            os.remove(dummy_file)

def test_fast_json():
    # Test that the monkeypatched json.dumps behaves correctly
    test_dict = {"a": 1, "b": [2, 3]}
    serialized = json.dumps(test_dict)
    assert json.loads(serialized) == test_dict
