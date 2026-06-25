import torch
import numpy as np
from qan_transformers.math.e8_projection import generate_e8_coordinates, project_e8_to_quasicrystal

def e8_quasicrystal_projection():
    # 1. Generate 240 root coordinates in 8D
    roots_8d = generate_e8_coordinates(norm=np.sqrt(2.0))
    
    # 2. Project 8D coordinates to 3D concentric shells using the Icosian method
    coords_3d = project_e8_to_quasicrystal(roots_8d, method="icosian")
    
    # 3. Calculate norms to verify grouping into 5 discrete shells
    norms = np.linalg.norm(coords_3d, axis=1)
    unique_shells = np.unique(np.round(norms, 4))
    
    return coords_3d, unique_shells

def test_e8_projection_snippet():
    coords_3d, unique_shells = e8_quasicrystal_projection()
    
    # 240 root vectors should be generated and projected
    assert coords_3d.shape == (240, 3)
    
    # Check that it groups into 5 discrete shells (as described in the document)
    # The document says: "organize themselves into exactly 5 concentric shells"
    print(f"Unique shells found: {unique_shells}")
    assert len(unique_shells) == 5, f"Expected 5 concentric shells, found {len(unique_shells)}"
