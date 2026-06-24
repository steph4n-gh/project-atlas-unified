import numpy as np
from qan_transformers.math.e8_projection import (
    generate_e8_coordinates,
    project_e8_to_quasicrystal,
    verify_quasicrystalline_symmetries,
)

def test_icosian_symmetries():
    coords_8d = generate_e8_coordinates()
    coords_3d = project_e8_to_quasicrystal(coords_8d, method="icosian")
    
    results = verify_quasicrystalline_symmetries(coords_3d)
    
    assert results["passes_icosahedral"], f"Icosian failed icosahedral symmetry. Max error: {results['icosahedral_max_error']}"
    assert results["passes_inversion"], f"Icosian failed inversion symmetry. Max error: {results['inversion_max_error']}"
    assert results["passes_symmetry"], "Icosian failed overall symmetry verification"
    assert results["icosahedral_max_error"] < 1e-7
    assert results["inversion_max_error"] < 1e-7

def test_coxeter_symmetries():
    coords_8d = generate_e8_coordinates()
    coords_3d = project_e8_to_quasicrystal(coords_8d, method="coxeter")
    
    results = verify_quasicrystalline_symmetries(coords_3d)
    
    assert results["passes_icosahedral"], f"Coxeter failed icosahedral symmetry. Max error: {results['icosahedral_max_error']}"
    assert results["passes_inversion"], f"Coxeter failed inversion symmetry. Max error: {results['inversion_max_error']}"
    assert results["passes_symmetry"], "Coxeter failed overall symmetry verification"
    assert results["icosahedral_max_error"] < 1e-7
    assert results["inversion_max_error"] < 1e-7
