import torch
import torch.nn.functional as F

_SHARED_E8_ROOTS = {}
_SHARED_E8_ROOTS_8D = {}

def get_shared_e8_roots(lvl):
    if lvl not in _SHARED_E8_ROOTS:
        from qan_transformers.math.e8_projection import generate_dynamic_e8_coordinates, project_e8_to_quasicrystal
        roots_8d = generate_dynamic_e8_coordinates(lvl)
        roots_3d = torch.tensor(project_e8_to_quasicrystal(roots_8d), dtype=torch.float32)
        roots_3d_norm = F.normalize(roots_3d, p=2, dim=-1, eps=1e-6)
        _SHARED_E8_ROOTS[lvl] = (roots_3d, roots_3d_norm)
    return _SHARED_E8_ROOTS[lvl]

def get_shared_e8_roots_8d(lvl):
    if lvl not in _SHARED_E8_ROOTS_8D:
        from qan_transformers.math.e8_projection import generate_dynamic_e8_coordinates
        roots_8d = torch.tensor(generate_dynamic_e8_coordinates(lvl), dtype=torch.float32)
        roots_8d_norm = F.normalize(roots_8d, p=2, dim=-1, eps=1e-6)
        _SHARED_E8_ROOTS_8D[lvl] = (roots_8d, roots_8d_norm)
    return _SHARED_E8_ROOTS_8D[lvl]
