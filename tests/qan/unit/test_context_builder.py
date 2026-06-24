import pytest
from qan_transformers.math.context_builder import format_xml_context, build_tree_structure

def test_format_xml_context_default():
    files = {
        "src/main.py": "print('hello')",
        "README.md": "# Title"
    }
    corpus = format_xml_context(files)
    
    assert "GossetGate" in corpus
    assert '<workspace_context>' in corpus
    assert '<file path="src/main.py">' in corpus
    assert '<![CDATA[\nprint(\'hello\')\n]]>' in corpus
    assert '<file path="README.md">' in corpus
    assert '<![CDATA[\n# Title\n]]>' in corpus
    assert '</workspace_context>' in corpus

def test_format_xml_context_custom_template():
    files = {
        "main.py": "foo()"
    }
    custom_template = "INSTRUCTIONS\nFILES:\n{file_contents}\nEND"
    corpus = format_xml_context(files, template=custom_template)
    
    assert corpus.startswith("INSTRUCTIONS")
    assert "FILES:\n<file path=\"main.py\">\n<![CDATA[\nfoo()\n]]>\n</file>\nEND" in corpus

def test_build_tree_structure():
    files = {
        "src/main.py": {"size": 100, "tokens": 10},
        "src/utils/helpers.py": {"size": 200, "tokens": 20},
        "README.md": {"size": 50, "tokens": 5}
    }
    tree = build_tree_structure(files)
    
    # Root level assertions
    assert "src" in tree
    assert tree["src"]["type"] == "directory"
    assert "README.md" in tree
    assert tree["README.md"]["type"] == "file"
    assert tree["README.md"]["size"] == 50
    assert tree["README.md"]["tokens"] == 5
    
    # Nested level assertions
    src_children = tree["src"]["children"]
    assert "main.py" in src_children
    assert src_children["main.py"]["type"] == "file"
    assert src_children["main.py"]["size"] == 100
    assert src_children["main.py"]["tokens"] == 10
    
    assert "utils" in src_children
    assert src_children["utils"]["type"] == "directory"
    
    # Double nested assertions
    utils_children = src_children["utils"]["children"]
    assert "helpers.py" in utils_children
    assert utils_children["helpers.py"]["type"] == "file"
    assert utils_children["helpers.py"]["size"] == 200
    assert utils_children["helpers.py"]["tokens"] == 20

def test_wrap_position_ids():
    import torch
    import numpy as np
    from qan_transformers.math.context_builder import wrap_position_ids
    
    # Test scalar
    assert wrap_position_ids(4100, 4096) == 4
    
    # Test list
    assert wrap_position_ids([4100, 4098], 4096) == [4, 2]
    
    # Test PyTorch Tensor
    t = torch.tensor([4100, 4098])
    res_t = wrap_position_ids(t, 4096)
    assert torch.equal(res_t, torch.tensor([4, 2]))
    
    # Test NumPy array
    arr = np.array([4100, 4098])
    res_arr = wrap_position_ids(arr, 4096)
    assert np.array_equal(res_arr, np.array([4, 2]))
