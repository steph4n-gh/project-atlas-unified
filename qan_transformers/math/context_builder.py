import re
import os

DEFAULT_SYSTEM_TEMPLATE = (
    "You are GossetGate, a premium AI developer assistant powered by the Quasicrystalline Attention Network.\n"
    "Below is the loaded workspace/document context. The files have been prefills-loaded directly into your attention cache:\n\n"
    "<workspace_context>\n"
    "{file_contents}\n"
    "</workspace_context>\n\n"
    "Crucial Guidelines:\n"
    "1. You have 100% visibility into these files. When the user asks about classes, functions, or imports, search this context first.\n"
    "2. If you answer referencing code from a specific file, always mention the file path explicitly (e.g. `[main.py](file:///path/to/main.py)`).\n"
    "3. Keep code modifications accurate, minimal, and fully compatible with the existing structure."
)

CDATA_END_RE = re.compile(r'\]\]\s*\>')

def build_xml_file_block(path: str, content: str) -> str:
    """
    Wraps file path and content in structured XML tags.
    Uses CDATA sections to preserve raw code symbols, syntax, and formatting.
    """
    # Clean CDATA markers if already present to prevent nested errors
    if ']]>' in content:
        sanitized_content = CDATA_END_RE.sub(']]&gt;', content)
    else:
        sanitized_content = content
    return f'<file path="{path}">\n<![CDATA[\n{sanitized_content}\n]]>\n</file>'

_XML_BLOCK_CACHE = {}

def format_xml_context(files: dict, template: str = None) -> str:
    """
    Takes a dictionary of {file_path: file_content} and returns a single combined string
    structured as a system prompt instruction enclosing the XML-demarcated codebase context.
    """
    global _XML_BLOCK_CACHE
    if template is None:
        template = DEFAULT_SYSTEM_TEMPLATE
        
    blocks = []
    for path, content in files.items():
        try:
            if os.path.isabs(path) and os.path.exists(path):
                stat = os.stat(path)
                cache_key = (path, stat.st_size, stat.st_mtime)
            else:
                cache_key = (path, len(content), hash(content))
        except Exception:
            cache_key = (path, len(content), hash(content))
            
        if cache_key in _XML_BLOCK_CACHE:
            block = _XML_BLOCK_CACHE[cache_key]
        else:
            block = build_xml_file_block(path, content)
            _XML_BLOCK_CACHE[cache_key] = block
        blocks.append(block)
        
    if len(_XML_BLOCK_CACHE) > 5000:
        _XML_BLOCK_CACHE.clear()
        
    file_contents_str = "\n\n".join(blocks)
    return template.format(file_contents=file_contents_str)

def build_tree_structure(files: dict) -> dict:
    """
    Parses a flat dictionary of {file_path: file_metadata} into a nested tree structure
    suitable for UI tree visualizers.
    """
    tree = {}
    # Win 114: Prefix node caching in directory tree builder
    dirs_cache = {}
    
    for path, meta in files.items():
        parts = path.split('/')
        current = tree
        prefix = ""
        for part in parts[:-1]:
            prefix = f"{prefix}/{part}" if prefix else part
            if prefix not in dirs_cache:
                # Directory node
                if part not in current or not isinstance(current[part], dict) or current[part].get("type") == "file":
                    current[part] = {"type": "directory", "children": {}}
                dirs_cache[prefix] = current[part]["children"]
            current = dirs_cache[prefix]
        
        filename = parts[-1]
        size = meta.get("size")
        if size is None:
            size = len(meta.get("content", ""))
        current[filename] = {
            "type": "file",
            "path": path,
            "size": size,
            "tokens": meta.get("tokens", 0)
        }
    return tree

def wrap_position_ids(position_ids, max_pos):
    """
    Vectorized wrap-around for position IDs/coordinates to stay within sequence length limits.
    Efficiently handles PyTorch tensors, NumPy arrays, lists, or scalars.
    """
    if hasattr(position_ids, "device"):  # PyTorch tensor or similar
        import torch
        if isinstance(position_ids, torch.Tensor):
            return torch.remainder(position_ids, max_pos)
    
    # Check if numpy array
    if hasattr(position_ids, "ndim"):
        import numpy as np
        if isinstance(position_ids, np.ndarray):
            return np.remainder(position_ids, max_pos)
            
    if isinstance(position_ids, (list, tuple)):
        return [p % max_pos for p in position_ids]
        
    return position_ids % max_pos

def crawl_codebase(folder_path: str, ignored_dirs=None, supported_extensions=None) -> dict:
    """
    Recursively walks the codebase folder and returns a dictionary mapping
    relative file paths to their text/code contents, ignoring binary, cache,
    and hidden directories.
    Optimized with set-based memberships and pre-compiled filters.
    """
    if ignored_dirs is None:
        ignored_dirs = {".git", ".venv", "node_modules", "__pycache__", ".pytest_cache", "build", "dist", ".agents"}
    else:
        ignored_dirs = set(ignored_dirs)
        
    if supported_extensions is None:
        supported_extensions = {
            # Source code
            ".py", ".js", ".ts", ".c", ".cpp", ".h", ".rs", ".sh", ".go", ".java", ".kt",
            # Config & Web markup
            ".json", ".toml", ".yaml", ".yml", ".html", ".css", ".md", ".ini", ".cfg"
        }
    else:
        supported_extensions = set(supported_extensions)
        
    files_dict = {}
    
    for root, dirs, files in os.walk(folder_path):
        # Exclude ignored directories in-place using fast set lookup
        dirs[:] = [d for d in dirs if d not in ignored_dirs and not d.startswith('.')]
        
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in supported_extensions:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, folder_path)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    files_dict[rel_path] = content
                except Exception:
                    continue
                    
    return files_dict
