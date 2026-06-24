import os
import torch
import numpy as np
from qan_transformers.math.e8_swap import AdelicMemorySwapGridDB

_ASCII_SIN_LOOKUP = torch.sin(torch.arange(256, dtype=torch.float32)) + 1.0

class LatticeIndexer:
    def __init__(self, d_model: int = 64, db: AdelicMemorySwapGridDB = None):
        """
        Lattice Indexer for RAG.
        Chunks files, embeds them, and indexes them in the E8 memory swap database.
        """
        self.d_model = d_model
        self.db = db if db is not None else AdelicMemorySwapGridDB(d_model=d_model)
        self.chunks = []

    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 100) -> list:
        """
        Chunks text into overlapping blocks.
        """
        if not text:
            return []
        step = chunk_size - overlap
        if step <= 0:
            return [text[:chunk_size]]
        return [text[i:i+chunk_size] for i in range(0, len(text), step)]

    def embed_chunks(self, chunks: list) -> torch.Tensor:
        """
        Batched generation of deterministic vectors of dimension d_model from a list of text chunks.
        """
        if not chunks:
            return torch.empty((0, self.d_model), dtype=torch.float32)
            
        # Encode strings to bytes first to ensure correct byte lengths for Unicode/non-ASCII characters
        encoded_chunks = [c.encode('ascii', errors='ignore') for c in chunks]
        lengths = [len(b) for b in encoded_chunks]
        max_len = max(lengths)
        if max_len == 0:
            return torch.zeros((len(chunks), self.d_model), dtype=torch.float32)
            
        num_chunks = len(chunks)
        vals_np = np.zeros((num_chunks, max_len), dtype=np.uint8)
        for idx, b in enumerate(encoded_chunks):
            if len(b) > 0:
                vals_np[idx, :len(b)] = np.frombuffer(b, dtype=np.uint8)
                
        vals = torch.from_numpy(vals_np).to(torch.int32)
        arange = torch.arange(1, max_len + 1, dtype=torch.int32).unsqueeze(0)
        
        indices = (vals * arange) % self.d_model
        weights = _ASCII_SIN_LOOKUP[vals.long()]
        
        mask = torch.arange(max_len).unsqueeze(0) < torch.tensor(lengths, dtype=torch.int32).unsqueeze(1)
        # Win 71: Zero-Allocation Branch-Free Masking in RAG replacing torch.where/zeros_like
        weights = mask.to(weights.dtype) * weights
        
        vecs = torch.zeros((len(chunks), self.d_model), dtype=torch.float32)
        vecs.scatter_add_(1, indices.long(), weights)
        
        norms = torch.linalg.norm(vecs, dim=1, keepdim=True)
        vecs = torch.where(norms > 0, vecs / norms, vecs)
        
        return vecs

    def embed_chunk(self, chunk: str) -> torch.Tensor:
        """
        Generates a deterministic vector of dimension d_model from the text chunk.
        """
        if not chunk:
            return torch.zeros(self.d_model, dtype=torch.float32)
        return self.embed_chunks([chunk])[0]

    def index_file(self, file_path: str):
        """
        Chunks, embeds, and indexes a single file.
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception:
            # Skip unreadable/binary files safely
            return
            
        chunks = self.chunk_text(content)
        if not chunks:
            return
            
        # Batched embedding generation and database insertion
        keys = self.embed_chunks(chunks)
        self.db.swap_out(keys, keys)
        
        step = 500 - 100
        for idx, chunk in enumerate(chunks):
            start = idx * step
            end = min(start + 500, len(content))
            self.chunks.append({"file": file_path, "start": start, "end": end})

    def index_directory(self, dir_path: str, batch_size: int = 512):
        """
        Recursively indexes all supported files within a directory in batched fashion.
        """
        # Collect target file paths using fast os.scandir
        target_files = []
        ignored_dirs = {".git", ".venv", "node_modules", "__pycache__", ".pytest_cache", "build", "dist"}
        supported_extensions = {'.txt', '.py', '.md', '.json', '.js', '.ts', '.c', '.cpp', '.h', '.html', '.css', '.toml', '.yaml', '.yml'}
        
        def scan_dir(path):
            try:
                for entry in os.scandir(path):
                    if entry.is_dir(follow_symlinks=False):
                        if entry.name not in ignored_dirs and not entry.name.startswith('.'):
                            scan_dir(entry.path)
                    elif entry.is_file(follow_symlinks=False):
                        _, ext = os.path.splitext(entry.name)
                        if ext.lower() in supported_extensions:
                            target_files.append(entry.path)
            except PermissionError:
                pass
                
        scan_dir(dir_path)
                    
        def process_file(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                chunks = self.chunk_text(content)
                step = 500 - 100
                results = []
                for idx, chunk in enumerate(chunks):
                    start = idx * step
                    end = min(start + 500, len(content))
                    results.append((chunk, file_path, start, end))
                return results
            except Exception:
                return []
                
        # Walk and chunk files in parallel
        from concurrent.futures import ThreadPoolExecutor
        all_results = []
        num_workers = max(1, min(32, os.cpu_count() or 4))
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = executor.map(process_file, target_files)
            for file_results in results:
                all_results.extend(file_results)
                
        # Now batch upload to database using an overlapped pipeline
        import queue
        from threading import Thread

        upload_queue = queue.Queue(maxsize=4)
        
        def db_uploader():
            while True:
                item = upload_queue.get()
                if item is None:
                    break
                keys, chunks_batch, files_batch, starts_batch, ends_batch = item
                try:
                    self.db.swap_out(keys, keys)
                    for f_p, start, end in zip(files_batch, starts_batch, ends_batch):
                        self.chunks.append({"file": f_p, "start": start, "end": end})
                finally:
                    upload_queue.task_done()
                    
        uploader_thread = Thread(target=db_uploader, daemon=True)
        uploader_thread.start()
        
        pending_chunks = []
        pending_files = []
        pending_starts = []
        pending_ends = []
        for chunk, file_path, start, end in all_results:
            pending_chunks.append(chunk)
            pending_files.append(file_path)
            pending_starts.append(start)
            pending_ends.append(end)
            
            if len(pending_chunks) >= batch_size:
                keys = self.embed_chunks(pending_chunks)
                upload_queue.put((keys, pending_chunks, pending_files, pending_starts, pending_ends))
                pending_chunks = []
                pending_files = []
                pending_starts = []
                pending_ends = []
                
        if pending_chunks:
            keys = self.embed_chunks(pending_chunks)
            upload_queue.put((keys, pending_chunks, pending_files, pending_starts, pending_ends))
            
        # Wait for all uploads to complete
        upload_queue.join()
        upload_queue.put(None)
        uploader_thread.join()

    def query(self, query_text: str, distance_threshold: float = 2.05) -> list:
        """
        Queries the E8 database using E8 neighborhood search (distance squared <= distance_threshold).
        Returns a list of dictionaries with matching chunks: [{"file": ..., "text": ...}]
        """
        if not self.chunks or self.db.grid_coords is None or self.db.grid_coords.shape[0] == 0:
            return []
            
        # Generate query embedding vector
        q_vec = self.embed_chunk(query_text)
        
        # Project using the database's target projection
        device = self.db.grid_coords.device
        dtype = self.db.grid_coords.dtype
        
        if not hasattr(self.db, "W_p_target"):
            self.db._init_default_projection(device, dtype, is_draft=False)
            
        W_p = self.db.W_p_target.to(device=device, dtype=dtype)
        q_vec = q_vec.to(device=device, dtype=dtype)
        
        # Project and quantize to E8 lattice coordinates
        q_8d = q_vec @ W_p
        q_quant = self.db._quantize(q_8d.unsqueeze(0)) # shape [1, 8]
        
        # Win 100: Precomputed Coordinate Maps and zero-copy caching in Lattice RAG Indexer
        if getattr(self, "_cached_grid_coords_id", None) != id(self.db.grid_coords) or getattr(self, "_cached_device", None) != device or getattr(self, "_cached_dtype", None) != dtype:
            self._cached_grid_coords_id = id(self.db.grid_coords)
            self._cached_device = device
            self._cached_dtype = dtype
            coords = self.db.grid_coords.to(device=device, dtype=dtype)
            self._cached_coords = coords
            self._cached_coords_norm2 = torch.sum(coords.square(), dim=-1).unsqueeze(0)
            
        coords = self._cached_coords
        coords_norm2 = self._cached_coords_norm2
        
        q_norm2 = torch.sum(q_quant.square(), dim=-1, keepdim=True)
        prod = torch.matmul(q_quant, coords.t())
        dist2 = q_norm2 + coords_norm2 - 2 * prod  # shape [1, M]
        dist2 = dist2[0]  # shape [M]
        
        # Mask matching neighborhood coordinates
        mask = dist2 <= distance_threshold
        matched_indices = torch.where(mask)[0]
        
        if len(matched_indices) == 0:
            return []
            
        matched_dists = dist2[matched_indices]
        
        # Limit to top 10 closest chunks
        top_k = min(10, len(matched_indices))
        sorted_dists, sort_idx = torch.topk(matched_dists, k=top_k, largest=False)
        top_indices = matched_indices[sort_idx].cpu().tolist()
        
        # Map matching indices back to chunks
        results = []
        for idx in top_indices:
            if idx < len(self.chunks):
                chunk_meta = self.chunks[idx]
                file_path = chunk_meta["file"]
                if "text" in chunk_meta:
                    chunk_text = chunk_meta["text"]
                else:
                    start = chunk_meta["start"]
                    end = chunk_meta["end"]
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read(end)
                            chunk_text = content[start:end]
                    except Exception:
                        chunk_text = ""
                results.append({"file": file_path, "text": chunk_text})
        return results


def crawl_codebase(folder_path: str) -> str:
    """
    Recursively walks the codebase folder and concatenates all text/code files
    into a single corpus, ignoring binary, cache, and hidden directories.
    """
    ignored_dirs = {".git", ".venv", "node_modules", "__pycache__", ".pytest_cache", "build", "dist", ".agents"}
    supported_extensions = {
        # Source code
        ".py", ".js", ".ts", ".c", ".cpp", ".h", ".rs", ".sh", ".go", ".java", ".kt",
        # Config & Web markup
        ".json", ".toml", ".yaml", ".yml", ".html", ".css", ".md", ".ini", ".cfg"
    }
    
    corpus_parts = []
    
    def scan_dir(path):
        try:
            for entry in os.scandir(path):
                if entry.is_dir(follow_symlinks=False):
                    if entry.name not in ignored_dirs and not entry.name.startswith('.'):
                        scan_dir(entry.path)
                elif entry.is_file(follow_symlinks=False):
                    _, ext = os.path.splitext(entry.name)
                    if ext.lower() in supported_extensions:
                        rel_path = entry.path[len(folder_path):].lstrip(os.sep)
                        try:
                            with open(entry.path, 'r', encoding='utf-8', errors='ignore') as f:
                                content = f.read()
                            header = f"\n--- FILE: {rel_path} ---\n"
                            corpus_parts.append(header + content)
                        except Exception:
                            continue
        except PermissionError:
            pass

    scan_dir(folder_path)
    return "".join(corpus_parts)


