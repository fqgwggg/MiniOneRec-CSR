"""Deterministic CSR construction, validation and immutable serialization."""
import hashlib
import json
from pathlib import Path
import numpy as np
from .reference_constraints import canonical_paths

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()

class CSRIndex:
    def __init__(self, paths, eos=2):
        self.paths = canonical_paths(paths, eos)
        self.eos = eos
        prefixes = {()}
        for path in self.paths:
            prefixes.update(path[:d] for d in range(1, len(path)))
        ordered = sorted(prefixes, key=lambda p: (len(p), p))
        node_ids = {p: i for i, p in enumerate(ordered)}
        terminal_id = len(ordered) if self.paths else None
        edges = [dict() for _ in range(len(ordered) + bool(self.paths))]
        for path in self.paths:
            for depth, token in enumerate(path):
                parent = node_ids[path[:depth]]
                child = terminal_id if token == eos else node_ids[path[:depth+1]]
                if token in edges[parent] and edges[parent][token] != child:
                    raise ValueError('Conflicting token transition')
                edges[parent][token] = child
        ptr, tokens, children = [0], [], []
        for row in edges:
            for token, child in sorted(row.items()):
                tokens.append(token)
                children.append(child)
            ptr.append(len(tokens))
        if max(len(tokens), len(edges), max(tokens, default=0)) >= 2**31:
            raise ValueError('int32 capacity exceeded')
        self.row_ptr = np.array(ptr, dtype=np.int32)
        self.token_ids = np.array(tokens, dtype=np.int32)
        self.next_nodes = np.array(children, dtype=np.int32)
        self.terminal = np.zeros(len(edges), dtype=bool)
        if terminal_id is not None:
            self.terminal[terminal_id] = True
        self.validate()

    def validate(self):
        p, t, n = self.row_ptr, self.token_ids, self.next_nodes
        if len(p) != len(self.terminal)+1 or p[0] != 0 or p[-1] != len(t) or len(t) != len(n) or (np.diff(p) < 0).any():
            raise ValueError('Invalid CSR pointers')
        if (t < 0).any() or (n < 0).any() or (n >= len(self.terminal)).any():
            raise ValueError('CSR token/node range error')
        for i in range(len(self.terminal)):
            row = t[p[i]:p[i+1]]
            if len(row) > 1 and (np.diff(row) <= 0).any():
                raise ValueError('CSR edges must have unique sorted tokens')
            if self.terminal[i] and len(row):
                raise ValueError('Terminal node has outgoing edges')
        return True

    @property
    def metadata(self):
        degrees = np.diff(self.row_ptr)
        return {'nodes': len(self.terminal), 'edges': len(self.token_ids), 'max_degree': int(degrees.max(initial=0)),
                'array_bytes': sum(a.nbytes for a in [self.row_ptr, self.token_ids, self.next_nodes, self.terminal]),
                'padded_edge_slots': int(len(self.terminal)*degrees.max(initial=0)),
                'constraint_set_sha256': digest(self.paths), 'eos': self.eos}

    def padded(self):
        width = max(1, self.metadata['max_degree'])
        tokens = np.full((len(self.terminal), width), -1, dtype=np.int64)
        nodes = np.zeros_like(tokens)
        valid = np.zeros_like(tokens, dtype=bool)
        for row in range(len(self.terminal)):
            a, b = self.row_ptr[row:row+2]
            tokens[row, :b-a] = self.token_ids[a:b]
            nodes[row, :b-a] = self.next_nodes[a:b]
            valid[row, :b-a] = True
        return tokens, nodes, valid

    def save(self, directory, identity):
        directory = Path(directory)
        manifest = {**self.metadata, 'identity': identity, 'paths': self.paths, 'index_version': 'csr_v1'}
        if directory.exists():
            loaded = self.load(directory)
            old = json.loads((directory/'manifest.json').read_text())
            if loaded.paths != self.paths or old['identity'] != identity:
                raise ValueError('Immutable index already exists with different identity')
            return old
        directory.mkdir(parents=True, exist_ok=False)
        np.savez(directory/'arrays.npz', row_ptr=self.row_ptr, token_ids=self.token_ids, next_nodes=self.next_nodes, terminal=self.terminal)
        manifest['arrays_sha256'] = file_hash(directory/'arrays.npz')
        (directory/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        return manifest

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        manifest = json.loads((directory/'manifest.json').read_text())
        if file_hash(directory/'arrays.npz') != manifest['arrays_sha256']:
            raise ValueError('Index fingerprint mismatch')
        obj = cls(manifest['paths'], manifest['eos'])
        with np.load(directory/'arrays.npz', allow_pickle=False) as arrays:
            for name in ['row_ptr', 'token_ids', 'next_nodes', 'terminal']:
                if not np.array_equal(getattr(obj, name), arrays[name]) or getattr(obj, name).dtype != arrays[name].dtype:
                    raise ValueError('Stored CSR differs from canonical paths')
        if obj.metadata != {k: manifest[k] for k in obj.metadata}:
            raise ValueError('Index metadata mismatch')
        return obj
