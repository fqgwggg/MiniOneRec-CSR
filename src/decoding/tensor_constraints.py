"""Stateless batched traversal; depth loop only, no Python beam loop."""
import numpy as np
import torch
from .reference_constraints import validate_scores

class TensorConstraints:
    def __init__(self, index, device='cpu'):
        index.validate()
        self.ptr = torch.as_tensor(index.row_ptr.astype(np.int64), device=device)
        # Dedicated sentinel edge; invalid/padded positions never address the next row.
        self.tokens = torch.as_tensor(np.append(index.token_ids, -1).astype(np.int64), device=device)
        self.children = torch.as_tensor(np.append(index.next_nodes, 0).astype(np.int64), device=device)
        self.terminal = torch.as_tensor(index.terminal.copy(), device=device)
        self.edge_count = len(index.token_ids)
        self.width = max(1, index.metadata['max_degree'])
        self.offsets = torch.arange(self.width, device=device)

    def _edges(self, nodes, active):
        safe = nodes.clamp(min=0, max=len(self.terminal)-1)
        positions = self.ptr[safe, None] + self.offsets[None, :]
        valid = (positions < self.ptr[safe+1, None]) & active[:, None]
        safe_positions = torch.where(valid, positions, self.edge_count)
        return self.tokens[safe_positions], self.children[safe_positions], valid

    def allowed_next(self, prefix_ids, prefix_lengths):
        if prefix_ids.ndim != 2 or prefix_lengths.shape != (prefix_ids.shape[0],):
            raise ValueError('Expected prefix_ids[Q,L] and lengths[Q]')
        if prefix_ids.dtype != torch.long or prefix_lengths.dtype != torch.long:
            raise ValueError('Prefix IDs and lengths must be torch.long')
        if prefix_ids.device != self.ptr.device or prefix_lengths.device != self.ptr.device:
            raise ValueError('Prefix and index devices differ')
        nodes = torch.zeros(prefix_ids.shape[0], dtype=torch.long, device=prefix_ids.device)
        alive = (prefix_lengths >= 0) & (prefix_lengths <= prefix_ids.shape[1])
        for depth in range(prefix_ids.shape[1]):
            consume = depth < prefix_lengths
            tokens, children, valid = self._edges(nodes, alive & ~self.terminal[nodes])
            matches = valid & (tokens == prefix_ids[:, depth, None])
            found = matches.any(dim=1)
            selected = children.gather(1, matches.to(torch.long).argmax(dim=1, keepdim=True)).squeeze(1)
            alive = alive & (~consume | found)
            nodes = torch.where(consume & found, selected, nodes)
        status = torch.where(alive, torch.where(self.terminal[nodes], 1, 0), 2)
        tokens, _, valid = self._edges(nodes, alive & ~self.terminal[nodes])
        return tokens, valid, status

    def mask_scores(self, prefix_ids, lengths, scores):
        validate_scores(scores)
        if scores.shape[0] != prefix_ids.shape[0] or scores.device != self.ptr.device:
            raise ValueError('Score/prefix/device mismatch')
        tokens, valid, status = self.allowed_next(prefix_ids, lengths)
        vocab = scores.shape[1]
        if ((tokens >= vocab) & valid).any():
            raise ValueError('Token outside model vocabulary')
        columns = torch.where(valid, tokens, vocab)
        counts = torch.zeros((scores.shape[0], vocab+1), dtype=torch.int32, device=scores.device)
        counts.scatter_add_(1, columns, valid.to(torch.int32))
        mask = counts[:, :vocab] > 0
        return scores.masked_fill(~mask, -torch.inf), status
