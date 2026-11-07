"""Independent dictionary oracle; no model imports or I/O at import."""
import torch

def canonical_paths(paths, eos):
    result = set()
    for path in paths:
        path = tuple(path)
        if len(path) < 1 or any(type(t) is not int or t < 0 for t in path):
            raise ValueError('Paths require nonnegative integer tokens')
        if path[-1] != eos or eos in path[:-1]:
            raise ValueError('Complete paths must end in exactly one EOS')
        result.add(path)
    return sorted(result)

class ReferenceConstraints:
    def __init__(self, paths, eos=2):
        self.paths = canonical_paths(paths, eos)
        self.eos = eos
        self.successors = {(): set()}
        self.terminals = set(self.paths)
        for path in self.paths:
            for depth, token in enumerate(path):
                self.successors.setdefault(path[:depth], set()).add(token)
                self.successors.setdefault(path[:depth+1], set())

    def next(self, prefix):
        prefix = tuple(prefix)
        if prefix not in self.successors:
            return [], 'invalid'
        return sorted(self.successors[prefix]), 'terminal' if prefix in self.terminals else 'active'

    def allowed_next(self, prefix_ids, prefix_lengths):
        rows, statuses = [], []
        for row, length in zip(prefix_ids.tolist(), prefix_lengths.tolist()):
            if length < 0 or length > len(row):
                tokens, status = [], 'invalid'
            else:
                tokens, status = self.next(row[:length])
            rows.append(tokens)
            statuses.append({'active': 0, 'terminal': 1, 'invalid': 2}[status])
        return rows, torch.tensor(statuses, device=prefix_ids.device)

    def mask_scores(self, prefix_ids, lengths, scores):
        validate_scores(scores)
        rows, status = self.allowed_next(prefix_ids, lengths)
        result = torch.full_like(scores, -torch.inf)
        for i, tokens in enumerate(rows):
            if tokens:
                if max(tokens) >= scores.shape[1]:
                    raise ValueError('Token outside model vocabulary')
                result[i, tokens] = scores[i, tokens]
        return result, status

def validate_scores(scores):
    if scores.ndim != 2 or not scores.is_floating_point():
        raise ValueError('scores must be a floating Q x V tensor')
    if torch.isnan(scores).any() or torch.isposinf(scores).any():
        raise ValueError('NaN and +Inf scores are unsupported')

def generated_suffix(input_ids, prompt_width):
    if input_ids.ndim != 2 or type(prompt_width) is not int or not 0 <= prompt_width <= input_ids.shape[1]:
        raise ValueError('Invalid padded prompt width')
    return input_ids[:, prompt_width:]

def formal_gate(contract):
    required = ['tokenizer_fingerprint', 'checkpoint_sha256', 'prompt_protocol', 'score_protocol_id', 'termination_protocol']
    if contract.get('real_tokenizer_verified') is not True or any(not contract.get(k) or contract[k] == 'pending' for k in required):
        raise ValueError('Tokenizer/model contract validation is required')
