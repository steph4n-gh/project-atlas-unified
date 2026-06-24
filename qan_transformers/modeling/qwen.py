from qan_transformers.modeling.attention import DenseAttention

class QwenAttention(DenseAttention):
    """
    Standard dense self-attention layer for Qwen model configuration.
    """
    pass


def speculative_verify_superposition_qwen(target_logits, candidate_tokens, amplitudes):
    """
    Wave-packet collapse index selection algorithm for Qwen model configuration.
    """
    import torch
    B, C, T, V = target_logits.shape
    
    best_c = 0
    best_accepted_len = -1
    best_correction_token = None
    best_amplitude = -1.0
    
    for c in range(C):
        accepted_len = 0
        correction_token = None
        for t in range(T):
            target_pred = target_logits[0, c, t].argmax(dim=-1).item()
            candidate_tok = candidate_tokens[0, c, t].item()
            if candidate_tok == target_pred:
                accepted_len += 1
            else:
                correction_token = target_pred
                break
        
        # Selection rule: longest prefix length, tie-break by highest amplitude
        amplitude = amplitudes[c].item()
        if (accepted_len > best_accepted_len) or \
           (accepted_len == best_accepted_len and amplitude > best_amplitude):
            best_c = c
            best_accepted_len = accepted_len
            best_correction_token = correction_token
            best_amplitude = amplitude
            
    return best_c, best_accepted_len, best_correction_token
