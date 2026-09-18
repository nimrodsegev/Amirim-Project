"""
Lookahead analysis: For each i, how many phrases can predict the last i tokens?
"""
import torch
from dataclasses import dataclass
from transformers import PreTrainedModel, PreTrainedTokenizer


@dataclass
class LookaheadConfig:
    """Configuration for lookahead analysis."""
    patchscope_prompt: str = "Repeat this: X"
    replace_token: str = "X"
    num_patchscope_tokens: int = 5
    device: str = "cuda"


def analyze_phrase_lookahead(model: PreTrainedModel,
                             tokenizer: PreTrainedTokenizer,
                             phrase: str,
                             config: LookaheadConfig) -> dict:
    """
    Analyze at which positions the model can predict the remaining tokens.
    
    Returns:
        Dictionary mapping i (lookahead distance) to boolean (success/fail)
        Example: {1: True, 2: False} means it predicted last 1 token but not last 2
    """
    # Add BOS token
    input_ids = tokenizer.encode(phrase)
    if tokenizer.bos_token_id and input_ids[0] != tokenizer.bos_token_id:
        input_ids = [tokenizer.bos_token_id] + input_ids
    
    # Need at least 3 tokens: [BOS, token1, token2]
    if len(input_ids) < 3:
        return {}
    
    phrase_tokens = input_ids[1:]  # Skip BOS
    n_tokens = len(phrase_tokens)
    
    # Initialize results: {i: False for all possible i}
    lookahead_results = {i: False for i in range(1, n_tokens)}
    
    # Hook for pre-norm
    def pre_norm_capture_hook(module, input):
        module.captured_input = input[0].clone()
        return input
    
    hook = model.model.norm.register_forward_pre_hook(pre_norm_capture_hook)
    
    # Get hidden states
    with torch.no_grad():
        model_inputs = torch.tensor(input_ids).unsqueeze(0).to(model.device)
        outputs = model(input_ids=model_inputs, output_hidden_states=True)
        last_hidden_state_wo_norm = model.model.norm.captured_input.clone()
        hook.remove()
        
        # Prepare patchscope prompt
        patchscope_input_ids = tokenizer.encode(config.patchscope_prompt)
        patchscope_tensor = torch.tensor(patchscope_input_ids).to(model.device)
        
        # Find replacement positions
        replace_token_ids = tokenizer.encode(config.replace_token, add_special_tokens=False)
        replace_mask = patchscope_tensor == replace_token_ids[0]
        space_replace_ids = tokenizer.encode(' ' + config.replace_token, add_special_tokens=False)
        if len(space_replace_ids) == 1:
            replace_mask = replace_mask | (patchscope_tensor == space_replace_ids[0])
        
        if not replace_mask.any():
            return lookahead_results
        
        replace_positions = replace_mask.nonzero(as_tuple=False).flatten()
        base_inputs_embeds = model.get_input_embeddings()(patchscope_tensor).unsqueeze(0)
        
        # Check each position (skip BOS, skip last)
        for token_idx in range(1, len(input_ids) - 1):
            phrase_pos = token_idx - 1  # Position in phrase (without BOS)
            
            # How many tokens remain after this position?
            tokens_remaining = n_tokens - phrase_pos - 1
            
            # Get the last 'tokens_remaining' tokens
            last_i_tokens = phrase_tokens[-(tokens_remaining):]
            rest_of_phrase = tokenizer.decode(last_i_tokens, skip_special_tokens=True)
            
            # Collect embeddings from all layers
            layer_embeddings = []
            for layer_idx in range(1, len(outputs.hidden_states)):
                if layer_idx < len(outputs.hidden_states) - 1:
                    layer_embedding = outputs.hidden_states[layer_idx][:, token_idx]
                else:
                    layer_embedding = last_hidden_state_wo_norm[:, token_idx]
                layer_embeddings.append(layer_embedding.squeeze(0))
            
            layer_embeddings = torch.stack(layer_embeddings, dim=0)
            batched_inputs_embeds = base_inputs_embeds.repeat(len(layer_embeddings), 1, 1)
            
            for batch_idx in range(len(layer_embeddings)):
                for pos in replace_positions:
                    batched_inputs_embeds[batch_idx, pos] = layer_embeddings[batch_idx]
            
            # Generate
            try:
                patchscope_outputs = model.generate(
                    inputs_embeds=batched_inputs_embeds,
                    max_new_tokens=config.num_patchscope_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                
                # Check each layer
                for layer_output in patchscope_outputs:
                    generated_text = tokenizer.decode(layer_output, skip_special_tokens=True)
                    
                    # Check if rest of phrase appears
                    if rest_of_phrase.lower() in generated_text.lower():
                        # Mark i=tokens_remaining as success
                        lookahead_results[tokens_remaining] = True
                        break  # Found in this layer, move to next position
                        
            except Exception as e:
                continue
    
    return lookahead_results


def analyze_dataset_lookahead(model: PreTrainedModel,
                              tokenizer: PreTrainedTokenizer,
                              phrases: list,
                              config: LookaheadConfig) -> dict:
    """
    Analyze all phrases and aggregate results by lookahead distance i.
    
    Returns:
        Dictionary: {i: {'tested': count, 'success': count}}
    """
    # Initialize aggregated results
    aggregated = {}
    
    for phrase in phrases:
        results = analyze_phrase_lookahead(model, tokenizer, phrase, config)
        
        # Aggregate
        for i, success in results.items():
            if i not in aggregated:
                aggregated[i] = {'tested': 0, 'success': 0}
            
            aggregated[i]['tested'] += 1
            if success:
                aggregated[i]['success'] += 1
    
    return aggregated
