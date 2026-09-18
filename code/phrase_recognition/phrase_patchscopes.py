"""
Patchscopes logic for checking phrase recognition.
FIXED VERSION - checks if REST of phrase appears in any layer
"""
import torch
from dataclasses import dataclass
from typing import Optional
from transformers import PreTrainedModel, PreTrainedTokenizer


@dataclass
class PhraseRecognitionConfig:
    """Configuration for phrase recognition analysis."""
    patchscope_prompt: str = "Repeat this: X"
    replace_token: str = "X"
    num_patchscope_tokens: int = 5
    enable_rescaling: bool = False
    amplifying_factor: float = 1.0
    device: str = "cuda"


def check_phrase_recognition(model: PreTrainedModel,
                            tokenizer: PreTrainedTokenizer,
                            phrase: str,
                            config: PhraseRecognitionConfig) -> bool:
    """
    Check if model recognizes the full phrase before reaching the final token.
    
    NEW LOGIC: For each token position, check if the REST of the phrase 
    appears in ANY layer's patchscope output.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        phrase: Multi-word phrase to check (e.g., "Iguazu National Park")
        config: Configuration for patchscopes
        
    Returns:
        True if any position recognizes the rest of the phrase in any layer
    """
    # Re-encode to get BOS token
    input_ids = tokenizer.encode(phrase)
    if tokenizer.bos_token_id and input_ids[0] != tokenizer.bos_token_id:
        input_ids = [tokenizer.bos_token_id] + input_ids
    
    # Need at least 3 tokens: [BOS, token1, token2]
    if len(input_ids) < 3:
        return False
    
    # Decode the original phrase tokens (without BOS)
    phrase_tokens = input_ids[1:]  # Skip BOS
    
    # Hook for pre-norm final layer
    def pre_norm_capture_hook(module, input):
        module.captured_input = input[0].clone()
        return input
    
    hook = model.model.norm.register_forward_pre_hook(pre_norm_capture_hook)
    
    # Get hidden states
    with torch.no_grad():
        model_inputs = torch.tensor(input_ids).unsqueeze(0).to(model.device)
        outputs = model(input_ids=model_inputs, output_hidden_states=True)
        
        # Get pre-norm final layer
        last_hidden_state_wo_norm = model.model.norm.captured_input.clone()
        hook.remove()
        
        # Prepare patchscope prompt
        patchscope_input_ids = tokenizer.encode(config.patchscope_prompt)
        patchscope_tensor = torch.tensor(patchscope_input_ids).to(model.device)
        
        # Find replacement positions
        replace_token_ids = tokenizer.encode(config.replace_token, add_special_tokens=False)
        replace_mask = patchscope_tensor == replace_token_ids[0]
        
        # Also check space version
        space_replace_ids = tokenizer.encode(' ' + config.replace_token, add_special_tokens=False)
        if len(space_replace_ids) == 1:
            replace_mask = replace_mask | (patchscope_tensor == space_replace_ids[0])
        
        if not replace_mask.any():
            return False
        
        replace_positions = replace_mask.nonzero(as_tuple=False).flatten()
        base_inputs_embeds = model.get_input_embeddings()(patchscope_tensor).unsqueeze(0)
        
        # Check each token position (skip BOS at 0, skip last token)
        for token_idx in range(1, len(input_ids) - 1):
            
            # Get the REST of the phrase after this position
            # token_idx points to input_ids, so phrase position is token_idx - 1
            phrase_pos = token_idx - 1
            rest_of_phrase_tokens = phrase_tokens[phrase_pos + 1:]  # Everything after current token
            rest_of_phrase = tokenizer.decode(rest_of_phrase_tokens, skip_special_tokens=True)
            
            # Collect embeddings from all layers for this position
            layer_embeddings = []
            for layer_idx in range(1, len(outputs.hidden_states)):
                if layer_idx < len(outputs.hidden_states) - 1:
                    layer_embedding = outputs.hidden_states[layer_idx][:, token_idx]
                else:
                    layer_embedding = last_hidden_state_wo_norm[:, token_idx]
                layer_embeddings.append(layer_embedding.squeeze(0))
            
            layer_embeddings = torch.stack(layer_embeddings, dim=0)
            
            # Batched patchscope generation (all layers at once)
            batched_inputs_embeds = base_inputs_embeds.repeat(len(layer_embeddings), 1, 1)
            
            for batch_idx in range(len(layer_embeddings)):
                for pos in replace_positions:
                    batched_inputs_embeds[batch_idx, pos] = layer_embeddings[batch_idx]
            
            # Generate for all layers
            try:
                patchscope_outputs = model.generate(
                    inputs_embeds=batched_inputs_embeds,
                    max_new_tokens=config.num_patchscope_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                
                # Check each layer's output
                for layer_output in patchscope_outputs:
                    generated_text = tokenizer.decode(layer_output, skip_special_tokens=True)
                    
                    # Case-insensitive match for REST of phrase
                    if rest_of_phrase.lower() in generated_text.lower():
                        return True  # SUCCESS! Found the rest in this layer
                        
            except Exception as e:
                continue
    
    return False


def batch_check_phrase_recognition(model: PreTrainedModel,
                                  tokenizer: PreTrainedTokenizer,
                                  phrases: list,
                                  config: PhraseRecognitionConfig) -> list:
    """Check multiple phrases."""
    results = []
    for phrase in phrases:
        result = check_phrase_recognition(model, tokenizer, phrase, config)
        results.append(result)
    return results
