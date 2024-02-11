import os
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['DATASETS_VERBOSITY'] = 'error'

import argparse
from datasets import load_dataset
import math
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import time

models = {          
    "tinyllama_1b": "TinyLlama/TinyLlama-1.1B-intermediate-step-1195k-token-2.5T", # 4.10 GB
    "croissantllm_1b" : "croissantllm/CroissantLLMBase", # 5.01 GB
    "stablelm2_1b" : "stabilityai/stablelm-2-1_6b", # 3.06 GB
    "qwen1.5_0.5b" : "Qwen/Qwen1.5-0.5B" , # 1.16 GB
    "qwen1.5_1b" : "Qwen/Qwen1.5-1.8B" , # 3.43 GB
    "olmo_1b" : "allenai/OLMo-1B", # 4.39 GB
     
    "redpajama_3b" : "togethercomputer/RedPajama-INCITE-Base-3B-v1", # 5.30 GB
    "btlm_3b" : "cerebras/btlm-3b-8k-base", #  4.93 GB
    "openllama2_3b" : "openlm-research/open_llama_3b_v2", #  6.38 GB
    "stablelm_3b" : "stabilityai/stablelm-3b-4e1t", # 5.21 GB
    "phi2_3b" : "microsoft/phi-2", # 5.18 GB
    "qwen1.5_4b" : "Qwen/Qwen1.5-4B" , # 7.37 GB
    "minicpm_3b" : "openbmb/MiniCPM-2B-sft-bf16", # 5.08 GB

    "bloomz_7b" : "bigscience/bloomz-7b1-mt", # 13.18 GB
    "falcon_7b" : "tiiuae/falcon-7b", # 13.45 GB       
    "redpajama_7b" : "togethercomputer/RedPajama-INCITE-7B-Base", # 12.90 GB
    "mpt_7b" : "mosaicml/mpt-7b", # 12.39 GB
    "mpt_7b_8k" : "mosaicml/mpt-7b-8k", # 12.39 GB
    "openllama2_7b" : "openlm-research/open_llama_7b_v2", # 12.55 GB
    "llama2_7b" : "meta-llama/Llama-2-7b-hf", # 12.55 GB
    "llama2_7b_32k" : "togethercomputer/LLaMA-2-7B-32K", # 12.55 GB
    "mistral_7b" : "mistralai/Mistral-7B-v0.1", # 13.49 GB
    "qwen_7b" : "Qwen/Qwen-7B", # 14.38 GB
    "yi_6b" : "01-ai/Yi-6B", # 11.29 GB
    "decilm_7b" : "Deci/DeciLM-7B", # 13.12 GB
    "qwen1.5_7b" : "Qwen/Qwen1.5-7B" , # 14.39 GB
    "olmo_7b" : "allenai/OLMo-7B", # 25.66 GB
    
    "openllama1_13b" : "openlm-research/open_llama_13b", # 24.24 GB
    "llama2_13b" : "meta-llama/Llama-2-13b-hf", # 24.25 GB
    "qwen_14b" : "Qwen/Qwen-14B", # 26.39 GB
    "solar_10b" : "upstage/SOLAR-10.7B-v1.0", # 19.99 GB
    "qwen1.5_14b" : "Qwen/Qwen1.5-14B" , # 26.40 GB
     
    "mpt_30b" : "mosaicml/mpt-30b", # 55.80 GB 
    "codellama_34b" : "codellama/CodeLlama-34b-hf", # 62.86 GB 
    "yi_34b" : "01-ai/Yi-34B", # 64.06 GB    
     
    "falcon_40b" : "tiiuae/falcon-40b", # 77.93 GB
    "alfred_40b": "lightonai/alfred-40b-1023", # 77.93 GB
    "mixtral_8x7B" : "mistralai/Mixtral-8x7B-v0.1" # 86.99 GB
}

datasets = {
    "fr" : "frenchtext/banque-fr-2311",
    "en" : "frenchtext/bank-en-2401",
    "de" : "frenchtext/bank-de-2401",
    "es" : "frenchtext/bank-es-2401"
}

def get_dataset_batches(dataset, batch_size=32):
    filtered_dataset = dataset.filter(lambda example: example["Words"]>15)
    sorted_dataset = filtered_dataset.sort("Words",reverse=True)
    
    dataset_length = len(sorted_dataset)
    for start_idx in range(0, dataset_length, batch_size):
        end_idx = min(start_idx + batch_size, dataset_length)
        yield sorted_dataset[start_idx:end_idx]
        
def get_encoding_offsets(encoding):
    start_token_idx = 0
    while encoding.special_tokens_mask[start_token_idx]==1: start_token_idx+=1
    start_index = encoding.offsets[start_token_idx][0]
    end_token_idx = len(encoding.offsets)-1
    while encoding.special_tokens_mask[end_token_idx]==1: end_token_idx-=1
    end_index = encoding.offsets[end_token_idx][1]
    return (start_index,end_index)     

def encode_dataset_batch(tokenizer, dataset_batch, stride=256):
    
    # SPECIAL CASE: tiktoken tokenizer does not implement truncation=True, return_overflowing_tokens=True, and encodings offsets
    # => we must implement it manually on top of Huggingface tokenizers
    if hasattr(tokenizer,"tokenizer") and tokenizer.tokenizer.__class__.__module__.startswith("tiktoken"):
        encodings = tokenizer(text = dataset_batch["Text"], add_special_tokens=True, 
                      padding="longest", 
                      # 2020: https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html#tensor-core-shape
                      # However now in 2023, this is less and less true, newer drivers and cuda versions are smarter about this and will be able to use tensorcores even without this aligned padding
                      pad_to_multiple_of=16, return_tensors="pt")
        
        input_tensor = encodings['input_ids']
        attention_mask = encodings['attention_mask']
       
        batch_size = input_tensor.size(0)
        encodings_length = input_tensor.size(1)
        texts_length = torch.tensor([len(text) for text in dataset_batch["Text"]])
        
        max_length = tokenizer.model_max_length 
        
        if encodings_length > max_length:
        
            unfolded_tensor, overflow_to_sample_mapping = truncate_tensor_with_overflow(input_tensor, padding_value=tokenizer.pad_token_id, max_length=max_length, stride=stride)
            unfolded_mask, _ = truncate_tensor_with_overflow(attention_mask, padding_value=0, max_length=max_length, stride=stride)

            encodings['input_ids'] = unfolded_tensor
            encodings['attention_mask'] = unfolded_mask
            encodings['overflow_to_sample_mapping'] = overflow_to_sample_mapping
            
            offset = max_length - stride
            overflow_lines = 1 + math.ceil((encodings_length - max_length)/offset)
            last_line_padding = overflow_lines*offset + stride - encodings_length
            
            tokens_per_sample = attention_mask.sum(1).tolist()
            start_indexes = []
            end_indexes = []
            for sample_tokens in tokens_per_sample:
                start_indexes.append(torch.clamp(torch.arange(0,overflow_lines*offset,offset), max=sample_tokens)/sample_tokens)
                end_indexes.append(torch.clamp(torch.arange(max_length,encodings_length+last_line_padding+1,offset), max=sample_tokens)/sample_tokens)                
            overflow_to_sample_offset = torch.stack((torch.concat(start_indexes),torch.concat(end_indexes)))

            texts_length_multiplier = torch.repeat_interleave(texts_length, overflow_lines).unsqueeze(0)
            otso = (overflow_to_sample_offset*texts_length_multiplier).int()
            encodings['overflow_to_sample_offset'] = [(otso[0,i].item(),otso[1,i].item()) for i in range(otso.size(1))]
            
        else:
            
            encodings['overflow_to_sample_mapping'] = torch.zeros(batch_size, dtype=torch.int32)
            encodings['overflow_to_sample_offset'] = [(0,texts_length[i].item()) for i in range(batch_size)]
    
    # GENERAL CASE: just rely on Huggingface tokenizers for truncation
    else:
        encodings = tokenizer(text = dataset_batch["Text"], add_special_tokens=True, 
                          padding="longest", truncation=True, return_overflowing_tokens=True, stride=stride,
                          # 2020: https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html#tensor-core-shape
                          # However now in 2023, this is less and less true, newer drivers and cuda versions are smarter about this and will be able to use tensorcores even without this aligned padding
                          pad_to_multiple_of=16, return_tensors="pt")

        encodings["overflow_to_sample_offset"] = list(map(get_encoding_offsets, encodings.encodings))
    
    encodings["overflow_to_sample_uri"] = list(map(lambda sample_id: dataset_batch["Uri"][sample_id.item()], encodings["overflow_to_sample_mapping"]))

    return encodings

def truncate_tensor_with_overflow(input_tensor, padding_value, max_length=2048, stride=256):
    batch_length = input_tensor.size(0)
    encoding_length = input_tensor.size(1)

    offset = max_length - stride
    overflow_lines = 1 + math.ceil((encoding_length - max_length)/offset)
    last_line_padding = overflow_lines*offset + stride - encoding_length

    padded_tensor = F.pad(input_tensor, (0,last_line_padding), "constant", padding_value)
    unfolded_tensor = padded_tensor.unfold(1, max_length, offset).reshape(-1, max_length)

    overflow_to_sample_mapping = torch.arange(batch_length).repeat_interleave(overflow_lines)
 
    return unfolded_tensor, overflow_to_sample_mapping 

def get_encodings_batches(tokenizer, dataset, batch_size=32, stride=256):
    for dataset_batch in get_dataset_batches(dataset, batch_size):
        encodings = encode_dataset_batch(tokenizer, dataset_batch, stride)
        
        encodings_length = encodings['input_ids'].size(0)
        for start_idx in range(0, encodings_length, batch_size):
            end_idx = min(start_idx + batch_size, encodings_length)
            yield {key: encodings[key][start_idx:end_idx] for key in encodings.data.keys()}

def load_model(model_id, model_name):
    initial_mem_allocated = torch.cuda.memory_allocated(0)
    
    if model_id=="stablelm2_1b" or model_id=="olmo_1b" or model_id=="olmo_7b":
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    elif model_id=="stablelm_3b":
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=myhftoken)
    elif model_id=="qwen_7b" or model_id=="qwen_14b":
        # https://github.com/QwenLM/Qwen/blob/main/tokenization_note.md#special-tokens
        tokenizer = AutoTokenizer.from_pretrained(model_name, cpad_token = '<|endoftext|>', trust_remote_code=True)
    elif model_id=="yi_34b":
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        if model_id=="qwen_7b" or model_id=="qwen_14b":
            tokenizer.pad_token = '<|endoftext|>'
        else:
            tokenizer.pad_token = tokenizer.eos_token

    if model_id=="tinyllama_1b":
        # torch_dtype="auto" loads the model in fp32, which is not compatible with flash attention
        model = AutoModelForCausalLM.from_pretrained(model_name, use_safetensors=True, device_map="auto", torch_dtype=torch.float16, attn_implementation="flash_attention_2")
    elif model_id=="croissantllm_1b":
        model = AutoModelForCausalLM.from_pretrained(model_name, use_safetensors=False, device_map="auto", torch_dtype=torch.float16, attn_implementation="flash_attention_2")
    elif model_id=="stablelm2_1b" or model_id=="minicpm_3b":
        model = AutoModelForCausalLM.from_pretrained(model_name, use_safetensors=True, device_map="auto", torch_dtype="auto", attn_implementation="flash_attention_2", trust_remote_code=True)
    elif model_id=="olmo_1b":
        # no flash attention support for olmo as of 02/04/2024
        model = AutoModelForCausalLM.from_pretrained(model_name, use_safetensors=True, device_map="auto", torch_dtype=torch.float16, attn_implementation="eager")
    elif model_id=="olmo_7b":
        # no flash attention support for olmo as of 02/04/2024
        model = AutoModelForCausalLM.from_pretrained(model_name, use_safetensors=False, device_map="auto", torch_dtype=torch.float16, attn_implementation="eager")
    elif model_id=="btlm_3b":
        # no flash attention support as of 01/07/2024, using device_map triggers a fatal error
        model = AutoModelForCausalLM.from_pretrained(model_name, use_safetensors=True, torch_dtype="auto", attn_implementation="eager", trust_remote_code=True).to('cuda')
        # max context length supported without flahs attention on a RTX 4090
        tokenizer.model_max_length = 4096
    elif model_id=="stablelm_3b":
        model = AutoModelForCausalLM.from_pretrained(model_name, use_safetensors=True, device_map="auto", torch_dtype="auto", attn_implementation="flash_attention_2", trust_remote_code=True, token=myhftoken)
    elif model_id=="phi2_3b":
        # no flash attention support for phi2 as of 01/07/2024
        # for qwen: latest version of flash_attn installed, but module dropout_layer_norm not found
        model = AutoModelForCausalLM.from_pretrained(model_name, use_safetensors=True, device_map="auto", torch_dtype="auto", attn_implementation="eager", trust_remote_code=True)
    elif model_id=="bloomz_7b" or model_id=="mpt_7b":
        # no flash attention support as of 01/08/2024
        model = AutoModelForCausalLM.from_pretrained(model_name, use_safetensors=True, device_map="auto", torch_dtype="auto", attn_implementation="eager")
    elif model_id=="decilm_7b":
        model = AutoModelForCausalLM.from_pretrained(model_name, use_safetensors=True, device_map="auto", torch_dtype="auto", attn_implementation="flash_attention_2", trust_remote_code=True)
    elif model_id=="openllama1_13b":
        # Chunking error during model conversion to safetensors
        model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True, device_map="auto", torch_dtype="auto", attn_implementation="flash_attention_2")
    elif model_id=="llama2_13b" or model_id=="solar_10b":
        model = AutoModelForCausalLM.from_pretrained(model_name, use_safetensors=True, load_in_8bit=True, device_map="auto", torch_dtype="auto", attn_implementation="flash_attention_2")
    elif model_id=="qwen_7b":
        model = AutoModelForCausalLM.from_pretrained(model_name, use_safetensors=True, device_map="auto", torch_dtype="auto", attn_implementation="eager", trust_remote_code=True)
    elif model_id=="qwen_14b":
        # no flash attention support for qwen_14b as of 02/01/2024
        model = AutoModelForCausalLM.from_pretrained(model_name, use_safetensors=True, load_in_8bit=True, device_map="auto", torch_dtype="auto", attn_implementation="eager", trust_remote_code=True)
    elif model_id=="qwen1.5_14b":
        model = AutoModelForCausalLM.from_pretrained(model_name, use_safetensors=True, load_in_8bit=True, device_map="auto", torch_dtype="auto", attn_implementation="flash_attention_2")
    elif model_id=="mpt_30b":
        # no flash attention support as of 01/18/2024
        model = AutoModelForCausalLM.from_pretrained(model_name, use_safetensors=False, load_in_4bit=True, device_map="auto", torch_dtype="auto")
    elif model_id=="codellama_34b":
        model = AutoModelForCausalLM.from_pretrained(model_name, use_safetensors=True, load_in_4bit=True, device_map="auto", torch_dtype="auto", attn_implementation="flash_attention_2")
    elif model_id=="yi_34b": 
        model = AutoModelForCausalLM.from_pretrained(model_name, use_safetensors=True, load_in_4bit=True, device_map="auto", torch_dtype="auto", attn_implementation="flash_attention_2")
    elif model_id=="falcon_40b":
        model = AutoModelForCausalLM.from_pretrained(model_name, use_safetensors=False, load_in_4bit=True, device_map=0, torch_dtype="auto", attn_implementation="flash_attention_2")
    elif model_id=="alfred_40b":
        model = AutoModelForCausalLM.from_pretrained(model_name, use_safetensors=False, trust_remote_code=True, load_in_4bit=True, device_map="auto", torch_dtype="auto", attn_implementation="flash_attention_2")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, use_safetensors=True, device_map="auto", torch_dtype="auto", attn_implementation="flash_attention_2")
    
    print()
    print(f"Model {model_name} properties:")
    
    load_mem_allocated = torch.cuda.memory_allocated(0)
    print(f"- memory allocated: {(load_mem_allocated-initial_mem_allocated)/1024/1024:.2f} MB")
    
    if model_id=="bloomz_7b":
        tokenizer.model_max_length = model.config.seq_length
    elif model_id=="olmo_1b" or model_id=="olmo_7b":
        tokenizer.model_max_length = model.config.max_sequence_length
    elif model_id=="mpt_7b" or model_id=="mpt_30b":
        pass
    else:
        # IMPORTANT fix: https://github.com/huggingface/transformers/issues/16186
        tokenizer.model_max_length = int(min(tokenizer.model_max_length, model.config.max_position_embeddings))

    if model_id=="stablelm2_1b" or model_id=="qwen_7b" or model_id=="qwen_14b" or model_id=="yi_34b":
        print(f"- model vocabulary: {tokenizer.vocab_size}")
    else:
        print(f"- model vocabulary: {len(tokenizer.vocab)}")

    # Memory limit of RTX 4090
    if tokenizer.model_max_length>8192:
        tokenizer.model_max_length = 8192

    if model_id=="decilm_7b" or model_id=="codellama_34b" or model_id=="qwen1.5_7b" or model_id=="qwen1.5_14b":
        tokenizer.model_max_length = 4096
    elif model_id=="mpt_30b":
        tokenizer.model_max_length = 2048

    print(f"- model sequence length: {int(tokenizer.model_max_length)}")

    print(f"- model torch dtype: {model.dtype}")
    
    return tokenizer, model

class PPLu():
    
    def __init__(self, dataset_iterator, tokenizer, device):
        if hasattr(tokenizer,"vocab"):
            self.vocab_size = len(tokenizer.vocab)
        else:
            self.vocab_size = tokenizer.vocab_size
        dataset_token_id_counts = torch.zeros(self.vocab_size+1, dtype=torch.int64)
        dataset_tokens_count = 0
        
        for idx,dataset_batch in enumerate(dataset_iterator):
            encodings = tokenizer(text = dataset_batch["Text"], add_special_tokens=True, padding="longest", return_tensors="pt")
            
            # Padding tokens should be ignored: count them as token_id=vocabulary_size
            token_ids = encodings.input_ids*encodings.attention_mask + self.vocab_size*(1-encodings.attention_mask)
            
            token_id_counts = torch.bincount(token_ids.view(-1), minlength=self.vocab_size+1)
            tokens_count = encodings.attention_mask.sum()

            dataset_token_id_counts += token_id_counts
            dataset_tokens_count += tokens_count
            if idx%100==9: print(f"... {dataset_tokens_count:,} tokens")
        
        # Then discard the tokens count for token_id=vocabulary_size
        self.token_id_probs =  (dataset_token_id_counts[:-1] / dataset_tokens_count).unsqueeze(1).to(device)
        self.perplexity_loss = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
        print(f"Done: {dataset_tokens_count:,} tokens")

    def __call__(self, input_ids, attention_mask, output_logits):
        # Next-token prediction: shift prediction scores and input ids by one
        logits = output_logits[:, :-1, :].permute(0, 2, 1).contiguous()
        labels = input_ids[:, 1:].contiguous()
        labels_to_ignore = attention_mask[:, 1:]

        # Number of tokens predicted, ignoring padding tokens
        predicted_tokens_count_r = labels_to_ignore.sum(dim=1)
        # ... make sure we don't divide by 0 below ...
        predicted_tokens_count = predicted_tokens_count_r.clamp(min=1)
        
        # Cross entropy loss (ignore_index=-100)
        labels_for_crossentropy = labels*labels_to_ignore -100*(1-labels_to_ignore)
        batch_perplexity_losses = (1/predicted_tokens_count)*self.perplexity_loss(logits, labels_for_crossentropy).sum(1)
        
        # Unigram probability loss
        labels_probs = F.embedding(labels, self.token_id_probs).squeeze()
        # prob = 1 for padding tokens => log prob = 0, ignored in the sum below
        labels_probs = labels_probs*labels_to_ignore + (1-labels_to_ignore) 
        batch_unigram_losses = -(1/predicted_tokens_count)*torch.log(labels_probs).sum(dim=1)
        
        # Unigram-nomralized perplexities
        perplexities = torch.exp(batch_perplexity_losses)
        unigram_normalized_perplexities = torch.exp(batch_perplexity_losses - batch_unigram_losses)
        
        return predicted_tokens_count_r, batch_perplexity_losses, batch_unigram_losses, perplexities, unigram_normalized_perplexities
    
class NormalizedPerplexityLogger:
    def __init__(self, dataset_name, split, model_name):
        self.filename = f"{dataset_name.replace('/','_')}_{split}_{model_name.replace('/','_')}_pplu.csv"
        self.file = open(self.filename, 'w')
        
    def log_batch(self, ppl, pplu, uri, span):
        self.file.write(f"{ppl},{pplu},{uri},{span}\n")

def display_perplexities(pred_tokens_count, ppl_losses, unigram_losses):        
    pt_pred_tokens_count = torch.Tensor(pred_tokens_count)
    total_pred_tokens_count = pt_pred_tokens_count.sum().item()
    
    pt_ppl_losses = torch.Tensor(ppl_losses)
    pt_unigram_losses = torch.Tensor(unigram_losses)    
    pt_pplu_losses = pt_ppl_losses - pt_unigram_losses

    ppl = math.exp((pt_ppl_losses*pt_pred_tokens_count).sum().item() / total_pred_tokens_count)
    pplu = math.exp((pt_pplu_losses*pt_pred_tokens_count).sum().item() / total_pred_tokens_count)

    print(f"- perplexity = {ppl:.3f}")
    print(f"- unigram-normalized perplexity = {pplu*1000:.3f} (x1000)")
        
with open("/workspace/hftoken", 'r') as file:
    myhftoken = file.read().strip()
        
def test_perplexity(model_id, model_name, tokenizer, model, dataset_name, dataset):
    print("Training unigram model")
    start_time = time.time()
    pplu_loss = PPLu(get_dataset_batches(dataset), tokenizer, model.device)
    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes = elapsed_time // 60
    seconds = elapsed_time % 60
    print(f"Elapsed time: {int(minutes)} minutes and {seconds:.2f} seconds")
    
    if model_id=="tinyllama_1b" or model_id=="croissantllm_1b" or model_id=="olmo_1b" or model_id=="redpajama_3b" or model_id=="openllama2_3b":
        batch_size = 16
    elif model_id=="redpajama_7b":
        batch_size = 8
    elif model_id=="stablelm_3b" or model_id=="phi2_3b" or model_id=="falcon_7b" or model_id=="olmo_7b" or model_id=="mpt_7b"or model_id=="openllama1_13b":
        batch_size = 6
    elif model_id=="stablelm2_1b" or model_id=="btlm_3b" or model_id=="minicpm_3b" or model_id=="llama2_7b":
        batch_size = 4
    elif model_id=="yi_6b" or model_id=="llama2_13b":
        batch_size = 3
    elif model_id=="bloomz_7b" or model_id=="llama2_7b_32k" or model_id=="mistral_7b"or model_id=="decilm_7b" or model_id=="solar_10b":
        batch_size = 2
    elif model_id=="mpt_30b" or model_id=="codellama_34b" or model_id=="yi_34b" or model_id.startswith("qwen"):
        batch_size = 1
    stride = 256
    
    print()
    print("Computing perplexity")
    print(f"- batch_size={batch_size}")
    print()
    logger = NormalizedPerplexityLogger(dataset_name, "valid", model_name)
    pred_tokens_count = [] 
    ppl_losses = []   
    unigram_losses = [] 
    start_time = time.time()
    for idx,encodings_batch in enumerate(get_encodings_batches(tokenizer, dataset, batch_size=batch_size, stride=stride)):       
        with torch.no_grad():
            # predict next token
            inputs = encodings_batch["input_ids"].to(model.device)
            attention_mask = encodings_batch["attention_mask"].to(model.device)
            outputs = model(input_ids=inputs, attention_mask=attention_mask, use_cache=False, output_attentions=False, output_hidden_states=False)

            batch_pred_tokens_count, batch_ppl_losses, batch_unigram_losses, batch_ppl, batch_pplu = pplu_loss(inputs, attention_mask, outputs.logits)

            pred_tokens_count.extend(batch_pred_tokens_count.tolist())
            ppl_losses.extend(batch_ppl_losses.tolist())
            unigram_losses.extend(batch_unigram_losses.tolist())

        for ppl,pplu,uri,span in zip(batch_ppl.tolist(), batch_pplu.tolist(), encodings_batch["overflow_to_sample_uri"], encodings_batch["overflow_to_sample_offset"]):
            logger.log_batch(ppl, pplu, uri, span)

        if idx%100 == 0:
            print(f"{(idx+1)*batch_size} encodings processed")
            display_perplexities(pred_tokens_count, ppl_losses, unigram_losses)

    print(f"FINAL RESULT: {(idx+1)*batch_size} encodings processed")
    display_perplexities(pred_tokens_count, ppl_losses, unigram_losses)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes = elapsed_time // 60
    seconds = elapsed_time % 60
    print(f"Elapsed time: {int(minutes)} minutes and {seconds:.2f} seconds")   
    print()
    print(f"- max memory allocated: {torch.cuda.max_memory_allocated(0)/1024/1024:.2f} MB")
    
def main(model_id, test_function, dataset_lang=None):
    if model_id in models:
        model_name = models[model_id]
        print()
        print(f"Loading model {model_name}")
        tokenizer, model = load_model(model_id, model_name)
    else:
        print(f"Model id {model_name} does not exist")
        sys.exit(1)
    
    if dataset_lang:
        if dataset_lang in datasets:
            dataset_name = datasets[dataset_lang]
            
            from datasets.utils import disable_progress_bar
            disable_progress_bar()
            
            print()
            print(f"Loading dataset {dataset_name}")
            dataset = load_dataset(dataset_name, split="valid", token=myhftoken)
            
            print()
            print(f"Dataset {dataset_name} properties:")
            print(f"- dataset examples: {len(dataset)}")
            print(f"- dataset words: {sum(dataset['Words'])}")
            print(f"- dataset chars: {sum(dataset['Chars'])}")
        else:
            print(f"Dataset key {dataset_lang} does not exist")
            sys.exit(1)
    else:
        dataset = None
    
    if test_function=="perplexity":
        if not dataset:
            print("Dataset is mandatory to compute perplexity")
            sys.exit(1)
        print()
        print(f"Computing perplexity of model {model_name} on dataset {dataset_name}")
        print()
        test_perplexity(model_id, model_name, tokenizer, model, dataset_name, dataset)
    else:
        print(f"Test function {test_function} does not exist")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for testing large language models.")
    parser.add_argument("model_id", type=str, help="The id of the model to test.")
    parser.add_argument("test_function", type=str, help="The test function to run.")
    parser.add_argument("dataset_lang", type=str, help="The language of the dataset.")

    args = parser.parse_args()
    main(args.model_id, args.test_function, args.dataset_lang)