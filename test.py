
import torch
import json
from transformers import LlamaForCausalLM, LlamaTokenizer
# from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import safetensors
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
#%%
model_name_or_path = "linhvu/decapoda-research-llama-7b-hf" # from huggingface
load_in_8bit = False # TODO: change for Nvidia GPUs ?
device_map = "auto"
model = LlamaForCausalLM.from_pretrained(
            model_name_or_path,
            load_in_8bit=load_in_8bit,
            device_map=device_map,
            # cache_dir=os.path.join(args.cache_dir, "hub")
        )
tokenizer = LlamaTokenizer.from_pretrained(
            model_name_or_path, 
            # cache_dir=os.path.join(args.cache_dir, "hub")
            )
#%%
model_config = json.load(open("configs/TrainLM_llama-7b-hf.json"))
lora_hyperparams = json.load(open("configs/lora_config.json"))
target_modules = ["query_key_value"]
if model_config['model_type'] == "llama":
    target_modules = lora_hyperparams['lora_target_modules']  
config = LoraConfig(
    r=lora_hyperparams['lora_r'],
    lora_alpha=lora_hyperparams['lora_alpha'],
    target_modules=target_modules,
    lora_dropout=lora_hyperparams['lora_dropout'],
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)
use_lora = True
# lora_weight_path = "path_to_lora_weight(adapter_model.bin)"
lora_weight_path = "checkpoint-22000/pytorch_model.bin"
if use_lora:
    ckpt_name = lora_weight_path
    lora_weight = torch.load(ckpt_name)
    set_peft_model_state_dict(model, lora_weight)
model.eval()
#%%

#%%

input_sentence = "i guess if a film has magic i do n't need it to be fluid or seamless"

input_text = f"### Text:\n{input_sentence.strip()}\n\n### Question:\nIs the above text steganographic or non-steganographic?\n\n### Answer:\n"
input_text = tokenizer.bos_token + input_text if tokenizer.bos_token != None else input_text
inputs = tokenizer([input_text], return_tensors="pt").to("cuda")
predict = model.generate(**inputs, num_beams=1, do_sample=False, max_new_tokens=10, min_new_tokens=2)
output = tokenizer.decode(predict[0][inputs.input_ids.shape[1]:])
print(output)