import torch
import transformers
from goldenverba.components.generation.hf_models.HuggingFace import (
    HuggingFaceOptions,
)

if torch.cuda.is_available():
    torch.set_default_device("cuda")
elif torch.backends.mps.is_available():
    torch.set_default_device("mps")
elif torch.backends.xla.is_available():
    torch.set_default_device("xla")
else:
    torch.set_default_device("cpu")

options = HuggingFaceOptions()

user_msg = input("Enter your message: ")

model = transformers.AutoModelForCausalLM.from_pretrained("", device_map="auto")

# https://github.com/huggingface/transformers/issues/27132
# please use the slow tokenizer since fast and slow tokenizer produces different tokens
tokenizer = transformers.AutoTokenizer.from_pretrained(
    "options.selected_option",
    use_fast=False,
)

system_message = "You are Orca, an AI language model created by Microsoft. You\
are a friendly and helpful assistant. You carefully follow instructions \
and try to provide the best answer based on facts.."
user_message = user_msg

prompt = f"\
    <|im_start|>system\n{system_message}<|im_end|>\
    \n<|im_start|>user\n{user_message}<|im_end|>\
    \n<|im_start|>assistant"

inputs = tokenizer(prompt, return_tensors="pt")
output_ids = model.generate(
    inputs["input_ids"],
)
answer = tokenizer.batch_decode(output_ids)[0]

print(answer)

# This example continues showing how to add a second turn message by the user to the conversation
second_turn_user_message = user_msg

# we set add_special_tokens=False because we dont want to automatically add a bos_token between messages
second_turn_message_in_markup = f"\
    \n<|im_start|>user\n{second_turn_user_message}<|im_end|>\
    \n<|im_start|>assistant"
second_turn_tokens = tokenizer(
    second_turn_message_in_markup, return_tensors="pt", add_special_tokens=False
)
second_turn_input = torch.cat([output_ids, second_turn_tokens["input_ids"]], dim=1)

output_ids_2 = model.generate(
    second_turn_input,
)
second_turn_answer = tokenizer.batch_decode(output_ids_2)[0]

print(second_turn_answer)
