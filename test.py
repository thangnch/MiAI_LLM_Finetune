import torch
from transformers import MllamaForConditionalGeneration, AutoProcessor

model_id ="/root/thangnc_model/Llama-3.2-11B-Vision-Cheque"

model = MllamaForConditionalGeneration.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16)
processor = AutoProcessor.from_pretrained(model_id)

messages = [
    [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": "https://raw.githubusercontent.com/thangnch/MiAI_OrangePi5B/refs/heads/main/image.jpg"},
                {"type": "text", "text": "Extract information from bank cheque"}
            ]
        }
    ],
]
inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=1024)
print(processor.decode(output[0]))