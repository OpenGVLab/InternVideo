import torch
import torch.nn.functional as F

from transformers import BertTokenizer

from models.vindlu_blip_T5 import VindLU_BLIP_T5
from utils.config_utils import setup_main

config = setup_main()

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = VindLU_BLIP_T5(config, tokenizer=tokenizer)

model.cuda()

# Data
image = torch.rand(1, 12, 3, 224, 224)
image = image.cuda()
max_input_len, max_output_len = 25, 5
raw_question = raw_text_input = ["What is in the picture?"]
raw_answer = raw_text_output = ["A dog"]

# Train mode
_, input_t5 = model.encode_vision(image)
input_t5 = model.t5_proj(input_t5)
atts_t5 = torch.ones(input_t5.size()[:-1], dtype=torch.long).to(input_t5.device)

input_tokens = model.tokenize(raw_text_input, input_t5.device, max_input_len)
output_tokens = model.tokenize(
    raw_text_output, input_t5.device, max_output_len
)

encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)

targets = output_tokens.input_ids.masked_fill(
    output_tokens.input_ids == model.t5_tokenizer.pad_token_id, -100
)

inputs_embeds = model.t5_model.encoder.embed_tokens(input_tokens.input_ids)
inputs_embeds = torch.cat([input_t5, inputs_embeds], dim=1)

outputs = model.t5_model(
    inputs_embeds=inputs_embeds,
    attention_mask=encoder_atts,
    decoder_attention_mask=output_tokens.attention_mask,
    return_dict=True,
    labels=targets,
)

print("Shape of logits: ", outputs.logits.shape)
print("Loss: ", outputs.loss)
print("Indexes: ", outputs.logits.argmax(dim=-1))
print(F.cross_entropy(outputs.logits.view(-1, outputs.logits.size(-1)), targets.view(-1), ignore_index=-100))

# Inference mode
answer_tokens_start = torch.emtpy(
    (len(raw_question), 1), dtype=torch.long, device=input_t5.device
)
answer_tokens_start.fill_(model.t5_tokenizer.pad_token_id)

outputs = model.t5_model.generate(
    inputs_embeds=inputs_embeds,
    attention_mask=encoder_atts,
    decoder_input_ids=answer_tokens_start,
    return_dict=True,
)

print("Indexes: ", outputs.logits.argmax(dim=-1))