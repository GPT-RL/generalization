import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "gpt2"
gpt2 = AutoModelForCausalLM.from_pretrained(model_name, return_dict_in_generate=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)

input_ids = tokenizer("Today is a nice day", return_tensors="pt").input_ids

generated_outputs = gpt2.generate(
    input_ids, do_sample=True, num_return_sequences=3, output_scores=True
)
for sequence in generated_outputs.sequences:
    print(tokenizer.decode(sequence, skip_special_tokens=True))


# only use id's that were generated
# gen_sequences has shape [3, 15]
gen_sequences = generated_outputs.sequences[:, input_ids.shape[-1] :]

# let's stack the logits generated at each step to a tensor and transform
# logits to probs
probs = torch.stack(generated_outputs.scores, dim=1).softmax(
    -1
)  # -> shape [3, 15, vocab_size]

# now we need to collect the probability of the generated token
# we need to add a dummy dim in the end to make gather work
gen_probs = torch.gather(probs, 2, gen_sequences[:, :, None]).squeeze(-1)

# now we can do all kinds of things with the probs

# 1) the probs that exactly those sequences are generated again
# those are normally going to be very small
unique_prob_per_sequence = gen_probs.prod(-1)
print("probs:", unique_prob_per_sequence)

# 2) normalize the probs over the three sequences
normed_gen_probs = gen_probs / gen_probs.sum(0)
if not normed_gen_probs[:, 0].sum() == 1.0:
    print("probs are not normalized:", normed_gen_probs[:, 0].sum())

# 3) compare normalized probs to each other like in 1)
unique_normed_prob_per_sequence = normed_gen_probs.prod(-1)
