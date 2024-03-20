import time

import torch
from datasets import load_dataset


@torch.no_grad()
def prediction_accuracy(model, tokenizer, batch_size, samples=None):
    test_dataset = load_dataset("lambada", split=["test"])[0]
    model.eval()
    # The task is to predict the last token of the input.
    total, hit = 0, 0
    start = time.time()
    for batch in test_dataset.iter(batch_size=batch_size):
        inputs = tokenizer(batch["text"], return_tensors="pt", padding=True)
        input_ids = inputs.input_ids.to(model.device)
        attention_mask = inputs.attention_mask.to(model.device)
        labels = input_ids[:, -1]
        # Pass only the first tokens
        outputs = model(input_ids[:, :-1], attention_mask=attention_mask[:, :-1])
        preds = outputs.logits[:, -1, :].argmax(dim=-1)
        total += labels.size(0)
        hit += (preds == labels).sum().item()
        if samples is not None and total >= samples:
            break
    end = time.time()
    acc = hit / total
    print(f"{total} sequences evaluated in {end - start:.2f} s. accuracy = {acc:.2f}")
    return acc
