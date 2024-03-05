import argparse
import sys

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from quanto import Calibration, freeze, qfloat8, qint4, qint8, qtype, quantize


@torch.no_grad()
def _calibrate(model, tokenizer, dataset, device, batch_size, num_batches):
    i = 1
    for batch in dataset.iter(batch_size=batch_size):
        inputs = tokenizer(batch["text"], return_tensors="pt", padding=True)
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)
        model(input_ids, attention_mask=attention_mask)
        i += 1
        if i > num_batches:
            break


@torch.no_grad()
def _perplexity(model, tokenizer, dataset, device, stride=512, seq_len=None):
    max_length = model.config.max_position_embeddings
    encodings = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")
    if seq_len is None:
        seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)

            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    return torch.exp(torch.stack(nlls).mean()).item()


class Perplexity:
    """
    A class for calculating the perplexity of a language model.
    """

    def __init__(self, model, tokenizer, dataset_path="wikitext", dataset_name=None, split="test", text_column="text"):
        """
        Calculate perplexity using the same method as seen in llama.cpp.

        Parameters
        ----------
        model : AutoModelForCausalLM
            The language model for which the perplexity is calculated.
        tokenizer : AutoTokenizer
            The tokenizer corresponding to the model.
        device : str, optional
            The device to run the calculations on. If auto, the device that your model uses
            will be the device used for these calculations. Default is 'auto'.
        dataset_path : str, optional
            The path to the dataset on the Hugging Face dataset hub. Default is 'wikitext'.
        dataset_name : str, optional
            The name of the dataset. Default is None.
        split : str, optional
            The split of the dataset to use. Default is 'test'.
        text_column : str, optional
            The name of the column in the dataset that contains the text data. Default is 'text'.
        """
        self._model = model
        self._tokenizer = tokenizer
        self._dataset_path = dataset_path
        self._dataset_name = dataset_name
        self._split = split
        self._text_column = text_column
        self._text = self._prepare_data()

    def _get_device(self):
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda:0"
        else:
            return "cpu"

    def _prepare_data(self):
        """
        Prepares the dataset by loading and formatting.

        Returns
        -------
        str
            The formatted dataset as a single string.
        """
        if self._dataset_path == "wikitext":
            self._dataset_name = "wikitext-2-raw-v1"

        # Load the dataset
        data = load_dataset(self._dataset_path, self._dataset_name, split=self._split)
        # Format the text column of the dataset
        text_list = [" \n" if s == "" else s for s in data[self._text_column]]
        return "".join(text_list)

    @staticmethod
    def softmax(logits):
        """
        Static method for applying the softmax function.

        Parameters
        ----------
        logits : np.ndarray
            The input to the softmax function.

        Returns
        -------
        np.ndarray
            The output of the softmax function.
        """
        e_x = np.exp(logits - np.max(logits))
        return e_x / e_x.sum(axis=0)

    def calculate_perplexity(self, n_ctx=512, n_batch=512):
        """
        Calculates the perplexity of the language model.

        Parameters
        ----------
        n_ctx : int
            The context size.
        n_batch : int
            The batch size.

        Returns
        -------
        list
            The list of perplexity scores calculated.
        """
        # Tokenize the text
        self._tokenizer.model_max_length = sys.maxsize
        tokens = self._tokenizer(self._text, truncation=False, return_tensors="pt").input_ids.to(self._model.device)

        nll = 0.0  # Negative log likelihood
        count = 0  # Counter for processed tokens
        curr_ppl = 0
        all_perplexity = []

        with tqdm(range(len(tokens[0]) // n_ctx), desc="Perplexity: - ") as progress:
            for i in progress:
                # Process each batch of tokens
                nll, count = self._process_batch(i, n_ctx, n_batch, tokens, nll, count)

                # Calculate and display the current perplexity
                curr_ppl = np.exp(nll / count)
                all_perplexity.append(curr_ppl)
                progress.set_description(f"Perplexity: {curr_ppl:.4f}")

        return all_perplexity

    def _process_batch(self, i, n_ctx, n_batch, tokens, nll, count):
        """
        Processes each batch of tokens.

        Parameters
        ----------
        i : int
            The batch index.
        n_ctx : int
            The context size.
        n_batch : int
            The batch size.
        tokens : torch.Tensor
            The tokenized text.
        nll : float
            The current negative log likelihood.
        count : int
            The current count of processed tokens.

        Returns
        -------
        float
            The updated negative log likelihood.
        int
            The updated count of processed tokens.
        """
        start = i * n_ctx
        end = start + n_ctx

        num_batches = (n_ctx + n_batch - 1) // n_batch

        logits = []

        for j in range(num_batches):
            batch_start = start + j * n_batch
            batch_size = min(end - batch_start, n_batch)

            token_org = tokens[0][batch_start].item()

            if j == 0:
                # Replace the first token with the BOS token
                tokens[0][batch_start] = self._tokenizer.bos_token_id

            # Compute the logits for the current batch of tokens
            batch_logits = self._compute_batch_logits(tokens, batch_start, batch_size)

            tokens[0][batch_start] = token_org

            logits.append(batch_logits)

        # We rely on the fact that attention in the forward pass only looks at previous
        # tokens here, so the logits returned for each token are an accurate representation
        # of what the model would have predicted at that point.
        #
        # Example, we have a context window of 512, we will compute perplexity for each of the
        # last 256 tokens.  Then, we split the input up into context window size chunks to
        # process the entire prompt.

        for j in range(min(512, n_ctx // 2), n_ctx - 1):
            tok_logits = logits[0][0][j].cpu().numpy()
            # Compute the probability of the next token
            prob = self.softmax(tok_logits)[tokens[0][start + j + 1]]

            # Update the negative log likelihood and the count of processed tokens
            nll += -np.log(prob, where=prob > 0)
            count += 1

        return nll, count

    def _compute_batch_logits(self, tokens, batch_start, batch_size):
        """
        Computes the logits for a batch of tokens.

        Parameters
        ----------
        tokens : torch.Tensor
            The tokenized text.
        batch_start : int
            The start index of the batch.
        batch_size : int
            The size of the batch.

        Returns
        -------
        torch.Tensor
            The logits for the batch of tokens.
        """
        # Compute the logits without keeping track of gradients
        with torch.no_grad():
            outputs = self._model(tokens[:, batch_start : batch_start + batch_size])
        return outputs.logits.detach()


def perplexity(
    model_id: str,
    weights: qtype,
    activations: qtype,
    device: torch.device,
    batch_size: int = 32,
    seed: int = 1,
    stride: int = 512,
):
    dtype = torch.float32 if device.type == "cpu" else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, low_cpu_mem_usage=True).to(device)
    test_set = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    if weights is not None:
        print("Quantizing")
        quantize(model, weights=weights, activations=activations)
        if activations is not None:
            print("Calibrating")
            with Calibration():
                _calibrate(model, tokenizer, test_set, device, batch_size=batch_size, num_batches=4)
        print("Freezing")
        freeze(model)

    print("Evaluating perplexity")
    ppl = Perplexity(model, tokenizer)
    ppl_value = np.mean(ppl.calculate_perplexity())
    return ppl_value


def keyword_to_qtype(k):
    return {"none": None, "int4": qint4, "int8": qint8, "float8": qfloat8}[k]


def main():
    parser = argparse.ArgumentParser(description="Generate bechmark")
    parser.add_argument(
        "--model", type=str, default="princeton-nlp/Sheared-LLaMA-1.3B", help="The model to use for benchmark"
    )
    parser.add_argument("--device", type=str, default=None, help="The device to use for benchmark.")
    parser.add_argument("--stride", type=int, default=512, help="The stride to use when evaluating perplexity.")
    parser.add_argument("--batch_size", type=int, default=32, help="The batch_size for evaluation (and calibration).")
    parser.add_argument(
        "--weights",
        type=str,
        default="none",
        choices=["none", "int4", "int8", "float8"],
    )
    parser.add_argument(
        "--activations",
        type=str,
        default="none",
        choices=["none", "int8", "float8"],
    )
    args = parser.parse_args()
    if args.device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    weights = keyword_to_qtype(args.weights)
    activations = keyword_to_qtype(args.activations)
    print(f"{args.model} (w: {args.weights}, a: {args.activations})")
    result = perplexity(args.model, weights, activations, device, batch_size=args.batch_size, stride=args.stride)
    print(result)


if __name__ == "__main__":
    main()
