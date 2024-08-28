# Quantize Stable Diffusion examples

## Running locally with PyTorch

### Installing the dependencies

Before running the scripts, make sure to install the library's training dependencies:

**Important**

To make sure you can successfully run the latest versions of the example scripts, we highly recommend **installing from source** and keeping the install up to date as we update the example scripts frequently and install some example-specific requirements. To do this, execute the following steps in a new virtual environment:
```bash
git clone https://github.com/huggingface/quanto
cd quanto
pip install -e .
```

Then cd in the `examples/vision/StableDiffusion` folder and run
```bash
pip install -r requirements.txt
```

**Now, we can launch the image generation script:**

```bash
python quantize_StableDiffusion.py --batch_size=1 --torch_dtype="fp32"
```

To better track our training experiments, we're using the following flags in the command above:

* `batch_size` Batch size is the number of samples used in one iteration of training.

* `torch_dtype` {fp32,fp16,bf16}
* `unet_qtype` {fp8,int8,int4,none}

Our experiments were conducted on a single 24GB A10 GPU.
```bash
fp16-fp16

batch_size: 1, torch_dtype: fp16, unet_dtype: none  in 3.307 seconds.Memory: 3.192GB.
```

```bash
bf16-int8

batch_size: 1, torch_dtype: bf16, unet_dtype: int8  in 3.918 seconds.Memory: 2.644GB.
```

```bash
fp16-int8

batch_size: 1, torch_dtype: fp16, unet_dtype: int8  in 3.920 seconds.Memory: 2.634GB.
``` 

will both get high-quality images at fast speed generation