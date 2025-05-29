# Text-Gen-JEPA

Text-Gen-JEPA is a JEPA inspired model specialized for text prediction (and generation w/ post-training).

In this repo I am crafting `text-gen-jepa-base`, a small base model that predict the next token of a sentence.

## Setting up

I recommand you to use [uv](https://docs.astral.sh/uv/), but you can also just use `pip`.

```bash
uv venv .venv
```

```bash
uv pip install -r requirements.txt
```

And finally, you can run the `setup.sh` script to create the checkpoints structure to store weights during training.

```bash
sh setup.sh
```

## NOTES:

* During the training for each epoch we are training on the WHOLE `train` dataset, it can be really big and long. It's why I recommand you to stop the training early when you check the validation loss increases for too long.

* This model ONLY generate from left to right (like a GPT model). In the future I will fork this repo to adapt it to bidirectionnal generation (like BERT).

## Process

1. An encoder $E_a$ transform the token sequence $x=[x_1, ..., x_n]$ into a latent representation $s_x$ of dimension $(n,d)$, where $n$ is the numben and $d$ the r of tokeembedding dimension.
2. Another instance of the encoder $E_b$ transform the token $y=x_{n+1}$ into a latent representation $s_y$.
3. A Predictor $P$ predict the latent representation $\hat{s}_y$ from $s_x$.
4. Finaly, we compare the distance between $s_y$ and $\hat{s}_y$.

## Training

* As a first step I am gonna train the model using [tiny_shakespeare dataset](https://huggingface.co/datasets/karpathy/tiny_shakespeare) (because it's light and easy to use).

## License

MIT