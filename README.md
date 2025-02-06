# Hibiki: High-Fidelity Simultaneous Speech-To-Speech Translation

[[Read the paper]][hibiki]
[[Samples]](https://huggingface.co/spaces/kyutai/hibiki-samples)
[[HuggingFace]](https://huggingface.co/kyutai/hibiki-2b-pytorch-bf16)

## Running the model

We provide inference code for PyTorch, Rust, MLX for macOS, and MLX-swift
for iOS.

### PyTorch

In order to translate an audio file using Hibiki/PyTorch, install the
`moshi` package via.
```bash
pip install moshi
```

Then you can retrieve some sample files from [kyutai-labs/moshi](https://github.com/kyutai-labs/moshi/tree/main/data)
and translate them via the following:
```bash
wget https://github.com/kyutai-labs/moshi/raw/refs/heads/main/data/sample_fr_hibiki_crepes.mp3
python -m moshi.run_inference sample_fr_hibiki_crepes.mp3 out_en.wav --hf-repo kyutai/hibiki-1b-pytorch-bf16
```


You can specify some classifier-free guidance using the `--cfg-coef` parameter.
The default value is 1, the higher the value, the closer the generated voice
should be to the original voice. A typical value to use is 3. 

### MLX

In order to translate an audio file using Hibiki/MLX, install the
`moshi_mlx` package via the following command. You need at least version `0.2.1`
of this package.

```bash
pip install moshi_mlx
```

Then you can retrieve some sample files from [kyutai-labs/moshi](https://github.com/kyutai-labs/moshi/tree/main/data)
and translate them via the following:
```bash
wget https://github.com/kyutai-labs/moshi/raw/refs/heads/main/data/sample_fr_hibiki_crepes.mp3
python -m moshi_mlx.run_inference sample_fr_hibiki_crepes.mp3 out_en.wav --hf-repo kyutai/hibiki-1b-mlx-bf16
```

You can specify some classifier-free guidance using the `--cfg-coef` parameter.
The default value is 1, the higher the value, the closer the generated voice
should be to the original voice. A typical value to use is 3. 

### MLX-Swift

More instructions to come soon.

### Rust

More instructions to come soon.

## License

The present code is provided under the MIT license for the Python parts, and Apache license for the Rust backend.
The web client code is provided under the MIT license.

The weights for the models are released under the CC-BY 4.0 license.

## Citation

If you use Hibiki, please cite the following paper,

```
@misc{kyutai2025hibiki,
      title={High-Fidelity Simultaneous Speech-To-Speech Translation}, 
      author={Tom Labiausse and Laurent Mazar\'e and Edouard Grave and
      Patrick P\'erez and Alexandre D\'efossez and Neil Zeghidour},
      year={2025},
      eprint={2502.03382},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.03382}, 
}
```



[hibiki]: https://arxiv.org/abs/2502.03382
