# Hibiki: a speech-text foundation model for real time dialogue

[[Read the paper]][hibiki]
[[Samples]](https://huggingface.co/spaces/kyutai/hibiki-samples)
[[HuggingFace]](https://huggingface.co/kmhf/hibiki-2b-pytorch-bf16)

## Running the model

We provide inference codebase in PyTorch, MLX, and Rust.

### PyTorch

In order to translate an audio file using Hibiki/PyTorch, install the
`moshi` package via.
```bash
pip install moshi
```

Then run:
```bash
python -m moshi.run_inference in_fr.wav out_en.wav --hf-repo kyutai/hibiki-1b-pytorch-bf16
```


### MLX

In order to translate an audio file using Hibiki/MLX, install the
`moshi_mlx` package via.
```bash
pip install moshi_mlx
```

Then run:
```bash
python -m moshi_mlx.run_inference in_fr.wav out_en.wav --hf-repo kyutai/hibiki-1b-mlx-bf16
```

### Rust

Soon

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
