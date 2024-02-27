# FractalFormer
this is a project where I create self-similarity at (hopefully) all levels of a decoder-only transformer. The idea is to take all of the things learned in my [matryoshkaGPT](https://github.com/evintunador/matryoshkaGPT) replication project and instead of having a series of single russian nesting dolls inside of each other, each "inside" contains multiple similar russian nesting dolls. Think of it like how each triangle in [Surpinski's Triangle](https://en.wikipedia.org/wiki/Sierpiński_triangle) has three triangles within it. I think at some point this will allow me to do interesting things such as
- a new weird version of MOE where all experts exist simultaneously rather than being gated
- infinite fusion of transformer models of a given size into transformer models of a larger size
- take advantage of the fact that language has a fractal-structure[^1](https://arxiv.org/pdf/2402.01825.pdf)[^2](uage and Its Dimensions - Intrinsic Dimensions of Language Fractal Structures) to create an (infinitely?) extendable maximum context length
- specialize a model for use with [conversational swarm intelligence](https://youtu.be/XBheCYnwdpM)

Relevant inspiration papers:
[Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147)
[MatFormer](https://arxiv.org/pdf/2310.07707.pdf)
[2D Matryoshka Sentence Embeddings](https://arxiv.org/pdf/2402.14776.pdf)