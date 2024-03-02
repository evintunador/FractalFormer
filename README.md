# FractalFormer
this is a project where I create self-similarity at (hopefully) all levels of a decoder-only transformer. The idea is to take all of the things learned in my [matryoshkaGPT](https://github.com/evintunador/matryoshkaGPT) replication project and instead of having a series of single russian nesting dolls inside of each other, each "inside" contains multiple similar russian nesting dolls. Think of it like how each triangle in [Surpinski's Triangle](https://en.wikipedia.org/wiki/Sierpiński_triangle) has three triangles within it. I think at some point this will allow me to do interesting things such as
- multiple small speculation models for speculative decoding in parallel to increase chances of a match
- a new weird version of MOE where all experts exist simultaneously rather than being gated
    - expert-based speculative decoding
- infinite fusion of transformer models of a given size into transformer models of a larger size. if this works it'd allow for highly personalized models
- take advantage of the fact that language has a fractal-structure[^1](https://arxiv.org/pdf/2402.01825.pdf)[^2](https://arxiv.org/pdf/2311.10217.pdf) to create an (infinitely?) extendable maximum context length. more on this later
- specialize a model for use with [conversational swarm intelligence](https://youtu.be/XBheCYnwdpM)
- i think if i properly mess with context lengths and combine tokens like in [Multi-Word Tokenization for Sequence Compression](https://arxiv.org/abs/2402.09949) along with combining residual states then i might be able to create a way to have very long context windows taking advantage of langauge's fractal structure
- what if instead of each model keeping its own residual state, the smallest models kept their own but every submodel actually contributed to the residual state of its super-models? Like during the residual connection we concatenated the states of the sub-models and added them to the states of the super-models? That'd solve this "how do i connect them" problem without introducing any additional weights. and then sub-models could still act as speculative decoders & whatnot for the super-models. and maybe whether or not their residual state gets added in is where we implement the gate for the MOE idea! oooh I like this. And I've been saying multimodality but i feel like this would apply equally well to that [psychology of consciousness paper](https://arxiv.org/abs/2308.08708)'s idea of separate modules for different modalities

Relevant inspiration papers:
- [Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147)
- [MatFormer](https://arxiv.org/pdf/2310.07707.pdf)
- [2D Matryoshka Sentence Embeddings](https://arxiv.org/pdf/2402.14776.pdf)
- [A Language and Its Dimensions: Intrinsic Dimensions of Language Fractal Structures](https://arxiv.org/pdf/2311.10217.pdf)
- [Fractal Patterns May Unravel the Intelligence in Next-Token Prediction](https://arxiv.org/pdf/2402.01825.pdf)