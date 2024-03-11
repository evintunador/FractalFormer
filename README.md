# FractalFormer
this is a project where I create self-similarity at (hopefully) all levels of a decoder-only transformer. The idea is to take all of the things learned in my [matryoshkaGPT](https://github.com/evintunador/matryoshkaGPT) replication project and instead of having a series of single russian nesting dolls inside of each other, each "inside" contains multiple similar russian nesting dolls. Think of it like how each triangle in [Surpinski's Triangle](https://en.wikipedia.org/wiki/Sierpiński_triangle) has three triangles within it. I think at some point this will allow me to do interesting things such as
- multiple small models for MOE speculative decoding in parallel to increase chances of a match
- a new weird version of MOE where all experts exist simultaneously rather than being gated
- infinite fusion of transformer models of a given size into transformer models of a larger size
- take advantage of the fact that language has a fractal-structure[^1](https://arxiv.org/pdf/2402.01825.pdf)[^2](https://arxiv.org/pdf/2311.10217.pdf) to create an (infinitely? effectively infinitely?) extendable maximum context length if i can figure out how to properly borrow ideas from my previous [next-concept prediction](https://youtu.be/my59-MHNNcU) project and/or from [Multi-Word Tokenization for Sequence Compression](https://arxiv.org/abs/2402.09949). more on this later
- specialize a model for use with [conversational swarm intelligence](https://youtu.be/XBheCYnwdpM)
- i think i can eventually meet the criteria for consciousness as defined in [psychology of consciousness paper](https://arxiv.org/abs/2308.08708)

# Repo Guide
- `FractalFormer_base.ipynb`: currently the only file that is both functional and readable. This is where i recommend you start if you're curious about the project because it's not only heavily commented but also has extensive print statements so you can see for yourself what's happening. I do however need to go update all the images and give more thorough walkthroughs in the pre-code markdown cells. If you'd like to help me out with a boring task then changing the code from using `verbose` as a global variable to using the actual relevant entries in `config.verbose` dictionary throughout this file and in `config.py`, `tokenizer.py`, and `FractalFormer_base.py` would be a huge help. Check out the following video I made on this file:

[![Error displaying thumbnail. Click here for video](https://img.youtube.com/vi/MJnIxpZhTk0/0.jpg)](https://www.youtube.com/watch?v=MJnIxpZhTk0)

- `FractalFormer_ModelMerging.ipynb`: This document is currently just a copy of the previous but has been renamed, meaning I've not yet started on it. If you'd like to contribute, then this is the place to do it. Basically I'd like to train multiple separate non-FractalFormer small models, freeze their weights, and then concatenate & merge them into a proper FractalFormer as defined in the previous document. If you'd like to contribute and want more details on the task at hand let me know.
- `FractalFormer_UpstreamResidual.ipynb`: This is the file I'm currently working on. Not sure I can fully convey why i'm doing what i'm doing here as I'm still working largely off of intuition. Basically, in the base version when you perform inference you have to choose between which of the models you want to run and they all are capable of running separately, but here in UpstreamResidaul what I want to do is for any given model you want to run inference on, all of its sub-models will also run in parallel and their residual states will be concatenated and added to the model of interest. This is essentially how i create a connection between all the models in my eventual goal into creating a kind of hive-mind.
- `FractalFormer_DownstreamResidual.ipynb`: like the previous document except instead of sending the residual streams from the small models up to the big ones, i split apart the residual streams of the big model and send it down to the small ones. I think this may be useful for my MOE idea down the line
- `FractalFormer_InbuiltTokenizer.ipynb`: the idea here is to use byte-level tokens and let the model essentially create its own tokens, thereby completely getting rid of the tokenization step in language modeling. I'm messing around with different potential ways to do this over in `weird_embeddings.ipynb` but we're a ways off from me having something concrete to explain.
-  `config.py`, `tokenizer.py`, and `FractalFormer_base.py` are all code directly copied from `FractalFormer_base.ipynb` so that the classes & functions can then be imported into the other files
- `input.txt` is just TinyShakespare
- `tokenizers/tokenizer.model` is a very simple tokenizer that takes the 65 unqiue characters in tinyshakespeare and turns them into 128 tokens. Originally made for the repo that I build all my models off of [here](https://github.com/evintunador/base_model)
- `models/` contains all of the models i've trained so far, which as of right now are only the base versions. I don't think i'll be going past 1 million parameters for the foreseeable future
- `images/` is where i put drawings that help demonstrate what's happening in the code. 


# Relevant inspiration papers:
- [Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147)
- [MatFormer](https://arxiv.org/pdf/2310.07707.pdf)
- [2D Matryoshka Sentence Embeddings](https://arxiv.org/pdf/2402.14776.pdf)
- [A Language and Its Dimensions: Intrinsic Dimensions of Language Fractal Structures](https://arxiv.org/pdf/2311.10217.pdf)
- [Fractal Patterns May Unravel the Intelligence in Next-Token Prediction](https://arxiv.org/pdf/2402.01825.pdf)