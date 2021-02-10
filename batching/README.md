# Code optimization

The goal of this document is to describe a couple of code optimization
techniques (batching, vectorization, GPU support) which allow to speed up
PyTorch applications and the training process in particular.

The code from the previous session, relying on a pre-trained BERT embedding
model, is assumed as a starting point for the current session.

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->


<!-- END doctoc generated TOC please keep comment here to allow auto update -->


## Batching



[fasttext-models]: https://fasttext.cc/docs/en/crawl-vectors.html#models "Official fastText models for 157 languages"
[fasttext-en-100]: https://user.phil.hhu.de/~waszczuk/treegrasp/fasttext/cc.en.100.bin.gz
[fasttext-python-usage-overview]: https://fasttext.cc/docs/en/python-module.html#usage-overview
[fasttext-reduce-dim]: https://fasttext.cc/docs/en/crawl-vectors.html#adapt-the-dimension
[bert-as-service]: https://github.com/hanxiao/bert-as-service
[bert-small-models]: https://github.com/google-research/bert/#bert
