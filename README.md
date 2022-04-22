# <img src="docs/source/images/logo/libkge-header-2880.png" alt="LibKGE: A knowledge graph embedding library" width="80%">

LibKGE是一个基于PyTorch的库，用于高效的训练、评估和超参数优化 [knowledge graph embeddings](https://ieeexplore.ieee.org/document/8047276) (KGE). 
它是高度可配置的，易于使用和可扩展的。其他KGE框架有 [listed below](#other-kge-frameworks).

LibKGE的主要目标是促进对KGE模型和训练方法的可重复的研究（以及有意义的比较）。在我们的文章中 [ICLR 2020 paper](https://github.com/uma-pi1/kge-iclr20) 有更多介绍
(see [video](https://iclr.cc/virtual_2020/poster_BkxSmlBFvr.html)), 
训练策略和超参数的选择对模型的性能有很大的影响，往往比模型类别本身的影响更大。LibKGE旨在为训练、超参数优化和评估策略提供清晰的实现，
可用于任何模型。框架中实现的每一个潜在的旋钮或启发式方法都通过*有据可查的*配置文件明确地暴露出来。(e.g., see [here](kge/config-default.yaml) and
[here](kge/model/embedder/lookup_embedder.yaml)). 
LibKGE还提供了最常见的KGE模型，并且可以很容易地添加新的模型（欢迎贡献！）。

对于链接预测任务，基于规则的系统，如 [AnyBURL](http://web.informatik.uni-mannheim.de/AnyBURL/) 是KGE的一个有竞争力的选择。

## 快速开始

```sh
#在开发模式下检索并安装项目 
git clone https://github.com/uma-pi1/kge.git
cd kge
pip install -e .

# 下载和预处理数据集
cd data
sh download_all.sh
cd ..

# 在toy数据集上训练一个例子模型（当你有一个gpu时，你可以省略'--job.device cpu'。)
kge start examples/toy-complex-train.yaml --job.device cpu

# 训练完成结果保存到kge/local目录下：
tree experiments/20220421-105739-toy-complex-train/
experiments/20220421-105739-toy-complex-train/
├── checkpoint_00000.pt
├── checkpoint_00005.pt
├── checkpoint_00010.pt
├── checkpoint_00015.pt
├── checkpoint_00020.pt
├── checkpoint_best.pt
├── config
│   ├── 3b87a893.yaml
│   └── 8e6d6792.yaml
├── config.yaml
├── kge.log
└── trace.yaml

1 directory, 11 files
```

## 目录

1. [Features](#features)
2. [Results and pretrained models](#results-and-pretrained-models)
3. [Using LibKGE](#using-libkge)
4. [Currently supported KGE models](#currently-supported-kge-models)
5. [Extending LibKGE](#extending-libkge)
6. [FAQ](#faq)
7. [Known issues](#known-issues)
8. [Changelog](CHANGELOG.md)
9. [Other KGE frameworks](#other-kge-frameworks)
10. [How to cite](#how-to-cite)

## 特点

 - **训练**
   - 训练形式: negative sampling, 1vsAll, KvsAll
   - Losses: binary cross entropy (BCE), Kullback-Leibler divergence (KL),
     margin ranking (MR), squared error (SE)
   - 支持PyTorch的所有优化器和学习率调度器，可以针对不同的参数单独选择（例如，实体和关系嵌入的参数不同）。
   - Learning rate warmup
   - Early stopping
   - Checkpointing
   - Stop (e.g., via `Ctrl-C`) and resume at any time
   - 自动内存管理以支持大批次的工作  (see config key `train.subbatch_auto_tune`)
 - **超参数优化**
   - Grid search, manual search, quasi-random search (using
     [Ax](https://ax.dev/)), Bayesian optimization (using [Ax](https://ax.dev/))
   - Highly parallelizable (multiple CPUs/GPUs on single machine)
   - Stop and resume at any time
 - **评估**
   - Entity ranking metrics: Mean Reciprocal Rank (MRR), HITS@k with/without filtering
   - Drill-down by: relation type, relation frequency, head or tail
 - **扩展日志和tracing**
   - 关于训练、超参数调整和评估的详细进展信息以机器可读的格式记录下来
   - 将所有/选定的部分追踪数据快速导出到CSV或YAML文件中，以方便分析。
 - **KGE models**
   - 所有模型都可以在有或没有交互关系的情况下使用 
   - [RESCAL](http://www.icml-2011.org/papers/438_icmlpaper.pdf) ([code](kge/model/rescal.py), [config](kge/model/rescal.yaml))
   - [TransE](https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data) ([code](kge/model/transe.py), [config](kge/model/transe.yaml))
   - [TransH](https://ojs.aaai.org/index.php/AAAI/article/view/8870) ([code](kge/model/transh.py), [config](kge/model/transh.yaml))
   - [DistMult](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/ICLR2015_updated.pdf) ([code](kge/model/distmult.py), [config](kge/model/distmult.yaml))
   - [ComplEx](http://proceedings.mlr.press/v48/trouillon16.pdf) ([code](kge/model/complex.py), [config](kge/model/complex.yaml))
   - [ConvE](https://arxiv.org/abs/1707.01476)  ([code](kge/model/conve.py), [config](kge/model/conve.yaml))
   - [RelationalTucker3](https://arxiv.org/abs/1902.00898)/[TuckER](https://arxiv.org/abs/1901.09590) ([code](kge/model/relational_tucker3.py), [config](kge/model/relational_tucker3.yaml))
   - [CP](https://arxiv.org/abs/1806.07297) ([code](kge/model/cp.py), [config](kge/model/cp.yaml))
   - [SimplE](https://arxiv.org/abs/1802.04868) ([code](kge/model/simple.py), [config](kge/model/simple.yaml))
   - [RotatE](https://arxiv.org/abs/1902.10197) ([code](kge/model/rotate.py), [config](kge/model/rotate.yaml))
   - [Transformer ("No context" model)](https://arxiv.org/abs/2008.12813) ([code](kge/model/transformer.py), [config](kge/model/transformer.yaml))
 - **Embedders**
   - Lookup embedder ([code](kge/model/embedder/lookup_embedder.py), [config](kge/model/embedder/lookup_embedder.yaml))
   - Projection embedder ([code](kge/model/embedder/projection_embedder.py), [config](kge/model/embedder/projection_embedder.yaml))


## 结果和预训练的模型

我们在下面列出一些用LibKGE获得的例子结果（测试数据上的过滤MRR和HITS@k）。这些结果是通过运行自动超参数搜索获得的，如[此处]所述(https://github.com/uma-pi1/kge-iclr20)。

这些结果不一定是使用LibKGE所能达到的最佳结果，但它们具有可比性，因为每个模型的超参数优化都采用了共同的实验设置（和同等的工作量）。
由于我们使用**过滤的MRR进行模型选择**，我们的结果可能不能说明其他验证指标（如HITS@10，该指标已被用于其他地方的模型选择）可实现的模型性能。

我们报告了整个测试集的性能数字，**包括在训练期间没有看到的实体的三元组**。
在现有的KGE实现中，这种做法并不一致：一些框架从测试集中移除未见过的实体，这导致了性能的明显提高（例如，对于这种评估方法，我们的WN18RR MRR数字大约增加了+3pp）。

我们还为这些结果提供了预训练的模型。每个预训练的模型都以LibKGE checkpoint的形式给出，其中包含了模型以及其他信息（如正在使用的配置）。关于如何使用checkpoint，请参见下面的文档。

#### FB15K-237 (Freebase)

|                                                                                                       |   MRR | Hits@1 | Hits@3 | Hits@10 |                                                                                      Config file |                                                                              Pretrained model |
|-------------------------------------------------------------------------------------------------------|------:|-------:|-------:|--------:|-------------------------------------------------------------------------------------------------:|----------------------------------------------------------------------------------------------:|
| [RESCAL](http://www.icml-2011.org/papers/438_icmlpaper.pdf)                                           | 0.356 |  0.263 |  0.393 |   0.541 | [config.yaml](http://web.informatik.uni-mannheim.de/pi1/iclr2020-models/fb15k-237-rescal.yaml)    | [1vsAll-kl](http://web.informatik.uni-mannheim.de/pi1/iclr2020-models/fb15k-237-rescal.pt) |
| [TransE](https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data) | 0.313 |  0.221 |  0.347 |   0.497 | [config.yaml](http://web.informatik.uni-mannheim.de/pi1/iclr2020-models/fb15k-237-transe.yaml)    | [NegSamp-kl](http://web.informatik.uni-mannheim.de/pi1/iclr2020-models/fb15k-237-transe.pt) |
| [DistMult](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/ICLR2015_updated.pdf)  | 0.343 |  0.250 |  0.378 |   0.531 | [config.yaml](http://web.informatik.uni-mannheim.de/pi1/iclr2020-models/fb15k-237-distmult.yaml)  | [NegSamp-kl](http://web.informatik.uni-mannheim.de/pi1/iclr2020-models/fb15k-237-distmult.pt) |
| [ComplEx](http://proceedings.mlr.press/v48/trouillon16.pdf)                                           | 0.348 |  0.253 |  0.384 |   0.536 | [config.yaml](http://web.informatik.uni-mannheim.de/pi1/iclr2020-models/fb15k-237-complex.yaml)   | [NegSamp-kl](http://web.informatik.uni-mannheim.de/pi1/iclr2020-models/fb15k-237-complex.pt) |
| [ConvE](https://arxiv.org/abs/1707.01476)                                                             | 0.339 |  0.248 |  0.369 |   0.521 | [config.yaml](http://web.informatik.uni-mannheim.de/pi1/iclr2020-models/fb15k-237-conve.yaml)     | [1vsAll-kl](http://web.informatik.uni-mannheim.de/pi1/iclr2020-models/fb15k-237-conve.pt) |
| [RotatE](https://openreview.net/pdf?id=HkgEQnRqYQ)                                                    | 0.333 |  0.240 |  0.368 |   0.522 | [config.yaml](http://web.informatik.uni-mannheim.de/pi1/libkge-models/fb15k-237-rotate.yaml)      | [NegSamp-bce](http://web.informatik.uni-mannheim.de/pi1/libkge-models/fb15k-237-rotate.pt) |

#### WN18RR (Wordnet)

|                                                                                                       |   MRR | Hits@1 | Hits@3 | Hits@10 |                                                                                 Config file |                                                                        Pretrained model |
|-------------------------------------------------------------------------------------------------------|------:|-------:|-------:|--------:|--------------------------------------------------------------------------------------------:|----------------------------------------------------------------------------------------:|
| [RESCAL](http://www.icml-2011.org/papers/438_icmlpaper.pdf)                                           | 0.467 |  0.439 |  0.480 |   0.517 |   [config.yaml](http://web.informatik.uni-mannheim.de/pi1/iclr2020-models/wnrr-rescal.yaml) |   [KvsAll-kl](http://web.informatik.uni-mannheim.de/pi1/iclr2020-models/wnrr-rescal.pt) |
| [TransE](https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data) | 0.228 |  0.053 |  0.368 |   0.520 |   [config.yaml](http://web.informatik.uni-mannheim.de/pi1/iclr2020-models/wnrr-transe.yaml) |  [NegSamp-kl](http://web.informatik.uni-mannheim.de/pi1/iclr2020-models/wnrr-transe.pt) |
| [DistMult](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/ICLR2015_updated.pdf)  | 0.452 |  0.413 |  0.466 |   0.530 | [config.yaml](http://web.informatik.uni-mannheim.de/pi1/iclr2020-models/wnrr-distmult.yaml) | [KvsAll-kl](http://web.informatik.uni-mannheim.de/pi1/iclr2020-models/wnrr-distmult.pt) |
| [ComplEx](http://proceedings.mlr.press/v48/trouillon16.pdf)                                           | 0.475 |  0.438 |  0.490 |   0.547 |  [config.yaml](http://web.informatik.uni-mannheim.de/pi1/iclr2020-models/wnrr-complex.yaml) |  [1vsAll-kl](http://web.informatik.uni-mannheim.de/pi1/iclr2020-models/wnrr-complex.pt) |
| [ConvE](https://arxiv.org/abs/1707.01476)                                                             | 0.442 |  0.411 |  0.451 |   0.504 |    [config.yaml](http://web.informatik.uni-mannheim.de/pi1/iclr2020-models/wnrr-conve.yaml) |    [KvsAll-kl](http://web.informatik.uni-mannheim.de/pi1/iclr2020-models/wnrr-conve.pt) |
| [RotatE](https://openreview.net/pdf?id=HkgEQnRqYQ)                                                    | 0.478 |  0.439 |  0.494 |   0.553 |    [config.yaml](http://web.informatik.uni-mannheim.de/pi1/libkge-models/wnrr-rotate.yaml) |     [NegSamp-bce](http://web.informatik.uni-mannheim.de/pi1/libkge-models/wnrr-rotate.pt) |

#### FB15K (Freebase)

|                                                                                                       |   MRR | Hits@1 | Hits@3 | Hits@10 |                                                                                      Config file |                                                                              Pretrained model |
|-------------------------------------------------------------------------------------------------------|------:|-------:|-------:|--------:|-------------------------------------------------------------------------------------------------:|----------------------------------------------------------------------------------------------:|
| [RESCAL](http://www.icml-2011.org/papers/438_icmlpaper.pdf)                                           | 0.644 | 0.544  | 0.708  |   0.824 |    [config.yaml](http://web.informatik.uni-mannheim.de/pi1/libkge-models/fb15k-rescal.yaml) |     [NegSamp-kl](http://web.informatik.uni-mannheim.de/pi1/libkge-models/fb15k-rescal.pt) |
| [TransE](https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data) | 0.676 | 0.542  | 0.787  |   0.875 |   [config.yaml](http://web.informatik.uni-mannheim.de/pi1/libkge-models/fb15k-transe.yaml) |   [NegSamp-bce](http://web.informatik.uni-mannheim.de/pi1/libkge-models/fb15k-transe.pt) |
| [DistMult](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/ICLR2015_updated.pdf)  | 0.841 | 0.806  | 0.863  |   0.903 | [config.yaml](http://web.informatik.uni-mannheim.de/pi1/libkge-models/fb15k-distmult.yaml) | [1vsAll-kl](http://web.informatik.uni-mannheim.de/pi1/libkge-models/fb15k-distmult.pt) |
| [ComplEx](http://proceedings.mlr.press/v48/trouillon16.pdf)                                           | 0.838 | 0.807  | 0.856  |   0.893 |  [config.yaml](http://web.informatik.uni-mannheim.de/pi1/libkge-models/fb15k-complex.yaml) |  [1vsAll-kl](http://web.informatik.uni-mannheim.de/pi1/libkge-models/fb15k-complex.pt) |
| [ConvE](https://arxiv.org/abs/1707.01476)                                                             | 0.825 | 0.781  | 0.855  |   0.896 |    [config.yaml](http://web.informatik.uni-mannheim.de/pi1/libkge-models/fb15k-conve.yaml) |     [KvsAll-bce](http://web.informatik.uni-mannheim.de/pi1/libkge-models/fb15k-conve.pt) |
| [RotatE](https://openreview.net/pdf?id=HkgEQnRqYQ)                                                    | 0.783 | 0.727  | 0.820  |   0.877 |    [config.yaml](http://web.informatik.uni-mannheim.de/pi1/libkge-models/fb15k-rotate.yaml) |     [NegSamp-kl](http://web.informatik.uni-mannheim.de/pi1/libkge-models/fb15k-rotate.pt) |

#### WN18 (Wordnet)

|                                                                                                       |   MRR | Hits@1 | Hits@3 | Hits@10 |                                                                                 Config file |                                                                        Pretrained model |
|-------------------------------------------------------------------------------------------------------|------:|-------:|-------:|--------:|--------------------------------------------------------------------------------------------:|----------------------------------------------------------------------------------------:|
| [RESCAL](http://www.icml-2011.org/papers/438_icmlpaper.pdf)                                           | 0.948 | 0.943  | 0.951  |   0.956 |   [config.yaml](http://web.informatik.uni-mannheim.de/pi1/libkge-models/wn18-rescal.yaml) |   [1vsAll-kl](http://web.informatik.uni-mannheim.de/pi1/libkge-models/wn18-rescal.pt) |
| [TransE](https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data) | 0.553 | 0.315  | 0.764  |   0.924 |   [config.yaml](http://web.informatik.uni-mannheim.de/pi1/libkge-models/wn18-transe.yaml) |  [NegSamp-bce](http://web.informatik.uni-mannheim.de/pi1/libkge-models/wn18-transe.pt) |
| [DistMult](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/ICLR2015_updated.pdf)  | 0.941 | 0.932  | 0.948  |   0.954 | [config.yaml](http://web.informatik.uni-mannheim.de/pi1/libkge-models/wn18-distmult.yaml) | [1vsAll-kl](http://web.informatik.uni-mannheim.de/pi1/libkge-models/wn18-distmult.pt) |
| [ComplEx](http://proceedings.mlr.press/v48/trouillon16.pdf)                                           | 0.951 | 0.947  | 0.953  |   0.958 |  [config.yaml](http://web.informatik.uni-mannheim.de/pi1/libkge-models/wn18-complex.yaml) |  [KvsAll-kl](http://web.informatik.uni-mannheim.de/pi1/libkge-models/wn18-complex.pt) |
| [ConvE](https://arxiv.org/abs/1707.01476)                                                             | 0.947 | 0.943  | 0.949  |   0.953 |    [config.yaml](http://web.informatik.uni-mannheim.de/pi1/libkge-models/wn18-conve.yaml) |    [1vsAll-kl](http://web.informatik.uni-mannheim.de/pi1/libkge-models/wn18-conve.pt) |
| [RotatE](https://openreview.net/pdf?id=HkgEQnRqYQ)                                                    | 0.946 | 0.943  | 0.948  |   0.953 |    [config.yaml](http://web.informatik.uni-mannheim.de/pi1/libkge-models/wn18-rotate.yaml) |     [NegSamp-kl](http://web.informatik.uni-mannheim.de/pi1/libkge-models/wn18-rotate.pt) |

#### Wikidata5M (Wikidata)

LibKGE支持大型数据集，如Wikidata5M（4.8M实体）。下面给出的结果是通过自动超参数搜索找到的，类似于上面用于较小数据集的超参数搜索，但有些值是固定的
（用共享负采样训练，嵌入维度：128，批次大小：1024，优化器：Adagrad，正规化：加权）。
我们在20个epoch中运行了30个伪随机配置，然后在200个epoch中重新运行在验证数据上表现最好的配置。

|                                                             |   MRR | Hits@1 | Hits@3 | Hits@10 |                                                                                    Config file |                                                                            Pretrained model |
|-------------------------------------------------------------|------:|-------:|-------:|--------:|-----------------------------------------------------------------------------------------------:|--------------------------------------------------------------------------------------------:|
| [ComplEx](http://proceedings.mlr.press/v48/trouillon16.pdf) | 0.301 |  0.245 |  0.331 |   0.397 | [config.yaml](http://web.informatik.uni-mannheim.de/pi1/libkge-models/wikidata5m-complex.yaml) | [NegSamp-kl](http://web.informatik.uni-mannheim.de/pi1/libkge-models/wikidata5m-complex.pt) |

#### Yago3-10 (YAGO)
下面给出的结果是由用于Wikidata5M的相同的自动超参数搜索发现的。我们重演了在验证数据上表现最好的配置，并进行了400次epoch。

|                                                             |   MRR | Hits@1 | Hits@3 | Hits@10 |                                                                                    Config file |                                                                            Pretrained model |
|-------------------------------------------------------------|------:|-------:|-------:|--------:|-----------------------------------------------------------------------------------------------:|--------------------------------------------------------------------------------------------:|
| [ComplEx](http://proceedings.mlr.press/v48/trouillon16.pdf) | 0.551 |  0.476 |  0.596 |   0.682 | [config.yaml](http://web.informatik.uni-mannheim.de/pi1/libkge-models/yago3-10-complex.yaml) | [NegSamp-kl](http://web.informatik.uni-mannheim.de/pi1/libkge-models/yago3-10-complex.pt) |

#### CoDEx

[CoDEx](https://github.com/tsafavi/codex) 是一个基于Wikidata的KG完成基准。这里的结果是使用用于Freebase和WordNet数据集的自动超参数搜索获得的，
但CoDEx-M和CoDEx-L的epoch和Ax试验较少。 See the [CoDEx paper](https://arxiv.org/pdf/2009.07810.pdf) (EMNLP 2020) for details.

##### CoDEx-S

|  | MRR | Hits@1 | Hits@3 | Hits@10 | Config file | Pretrained model |
|---------|----:|----:|-------:|--------:|------------:|-----------------:|
| RESCAL | 0.404 | 0.293 | 0.4494 | 0.623 | [config.yaml](https://github.com/tsafavi/codex/tree/master/models/link-prediction/codex-s/rescal/config.yaml) | [1vsAll-kl](https://www.dropbox.com/s/v209jchl93mmeuv/codex-s-lp-rescal.pt?dl=0) |
| TransE | 0.354 | 0.219 | 0.4218 | 0.634 | [config.yaml](https://github.com/tsafavi/codex/tree/master/models/link-prediction/codex-s/transe/config.yaml) | [NegSamp-kl](https://www.dropbox.com/s/8brqhb4bd5gnktc/codex-s-lp-transe.pt?dl=0) |
| ComplEx | 0.465 | 0.372 | 0.5038 | 0.646 | [config.yaml](https://github.com/tsafavi/codex/tree/master/models/link-prediction/codex-s/complex/config.yaml) | [1vsAll-kl](https://www.dropbox.com/s/kk3pgdnyddsdzn9/codex-s-lp-complex.pt?dl=0) |
| ConvE | 0.444 | 0.343 | 0.4926  | 0.635 | [config.yaml](https://github.com/tsafavi/codex/tree/master/models/link-prediction/codex-s/conve/config.yaml) | [1vsAll-kl](https://www.dropbox.com/s/atvu77pzed6mcgh/codex-s-lp-conve.pt?dl=0) |
| TuckER | 0.444 | 0.339 | 0.4975 | 0.638 | [config.yaml](https://github.com/tsafavi/codex/tree/master/models/link-prediction/codex-s/tucker/config.yaml) | [KvsAll-kl](https://www.dropbox.com/s/f87xloe2g3f4fvy/codex-s-lp-tucker.pt?dl=0) 

##### CoDEx-M

|  | MRR | Hits@1 | Hits@3 |Hits@10 | Config file | Pretrained model |
|---------|----:|----:|-------:|--------:|------------:|-----------------:|
| RESCAL | 0.317 | 0.244 | 0.3477 | 0.456 | [config.yaml](https://github.com/tsafavi/codex/tree/master/models/link-prediction/codex-m/rescal/config.yaml) | [1vsAll-kl](https://www.dropbox.com/s/e3kp3eu4nnknn5b/codex-m-lp-rescal.pt?dl=0) |
| TransE | 0.303 | 0.223 | 0.3363 | 0.454 | [config.yaml](https://github.com/tsafavi/codex/tree/master/models/link-prediction/codex-m/transe/config.yaml) | [NegSamp-kl](https://www.dropbox.com/s/y8uucaajpofct3x/codex-m-lp-transe.pt?dl=0) |
| ComplEx | 0.337 | 0.262 | 0.3701 | 0.476 | [config.yaml](https://github.com/tsafavi/codex/tree/master/models/link-prediction/codex-m/complex/config.yaml) | [KvsAll-kl](https://www.dropbox.com/s/psy21fvbn5pbmw6/codex-m-lp-complex.pt?dl=0) |
| ConvE | 0.318 | 0.239 | 0.3551 | 0.464 | [config.yaml](https://github.com/tsafavi/codex/tree/master/models/link-prediction/codex-m/conve/config.yaml) | [NegSamp-kl](https://www.dropbox.com/s/awjhlrfjrgz9phi/codex-m-lp-conve.pt?dl=0) |
| TuckER | 0.328 | 0.259 | 0.3599 | 0.458 | [config.yaml](https://github.com/tsafavi/codex/tree/master/models/link-prediction/codex-m/tucker/config.yaml) | [KvsAll-kl](https://www.dropbox.com/s/so5l2owtx7wcos1/codex-m-lp-tucker.pt?dl=0) |


##### CoDEx-L

|  | MRR | Hits@1 | Hits@3 | Hits@10 | Config file | Pretrained model |
|---------|----:|----:|-------:|--------:|------------:|-----------------:|
| RESCAL | 0.304 | 0.242 | 0.3313 | 0.419 | [config.yaml](https://github.com/tsafavi/codex/tree/master/models/link-prediction/codex-l/rescal/config.yaml) | [1vsAll-kl](https://www.dropbox.com/s/wvbef9u98vmkbi8/codex-l-lp-rescal.pt?dl=0) |
| TransE | 0.187 | 0.116 | 0.2188 | 0.317 | [config.yaml](https://github.com/tsafavi/codex/tree/master/models/link-prediction/codex-l/transe/config.yaml) | [NegSamp-kl](https://www.dropbox.com/s/s9d682b49tuq5mc/codex-l-lp-transe.pt?dl=0) |
| ComplEx | 0.294 | 0.237 | 0.3179 | 0.400 | [config.yaml](https://github.com/tsafavi/codex/tree/master/models/link-prediction/codex-l/complex/config.yaml) | [1vsAll-kl](https://www.dropbox.com/s/jqubvr77og2pvzv/codex-l-lp-complex.pt?dl=0) |
| ConvE | 0.303 | 0.240 | 0.3298 | 0.420 | [config.yaml](https://github.com/tsafavi/codex/tree/master/models/link-prediction/codex-l/conve/config.yaml) | [1vsAll-kl](https://www.dropbox.com/s/qcfjy6i1sqbec0z/codex-l-lp-conve.pt?dl=0) |
| TuckER | 0.309 | 0.244 | 0.3395 | 0.430 | [config.yaml](https://github.com/tsafavi/codex/tree/master/models/link-prediction/codex-l/tucker/config.yaml) | [KvsAll-kl](https://www.dropbox.com/s/j8u4nqwzz3v7jw1/codex-l-lp-tucker.pt?dl=0) |

## 使用 LibKGE
LibKGE支持KGE模型的训练、评估和超参数调整。
每个任务的设置可以通过YAML格式的配置文件或命令行来指定。可用设置的默认值和用法可以在[config-default.yaml](kge/config-default.yaml)以及特定于模型和嵌入的配置文件（如[lookup_embedder.yaml](kge/model/embedder/lookup_embedder.yaml)).

#### 训练一个模型
首先创建一个配置文件，如：

```yaml
job.type: train
dataset.name: fb15k-237

train:
  optimizer: Adagrad
  optimizer_args:
    lr: 0.2

valid:
  every: 5
  metric: mean_reciprocal_rank_filtered

model: complex
lookup_embedder:
  dim: 100
  regularize_weight: 0.8e-7
```

要开始训练，请运行以下其中一项。

```sh
# 将该文件作为`config.yaml`存储在你选择的新文件夹中。然后用以下方法启动或恢复训练工作。
kge resume <folder>

# 或者，将配置存储在任何地方，并使用start命令创建一个新的文件夹
#   <kge-home>/local/experiments/<date>-<config-file-name>
# 并在那里开始训练。
kge start <config-file>

# 在这两种情况下，可以在命令行上修改配置选项, too: e.g.,
kge start <config-file> config.yaml --job.device cuda:0 --train.optimizer Adam
```

各种checkpoint（包括模型参数和配置选项）将在训练期间被创建。这些checkpoint可用于恢复训练（或任何其他工作类型，如超参数搜索工作）。

#### 恢复训练

所有LibKGE的job都可以被中断（例如，通过`Ctrl-C`）和恢复（从它的某个checkpoint）。要恢复一个job，请使用。

```sh
kge resume <folder>

# Change the device when resuming
kge resume <folder> --job.device cuda:1
```

默认情况下，使用最后一个checkpoint文件。checkpoint的文件名可以用`--checkpoint'来覆盖。

#### 评估一个训练过的模型

为了评估训练好的模型，运行以下内容。

```sh
# 在验证分割上评估一个模型
kge valid <folder>

# 在测试分割上评估一个模型
kge test <folder>
```
默认情况下，使用名为``checkpoint_best.pt``的checkpoint文件（存储迄今为止的最佳验证结果）。checkpoint的文件名可以用`--checkpoint'来覆盖。

#### 超参数优化

LibKGE支持各种形式的超参数优化，如网格搜索或贝叶斯优化。搜索类型和搜索空间在配置文件中指定。
例如，您可以使用[Ax](https://ax.dev/)进行SOBOL（伪随机）和贝叶斯优化。

下面的配置文件定义了10个SOBOL试验（arms）的搜索，然后是20个贝叶斯优化试验。

```yaml
job.type: search
search.type: ax

dataset.name: wnrr
model: complex
valid.metric: mean_reciprocal_rank_filtered

ax_search:
  num_trials: 30
  num_sobol_trials: 10  # remaining trials are Bayesian
  parameters:
    - name: train.batch_size
      type: choice
      values: [256, 512, 1024]
    - name: train.optimizer_args.lr
      type: range
      bounds: [0.0003, 1.0]
    - name: train.type
      type: fixed
      value: 1vsAll
```
试验可以在几个设备上并行运行。

```sh
# Run 4 trials in parallel evenly distributed across two GPUs
kge resume <folder> --search.device_pool cuda:0,cuda:1 --search.num_workers 4

# Run 3 trials in parallel, with per GPUs capacity
kge resume <folder> --search.device_pool cuda:0,cuda:1,cuda:1 --search.num_workers 3
```

#### 输出和分析日志和checkpoint

大量的日志被存储为YAML文件（超参数搜索、训练、验证）。LibKGE提供了一种方便的方法，可以将日志数据导出为CSV。

```sh
kge dump trace <folder>
```

上面的命令产生的CSV输出，如 [this output for a training job](docs/examples/dump-example-model.csv) or [this output for a search
job](https://github.com/uma-pi1/kge-iclr20/blob/master/data_dumps/iclr2020-fb15k-237-all-trials.csv).
可以根据需要在CSV文件中添加额外的配置选项或指标 (using a [keys
file](https://github.com/uma-pi1/kge-iclr20/blob/master/scripts/iclr2020_keys.conf)).

有关checkpoint的信息（如使用的配置、训练损失、验证指标或探索的超参数配置）也可以从命令行导出（作为YAML）。

```sh
kge dump checkpoint <checkpoint>
```

配置文件也可以以各种格式导出。
```sh
# 只导出与默认值不同的配置选项
kge dump config <config-or-folder-or-checkpoint>

# dump the configuration as is
kge dump config <config-or-folder-or-checkpoint> --raw

# dump the expanded config including all configuration keys
kge dump config <config-or-folder-or-checkpoint> --full

```

#### 帮助和其他命令

```sh
# help on all commands
kge --help

# help on a specific command
kge dump --help
```

#### 在一个应用程序中使用预训练的模型
使用用LibKGE训练的模型是很简单的。在下面的例子中，我们加载一个checkpoint，并预测两个主题关系对的最合适对象。
('Dominican Republic', 'has form of government', ?) and
('Mighty Morphin Power Rangers', 'is tv show with actor', ?).

```python
import torch
from kge.model import KgeModel
from kge.util.io import load_checkpoint

# 该checkpoint的下载链接在上述结果中给出。
checkpoint = load_checkpoint('fb15k-237-rescal.pt')
model = KgeModel.create_from(checkpoint)

s = torch.Tensor([0, 2,]).long()             # subject indexes
p = torch.Tensor([0, 1,]).long()             # relation indexes
scores = model.score_sp(s, p)                # scores of all objects for (s,p,?)
o = torch.argmax(scores, dim=-1)             # index of highest-scoring objects

print(o)
print(model.dataset.entity_strings(s))       # convert indexes to mentions
print(model.dataset.relation_strings(p))
print(model.dataset.entity_strings(o))

# Output (slightly revised for readability):
#
# tensor([8399, 8855])
# ['Dominican Republic'        'Mighty Morphin Power Rangers']
# ['has form of government'    'is tv show with actor']
# ['Republic'                  'Johnny Yong Bosch']
```

For other scoring functions (score_sp, score_po, score_so, score_spo), see [KgeModel](kge/model/kge_model.py#L455).

#### 使用你自己的数据集

要使用你自己的数据集，在`data`文件夹中创建一个子文件夹`mydataset`（=数据集名称）。你可以通过在job的配置文件中指定`dataset.name: mydataset`来使用你的数据集。
每个数据集都由一个`dataset.yaml`文件描述，需要存储在`mydataset`文件夹中。在执行完[快速入门说明](#quick-start)后，看看`data/toy/dataset.yaml'下提供的toy例子。配置键和文件格式都有记录 [here](https://github.com/uma-pi1/kge/blob/2b693e31c4c06c71336f1c553727419fe01d4aa6/kge/config-default.yaml#L48).

您的数据可以被自动预处理并转换为LibKGE所要求的格式。下面是`toy`数据集的相关部分，请看。

```sh
# download
curl -O http://web.informatik.uni-mannheim.de/pi1/kge-datasets/toy.tar.gz
tar xvf toy.tar.gz

# preprocess
python preprocess/preprocess_default.py toy
```


## 目前支持的KGE模型

LibKGE目前实现的KGE模型列在 [features](#features).

[examples](examples)文件夹包含一些配置文件，作为如何训练这些模型的例子。

我们欢迎为扩大支持的模型列表做出贡献! 请参阅[CONTRIBUTING](CONTRIBUTING.md)了解详情，并可自由地最初打开一个问题。

## 扩展 LibKGE

LibKGE可以通过新的训练、评估或搜索工作以及新的模型和嵌入器进行扩展。

KGE模型实现了 "KgeModel "类，通常由 "KgeEmbedder "和 "KgeScorer "组成，
前者将每个主语、关系和对象与一个嵌入相关联，后者则根据嵌入情况对三元祖进行评分。
所有这些基类都定义在 [kge_model.py](kge/model/kge_model.py). 

KGE工作执行训练、评估和超参数搜索。 The relevant base classes are [Job](kge/job/job.py), [TrainingJob](kge/job/train.py), [EvaluationJob](kge/job/eval.py), and [SearchJob](kge/job/search.py).

要添加一个组件，例如`mycomp`（=一个模型、嵌入器或job），并实现`MyClass`，你需要。

1. 创建一个配置文件`mycomp.yaml`。你可以把这个文件直接存放在LibKGE的模块文件夹中（例如，`<kge-home>/kge/model/`）或你自己的模块文件夹中。如果您打算将您的代码贡献给LibKGE，我们建议直接在LibKGE模块文件夹中开发。如果您只是想在LibKGE之外玩玩或发布您的代码，请使用您自己的模块。

2. 在`mycomp.yaml`中定义你的组件所需的所有选项，它们的默认值，以及它们的类型。我们建议遵循LibKGE的核心理念，以这种方式定义每一个能影响实验结果的选项。请注意整数(0)与浮点数(0.0)的关系；例如，`float_option: 0 "是不正确的，因为它被解释为一个整数。

3. 在你选择的模块中实现`MyClass'。在`mycomp.yaml`中，添加键`mycomp.class_name`和值`MyClass`。如果你遵循LibKGE的目录结构（`mycomp.yaml`用于配置，`mycomp.py`用于实现），那么确保`MyClass`在`__init__.py`中被导入。
(e.g., as done [here](kge/model/__init__.py)).

4. 要在一个实验中使用你的组件，通过`modules`键注册你的模块，通过实验配置文件中的`import`键注册其配置。参见 [config-default.yaml](kge/config-default.yaml) 以了解这些键的描述。例如，在`myexp_config.yaml`中，添加。

   ```yaml
   modules: [ kge.job, kge.model, kge.model.embedder, mymodule ]
   import: [ mycomp ]
   ```

## FAQ

#### 配置选项是否在某处有记录?
Yes, see [config-default.yaml](https://github.com/uma-pi1/kge/blob/master/kge/config-default.yaml) as well as the configuration files for each component listed [above](#features).

#### 命令行选项是否在某处有记录?
Yes, try `kge --help`. You may also obtain help for subcommands, e.g., try `kge dump --help` or `kge dump trace --help`.

#### LibKGE的内存用完了。我可以做什么呢？
- For training, set `train.subbatch_auto_tune` to true (equivalent result, less memory but slower).
- For evaluation, set `entity_ranking.chunk_size` to, say, 10000 (equivalent result, less memory but slightly slower, the more so the smaller the chunk size).
- Change hyperparameters (non-equivalent result): e.g., decrease the batch size, use negative sampling, use less samples).

## Known issues

## Changelog

See [here](CHANGELOG.md).


## Other KGE frameworks

Other KGE frameworks:
 - [Graphvite](https://graphvite.io/)
 - [AmpliGraph](https://github.com/Accenture/AmpliGraph)
 - [OpenKE](https://github.com/thunlp/OpenKE)
 - [PyKEEN](https://github.com/SmartDataAnalytics/PyKEEN)
 - [Pykg2vec](https://github.com/Sujit-O/pykg2vec)
 - [Dist-KGE](https://github.com/uma-pi1/dist-kge), a parallel variant of LibKGE

KGE projects for publications that also implement a few models:
 - [ConvE](https://github.com/TimDettmers/ConvE)
 - [KBC](https://github.com/facebookresearch/kbc)

PRs to this list are welcome.

## How to cite

Please cite the following publication to refer to the experimental study about the impact of training methods on KGE performance:

```
@inproceedings{
  ruffinelli2020you,
  title={You {CAN} Teach an Old Dog New Tricks! On Training Knowledge Graph Embeddings},
  author={Daniel Ruffinelli and Samuel Broscheit and Rainer Gemulla},
  booktitle={International Conference on Learning Representations},
  year={2020},
  url={https://openreview.net/forum?id=BkxSmlBFvr}
}
```

If you use LibKGE, please cite the following publication:

```
@inproceedings{
  libkge,
  title="{L}ib{KGE} - {A} Knowledge Graph Embedding Library for Reproducible Research",
  author={Samuel Broscheit and Daniel Ruffinelli and Adrian Kochsiek and Patrick Betz and Rainer Gemulla},
  booktitle={Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations},
  year={2020},
  url={https://www.aclweb.org/anthology/2020.emnlp-demos.22},
  pages = "165--174",
}
```
