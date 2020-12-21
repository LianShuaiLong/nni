EfficientNet
============

`EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks <https://arxiv.org/abs/1905.11946>`__

如论文中 3.3 所述，使用遍历搜索来找到 EfficientNet-B1 的 alpha, beta 和 gamma 的最好组合。 搜索空间，Tuner，配置示例如下。

说明
------------

:githublink:`示例代码 <examples/trials/efficientnet>`


#. 将示例代码目录设为当前工作目录。
#. 运行 ``git clone https://github.com/ultmaster/EfficientNet-PyTorch`` 来克隆 `ultmaster 修改过的版本 <https://github.com/ultmaster/EfficientNet-PyTorch>`__ 。 修改尽可能接近原始的 `Tensorflow 版本 <https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet>`__ （包括 EMA，标记平滑度等等。）；另外添加了代码从 Tuner 获取参数并回调中间和最终结果。 Clone it into ``EfficientNet-PyTorch``\ ; the files like ``main.py``\ , ``train_imagenet.sh`` will appear inside, as specified in the configuration files.
#. Run ``nnictl create --config config_local.yml`` (use ``config_pai.yml`` for OpenPAI) to find the best EfficientNet-B1. Adjust the training service (PAI/local/remote), batch size in the config files according to the environment.

For training on ImageNet, read ``EfficientNet-PyTorch/train_imagenet.sh``. Download ImageNet beforehand and extract it adhering to `PyTorch format <https://pytorch.org/docs/stable/torchvision/datasets.html#imagenet>`__ and then replace ``/mnt/data/imagenet`` in with the location of the ImageNet storage. This file should also be a good example to follow for mounting ImageNet into the container on OpenPAI.

Results
-------

The follow image is a screenshot, demonstrating the relationship between acc@1 and alpha, beta, gamma.


.. image:: ../../img/efficientnet_search_result.png
   :target: ../../img/efficientnet_search_result.png
   :alt: 

