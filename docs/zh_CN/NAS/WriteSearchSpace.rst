Write A .. role:: raw-html(raw)
   :format: html

Search Space
====================

通常，搜索空间是要在其中找到最好结构的候选项。 无论是经典 NAS 还是 One-Shot NAS，不同的搜索算法都需要搜索空间。 NNI 提供了统一的 API 来表达神经网络架构的搜索空间。

搜索空间可基于基础模型来构造。 这也是在已有模型上使用 NAS 的常用方法。 Take `MNIST on PyTorch <https://github.com/pytorch/examples/blob/master/mnist/main.py>`__ as an example. Note that NNI provides the same APIs for expressing search space on PyTorch and TensorFlow.

.. code-block:: python

   from nni.nas.pytorch import mutables

   class Net(nn.Module):
       def __init__(self):
           super(Net, self).__init__()
           self.conv1 = mutables.LayerChoice([
               nn.Conv2d(1, 32, 3, 1),
               nn.Conv2d(1, 32, 5, 3)
           ])  # try 3x3 kernel and 5x5 kernel
           self.conv2 = nn.Conv2d(32, 64, 3, 1)
           self.dropout1 = nn.Dropout2d(0.25)
           self.dropout2 = nn.Dropout2d(0.5)
           self.fc1 = nn.Linear(9216, 128)
           self.fc2 = nn.Linear(128, 10)

       def forward(self, x):
           x = self.conv1(x)
           x = F.relu(x)
           # ... same as original ...
           return output

以上示例在 conv1 上添加了 conv5x5 的选项。 修改非常简单，只需要声明 ``LayerChoice`` 并将原始的 conv3x3 和新的 conv5x5 作为参数即可。 就这么简单！ 不需要修改 forward 函数。 可将 conv1 想象为没有 NAS 的模型。

如何表示可能的连接？ 通过 ``InputChoice`` 来实现。 要在 MNIST 示例上使用跳过连接，需要增加另一层 conv3。 下面的示例中，从 conv2 的可能连接加入到了 conv3 的输出中。

.. code-block:: python

   from nni.nas.pytorch import mutables

   class Net(nn.Module):
       def __init__(self):
           # ... same ...
           self.conv2 = nn.Conv2d(32, 64, 3, 1)
           self.conv3 = nn.Conv2d(64, 64, 1, 1)
           # declaring that there is exactly one candidate to choose from
           # search strategy will choose one or None
           self.skipcon = mutables.InputChoice(n_candidates=1)
           # ... same ...

       def forward(self, x):
           x = self.conv1(x)
           x = F.relu(x)
           x = self.conv2(x)
           x0 = self.skipcon([x])  # choose one or none from [x]
           x = self.conv3(x)
           if x0 is not None:  # skipconnection is open
               x += x0
           x = F.max_pool2d(x, 2)
           # ... same ...
           return output

Input Choice 可被视为可调用的模块，它接收张量数组，输出其中部分的连接、求和、平均（默认为求和），或没有选择时输出 ``None``。 Like layer choices, input choices should be **initialized in ``__init__`` and called in ``forward``**. 这会让搜索算法找到这些 Choice，并进行所需的准备。

``LayerChoice`` and ``InputChoice`` are both **mutables**. Mutable means "changeable". As opposed to traditional deep learning layers/modules which have fixed operation types once defined, models with mutable are essentially a series of possible models.

Users can specify a **key** for each mutable. By default, NNI will assign one for you that is globally unique, but in case users want to share choices (for example, there are two ``LayerChoice``\ s with the same candidate operations and you want them to have the same choice, i.e., if first one chooses the i-th op, the second one also chooses the i-th op), they can give them the same key. The key marks the identity for this choice and will be used in the dumped checkpoint. So if you want to increase the readability of your exported architecture, manually assigning keys to each mutable would be a good idea. For advanced usage on mutables (e.g., ``LayerChoice`` and ``InputChoice``\ ), see `Mutables <./NasReference.rst>`__.

With search space defined, the next step is searching for the best model from it. Please refer to `classic NAS algorithms <./ClassicNas.md>`__ and `one-shot NAS algorithms <./NasGuide.rst>`__ for how to search from your defined search space.
