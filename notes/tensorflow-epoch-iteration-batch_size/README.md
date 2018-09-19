### Epoch Iteration Batchsize in Tensorflow

- epoch： 1個epoch等於使用訓練集中的全部樣本訓練一次。
  - 所有的訓練樣本完成一次Forword運算以及一次BP運算

- iteration： 1個iteration等於使用batchsize個樣本訓練一次。
  - 每一次迭代都是一次權重更新，每一次權重更新需要batch size個數據進行Forward運算得到損失函式，再BP演算法更新引數。

- batchsize： 批次大小。在深度學習中，一般採用SGD訓練，即每次訓練在訓練集中取batchsize個樣本訓練
  - 一次Forword運算以及BP運算中所需要的訓練樣本數目，其實深度學習每一次引數的更新所需要損失函式並不是由一個{data：label}獲得的，而是由一組資料加權得到的，這一組資料的數量就是[batch size]。當然batch size 越大，所需的記憶體就越大，要量力而行

```
舉個例子，訓練集有1000個樣本，batchsize=10，那麼：
訓練完整個樣本集需要：100次iteration，1次epoch。
```

> stochastic gradient descent (SGD)
> 每次計算梯度只用一個樣本，這樣做的好處是計算快，而且很適合online-learning數據流式到達的場景，但缺點是單個sample產生的梯度估計往往很不准，所以得採用很小的learning rate，而且由於現代的計算框架CPU/GPU的多線程工作，單個sample往往很難佔滿CPU/GPU的使用率，導致計算資源浪費。

最後可以得到一個公式：

`one epoch = numbers of iterations = N = 訓練樣本的數量/batch size`

`一次epoch 總處理數量 = iterations次數 * batch_size大小`

# Reference

- [知乎：深度機器學習中的batch的大小對學習效果有何影響？](https://www.zhihu.com/question/32673260)
- [如何理解深度學習分佈式訓練中的large batch size與learning rate的關係？](https://www.leiphone.com/news/201710/RIIlL7LdIlT1Mvm8.html)
- [深度學習中 epoch，[batch size], iterations概念解釋](aaf080cd240a7365d11be2d8302322703ffe305e3f45ab7957ac6a56f3b90234)