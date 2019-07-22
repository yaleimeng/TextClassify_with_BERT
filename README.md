# TextClassify_with_BERT
使用BERT模型做文本分类；面向工业用途

+ 自己研究了当前开源的使用BERT做文本分类的许多存储库，各有各的缺点。通病就是面向学术，不考虑实际应用。
+ 使用txt、tsV、csv等不同数据集也就算了。有些项目甚至似乎存在bug。所以还是要自己动手解决。
+ 先占个位置，随后更新，欢迎收藏加关注。

## 与同类代码库相比的亮点：
- 在自定义Processor类中添加了针对单条文本的处理方法。在常规的针对test数据集的predict方法之外添加了针对单条数据的类别预测方法。
- 编写了简单的web服务，直接搭建对外服务的API接口。
- estimator 从 tf.contrib.tpu.TPUEstimator换成 tf.estimator.Estimator，以便在 gpu 上更高效运行。于此同时 model_fn 里tf.contrib.tpu.TPUEstimatorSpec 也修改成 tf.estimator.EstimatorSpec形式，相关调用参数也做了调整。在转换成较普通的 estimator 后便可以使用常用的方式处理，如生成用于部署的 *.pb 文件等。

## 使用方法：
1. 单机运行
+ 训练+评估：运行train_eval.py
+ 测试：运行predict_GPU.py
2. 搭建分类预测服务
+ 运行server.py
