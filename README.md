# TextClassify_with_BERT
使用BERT模型做文本分类；面向工业用途

+ 自己研究了当前开源的使用BERT做文本分类的许多存储库，各有各的缺点。通病就是面向学术，不考虑实际应用。
+ 使用txt、tsv、csv等不同数据集也就算了，有些项目甚至似乎存在bug。所以还是要自己动手解决。
+ 已经基本可用，欢迎收藏加关注。有问题可提issue交流。

## 与同类代码库相比的亮点：
- 在自定义Processor类中添加了针对单条文本的处理方法。在常规的针对test数据集的predict方法之外添加了针对单条数据的类别预测方法。
- 编写了简单的web服务，直接搭建对外服务的API接口。
- estimator 从 tf.contrib.tpu.TPUEstimator换成 tf.estimator.Estimator，以便在 gpu 上更高效运行。于此同时 model_fn 里tf.contrib.tpu.TPUEstimatorSpec 也修改成 tf.estimator.EstimatorSpec形式，相关调用参数也做了调整。
- 在转换成较普通的 estimator 后便可以使用常用方式处理，如生成用于部署的 *.pb 文件等。

## 使用方法：
0. 准备工作
+ 首先要下载[BERT-Base中文模型](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip)或者从[Chinese-BERT-wwm](https://github.com/ymcui/Chinese-BERT-wwm)下载全词mask版本并解压到合适的目录；后面需要作为model_dir配置。
1. 单机运行
+ 在arguments.py中修改运行参数。主要是**数据目录、BERT目录、模型目录、序列长度、batch大小、学习率**等。
> 如果仅对test.txt执行预测，只需要把 do_predict 设为True，do_train 与do_eval 设置为false。
+ 训练+评估：运行train_eval.py </br>
> 如果上面步骤进展顺利，恭喜，在output/下面已经生成了训练和评估的结果。屏幕上也打印出了评估的准确率。</br>
+ 测试：先修改自己要测试的问题，运行predict_GPU.py</br>
2. 使用自己的真实数据集。修改xunlian.sh参数，重新进行训练和评估。
3. 搭建分类预测服务
+ 使用自己的pb模型+[开源框架](https://github.com/macanv/BERT-BiLSTM-CRF-NER)。 【强烈推荐】
+ 运行server.py 【仅供玩耍】
+ 有pb模型自己使用TensorFlow Serving部署
4. 关于用bert-base搭建服务的简介：
+ 在服务器端、客户端安装：pip install bert-base
+ 在PB模型所在目录下新建一个 run.sh文件，写入以下内容：
> bert-base-serving-start \ </br>
    -model_dir ./ \   # 训练输出目录【主要是其他Mode用到】。</br>
    -bert_model_dir F:\chinese_L-12_H-768_A-12   # BERT自身的目录。</br>
    -model_pb_dir ./ \      # pb模型路径，就填./ 当前目录就行了。。 </br>
    -mode CLASS       \     # 运行模式，分类就是CLASS</br>
    -max_seq_len 200  \     # 最大序列长度。要跟训练时参数一致。</br>
+ 运行编写好的run.sh，如果没报错，提示已经开始监听的话就表示服务成功开启。可以使用客户端运行示例了。</br>
非本机运行的话，构造BertClient时需要设置ip参数，例如BertClient(ip=192.168.20.20 ）。
