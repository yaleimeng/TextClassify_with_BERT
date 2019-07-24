# TextClassify_with_BERT
使用BERT模型做文本分类；面向工业用途

+ 自己研究了当前开源的使用BERT做文本分类的许多存储库，各有各的缺点。通病就是面向学术，不考虑实际应用。
+ 使用txt、tsv、csv等不同数据集也就算了，有些项目甚至似乎存在bug。所以还是要自己动手解决。
+ 已经基本可用，欢迎收藏加关注。有问题可提issue交流。

## 与同类代码库相比的亮点：
- 在自定义Processor类中添加了针对单条文本的处理方法。在常规的针对test数据集的predict方法之外添加了针对单条数据的类别预测方法。
- 编写了简单的web服务，直接搭建对外服务的API接口。
- estimator 从 tf.contrib.tpu.TPUEstimator换成 tf.estimator.Estimator，以便在 gpu 上更高效运行。于此同时 model_fn 里tf.contrib.tpu.TPUEstimatorSpec 也修改成 tf.estimator.EstimatorSpec形式，相关调用参数也做了调整。
- 在转换成较普通的 estimator 后便可以使用常用方式处理，如生成用于部署的 *.pb 文件等。【待完成】

## 使用方法：
0.准备工作
+ 首先要下载BERT-Base[中文模型](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip)并解压到合适的目录；后面需要作为model_dir配置。
1. 单机运行
+ 训练+评估：运行xunlian.sh，实质上是运行train_eval.py并提供一堆参数 </br>
内容如下：
> export BERT_BASE_DIR=/home/itibia/chinese_L-12_H-768_A-12   # 这是BERT原始模型所在目录，根据实际修改

> python train_eval.py \  </br>
  --task_name cnews \   </br>
  --do_train  \         # 表示执行训练</br>
  --do_eval  \          # 表示执行评估</br>
  --do_predict false  \    # 表示不执行预测</br>
  --data_dir ./data/ \   # 数据集所在目录</br>
  --vocab_file $BERT_BASE_DIR/vocab.txt \   # Bert词汇表</br>
  --bert_config_file $BERT_BASE_DIR/bert_config.json \  # bert配置文件</br>
  --init_checkpoint $BERT_BASE_DIR/bert_model.ckpt \    # bert检查点</br>
  --max_seq_length 200 \        # 最大序列长度。对于新闻等长文本，可设置大一点。短文本可设短一些</br>
  --train_batch_size 32 \       # batch大小。结合自身GPU算力进行调整。如果训练时提示资源耗尽就调小一点。</br>
  --learning_rate 3e-5 \  </br>
  --num_train_epochs 5.0 \ </br>
  --output_dir ./output/ \     # ckpt模型输出目录</br>
  --local_rank 3  </br>

如果上面步骤进展顺利，恭喜，在output/下面已经生成了训练和评估的结果。屏幕上也打印出了评估的准确率。</br>
如果仅对test.txt执行预测，只需要把--do_predict 的false去掉，--do_train  与 --do_eval 都删除或者设置为false。

+ 测试：运行predict_GPU.py</br>
首先需要在BertClass类中根据实际情况修改init()中定义的几个成员变量。然后修改自己要测试的问题。</br>

2. 使用自己的数据集。修改xunlian.sh参数，重新进行训练和评估。

3. 搭建分类预测服务
+ 运行server.py
+ 或者自己使用TensorFlow Serving部署。  simple_tensorflow_serving --model_base_path="./pb_dir"
