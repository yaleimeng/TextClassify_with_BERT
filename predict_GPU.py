# -*- coding: utf-8 -*-
'''
@author: yaleimeng@sina.com
@license: (C) Copyright 2019
@desc: 项目描述。
@DateTime: Created on 2019/7/19, at 下午 04:13 by PyCharm
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, csv, random, collections
import modeling, optimization, tokenization
import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS

## 【必备参数已经在Bert_Class类中给出。】[Other parameters
flags.DEFINE_string("init_checkpoint", None, "Initial checkpoint (usually from a pre-trained BERT model).")
flags.DEFINE_bool("do_lower_case", True, "Whether to lower case the input text. Should be True for uncased "
                                         "models and False for cased models.")
flags.DEFINE_integer("max_seq_length", 200, "The maximum total input sequence length after WordPiece tokenization. "
                                            "Sequences longer than this will be truncated, and sequences shorter "
                                            "than this will be padded.")
flags.DEFINE_bool("do_train", False, "Whether to run training.")
flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")
flags.DEFINE_bool("do_predict", False, "Whether to run the model in inference mode on the test set.")
flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")
flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")
flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")
flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")
flags.DEFINE_float("num_train_epochs", 5.0, "Total number of training epochs to perform.")
flags.DEFINE_float("warmup_proportion", 0.1, "Proportion of training to perform linear learning rate warmup for. "
                                             "E.g., 0.1 = 10% of training.")
flags.DEFINE_integer("save_checkpoints_steps", 1000, "How often to save the model checkpoint.")
flags.DEFINE_integer("iterations_per_loop", 1000, "How many steps to make in each estimator call.")

# flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")
tf.flags.DEFINE_string("tpu_name", None, "The Cloud TPU to use for training. This should be either the name "
                                         "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
                                         "url.")
tf.flags.DEFINE_string("tpu_zone", None, "[Optional] GCE zone where the Cloud TPU is located in. If not "
                                         "specified, we will attempt to automatically detect the GCE project from "
                                         "metadata.")
tf.flags.DEFINE_string("gcp_project", None, "[Optional] Project name for the Cloud TPU-enabled project. If not "
                                            "specified, we will attempt to automatically detect the GCE project from "
                                            "metadata.")
tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")
flags.DEFINE_integer("num_tpu_cores", 8, "Only used if `use_tpu` is True. Total number of TPU cores to use.")


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.

    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.

    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids, input_mask, segment_ids, label_id, is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.is_real_example = is_real_example


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with tf.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class SelfProcessor(DataProcessor):
    """Processor for the FenLei data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        file_path = os.path.join(data_dir, 'train.txt')  # cnews.train.txt
        with open(file_path, 'r', encoding="utf-8") as f:
            reader = f.readlines()
        random.seed(0)
        random.shuffle(reader)  # 注意要shuffle

        examples, self.labels = [], []
        for index, line in enumerate(reader):
            guid = 'train-%d' % index
            split_line = line.strip().split("\t")
            text_a = tokenization.convert_to_unicode(split_line[1])
            text_b = None
            label = split_line[0]
            examples.append(InputExample(guid=guid, text_a=text_a,
                                         text_b=text_b, label=label))
            self.labels.append(label)
        return examples

    def get_dev_examples(self, data_dir):
        file_path = os.path.join(data_dir, 'val.txt')
        with open(file_path, 'r', encoding="utf-8") as f:
            reader = f.readlines()
        random.shuffle(reader)

        examples = []
        for index, line in enumerate(reader):
            guid = 'dev-%d' % index
            split_line = line.strip().split("\t")
            text_a = tokenization.convert_to_unicode(split_line[1])
            text_b = None
            label = split_line[0]
            examples.append(InputExample(guid=guid, text_a=text_a,
                                         text_b=text_b, label=label))

        return examples

    def get_test_examples(self, data_dir):
        file_path = os.path.join(data_dir, 'test.txt')
        with open(file_path, 'r', encoding="utf-8") as f:
            reader = f.readlines()
        # random.shuffle(reader)  # 测试集不打乱数据，便于比较

        examples = []
        for index, line in enumerate(reader):
            guid = 'test-%d' % index
            split_line = line.strip().split("\t")
            text_a = tokenization.convert_to_unicode(split_line[1])
            text_b = None
            label = split_line[0]
            examples.append(InputExample(guid=guid, text_a=text_a,
                                         text_b=text_b, label=label))
        return examples

    def one_example(self, sentence):
        guid, label = 'pred-0', self.labels[0]
        text_a, text_b = sentence, None
        return InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)

    def get_labels(self):
        return sorted(set(self.labels), key=self.labels.index)  # 使用有序列表而不是集合。保证了标签正确


def convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        return InputFeatures(
            input_ids=[0] * max_seq_length,
            input_mask=[0] * max_seq_length,
            segment_ids=[0] * max_seq_length,
            label_id=0,
            is_real_example=False)

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    tokens_a = tokenizer.tokenize(example.text_a)
    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)

    if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    label_id = label_map[example.label]
    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label: %s (id = %d)" % (example.label, label_id))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=label_id,
        is_real_example=True)
    return feature


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, ):
    """Creates a classification model."""
    model = modeling.BertModel(
        config=bert_config, is_training=is_training,
        input_ids=input_ids, input_mask=input_mask,
        token_type_ids=segment_ids, use_one_hot_embeddings=False)

    # In the demo, we are doing a simple classification task on the entire segment.
    #
    # If you want to use the token-level output, use model.get_sequence_output() instead.
    output_layer = model.get_pooled_output()

    hidden_size = output_layer.shape[-1].value

    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        if is_training:
            # I.e., 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        probabilities = tf.nn.softmax(logits, axis=-1)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)

        return (loss, per_example_loss, logits, probabilities)


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train, num_warmup, ):
    """Returns `model_fn` closure for GPU Estimator."""

    def model_gpu(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for GPU 版本的 Estimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss, per_example_loss, logits, probabilities) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids, num_labels)

        tvars = tf.trainable_variables()
        initialized_variable_names = {}

        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)

        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(total_loss, learning_rate, num_train, num_warmup, False)
            output_spec = tf.estimator.EstimatorSpec(mode=mode, loss=total_loss, train_op=train_op, )
        elif mode == tf.estimator.ModeKeys.EVAL:
            def metric_fn(per_example_loss, label_ids, logits, is_real_example):
                predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                accuracy = tf.metrics.accuracy(
                    labels=label_ids, predictions=predictions, weights=is_real_example)
                loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
                return {"eval_accuracy": accuracy, "eval_loss": loss, }

            metrics = metric_fn(per_example_loss, label_ids, logits, True)
            output_spec = tf.estimator.EstimatorSpec(mode=mode, loss=total_loss, eval_metric_ops=metrics)
        else:
            output_spec = tf.estimator.EstimatorSpec(mode=mode, predictions={"probabilities": probabilities}, )
        return output_spec

    return model_gpu


# This function is not used by this file but is still used by the Colab and people who depend on it.
def input_fn_builder(features, seq_length, is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []
    all_label_ids = []

    for feature in features:
        all_input_ids.append(feature.input_ids)
        all_input_mask.append(feature.input_mask)
        all_segment_ids.append(feature.segment_ids)
        all_label_ids.append(feature.label_id)

    def input_fn(params):
        """The actual input function."""
        batch_size = 64

        num_examples = len(features)

        # This is for demo purposes and does NOT scale to large data sets. We do   not use Dataset.from_generator()
        # because that uses tf.py_func which is not TPU compatible. The right way to load data is with TFRecordReader.
        d = tf.data.Dataset.from_tensor_slices({
            "input_ids":
                tf.constant(all_input_ids, shape=[num_examples, seq_length], dtype=tf.int32),
            "input_mask":
                tf.constant(all_input_mask, shape=[num_examples, seq_length], dtype=tf.int32),
            "segment_ids":
                tf.constant(all_segment_ids, shape=[num_examples, seq_length], dtype=tf.int32),
            "label_ids":
                tf.constant(all_label_ids, shape=[num_examples], dtype=tf.int32),
        })

        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.batch(batch_size=batch_size)
        return d

    return input_fn


class Bert_Class():

    def __init__(self):
        self.data_dir = '/home/Classification/data'
        self.vocab_file = '/home/chinese_L-12_H-768_A-12/vocab.txt'
        self.config_file = '/home/chinese_L-12_H-768_A-12/bert_config.json'
        self.init_check = '/home/chinese_L-12_H-768_A-12/bert_model.ckpt'
        self.task = 'FenLei'
        self.out_dir = 'output'  # "The output directory where the model checkpoints will be written."
        self.prepare()

    def prepare(self):
        tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case, FLAGS.init_checkpoint)
        self.config = modeling.BertConfig.from_json_file(self.config_file)

        if FLAGS.max_seq_length > self.config.max_position_embeddings:
            raise ValueError(
                "Cannot use sequence length %d because the BERT model "
                "was only trained up to sequence length %d" %
                (FLAGS.max_seq_length, self.config.max_position_embeddings))

        # tf.gfile.MakeDirs(self.out_dir)
        self.tokenizer = tokenization.FullTokenizer(vocab_file=self.vocab_file, do_lower_case=FLAGS.do_lower_case)

        self.train_examples, num_train_steps, num_warmup_steps = None, None, None
        self.processor = SelfProcessor()
        self.train_examples = self.processor.get_train_examples(self.data_dir)
        global label_list
        label_list = self.processor.get_labels()

        run_config = tf.estimator.RunConfig(
            model_dir=self.out_dir, save_checkpoints_steps=FLAGS.save_checkpoints_steps, )
        # tf_random_seed=None,   save_summary_steps=100,  session_config=None, keep_checkpoint_max=5,
        # keep_checkpoint_every_n_hours=10000, log_step_count_steps=100, )

        self.train_steps = int(len(self.train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(self.train_steps * FLAGS.warmup_proportion)

        model_fn = model_fn_builder(bert_config=self.config, num_labels=len(label_list),
                                    init_checkpoint=FLAGS.init_checkpoint, learning_rate=FLAGS.learning_rate,
                                    num_train=num_train_steps, num_warmup=num_warmup_steps)

        self.estimator = tf.estimator.Estimator(model_fn=model_fn, config=run_config, )

    def yuce(self, sentence):
        exam = self.processor.one_example(sentence)  # 待预测的样本列表
        feature = convert_single_example(0, exam, label_list, FLAGS.max_seq_length, self.tokenizer)

        predict_input_fn = input_fn_builder(features=[feature, ],
                                            seq_length=FLAGS.max_seq_length, is_training=False,
                                            drop_remainder=False)
        result = self.estimator.predict(input_fn=predict_input_fn)  # 执行预测操作，得到一个生成器。
        gailv = list(result)[0]["probabilities"].tolist()
        pos = gailv.index(max(gailv))  # 定位到最大概率值索引，
        return label_list[pos]


if __name__ == "__main__":
    toy = Bert_Class()
    import time

    aaa = time.clock()
    sentence = '中国高铁动车组通常运行速度是多少？'
    print(toy.yuce(sentence), sentence)
    bbb = time.clock()
    print('预测用时：', bbb - aaa)
    sentence = '梨树种下之后，过几年能结果子？'
    print(toy.yuce(sentence), sentence)
    aaa = time.clock()
    print('预测用时：', aaa - bbb)
