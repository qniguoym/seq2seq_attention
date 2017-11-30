import tensorflow as tf
import os
import sys
root_dir='/'.join(os.path.realpath(__file__).split('/')[:-2])
sys.path.append(root_dir)
import numpy as np
import data_utils
import export_func
from grpc.beta import implementations
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
import operator
import pypinyin
def read_data(source_path, max_size=None):#[[[source],[target]]]
        source_data=[]
        with open(source_path) as source_file:
                source = source_file.readline()
                counter = 0
                while source and (not max_size or counter < max_size):
                    counter += 1
                    source_ids = [int(x) for x in source.split()]
                    source_data.append(source_ids)
                    source = source_file.readline()
        return source_data
class Model():
    def __init__(self,is_training=True):
        self.epochs=1000000
        self.rnn_size=50
        self.num_layers=2
        self.encoding_embedding_size=15
        self.decoding_embedding_size=15
        self.learning_rate=0.001
        self.root_dir='/'.join(os.path.realpath(__file__).split('/')[:-2])
        self.data_dir=os.path.join(self.root_dir,'data')
        self.model_dir=os.path.join(self.data_dir,'model')
        self.src_train_han, self.src_train_pin, self.dest_train, self.src_dev_han,self.src_dev_pin,self.dest_dev, self.source_vocab_path, self.target_vocab_path=self.get_path()
        self.target_int_to_letter,self.target_letter_to_int,=data_utils.getvocab(self.target_vocab_path)
        self.source_int_to_letter,self.source_letter_to_int=data_utils.getvocab(self.source_vocab_path)
        self.source_vocab_size=len(self.source_letter_to_int)
        self.target_vocab_size=len(self.target_letter_to_int)
        if is_training==False:
            self.batch_size=1
        self.build()
    def get_path(self):
        root_dir='/'.join(os.path.realpath(__file__).split('/')[:-2])
        data_dir=os.path.join(root_dir,'data')
        train_path = os.path.join(data_dir, "train")

        dev_path = os.path.join(data_dir, "dev")
        # Create vocabularies of the appropriate sizes.
        vocab_source = os.path.join(data_dir, "vocab/vocab_source.txt")
        vocab_target = os.path.join(data_dir, "vocab/vocab_target.txt")

        # Create token ids for the training data.
        src_train_hanids_path = os.path.join(train_path, "content_train_hanid.txt")
        dest_train_ids_path = os.path.join(train_path, "time_train_id.txt")
        src_train_pinids_path = os.path.join(train_path, "content_train_pinid.txt")
        # Create token ids for the development data.
        src_dev_hanids_path = os.path.join(dev_path, "content_dev_hanid.txt")
        dest_dev_ids_path = os.path.join(dev_path, "time_dev_id.txt")
        src_dev_pinids_path = os.path.join(dev_path, "content_dev_pinid.txt")
        return (src_train_hanids_path, src_train_pinids_path, dest_train_ids_path,
              src_dev_hanids_path, src_dev_pinids_path, dest_dev_ids_path,
              vocab_source,vocab_target)
    def pad_sentence_batch(self,sentence_batch, pad_int):
        max_sentence = max([len(sentence) for sentence in sentence_batch])
        return [sentence + [pad_int] * (max_sentence - len(sentence)) for sentence in sentence_batch]
    def get_batches(self,targets, sources, sources_pin, batch_size, source_pad_int, target_pad_int):
        for batch_i in range(0, len(sources)//batch_size):
            start_i = batch_i * batch_size
            sources_batch = sources[start_i:start_i + batch_size]
            sources_batch_pin = sources_pin[start_i:start_i + batch_size]
            targets_batch = targets[start_i:start_i + batch_size]
            # 补全序列
            pad_sources_batch = np.array(self.pad_sentence_batch(sources_batch, source_pad_int))
            pad_sources_batch_pin = np.array(self.pad_sentence_batch(sources_batch_pin, source_pad_int))
            pad_targets_batch = np.array(self.pad_sentence_batch(targets_batch, target_pad_int))

            # 记录每条记录的长度
            targets_lengths = []
            for target in targets_batch:
                targets_lengths.append(len(target))

            source_lengths = []
            for source in sources_batch:
                source_lengths.append(len(source))

            yield pad_targets_batch, pad_sources_batch,pad_sources_batch_pin, targets_lengths, source_lengths
    def get_inputs(self):
        self.input_data=tf.placeholder(dtype=tf.int32,shape=[None,None],name="inputs")
        self.input_data_pin=tf.placeholder(dtype=tf.int32,shape=[None,None],name="inputs_pin")
        self.targets=tf.placeholder(dtype=tf.int32,shape=[None,None],name="inputs")
        self.lr=tf.placeholder(tf.float32,name="learning_rate")
        # 定义target序列最大长度（之后target_sequence_length和source_sequence_length会作为feed_dict的参数）
        self.target_sequence_length = tf.placeholder(tf.int32, (None,), name='target_sequence_length')
        self.batch_size = tf.shape(self.input_data)[0]
        self.max_target_sequence_length = tf.reduce_max(self.target_sequence_length, name='max_target_len')
        self.source_sequence_length = tf.placeholder(tf.int32, (None,), name='source_sequence_length')
    def process_decoder_input(self,data, vocab_to_int, batch_size):
        #补充<GO>，并移除最后一个字符 cut掉最后一个字符
        ending = tf.strided_slice(data, [0, 0], [batch_size, -1], [1, 1])
        decoder_input = tf.concat([tf.fill([batch_size, 1], vocab_to_int['_GO']), ending], 1)
        return decoder_input
    def get_encoder_layer(self):
        # Encoder embedding
        weights=tf.get_variable("weights",initializer=tf.random_normal(shape=[self.source_vocab_size,self.encoding_embedding_size],stddev=0.1))
        #encoder_embed_input1 = tf.contrib.layers.embed_sequence(input_data, source_vocab_size, encoding_embedding_size)
        encoder_embed_input1 = tf.nn.embedding_lookup(weights,self.input_data)
        encoder_embed_input2 = tf.nn.embedding_lookup(weights,self.input_data_pin)
        encoder_embed_input = tf.concat([encoder_embed_input1,encoder_embed_input2],-1)

        # RNN cell
        def get_lstm_cell(rnn_size):
            lstm_cell = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            return lstm_cell

        cell = tf.contrib.rnn.MultiRNNCell([get_lstm_cell(self.rnn_size) for _ in range(self.num_layers)])

        self.encoder_output, self.encoder_state = tf.nn.dynamic_rnn(cell, encoder_embed_input,
                                                          sequence_length=self.source_sequence_length, dtype=tf.float32)
    def decoding_layer(self):
        memory=self.encoder_output
        attention_mechanism=tf.contrib.seq2seq.BahdanauAttention(num_units=self.rnn_size,memory=memory,memory_sequence_length=self.source_sequence_length)
        def get_decoder_cell(rnn_size):
            decoder_cell = tf.contrib.rnn.LSTMCell(rnn_size,initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            return decoder_cell
        cell = tf.contrib.rnn.MultiRNNCell([get_decoder_cell(self.rnn_size) for _ in range(self.num_layers)])

        attn_cell=tf.contrib.seq2seq.AttentionWrapper(cell, attention_mechanism, attention_layer_size=self.rnn_size)
        out_cell=tf.contrib.rnn.OutputProjectionWrapper(attn_cell,self.target_vocab_size)

        decoder_embeddings = tf.Variable(tf.random_uniform([self.target_vocab_size, self.decoding_embedding_size]))
        decoder_embed_input = tf.nn.embedding_lookup(decoder_embeddings, self.decoder_input)

        # 4. Training decoder
        with tf.variable_scope("decode"):
            # 得到help对象
            training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_embed_input,
                                                                sequence_length=self.target_sequence_length,
                                                                time_major=False)
            #当time_major=False,inputs的shape为[batch_size,sequence_length,embedding_size]
            #当time_major=True,inputs的shape为[sequence_length,batch_size,embedding_size]

            # 构造decoder
            training_decoder = tf.contrib.seq2seq.BasicDecoder(out_cell,
                                                               training_helper,
                                                               initial_state=out_cell.zero_state(dtype=tf.float32,batch_size=self.batch_size))
            #encoder_state是encoder的final state,直接将encoder的final_state作为这个参数输入即可
            training_decoder_output, _,_ = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                                           impute_finished=True,
                                                                           maximum_iterations=self.max_target_sequence_length)
            #impute_finished 状态被复制,输出置为0
        # 5. Predicting decoder
        # 与training共享参数
        with tf.variable_scope("decode", reuse=True):
            # 创建一个常量tensor并复制为batch_size的大小
            start_tokens = tf.tile(tf.constant([self.target_letter_to_int['_GO']], dtype=tf.int32), [self.batch_size],
                                   name='start_tokens')
            predicting_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(decoder_embeddings,
                                                                    start_tokens,
                                                                    self.target_letter_to_int['_EOS'])
            #这是用于inference阶段的helper,将output输出后的logits使用argmax获得id再经过embedding_layer来获取下一时刻的输出
            predicting_decoder = tf.contrib.seq2seq.BasicDecoder(out_cell,
                                                            predicting_helper,
                                                            initial_state=out_cell.zero_state(dtype=tf.float32,batch_size=self.batch_size))
            predicting_decoder_output, _ ,_= tf.contrib.seq2seq.dynamic_decode(predicting_decoder,
                                                                impute_finished=True,
                                                                maximum_iterations=self.max_target_sequence_length)

        return training_decoder_output, predicting_decoder_output
    def seq2seq_model(self):
        # 获取encoder的状态输出
        self.get_encoder_layer()
        # 预处理后的decoder输入
        self.decoder_input = self.process_decoder_input(self.targets, self.target_letter_to_int, self.batch_size)

        # 将状态向量与输入传递给decoder
        self.training_decoder_output, self.predicting_decoder_output = self.decoding_layer()
    def build(self):
        self.train_graph = tf.Graph()
        with self.train_graph.as_default():
            # 获得模型输入
            self.get_inputs()
            self.seq2seq_model()
            self.training_logits = tf.identity(self.training_decoder_output.rnn_output, 'logits')
            self.predicting_logits = tf.identity(self.predicting_decoder_output.sample_id, name='predictions')
            masks = tf.sequence_mask(self.target_sequence_length, self.max_target_sequence_length, dtype=tf.float32, name='masks')
            with tf.name_scope("optimization"):
                # Loss function
                self.cost = tf.contrib.seq2seq.sequence_loss(
                    self.training_logits,#18代表了候选词表的数量
                    self.targets,
                    masks)
                # Optimizer
                optimizer = tf.train.AdamOptimizer(self.lr)
                # Gradient Clipping
                gradients = optimizer.compute_gradients(self.cost)
                capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
                self.train_op = optimizer.apply_gradients(capped_gradients)
def train():
    model=Model()
    train_source = read_data(model.src_train_han)
    train_source_pin = read_data(model.src_train_pin)
    train_target = read_data(model.dest_train)
    valid_source = read_data(model.src_dev_han)
    valid_source_pin = read_data(model.src_dev_pin)
    valid_target = read_data(model.dest_dev)

    batch_size=128
    (valid_targets_batch, valid_sources_batch,valid_sources_batch_pin, valid_targets_lengths, valid_sources_lengths) = next(model.get_batches(valid_target, valid_source, valid_source_pin,batch_size,model.source_letter_to_int['_PAD'],model.target_letter_to_int['_PAD']))
    display_step = 12
    checkpoint = model.model_dir
    min=10
    flag=0

    with tf.Session(graph=model.train_graph) as sess:
        sess.run(tf.global_variables_initializer())
        for epoch_i in range(1, model.epochs+1):
            for batch_i, (targets_batch, sources_batch,sources_batch_pin, targets_lengths, sources_lengths) in enumerate(model.get_batches(train_target, train_source, train_source_pin,batch_size, model.source_letter_to_int['_PAD'],model.target_letter_to_int['_PAD'])):
                pp,tt,_, loss = sess.run(
                    [model.predicting_logits,model.targets,model.train_op, model.cost],
                    {model.input_data: sources_batch,
                     model.input_data_pin:sources_batch_pin,
                     model.targets: targets_batch,
                     model.lr: model.learning_rate,
                     model.target_sequence_length: targets_lengths,
                     model.source_sequence_length: sources_lengths})
                if batch_i % display_step == 0:
                    # 计算validation loss
                    p,tar,validation_loss = sess.run(
                    [model.predicting_logits,model.targets,model.cost],
                    {model.input_data: valid_sources_batch,
                     model.input_data_pin:valid_sources_batch_pin,
                     model.targets: valid_targets_batch,
                     model.lr: model.learning_rate,
                     model.target_sequence_length: valid_targets_lengths,
                     model.source_sequence_length: valid_sources_lengths})
                    print (pp[1],tt[1])
                    print (p[1],tar[1])
                    print('Epoch {:>3}/{} Batch {:>4}/{} - Training Loss: {:>6.3f}  - Validation loss: {:>6.3f}'
                          .format(epoch_i,
                                  model.epochs,
                                  batch_i,
                                  len(train_source) // model.batch_size,
                                  loss,
                                  validation_loss))
                    if min>validation_loss:
                        min=validation_loss
                        export_func.export(model,sess,signature_name='seq2seq',export_path=checkpoint,version=epoch_i*(batch_i+1))
                        print('Model Trained and Saved')
                        flag=1
                        break
            if flag==1:
                break
def predict(order):
    order=data_utils.preprocess(order)
    data_dir=os.path.join(root_dir,'data')
    source_vocab_path = os.path.join(data_dir, "vocab/vocab_source.txt")
    target_vocab_path = os.path.join(data_dir, "vocab/vocab_target.txt")
    target_int_to_letter,target_letter_to_int=data_utils.getvocab(target_vocab_path)
    source_int_to_letter,source_letter_to_int=data_utils.getvocab(source_vocab_path)
    source=[data_utils.sentenceofhan_to_token_ids(order,source_letter_to_int)]
    source_pin=[data_utils.sentenceofpin_to_token_ids2(order,source_letter_to_int)]
    target=[[0,0,0,0,0,0,0,0,0,0,0,0]]
    lr=0
    source_len=[len(source[0])]
    target_len=[len(target[0])]
    hostport='192.168.31.186:6000'
    host,port=hostport.split(':')
    #grpc
    channel=implementations.insecure_channel(host,int(port))
    stub=prediction_service_pb2.beta_create_PredictionService_stub(channel)
    #build request
    request= predict_pb2.PredictRequest()
    request.model_spec.name='seq2seq'
    request.model_spec.signature_name='seq2seq'
    request.inputs['input_data'].CopyFrom(tf.contrib.util.make_tensor_proto(source,dtype=tf.int32))
    request.inputs['input_data_pin'].CopyFrom(tf.contrib.util.make_tensor_proto(source_pin,dtype=tf.int32))
    request.inputs['targets'].CopyFrom(tf.contrib.util.make_tensor_proto(target,dtype=tf.int32))
    request.inputs['lr'].CopyFrom(tf.contrib.util.make_tensor_proto(lr,dtype=tf.float32))
    request.inputs['source_sequence_length'].CopyFrom(tf.contrib.util.make_tensor_proto(source_len,dtype=tf.int32))
    request.inputs['target_sequence_length'].CopyFrom(tf.contrib.util.make_tensor_proto(target_len,dtype=tf.int32))
    model_result=stub.Predict(request,60.0)
    output=np.array(model_result.outputs['output'].int_val)
    ans_=[target_int_to_letter.get(i) for i in output]
    ans_=''.join(ans_)
    ans_=ans_.replace('S','')
    st='NN:NN'
    et='NN:NN'
    if 'E' in ans_:
        tmp=ans_.split('E')
        if len(tmp)==2:
            st=tmp[0]
            et=tmp[1]
    print (st,et)
    return st,et

if __name__ == "__main__":
    #train()
    with open(os.path.join(data_utils.data_dir,'test/test.txt')) as f:
        for line in f:
            line='查看汇嘉时代广场东门今年9月5日凌晨1:47的录像，给我付三倍速回放'
            print (line)
            predict(line)
            break



