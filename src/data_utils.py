import os
from pypinyin import lazy_pinyin
_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_UNK = "_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]
PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3
dic={'':'','0':'零','1':'一','2':'二','3':'三','4':'四','5':'五','6':'六','7':'七','8':'八','9':'九','10':'十','11':'十一','12':'十二','13':'十三','14':'十四','15':'十五','16':'十六','17':'十七','18':'十八','19':'十九','20':'二十','21':'二十一','22':'二十二','23':'二十三','24':'二十四','25':'二十五','26':'二十六','27':'二十七','28':'二十八','29':'二十九','30':'三十','31':'三十一','32':'三十二','33':'三十三','34':'三十四','35':'三十五','36':'三十六','37':'三十七','38':'三十八','39':'三十九','40':'四十','41':'四十一','42':'四十二','43':'四十三','44':'四十四','45':'四十五','46':'四十六','47':'四十七','48':'四十八','49':'四十九','50':'五十','51':'五十一','52':'五十二','53':'五十三','54':'五十四','55':'五十五','56':'五十六','57':'五十七','58':'五十八','59':'五十九'}
dic2={'':'','0':'零','1':'一','2':'两','3':'三','4':'四','5':'五','6':'六','7':'七','8':'八','9':'九','10':'十','11':'十一','12':'十二','13':'十三','14':'十四','15':'十五','16':'十六','17':'十七','18':'十八','19':'十九','20':'二十','21':'二十一','22':'二十二','23':'二十三','24':'二十四'}
root_dir='/'.join(os.path.realpath(__file__).split('/')[:-2])
data_dir=os.path.join(root_dir,'data')
model_dir=os.path.join(data_dir,'model')
def han(num):#月/日
    if len(num)==2:
        if num[0]=='0' and num[1]=='0':
            return ''
        elif num[0]=='0':
            return dic[num[0]]+dic[num[1]]
        else:
            return dic[num]
    if len(num)==1:
        return dic[num[0]]
    else:
        return ''
def num2han(line):
    if '年' in line:
        pos=line.index('年')
        while pos-1>=0 and line[pos-1].isdigit():
            pos=pos-1
            line=line[:pos]+dic[line[pos]]+line[pos+1:]
    if '月' in line:
        pos=line.index('月')
        num=''
        while pos-1>=0 and line[pos-1].isdigit():
            pos=pos-1
            num=line[pos]+num
        line=line[:pos]+dic[num]+line[pos+len(num):]
    if '日' in line:
        pos=line.index('日')
        num=''
        while pos-1>=0 and line[pos-1].isdigit():
            pos=pos-1
            num=line[pos]+num
        line=line[:pos]+dic[num]+line[pos+len(num):]
    if ':' in line:
        pos=line.index(':')
        num=''
        while pos-1>=0 and line[pos-1].isdigit():
            pos=pos-1
            num=line[pos]+num
        line=line[:pos]+dic2[num]+line[pos+len(num):]
        pos=line.index(':')
        tmp=pos
        num=''
        while pos+1<len(line) and line[pos+1].isdigit():
            pos=pos+1
            num=num+line[pos]
        rep=han(num)
        line=line[:tmp]+"&"+rep+line[pos+1:]
    if ':' in line:
        pos=line.index(':')
        num=''
        while pos-1>=0 and line[pos-1].isdigit():
            pos=pos-1
            num=line[pos]+num
        line=line[:pos]+dic2[num]+line[pos+len(num):]
        pos=line.index(':')
        tmp=pos
        num=''
        while pos+1<len(line) and line[pos+1].isdigit():
            pos=pos+1
            num=num+line[pos]
        rep=han(num)
        line=line[:tmp]+"&"+rep+line[pos+1:]
    if '点' in line:
        pos=line.index('点')
        num=''
        while pos-1>=0 and line[pos-1].isdigit():
            pos=pos-1
            num=line[pos]+num
        line=line[:pos]+dic2[num]+line[pos+len(num):]
        pos=line.index('点')
        tmp=pos
        num=''
        while pos+1<len(line) and line[pos+1].isdigit():
            pos=pos+1
            num=num+line[pos]
        rep=han(num)
        print (line)
        line=line[:tmp]+"&"+rep+line[pos+1:]
    if '点' in line:
        pos=line.index('点')
        num=''
        while pos-1>=0 and line[pos-1].isdigit():
            pos=pos-1
            num=line[pos]+num
        line=line[:pos]+dic2[num]+line[pos+len(num):]
        pos=line.index('点')
        tmp=pos
        num=''
        while pos+1<len(line) and line[pos+1].isdigit():
            pos=pos+1
            num=num+line[pos]
        rep=han(num)
        line=line[:tmp]+"&"+rep+line[pos+1:]
    line=line.replace('&','点')
    if '-' in line:
        line=line.replace('-','负')
    for i in line:
        if i in dic:
            line=line.replace(i,dic[i])
    return line
def preprocess(order):
    return num2han(order)
def getvocab(vocab_name):
    f=os.path.join(data_dir,vocab_name)
    vocab=[]
    with open(f) as f:
        for line in f:
            line=line.strip()
            vocab.append(line)
    int_to_vocab={idx:word for idx,word in enumerate(vocab)}
    vocab_to_int={word:idx for idx,word in enumerate(vocab)}
    return int_to_vocab,vocab_to_int
def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size=None):
    vocab = {}
    with open(data_path) as f:
        for line in f:
            for word in line:
                word=word.strip()#将字典中的空白符号去除掉
                if word in vocab:
                    vocab[word] += 1
                else:
                    vocab[word] = 1
    vocab_list = sorted(vocab, key=vocab.get, reverse=True)
    if max_vocabulary_size:
        if len(vocab_list) > max_vocabulary_size:
            vocab_list = vocab_list[:max_vocabulary_size]
    with open(vocabulary_path, 'w') as vocab_file:
        for le in _START_VOCAB:
            vocab_file.write(le+'\n')
        for w in vocab_list:
            vocab_file.write(w + '\n')
def initialize_vocabulary(vocabulary_path):
    rev_vocab = []
    with open(vocabulary_path) as f:
        rev_vocab.extend(f.readlines())
    rev_vocab = [line.strip() for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab
def sentenceofhan_to_token_ids(sentence, vocabulary):
    token_ids=[]
    for word in sentence:
        id=vocabulary.get(word)
        if id is not None:
            pin=''.join(lazy_pinyin(word,0))
            lens=len(pin)
            while lens>0:
                token_ids.append(id)
                lens=lens-1
        else:
            token_ids.append(vocabulary.get(_UNK))
    return token_ids
def sentenceofpin_to_token_ids(sentence, vocabulary):
    token_ids=[]
    for word in sentence:
        if word in vocabulary:
            token_ids.append(vocabulary.get(word))
        else:
            for i in word:
                if i in vocabulary:
                    token_ids.append(vocabulary.get(i))
                else:
                    token_ids.append(vocabulary.get(_UNK))
    return token_ids
def sentenceofpin_to_token_ids2(sentence, vocabulary):
    token_ids=[]
    for word in sentence:
        if word in vocabulary:
            pinyin=lazy_pinyin(word,0)[0]
            for i in pinyin:
                token_ids.append(vocabulary.get(i))
        else:
            token_ids.append(vocabulary.get(_UNK))
    return token_ids
def data_to_token_ids(data_path, target_path_han, vocabulary_path,target_path_pin=None):
    if target_path_pin:
        vocab, _ = initialize_vocabulary(vocabulary_path)
        tokens_file1=open(target_path_han,'w')
        tokens_file2=open(target_path_pin,'w')
        with open(data_path) as data_file:
            counter = 0
            for line in data_file:
                counter += 1
                line=line.strip()
                token_ids = sentenceofhan_to_token_ids(line, vocab)
                tokens_file1.write(' '.join([str(tok) for tok in token_ids])+'\n')

                _line=[]
                for word in line:
                    if word in vocab:
                        _line.append(word)
                    else:
                        _line.append(_UNK)
                pinyin=(lazy_pinyin(_line,0))
                token_ids = sentenceofpin_to_token_ids(pinyin, vocab)
                tokens_file2.write(' '.join([str(tok) for tok in token_ids])+'\n')
        tokens_file1.close()
        tokens_file2.close()
    else:
        vocab, _ = initialize_vocabulary(vocabulary_path)
        tokens_file1=open(target_path_han,'w')
        with open(data_path) as data_file:
            for line in data_file:
                line=line.strip()
                token_ids = sentenceofhan_to_token_ids(line, vocab)
                tokens_file1.write(' '.join([str(tok) for tok in token_ids])+'\n')
        tokens_file1.close()
def prepare_headline_data(data_dir, vocabulary_size=None):
    data_label=os.path.join(data_dir,'origin/data_label.txt')
    train_path = os.path.join(data_dir, "train")
    src_train_path = os.path.join(train_path, "content-train.txt")
    dest_train_path = os.path.join(train_path, "time-train.txt")

    dev_path = os.path.join(data_dir, "dev")
    src_dev_path = os.path.join(dev_path, "content-dev.txt")
    dest_dev_path = os.path.join(dev_path, "time-dev.txt")

    # Create vocabularies of the appropriate sizes.
    vocab_source = os.path.join(data_dir, "vocab/vocab_source.txt")
    vocab_target = os.path.join(data_dir, "vocab/vocab_target.txt")
    create_vocabulary(vocab_source,data_label)

    # Create token ids for the training data.
    src_train_hanids_path = os.path.join(train_path, "content_train_hanid.txt")
    dest_train_ids_path = os.path.join(train_path, "time_train_id.txt")
    src_train_pinids_path = os.path.join(train_path, "content_train_pinid.txt")
    data_to_token_ids(src_train_path, src_train_hanids_path, vocab_source, src_train_pinids_path)
    data_to_token_ids(dest_train_path, dest_train_ids_path, vocab_target,dest_train_ids_path)

    # Create token ids for the development data.
    src_dev_hanids_path = os.path.join(dev_path, "content_dev_hanid.txt")
    dest_dev_ids_path = os.path.join(dev_path, "time_dev_id.txt")
    src_dev_pinids_path = os.path.join(dev_path, "content_dev_pinid.txt")
    data_to_token_ids(src_dev_path, src_dev_hanids_path, vocab_source, src_dev_pinids_path)
    data_to_token_ids(dest_dev_path, dest_dev_ids_path,vocab_target)

    return (src_train_hanids_path, src_train_pinids_path, dest_train_ids_path,
          src_dev_hanids_path, src_dev_pinids_path, dest_dev_ids_path,
          vocab_source,vocab_target)
if __name__=='__main__':
    a=preprocess('我要看2016年12月6日晚上11:33，红光山北门的录像，有三倍速回放。')
    print (a)