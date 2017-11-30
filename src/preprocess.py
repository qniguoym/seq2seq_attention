import os
import pypinyin
root_dir='/'.join(os.path.realpath(__file__).split('/')[:-2])
data_dir=os.path.join(root_dir,'data')
model_dir=os.path.join(data_dir,'model')
def getdatalabel():
    fw=open(os.path.join(data_dir,'origin/data_label.txt'),'w')
    with open(os.path.join(data_dir,'origin/trainData_time.txt')) as f:
        for line in f:
            line=line.strip()
            line=line.split('\t')
            tmp=pypinyin.lazy_pinyin(line[0],0)
            tmp=' '.join(tmp)
            fw.write(line[0]+'\t'+tmp+'\t'+line[1]+'\n')
def dat_lab():
    fw1=open(os.path.join(data_dir,'origin/data.txt'),'w')
    fw2=open(os.path.join(data_dir,'origin/label.txt'),'w')
    with open(os.path.join(data_dir,'origin/trainData_time.txt')) as f:
        for line in f:
            line=line.strip()
            line=line.split('\t')
            fw1.write(line[0]+'\n')
            fw2.write(line[1]+'\n')
def divide():
    fw1=open(os.path.join(data_dir,'train/content-train.txt'),'w')
    fw2=open(os.path.join(data_dir,'train/time-train.txt'),'w')
    fw3=open(os.path.join(data_dir,'dev/content-dev.txt'),'w')
    fw4=open(os.path.join(data_dir,'dev/time-dev.txt'),'w')
    with open(os.path.join(data_dir,'origin/data.txt')) as f:
        i=0
        for line in f:
            line=line.strip()
            if i<400:
                fw3.write(line+'\n')
            else:
                fw1.write(line+'\n')
            i=i+1
    with open(os.path.join(data_dir,'origin/label.txt')) as f:
        i=0
        for line in f:
            line=line.strip()
            if i<400:
                fw4.write(line+'\n')
            else:
                fw2.write(line+'\n')
            i=i+1
if __name__=='__main__':
    divide()
