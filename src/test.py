import os
root_dir='/'.join(os.path.realpath(__file__).split('/')[:-2])
data_dir=os.path.join(root_dir,'data')
model_dir=os.path.join(data_dir,'model')
with open(os.path.join(data_dir,'ans1.txt')) as f:
    count=0
    sum=0
    for line in f:
        count=count+1
        line=line.split('\t')
        a=''
        for i in line[0]:
            if i.isdigit():
                a=a+i
        b=''
        for i in line[0]:
            if i.isdigit():
                b=b+i
        if a==b:
            sum=sum+1
    print (count,sum,float(sum)/count)

