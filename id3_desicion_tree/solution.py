import sys
import desiciontree

arg_len=len(sys.argv)
data=open(r"{}".format(sys.argv[1]),'r')
test_data=open(r"{}".format(sys.argv[2],'r'))
if arg_len==4:
    max_depth=int(sys.argv[3])
else:
    max_depth=''

d=[]
test_d=[]
arr=data.readline().strip().split(',')
attr=arr[:-1]
y=arr[-1]
vx={}
vy=set()

for i in attr:
    vx[i]=set()
line=data.readline()

while(line):
    arr=line.strip().split(',')
    pod={}
    for i in range(len(arr)-1):
        x=arr[i]
        at=attr[i]
        pod[at]=x
        vx[at].add(x)
    y_val=arr[-1]
    pod[y]=y_val
    vy.add(y_val)
    d.append(pod)
    line=data.readline()
test_data.readline()
line=test_data.readline()

while(line):
    arr=line.strip().split(',')
    pod={}
    for i in range(len(arr)-1):
        x=arr[i]
        at=attr[i]
        pod[at]=x

    y_val=arr[-1]
    pod[y]=y_val

    test_d.append(pod)
    line=test_data.readline()
model=desiciontree.desiciontree(attr,y,vx,vy,max_depth)

model.fit(d)

print('[BRANCHES]:')
model.ispis()
model.predict(test_d)
