import math
from Node import Node
from cvor import cvor

class desiciontree:
    def __init__(self,attr,y,vx,vy,max_depth):
        self.attr=attr
        self.y=y
        self.vx=vx
        self.vy=vy
        self.max_depth=max_depth

    def id3(self,d,d_parent,attr,y,depth):
        if self.max_depth=='' or depth<=self.max_depth:
            broji_y_parent = {}
            for i in self.vy:
                broji_y_parent[i]=0
            for i in d_parent:
                broji_y_parent[i[y]]+=1
            pom={}
            for i in broji_y_parent.keys():
                val=broji_y_parent[i]
                if val>0:
                    pom[i]=val
            broji_y_parent=pom
            broji_y_parent = sorted(broji_y_parent.items(), key=lambda a: (-a[1], a[0]))

            if not d:
                nod=Node(broji_y_parent[0][0],None,depth)

                return nod
            else:
                broji_y={}
                for i in d:
                    v=i[y]
                    if(broji_y.get(v,0)!=0):
                        broji_y[v]+=1
                    else:
                        broji_y[v]=1
                broji_y=sorted(broji_y.items(),key=lambda a:(-a[1],a[0]))

                v=broji_y[0][0]
                if not attr:
                    nod=Node(v,None,depth)

                    return nod
                all_v=True
                for i in d:
                    if i[y]!=v:
                        all_v=False
                        break
                if(all_v):
                    nod=Node(v,None,depth)

                    return nod
                e_parent=0
                d_parent_size=len(d_parent)
                for i in broji_y_parent:
                    py=i[1]/d_parent_size

                    e_parent-=py*math.log2(py)

                ig_map={}
                for i in attr:

                    sum_e=0
                    for j in self.vx[i]:
                        e=0
                        opet_broji_y={}
                        for k in self.vy:
                            opet_broji_y[k]=0
                        m_size=0
                        for l in d:
                            if(l[i]==j):
                                opet_broji_y[l[y]]+=1
                                m_size+=1

                        if(m_size>0):
                            for m in opet_broji_y.values():
                                py=m/m_size

                                if(py>0):
                                    e -= py * math.log2(py)

                        dij=m_size/len(d)
                        zbr=dij*e
                        sum_e+=zbr

                    ig_map[i]=e_parent-sum_e

                ig_map=sorted(ig_map.items(),key=lambda a:(-a[1],a[0]))
                print('ig map'+str(ig_map))
                subtrees={}
                x_max=ig_map[0][0]
                pom=[]
                for i in attr:
                    if i!=x_max:
                        pom.append(i)

                new_depth=depth+1
                for i in self.vx[x_max]:

                    new_d=[]
                    for j in d:
                        if(j[x_max]==i):
                            new_d.append(j)

                    t=self.id3(new_d,d,pom,y,new_depth)
                    subtrees[i]=t
                nod=Node(x_max,subtrees,depth)

                return nod
        else:
            broji_y = {}
            if len(d)>0:
                for i in d:
                    v = i[y]
                    if (broji_y.get(v, 0) != 0):
                        broji_y[v] += 1
                    else:
                        broji_y[v] = 1

            else:
                print(d_parent)
                for i in d_parent:
                    v = i[y]
                    if (broji_y.get(v, 0) != 0):
                        broji_y[v] += 1
                    else:
                        broji_y[v] = 1
            broji_y = sorted(broji_y.items(), key=lambda a: (-a[1], a[0]))
            nod = Node(broji_y[0][0], None, depth)

            return nod














    def fit(self,data):
        self.data=data
        self.tree= self.id3(data,data,self.attr,self.y,1)

    def predict(self,test_data):
        parovi=[]
        st='[PREDICTIONS]:'
        for i in test_data:
            mp={}

            n=self.tree
            nx=n.x
            sub=n.subtrees
            while(sub is not None):


                try:
                    vx = i[nx]
                    n=sub[vx]
                    sub=n.subtrees
                    mp[nx]=vx
                    nx=n.x
                except:


                    broji_y={}
                    for j in self.vy:
                        broji_y[j]=0
                    for j in self.data:
                        podskup=True
                        for k in mp.keys():

                            if j[k]!=mp[k]:
                                podskup=False

                        if(podskup==True):
                            broji_y[j[self.y]]+=1
                    broji_y=sorted(broji_y.items(),key=lambda a:(-a[1],a[0]))

                    nx=broji_y[0][0]
                    sub=None


            parovi.append((nx,i[self.y]))
            s=' '+nx
            st+=s
        print(st)
        tocni=0
        for i in parovi:
            if i[0]==i[1]:
                tocni+=1
        omj=tocni/len(test_data)
        s='[ACCURACY]: {:.5f}'.format(omj)
        print(s)
        print('[CONFUSION_MATRIX]:')
        vy_test=set()
        for i in parovi:
            vy_test.add(i[0])
            vy_test.add(i[1])
        vy_test=sorted(vy_test)
        conf_matr={}
        for i in vy_test:
            for j in vy_test:
                conf_matr[(i,j)]=0
        for i in parovi:
            pred=i[0]
            tr=i[1]
            conf_matr[(pred,tr)]+=1
        for i in vy_test:
            s=''
            for j in vy_test:
                st=str(conf_matr[(j,i)])+' '
                s+=st
            print(s.strip())








    def path(self,c):
        s=[]

        n=c.node
        p=c.parent
        pr=c.prijelaz
        while(c is not None):
            if n.subtrees is None:
                st=pr+' '+n.x
                s.append(st)
            else:
                if pr is not None:
                    st = pr + ' ' + str(n.depth) + ':' + n.x + '='
                else:
                    st = str(n.depth) + ':' + n.x + '='

                s.append(st)
            c=p
            if(c is not None):
                p=c.parent
                n=c.node
                pr=c.prijelaz
        s.reverse()
        st=''
        for i in s:
            st+=i
        print(st)

    def ispis(self):
        if(self.max_depth==0):
            print(self.tree.x)
            return
        open=[]
        c=cvor(self.tree,None,None)
        visited=set()
        open.append(c)
        while(open):
            n=open[0]
            open.remove(n)
            pom=n.node

            subtr=pom.subtrees
            if(subtr is None):
                self.path(n)
            else:
                visited.add(n)

                for i in subtr.keys():
                    cv=cvor(subtr[i],n,i)
                    if (not cv in visited):

                        open.insert(0,cv)



