# ACM模板

**整理人：李开，孔繁初，王德涵**

[TOC]

## 前言

略

## 模板代码

### 起始模板

```cpp
#define INL inline
#define REG register
#define U unsigned
#define M ((l+r)>>1)
#define _rep(i,a,b) for(int i=a;i<=b;i++)
#define _for(i,a,b) for(int i=a;i<b;i++)
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <map>
#include <memory.h>
#include <queue>
#include <set>
#include <stack>
#include <string>
#include <vector>
#include <unordered_map>
using namespace std;
typedef long long ll;
typedef unsigned long long ull;

int main(){
    
    return 0;
}
```

### 对拍模板（待补充）

### 造数据模板（待补充）

## 计算几何

```cpp
const double eps = 1e-9;

struct Point {
    double x, y;
    Point(double x, double y): x(x), y(y) {}
};

typedef Point Vector;

Vector operator + (Vector A, Vector B) { return Vector(A.x+B.x, A.y+B.y); }
Vector operator - (Point A, Point B) { return Vector(A.x-B.x, A.y-B.y); }
Vector operator * (Vector A, double t) { return Vector(A.x * t, A.y * t); }
Vector operator / (Vector A, double t) { return Vector(A.x / t, A.y / t); }

int dcmp(double x) { if(fabs(x) < eps) return 0; else return x > 0 ? 1 : -1; }
bool operator == (Point A, Point B) { return dcmp(A.x-B.x) == 0 && dcmp(A.y-B.y) == 0; }

double Dot(Vector A, Vector B) { return A.x * B.x + A.y * B.y; }

double Cross(Vector A, Vector B) { return A.y * B.x - A.x * B.y; }

double length(Vector A) { return sqrt(A.x * A.x + A.y * A.y); }

Vector Normal(Vector A) {
    double L = length(A);
    return Vector(-A.y/L, A.x/L);
}

Point GetLineInsection(Point P, Vector v, Point Q, Vector w) {
    Vector u = P - Q;
    double t = Cross(w, u) / Cross(v, w);
    return P + v * t;
    //return t1 >= 0 && t2 >= 0 ? P + v * t1 : nullptr;
}

bool SegmentProperIntersection(Point a1, Point a2, Point b1, Point b2) {
    double c1 = Cross(a2-a1, b1-a1), c2 = Cross(a2-a1, b2-a1);
    double c3 = Cross(b2-b1, a1-b1), c4 = Cross(b2-b1, a2-b1);
    return dcmp(c1)*dcmp(c2) < 0 && dcmp(c3)*dcmp(c4) < 0;
}

double DistanceToLine(Point P, Point A, Point B) {
    Vector v1 = B-A, v2 = P-A;
    return fabs(Cross(v1, v2) / length(v1));
}

double Area(Point A, Point B, Point C) {
    double LA, LB, LC;
    LA = length(A);
    LB = length(B);
    LC = length(C);
    double p = (LA+LB+LC) / 2;
    return sqrt(p * (p-LA) * (p-LB) * (p-LC));
}

double Area2(Point A, Point B, Point C) {
    return Cross(B-A, C-A);
}
```

## 数据结构

### 树状数组

最简单的数据结构，也是功能最少的，只能支持符合前缀和性质的操作，比如区间加减、异或；连区间最值都维护不了。

可以用来套主席树，比一般的树套树的常数更小。

模板：[洛谷3374 【模板】树状数组 1](https://www.luogu.com.cn/problem/P3374)

```cpp
const int maxn=114514*5;
int tree[maxn],n,m;
INL void update(REG int x,REG int pos){//单点加
    while(pos<=n){
        tree[pos]+=x;
        pos+=lowbit(pos);
    }
}
INL int query(REG int pos){//查询前缀和（注意是前缀和，所以区间查询要查询两次）
    REG int tmp=0;
    while(pos){
        tmp+=tree[pos];
        pos-=lowbit(pos);
    }
    return tmp;
}
int main(){
    n=read();
    m=read();
    for(REG int i=1,tmp;i<=n;i++){
        tmp=read();
        update(tmp,i);
    }
    while(m--){
        REG int op=read(),x=read(),y=read();
        switch (op){
        case 1:
            update(y,x);
            break;
        
        case 2:
            printf("%d\n",query(y)-query(x-1));
            break;
        }
    }
    return 0;
}
```

### 线段树

线段树能干的事海了去了，只要是符合结合律（不严谨，实际上是满足幺半群性质，一般认为是符合结合律即可）的信息基本都能维护。

笑死，原理这么简单，要啥板子，直接手搓

#### 李超树（待补充）



#### 吉老师线段树（势能线段树，待补充）

吉老师线段树是一种**需要暴力向下递归**的线段树，这种线段树需要**势能分析**来证明复杂度。

一般吉老师线段树的作用是**区间历史最值**和**区间取最值操作**。

~~一般吉老师线段树都很难写。~~



#### 主席树（可持久化线段树）

不带修、符合前缀和性质的信息就可以用主席树存，时空复杂度都很优。

**注意主席树（以及各种动态开点线段树）开空间的逻辑：$操作次数 \times \log (下标值域大小)$ **

模板1：[洛谷3919 【模板】可持久化线段树 1（可持久化数组）](https://www.luogu.com.cn/problem/P3919)

```cpp
const int maxn=1145140;
char buf[100005],*p1=buf,*p2=buf;
inline char nc(){
    if(p1==p2){
        p1=buf;
        p2=p1+fread(buf,1,100000,stdin);
        if(p1==p2){
            return EOF;
        }
    }
    return *p1++;
}

class PerSegTree{
    #define LS tree[pos].lson
    #define RS tree[pos].rson
    private:
    struct Node{
        int lson,rson,val;
    }tree[1145140*2*20];
    int siz;
    int copyNode(int pos){//主席树核心操作：复制节点
        tree[++siz]=tree[pos];
        return siz;
    }
    public:
    void update(int tar,int v,int l,int r,int& pos){
        pos=copyNode(pos);
        if(l==r){
            tree[pos].val=v;
            return;
        }
        if(tar<=M){
            update(tar,v,l,M,LS);
        }
        else{
            update(tar,v,M+1,r,RS);
        }
    }
    int query(int tar,int l,int r,int pos){
        if(l==r){
            return tree[pos].val;
        }
        if(tar<=M){
            return query(tar,l,M,LS);
        }
        else{
            return query(tar,M+1,r,RS);
        }
    }
    #undef LS
    #undef RS
}seg;
int n,root[maxn];

int main(){
    n=read();
    int m=read();
    for(int i=1;i<=n;i++){
        seg.update(i,read(),1,n,root[0]);
    }
    for(int i=1;i<=m;i++){
        int lasv=read(),op=read(),idx=read();
        root[i]=root[lasv];
        if(op==1){
            int val=read();
            seg.update(idx,val,1,n,root[i]);
        }
        else{
            printf("%d\n",seg.query(idx,1,n,root[i]));
        }
    }
    return 0;
}
```

模板2：[洛谷3834 【模板】可持久化线段树 2](https://www.luogu.com.cn/problem/P3834)（主席树的起源&灵魂）

```cpp
const int maxn=114514<<1,shift=1000000000,ub=2000000005;
char buf[100005],*p1=buf,*p2=buf;
inline char nc(){
    if(p1==p2){
        p1=buf;
        p2=p1+fread(buf,1,100000,stdin);
        if(p1==p2){
            return EOF;
        }
    }
    return *p1++;
}

class PerSegTree{
    #define LS tree[pos].lson
    #define RS tree[pos].rson
    private:
    struct Node{
        int lson,rson,siz;
    }tree[maxn*33];
    int siz;
    int copyNode(int pos){
        tree[++siz]=tree[pos];
        return siz;
    }
    void upload(int pos){
        tree[pos].siz=tree[LS].siz+tree[RS].siz;
    }
    public:
    void insert(ll v,ll l,ll r,int& pos){
        pos=copyNode(pos);
        if(l==r){
            tree[pos].siz++;
            return;
        }
        if(v<=M){
            insert(v,l,M,LS);
        }
        else{
            insert(v,M+1,r,RS);
        }
        upload(pos);
    }
    int queryKth(int k,ll l,ll r,int lpos,int rpos){
        if(l==r){
            return l;
        }
        if(k>=tree[tree[rpos].lson].siz-tree[tree[lpos].lson].siz){
            return queryKth(k-tree[tree[rpos].lson].siz+tree[tree[lpos].lson].siz,M+1,r,tree[lpos].rson,tree[rpos].rson);
        }
        else{
            return queryKth(k,l,M,tree[lpos].lson,tree[rpos].lson);
        }
    }
    void display(ll l,ll r,int pos){//主席树的遍历函数，用于调试
        if(!pos){
            return;
        }
        if(l==r){
            cout<<l-shift<<' ';
            return;
        }
        display(l,M,LS);
        display(M+1,r,RS);
    }
}seg;

int n,root[maxn];

int main(){
    n=read();
    int m=read();
    for(int i=1;i<=n;i++){
        root[i]=root[i-1];
        ll v=read();
        seg.insert(v+shift,0,ub,root[i]);//题目的值域涉及负数，但线段树/主席树的下标必须非负，所以要平移一下
    }
    while(m--){
        int l=read(),r=read(),k=read();
        printf("%d\n",seg.queryKth(k-1,0,ub,root[l-1],root[r])-shift);
    }
    return 0;
}
```

#### 线段树上二分

如果我们遇到这样一种操作，“二分一个下标，再在线段树上查询二分出来的这个下标的信息”，即“先二分再上树”，那我们就可以合并“二分”和“上树”，写成线段树上二分。

**实际上，主席树模板2中的$queryKth$函数就是一个线段树（值域树）上二分的例子。**

模板就不放了，参考 $\uparrow$ 即可

#### 线段树分裂/合并

顾名思义，线段树分裂就是从一个下标位置把线段树劈成两半（类似于Treap的split操作），合并就是把两颗线段树并成一颗；

线段树分裂毫无疑问是单次$O(\log n)$的，而线段树合并可以证明是均摊$O(\log n)$的。

除了你能想到的那些常见作用之外，线段树分裂/合并还可以用于**区间排序操作**。

 模板：[洛谷5494 【模板】线段树分裂](https://www.luogu.com.cn/problem/P5494)

```cpp
const int maxn=2*114514;
char buf[100005],*p1=buf,*p2=buf;
inline char nc(){
    if(p1==p2){
        p1=buf;
        p2=p1+fread(buf,1,100000,stdin);
        if(p1==p2){
            return EOF;
        }
    }
    return *p1++;
}

int n,m;
class SegTree{
    #define LS tree[pos].lson
    #define RS tree[pos].rson
    private:
    struct treenode{
        int lson,rson;
        ll siz;
    }tree[maxn<<6];
    int cntr;
    stack<int> recyc;//节点回收
    INL void delet(int pos){
        recyc.push(pos);
    }
    INL int addNode(){
        int pos=recyc.empty()?(++cntr):recyc.top();
        if(!recyc.empty()){
            recyc.pop();
        }
        LS=RS=0;
        tree[pos].siz=0;
        return pos;
    }
    INL void upload(int pos){
        tree[pos].siz=tree[LS].siz+tree[RS].siz;
    }
    public:
    void split(int al,int ar,int l,int r,int& pos,int& npos){//线段树分裂，将[l,r]范围的节点分裂出来给npos
        if(!pos){
            return;
        }
        if(al<=l&&ar>=r){
            npos=pos;
            pos=0;
            return;
        }
        if(!npos){
            npos=addNode();
        }
        if(al<=M){
            split(al,ar,l,M,LS,tree[npos].lson);
        }
        if(ar>M){
            split(al,ar,M+1,r,RS,tree[npos].rson);
        }
        upload(pos);
        upload(npos);
    }
    int merge(int pos,int rpos){//线段树合并，将rpos合并给pos，抛弃rpos
        if(!pos||!rpos){
            return pos^rpos;
        }
        tree[pos].siz+=tree[rpos].siz;
        LS=merge(LS,tree[rpos].lson);
        RS=merge(RS,tree[rpos].rson);
        delet(rpos);
        return pos;
    }
    void insert(int tar,ll d,int l,int r,int& pos){
        if(!pos){
            pos=addNode();
        }
        if(l==r){
            tree[pos].siz+=d;
            return;
        }
        if(tar<=M){
            insert(tar,d,l,M,LS);
        }
        else{
            insert(tar,d,M+1,r,RS);
        }
        upload(pos);
    }
    ll query(int al,int ar,int l,int r,int pos){
        if(!pos){
            return 0;
        }
        if(al<=l&&ar>=r){
            return tree[pos].siz;
        }
        ll ans=0;
        if(al<=M){
            ans+=query(al,ar,l,M,LS);
        }
        if(ar>M){
            ans+=query(al,ar,M+1,r,RS);
        }
        return ans;
    }
    int searchKth(int k,int pos){
        int l=1,r=n;
        while(l!=r){
            if(tree[pos].siz<=k){
                return -1;
            }
            if(tree[LS].siz<=k){
                k-=tree[LS].siz;
                pos=RS;
                l=M+1;
            }
            else{
                pos=LS;
                r=M;
            }
        }
        return l;
    }
}seg;
int root[maxn],rcntr=2;
int main(){
    n=read();
    m=read();
    for(int i=1;i<=n;i++){
        int d=read();
        seg.insert(i,d,1,n,root[1]);
    }
    while(m--){
        int op=read(),x=read(),y=read(),z;
        if(op==0){
            z=read();
            seg.split(y,z,1,n,root[x],root[rcntr++]);
        }
        else if(op==1){
            seg.merge(root[x],root[y]);
        }
        else if(op==2){
            z=read();
            seg.insert(z,y,1,n,root[x]);
        }
        else if(op==3){
            z=read();
            printf("%lld\n",seg.query(y,z,1,n,root[x]));
        }
        else{
            printf("%d\n",seg.searchKth(y-1,root[x]));
        }
    }
    return 0;
}
```

### 平衡树

线段树能干的平衡树都能干；平衡树的一个突出特点是**能够任意变换形态**，这使得平衡树还可以维护**区间反转、区间删除**等操作，这种能力是线段树不具备的。

平衡树普遍**常数较大**；对于这里使用的Treap要特别注意：**merge和split（也就是形态变换）的常数比在树上进行搜索要大得多，所以实现的时候要尽量避免形态变换**。

模板：[洛谷3369 【模板】普通平衡树](https://www.luogu.com.cn/problem/P3369)

```cpp
const int maxn=114514;
class Treap{
    #define LS tree[pos].lson
    #define RS tree[pos].rson
    private:
    struct node{
        int val,rep,siz,key,lson,rson;
    }tree[maxn];
    stack<int> recyc;//用于节点回收的栈
    int siz,root;
    int newNode(int val){
        if(recyc.empty()){//回收节点，需要清空之前的数据
            tree[++siz].val=val;
            tree[siz].rep=tree[siz].siz=1;
            tree[siz].key=rand();
            return siz;
        }
        int pos=recyc.top();
        recyc.pop();
        tree[pos].val=val;
        tree[pos].rep=tree[pos].siz=1;
        tree[pos].key=rand();
        LS=RS=0;
        return pos;
    }
    void upload(int pos){
        tree[pos].siz=tree[LS].siz+tree[pos].rep+tree[RS].siz;
    }
    void split(int pos,int lsiz,int& l,int& r){//treap的分裂操作
        if(!pos){
            l=r=0;
            return;
        }
        if(tree[LS].siz+tree[pos].rep<=lsiz){//此处将相同值合并为了同一节点，不合并时直接将tree[pos].rep替换成1即可
            l=pos;
            split(RS,lsiz-tree[LS].siz-tree[pos].rep,RS,r);
        }
        else{
            r=pos;
            split(LS,lsiz,l,LS);
        }
        upload(pos);
    }
    int merge(int lpos,int rpos){//treap的合并操作
        if(!lpos||!rpos){
            return lpos^rpos;
        }
        if(tree[lpos].key>tree[rpos].key){
            tree[lpos].rson=merge(tree[lpos].rson,rpos);
            upload(lpos);
            return lpos;
        }
        else{
            tree[rpos].lson=merge(lpos,tree[rpos].lson);
            upload(rpos);
            return rpos;
        }
    }
    void display(int pos){//将treap拍扁并打印，用于调试
        if(!pos){
            return;
        }
        display(LS);
        cout<<tree[pos].val<<'*'<<tree[pos].rep<<", ";
        display(RS);
    }
    public:
    int rank(int x){//查询严格小于x的值得数量
        int pos=root,r=0;
        while(pos){
            if(tree[pos].val==x){
                return r+tree[LS].siz;
            }
            if(tree[pos].val<x){
                r+=tree[LS].siz+tree[pos].rep;
                pos=RS;
            }
            if(tree[pos].val>x){
                pos=LS;
            }
        }
        return r;
    }
    int search(int rk){//根据排名找值，rk表示严格小于目标的值的数量
        int pos=root;
        while(pos){
            if(tree[LS].siz<=rk&&tree[LS].siz+tree[pos].rep>rk){
                return tree[pos].val;
            }
            else if(tree[LS].siz+tree[pos].rep<=rk){
                rk-=tree[LS].siz+tree[pos].rep;
                pos=RS;
            }
            else{
                pos=LS;
            }
        }
        return 0;
    }
    void insert(int x){
        int lpart=0,rpart=0;
        split(root,rank(x),lpart,rpart);
        int pos=rpart;
        stack<int> trace;
        while(LS){
            trace.push(pos);
            pos=LS;
        }
        if(tree[pos].val==x){
            tree[pos].rep++;
            tree[pos].siz++;
            while(!trace.empty()){
                upload(trace.top());
                trace.pop();
            }
            root=merge(lpart,rpart);
        }
        else{
            root=merge(lpart,merge(newNode(x),rpart));
        }
    }
    void remove(int x){
        int lpart=0,tar=0,rpart=0;
        split(root,rank(x+1),root,rpart);
        split(root,rank(x),lpart,tar);
        if(tree[tar].rep==1){
            recyc.push(tar);
            root=merge(lpart,rpart);
        }
        else{
            tree[tar].rep--;
            tree[tar].siz--;
            root=merge(lpart,merge(tar,rpart));
        }
    }
    int lowerBound(int x){//x的前驱
        int lpart=0,rpart=0;
        split(root,rank(x),lpart,rpart);
        int pos=lpart,ans;
        while(RS){
            pos=RS;
        }
        ans=tree[pos].val;
        root=merge(lpart,rpart);
        return ans;
    }
    int upperBound(int x){//x的后继
        int lpart=0,rpart=0;
        split(root,rank(x+1),lpart,rpart);
        int pos=rpart,ans;
        while(LS){
            pos=LS;
        }
        ans=tree[pos].val;
        root=merge(lpart,rpart);
        return ans;
    }
    void display(){//将treap拍扁并打印，用于调试
        display(root);
        cout<<endl;
    }
}treap;

int main(){
    int n=read();
    while(n--){
        int op=read(),x=read();
        if(op==1){
            treap.insert(x);
        }
        else if(op==2){
            treap.remove(x);
        }
        else if(op==3){
            printf("%d\n",treap.rank(x)+1);
        }
        else if(op==4){
            printf("%d\n",treap.search(x-1));
        }
        else if(op==5){
            printf("%d\n",treap.lowerBound(x));
        }
        else{
            printf("%d\n",treap.upperBound(x));
        }
    }
    return 0;
}
```

#### 可持久化平衡树

由于平衡树能任意变换形态，可持久化平衡树能搞**区间复制**（理论上块状链表应该也可以搞，但是复杂度劣于平衡树而且我没写过）。

**注意能进行可持久化的平衡树仅Treap一家。**

区间复制操作模板：[HDU 7152 - Copy](https://vjudge.net/problem/HDU-7152)

```cpp
const int maxn=114514;
int n,root[maxn];

class Treap{
	#define LS tree[pos].lson
	#define RS tree[pos].rson
	private:
	struct Node{
		int key,siz,val,lson,rson;
	}tree[12000000];
	int siz;//注意，为了方便使用，可持久化平衡树的根节点存储在结构体外
	vector<int> cache;//重构时对结构进行暂存的数组
	int newNode(int val){
		tree[++siz].key=rand();
		tree[siz].siz=1;
		tree[siz].val=val;
		tree[siz].lson=tree[siz].rson=0;
		return siz;
	}
	int copyNode(int pos){
		tree[++siz]=tree[pos];
		return siz;
	}
	void upload(int pos){
		if(pos){
			tree[pos].siz=tree[LS].siz+1+tree[RS].siz;
		}
    }
    void split(int pos,int lsiz,int& l,int& r){
        if(!pos){
            l=r=0;
            return;
        }
        if(tree[LS].siz+1<=lsiz){
            l=copyNode(pos);
            split(RS,lsiz-tree[LS].siz-1,tree[l].rson,r);
        }
        else{
            r=copyNode(pos);
            split(LS,lsiz,l,tree[r].lson);
        }
        upload(l);
        upload(r);
    }
    int merge(int lpos,int rpos){
        if(!lpos||!rpos){
            return lpos^rpos;
        }
        if(tree[lpos].key>tree[rpos].key){
            lpos=copyNode(lpos);
            tree[lpos].rson=merge(tree[lpos].rson,rpos);
            upload(lpos);
            return lpos;
        }
        else{
            rpos=copyNode(rpos);
            tree[rpos].lson=merge(lpos,tree[rpos].lson);
            upload(rpos);
            return rpos;
        }
    }
    void getRebuildCache(int pos){
        if(!pos||cache.size()>=n){
            return;
		}
		getRebuildCache(LS);
		cache.push_back(tree[pos].val);
		getRebuildCache(RS);
	}
	public:
	void append(int& root,int val){
		root=merge(root,newNode(val));
	}
	void copy(int& root,int al,int ar){
		int lres=0,tar=0,rres=0;
		split(root,ar,tar,rres);
		split(tar,al-1,lres,tar);
		root=merge(lres,merge(tar,merge(tar,rres)));
		split(root,n,root,rres);
	}
	int query(int root,int idx){//本题的查询操作是复杂度大头，尽管搜索写起来更费事，但比简洁的几行形态变换要来得更快
		int pos=root;
		while(pos){
			if(tree[LS].siz+1==idx){
				return tree[pos].val;
			}
			if(tree[LS].siz+1<idx){
				idx-=tree[LS].siz+1;
				pos=RS;
			}
			else{
				pos=LS;
			}
		}
		return 0;
	}
	void clear(){
		siz=0;
	}
	int size(){
		return siz;
	}
	void rebuild(int& root){
		getRebuildCache(root);
		clear();
		root=0;
		for(U i=0;i<cache.size();i++){
			root=merge(root,newNode(cache[i]));
		}
		cache.clear();
	}
    /*可持久化平衡树一般都具有非常巨大的空间消耗，对于本题这种并不涉及历史版本查询的“可持久化”，可以采用定期重构的方法解决
    打删除标记、进行节点回收的方法其实更难写，对细节处理要求更高*/
}treap;

int main(){
	int T=read();
	while(T--){
		n=read();
		int q=read();
		for(int i=1;i<=n;i++){
			int x=read();
			treap.append(root[0],x);
		}
		int ans=0;
		for(int i=1;i<=q;i++){
			root[i]=root[i-1];
			int op=read();
			if(op==1){
				int l=read(),r=read();
				treap.copy(root[i],l,r);
			}
			else if(op==2){
				int x=read();
				ans^=treap.query(root[i],x);
			}
			if(treap.size()>=10000000){//平衡树大小超过1e7就重构
				treap.rebuild(root[i]);
			}
		}
		printf("%d\n",ans);
		treap.clear();
		root[0]=0;
	}
	return 0;
}
```

### k-D树（待补充）



### 树套树（待补充）

一个比较反直觉的特点：树套树的应用面很窄，如果有区间操作的话基本就是告别树套树了，因为**在外层树进行区间信息的上下传基本上都是灾难**。



### 分块/块状链表（待补充）



### 莫队（待补充）



### 左偏树（待补充）

左偏树目前没见过啥花活，就仅仅是可以合并的堆而已，现在似乎已经绝迹了（悲）



## 树形结构

### Link-Cut-Tree

处理树上路径问题的利器，尤其是树是**动态连边**的时候，基本只能用LCT来做。

模板：[洛谷3690 【模板】动态树（Link Cut Tree）](https://www.luogu.com.cn/problem/P3690)

```cpp
const int maxn=3*114514;
char buf[100005],*p1=buf,*p2=buf;
inline char nc(){
    if(p1==p2){
        p1=buf;
        p2=p1+fread(buf,1,100000,stdin);
        if(p1==p2){
            return EOF;
        }
    }
    return *p1++;
}

class LCT{
    #define LS tree[pos].son[0]
    #define RS tree[pos].son[1]
    #define REVS tree[pos].son[which^1]
    #define FA tree[pos].fa
    #define isrson(pos) (tree[tree[pos].fa].son[1]==pos)
    #define notroot(pos) (tree[tree[pos].fa].son[0]==pos||isrson(pos))

    private:
    bool flip[maxn];
    INL void upload(REG int pos){
        tree[pos].sum=tree[LS].sum^tree[RS].sum^tree[pos].val;//注意这个模板里求的是异或和！！使用时要根据题目要求修改上推函数，注意使用平衡树的逻辑
    }
    INL void flipside(REG int pos){
        swap(LS,RS);
        flip[pos]^=1;
    }
    INL void download(REG int pos){
        if(flip[pos]){
            flipside(LS);
            flipside(RS);
            flip[pos]=0;
        }
    }

    public:
    struct lctnode{
        int son[2],fa,val,sum;
    }tree[maxn];
    INL void rotate(REG int pos){//splay特有的上旋
        REG bool which=isrson(pos);
        REG int f=FA,g=tree[f].fa,r=REVS;
        if(notroot(f)){
            tree[g].son[isrson(f)]=pos;
        }
        REVS=f;
        tree[f].son[which]=r;
        if(r!=0){
            tree[r].fa=f;
        }
        tree[f].fa=pos;
        FA=g;
        upload(f);
    }
    int stac[maxn];
    INL void splay(REG int pos){
        REG int npos=pos,frt=0;
        stac[frt++]=npos;
        while(notroot(npos)){
            stac[frt++]=npos=tree[npos].fa;
        }
        while(frt>0){
            download(stac[--frt]);
        }
        for(REG int f=FA;notroot(pos);rotate(pos),f=FA){
            if(notroot(f)){
                rotate((isrson(pos)==isrson(f))?f:pos);
            }
        }
        upload(pos);
    }
    INL void access(REG int pos){//将pos到根的路径打通成一条链
        for(REG int f=0;pos!=0;f=pos,pos=FA){
            splay(pos);
            RS=f;
            upload(pos);
        }
    }
    INL void makeroot(REG int pos){//将pos换成整棵树的根，一般认为LCT的大常数就是这个操作害的
        access(pos);
        splay(pos);
        flipside(pos);
    }
    INL int findroot(REG int pos){//找到这个节点所在的树根，用来判断连通性
        access(pos);
        splay(pos);
        while(LS!=0){
            download(pos);
            pos=LS;
        }
        splay(pos);
        return pos;
    }
    INL void extract(REG int x,REG int y){//将x到y路径上的信息都提取出来，这里没有检验连通性
        makeroot(x);
        access(y);
        splay(y);//此时x->y的信息全部储存在tree[y]上
    }
    INL void link(REG int x,REG int y){//在x和y之间连边，此处检验了连边的合法性
        makeroot(x);
        if(findroot(y)!=x){
            tree[x].fa=y;
        }
    }
    INL void cut(REG int x,REG int y){//切断x和y之间的边，此处检验了切边的合法性
        makeroot(x);
        if(findroot(y)==x&&tree[y].fa==x&&tree[y].son[0]==0){
            tree[y].fa=tree[x].son[1]=0;
            upload(x);
        }
    }
    #undef LS
    #undef RS
    #undef REVS
    #undef FA
    #undef isrson
    #undef notroot
}lct;

int opr0(int x,int y){
    lct.extract(x,y);
    return lct.tree[y].sum;
}

void opr3(int x,int y){
    lct.splay(x);
    lct.tree[x].val=y;
}

int n,m;
int main(){
    n=read();
    m=read();
    for(int i=1;i<=n;i++){
        lct.tree[i].val=read();//作为数据结构，LCT有着很反直觉的“数组下标即对应点号”的特点
    }
    for(int i=0;i<m;i++){
        int op=read(),x=read(),y=read();
        if(op==0){
            printf("%d\n",opr0(x,y));
        }
        else if(op==1){
            lct.link(x,y);
        }
        else if(op==2){
            lct.cut(x,y);
        }
        else{
            opr3(x,y);
        }
    }
    return 0;
}
```



### 重链剖分

重链剖分能且仅能处理静态树问题，而且某些静态树问题（链染色）的时空和代码复杂度都高于LCT；它存在的意义是常数较小（懂不懂两个log跑得比一个log还快的含金量啊？）且原理比较简单，实现起来比较好看。此外，它还是**树上启发式合并**的理论基础。

模板：[洛谷3384 【模板】轻重链剖分/树链剖分](https://www.luogu.com.cn/problem/P3384)

```cpp
const int maxn=114514;
char buf[100005],*p1=buf,*p2=buf;
inline char nc(){
    if(p1==p2){
        p1=buf;
        p2=p1+fread(buf,1,100000,stdin);
        if(p1==p2){
            return EOF;
        }
    }
    return *p1++;
}

ll n,root,mod,vals[maxn];
class SegTree{
    private:
    ll tree[maxn<<2],tag[maxn<<2];
    void upload(int pos){
        tree[pos]=tree[pos<<1]+tree[pos<<1|1];
    }
    void download(int l,int r,int pos){
        if(tag[pos]){
            tree[pos<<1]=(tree[pos<<1]+tag[pos]*(M-l+1))%mod;
            tag[pos<<1]=(tag[pos<<1]+tag[pos])%mod;
            tree[pos<<1|1]=(tree[pos<<1|1]+tag[pos]*(r-M))%mod;
            tag[pos<<1|1]=(tag[pos<<1|1]+tag[pos])%mod;
            tag[pos]=0;
        }
    }
    public:
    void update(int al,int ar,ll d,int l,int r,int pos){
        if(al<=l&&ar>=r){
            tree[pos]=(tree[pos]+d*(r-l+1))%mod;
            tag[pos]=(tag[pos]+d)%mod;
            return;
        }
        download(l,r,pos);
        if(al<=M){
            update(al,ar,d,l,M,pos<<1);
        }
        if(ar>M){
            update(al,ar,d,M+1,r,pos<<1|1);
        }
        upload(pos);
    }
    ll query(int al,int ar,int l,int r,int pos){
        if(al<=l&&ar>=r){
            return tree[pos];
        }
        download(l,r,pos);
        ll ans=0;
        if(al<=M){
            ans=query(al,ar,l,M,pos<<1);
        }
        if(ar>M){
            ans=(ans+query(al,ar,M+1,r,pos<<1|1))%mod;
        }
        return ans;
    }
}seg;//区间修改、区间查询的线段树板子

vector<int> graf[maxn];
int siz[maxn],dep[maxn],fa[maxn],cei[maxn],dfn[maxn],cntr=1;
int dfs1(int pos,int ndep,int nf){//第一趟dfs，处理深度、父节点、子树大小和重儿子。在这个板子中，graf[pos][0]就是pos的重儿子（如果不是父节点的话）
    dep[pos]=ndep;
    fa[pos]=nf;
    siz[pos]=1;
    int maxi=-1;
    for(U int i=0;i<graf[pos].size();i++){
        int tar=graf[pos][i];
        if(tar==nf){
            continue;
        }
        siz[pos]+=dfs1(tar,ndep+1,pos);
        if(maxi==-1||siz[tar]>siz[graf[pos][maxi]]){
            maxi=i;
        }
    }
    if(maxi!=-1&&maxi){
        swap(graf[pos][0],graf[pos][maxi]);
    }
    return siz[pos];
}
void dfs2(int pos,int ncei){//第二趟dfs，处理欧拉序和链顶
    cei[pos]=ncei;
    dfn[pos]=cntr++;
    if(graf[pos][0]==fa[pos]){
        return;
    }
    dfs2(graf[pos][0],ncei);
    for(U int i=1;i<graf[pos].size();i++){
        int tar=graf[pos][i];
        if(tar==fa[pos]){
            continue;
        }
        dfs2(tar,tar);
    }
}
//以上是重链剖分的两个核心dfs

void opr1(int x,int y,ll z){//对链的修改操作
    while(cei[x]!=cei[y]){
        if(dep[cei[x]]<dep[cei[y]]){//当两个端点不在同一个重链时，优先跳链顶更深的那个，防止跳过LCA
            swap(x,y);
        }
        seg.update(dfn[cei[x]],dfn[x],z,1,n,1);//注意，线段树中的下标对应的都是欧拉序，不是点号
        x=fa[cei[x]];
    }
    if(dep[x]<dep[y]){
        swap(x,y);
    }
    seg.update(dfn[y],dfn[x],z,1,n,1);
}

ll opr2(int x,int y){//对链的查询操作
    ll ans=0;
    while(cei[x]!=cei[y]){//跳法同上
        if(dep[cei[x]]<dep[cei[y]]){
            swap(x,y);
        }
        ans=(ans+seg.query(dfn[cei[x]],dfn[x],1,n,1))%mod;
        x=fa[cei[x]];
    }
    if(dep[x]<dep[y]){
        swap(x,y);
    }
    return (ans+seg.query(dfn[y],dfn[x],1,n,1))%mod;
}

void opr3(int x,ll z){//对子树的修改操作，利用了同一个子树内欧拉序连续的性质
    seg.update(dfn[x],dfn[x]+siz[x]-1,z,1,n,1);
}

ll opr4(int x){//对子树的查询操作
    return seg.query(dfn[x],dfn[x]+siz[x]-1,1,n,1);
}

int main(){
    n=read();
    int m=read();
    root=read();
    mod=read();
    for(int i=1;i<=n;i++){
        vals[i]=read()%mod;
    }
    for(int i=1;i<n;i++){
        int x=read(),y=read();
        graf[x].push_back(y);
        graf[y].push_back(x);
    }
    dfs1(root,1,0);
    dfs2(root,root);
    for(int i=1;i<=n;i++){
        seg.update(dfn[i],dfn[i],vals[i],1,n,1);
    }
    while(m--){
        int op=read();
        if(op==1){
            int x=read(),y=read();
            ll z=read();
            opr1(x,y,z);
        }
        else if(op==2){
            int x=read(),y=read();
            printf("%lld\n",opr2(x,y));
        }
        else if(op==3){
            int x=read();
            ll z=read();
            opr3(x,z);
        }
        else{
            int x=read();
            printf("%lld\n",opr4(x));
        }
    }
    return 0;
}
```



### 长链剖分

目前可以认为长链剖分的作用有且仅有$O(1)$求k级祖先。

模板：[洛谷5903 【模板】树上 k 级祖先](https://www.luogu.com.cn/problem/P5903)

```cpp
const int maxn=114514*5;
vector<int> graf[maxn],anc[maxn],offs[maxn];
int maxd[maxn],dep[maxn],multi[maxn][32],cei[maxn],dfn[maxn],hibit[maxn],lg2[maxn],cntr=1;
int n;
ull q,s;
INL U get(U x){
    x^=x<<13;
    x^=x>>17;
    x^=x<<5;
    return s=x;
}
void getlog(){
    int ans=0;
    for(int i=1;i<=n;i++){
        if((1<<ans)==i){
            ans++;
        }
        lg2[i]=ans-1;
    }
}
int dfs1(int pos,int ndep,int nf){
    maxd[pos]=dep[pos]=ndep;
    multi[pos][0]=nf;
    for(U i=0;multi[pos][i];i++){
        multi[pos][i+1]=multi[multi[pos][i]][i];
    }
    int maxi=-1;
    for(U int i=0;i<graf[pos].size();i++){
        int tar=graf[pos][i];
        if(tar==nf){
            continue;
        }
        maxd[pos]=max(maxd[pos],dfs1(tar,ndep+1,pos));
        if(maxi==-1||maxd[tar]>maxd[graf[pos][maxi]]){
            maxi=i;
        }
    }
    if(maxi!=-1&&maxi){
        swap(graf[pos][0],graf[pos][maxi]);
    }
    return maxd[pos];
}
void dfs2(int pos,int ncei){
    cei[pos]=ncei;
    dfn[pos]=cntr++;
    if(graf[pos][0]==multi[pos][0]){
        return;
    }
    dfs2(graf[pos][0],ncei);
    for(U int i=1;i<graf[pos].size();i++){
        int tar=graf[pos][i];
        if(tar==multi[pos][0]){
            continue;
        }
        dfs2(tar,tar);
    }
}
ull kthanc(int pos,int k){//查询k级祖先
    if(!k){
        return pos;
    }
    int pas=cei[multi[pos][lg2[k]]];
    if(dep[pos]-dep[pas]==k){
        return pas;
    }
    else if(dep[pos]-dep[pas]<k){
        return anc[pas][k-(dep[pos]-dep[pas])-1];
    }
    else{
        return offs[pas][(dep[pos]-dep[pas])-k-1];
    }
}
int main(){
    n=read();
    q=read();
    s=read();
    getlog();
    int root=0;
    for(int i=1;i<=n;i++){
        int f=read();
        if(f){
            graf[f].push_back(i);
            graf[i].push_back(f);
        }
        else{
            root=i;
        }
    }
    dfs1(root,1,0);
    dfs2(root,root);
    
    for(int i=1;i<=n;i++){
        if(i==cei[i]){
            int pos=i;
            while(graf[pos][0]!=multi[pos][0]){
                pos=graf[pos][0];
                offs[i].push_back(pos);
            }
            pos=multi[i][0];
            for(U j=0;j<offs[i].size()&&pos;j++){
                anc[i].push_back(pos);
                pos=multi[pos][0];
            }
            anc[i].push_back(pos);
        }
    }//千万不要忘了这一段也是模板的一部分！！！！！
    
    ull ans=0,rslt=0,x,k;
    for(ull i=1;i<=q;i++){
        x=((get(s)^ans)%n)+1;
        k=(get(s)^ans)%dep[x];
        rslt^=i*(ans=kthanc(x,k));
    }
    cout<<rslt;
    return 0;
}
```




## 点分治

点分治通常与树上路径问题强相关，如某种性质的路径是否存在、某种路径的数量、**所有**路径的和等问题。

模板：[洛谷3806 【模板】点分治1](https://www.luogu.com.cn/problem/P3806)

```cpp
const int maxn=11451;
char buf[100005],*p1=buf,*p2=buf;
inline char nc(){
    if(p1==p2){
        p1=buf;
        p2=p1+fread(buf,1,100000,stdin);
        if(p1==p2){
            return EOF;
        }
    }
    return *p1++;
}

struct Edge{
    int f,t,len;
    Edge(int _x=0,int _y=0,int _len=0){
        f=_x;t=_y;len=_len;
    }
};
vector<Edge> eds;
vector<int> graf[maxn];
int n,k,siz[maxn],subsiz;
stack<int> buckModify;
bool vis[maxn],ans,buck[11451400];//注意！此处的vis数组是记录“有没有算过以点p为根的子树的答案”
int foc,maxs;//注意，foc是当前重心，是全 局 变 量，这是个坑

void updateBuck(int idx){//更新桶内信息
    if(!buck[idx]){
        buck[idx]=1;
        buckModify.push(idx);
    }
}

void clearBuck(){//根据栈的记录滚掉所有桶内信息
    while(!buckModify.empty()){
        buck[buckModify.top()]=0;
        buckModify.pop();
    }
}

void getFocus(int pos,int fa){//得到当前子树的重心
    int thisMaxs=0;
    siz[pos]=1;
    for(U i=0;i<graf[pos].size();i++){
        Edge e=eds[graf[pos][i]];
        if(vis[e.t]||e.t==fa){
            continue;
        }
        getFocus(e.t,pos);
        siz[pos]+=siz[e.t];
        thisMaxs=max(thisMaxs,siz[e.t]);
    }
    if(max(thisMaxs,subsiz-thisMaxs-1)<maxs){
        maxs=max(thisMaxs,subsiz-thisMaxs-1);
        foc=pos;
    }
}

void dfsAns(int pos,int fa,int dep){//“从这个子树出发”来更新答案
    if(dep>k){
        return;
    }
    ans|=buck[k-dep];
    if(ans){
        return;
    }
    for(U i=0;(!ans)&&i<graf[pos].size();i++){
        Edge e=eds[graf[pos][i]];
        if(vis[e.t]||e.t==fa){
            continue;
        }
        dfsAns(e.t,pos,dep+e.len);
    }
}

void mergeInfo(int pos,int fa,int dep){//将这个子树的信息合并入桶，不更新答案
    if(dep>k){
        return;
    }
    updateBuck(dep);
    for(U i=0;i<graf[pos].size();i++){
        Edge e=eds[graf[pos][i]];
        if(vis[e.t]||e.t==fa){
            continue;
        }
        mergeInfo(e.t,pos,dep+e.len);
    }
}

void solve(int pos){//得到以从pos出发或经过pos的答案（从pos出发就是buck[0]=1）
    buck[0]=1;
    for(U i=0;(!ans)&&i<graf[pos].size();i++){
        Edge e=eds[graf[pos][i]];
        if(vis[e.t]){
            continue;
        }
        dfsAns(e.t,pos,e.len);
        mergeInfo(e.t,pos,e.len);
    }
}

void dvd(int pos){//点分治主体
    maxs=1145141919;//初始化maxs为极大值，保证foc更新
    getFocus(pos,0);
    vis[foc]=1;//得到重心并标记
    solve(foc);
    if(ans){//这道题是询问存在性，所以得到答案后可以直接终止点分治过程
        return;
    }
    clearBuck();//foc为根的子树答案已经计算完成了，清空桶
    int currentSiz=subsiz,thisFoc=foc;//注意！这俩数据接下来的分治过程都会用到，但都是全局变量，会受影响，所以要记录下来
    for(U i=0;(!ans)&&i<graf[thisFoc].size();i++){
        Edge e=eds[graf[thisFoc][i]];
        if(vis[e.t]){
            continue;
        }
        if(siz[e.t]<siz[thisFoc]){
            subsiz=siz[e.t];
        }
        else{
            subsiz=currentSiz-siz[thisFoc];
        }
        dvd(e.t);
    }
}

int main(){
    subsiz=n=read();
    int m=read();
    for(int i=1;i<n;i++){
        int x=read(),y=read(),len=read();
        eds.push_back(Edge(x,y,len));
        graf[x].push_back(eds.size()-1);
        eds.push_back(Edge(y,x,len));
        graf[y].push_back(eds.size()-1);
    }
    while(m--){
        k=read();
        subsiz=n;
        dvd(1);
        printf(ans?"AYE\n":"NAY\n");
        ans=0;
        clearBuck();
        for(int i=1;i<=n;i++){
            vis[i]=0;
        }
    }
    return 0;
}
```



## 树上启发式合并

树上启发式合并的问题通常都和点分治很像，甚至某些题本身就也可以用点分治来写；二者的不同在于树上启发式合并处理的是**有根树**，处理时保留了原来的祖先后代关系；点分治处理的是**无根树**，处理时会**抛弃**祖先后代关系。

树上启发式合并问题的一般特点：处理路径或**子树**相关问题；处理某个点的答案时用到的信息难以**显式地转移给父节点**。

比如，计算子树的**权值和**，子节点的权值和可以直接加到父节点上，就是可以显式地转移给父节点；而计算子树的**颜色数**，我们需要给颜色开桶，那么子节点的桶如果暴力转移到父节点的话单次转移的复杂度就是$O(n)$，时空复杂度都不对，就不能显式地转移给父节点。

模板（万  恶  之  源）：[Codeforces 741D](https://codeforces.com/problemset/problem/741/D)

```cpp
const int maxn=114514*5;
int n,val[maxn];
vector<int> graf[maxn];
int siz[maxn],dep[maxn],fa[maxn],cei[maxn],dfn[maxn],cntr=1;
int buck[1<<22],ans[maxn],nowAns;
stack<int> buckModify;
int dfs1(int pos,int ndep,int nf){
    dep[pos]=ndep;
    fa[pos]=nf; 
    siz[pos]=1;
    val[pos]^=val[nf];
    int maxi=-1;
    for(U int i=0;i<graf[pos].size();i++){
        int tar=graf[pos][i];
        if(tar==nf){
            continue;
        }
        siz[pos]+=dfs1(tar,ndep+1,pos);
        if(maxi==-1||siz[tar]>siz[graf[pos][maxi]]){
            maxi=i;
        }
    }
    if(maxi!=-1&&maxi){
        swap(graf[pos][0],graf[pos][maxi]);
    }
    return siz[pos];
}
void dfs2(int pos,int ncei){
    cei[pos]=ncei;
    dfn[pos]=cntr++;
    if(graf[pos].empty()||graf[pos][0]==fa[pos]){
        return;
    }
    dfs2(graf[pos][0],ncei);
    for(U int i=1;i<graf[pos].size();i++){
        int tar=graf[pos][i];
        if(tar==fa[pos]){
            continue;
        }
        dfs2(tar,tar);
    }
}//以上是重链剖分板子

void updateBuck(int idx,int val){//更新桶内信息
    buckModify.push(idx);
    buck[idx]=max(buck[idx],val);
}

void dfsAns(int pos,int rootDep){//“从这个子树出发”来更新答案
    nowAns=max(nowAns,dep[pos]+buck[val[pos]]-2*rootDep);
    for(int i=0;i<22;i++){
        nowAns=max(nowAns,dep[pos]+buck[val[pos]^(1<<i)]-2*rootDep);
    }
    for(U i=0;i<graf[pos].size();i++){
        if(graf[pos][i]==fa[pos]){
            continue;
        }
        dfsAns(graf[pos][i],rootDep);
    }
}

void mergeInfo(int pos){//将这个子树的信息合并入桶，不更新答案
    updateBuck(val[pos],dep[pos]);
    for(U i=0;i<graf[pos].size();i++){
        if(graf[pos][i]==fa[pos]){
            continue;
        }
        mergeInfo(graf[pos][i]);
    }
}

void removeInfo(){//根据栈的记录滚掉所有桶内信息
    while(!buckModify.empty()){
        buck[buckModify.top()]=-1145141919;
        buckModify.pop();
    }
}

void dfs(int pos){//树上启发式合并主体
    for(U i=1;i<graf[pos].size();i++){//先遍历一遍轻儿子得到所有轻儿子的答案
        if(graf[pos][i]==fa[pos]){
            continue;
        }
        dfs(graf[pos][i]);
        removeInfo();//每算完一个轻儿子要清空桶
    }

    if(!graf[pos].empty()/*这个主要是防一手n=1，如果保证n>=2的话可以去掉*/&&graf[pos][0]!=fa[pos]){//计算重儿子的答案
        dfs(graf[pos][0]);
    }

    for(U i=1;i<graf[pos].size();i++){//再遍历一遍轻儿子，计算pos的答案
        if(graf[pos][i]==fa[pos]){
            continue;
        }
        dfsAns(graf[pos][i],dep[pos]);
        mergeInfo(graf[pos][i]);
    }//注意！这里只得到了“跨越pos的所有路径”的答案，而没有得到“从pos出发的路径”的答案

    nowAns=max(nowAns,buck[val[pos]]-dep[pos]);
    for(int i=0;i<22;i++){
        nowAns=max(nowAns,buck[val[pos]^(1<<i)]-dep[pos]);
    }//此处计算“从pos出发的路径”的答案
    ans[pos]=nowAns;
    updateBuck(val[pos],dep[pos]);//将pos的信息并入桶
    nowAns=0;//将记录当前答案的变量清零
}

void getAns(int pos){
    for(U i=0;i<graf[pos].size();i++){
        if(graf[pos][i]==fa[pos]){
            continue;
        }
        getAns(graf[pos][i]);
        ans[pos]=max(ans[pos],ans[graf[pos][i]]);
    }
}
/*注意，我们刚刚对每个节点计算的都是“过这个节点的路径”的答案，而不是“以这个节点为根的子树”内的所有答案；
所以我们还要把每个节点的答案与子树内的所有节点取max，一遍dfs即可*/

int main(){
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    for(int i=0;i<(1<<22);i++){
        buck[i]=-1145141919;
    }
    cin>>n;
    for(int i=2;i<=n;i++){
        int f;
        string v;
        cin>>f>>v;
        graf[f].push_back(i);
        graf[i].push_back(f);
        val[i]=(1<<(v[0]-'a'));
    }
    dfs1(1,1,0);
    dfs2(1,1);
    dfs(1);
    getAns(1);
    for(int i=1;i<=n;i++){
        printf("%d ",ans[i]);
    }
    return 0;
}
```



## 虚树

某些树上dp题会给你多次询问，每次询问都仅与一个给定点集有关，此时我们显然不能对于每个询问都扫一遍整棵树，于是虚树应运而生。

虚树就是一颗仅包含给定点集中的点以及这些点的LCA的树，在这棵树上dp的时间复杂度就是$O(点集大小)$而不是$O(n)$了。

虚树一般和重链剖分配合食用。

虚树题通常都很模板化，现在似乎已经绝迹了（悲）

模板：[洛谷2495 \[SDOI2011] 消耗战](https://www.luogu.com.cn/problem/P2495)

```cpp
const int maxn=5*114514;
struct edge{
	int f,t;
	ll len;
	INL edge(){
		f=t=len=-1;
	}
	INL edge(REG int _f,REG int _t,REG ll _len){
		f=_f;t=_t;len=_len;
	}
};
vector<edge> eds;
vector<int> graf[maxn];
int n,m;
int siz[maxn],fa[maxn],dep[maxn],cei[maxn],dfn[maxn];
ll shrt[maxn];
namespace cut{
	int cntr=1;
	int dfs1(int pos,int f,int ndep,ll nshrt){
		siz[pos]=1;
		fa[pos]=f;
		dep[pos]=ndep;
		shrt[pos]=nshrt;
		REG int maxi=-1;
		for(REG unsigned i=0;i<graf[pos].size();i++){
			REG edge e=eds[graf[pos][i]];
			if(e.t==f){
				continue;
			}
			siz[pos]+=dfs1(e.t,pos,ndep+1,min<ll>(e.len,nshrt));
			if(maxi==-1||siz[e.t]>siz[eds[graf[pos][maxi]].t]){
				maxi=i;
			}
		}
		if(maxi!=-1&&maxi){
			swap<int>(graf[pos][0],graf[pos][maxi]);
		}
		return siz[pos];
	}
	void dfs2(int pos,int ncei){
		cei[pos]=ncei;
		dfn[pos]=cntr++;
		if(eds[graf[pos][0]].t==fa[pos]){
			return;
		}
		dfs2(eds[graf[pos][0]].t,ncei);
		for(REG unsigned i=1;i<graf[pos].size();i++){
			REG edge e=eds[graf[pos][i]];
			if(e.t==fa[pos]){
				continue;
			}
			dfs2(e.t,e.t);
		}
	}
	INL int lca(REG int x,REG int y){
		while(cei[x]!=cei[y]){
			if(dep[cei[x]]>dep[cei[y]]){
				swap<int>(x,y);
			}
			y=fa[cei[y]];
		}
		return dep[x]>dep[y]?y:x;
	}
}//以上是重链剖分求LCA
struct cldakicpc{
	int pos,id;
	INL cldakicpc(){
		pos=id=-1;
	}
	INL cldakicpc(REG int _pos,REG int _id){
		pos=_pos;id=_id;
	}
}lst[maxn];
INL bool cmp(cldakicpc p1,cldakicpc p2){
	return p1.id<p2.id;
}
int stak[maxn],frt;
bool flags[maxn];
ll dp[maxn];
vector<int> ftree[maxn];
INL void insert(REG int x){//向虚树中插入节点
    if(!frt){
        stak[++frt]=x;
        return;
    }
    REG int anc=cut::lca(stak[frt],x);
    while(dep[anc]<dep[stak[frt-1]]){
        ftree[stak[frt-1]].push_back(stak[frt]);
        frt--;
    }
    if(dep[anc]<dep[stak[frt]]){
        ftree[anc].push_back(stak[frt--]);
    }
    if(stak[frt]!=anc){
        stak[++frt]=anc;
    }
    stak[++frt]=x;
}
void dfs3(int pos){//边dp边删除虚树
	dp[pos]=shrt[pos];
	if(flags[pos]){
		flags[pos]=0;
        for(REG unsigned i=0;i<ftree[pos].size();i++){
			dfs3(ftree[pos][i]);
		}
	}
	else{
		REG ll tmp=0;
		for(REG unsigned i=0;i<ftree[pos].size();i++){
			REG int tar=ftree[pos][i];
			dfs3(tar);
			tmp+=dp[tar];
		}
		dp[pos]=min<ll>(dp[pos],tmp);
	}
	ftree[pos].clear();
}
int main(){
	n=read();
	for(REG int i=0;i<n-1;i++){
		REG int x=read(),y=read();
		REG ll l=read();
		eds.push_back(edge(x,y,l));
		eds.push_back(edge(y,x,l));
		graf[x].push_back(i<<1);
		graf[y].push_back(i<<1|1);
	}
	cut::dfs1(1,0,0,0x7FFFFFFFFFFFFFFF);
	cut::dfs2(1,1);
	m=read();
	for(REG int i=0;i<m;i++){
		REG int p=read();
		for(REG int j=0;j<p;j++){
			lst[j].pos=read();
			lst[j].id=dfn[lst[j].pos];
			flags[lst[j].pos]=1;
		}
		sort(lst,lst+p,cmp);
		if(lst->pos!=1){
			insert(1);
		}
		for(REG int j=0;j<p;j++){
			insert(lst[j].pos);
		}
		while(frt>1){
			ftree[stak[frt-1]].push_back(stak[frt]);
			frt--;
		}
		frt=0;//构造虚树
		dfs3(1);
		printf("%lld\n",dp[1]);
	}
	return 0;
}
```

模板：[洛谷3233 \[HNOI2014]世界树](https://www.luogu.com.cn/problem/P3233)

这道题比上一道麻烦很多，重链剖分的作用也不仅限于求LCA。

```cpp
#define INL inline
#define REG register
#include <algorithm>
#include <cstdio>
#include <iostream>
#include <vector>
using namespace std;
INL int read(){
    REG int sum=0,sign=1;
    REG char tmp=getchar();
    while(tmp<'0'||tmp>'9'){
        if(tmp=='-'){
            sign=-1;
        }
        tmp=getchar();
    }
    while(tmp>='0'&&tmp<='9'){
        sum=(sum<<1)+(sum<<3)+(tmp-'0');
        tmp=getchar();
    }
    return sum*sign;
}
const int maxn=3*114514;
int n,m;
vector<int> graf[maxn];
int siz[maxn],fa[maxn],dep[maxn],cei[maxn],dfn[maxn],pid[maxn];
namespace cut{
    int cntr=1;
    int dfs1(int pos,int f,int ndep){
        siz[pos]=1;
        fa[pos]=f;
        dep[pos]=ndep;
        REG int maxi=-1;
        for(REG unsigned i=0;i<graf[pos].size();i++){
            REG int tar=graf[pos][i];
            if(tar==f){
                continue;
            }
            siz[pos]+=dfs1(tar,pos,ndep+1);
            if(maxi==-1||siz[tar]>siz[graf[pos][maxi]]){
                maxi=i;
            }
        }
        if(maxi!=-1&&maxi){
            swap<int>(graf[pos][0],graf[pos][maxi]);
        }
        return siz[pos];
    }
    void dfs2(int pos,int ncei){
        cei[pos]=ncei;
        dfn[pos]=cntr;
        pid[cntr++]=pos;
        if(graf[pos][0]==fa[pos]){
            return;
        }
        dfs2(graf[pos][0],ncei);
        for(REG unsigned i=1;i<graf[pos].size();i++){
            REG int tar=graf[pos][i];
            if(tar==fa[pos]){
                continue;
            }
            dfs2(tar,tar);
        }
    }
    INL int lca(REG int x,REG int y){
        while(cei[x]!=cei[y]){
            if(dep[cei[x]]>dep[cei[y]]){
                swap<int>(x,y);
            }
            y=fa[cei[y]];
        }
        return dep[x]>dep[y]?y:x;
    }
    INL int upper(REG int pos,REG int d){//求d级祖先
        while(dep[cei[pos]]>dep[pos]-d){
            d-=dep[pos]-dep[cei[pos]]+1;
            pos=fa[cei[pos]];
        }
        return pid[dfn[pos]-d];
    }
    INL int dist(REG int x,REG int y){
        REG int anc=lca(x,y);
        return dep[x]+dep[y]-(dep[anc]<<1);
    }
}
bool flags[maxn];
struct cldakicpc{
    int pos,key;
}lst[maxn];
INL bool cmp(REG cldakicpc p1,REG cldakicpc p2){
    return p1.key<p2.key;
}
int backup[maxn],stak[maxn],frt;
vector<int> ftree[maxn];
INL void insert(REG int pos){
    if(!frt){
        stak[++frt]=pos;
        return;
    }
    REG int anc=cut::lca(stak[frt],pos);
    while(dep[anc]<dep[stak[frt-1]]){
        ftree[stak[frt-1]].push_back(stak[frt]);
        ftree[stak[frt]].push_back(stak[frt-1]);
        frt--;
    }
    if(dep[anc]<dep[stak[frt]]){
        ftree[anc].push_back(stak[frt]);
        ftree[stak[frt]].push_back(anc);
        frt--;
    }
    if(anc!=stak[frt]){
        stak[++frt]=anc;
    }
    stak[++frt]=pos;
}
int sfa[maxn],sdsiz[maxn],near[maxn],ctrl[maxn];
void dfs1(int pos,int f){
    sfa[pos]=f;
    sdsiz[pos]=siz[pos];
    for(REG unsigned i=0;i<ftree[pos].size();i++){
        REG int tar=ftree[pos][i];
        if(tar==f){
            continue;
        }
        dfs1(tar,pos);
        sdsiz[pos]-=siz[cut::upper(tar,dep[tar]-dep[pos]-1)];
        if(!near[pos]||dep[near[tar]]<dep[near[pos]]||(dep[near[tar]]==dep[near[pos]]&&near[tar]<near[pos])){
            near[pos]=near[tar];
        }
    }
    if(flags[pos]){
        near[pos]=pos;
    }
}
INL int selenear(REG int pos,REG int pos1,REG int pos2){
    REG int ans=pos1,ndis=dep[pos1]-dep[pos],tmp;
    if((tmp=cut::dist(pos,pos2))<=ndis){
        if(tmp<ndis||(tmp==ndis&&pos2<ans)){
            ans=pos2;
            ndis=tmp;
        }
    }
    return ans;
}
void dfs2(int pos){
    if(pos!=1&&!flags[pos]){
        near[pos]=selenear(pos,near[pos],near[sfa[pos]]);
    }
    for(REG unsigned i=0;i<ftree[pos].size();i++){
        REG int tar=ftree[pos][i];
        if(tar==sfa[pos]){
            continue;
        }
        dfs2(tar);
    }
}
void dfs3(int pos){
    if(pos==1){
        ctrl[near[pos]]+=sdsiz[pos];
    }
    for(REG unsigned i=0;i<ftree[pos].size();i++){
        REG int tar=ftree[pos][i];
        if(tar==sfa[pos]){
            continue;
        }
        //cout<<pos<<' '<<tar<<' '<<near[pos]<<' '<<near[tar]<<' '<<ctrl[near[pos]]<<' '<<ctrl[near[tar]];
        if(near[pos]==near[tar]){
            ctrl[near[pos]]+=siz[cut::upper(tar,dep[tar]-dep[pos]-1)]-siz[tar]+sdsiz[tar];
        }
        else{
            REG int ldis=cut::dist(tar,near[tar]),rdis=cut::dist(pos,near[pos]);
            if((ldis+rdis+dep[tar]-dep[pos])&1){
                REG int ltop=cut::upper(tar,((ldis+rdis+dep[tar]-dep[pos]-1)>>1)-ldis);
                ctrl[near[tar]]+=siz[ltop]-siz[tar]+sdsiz[tar];
                if(fa[ltop]!=pos){
                    ctrl[near[pos]]+=siz[cut::upper(tar,dep[tar]-dep[pos]-1)]-siz[ltop];
                }
            }
            else{
                REG int midbl,mid=cut::upper(tar,((ldis+rdis+dep[tar]-dep[pos])>>1)-ldis);
                if(mid!=tar){
                    midbl=cut::upper(tar,((ldis+rdis+dep[tar]-dep[pos])>>1)-ldis-1);
                    if(mid!=pos){
                        if(near[tar]<near[pos]){
                            ctrl[near[tar]]+=siz[mid]-siz[midbl];
                        }
                        else{
                            ctrl[near[pos]]+=siz[mid]-siz[midbl];
                        }
                        ctrl[near[pos]]+=siz[cut::upper(tar,dep[tar]-dep[pos]-1)]-siz[mid];
                    }
                    ctrl[near[tar]]+=siz[midbl]-siz[tar]+sdsiz[tar];
                }
                else{
                    ctrl[near[tar]]+=sdsiz[tar];
                    ctrl[near[pos]]+=siz[cut::upper(tar,dep[tar]-dep[pos]-1)]-siz[mid];
                }
            }
        }
        //cout<<' '<<ctrl[near[pos]]<<' '<<ctrl[near[tar]]<<endl;
        dfs3(tar);
    }
}
void dfsclear(int pos){
    for(REG unsigned i=0;i<ftree[pos].size();i++){
        REG int tar=ftree[pos][i];
        if(tar==sfa[pos]){
            continue;
        }
        dfsclear(tar);
    }
    ftree[pos].clear();
    near[pos]=ctrl[pos]=0;
    flags[pos]=0;
}
int main(){
    n=read();
    for(REG int i=0;i<n-1;i++){
        REG int x=read(),y=read();
        graf[x].push_back(y);
        graf[y].push_back(x);
    }
    cut::dfs1(1,0,0);
    cut::dfs2(1,1);
    m=read();
    while(m--){
        REG int p=read();
        for(REG int i=0;i<p;i++){
            backup[i]=lst[i].pos=read();
            lst[i].key=dfn[lst[i].pos];
            flags[lst[i].pos]=1;
        }
        if(p==1){
            printf("%d\n",n);
            continue;
        }
        sort(lst,lst+p,cmp);
        if(lst->pos!=1){
            insert(1);
        }
        for(REG int i=0;i<p;i++){
            insert(lst[i].pos);
        }
        while(frt>1){
            ftree[stak[frt-1]].push_back(stak[frt]);
            ftree[stak[frt]].push_back(stak[frt-1]);
            frt--;
        }
        frt=0;
        dfs1(1,0);
        dfs2(1);
        dfs3(1);
        for(REG int i=0;i<p;i++){
            printf("%d ",ctrl[backup[i]]);
        }
        printf("\n");
        /*for(REG int i=0;i<p;i++){
            printf("%d ",sdsiz[backup[i]]);
        }
        printf("\n");*/
        dfsclear(1);
    }
    return 0;
}
```

### 可持久化线段树

```cpp
/**
 * @file Chairman_tree.cpp
 * @author throusea (1353272517@qq.com)
 * @brief 主席树经典问题
 * @version 0.1
 * @date 2022-02-23
 * 
 * @copyright Copyright (c) 2022
 * 
 */
#include <bits/stdc++.h>
using namespace std;
const int maxn = 2e5+7;
struct node {
    int sum;
    node *ch[2];
};
node* root[maxn];
int a[maxn];
void copy_node (node* &p, node* &d) {
    d->ch[0] = p->ch[0];
    d->ch[1] = p->ch[1];
    d->sum = p->sum+1;
}

void insert(node* p, node* &d, int L, int R, int x) {
    int M = (L+R)/2;
    d = new node();
    if (p == nullptr)
        d->sum = 1;
    else
        copy_node(p,d);

    if (L == R) return;
    if (x <= M) 
        insert(p == nullptr ? p : p->ch[0], d->ch[0], L, M, x);
    else 
        insert(p == nullptr ? p : p->ch[1], d->ch[1], M+1, R, x);
}

int ans;
void query(node* p, node* d, int L, int R, int k) {
    int l_sum = 0;
    if (L == R) {
        ans = L;
        return;
    }
    if (d->ch[0] != nullptr) 
        l_sum = d->ch[0]->sum;
    if (p != nullptr && p->ch[0] != nullptr)
        l_sum -= p->ch[0]->sum;
    int sum = d->sum;
    if (p != nullptr) sum -= p->sum;
    int M = (L+R)/2;
    if (l_sum >= k)
        query(p == nullptr ? p : p->ch[0], d->ch[0], L, M, k);
    else
        query(p == nullptr ? p : p->ch[1], d->ch[1], M+1, R, k-l_sum);
}

int mp[maxn];
int* dsct(int *arr, int len) {
    int *b = new int[len+1];
    for (int i = 1; i <= len; i++)
    b[i] = arr[i];
    sort(b+1, b+len+1);
    int cnt = unique(b+1, b+len+1) - b - 1;
    int *c = new int[len+1];
    for (int i = 1; i <= len; i++) {
        c[i] = lower_bound(b+1, b+cnt+1, arr[i])-b;
        mp[c[i]] = i;
    }
    return c;
}

int main() {
    int n, q;
    scanf("%d%d",&n,&q);
    for (int i = 1; i <= n; i++)
    scanf("%d",&a[i]);
    int *c = dsct(a, n); //离散后的数组
    for (int i = 1; i <= n; i++)
    insert(root[i-1], root[i], 1, n, c[i]);

    while (q--) {
        int L, R, k;
        scanf("%d%d%d",&L,&R,&k);
        query(root[L-1], root[R], 1, n, k);
        printf("%d\n", a[mp[ans]]);
    }
    return 0;
}
```

## 字符串 模板汇总

### 字符串哈希

啥？你说这玩意还需要模板？

### Trie树

（待补充）

### KMP

模板：[洛谷3375 【模板】KMP字符串匹](https://www.luogu.com.cn/problem/P3375)

```cpp
string s1,s2;//s1是文本串，s2是模式串
int kmp[1145140];
int main(){
    ios::sync_with_stdio(0);
    cin>>s1>>s2;
    kmp[0]=-1;
    for(int i=1,j=-1;i<s2.length();i++){
        while(j>=0&&s2[i]!=s2[j+1]){
            j=kmp[j];
        }
        if(s2[i]==s2[j+1]){
            j++;
        }
        kmp[i]=j;
    }
    for(int i=0,j=-1;i<s1.length();i++){
        while(j>=0&&s1[i]!=s2[j+1]){
            j=kmp[j];
        }
        if(s1[i]==s2[j+1]){
            j++;
        }
        if(j==s2.length()-1){
            cout<<i-j+1<<endl;
            j=kmp[j];
        }
    }
    for(int i=0;i<s2.length();i++){
        cout<<kmp[i]+1<<' ';
    }
    return 0;
}
```

#### exKMP

（待补充）

### AC自动机

模板：[洛谷3808 【模板】AC 自动机（简单版）](https://www.luogu.com.cn/problem/P3808)

```cpp
const int maxn=1145140;
class ACAM{
    private:
    struct node{
        int num,fail,son[26];
    }tree[maxn];
    int cntr=1;
    public:
    void add(string s){//ACAM的核心是trie树，加入一个模式串的过程和trie树完全一致
        int pos=1;
        for(U i=0;i<s.length();i++){
            int& nextpos=tree[pos].son[s[i]-'a'];
            if(!nextpos){
                nextpos=++cntr;
            }
            pos=nextpos;
        }
        tree[pos].num++;
    }
    void buildFail(){//建立fail指针，fail指针指向目前状态的最长后缀状态
        for(int i=0;i<26;i++){
            tree[0].son[i]=1;
        }
        queue<int> q;
        q.push(1);
        while(!q.empty()){
            int pos=q.front();
            q.pop();
            for(int i=0;i<26;i++){
                int& nextpos=tree[pos].son[i];
                int ffail=tree[pos].fail;
                if(!nextpos){
                    nextpos=tree[ffail].son[i];
                    continue;
                }
                tree[nextpos].fail=tree[ffail].son[i];
                q.push(nextpos);
            }
        }
    }
    int calc(string s){//这个函数是用来计算一个文本串的多匹配的，注意为了去重，每走过一个状态就把这个状态的计数器清空了
        int ans=0,pos=1;
        for(U i=0;i<s.length();i++){
            int nextpos=tree[pos].son[s[i]-'a'];
            while(pos>1&&tree[nextpos].num!=-1){
                ans+=tree[nextpos].num;
                tree[nextpos].num=-1;
                nextpos=tree[nextpos].fail;
            }
            pos=tree[pos].son[s[i]-'a'];
        }
        return ans;
    }
}acam;
int main(){
    ios::sync_with_stdio(0);
    int n;
    cin>>n;
    string s;
    while(n--){
        cin>>s;
        acam.add(s);
    }
    acam.buildFail();
    cin>>s;
    cout<<acam.calc(s);
    return 0;
}
```

### SAM

模板：[洛谷3804 【模板】后缀自动机 (SAM)](https://www.luogu.com.cn/problem/P3804)

```cpp
const int maxn=1145140;
string s;

class SAM{
    public:
    struct state{
        int len,siz,link,nxt[26];
    }st[maxn<<1];
    int siz,last;
    vector<int> graf[maxn<<1];
    void init(){
        siz=1;
        st[0].len=0;
        st[0].siz=0;
        st[0].link=-1;
    }
    void extend(int c){
        int cur=siz++;
        st[cur].len=st[last].len+1;
        st[cur].siz=1;
        int p=last;
        while(p!=-1&&!st[p].nxt[c]){
            st[p].nxt[c]=cur;
            p=st[p].link;
        }
        if(p==-1){
            st[cur].link=0;
        }
        else{
            int q=st[p].nxt[c];
            if(st[p].len+1==st[q].len){
                st[cur].link=q;
            }
            else{
                int clone=siz++;
                st[clone].siz=0;
                st[clone].len=st[p].len+1;
                st[clone].link=st[q].link;
                st[cur].link=st[q].link=clone;
                memcpy(st[clone].nxt,st[q].nxt,26*sizeof(int));
                while(p!=-1&&st[p].nxt[c]==q){
                    st[p].nxt[c]=clone;
                    p=st[p].link;
                }
            }
        }
        last=cur;
    }
    void construct_tree(){
        for(int i=1;i<siz;i++){
            graf[st[i].link].push_back(i);
        }
    }
    void dfs(int pos){
        for(U i=0;i<graf[pos].size();i++){
            dfs(graf[pos][i]);
            st[pos].siz+=st[graf[pos][i]].siz;
        }
    }
    ll getans(){
        ll ans=0;
        for(int i=1;i<siz;i++){
            if(st[i].siz>1){
                ans=max(ans,(ll)st[i].len*st[i].siz);
            }
        }
        return ans;
    }
}sam;

int main(){
    cin>>s;
    sam.init();//千万不要忘了初始化！！！
    for(U i=0;i<s.length();i++){
        sam.extend(s[i]-'a');//注意这个板子在插入字符的时候插入的实际是字符-'a'！！！
    }
    sam.construct_tree();
    sam.dfs(0);
    cout<<sam.getans();
    return 0;
}
```

另一个模板：[洛谷1368 【模板】最小表示法](https://www.luogu.com.cn/problem/P1368)

```cpp
const int maxn=3*114514;
int n,lst[maxn];
class SAM{
    public:
    struct state{
        int len,link;
        map<int,int> nxt;//注意这里的转移使用了map，不同于上一个模板
    }st[maxn<<2];
    int siz,last;
    void init(){
        siz=1;
        st[0].len=0;
        st[0].link=-1;
    }
    void extend(int c){
        int cur=siz++;
        st[cur].len=st[last].len+1;
        int p=last;
        while(p!=-1&&!st[p].nxt[c]){
            st[p].nxt[c]=cur;
            p=st[p].link;
        }
        if(p==-1){
            st[cur].link=0;
        }
        else{
            int q=st[p].nxt[c];
            if(st[p].len+1==st[q].len){
                st[cur].link=q;
            }
            else{
                int clone=siz++;
                st[clone].len=st[p].len+1;
                st[clone].link=st[q].link;
                st[cur].link=st[q].link=clone;
                st[clone].nxt=st[q].nxt;
                while(p!=-1&&st[p].nxt[c]==q){
                    st[p].nxt[c]=clone;
                    p=st[p].link;
                }
            }
        }
        last=cur;
    }
    void output(){
        int p=0,cntr=0;
        while(cntr<n){
            cout<<(*st[p].nxt.begin()).first<<' ';
            p=(*st[p].nxt.begin()).second;
            cntr++;
        }
    }
}sam;

int main(){
    cin>>n;
    sam.init();
    for(int i=0;i<n;i++){
        cin>>lst[i];
        sam.extend(lst[i]);
    }
    for(int i=0;i<n;i++){
        sam.extend(lst[i]);
    }
    sam.output();
    return 0;
}
```

### Manacher

求回文半径，没啥花活

模板：[洛谷3805 【模板】manacher 算法](https://www.luogu.com.cn/problem/P3805)

```cpp
vector<int> ext;//存放处理过后的字符串
int rad[11451400<<1];//存储回文半径，注意要开最大长度的两倍！！
void manachar(string s){
    ext.clear();
    ext.push_back(-1);
    ext.push_back(-1);
    for(unsigned i=0;i<s.length();i++){//不难发现，处理后的字符串的下标映射是s[i]=ext[2*i+2]
        ext.push_back(s[i]);
        ext.push_back(-1);
    }
    unsigned edge=0,axis=0;
    for(unsigned i=1;i<ext.size();i++){
        if(i<edge){
            rad[i]=min((unsigned)rad[(axis<<1)-i],rad[axis]-(i-axis));
        }
        else{
            rad[i]=1;
        }
        while(i-rad[i]>=0&&i+rad[i]<ext.size()&&ext[i-rad[i]]==ext[i+rad[i]]){
            rad[i]++;
        }
        if(i+rad[i]>edge){
            edge=i+rad[i];
            axis=i;
        }
    }
}
int main(){
    freopen("LG3805.in","r",stdin);
    string s;
    ios::sync_with_stdio(0);
    cin>>s;
    manachar(s);
    int maxans=0;
    for(unsigned i=0;i<ext.size();i++){
        maxans=max(maxans,rad[i]-1);
    }
    cout<<maxans;
    return 0;
}
```

### 回文树

回文树是一种可以存储一个串中所有回文子串的高效数据结构，使用回文树可以简单高效地解决一系列涉及回文串的问题。

回文树实际上是两棵树，偶数长度的回文串的根节点是$0$，奇数长度的根节点是$-1$。

一个节点的$fail$指针指向的是这个节点所代表的回文串的最长回文后缀所对应的节点，但是转移边并非代表在原节点代表的回文串后加一个字符，而是表示在原节点代表的回文串前后各加一个相同的字符（不难理解，因为要保证存的是回文串）。

每个节点代表一个本质不同的回文子串，每个节点上的$len$值表示此节点对应回文子串的长度，每个节点的$fail$指针指向这个节点的最长的回文后缀。

模板：[G-Magic Spells_"蔚来杯"2022牛客暑期多校训练营9](https://ac.nowcoder.com/acm/contest/33194/G)

```cpp
const int maxn = 114514*3;
int k;

class PAM{
    private:
    int sz, tot, last;
    int cnt[maxn],ch[maxn][26],len[maxn],fail[maxn];
    char s[maxn];
    int node(int l){  // 建立一个新节点，长度为 l
        sz++;
        memset(ch[sz],0,sizeof(ch[sz]));
        len[sz]=l;
        fail[sz]=cnt[sz]=0;
        return sz;
    }
	public:
    void clear(){  // 初始化
        sz=-1;
        last=0;
        s[tot=0]='$';
        node(0);
        node(-1);
        fail[0]=1;
    }

    int getfail(int x){  // 找后缀回文
        while(s[tot-len[x]-1]!=s[tot]){
            x=fail[x];
        }
        return x;
    }

    void insert(char c,int id){  // 建树
        s[++tot]=c;
        int now=getfail(last);
        if (!ch[now][c-'a']){
            int x=node(len[now]+2);
            fail[x]=ch[getfail(fail[now])][c-'a'];
            ch[now][c-'a']=x;
        }
        last=ch[now][c-'a'];
        cnt[last]|=id;
    }

    ll solve(){
        ll ans=0;
        for(int i=sz;i>=0;i--){
            cnt[fil[i]]|=cnt[i];
        }
        for(int i=2;i<=sz;i++){// 更新答案
            if(cnt[i]==(1<<k)-1){
                ans++;
            }
        }
        return ans;
    }
}pam;


int main(){
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    pam.clear();
    cin>>k;
    for(int i=0;i<k;i++){
        string s;
        cin>>s;
        for(int j=0;j<(int)s.length();j++){
            pam.insert(s[j],(1<<i));
        }
        if(i<k-1){
            pam.insert('#',(1<<i));
        }
    }
    cout<<pam.solve();
    return 0;
}
```



## 图论

### 最短路（待补充）

### 最小生成树（待补充）

```cpp
const int maxn = 1145141;

char buf[100005], *p1 = buf, *p2 = buf;
char nc(){
    if(p1 == p2){
        p1 = buf;
        p2 = p1+fread(buf, 1, 100000, stdin);
        if(p1 == p2)return EOF;
    }
    return *p1++;
}

struct node{
    int from;
    int to;
    int k;
    int next;
    bool operator < (node a) const {
        return a.k < k;
    }
}edge[maxn<<1];

int n, m;
int head[maxn];
int cnt = 0;
int fa[maxn];


void join(int x, int y, int k){
    edge[++cnt].next = head[x];
    edge[cnt].to = y;
    edge[cnt].from = x;
    edge[cnt].k = k;
    head[x] = cnt;
}

int getfather(int x){
    return (fa[x] == x) ? x :(fa[x] = getfather(fa[x]));
}

void pre(){
    for(int i = 1;i <= n;i++){
        fa[i] = i;
    }
}

priority_queue<node> q;

int kurskal(){
    for(int i = 1;i <= m*2;i+=2){
        q.push(edge[i]);
    }
    int ret = 0;
    while(!q.empty()){
        node x = q.top();
        q.pop();
        if(getfather(x.from) == getfather(x.to))continue;
        fa[getfather(x.from)] = getfather(x.to);
        ret += x.k;
    }

    for(int i = 2;i <= n;i++){
        if(getfather(1) != getfather(i))return -1;
    }

    return ret;
}

int main(){
    #ifndef ONLINE_JUDGE
    freopen("test.in", "r", stdin);
    freopen("test.out", "w", stdout);
    #endif

    n = read();
    m = read();
    pre();

    for(int i = 1;i <= m;i++){
        int x = read(), y = read(), k = read();
        join(x, y, k);
        join(y, x, k);
    }
    int ans = kurskal();
    if(ans == -1){
        cout<<"orz";
    }else{
        cout<<ans;
    }
    return 0;
}
```



### 连通块

#### Tarjan

```cpp
const int maxn = 1e6+5;

struct node{
    int next;
    int from;
    int to;
}edge[maxn<<1];

int n, m;
int a[maxn] = {0};
int head[maxn] = {0};
int cnt = 0;
int dfn[maxn];
int stk[maxn];
int tp = 0;
int tot = 0;
int color[maxn];
int low[maxn];
int val[maxn];
bool vis[maxn] = {0};
int rd[maxn] = {0};
int dis[maxn] = {0};
int t = 0;

void join(int x, int y){
    edge[++cnt].next = head[x];
    edge[cnt].to = y;
    edge[cnt].from = x;
    head[x] = cnt;
}

void tarjan(int x){
    //cout<<x<<endl;
    dfn[x] = ++t;
    low[x] = t;
    stk[++tp] = x;
    vis[x] = true;
    for(int i = head[x];i;i = edge[i].next){
        int to = edge[i].to;
        if(!dfn[to]){
            tarjan(to);
            low[x] =  min(low[x], low[to]);
        }else if(vis[to]){
            low[x] = min(low[x], dfn[to]);
        }
    }

    if(dfn[x] == low[x]){
        ++tot;
        while(true){
            color[stk[tp]] = tot;
            val[tot] += a[stk[tp]];
            vis[stk[tp]] = false;
            tp--;
            if(stk[tp+1] == x)break;
        }
    }
}

void tuopu(){
    queue<int> q;
    for(int i = 1;i <= tot;i++){
        if(rd[i] == 0){
            dis[i] = val[i];
            q.push(i);
        }
    }

    while(!q.empty()){
        int x = q.front();
        q.pop();
        for(int i = head[x];i;i = edge[i].next){
            int to = edge[i].to;
            dis[to] = max(dis[to], dis[x]+val[to]);
            rd[to]--;
            if(rd[to] == 0)q.push(to);
        }
    }
}


int main(){
    cin>>n>>m;
    for(int i = 1;i <= n;i++){
        scanf("%d", &a[i]);
    }

    for(int i = 1;i <= m;i++){
        int x, y;
        scanf("%d%d", &x, &y);
        join(x, y);
    }

    for(int i = 1;i <= n;i++){
        if(!dfn[i])tarjan(i);
    }

    memset(head, 0, sizeof(head));

    for(int i = 1;i <= m;i++){
        int x = edge[i].from;
        int y = edge[i].to;
        if(color[x] != color[y]){
            join(color[x], color[y]);
            rd[color[y]]++;
        }
    }

    tuopu();
    int ans = 0;
    for(int i = 1;i <= tot;i++){
        ans = max(ans, dis[i]);
    }
    cout<<ans<<endl;
    return 0;
}
```



### 网络流

#### 最大流

模板：[洛谷3376 【模板】网络最大流](https://www.luogu.com.cn/problem/P3376)

```cpp
const int maxn=5*1145;
int n,m,S,T;
struct edge{
    int f,t;
    ll cap,flow;
    edge(int _f=0,int _t=0,ll _cap=0){
        f=_f;t=_t;cap=_cap;flow=0;
    }
};
vector<edge> eds;
vector<int> graf[maxn];
INL void addedge(int x,int y,ll c){//加入一条x->y，容量为c的边
    eds.push_back(edge(x,y,c));
    eds.push_back(edge(y,x,0));
    graf[x].push_back(eds.size()-2);
    graf[y].push_back(eds.size()-1);
}
int lay[maxn];
unsigned cur[maxn];
bool flags[maxn];
bool getlay(){
    for(int i=0;i<=n;i++){
        lay[i]=-1;
        cur[i]=0;
        flags[i]=0;
    }
    queue<int> que;
    que.push(S);
    lay[S]=0;
    flags[S]=1;
    while(!que.empty()){
        int pos=que.front();
        que.pop();
        for(unsigned i=0;i<graf[pos].size();i++){
            edge e=eds[graf[pos][i]];
            if(!flags[e.t]&&e.cap>e.flow){
                lay[e.t]=lay[pos]+1;
                que.push(e.t);
                flags[e.t]=1;
            }
        }
    }
    return flags[T];
}
ll dinic(int pos,ll left){
    if(pos==T||!left){
        return left;
    }
    ll addf=0,nf;
    for(unsigned& i=cur[pos];i<graf[pos].size();i++){
        edge& e=eds[graf[pos][i]];
        if(lay[e.t]==lay[pos]+1&&(nf=dinic(e.t,min(left,e.cap-e.flow)))>0){
            addf+=nf;
            left-=nf;
            e.flow+=nf;
            eds[graf[pos][i]^1].flow-=nf;
            if(!left){
                break;
            }
        }
    }
    return addf;
}//最大流主体
int main(){
    n=read();m=read();S=read();T=read();//n是点数，m是边数，S是源点，T是汇点
    for(int i=0;i<m;i++){
        int x=read(),y=read();
        ll c=read();
        addedge(x,y,c);
    }
    ll maxf=0;//最大流
    while(getlay()){
        maxf+=dinic(S,2147483647);
    }
    cout<<maxf;
    return 0;
}
```

#### 最小费用最大流

模板：[洛谷3381 【模板】最小费用最大流](https://www.luogu.com.cn/problem/P3381)

```cpp
const int maxn=5*11451;
const ll inf=1145141919810000000;
int n,m,S,T;
struct edge{
    int f,t;
    ll cap,flow,len;
    INL edge(int _f=0,int _t=0,ll _cap=0,ll _len=0){
        f=_f;t=_t;cap=_cap;flow=0;len=_len;
    }
};
vector<edge> eds;
vector<int> graf[maxn];
ll shrt[maxn],addf[maxn],pre[maxn];
bool inq[maxn];
ll maxf,maxc;
INL void addedge(int x,int y,ll c,ll l){//加入一条x->y，容量为c，单位流量费用为l的边
    eds.push_back(edge(x,y,c,l));
    eds.push_back(edge(y,x,0,-l));
    graf[x].push_back(eds.size()-2);
    graf[y].push_back(eds.size()-1);
}
bool SPFA(){
    for(int i=0;i<=n;i++){
        shrt[i]=inf;
        pre[i]=-1;
        inq[i]=0;
    }
    queue<int> que;
    que.push(S);
    shrt[S]=0;
    addf[S]=inf;
    inq[S]=1;
    while(!que.empty()){
        int pos=que.front();
        que.pop();
        for(unsigned i=0;i<graf[pos].size();i++){
            edge e=eds[graf[pos][i]];
            if(e.cap>e.flow&&shrt[e.t]>shrt[pos]+e.len){
                shrt[e.t]=shrt[pos]+e.len;
                addf[e.t]=min(addf[pos],e.cap-e.flow);
                pre[e.t]=graf[pos][i];
                if(!inq[e.t]){
                    inq[e.t]=1;
                    que.push(e.t);
                }
            }
        }
        inq[pos]=0;
    }
    if(pre[T]==-1){
        return 0;
    }
    maxf+=addf[T];
    maxc+=addf[T]*shrt[T];
    for(int pos=T;pos!=S;pos=eds[pre[pos]].f){
        eds[pre[pos]].flow+=addf[T];
        eds[pre[pos]^1].flow-=addf[T];
    }
    return 1;
}//费用流主体
int main(){
    n=read();m=read();S=read();T=read();//n是点数，m是边数，S是源点，T是汇点
    for(int i=0;i<m;i++){
        int x=read(),y=read();
        ll c=read(),l=read();
        addedge(x,y,c,l);
    }
    while(SPFA());
    cout<<maxf<<' '<<maxc;//最大流和最小费用
    return 0;
}
```

#### 无源汇上下界可行流

首先对于原图中的每一条$u \rarr v$，流量上下界为$[l,r]$的边，**实际**连一条$u \rarr v$，流量上下界为$[0,r-l]$的边；

设$S'$为附加源点，$T'$为附加汇点，对于一个点$u$，$d_u = 入边的下界之和 - 出边的下界之和$；

若$d_u=0$，此时流量平衡；

若$d_u>0$，此时入流量过大，$S'$向$u$连流量为$d_u$的边；

若$d_u<0$，此时出流量过大，$u$向$T'$连流量为$-d_u$的边。

**连着$S'$或$T'$的边统称为附加边。**

在建图完毕之后跑$S'$到$T'$的最大流，若$S'$连出去的边全部满流，则存在可行流，否则不存在。

#### 有源汇上下界可行流

加入一条$S$到$T$的上界为$\infin$，下界为$0$的边转化为无源汇上下界可行流问题。

#### 有源汇上下界最大/最小流

跑完任意一个可行流之后，在残量网络上再跑一次$S$到$T$的最大流（最大流）/$T$到$S$的最大流（最小流），将可行流流量和最大流流量相加（最大流）/减（最小流）即为答案。

#### 有源汇最小费用可行流

按有源汇上下界可行流建边，**原图的边的费用与原图一致，附加边的费用为0。**

### 二分图匹配（待补充）

## 数学

### 扩展欧几里得（待补充）

### 筛法（欧拉筛，素数筛，待补充）

### 快速傅里叶变换（FFT）

```cpp
const int maxn = 3e6+5;
const double Pi = acos(-1.0);

int n, m;
int r[maxn] = {0};
int limit = 1;

struct complex{
    double x, y;
    complex(double xx = 0, double yy = 0){
        x = xx;
        y = yy;
    }
}a[maxn], b[maxn];

complex operator + (complex a, complex b){
    return complex(a.x + b.x, a.y + b.y);
}

complex operator - (complex a, complex b){
    return complex(a.x - b.x, a.y - b.y);
}

complex operator * (complex a, complex b){
    return complex(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}

void fft(complex *c, int type){
    for(int i = 0;i <= limit;i++){
        if(i < r[i])swap(c[i], c[r[i]]);
    }

    for(int mid = 1;mid < limit;mid <<= 1){
        complex wn(cos(Pi/(double)mid) , type*sin(Pi/(double)mid));
        for(int k = mid<<1, j = 0; j < limit; j += k){
            complex w(1, 0);
            for(int l = 0;l < mid;l++){
                complex x = c[j+l], y = w*c[j+l+mid];
                c[j+l] = x+y;
                c[j+l+mid] = x-y;
                w = wn*w;
            }
        }
    }
}

int main(){
    int n, m;
    cin>>n>>m;
    for(int i = 0;i <= n;i++){
        scanf("%lf", &a[i].x);
    }
    for(int j = 0;j <= m;j++){
        scanf("%lf", &b[j].x);
    }
    int l = 0;
    while(limit <= n+m)limit <<= 1, l++;
    for(int i = 0;i <= limit;i++){
        r[i] = (r[i>>1]>>1) | ((i&1)<<(l-1));
    }
    fft(a, 1);
    fft(b, 1);
    for(int i = 0;i <= limit;i++){
        a[i] = a[i] * b[i];
    }
    fft(a, -1);
    for(int i = 0;i <= n+m;i++){
        printf("%d ", (int)(a[i].x/limit+0.5));
    }
    return 0;
}
```

### 快速数论变换（NTT）

```cpp
const ll maxn = 3e6+5;
const double Pi = acos(-1.0);
const ll mod = 998244353;
const ll g = 3, invg = 332748118;

ll n, m;
ll r[maxn] = {0};
ll limit = 1;
ll a[maxn], b[maxn];

ll ksm(ll a, ll k){
    ll ret = 1;
    while(k){
        if(k&1){
            ret *= a;
            ret %= mod;
        }
        a *= a;
        a %= mod;
        k >>= 1;
    }
    return ret % mod;
}

void ntt(ll *c, ll type){
    for(ll i = 0;i < limit;i++){
        if(i < r[i])swap(c[i], c[r[i]]);
    }
    for(int i = 0;i < limit;i++){
        cout<<c[i]<<" ";
    }cout<<endl;

    for(ll mid = 1;mid < limit;mid <<= 1){
        ll wn = ksm((type == 1 ? g : invg), (mod-1)/(mid<<1));
        for(ll k = mid<<1, j = 0; j < limit; j += k){
            ll w = 1;
            for(ll l = 0;l < mid;l++){
                ll x = c[j+l], y = w*c[j+l+mid]%mod;
                c[j+l] = (x+y)%mod;
                c[j+l+mid] = (x-y+mod)%mod;
                w = wn*w%mod;
            }
        }
    }
}

int main(){
    ll n, m;
    cin>>n>>m;
    for(ll i = 0;i <= n;i++){
        scanf("%ld", &a[i]);
    }
    for(ll j = 0;j <= m;j++){
        scanf("%ld", &b[j]);
    }
    ll l = 0;
    while(limit <= n+m)limit <<= 1, l++;
    for(ll i = 0;i < limit;i++){
        r[i] = (r[i>>1]>>1) | ((i&1)<<(l-1));
    }
    ntt(a, 1);
    ntt(b, 1);
    for(ll i = 0;i <= n;i++){
        cout<<a[i]<<" ";
    }cout<<"!a"<<endl;
    for(ll j = 0;j <= m;j++){
        cout<<b[j]<<" ";
    }cout<<"!b"<<endl;
    for(ll i = 0;i <= limit;i++){
        a[i] = a[i] * b[i]%mod;
    }
    ntt(a, -1);
    ll inv = ksm(limit, mod-2);
    for(ll i = 0;i <= n+m;i++){
        printf("%ld ", (a[i]*inv)%mod);
    }
    return 0;
}
```



## 其他

### 哈希表

```cpp
/**
 * @file hash.cpp
 * @author your name (you@domain.com)
 * @brief 哈希表
 * @version 0.1
 * @date 2022-06-04
 * 
 * @copyright Copyright (c) 2022
 * 
 */
#include<iostream>
#define LL long long
using namespace std;
const int maxn = 10000 + 7;
const int hashsize = 1000003;
string strs[maxn];
int head[hashsize], nxt[maxn];
string st[maxn];
int convert(char c) {
    if('0' <= c && c <= '9') return c - '0';
    else if('A' <= c && c <= 'Z') return c - 'A' + 11;
    else return c - 'a' + 37;
}

int _hash(string s) {
    LL tol = 0;
    for(int i = 0; i < s.size(); i++)
    tol = (tol * 62 + convert(s[i])) % hashsize;
    return tol;
}

int k;
int try_to_insert(string s) {
    int h = _hash(s);
    int u = head[h];
    while(u) {
        if(st[u] == s) return 0;
        u = nxt[u];
    }
    nxt[++k] = head[h];
    st[k] = s;
    head[h] = k;
    return 1;
}

int main() {
    int n;
    cin>>n;
    int cnt = 0;
    for(int i = 1; i <= n; i++) {
        cin>>strs[i];
        if(try_to_insert(strs[i])) cnt++;
    }
    cout<<cnt<<endl;
    return 0;
}
```

