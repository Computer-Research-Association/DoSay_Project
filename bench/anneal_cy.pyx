# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
from libc.math cimport exp
from libc.stdint cimport uint64_t

DEF ROWS=9
DEF COLS=18
DEF NC=162
DEF MAXM=600
DEF MAXL=110

cdef uint64_t _S

cdef inline uint64_t xs() nogil:
    global _S
    cdef uint64_t x=_S
    x^=x<<13; x^=x>>7; x^=x<<17; _S=x
    return x
cdef inline double rd() nogil:
    return (xs()>>11)*(1.0/9007199254740992.0)
cdef inline int rr(int n) nogil:
    return <int>(xs()%<uint64_t>n)

cdef int find_moves(int* g, int* M) nogil:
    cdef int P[10][19]
    cdef int r,c,r1,r2,c1,c2,s,top,bot,lf,rt,n=0
    for r in range(10):
        for c in range(19): P[r][c]=0
    for r in range(ROWS):
        for c in range(COLS):
            P[r+1][c+1]=g[r*COLS+c]+P[r][c+1]+P[r+1][c]-P[r][c]
    for r1 in range(ROWS):
        for r2 in range(r1,ROWS):
            for c1 in range(COLS):
                for c2 in range(c1,COLS):
                    s=P[r2+1][c2+1]-P[r1][c2+1]-P[r2+1][c1]+P[r1][c1]
                    if s==10:
                        top=P[r1+1][c2+1]-P[r1][c2+1]-P[r1+1][c1]+P[r1][c1]
                        bot=P[r2+1][c2+1]-P[r2][c2+1]-P[r2+1][c1]+P[r2][c1]
                        lf=P[r2+1][c1+1]-P[r1][c1+1]-P[r2+1][c1]+P[r1][c1]
                        rt=P[r2+1][c2+1]-P[r1][c2+1]-P[r2+1][c2]+P[r1][c2]
                        if top>0 and bot>0 and lf>0 and rt>0:
                            M[4*n]=r1; M[4*n+1]=c1; M[4*n+2]=r2; M[4*n+3]=c2; n+=1
                    elif s>10: break
    return n

cdef int apply1(int* g,int r1,int c1,int r2,int c2) nogil:
    cdef int r,c,cl=0
    for r in range(r1,r2+1):
        for c in range(c1,c2+1):
            if g[r*COLS+c]>0:
                cl+=1; g[r*COLS+c]=0
    return cl

cdef int fewest_idx(int* g,int* M,int n) nogil:
    cdef int i,best=0,bc=1000000,k,r,c
    for i in range(n):
        k=0
        for r in range(M[4*i],M[4*i+2]+1):
            for c in range(M[4*i+1],M[4*i+3]+1):
                if g[r*COLS+c]>0: k+=1
        if k<bc: bc=k; best=i
    return best

cdef int rollout(int* g,int fr1,int fc1,int fr2,int fc2,double gp,int* Seq,int* outlen) nogil:
    cdef int M[MAXM*4]
    cdef int total=0, L=0, n, idx
    total+=apply1(g,fr1,fc1,fr2,fc2)
    Seq[0]=fr1;Seq[1]=fc1;Seq[2]=fr2;Seq[3]=fc2;L=1
    while True:
        n=find_moves(g,M)
        if n==0: break
        if rd()>=gp: idx=rr(n)
        else: idx=fewest_idx(g,M,n)
        total+=apply1(g,M[4*idx],M[4*idx+1],M[4*idx+2],M[4*idx+3])
        Seq[4*L]=M[4*idx];Seq[4*L+1]=M[4*idx+1];Seq[4*L+2]=M[4*idx+2];Seq[4*L+3]=M[4*idx+3];L+=1
        if L>=MAXL-1: break
    outlen[0]=L
    return total

cdef void build_prefix(int* b0,int* Seq,int L,int* pg,int* pc) nogil:
    cdef int i,step,acc=0
    for i in range(NC): pg[i]=b0[i]
    pc[0]=0
    for step in range(L):
        for i in range(NC): pg[(step+1)*NC+i]=pg[step*NC+i]
        acc+=apply1(&pg[(step+1)*NC], Seq[4*step],Seq[4*step+1],Seq[4*step+2],Seq[4*step+3])
        pc[step+1]=acc

cdef int pick_worst(int* pc,int n) nogil:
    cdef double w[MAXL]
    cdef double tot=0, rv, acc=0
    cdef int i,d
    for i in range(n):
        d=pc[i+1]-pc[i]-2
        w[i]=<double>(d*d)+0.1
        tot+=w[i]
    rv=rd()*tot
    for i in range(n):
        acc+=w[i]
        if acc>=rv: return i
    return n-1

def anneal_cy(signed char[:,:] board, int iters, unsigned long long seed, double T0=3.0, double gp=0.8):
    global _S
    _S = seed*2747636419 + 1
    if _S==0: _S=0x9E3779B97F4A7C15
    cdef int b0[NC], g[NC], M[MAXM*4]
    cdef int cur[MAXL*4], best[MAXL*4], tail[MAXL*4]
    cdef int pg[(MAXL+1)*NC]
    cdef int pc[MAXL+1]
    cdef int curlen=0, bestlen=0, taillen=0
    cdef int i,r,c,it,n,idx,t,att, curscore,bestscore,newscore, tailtot
    cdef double T, delta
    for r in range(ROWS):
        for c in range(COLS): b0[r*COLS+c]=board[r,c]
    for i in range(NC): g[i]=b0[i]
    n=find_moves(g,M)
    if n==0: return 0, []
    idx=fewest_idx(g,M,n)
    curscore=rollout(g, M[4*idx],M[4*idx+1],M[4*idx+2],M[4*idx+3], 1.0, cur, &curlen)
    bestscore=curscore; bestlen=curlen
    for i in range(curlen*4): best[i]=cur[i]
    build_prefix(b0, cur, curlen, pg, pc)
    for it in range(iters):
        T=T0*(1.0-(<double>it)/iters)+1e-6
        if curlen<2: break
        t=pick_worst(pc, curlen)
        for i in range(NC): g[i]=pg[t*NC+i]
        n=find_moves(g,M)
        if n<2: continue
        idx=rr(n); att=0
        while (M[4*idx]==cur[4*t] and M[4*idx+1]==cur[4*t+1] and M[4*idx+2]==cur[4*t+2] and M[4*idx+3]==cur[4*t+3]):
            idx=rr(n); att+=1
            if att>25: break
        tailtot=rollout(g, M[4*idx],M[4*idx+1],M[4*idx+2],M[4*idx+3], gp, tail, &taillen)
        newscore=pc[t]+tailtot
        delta=<double>(newscore-curscore)
        if delta>=0 or rd()<exp(delta/T):
            for i in range(taillen*4): cur[t*4+i]=tail[i]
            curlen=t+taillen; curscore=newscore
            build_prefix(b0, cur, curlen, pg, pc)
            if curscore>bestscore:
                bestscore=curscore; bestlen=curlen
                for i in range(curlen*4): best[i]=cur[i]
    return bestscore, [(best[4*i],best[4*i+1],best[4*i+2],best[4*i+3]) for i in range(bestlen)]
