int valid_c(const signed char* g, int* out) {
    int P[10][19];
    for (int r=0;r<10;r++) for(int c=0;c<19;c++) P[r][c]=0;
    for (int r=0;r<9;r++) for(int c=0;c<18;c++) P[r+1][c+1]=g[r*18+c]+P[r][c+1]+P[r+1][c]-P[r][c];
    int n=0;
    for (int r1=0;r1<9;r1++) for(int r2=r1;r2<9;r2++) for(int c1=0;c1<18;c1++) for(int c2=c1;c2<18;c2++){
        int s=P[r2+1][c2+1]-P[r1][c2+1]-P[r2+1][c1]+P[r1][c1];
        if(s==10){
            int top=P[r1+1][c2+1]-P[r1][c2+1]-P[r1+1][c1]+P[r1][c1];
            int bot=P[r2+1][c2+1]-P[r2][c2+1]-P[r2+1][c1]+P[r2][c1];
            int lf =P[r2+1][c1+1]-P[r1][c1+1]-P[r2+1][c1]+P[r1][c1];
            int rt =P[r2+1][c2+1]-P[r1][c2+1]-P[r2+1][c2]+P[r1][c2];
            if(top>0&&bot>0&&lf>0&&rt>0){out[n*4]=r1;out[n*4+1]=c1;out[n*4+2]=r2;out[n*4+3]=c2;n++;}
        } else if(s>10) break;
    }
    return n;
}
