#include <math.h>
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/stitching/stitcher.hpp>
#include <iostream>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>

#define K_NEAREST_NEIGHBOURS 2           //the paper set it to 4, may need to change
#define MATCH_ACCEPT_DISTANCE_RATO 0.8
#define RANSACREPROJTHRESHOLD 3
#define MAX_PROTENTIAL_MATCHED_IMAGE_NUM 6
#define MIN_MATCHED_POINT_NUM 7

#define BLEND_LV 3
#define WAVELENGTH_SIGMA 8.0

#define KMAX 10000

using namespace cv;
using namespace std;


//
// global variables
//

int refreshFlag=1; // indicate whether the display needs to be refreshed
int localMaximumSize=10;
int suppressionWinSize=5;
string path;

struct node {
    int index;
    node *n;
};

void reflesh(int pos, void *data)
{
    refreshFlag=1;
}

Mat mulFloatToUchar(Mat &d, Mat &u)
{
    assert(d.rows==u.rows);
    assert(d.cols==u.cols);
    assert(d.type()==CV_32FC1);
    assert(u.type()==CV_8UC1);
    
    Mat result=Mat::zeros(d.rows,d.cols,CV_32FC1);
    for(int r=0; r<result.rows; r++) for (int c=0; c<result.cols; c++) {
        result.at<float>(r,c)=d.at<float>(r,c)*u.at<uchar>(r,c);
    }
    return result;
}

Mat mulVec3fToFloat(Mat &v, Mat &d)
{
    assert(v.rows==d.rows);
    assert(v.cols==d.cols);
    assert(v.type()==CV_32FC3);
    assert(d.type()==CV_32FC1);
    
    Mat result=Mat::zeros(v.rows,v.cols,CV_32FC3);
    for(int r=0; r<result.rows; r++) for (int c=0; c<result.cols; c++) {
        result.at<Vec3f>(r,c)=v.at<Vec3f>(r,c)*d.at<float>(r,c);
    }
    return result;
}

Mat mulVec3fToUchar(Mat &v, Mat &u)
{
    assert(v.rows==u.rows);
    assert(v.cols==u.cols);
    assert(v.type()==CV_32FC3);
    assert(u.type()==CV_8UC1);
    
    Mat result=Mat::zeros(v.rows,v.cols,CV_32FC3);
    for(int r=0; r<result.rows; r++) for (int c=0; c<result.cols; c++) {
        result.at<Vec3f>(r,c)=v.at<Vec3f>(r,c)*u.at<uchar>(r,c);
    }
    return result;
}

void xDerivative(Mat &src, Mat &z)
{
    float x[9]={-1,0,1,-1,0,1,-1,0,1};
    //    float x[9]={-1,0,1,-2,0,2,-1,0,1};
    Mat kx=Mat(3,3,CV_32FC1,x);
    for(int i=0;i<src.rows;i++) for(int j=0;j<src.cols;j++)
        z.at<float>(i,j)=src.at<uchar>(i,j);
    filter2D(z, z, -1, kx);
    //    z+=765;
    //    z/=6;
}

void yDerivative(Mat &src, Mat &z)
{
    float y[9]={-1,-1,-1,0,0,0,1,1,1};
    //    float y[9]={-1,-2,-1,0,0,0,1,2,1};
    Mat ky=Mat(3,3,CV_32FC1,y);
    for(int i=0;i<src.rows;i++) for(int j=0;j<src.cols;j++)
        z.at<float>(i,j)=src.at<uchar>(i,j);
    filter2D(z, z, -1, ky);
    //    z+=765;
    //    z/=6;
}

void computeGradientVectors(Mat &src, Mat &dx, Mat &dy)
{
    dx=Mat(src.rows,src.cols,CV_32FC1);
    dy=Mat(src.rows,src.cols,CV_32FC1);
    xDerivative(src,dx);
    yDerivative(src,dy);
}

void calDev(int ii, int jj, Mat &params, Point3f ui, Point3f uj, Mat &first, Mat &second, Mat &rijk)
{
    int i=ii/4;
    int j=jj/4;
    int indexI=ii%4;
    int indexJ=jj%4;
    Mat Ki=Mat::eye(3, 3, CV_32FC1);
    Mat Kj=Mat::eye(3, 3, CV_32FC1);
    Ki.at<float>(0,0)=Ki.at<float>(1,1)=params.at<float>(i*4,0);
    Kj.at<float>(0,0)=Kj.at<float>(1,1)=params.at<float>(j*4,0);
    Mat Ki_inv=Ki.inv();
    Mat Kj_inv=Kj.inv();
    
    float rixn=params.at<float>(i*4+1,0);
    float riyn=params.at<float>(i*4+2,0);
    float rizn=params.at<float>(i*4+3,0);
    Mat ide=Mat::eye(3, 3, CV_32FC1);
    Mat rix=ide.clone();
    Mat riy=ide.clone();
    Mat riz=ide.clone();
    rix.at<float>(1,1)=rix.at<float>(2,2)=cos(rixn);
    rix.at<float>(1,2)=-sin(rixn);
    rix.at<float>(2,1)=sin(rixn);
    riy.at<float>(0,0)=riy.at<float>(2,2)=cos(riyn);
    riy.at<float>(0,2)=sin(riyn);
    riy.at<float>(2,0)=-sin(riyn);
    riz.at<float>(0,0)=riz.at<float>(1,1)=cos(rizn);
    riz.at<float>(0,1)=-sin(rizn);
    riz.at<float>(1,0)=sin(rizn);
    Mat Ri;
    Ri=riz*riy*rix;
    Mat Ri_t=Ri.t();
    
    float rjxn=params.at<float>(j*4+1,0);
    float rjyn=params.at<float>(j*4+2,0);
    float rjzn=params.at<float>(j*4+3,0);
    Mat rjx=ide.clone();
    Mat rjy=ide.clone();
    Mat rjz=ide.clone();
    rjx.at<float>(1,1)=rjx.at<float>(2,2)=cos(rjxn);
    rjx.at<float>(1,2)=-sin(rjxn);
    rjx.at<float>(2,1)=sin(rjxn);
    rjy.at<float>(0,0)=rjy.at<float>(2,2)=cos(rjyn);
    rjy.at<float>(0,2)=sin(rjyn);
    rjy.at<float>(2,0)=-sin(rjyn);
    rjz.at<float>(0,0)=rjz.at<float>(1,1)=cos(rjzn);
    rjz.at<float>(0,1)=-sin(rjzn);
    rjz.at<float>(1,0)=sin(rjzn);
    Mat Rj;
    Rj=rjz*rjy*rjx;
    Mat Rj_t=Rj.t();
    
    Mat ujl=Mat::zeros(3, 1, CV_32FC1);
    ujl.at<float>(0,0)=uj.x;
    ujl.at<float>(1,0)=uj.y;
    ujl.at<float>(2,0)=uj.z;
    cout<<Ri<<endl;
    cout<<Rj<<endl;
    Mat pijkp=Ki*Ri*Rj_t*Kj_inv*ujl;
    Mat pijk=pijkp.clone();
    pijk.at<float>(0,0)=pijk.at<float>(0,0)/pijk.at<float>(2,0);
    pijk.at<float>(1,0)=pijk.at<float>(1,0)/pijk.at<float>(2,0);
    pijk.at<float>(2,0)=pijk.at<float>(2,0)/pijk.at<float>(2,0);
    Mat firstPart=Mat::zeros(2, 3, CV_32FC1);
    firstPart.at<float>(0,0)=firstPart.at<float>(1,1)=1/pijkp.at<float>(2,0);
    firstPart.at<float>(0,2)=-pijkp.at<float>(0,0)/pow(pijkp.at<float>(2,0),2);
    firstPart.at<float>(1,2)=-pijkp.at<float>(1,0)/pow(pijkp.at<float>(2,0),2);
    Mat secondPart_1;
    Mat secondPart_2;
    if(indexI==0) { //f
        secondPart_1=ide*Ri*Rj_t*Kj_inv*ujl;
    }
    else if(indexI==1) { //thetaX
        Mat z=Mat::zeros(3, 3, CV_32FC1);
        z.at<float>(1,2)=-1;
        z.at<float>(2,1)=1;
        secondPart_1=Ki*(Ri*z)*Rj_t*Kj_inv*ujl;
    }
    else if(indexI==2) { //thetaY
        Mat z=Mat::zeros(3, 3, CV_32FC1);
        z.at<float>(0,2)=1;
        z.at<float>(2,0)=-1;
        secondPart_1=Ki*(Ri*z)*Rj_t*Kj_inv*ujl;
    }
    else if(indexI==3) { //thetaZ
        Mat z=Mat::zeros(3, 3, CV_32FC1);
        z.at<float>(0,1)=-1;
        z.at<float>(1,0)=1;
        secondPart_1=Ki*(Ri*z)*Rj_t*Kj_inv*ujl;
    }
    
    if(indexJ==0) { //f
        secondPart_2=-Ki*Ri*Rj_t*Kj_inv*Kj_inv*ujl;
    }
    else if(indexJ==1) { //thetaX
        Mat z=Mat::zeros(3, 3, CV_32FC1);
        z.at<float>(1,2)=-1;
        z.at<float>(2,1)=1;
        Mat zz=Rj*z;
        Mat zzz=zz.t();
        secondPart_2=Ki*Ri*zzz*Kj_inv*ujl;
    }
    else if(indexJ==2) { //thetaY
        Mat z=Mat::zeros(3, 3, CV_32FC1);
        z.at<float>(0,2)=1;
        z.at<float>(2,0)=-1;
        Mat zz=Rj*z;
        Mat zzz=zz.t();
        secondPart_2=Ki*Ri*zzz*Kj_inv*ujl;
    }
    else if(indexJ==3) { //thetaZ
        Mat z=Mat::zeros(3, 3, CV_32FC1);
        z.at<float>(0,1)=-1;
        z.at<float>(1,0)=1;
        Mat zz=Rj*z;
        Mat zzz=zz.t();
        secondPart_2=Ki*Ri*zzz*Kj_inv*ujl;
    }
    Mat z=(-1)*firstPart*secondPart_1;
    first=z;
    second=(-1)*firstPart*secondPart_2;
    rijk=Mat::zeros(2, 1, CV_32FC1);
    rijk.at<float>(0,0)=ui.x-pijk.at<float>(0,0);
    rijk.at<float>(1,0)=ui.y-pijk.at<float>(1,0);
    cout<<ui<<endl;
    cout<<pijk<<endl;
    cout<<rijk<<endl;
}

void calJ(vector<Mat> &imgs,
          vector< vector< vector<DMatch> > > &matchz,
          vector< vector< vector<uchar> > > &maskz,
          vector< pair<int,int> > &matchedImg,
          vector< vector<KeyPoint> > &kpz,
          Mat &params,
          Mat &JTJ,
          Mat &JTR,
          float &e)
{
    JTJ=Mat::zeros(params.rows, params.rows, CV_32FC1);
    JTR=Mat::zeros(params.rows, 1, CV_32FC1);
    e=0;
    for (int ii=0; ii<params.rows; ii++) {
        float forJTR=0;
        for (int jj=ii; jj<params.rows; jj++){
            float forJTJ=0;
            int i=ii/4;
            int j=jj/4;
            
            if (i==j) {
                //only update JTJ
                forJTJ=0;
                for (int a=0; a<matchedImg.size(); a++) {
                    if (matchedImg[a].second==i) {
                        j=matchedImg[a].first;
                        for (int k=0; k<matchz[j][i].size(); k++) {
                            if(maskz[j][i][k]==0)
                                continue;
                            Point3f ui,uj;
                            uj.x=kpz[j][matchz[j][i][k].queryIdx].pt.x-imgs[j].cols/2.0;
                            uj.y=-kpz[j][matchz[j][i][k].queryIdx].pt.y+imgs[j].rows/2.0;
                            uj.z=1;
                            ui.x=kpz[i][matchz[j][i][k].trainIdx].pt.x-imgs[i].cols/2.0;
                            ui.y=-kpz[i][matchz[j][i][k].trainIdx].pt.y+imgs[i].rows/2.0;
                            ui.z=1;
                            Mat firstPart,secondPart,fake1,fake2;
                            calDev(ii, j*4, params, ui, uj, firstPart, fake1, fake2);
                            calDev(jj, j*4, params, ui, uj, secondPart, fake1, fake2);
                            Mat z;
                            Mat firstPart_t=firstPart.t();
                            z=firstPart_t*secondPart;
                            forJTJ+=z.at<float>(0,0);
                            
                            
                            calDev(j*4, ii, params, uj, ui, fake1, firstPart, fake2);
                            calDev(j*4, jj, params, uj, ui, fake1, secondPart, fake2);
                            firstPart_t=firstPart.t();
                            z=firstPart_t*secondPart;
                            forJTJ+=z.at<float>(0,0);
                        }
                    }
                }
                JTJ.at<float>(ii,jj)=forJTJ;
                JTJ.at<float>(jj,ii)=forJTJ;
                continue;  //***
            }
            
            bool ifExist=false;
            for (int k=0; k<matchedImg.size(); k++) {
                if(matchedImg[k].first==j && matchedImg[k].second==i) {
                    ifExist=true;
                    break;
                }
            }
            if(!ifExist)
                continue;
            for (int k=0; k<matchz[j][i].size(); k++) {
                if(maskz[j][i][k]==0)
                    continue;
                Point3f ui,uj;
                ui.x=kpz[i][matchz[j][i][k].trainIdx].pt.x-imgs[i].cols/2.0;
                ui.y=-kpz[i][matchz[j][i][k].trainIdx].pt.y+imgs[i].rows/2.0;
                ui.z=1;
                uj.x=kpz[j][matchz[j][i][k].queryIdx].pt.x-imgs[j].cols/2.0;
                uj.y=-kpz[j][matchz[j][i][k].queryIdx].pt.y+imgs[j].rows/2.0;
                uj.z=1;
                Mat firstPart,secondPart_1,secondPart_2;
                calDev(ii, jj, params, ui, uj, firstPart, secondPart_1, secondPart_2);
                Mat z,firstPart_t;
                firstPart_t=firstPart.t();
                z=firstPart_t*secondPart_1;
                forJTJ+=z.at<float>(0,0);
                z=firstPart_t*secondPart_2;
                forJTR+=z.at<float>(0,0);
                e+=pow(secondPart_2.at<float>(0,0),2);
                e+=pow(secondPart_2.at<float>(1,0),2);
                
                
                
                calDev(jj, ii, params, uj, ui, firstPart, secondPart_1, secondPart_2);
                firstPart_t=firstPart.t();
                z=firstPart_t*secondPart_1;
                forJTJ+=z.at<float>(0,0);
                z=firstPart_t*secondPart_2;
                forJTR+=z.at<float>(0,0);
                e+=pow(secondPart_2.at<float>(0,0),2);
                e+=pow(secondPart_2.at<float>(1,0),2);
            }
            JTJ.at<float>(ii,jj)=forJTJ;
            JTJ.at<float>(jj,ii)=forJTJ;
        }
        JTR.at<float>(ii,0)=forJTR;
    }
    //    cout<<JTJ<<endl;
    //    cout<<JTR<<endl;
}

void LM(vector<Mat> &imgs,
        vector< vector< vector<DMatch> > > &matchz,
        vector< vector< vector<uchar> > > &maskz,
        vector< pair<int,int> > &matchedImg,
        vector< vector<KeyPoint> > &kpz,
        Mat &params)
{
    float epsilon1=pow(0.1,8);
    float epsilon2=pow(0.1,8);
    float tau=0.1;
    int k=0;
    Mat A,minusG;
    float v=2.0;
    float e;
    bool found;
    float mu;
    calJ(imgs, matchz, maskz, matchedImg, kpz, params, A, minusG, e);
    Mat ide=Mat::eye(A.rows, A.cols, CV_32FC1);
    float maxG=-FLT_MAX;
    for (int i=0; i<minusG.rows; i++) {
        if(abs(minusG.at<float>(i,0))>maxG)
            maxG=abs(minusG.at<float>(i,0));
    }
    assert(maxG>=0);
    found=(maxG<=epsilon1);
    float maxAii=-FLT_MAX;
    for (int i=0; i<A.rows; i++) {
        if(A.at<float>(i,i)>maxAii)
            maxAii=A.at<float>(i,i);
    }
    mu=tau*maxAii;
    while (!found && k<KMAX) {
        k++;
        Mat cp=ide*M_PI/16;
        float avgf=0;
        for (int i=0; i<imgs.size(); i++)
            avgf+=params.at<float>(i*4,0);
        avgf/=imgs.size();
        for (int i=0; i<imgs.size(); i++)
            cp.at<float>(i*4,0)=avgf;
        Mat zcp=cp*cp;
        cp=zcp.inv();
        Mat Ap=A+mu*ide;
        Mat hlm;
        solve(Ap, minusG, hlm);
        float hlmValue=0;
        for (int i=0; i<hlm.rows; i++)
            hlmValue+=pow(hlm.at<float>(i,0),2);
        hlmValue=sqrt(hlmValue);
        float paramsValue=0;
        for (int i=0; i<params.rows; i++)
            paramsValue+=pow(params.at<float>(i,0),2);
        paramsValue=sqrt(paramsValue);
        if (hlmValue<=epsilon2*(paramsValue+epsilon2)) {
            found=true;
        }
        else {
            Mat paramsNew=params+hlm;
            
            Mat new_A,new_minusG;
            float new_e;
            calJ(imgs, matchz, maskz, matchedImg, kpz, paramsNew, new_A, new_minusG, new_e);
            float rho;
            Mat denominator=(1.0/2.0)*hlm.t()*(mu*hlm+minusG);
            rho=(e-new_e)/denominator.at<float>(0,0);
            if (rho>0) {
                params=paramsNew;
                A=new_A;
                minusG=new_minusG;
                e=new_e;
                
                maxG=-FLT_MAX;
                for (int i=0; i<minusG.rows; i++) {
                    if(abs(minusG.at<float>(i,0))>maxG)
                        maxG=abs(minusG.at<float>(i,0));
                }
                assert(maxG>=0);
                found=(maxG<=epsilon1);
                mu=mu*max((1.0/3.0),(1-pow((2*rho-1),3)));
                v=2.0;
            }
            else {
                mu*=v;
                v*=2.0;
            }
        }
        
    }
}

Mat findH(vector<Point2f> &src, vector<Point2f> &dst, int n) {
    Mat A=Mat::zeros(2*n, 9, CV_32FC1);
    for (int i=0; i<n; i++) {
        int a=i*2;
        int b=a+1;
        A.at<float>(a,0)=src[i].x;
        A.at<float>(a,1)=src[i].y;
        A.at<float>(a,2)=1;
        A.at<float>(a,6)=src[i].x * dst[i].x * (-1);
        A.at<float>(a,7)=src[i].y * dst[i].x * (-1);
        A.at<float>(a,8)=dst[i].x * (-1);
        A.at<float>(b,3)=src[i].x;
        A.at<float>(b,4)=src[i].y;
        A.at<float>(b,5)=1;
        A.at<float>(b,6)=src[i].x * dst[i].y * (-1);
        A.at<float>(b,7)=src[i].y * dst[i].y * (-1);
        A.at<float>(b,8)=dst[i].y * (-1);
    }
    Mat w,u,vt;
    SVD::compute(A, w, u, vt, SVD::FULL_UV);
    //    cout<<"A in findH:\n"<<A<<endl;
    //    cout<<"w in findH:\n"<<w<<endl;
    //    cout<<"u in findH:\n"<<u<<endl;
    //    cout<<"vt in findH:\n"<<vt<<endl;
    //    for(int i=0;i<9;i++)
    //        cout<<vt.at<float>(i,8)<<endl;
    Mat H=Mat::zeros(3, 3, CV_32FC1);
    H.at<float>(0,0)=vt.at<float>(8,0)/vt.at<float>(8,8);
    H.at<float>(0,1)=vt.at<float>(8,1)/vt.at<float>(8,8);
    H.at<float>(0,2)=vt.at<float>(8,2)/vt.at<float>(8,8);
    H.at<float>(1,0)=vt.at<float>(8,3)/vt.at<float>(8,8);
    H.at<float>(1,1)=vt.at<float>(8,4)/vt.at<float>(8,8);
    H.at<float>(1,2)=vt.at<float>(8,5)/vt.at<float>(8,8);
    H.at<float>(2,0)=vt.at<float>(8,6)/vt.at<float>(8,8);
    H.at<float>(2,1)=vt.at<float>(8,7)/vt.at<float>(8,8);
    H.at<float>(2,2)=vt.at<float>(8,8)/vt.at<float>(8,8);
    //    cout<<"H in findH:\n"<<H<<endl;
    return H;
}

void getRandom(int begin, int end, int num, vector<int> &rst)
{
    rst.clear();
    const unsigned int n = end-begin;
    const unsigned long divisor = (RAND_MAX + 1) / n;
    unsigned int k;
    for(int i=0;i<num;) {
        do { k = std::rand() / divisor; } while (k >= n);
        bool ifExist=false;
        for(int j=0;j<rst.size();j++) {
            if(rst[j]==k) {
                ifExist=true;
                break;
            }
        }
        if(ifExist)
            continue;
        rst.push_back(k);
        i++;
    }
}

pair< pair<int, float> ,Mat> & betterModel(pair< pair<int, float> ,Mat> &a, pair< pair<int, float> ,Mat> &b)
{
    float a_f=a.first.first;
    float a_s=a.first.second;
    float b_f=b.first.first;
    float b_s=b.first.second;
    if(a_f==b_f) {
        if(a_s<b_s)
            return a;
        return b;
    }
    if(a_f>b_f)
        return a;
    return b;
}

bool equalModelH(pair< pair<int, float> ,Mat> &a, pair< pair<int, float> ,Mat> &b)
{
    if(a.first.first==b.first.first && a.first.second==b.first.second)
        return true;
    return false;
}

Mat ransacFindH(vector<Point2f> &src, vector<Point2f> &dst, float t, vector<uchar> &masks)
{
    int n=4;
    int d=4;
    float kThreshold=1000;
    
    srand((unsigned)time(0));
    srand(rand());
    srand(rand());
    pair< pair<int, float> ,Mat> better_H(pair<int, float>(-1,MAXFLOAT),Mat());
    
    masks.clear();
    for(int i=0;i<src.size();i++)
        masks.push_back(0);
    
    for(int i=0;i<kThreshold;i++) {
        vector<int> rdm;
        getRandom(0, src.size(), n, rdm);
        vector<Point2f> rdmSrc;
        vector<Point2f> rdmDst;
        for(int j=0;j<n;j++) {
            rdmSrc.push_back(src[rdm[j]]);
            rdmDst.push_back(dst[rdm[j]]);
        }
        Mat H;
        H=findH(rdmSrc, rdmDst, n);
        float ifNan=0;
        for(int a=0;a<H.rows;a++)
            for(int b=0;b<H.cols;b++)
                ifNan+=H.at<float>(a,b);
        if(isnan(ifNan)) {
            i--;
            cout<<"Nan Happened1!"<<endl;
            continue;
        }
        
        Mat pts=Mat::zeros(3, (int)src.size(), CV_32FC1);
        for(int j=0;j<src.size();j++) {
            pts.at<float>(0,j)=src[j].x;
            pts.at<float>(1,j)=src[j].y;
            pts.at<float>(2,j)=1;
        }
        Mat ppts=H*pts;
        //        cout<<H<<endl;
        vector< pair<int, float> > dists;
        dists.clear();
        for(int j=0;j<src.size();j++) {
            float x=ppts.at<float>(0,j)/ppts.at<float>(2,j);
            float y=ppts.at<float>(1,j)/ppts.at<float>(2,j);
            //            cout<<"("<<x<<", "<<y<<"), "<<"("<<image[j].x<<", "<<image[j].y<<")"<<endl;
            float z=sqrt(pow(x-dst[j].x,2)+pow(y-dst[j].y,2));
            dists.push_back(pair<int, float>(j,z));
        }
        
        //        //recompute
        //        rdmSrc.clear();
        //        rdmDst.clear();
        //        for(int j=0; j<dists.size() && dists[j].second<t; j++) {
        //            rdmSrc.push_back(src[dists[j].first]);
        //            rdmDst.push_back(dst[dists[j].first]);
        //        }
        //        if(rdmDst.size()<4) {
        //            //            cout<<"d<4! continue"<<endl;
        //            //            cout<<"t="<<t<<"; d="<<rdmDst.size()<<endl;
        //            continue;
        //        }
        //
        //        H=findH(rdmSrc, rdmDst, (int)rdmDst.size());
        //        ifNan=0;
        //        for(int a=0;a<H.rows;a++)
        //            for(int b=0;b<H.cols;b++)
        //                ifNan+=H.at<float>(a,b);
        //        if(isnan(ifNan)) {
        //            i--;
        //            cout<<"Nan Happened2!"<<endl;
        //            continue;
        //        }
        //        pts=Mat::zeros(3, (int)src.size(), CV_32FC1);
        //        for(int j=0;j<src.size();j++) {
        //            pts.at<float>(0,j)=src[j].x;
        //            pts.at<float>(1,j)=src[j].y;
        //            pts.at<float>(2,j)=1;
        //        }
        //        ppts=H*pts;
        //        //          cout<<H<<endl;
        //        dists.clear();
        //        for(int j=0;j<src.size();j++) {
        //            float x=ppts.at<float>(0,j)/ppts.at<float>(2,j);
        //            float y=ppts.at<float>(1,j)/ppts.at<float>(2,j);
        //            //                  cout<<"("<<x<<", "<<y<<"), "<<"("<<image[j].x<<", "<<image[j].y<<")"<<endl;
        //            float z=sqrt(pow(x-dst[j].x,2)+pow(y-dst[j].y,2));
        //            dists.push_back(pair<int, float>(j,z));
        //        }
        
        
        int inlines=0;
        float inlinesDist=0;
        for (int j=0; j<src.size(); j++) {
            if(dists[j].second>t)   continue;
            inlines++;
            inlinesDist+=dists[j].second;
        }
        if(inlines<d)
            continue;
        
        pair< pair<int, float> ,Mat> candidate(pair<int,float>(inlines,inlinesDist), H);
        pair< pair<int, float> ,Mat> z=better_H;
        better_H=betterModel(better_H, candidate);
        if(equalModelH(better_H, candidate)) {
            for (int j=0; j<masks.size(); j++)
                masks[j]=0;
            for (int j=0; j<src.size(); j++) {
                if(dists[j].second>t)   continue;
                masks[j]=1;
            }
        }
    }
    //    cout<<"# of inliers="<<better_H.first.first<<", average inlier distance error="<<better_H.first.second/better_H.first.first<<endl;
    return better_H.second;
}

void findHomoGraphyzATSingleImg(vector<Mat> &homographyzATSingleImg,
                                vector<Mat> &imgs,
                                vector< vector<Mat> > &homographyz,
                                vector<Mat> &imgCoordinates,
                                vector< vector<int> > &inlierNumBetweenMatch,
                                vector< pair<int,int> > &addOrder)
{
    addOrder.clear();
    Mat ide=Mat::zeros(3, 3, CV_32FC1);
    ide.at<float>(0,0)=1;
    ide.at<float>(1,1)=1;
    ide.at<float>(2,2)=1;
    homographyzATSingleImg.clear();
    for (int i=0; i<imgs.size(); i++)
        homographyzATSingleImg.push_back(ide);
    pair<int,int> m;
    int mn=-1;
    for (int i=0; i<imgs.size()-1; i++) for (int j=i+1; j<imgs.size(); j++) {
        if(i==j)    continue;
        if(inlierNumBetweenMatch[i][j]>mn) {
            m.first=i;
            m.second=j;
            mn=inlierNumBetweenMatch[i][j];
        }
    }
    //    m.first=0;
    //    for (int i=1; i<imgs.size(); i++) {
    //        if(inlierNumBetweenMatch[0][i]>mn) {
    //            m.second=i;
    //            mn=inlierNumBetweenMatch[0][i];
    //        }
    //    }
    
    if(mn<1 || m.first<0 || m.first>=imgs.size() || m.second<0 || m.second>=imgs.size() || m.first==m.second) {
        cout<<"No matching image found in findHomoGraphyzATSingleImg()."<<endl;
        exit(0);
    }
    homographyzATSingleImg[m.second]=ide;
    homographyzATSingleImg[m.first]=homographyz[m.first][m.second];
    addOrder.push_back(pair<int,int>(-1,m.second));
    addOrder.push_back(pair<int,int>(m.second,m.first));
    vector< pair<int,int> > maxConsistMatches;
    assert(inlierNumBetweenMatch[m.first].size()==imgs.size());
    for (int i=0; i<inlierNumBetweenMatch[m.first].size(); i++) {
        if(i==m.first || i==m.second)
            maxConsistMatches.push_back(pair<int,int>(-1,-1)); //-1==delete, not count
        else
            maxConsistMatches.push_back(pair<int,int>(m.first,inlierNumBetweenMatch[m.first][i]));
    }
    assert(inlierNumBetweenMatch[m.second].size()==imgs.size());
    for (int i=0; i<inlierNumBetweenMatch[m.second].size(); i++) {
        if(maxConsistMatches[i].first==-1) continue;
        if(maxConsistMatches[i].second<inlierNumBetweenMatch[m.second][i]) {
            maxConsistMatches[i].first=m.second;
            maxConsistMatches[i].second=inlierNumBetweenMatch[m.second][i];
        }
    }
    int completedNum=2;
    assert(maxConsistMatches.size()==imgs.size());
    while (completedNum<imgs.size()) {
        int x=-1;
        for (int i=0; i<maxConsistMatches.size(); i++) {
            if(maxConsistMatches[i].first==-1)  continue;
            if(x==-1 || maxConsistMatches[x].second<maxConsistMatches[i].second)
                x=i;
        }
        assert(x>-1);
        Mat z=homographyzATSingleImg[maxConsistMatches[x].first]*homographyz[x][maxConsistMatches[x].first];
        homographyzATSingleImg[x]=z;
        addOrder.push_back(pair<int,int>(maxConsistMatches[x].first,x));
        //        cout<<x<<":\n"<<homographyzATSingleImg[x]<<endl;
        //        cout<<maxConsistMatches[x].first<<":\n"<<homographyzATSingleImg[maxConsistMatches[x].first]<<endl;
        maxConsistMatches[x].first=-1;
        maxConsistMatches[x].second=-1;
        for (int i=0; i<maxConsistMatches.size(); i++) {
            if(maxConsistMatches[i].first==-1) continue;
            if(maxConsistMatches[i].second<inlierNumBetweenMatch[x][i]) {
                maxConsistMatches[i].first=x;
                maxConsistMatches[i].second=inlierNumBetweenMatch[x][i];
            }
        }
        completedNum++;
    }
    return;
}

int cmpMatchedNum(const pair<int,int> &a, const pair<int,int> &b) {
    return a.second>b.second;
}

int cmpMatchedInlierNum(const int& a, const int& b) {
    return a>b;
}

void getInitWeight(vector<Mat> &imgs, vector<Mat> &weight)
{
    weight.clear();
    for (int i=0; i<imgs.size(); i++) {
        int chalf=ceil(imgs[i].cols/2.0);
        int rhalf=ceil(imgs[i].rows/2.0);
        assert(chalf>=2 && rhalf>=2);
        float cunit=1.0/(chalf-1);
        float runit=1.0/(rhalf-1);
        Mat w=Mat::zeros(imgs[i].rows, imgs[i].cols, CV_32FC1);
        for (int r=0; r<imgs[i].rows; r++) for(int c=0;c<imgs[i].cols;c++) {
            int rr;
            int cc;
            if(r<=rhalf-1)
                rr=r;
            else
                rr=imgs[i].rows-1-r;
            if(c<=chalf-1)
                cc=c;
            else
                cc=imgs[i].cols-1-c;
            w.at<float>(r,c)=rr*runit*cc*cunit;
        }
        weight.push_back(w);
    }
}

void findWeight(vector<Mat> &imgs, vector<Mat> &homographyzATSingleImg, vector<Mat> &weight, Size imgSize)
{
    getInitWeight(imgs,weight);
    for(int i=0;i<weight.size();i++) {
        Mat z;
        warpPerspective(weight[i],z,homographyzATSingleImg[i],imgSize);
        weight[i]=z;
    }
    for (int r=0; r<weight[0].rows; r++)    for (int c=0; c<weight[0].cols; c++) {
        int n=0;
        for (int i=1; i<weight.size(); i++) {
            if(weight[i].at<float>(r,c)>weight[n].at<float>(r,c)) {
                weight[n].at<float>(r,c)=0;
                n=i;
            }
            else
                weight[i].at<float>(r,c)=0;
        }
        if(weight[n].at<float>(r,c)<=0)    continue;
        weight[n].at<float>(r,c)=1;
    }

}

void getBW(vector<Mat> &resultImgs, vector<Mat> &wmax, vector< vector<Mat> > &B, vector< vector<Mat> > &W, vector<Mat> &homographyzATSingleImg, Size imgSize, Mat &ifHas)
{
    B.clear();
    W.clear();
    Mat zz;
    for (int i=0; i<resultImgs.size(); i++) {
        B.push_back(vector<Mat>());
        W.push_back(vector<Mat>());
    }
    for (int i=0; i<resultImgs.size(); i++) {
        Mat inow=resultImgs[i].clone();
        Mat z;
        inow.convertTo(z, CV_32FC3);
        inow=z;
        Mat wnow=wmax[i].clone();
        wnow.convertTo(z, CV_32FC1);
        wnow=z;
        for (int j=0; j<BLEND_LV; j++) {
            float sigma=sqrt(2.0*j+1)*WAVELENGTH_SIGMA;
            int gaussianSize=ceil(sigma*5.0);
            if (gaussianSize%2==0)  gaussianSize++;
            Mat isigma;
            Mat bsigma;
            Mat wsigma;
            GaussianBlur(inow, isigma, Size(gaussianSize, gaussianSize), sigma);
            GaussianBlur(wnow, wsigma, Size(gaussianSize, gaussianSize), sigma);
            bsigma=inow-isigma;
            
            if(j==BLEND_LV-1) {
                bsigma=inow;
                if (sigma<350.0) {
                    sigma=350.0;
                    gaussianSize=ceil(sigma*1.5);
                    if (gaussianSize%2==0)  gaussianSize++;
                    GaussianBlur(wnow, wsigma, Size(gaussianSize, gaussianSize), sigma);
                }
            }
            Mat zz;
            warpPerspective(bsigma,zz,homographyzATSingleImg[i],imgSize,INTER_LINEAR,BORDER_REPLICATE);
            B[i].push_back(zz);
            W[i].push_back(wsigma);
            
            inow=isigma.clone();
            wnow=wsigma.clone();
        }
        
        Mat zz;
        Mat ifHasx=Mat::ones(resultImgs[i].rows, resultImgs[i].cols, CV_32FC1);
        warpPerspective(ifHasx,zz,homographyzATSingleImg[i],imgSize);
        ifHasx=zz;
        if(i==0)
            ifHas=ifHasx.clone();
        else {
            zz=ifHas+ifHasx;
            ifHas=zz;
        }
    }
    for (int r=0; r<ifHas.rows; r++) for (int c=0; c<ifHas.cols; c++) {
        if(ifHas.at<float>(r,c)!=0)
            ifHas.at<float>(r,c)=255;
    }
}

Mat getPanoramaImage(vector<Mat> &homographyzATSingleImg, vector<Mat> &imgCoordinates, vector<Mat> &imgs, vector< pair<int,int> > &addOrder)
{
    float xMin=DBL_MAX;
    float xMax=DBL_MIN;
    float yMin=DBL_MAX;
    float yMax=DBL_MIN;
    vector<Mat> boundz;
    for (int i=0; i<imgs.size(); i++) {
        Mat corners=Mat::zeros(imgs[i].rows*2+imgs[i].cols*2, 1, CV_32FC2);  //x-y-coordinate
        int num=0;
        for (int j=0; j<imgs[i].rows; j++) {
            corners.at<Vec2f>(num,0)[0]=0;
            corners.at<Vec2f>(num,0)[1]=j;
            num++;
        }
        for (int j=0; j<imgs[i].cols; j++) {
            corners.at<Vec2f>(num,0)[0]=j;
            corners.at<Vec2f>(num,0)[1]=0;
            num++;
        }
        for (int j=0; j<imgs[i].rows; j++) {
            corners.at<Vec2f>(num,0)[0]=imgs[i].cols-1;
            corners.at<Vec2f>(num,0)[1]=j;
            num++;
        }
        for (int j=0; j<imgs[i].cols; j++) {
            corners.at<Vec2f>(num,0)[0]=j;
            corners.at<Vec2f>(num,0)[1]=imgs[i].rows-1;
            num++;
        }
        perspectiveTransform(corners, corners, homographyzATSingleImg[i]);
        float xMinL=DBL_MAX;
        float xMaxL=DBL_MIN;
        float yMinL=DBL_MAX;
        float yMaxL=DBL_MIN;
        for (int j=0; j<num; j++) {
            xMinL=MIN(xMinL, corners.at<Vec2f>(j,0)[0]);
            xMaxL=MAX(xMaxL, corners.at<Vec2f>(j,0)[0]);
            yMinL=MIN(yMinL, corners.at<Vec2f>(j,0)[1]);
            yMaxL=MAX(yMaxL, corners.at<Vec2f>(j,0)[1]);
        }
        xMin=MIN(xMin, xMinL);
        xMax=MAX(xMax, xMaxL);
        yMin=MIN(yMin, yMinL);
        yMax=MAX(yMax, yMaxL);
        Mat bounds=Mat::zeros(4, 1, CV_32SC1);
        bounds.at<int>(0,0)=floor(xMinL);
        bounds.at<int>(1,0)=ceil(xMaxL);
        bounds.at<int>(2,0)=floor(yMinL);
        bounds.at<int>(3,0)=ceil(yMaxL);
        boundz.push_back(bounds);
    }
    int xMinInt=(int)floor(xMin);
    int xMaxInt=(int)ceil(xMax);
    int yMinInt=(int)floor(yMin);
    int yMaxInt=(int)ceil(yMax);
    int xRange=xMaxInt-xMinInt;
    int yRange=yMaxInt-yMinInt;
    Mat trans=Mat::eye(3, 3, CV_32FC1);
    trans.at<float>(0,2)=-xMinInt;
    trans.at<float>(1,2)=-yMinInt;
    for (int i=0; i<homographyzATSingleImg.size(); i++) {
        Mat z=trans*homographyzATSingleImg[i];
        homographyzATSingleImg[i]=z;
    }
    vector<Mat> weight;
    findWeight(imgs, homographyzATSingleImg, weight, Size(xRange,yRange));
    
    vector< vector<Mat> > B;
    vector< vector<Mat> > W;
    Mat ifHas;
    getBW(imgs, weight, B, W, homographyzATSingleImg, Size(xRange,yRange), ifHas);
    
    
//    for (int i=0; i<weight.size(); i++) {
//        Mat z=weight[i]*255;
//        imwrite(path+to_string(i)+".png", z);
//        for (int j=0; j<W[i].size(); j++) {
//            Mat zz=W[i][j]*255;
//            imwrite(path+to_string(i)+"_"+to_string(j)+".png", zz);
//        }
//    }
    
    Mat panoramaF=Mat::zeros(yRange, xRange, CV_32FC3);
    
    for (int j=0; j<BLEND_LV; j++) {
        Mat molecusar=mulVec3fToFloat(B[0][j], W[0][j]);
        Mat denominator=W[0][j].clone();
        for (int i=1; i<imgs.size(); i++) {
            Mat z=mulVec3fToFloat(B[i][j], W[i][j]);
            Mat zz=molecusar+z;
            molecusar=zz.clone();
            z=denominator+W[i][j];
            denominator=z.clone();
        }
        for (int r=0; r<denominator.rows; r++) for (int c=0; c<denominator.cols; c++) {
            if(denominator.at<float>(r,c)==0) continue;
            if(ifHas.at<float>(r,c)==0)
                molecusar.at<Vec3f>(r,c)=Vec3f(0,0,0);
            molecusar.at<Vec3f>(r,c)=molecusar.at<Vec3f>(r,c)/denominator.at<float>(r,c);
        }
        Mat z=panoramaF+molecusar;
        panoramaF=z.clone();
        molecusar.convertTo(z, CV_8UC3);
//        imwrite(path+to_string(j)+".png", z);
    }
    Mat panorama;
    panoramaF.convertTo(panorama, CV_8UC3);
    return panorama;
}

void help(void)
{
    cout << endl;
    cout << "Usage: [<app-name>] [<image-directory>]" << endl;
    cout << endl;
    cout << "Key summary:" << endl;
    cout << "---------------------------------" << endl;
    cout << endl;
    cout << "<ESC>: quit" << endl;
    cout << "'q' - quit" << endl;
    
    cout << "'h' - Display a short description of the program, its command line arguments, and the keys it supports." << endl;
    cout << endl;
}

int main(int argc, char *argv[])
{
    string winName="win";
    
    int key;
    int displayMode=1;   // the current display mode
    
    Mat gImg,zImg,zzImg,zzzImg;
    Mat cImg1,cImg2;
    Mat outImg1,outImg2;
    Directory dir;
    
    bool ifMy=true;
    
    SIFT sift;
    FlannBasedMatcher matcher;
    vector< vector<KeyPoint> > kpz;  //keypoints of image i
    vector<Mat> fdz;                 //featureDescriptions of image i
    vector< vector< vector<DMatch> > > matchz; //match points between image i & j
    vector< vector<int> > inlierNumBetweenMatch; //# of inline-match-points between image i & j
    vector< vector<Mat> > homographyz;
    vector<Mat> homographyzATSingleImg;
    vector< vector< vector<uchar> > > maskz;
    vector< vector< pair<int,int> > > pMatchOnImg; //matchz[i][j].size
    vector< pair<int,int> > matchedImg;
    vector<Mat> imgs;
    vector<Mat> imgCoordinates;
    
    if(argc<2){
        //        cImg1=imread("/Users/t/Desktop/a1.jpg");
        //        cImg2=imread("/Users/t/Desktop/a2.jpg");
        cout<<"Not enough parameters. Image directory is needed."<<endl;
        exit(0);
    }
    else {
        //        path=argv[1];
        path=argv[1];
        vector<string> imagePaths=dir.GetListFiles(path);
        if(imagePaths.size()==0) {
            cout<<"There's no image in this directory path. Program exits."<<endl;
            exit(0);
        }
        for(int i=0;i<imagePaths.size();i++) {
            if (imagePaths[i]==".DS_Store") {
                continue;
            }
            Mat img=imread(path+imagePaths[i], CV_LOAD_IMAGE_COLOR);
            if(img.empty()){
                cout<<"Could not read image \""<<imagePaths[i]<<"\""<<endl;
                exit(0);
            }
            cout<<"channel="<<img.channels()<<", depth="<<img.depth()<<", size=("<<img.rows<<" * "<<img.cols<<")"<<endl;
            imgs.push_back(img);
            Mat coordinate=Mat::zeros(img.rows, img.cols, CV_32FC2);
            for (int r=0;r<img.rows;r++) for (int c=0;c<img.cols;c++)
            {
                coordinate.at<Vec2f>(r,c)[0]=c;
                coordinate.at<Vec2f>(r,c)[1]=r;
            }
            imgCoordinates.push_back(coordinate);
        }
        for (int i=0; i<imgs.size(); i++) {
            matchz.push_back(vector< vector<DMatch> >(imgs.size()));
            inlierNumBetweenMatch.push_back(vector<int>(imgs.size())); //initialized with zero?
            homographyz.push_back(vector<Mat>(imgs.size())); //initialized with zero?
            maskz.push_back(vector< vector<uchar> >(imgs.size()));
            pMatchOnImg.push_back(vector< pair<int,int> >());
            //            kImageMatch.push_back(vector<int>());
        }
        cout<<homographyz[0][0]<<endl; //
    }
    
    cout << "OpenCV version: " << CV_VERSION <<  endl;
    
    // create a window
    namedWindow(winName, CV_WINDOW_AUTOSIZE);
    moveWindow(winName, 100, 100);
    resizeWindow(winName, 1200, 200);
    
    //    createTrackbar(thresholdString, winName, &threshold, 255, reflesh);
    
    cout<<"-------My version-------"<<endl;
    
//    float fs[10]={897.93,902.182,892.507,897.956,901.85,904.945,894.699,0,0};
//    for(int i=0;i<imgs.size();i++)
//    {
//        float f=1700;
//        Mat img=imgs[i];
//        Mat simg=Mat::zeros(imgs[i].rows, img.cols, CV_8UC3);
//        int ctoir=img.rows/2;
//        int ctoic=img.cols/2;
//        float alpha=f*3/M_PI;
//        for(int r=0;r<img.rows;r++) for(int c=0;c<img.cols;c++) {
//            float x=c-img.cols/2.0;
//            float y=img.rows/2.0-r;
//            int xt=round(atan(x/f)*alpha+ctoic);
//            int yt=round(-(y/sqrt(pow(f,2)+pow(x,2)))*alpha+ctoir);
//            simg.at<Vec3b>((int)yt,(int)xt)=img.at<Vec3b>(r,c);
//        }
//        imgs[i]=simg;
//    }
    
    //feature matching
    for (int i=0; i<imgs.size(); i++) {
        vector<KeyPoint> kps;
        Mat featureDescriptions;
        sift(imgs[i], Mat(), kps, featureDescriptions);
        kpz.push_back(kps);
        fdz.push_back(featureDescriptions);
    }
    
    for (int i=0; i<imgs.size()-1; i++) {
        for (int j=i+1; j<imgs.size(); j++) {
            vector<DMatch> z;
            matchz[i][j]=z;
            vector< vector<DMatch> > matches;
            matcher.knnMatch(fdz[i], fdz[j], matches, K_NEAREST_NEIGHBOURS);
            for (int k=0; k<matches.size(); k++) {
                DMatch m1=matches[k][0];
                DMatch m2=matches[k][1];
                if (m1.distance<MATCH_ACCEPT_DISTANCE_RATO*m2.distance)
                    matchz[i][j].push_back(m1);
                else if(m2.distance<MATCH_ACCEPT_DISTANCE_RATO*m1.distance)
                    matchz[i][j].push_back(m2);
            }
            matchz[j][i]=matchz[i][j];
            pMatchOnImg[i].push_back(pair<int,int>(j,matchz[i][j].size()));
            pMatchOnImg[j].push_back(pair<int,int>(i,matchz[i][j].size()));
        }
        sort(pMatchOnImg[i].begin(),pMatchOnImg[i].end(),cmpMatchedNum);
    }
    
    //image matching
    for (int i=0; i<pMatchOnImg.size(); i++) {
        for (int j=0; j<pMatchOnImg[i].size() && j<MAX_PROTENTIAL_MATCHED_IMAGE_NUM; j++) {
            if(pMatchOnImg[i][j].second<MIN_MATCHED_POINT_NUM) {
                cout<<"Not enough matching points to find homograph, n="<<pMatchOnImg[i][j].second<<", i="<<i<<", j="<<j<<endl;
                break;
            }
            vector<Point2f> srcPoints;
            vector<Point2f> dstPoints;
            int q=i;
            int t=pMatchOnImg[i][j].first;
            if(q==t) {
                cout<<"error#1 happened!"<<endl;
                continue;
            }
            if(q>t)
                continue;
            for (int k=0; k<matchz[q][t].size(); k++) {
                srcPoints.push_back(kpz[q][matchz[q][t][k].queryIdx].pt);
                dstPoints.push_back(kpz[t][matchz[q][t][k].trainIdx].pt);
            }
            vector<uchar> oa;
            maskz[q][t]=oa;
            //ransac
            //            homographyz[q][t]=cv::findHomography(srcPoints,dstPoints,CV_RANSAC,RANSACREPROJTHRESHOLD,maskz[q][t]); //q to t
            homographyz[q][t]=ransacFindH(srcPoints, dstPoints, RANSACREPROJTHRESHOLD, maskz[q][t]);
            maskz[t][q]=maskz[q][t];
            Mat z;
            homographyz[q][t].convertTo(z, CV_32FC1);
            homographyz[q][t]=z;
            homographyz[t][q]=homographyz[q][t].inv();
            //            cout<<homographyz[q][t]<<endl;
            //            cout<<homographyz[t][q]<<endl;
            
            //probabilistic model verification
            int nf=srcPoints.size();
            int ni=0;
            for(int m=0;m<maskz[q][t].size();m++)
                if(maskz[q][t][m]!=0)
                    ni++;
            if(ni>8.0+0.3*nf) {
                inlierNumBetweenMatch[q][t]=ni;
                inlierNumBetweenMatch[t][q]=ni;
                matchedImg.push_back(pair<int,int>(q,t));
                matchedImg.push_back(pair<int,int>(t,q));
            }
        }
    }
    vector< pair<int,int> > addOrder;
    findHomoGraphyzATSingleImg(homographyzATSingleImg, imgs, homographyz, imgCoordinates, inlierNumBetweenMatch, addOrder);
    
    
    //    vector<detail::MatchesInfo> matchesInfos;
    //    for (int i=0; i<imgs.size(); i++) {
    //        for (int j=0; j<imgs.size(); j++) {
    //            detail::MatchesInfo mi;
    //            mi.src_img_idx=i;
    //            mi.dst_img_idx=j;
    //            mi.matches=matchz[i][j];
    //            mi.inliers_mask=maskz[i][j];
    //            mi.num_inliers=inlierNumBetweenMatch[i][j];
    //            Mat z;
    //            if(i!=j)
    //                homographyz[i][j].convertTo(z, CV_64F);
    //            else
    //                z=Mat::eye(3, 3, CV_64F);
    //            assert(z.type()==CV_64F && z.size()==Size(3,3));
    //            mi.H=z.clone();
    //            bool ifMatch=false;
    //            for (int k=0; k<matchedImg.size(); k++) {
    //                if (matchedImg[k].first==i && matchedImg[k].second==j) {
    //                    ifMatch=true;
    //                    break;
    //                }
    //            }
    //            if(ifMatch)
    //                mi.confidence=1;
    //            else
    //                mi.confidence=inlierNumBetweenMatch[i][j]/(8.0+0.3*matchz[i][j].size());
    //            matchesInfos.push_back(mi);
    //        }
    //    }
    //    vector<detail::ImageFeatures> imageFeatures;
    //    for (int i=0; i<imgs.size(); i++) {
    //        detail::ImageFeatures imf;
    //        imf.img_idx=i;
    //        imf.img_size=Size(imgs[i].cols,imgs[i].rows);
    //        imf.keypoints=kpz[i];
    //        imf.descriptors=fdz[i];
    //        imageFeatures.push_back(imf);
    //    }
    //    vector<detail::CameraParams> cameraParams;
    //    for (int i=0; i<imgs.size(); i++)
    //        cameraParams.push_back(detail::CameraParams());
    //    detail::HomographyBasedEstimator estimator;
    //    estimator(imageFeatures, matchesInfos, cameraParams);
    //    for (size_t i = 0; i < cameraParams.size(); ++i)
    //    {
    //        Mat R;
    //        cameraParams[i].R.convertTo(R, CV_32F);
    //        cameraParams[i].R = R;
    //        cout<<"K :\n " << cameraParams[i].K()<<endl;
    //        cout<<"R :\n " << cameraParams[i].R<<endl;
    //    }
    //
    //    Mat params=Mat::zeros(4*imgs.size(), 1, CV_32FC1);
    //    for (int i=0; i<imgs.size(); i++) {
    //        Mat z=cameraParams[i].K();
    //        Mat k;
    //        z.convertTo(k, CV_32FC1);
    //        float thetaX,thetaY,thetaZ;
    //        thetaZ=atan(cameraParams[i].R.at<float>(1,0)/cameraParams[i].R.at<float>(0,0));
    //        thetaY=atan(-cameraParams[i].R.at<float>(2,0)/sqrt(pow(cameraParams[i].R.at<float>(2,1),2)+pow(cameraParams[i].R.at<float>(2,2),2)));
    //        thetaX=atan(cameraParams[i].R.at<float>(2,1)/cameraParams[i].R.at<float>(2,2));
    //        params.at<float>(i*4,0)=k.at<float>(0,0);
    //        params.at<float>(i*4+1,0)=thetaX;
    //        params.at<float>(i*4+2,0)=thetaY;
    //        params.at<float>(i*4+3,0)=thetaZ;
    //        cout<<thetaX<<endl<<thetaY<<endl<<thetaZ<<endl;
    //    }
    //    LM(imgs, matchz, maskz, matchedImg, kpz, params);
    //    cout<<params<<endl;
    
    //    detail::BundleAdjusterRay bundleAdjuster;
    //    bundleAdjuster.setConfThresh(1.0);
    //    bundleAdjuster(imageFeatures, matchesInfos, cameraParams);
    //
    //    // Find median focal length and use it as final image scale
    //    vector<double> focals;
    //    for (size_t i = 0; i < cameraParams.size(); ++i)
    //    {
    //        cout<<"Camera #:\n" << cameraParams[i].K()<<endl;
    //        focals.push_back(cameraParams[i].focal);
    //    }
    
    
    //    for(int i=0;i<imgCoordinates.size();i++) {
    //        cout<<"before:"<<endl;
    //        cout<<imgCoordinates[i]<<endl;
    //        cout<<homographyzATSingleImg[i]<<endl;
    //        perspectiveTransform(imgCoordinates[i], imgCoordinates[i], homographyzATSingleImg[i]);
    //        cout<<"after:"<<endl;
    //        cout<<imgCoordinates[i]<<endl;
    //    }
    //    for(int i=0;i<imgCoordinates.size();i++) {
    //        warpPerspective(imgs[i], imgs[i], homographyzATSingleImg[i], Size(imgs[i].cols*10,imgs[i].rows));
    //        imshow(to_string(i), imgs[i]);
    //    }
    
    Mat panorama=getPanoramaImage(homographyzATSingleImg, imgCoordinates, imgs, addOrder);
    imwrite(path+"panorama.png", panorama);
    //assume all image from same panoramas
    
    cout<<"\n --------Finish!-------"<<endl;
    
    
    // create the image pyramid
    
    // enter the keyboard event loop
    
    
    return 0;
}