#include <iostream>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <math.h>
#include <cmath>
#include <string>
#include <fstream>
#include <vector>
#include <deque>
#include "Timer.h"

using namespace cv;
using namespace std;


//
// global variables
//

int intervalSize=5;
int timeGaussianSize=5;
int windowSize=30;
double timeInterval=0.01;
float sigma=timeGaussianSize/5.0;
int dBlurSize=5;
float singularThreshold=0.001;
int st=0;
int confidenceParameter=20;

int refreshFlag=1; // indicate whether the display needs to be refreshed

void xDerivative(Mat &src, Mat &z)
{
    float x[9]={-1,0,1,-1,0,1,-1,0,1};
    Mat kx=Mat(3,3,CV_32FC1,x);
    src.convertTo(z, CV_32FC1);
    GaussianBlur(z, z, Size(dBlurSize,dBlurSize),dBlurSize/5.0);
    filter2D(z, z, -1, kx);
    z/=3;
}

void yDerivative(Mat &src, Mat &z)
{
    float y[9]={-1,-1,-1,0,0,0,1,1,1};
    Mat ky=Mat(3,3,CV_32FC1,y);
    src.convertTo(z, CV_32FC1);
    GaussianBlur(z, z, Size(dBlurSize,dBlurSize),dBlurSize/5.0);
    filter2D(z, z, -1, ky);
    z/=3;
}

Scalar getColor(float depth, float dx, float dy)
{
    Scalar color;
    float angle=atan2(dy,dx);
    while (angle<0)
        angle+=(2*M_PI);
    while (angle>(2*M_PI))
        angle-=(2*M_PI);
    float firstColor;
    float secondColor;
    int sec=floor(angle/(M_PI*2/3));
    assert(angle>=0 && angle<=2*M_PI);
    if(angle<(M_PI/3+sec*M_PI*2/3)) {
        firstColor=255;
        secondColor=255*(angle-sec*M_PI*2/3)/(M_PI/3);
    }
    else {
        firstColor=255*(2-(angle-sec*M_PI*2/3)/(M_PI/3));
        secondColor=255;
    }
    if(depth>1)
        depth=1;
    firstColor*=depth;
    secondColor*=depth;
    //    if(firstColor>255)
    //        firstColor=255;
    //    if(secondColor>255)
    //        secondColor=255;
    switch (sec) {
        case 0:
            color=Scalar(firstColor,secondColor,0);
            break;
        case 1:
            color=Scalar(0,firstColor,secondColor);
            break;
        case 2:
            color=Scalar(secondColor,0,firstColor);
            break;
        default:
            cout<<"error happened in getColor!"<<endl;
            exit(0);
            break;
    }
    return color;
}

void drawGradientVectors(Mat &src, Mat &dst, Mat &UVs, int interval)
{
    dst=Mat::zeros(src.rows, src.cols, CV_8UC3);
    
    //   dst=src.clone();
    Mat norm=UVs*interval;
    float lu,lv;
    for(int r=0;r<dst.rows;r+=interval) for(int c=0;c<dst.cols;c+=interval) {
        lu=norm.at<Vec2f>(r,c).val[0];
        lv=norm.at<Vec2f>(r,c).val[1];
        float depth=sqrt(pow(lu,2)+pow(lv,2))/10;
        Scalar color=getColor(depth,lu,lv);
        line(dst, Point(c,r), Point(c+0.5+lu,r+0.5+lv), color);
    }
}

void drawGradientVectors(Mat &src, Mat &dst, Mat &Us, Mat & Vs, Mat &confident, int interval, vector<Point2f> &ps, bool ifA=true)
{
    dst=Mat::zeros(src.rows, src.cols, CV_8UC3);
    
    //   dst=src.clone();
    Mat normU=Us;
    Mat normV=Vs;
    normU*=interval;
    normV*=interval;
    float lu,lv;
    if(ifA)
    {
        for(int r=0;r<dst.rows;r+=interval) for(int c=0;c<dst.cols;c+=interval) {
            lu=normU.at<float>(r,c);
            lv=normV.at<float>(r,c);
            //            float depth=sqrt(pow(lu,2)+pow(lv,2))/30;
            float depth=((confident.at<float>(r,c)>1)?(1):(confident.at<float>(r,c)))*sqrt(pow(lu,2)+pow(lv,2));
            if(depth<0.1 || (lu==0 && lv==0))
                continue;
            //            cout<<depth<<endl;
            Scalar color=getColor(depth,lu,lv);
            line(dst, Point(c,r), Point(c+0.5+lu,r+0.5+lv), color);
        }
    }
    else
    {
        for (int r=0; r<src.rows; r++) for (int c=0; c<src.cols; c++) {
            int v=src.at<float>(r,c);
            dst.at<Vec3b>(r,c)=Vec3b(v,v,v);
        }
        for(int i=0;i<ps.size();i++) {
            int r=ps[i].y;
            int c=ps[i].x;
            if(r<0 || r>=src.rows || c<0 || c>=src.cols)
                continue;
            lu=normU.at<float>(r,c);
            lv=normV.at<float>(r,c);
            float depth=sqrt(pow(lu,2)+pow(lv,2))/30;
            //            float depth=((confident.at<float>(r,c)>1)?(1):(confident.at<float>(r,c)))*sqrt(pow(lu,2)+pow(lv,2))/30;
            Scalar color=getColor(depth,lu,lv);
            line(dst, Point(c,r), Point(c+0.5+lu,r+0.5+lv), color);
        }
    }
}

void trackbarHandler(int pos, void *data)
{
    refreshFlag=1;
}

void derivativeOfGaussianKernel(double *kernel, int size, double sigma)
{
    assert(size%2==1);
    int mid=size/2;
    for(int i=0;i<size;i++)
        kernel[i]=exp(-(pow((double)(i-mid),2.0)/(2*pow(sigma,2.0))))*(-(i-mid)/pow(sigma,2.0));
}

Mat getIt(deque<Mat> &imgQueue, int kernelSize, double sigma, int deltaTime)
{
    assert(imgQueue.size()%2==1);
    double *kernel=new double[kernelSize];
    derivativeOfGaussianKernel(kernel, kernelSize, sigma);
    Mat z=Mat::zeros(imgQueue[0].rows, imgQueue[0].cols, imgQueue[0].type());
    for (int i=0; i<imgQueue.size(); i++) {
        Mat zz=z+kernel[imgQueue.size()-1-i]*imgQueue[i];
        z=zz;
    }
    Mat zz=z/deltaTime;
    delete kernel;
    //    zz=imgQueue[imgQueue.size()/2+1]-imgQueue[imgQueue.size()/2];
    return zz;
}

void getXYs(Mat &Ix, Mat &Iy, Mat &It, int window, int interval, Mat &Xs, Mat &Ys, Mat &confident, vector<Point2f> &ps, bool ifA)
{
    Mat sumFilter=Mat::ones(window, window, CV_32FC1);
    Mat X=Mat::zeros(It.rows, It.cols, CV_32FC1);
    Mat Y=Mat::zeros(It.rows, It.cols, CV_32FC1);
    for (int r=0; r<X.rows; r++) for (int c=0; c<X.cols; c++) {
        X.at<float>(r,c)=c;
        Y.at<float>(r,c)=r;
    }
    Mat Ix2=Ix.mul(Ix);
    Mat Iy2=Iy.mul(Iy);
    Mat IxIy=Ix.mul(Iy);
    Mat IxIt=Ix.mul(It);
    Mat IyIt=Iy.mul(It);
    filter2D(Ix2, Ix2, -1, sumFilter);
    filter2D(Iy2, Iy2, -1, sumFilter);
    filter2D(IxIy, IxIy, -1, sumFilter);
    filter2D(IxIt, IxIt, -1, sumFilter);
    filter2D(IyIt, IyIt, -1, sumFilter);
    IxIt*=-1;
    IyIt*=-1;
    Xs=Mat::zeros(It.rows, It.cols, CV_32FC1);
    Ys=Mat::zeros(It.rows, It.cols, CV_32FC1);
    confident=Mat::ones(It.rows, It.cols, CV_32FC1);
    float minConfident=FLT_MAX;
    float maxConfident=-1;
    if(ifA)
    {
        for (int r=0; r<It.rows; r+=interval) for (int c=0; c<It.cols; c+=interval) {
            Mat A=Mat::zeros(2, 2, CV_32FC1);
            A.at<float>(0,0)=Ix2.at<float>(r,c);
            A.at<float>(1,1)=Iy2.at<float>(r,c);
            A.at<float>(0,1)=A.at<float>(1,0)=IxIy.at<float>(r,c);
            Mat B=Mat::zeros(2, 1, CV_32FC1);
            B.at<float>(0,0)=IxIt.at<float>(r,c);
            B.at<float>(1,0)=IyIt.at<float>(r,c);
            Mat C;
            solve(A, B, C);
            Xs.at<float>(r,c)=C.at<float>(0,0);
            Ys.at<float>(r,c)=C.at<float>(1,0);
            Mat AT=A.t();
            Mat w,u,vt;
            SVD::compute(AT*A, w, u, vt, SVD::FULL_UV);
            float sglar=w.at<float>(1,0)/w.at<float>(0,0);
            if(sglar<singularThreshold) {
                Xs.at<float>(r,c)=0;
                Ys.at<float>(r,c)=0;
                continue;
            }
            confident.at<float>(r,c)=w.at<float>(1,0);
            if(minConfident>confident.at<float>(r,c))
                minConfident=confident.at<float>(r,c);
            if(maxConfident<confident.at<float>(r,c))
                maxConfident=confident.at<float>(r,c);
        }
    }
    else
    {
        for (int i=0;i<ps.size();i++) {
            int r=ps[i].y;
            int c=ps[i].x;
            if(r<0 || r>=Ix.rows || c<0 || c>=Ix.cols)
                continue;
            Mat A=Mat::zeros(2, 2, CV_32FC1);
            A.at<float>(0,0)=Ix2.at<float>(r,c);
            A.at<float>(1,1)=Iy2.at<float>(r,c);
            A.at<float>(0,1)=A.at<float>(1,0)=IxIy.at<float>(r,c);
            Mat B=Mat::zeros(2, 1, CV_32FC1);
            B.at<float>(0,0)=IxIt.at<float>(r,c);
            B.at<float>(1,0)=IyIt.at<float>(r,c);
            Mat C;
            solve(A, B, C);
            Xs.at<float>(r,c)=C.at<float>(0,0);
            Ys.at<float>(r,c)=C.at<float>(1,0);
            Mat AT=A.t();
            Mat w,u,vt;
            SVD::compute(AT*A, w, u, vt, SVD::FULL_UV);
            float sglar=w.at<float>(1,0)/w.at<float>(0,0);
            if(sglar<singularThreshold) {
                Xs.at<float>(r,c)=0;
                Ys.at<float>(r,c)=0;
                continue;
            }
            confident.at<float>(r,c)=w.at<float>(1,0);
            if(minConfident>confident.at<float>(r,c))
                minConfident=confident.at<float>(r,c);
            if(maxConfident<confident.at<float>(r,c))
                maxConfident=confident.at<float>(r,c);
        }
    }
    confident/=maxConfident;
    minConfident/=maxConfident;
    confident*=confidenceParameter;
    //    cout<<minConfident<<endl;
    //    cout<<minConfident<<"\t"<<maxConfident<<endl;
}

void getUVs(Mat &Ix, Mat &Iy, Mat &It, int window, int interval, Mat &Us, Mat &Vs, Mat &confident, vector<Point2f> &ps, bool ifA=true)
{
    Mat sumFilter=Mat::ones(window, window, CV_32FC1);
    //    Mat z=sumFilter/(window*window);
    //    sumFilter=z;
    Mat X=Mat::zeros(It.rows, It.cols, CV_32FC1);
    Mat Y=Mat::zeros(It.rows, It.cols, CV_32FC1);
    for (int r=0; r<It.rows; r++) for (int c=0; c<It.cols; c++) {
        X.at<float>(r,c)=c;
        Y.at<float>(r,c)=r;
    }
    
    Mat Ix2=Ix.mul(Ix);
    Mat Ix2X=Ix2.mul(X);
    Mat Ix2Y=Ix2.mul(Y);
    Mat IxIy=Ix.mul(Iy);
    Mat Ix2X2=Ix2X.mul(X);
    Mat Ix2Y2=Ix2Y.mul(Y);
    Mat Ix2XY=Ix2X.mul(Y);
    Mat IxIyX=IxIy.mul(X);
    Mat IxIyY=IxIy.mul(Y);
    Mat IxIyX2=IxIyX.mul(X);
    Mat IxIyY2=IxIyY.mul(Y);
    Mat IxIyXY=IxIyX.mul(Y);
    Mat Iy2=Iy.mul(Iy);
    Mat Iy2X=Iy2.mul(X);
    Mat Iy2Y=Iy2.mul(Y);
    Mat Iy2X2=Iy2X.mul(X);
    Mat Iy2Y2=Iy2Y.mul(Y);
    Mat Iy2XY=Iy2X.mul(Y);
    Mat IxIt=Ix.mul(It);
    Mat IyIt=Iy.mul(It);
    Mat IxItX=IxIt.mul(X);
    Mat IxItY=IxIt.mul(Y);
    Mat IyItX=IyIt.mul(X);
    Mat IyItY=IyIt.mul(Y);
    
    Mat sIx2;
    Mat sIx2X;
    Mat sIx2Y;
    Mat sIxIy;
    Mat sIx2X2;
    Mat sIx2Y2;
    Mat sIx2XY;
    Mat sIxIyX;
    Mat sIxIyY;
    Mat sIxIyX2;
    Mat sIxIyY2;
    Mat sIxIyXY;
    Mat sIy2;
    Mat sIy2X;
    Mat sIy2Y;
    Mat sIy2X2;
    Mat sIy2Y2;
    Mat sIy2XY;
    Mat sIxIt;
    Mat sIyIt;
    Mat sIxItX;
    Mat sIxItY;
    Mat sIyItX;
    Mat sIyItY;
    
    filter2D(Ix2, sIx2, -1, sumFilter);
    filter2D(Ix2X, sIx2X, -1, sumFilter);
    filter2D(Ix2Y, sIx2Y, -1, sumFilter);
    filter2D(IxIy, sIxIy, -1, sumFilter);
    filter2D(Ix2X2, sIx2X2, -1, sumFilter);
    filter2D(Ix2Y2, sIx2Y2, -1, sumFilter);
    filter2D(Ix2XY, sIx2XY, -1, sumFilter);
    filter2D(IxIyX, sIxIyX, -1, sumFilter);
    filter2D(IxIyY, sIxIyY, -1, sumFilter);
    filter2D(IxIyX2, sIxIyX2, -1, sumFilter);
    filter2D(IxIyY2, sIxIyY2, -1, sumFilter);
    filter2D(IxIyXY, sIxIyXY, -1, sumFilter);
    filter2D(Iy2, sIy2, -1, sumFilter);
    filter2D(Iy2X, sIy2X, -1, sumFilter);
    filter2D(Iy2Y, sIy2Y, -1, sumFilter);
    filter2D(Iy2X2, sIy2X2, -1, sumFilter);
    filter2D(Iy2Y2, sIy2Y2, -1, sumFilter);
    filter2D(Iy2XY, sIy2XY, -1, sumFilter);
    filter2D(IxIt, sIxIt, -1, sumFilter);
    filter2D(IyIt, sIyIt, -1, sumFilter);
    filter2D(IxItX, sIxItX, -1, sumFilter);
    filter2D(IxItY, sIxItY, -1, sumFilter);
    filter2D(IyItX, sIyItX, -1, sumFilter);
    filter2D(IyItY, sIyItY, -1, sumFilter);
    
    Us=Mat::zeros(It.rows, It.cols, CV_32FC1);
    Vs=Mat::zeros(It.rows, It.cols, CV_32FC1);
    confident=Mat::zeros(It.rows, It.cols, CV_32FC1);
    float minConfident=FLT_MAX;
    float maxConfident=-1;
    if(ifA)
    {
        for (int r=0; r<It.rows; r+=interval) for (int c=0; c<It.cols; c+=interval) {
            float x[6][6]={{sIx2.at<float>(r,c), sIx2X.at<float>(r,c), sIx2Y.at<float>(r,c),
                sIxIy.at<float>(r,c), sIxIyX.at<float>(r,c), sIxIyY.at<float>(r,c)},
                {sIx2X.at<float>(r,c), sIx2X2.at<float>(r,c), sIx2XY.at<float>(r,c),
                    sIxIyX.at<float>(r,c), sIxIyX2.at<float>(r,c), sIxIyXY.at<float>(r,c)},
                {sIx2Y.at<float>(r,c), sIx2XY.at<float>(r,c), sIx2Y2.at<float>(r,c),
                    sIxIyY.at<float>(r,c), sIxIyXY.at<float>(r,c), sIxIyY2.at<float>(r,c)},
                {sIxIy.at<float>(r,c), sIxIyX.at<float>(r,c), sIxIyY.at<float>(r,c),
                    sIy2.at<float>(r,c), sIy2X.at<float>(r,c), sIy2Y.at<float>(r,c)},
                {sIxIyX.at<float>(r,c), sIxIyX2.at<float>(r,c), sIxIyXY.at<float>(r,c),
                    sIy2X.at<float>(r,c), sIy2X2.at<float>(r,c), sIy2XY.at<float>(r,c)},
                {sIxIyY.at<float>(r,c), sIxIyXY.at<float>(r,c), sIxIyY2.at<float>(r,c),
                    sIy2Y.at<float>(r,c), sIy2XY.at<float>(r,c), sIy2Y2.at<float>(r,c)}};
            float y[6]={-sIxIt.at<float>(r,c), -sIxItX.at<float>(r,c), -sIxItY.at<float>(r,c),
                -sIyIt.at<float>(r,c), -sIyItX.at<float>(r,c), -sIyItY.at<float>(r,c)};
            Mat A=Mat(6,6,CV_32FC1,x);
            Mat B=Mat(6,1,CV_32FC1,y);
            Mat params;
            solve(A, B, params, DECOMP_NORMAL);
            //            float denominator=0;
            //            denominator+=pow(params.at<float>(1,0),2);
            //            denominator+=pow(params.at<float>(2,0),2);
            //            denominator+=pow(params.at<float>(4,0),2);
            //            denominator+=pow(params.at<float>(5,0),2);
            //            denominator=sqrt(denominator/2);
            //            if(denominator==0)
            //                continue;
            
            float rot[2][2]={{params.at<float>(1,0),params.at<float>(2,0)},
                {params.at<float>(4,0),params.at<float>(5,0)}};
            float trans[2]={params.at<float>(0,0),params.at<float>(3,0)};
            Mat rotation=Mat(2,2,CV_32FC1,rot);
            Mat translation=Mat(2,1,CV_32FC1,trans);
            Mat xy=Mat(2,1,CV_32FC1);
            xy.at<float>(0,0)=c;
            xy.at<float>(1,0)=r;
            Mat uv=translation+rotation*xy;
            Us.at<float>(r,c)=uv.at<float>(0,0);
            Vs.at<float>(r,c)=uv.at<float>(1,0);
            
            Mat AT=A.t();
            Mat w,u,vt;
            SVD::compute(AT*A, w, u, vt, SVD::FULL_UV);
            //            cout<<w<<endl;
            //            cout<<params<<endl;
            float sglar=w.at<float>(5,0)/w.at<float>(4,0);
            if(sglar<singularThreshold) {
                Us.at<float>(r,c)=0;
                Vs.at<float>(r,c)=0;
                continue;
            }
            confident.at<float>(r,c)=w.at<float>(5,0);
            if(minConfident>confident.at<float>(r,c))
                minConfident=confident.at<float>(r,c);
            if(maxConfident<confident.at<float>(r,c))
                maxConfident=confident.at<float>(r,c);
            
            //            if(sqrt(pow(Us.at<float>(r,c),2)+pow(Vs.at<float>(r,c),2))>20) {
            //                cout<<A<<endl;
            //                cout<<B<<endl;
            //                cout<<w<<endl;
            //                cout<<params<<endl;
            //                cout<<confident.at<float>(r,c)<<"\t"<<maxConfident<<endl;
            //            }
            
        }
    }
    else
    {
        for (int i=0;i<ps.size();i++) {
            int r=ps[i].y;
            int c=ps[i].x;
            if(r<0 || r>=Ix.rows || c<0 || c>=Ix.cols)
                continue;
            float x[6][6]={{sIx2.at<float>(r,c), sIx2X.at<float>(r,c), sIx2Y.at<float>(r,c),
                sIxIy.at<float>(r,c), sIxIyX.at<float>(r,c), sIxIyY.at<float>(r,c)},
                {sIx2X.at<float>(r,c), sIx2X2.at<float>(r,c), sIx2XY.at<float>(r,c),
                    sIxIyX.at<float>(r,c), sIxIyX2.at<float>(r,c), sIxIyXY.at<float>(r,c)},
                {sIx2Y.at<float>(r,c), sIx2XY.at<float>(r,c), sIx2Y2.at<float>(r,c),
                    sIxIyY.at<float>(r,c), sIxIyXY.at<float>(r,c), sIxIyY2.at<float>(r,c)},
                {sIxIy.at<float>(r,c), sIxIyX.at<float>(r,c), sIxIyY.at<float>(r,c),
                    sIy2.at<float>(r,c), sIy2X.at<float>(r,c), sIy2Y.at<float>(r,c)},
                {sIxIyX.at<float>(r,c), sIxIyX2.at<float>(r,c), sIxIyXY.at<float>(r,c),
                    sIy2X.at<float>(r,c), sIy2X2.at<float>(r,c), sIy2XY.at<float>(r,c)},
                {sIxIyY.at<float>(r,c), sIxIyXY.at<float>(r,c), sIxIyY2.at<float>(r,c),
                    sIy2Y.at<float>(r,c), sIy2XY.at<float>(r,c), sIy2Y2.at<float>(r,c)}};
            float y[6]={-sIxIt.at<float>(r,c), -sIxItX.at<float>(r,c), -sIxItY.at<float>(r,c),
                -sIyIt.at<float>(r,c), -sIyItX.at<float>(r,c), -sIyItY.at<float>(r,c)};
            Mat A=Mat(6,6,CV_32FC1,x);
            Mat B=Mat(6,1,CV_32FC1,y);
            Mat params;
            solve(A, B, params, DECOMP_NORMAL);
            
            float rot[2][2]={{params.at<float>(1,0),params.at<float>(2,0)},
                {params.at<float>(4,0),params.at<float>(5,0)}};
            float trans[2]={params.at<float>(0,0),params.at<float>(3,0)};
            Mat rotation=Mat(2,2,CV_32FC1,rot);
            Mat translation=Mat(2,1,CV_32FC1,trans);
            Mat xy=Mat(2,1,CV_32FC1);
            xy.at<float>(0,0)=c;
            xy.at<float>(1,0)=r;
            Mat uv=translation+rotation*xy;
            Us.at<float>(r,c)=uv.at<float>(0,0);
            Vs.at<float>(r,c)=uv.at<float>(1,0);
            
            Mat AT=A.t();
            Mat w,u,vt;
            SVD::compute(AT*A, w, u, vt, SVD::FULL_UV);
            float sglar=w.at<float>(5,0)/w.at<float>(4,0);
            if(sglar<singularThreshold) {
                Us.at<float>(r,c)=0;
                Vs.at<float>(r,c)=0;
                continue;
            }
            confident.at<float>(r,c)=w.at<float>(5,0);
            if(minConfident>confident.at<float>(r,c))
                minConfident=confident.at<float>(r,c);
            if(maxConfident<confident.at<float>(r,c))
                maxConfident=confident.at<float>(r,c);
        }
    }
    confident/=maxConfident;
    confident*=confidenceParameter;
    //    cout<<minConfident<<"\t"<<maxConfident<<endl;
}

void help(void)
{
    cout << endl;
    cout << "Usage:" << endl;
    cout << endl;
    cout << "Key summary:" << endl;
    cout << "---------------------------------" << endl;
    cout << endl;
    cout << "<ESC>: quit" << endl;
    cout << "'i' - reload the original image (i.e. cancel any previous processing)" << endl;
    cout << "'a' - dense optical flow - affine algorithm" << endl;
    cout << "'A' - optical flow on corner - affine algorithm" << endl;
    cout << "'l' - dense optical flow - Lucas-Kanade algorithm" << endl;
    cout << "'L' - optical flow on corner- Lucas-Kanade algorithm" << endl;
    cout << "'o' - dense optical flow - Gunnar Farneback algorithm by OpenCV" << endl;
    cout << "'d' - difference between two images" << endl;
    cout << "'c' - only show corners" << endl;
    cout << "'p' - pause/restart" << endl;
    cout << "'q' - quit" << endl;
    cout << "'h' - Display a short description of the program, its command line arguments, and the keys it supports." << endl;
    cout << endl;
}


int main(int argc, char *argv[])
{
    string winName="win";
    
    int displayMode=9;   // the current display mode
    int key;
    int i,j;
    
    VideoCapture cap;
    bool ifUseCamera=false;
    
    Mat inImg;
    Mat outImg;
    Mat grayImg;
    Mat lastInImg;
    deque<Mat> imgQueue;
    
    vector<Point2f> corners;
    vector<Point2f> emptyCorners;
    
    Timer timer;
    
    bool ifPause=false;
    
    // capture an image from a camera or read it from a file
    
    cap=VideoCapture(0);
    if(!cap.isOpened())  {
        cout<<"Cannot open camera!"<<endl;
        exit(0);
    }
    ifUseCamera=true;
    
    cout << "OpenCV version: " << CV_VERSION <<  endl;
    
    // create a window with three trackbars
    namedWindow(winName, CV_WINDOW_AUTOSIZE);
    moveWindow(winName, 100, 100);
    resizeWindow(winName, 1200, 100);
    createTrackbar("interval size", winName, &intervalSize, 101, trackbarHandler);
    createTrackbar("gaussian size", winName, &timeGaussianSize, 11, trackbarHandler);
    createTrackbar("window size", winName, &windowSize, 101, trackbarHandler);
    createTrackbar("singular threshold", winName, &st, 300, trackbarHandler);
    createTrackbar("confidence parameter", winName, &confidenceParameter, 300, trackbarHandler);
    
    // create the image pyramid
    
    
    // enter the keyboard event loop
    timer.start();
    while(1){
        if(ifUseCamera) {
            cap>>inImg;
            if (inImg.empty()){
                cout << "Could not grab image" << endl;
                exit(0);
            }
            //            for(i=0;i<cimg.rows;i++)    for(j=0;j<cimg.cols;j++)    for(k=0;k<cimg.channels();k++)
            //                cimg.at<Vec3b>(i,j).val[k]=255-cimg.at<Vec3b>(i,j).val[k];
        }
        key=cvWaitKey(10); // wait 10 ms for a key
        if(key==27) break;
        switch(key){
            case 'i':
                cout<<"i"<<endl;
                displayMode=7;
                refreshFlag=1;
                break;
            case 'A':
                cout<<"A"<<endl;
                displayMode=5;
                refreshFlag=1;
                break;
            case 'a':
                cout<<"a"<<endl;
                displayMode=1;
                refreshFlag=1;
                break;
            case 'L':
                cout<<"L"<<endl;
                displayMode=8;
                refreshFlag=1;
                break;
            case 'l':
                cout<<"l"<<endl;
                displayMode=9;
                refreshFlag=1;
                break;
            case 'c':
                cout<<"c"<<endl;
                displayMode=4;
                refreshFlag=1;
                break;
            case 'o':
                cout<<"o"<<endl;
                displayMode=6;
                refreshFlag=1;
                break;
            case 'd':
                cout<<"d"<<endl;
                displayMode=2;
                refreshFlag=1;
                break;
            case 'p':
                cout<<"p"<<endl;
                ifPause=!ifPause;
                break;
            case 'h':
                help();
                break;
            case 'q':
                exit(0);
                break;
        }
        
        if(ifPause)
            continue;
        
        //        while (timer.getElapsedTimeInSec()<timeInterval);
//        cout<<timer.getElapsedTime()<<endl;
        timer.stop();
        timer.start();
        
        
        intervalSize=(intervalSize==0)?(1):(intervalSize);
        timeGaussianSize=(timeGaussianSize%2==1)?(timeGaussianSize):(timeGaussianSize+1);
        windowSize=(windowSize%2==1)?(windowSize):(windowSize+1);
        singularThreshold=st/300.0;
        // update the display as necessary
        if(refreshFlag || ifUseCamera){
            Mat z;
            resize(inImg, inImg, Size(inImg.cols/3, inImg.rows/3));
            //            cout<<inImg.size()<<endl;
            cvtColor(inImg, grayImg, CV_BGR2GRAY);
            grayImg.convertTo(z, CV_32FC1);
            imgQueue.push_back(z);
            if(imgQueue.size()<timeGaussianSize+1)
                continue;
            while(imgQueue.size()>timeGaussianSize)
                imgQueue.pop_front();
            
            refreshFlag=0;
            //      cout<<cimg.channels()<<endl<<cimg.rows<<endl<<cimg.cols<<endl<<cimg.type()<<endl;
            Mat Ix,Iy;
            Mat img;
            Mat k;
            Mat Us,Vs;
            Mat confident;
            Mat It;
            switch(displayMode){
                case 1:
                    img=imgQueue[imgQueue.size()/2];
                    xDerivative(img, Ix);
                    yDerivative(img, Iy);
                    It=getIt(imgQueue, timeGaussianSize, sigma, 1);
                    getUVs(Ix, Iy, It, windowSize, intervalSize, Us, Vs, confident, emptyCorners, true);
                    drawGradientVectors(img, outImg, Us, Vs, confident, intervalSize, emptyCorners, true);
                    outImg.at<Vec3b>(windowSize-1,windowSize-1)=Vec3b(255,255,255);
                    imshow("camera",outImg);
                    break;
                case 8:
                    img=imgQueue[imgQueue.size()/2];
                    xDerivative(img, Ix);
                    yDerivative(img, Iy);
                    It=getIt(imgQueue, timeGaussianSize, sigma, 1);
                    corners.clear();
                    goodFeaturesToTrack(img,corners,300,0.1,1);
                    img.convertTo(z, CV_8UC1);
                    cornerSubPix(z,corners,Size(11,11),Size(-1,-1),TermCriteria(TermCriteria::COUNT+TermCriteria::EPS,30,0.1));
                    getXYs(Ix, Iy, It, windowSize, intervalSize, Us, Vs, confident, corners, false);
                    drawGradientVectors(img, outImg, Us, Vs, confident, intervalSize, corners, false);
                    for (i=0; i<corners.size(); i++)
                        circle( outImg, corners[i], 3,Scalar(0,0,0), 1 );
                    imshow("camera",outImg);
                    break;
                case 9:
                    img=imgQueue[imgQueue.size()/2];
                    xDerivative(img, Ix);
                    yDerivative(img, Iy);
                    It=getIt(imgQueue, timeGaussianSize, sigma, 1);
                    getXYs(Ix, Iy, It, windowSize, intervalSize, Us, Vs, confident, emptyCorners, true);
                    drawGradientVectors(img, outImg, Us, Vs, confident, intervalSize, emptyCorners, true);
                    outImg.at<Vec3b>(windowSize-1,windowSize-1)=Vec3b(255,255,255);
                    imshow("camera",outImg);
                    
                    break;
                case 6:
                    img=imgQueue[imgQueue.size()/2];
                    calcOpticalFlowFarneback(imgQueue[imgQueue.size()/2],imgQueue[imgQueue.size()/2+1],z,0.5,1,windowSize,1,5,1.1, OPTFLOW_FARNEBACK_GAUSSIAN);
                    drawGradientVectors(img, outImg, z, intervalSize);
                    imshow("camera",outImg);
                    break;
                case 2:
                    It=getIt(imgQueue, timeGaussianSize, sigma, 1);
                    It.convertTo(outImg, CV_8UC1);
                    //                    outImg=inImg.clone();
                    imshow("camera",outImg);
                    break;
                case 7:
                    img=imgQueue[imgQueue.size()/2];
                    img.convertTo(outImg, CV_8UC1);
                    imshow("camera",outImg);
                    break;
                case 4:
                    img=imgQueue[imgQueue.size()/2];
                    corners.clear();
                    goodFeaturesToTrack(img,corners,300,0.1,1);
                    img.convertTo(z, CV_8UC1);
                    img.convertTo(outImg, CV_8UC1);
                    cornerSubPix(z,corners,Size(11,11),Size(-1,-1),TermCriteria(TermCriteria::COUNT+TermCriteria::EPS,30,0.1));
                    for (i=0; i<corners.size(); i++) {
                        circle( outImg, corners[i], 3,  Scalar(0,0,0), 1 );
                    }
                    imshow("camera",outImg);
                    break;
                case 5:
                    img=imgQueue[imgQueue.size()/2];
                    xDerivative(img, Ix);
                    yDerivative(img, Iy);
                    It=getIt(imgQueue, timeGaussianSize, sigma, 1);
                    corners.clear();
                    goodFeaturesToTrack(img,corners,300,0.1,1);
                    img.convertTo(z, CV_8UC1);
                    cornerSubPix(z,corners,Size(11,11),Size(-1,-1),TermCriteria(TermCriteria::COUNT+TermCriteria::EPS,30,0.1));
                    getUVs(Ix, Iy, It, windowSize, intervalSize, Us, Vs, confident, corners, false);
                    drawGradientVectors(img, outImg, Us, Vs, confident, intervalSize, corners, false);
                    for (i=0; i<corners.size(); i++)
                        circle( outImg, corners[i], 3,Scalar(0,0,0), 1 );
                    imshow("camera",outImg);
                    break;
            }
        }
        
    }
    
    // release the images
    inImg.release();
    return 0;
}



////////////////////////////////////////////////////////////////////////
// EOF
////////////////////////////////////////////////////////////////////////