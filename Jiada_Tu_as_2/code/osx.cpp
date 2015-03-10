#include <math.h>
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <iostream>
#include <cmath>
#include "myPoint.h"
#include "Timer.h"

using namespace cv;
using namespace std;


//
// global variables
//

int refreshFlag=1; // indicate whether the display needs to be refreshed
int localMaximumSize=10;
int suppressionWinSize=5;

void reflesh(int pos, void *data)
{
    refreshFlag=1;
}

void trimDerivateByEdge(Mat &src, Mat &dx, Mat &dy)
{
    Mat zmat;
//    Mat grad=Mat(dx.rows,dx.cols,CV_64FC1);
//    double x=0;
//    for(int i=0;i<dx.rows;i++)  for(int j=0;j<dx.cols;j++) {
//        grad.at<double>(i,j)=sqrt(pow(dx.at<double>(i,j),2)+pow(dy.at<double>(i,j),2));
//        if(x<grad.at<double>(i,j))
//            x=grad.at<double>(i,j);
//    }
//    x/=threshold;
//    for(int i=0;i<dx.rows;i++)  for(int j=0;j<dx.cols;j++) {
//        if(x>grad.at<double>(i,j))
//            dx.at<double>(i,j)=dx.at<double>(i,j)=0;
//    }
    Canny(src, zmat, 0, 30);
    for(int i=0;i<zmat.rows;i++)  for(int j=0;j<zmat.cols;j++)
        if(zmat.at<uchar>(i,j)==0)
            dx.at<double>(i,j)=dx.at<double>(i,j)=0;
}

void xDerivative(Mat &src, Mat &z)
{
    double x[9]={-1,0,1,-1,0,1,-1,0,1};
    //    double x[9]={-1,0,1,-2,0,2,-1,0,1};
    Mat kx=Mat(3,3,CV_64FC1,x);
    for(int i=0;i<src.rows;i++) for(int j=0;j<src.cols;j++)
        z.at<double>(i,j)=src.at<uchar>(i,j);
    filter2D(z, z, -1, kx);
    //    z+=765;
    //    z/=6;
}

void yDerivative(Mat &src, Mat &z)
{
    double y[9]={-1,-1,-1,0,0,0,1,1,1};
    //    double y[9]={-1,-2,-1,0,0,0,1,2,1};
    Mat ky=Mat(3,3,CV_64FC1,y);
    for(int i=0;i<src.rows;i++) for(int j=0;j<src.cols;j++)
        z.at<double>(i,j)=src.at<uchar>(i,j);
    filter2D(z, z, -1, ky);
    //    z+=765;
    //    z/=6;
}

void computeGradientVectors(Mat &src, Mat &dx, Mat &dy)
{
    dx=Mat(src.rows,src.cols,CV_64FC1);
    dy=Mat(src.rows,src.cols,CV_64FC1);
    xDerivative(src,dx);
    yDerivative(src,dy);
}

void suppression(Mat &src, Mat &dst, int supWinSize, double largerPercent=0.1)
{
    dst=src.clone();
    int leftupSize=floor(supWinSize/2.0);
    int rightdownSize=floor(supWinSize/2.0)+1;
    int leftBounder;
    int rightBounder;
    int upBounder;
    int downBounder;
    vector<myPoint> ps;
    myPoint *zp=NULL;
    
    largerPercent+=1;
    
    for(int i=0;i<dst.rows;i++) for(int j=0;j<dst.cols;j++)
        if(dst.at<double>(i,j)<0)
            dst.at<double>(i,j)=0;
        else
            ps.push_back(myPoint(i,j,dst.at<double>(i,j)));
    sort(ps.begin(), ps.end(), cmpMyPoint);
    
    for(int i=0;i<ps.size();i++) {
        zp=&ps[i];
        if(dst.at<double>(zp->row,zp->col)<=0)
            continue;
    
        leftBounder=(zp->col-leftupSize)>0?(zp->col-leftupSize):0;
        rightBounder=(zp->col+rightdownSize)<dst.cols?(zp->col+rightdownSize):dst.cols;
        upBounder=(zp->row-leftupSize)>0?(zp->row-leftupSize):0;
        downBounder=(zp->row+rightdownSize)<dst.rows?(zp->row+rightdownSize):dst.rows;
        
        for (int ii=upBounder;ii<downBounder;ii++) for(int jj=leftBounder;jj<rightBounder;jj++) {
            if(ii==zp->row && jj==zp->col)
                continue;
            dst.at<double>(ii,jj)=0;
        }

//        //ANMS
//        for (int ii=upBounder;ii<downBounder;ii++) for(int jj=leftBounder;jj<rightBounder;jj++) {
//            if(ii==zp->row && jj==zp->col)
//                continue;
//            if(zp->value > largerPercent*src.at<double>(ii,jj))
//                continue;
//            else {
//                dst.at<double>(zp->row,zp->col)=0;
//                break;
//            }
//        }
    }
}

void localization(Mat & pointMat, Mat &dst, Mat &dx, Mat &dy, int windowSize)
{
    Mat tmat=Mat::zeros(pointMat.rows,pointMat.cols,pointMat.type());
    Mat ttmat=Mat::zeros(pointMat.rows,pointMat.cols,pointMat.type());
    Mat x=Mat::zeros(pointMat.rows, pointMat.cols, CV_64FC1);
    Mat y=Mat::zeros(pointMat.rows, pointMat.cols, CV_64FC1);
    for(int i=0; i<pointMat.rows; i++) for(int j=0;j<pointMat.cols;j++) {
        x.at<double>(i,j)=j;
        y.at<double>(i,j)=i;
    }
    Mat dxx=dx.mul(dx);
    Mat dxy=dx.mul(dy);
    Mat dyy=dy.mul(dy);
    Mat s1=dxx.mul(x)+dxy.mul(y);
    Mat s2=dxy.mul(x)+dyy.mul(y);
    Mat filter=Mat::ones(windowSize,windowSize,CV_64FC1);
    filter2D(dxx,dxx,-1,filter);
    filter2D(dxy,dxy,-1,filter);
    filter2D(dyy,dyy,-1,filter);
    filter2D(s1,s1,-1,filter);
    filter2D(s2,s2,-1,filter);
    Mat zmat,zsmat,p;
    for(int i=0;i<pointMat.rows;i++) for(int j=0;j<pointMat.cols;j++) {
        if(pointMat.at<double>(i,j)<=0)
            continue;
        zmat=Mat::zeros(2,2,CV_64FC1);
        zmat.at<double>(0,0)=dxx.at<double>(i,j);
        zmat.at<double>(0,1)=zmat.at<double>(1,0)=dxy.at<double>(i,j);
        zmat.at<double>(1,1)=dyy.at<double>(i,j);
        invert(zmat, zmat);
        zsmat=Mat::zeros(2,1,CV_64FC1);
        zsmat.at<double>(0,0)=s1.at<double>(i,j);
        zsmat.at<double>(1,0)=s2.at<double>(i,j);
        p=zmat*zsmat;
        int pc=(p.at<double>(0,0)+0.5);
        int pr=(p.at<double>(1,0)+0.5);
        if(pr>-1 && pr<pointMat.rows && pc>-1 && pc<pointMat.cols) {
            //            cout<<"pr:"<<pr<<"\tpc:"<<pc<<"\tvalue:"<<pointMat.at<double>(i,j)<<endl;
            if(pointMat.at<double>(i,j) < ttmat.at<double>(pr,pc))
                continue;
            tmat.at<double>(pr,pc)=pointMat.at<double>(i,j);
            ttmat.at<double>(pr,pc)=pointMat.at<double>(i,j);
            int m=5;
            for(int ii=pr-m;ii<pr+m+1;ii++)
                for(int jj=pc-m;jj<pc+m+1;jj++) {
                    if(ii==pr && jj==pc)
                        continue;
                    if(ii>-1 && ii<pointMat.rows && jj>-1 && jj<pointMat.cols) {
                        if(ttmat.at<double>(ii,jj)<ttmat.at<double>(pr,pc))
                            tmat.at<double>(ii,jj)=0;
                        else
                            tmat.at<double>(pr,pc)=0;
                    }
                }
        }
    }
    dst=tmat.clone();
}

void detectCorner(Mat &src, Mat &dst, double smooth, double sigma, int neighbor, double weight, bool ifMy)
{
//    cout<<"my corner detection"<<endl;
    Mat dx,dy;
    Mat zmat;
    Mat eigenValues;
    Mat eigenVectors;
    dst=Mat::zeros(src.rows, src.cols, CV_64FC1);
    
    if(ifMy)
    {
        if(smooth!=0)
            GaussianBlur(src, src, Size(smooth,smooth), sigma);
        computeGradientVectors(src, dx, dy);

       //   trimDerivateByEdge(src,dx,dy); //trim derivatives base on edge

       Mat dxx=dx.mul(dx);
       Mat dxy=dx.mul(dy);
       Mat dyy=dy.mul(dy);
       
       Mat filter=Mat::ones(neighbor,neighbor,CV_64FC1);
       filter2D(dxx,dxx,-1,filter);
       filter2D(dxy,dxy,-1,filter);
       filter2D(dyy,dyy,-1,filter);
    
       //harris
       for(int i=0;i<src.rows;i++) for(int j=0;j<src.cols;j++) {
            Mat zmat=Mat::zeros(2,2,CV_64FC1);
            zmat.at<double>(0,0)=dxx.at<double>(i,j);
            zmat.at<double>(0,1)=zmat.at<double>(1,0)=dxy.at<double>(i,j);
            zmat.at<double>(1,1)=dyy.at<double>(i,j);
            double det=determinant(zmat);
            double trace=cv::trace(zmat).val[0];
            dst.at<double>(i,j)=det-weight*trace*trace;
        }
    }
    else {
        Mat zsrc=src.clone();
        if(smooth!=0)
            GaussianBlur(zsrc, zsrc, Size(smooth,smooth), sigma);
        computeGradientVectors(zsrc, dx, dy);
        cornerHarris(src, dst, neighbor, smooth, weight, BORDER_DEFAULT );
        dst.convertTo(dst, CV_64FC1);
    }
    
    suppression(dst, dst, neighbor);
    localization(dst,dst,dx,dy,neighbor);
}

void getFeatures(Mat &cImg, Mat &outImg, vector<KeyPoint> &kp, Mat &featureDescriptions, double weight, double threshold, double smooth, double sigma, double neighbor, bool ifMy=true)
{
    kp.clear();
    Mat gImg;
    Mat zImg,zzImg,zzzImg;
    SiftDescriptorExtractor descriptor;
//    SIFT descriptor;
    int i,j;
    int CN;
    if(cImg.channels()!=1)
        cvtColor(cImg, gImg, CV_BGR2GRAY);
    else
        gImg=cImg.clone();
    outImg=Mat::zeros(gImg.rows, gImg.cols, CV_8UC3);
    for( j = 0; j < gImg.rows ; j++ )   for( i = 0; i < gImg.cols; i++ )
        outImg.at<Vec3b>(j,i)=Vec3b(gImg.at<uchar>(j,i),gImg.at<uchar>(j,i),gImg.at<uchar>(j,i));
    //  GaussianBlur(outImg, outImg, Size(smooth,smooth), sigma);
    detectCorner(gImg, zImg, smooth, sigma, neighbor, weight, ifMy);
    normalize( zImg, zzImg, 0, 255, NORM_MINMAX, CV_32FC1, Mat() );
    convertScaleAbs( zzImg, zzzImg );
    //outImg=Mat::zeros(gImg.rows, gImg.cols, CV_8UC3);
    CN=0;
    Mat dx,dy;
    computeGradientVectors(gImg,dx,dy);
    for( i = 0; i < zzImg.rows ; i++ )   for( j = 0; j < zzImg.cols; j++ ) {
        if( zzzImg.at<uchar>(i,j) > threshold ) {
//          circle( outImg, Point( j, i ), neighbor/2,  Scalar(0,255,0), 1 );
//          rectangle(outImg, Point(j-floor(neighbor/2),i-floor(neighbor/2)),
//                Point(j+ceil(neighbor/2),i+ceil(neighbor/2)), Scalar(0,255,0), 1);
            rectangle(outImg, Point(j-1,i-1),Point(j+1,i+1), Scalar(0,255,0), 2);
            CN++;
            KeyPoint zp=KeyPoint(j,i,16);
            zp.angle=atan2(dx.at<double>(i,j),dy.at<double>(i,j))*180/M_PI;
            if(zp.angle>180 || zp.angle<-180)
                cout<<"ANGLE ERROR!"<<endl;
            if(zp.angle<0)
                zp.angle+=360;
            kp.push_back(zp);
        }
    }
    cout<<"# of corners: "<<CN<<endl;
    descriptor.compute(gImg, kp, featureDescriptions);
//    descriptor.operator()(gImg, Mat(), kp, featureDescriptions);
    cout<<featureDescriptions.rows<<"  "<<featureDescriptions.cols<<endl;
//    cout<<featureDescriptions<<endl;
}

double euclidDistance(const Mat &a, const Mat &b) {
    double sum=0;
    for(int i=0;i<a.cols;i++)
        sum+=pow((a.at<float>(0,i)-b.at<float>(0,i)),2);
    return sqrt(sum);
}

int singleMatch(const Mat &vec1, Mat &descriptions, double successThreshold, double &value) {
    double distance;
    double min=successThreshold+1;
    int matchIndex=-1;
    for(int i=0;i<descriptions.rows;i++) {
        distance=euclidDistance(vec1, descriptions.row(i));
        if(distance<min) {
            min=distance;
            matchIndex=i;
        }
    }
//    cout<<"min: "<<min<<endl;
    if(min>=successThreshold)
        matchIndex=-1;
    value=min;
    return matchIndex;
}

void doMatch(vector<KeyPoint> &kp1, Mat &descriptions1,
             vector<KeyPoint> &kp2, Mat &descriptions2,
             vector< pair<KeyPoint,KeyPoint> > &matchResult, double successThreshold)
{
    matchResult.clear();
    vector< pair<bool,double> > matched;
    for(int i=0;i<kp2.size();i++)
        matched.push_back(pair<bool,double>(false,successThreshold+1));
    int matchIndex=-1;
    double value=successThreshold+1;
    for(int i=0;i<descriptions1.rows;i++) {
        value=successThreshold+1;
        matchIndex=singleMatch(descriptions1.row(i), descriptions2, successThreshold, value);
        if(matchIndex!=-1) {
            if(matched[matchIndex].first==false) {
                matchResult.push_back(pair<KeyPoint,KeyPoint>(kp1[i],kp2[matchIndex]));
                matched[matchIndex].first=true;
                matched[matchIndex].second=value;
            }
            else if(matched[matchIndex].second>value){
                for(int j=0;j<matchResult.size();j++)
                    if(matchResult[j].second.pt.x==kp2[matchIndex].pt.x && matchResult[j].second.pt.y==kp2[matchIndex].pt.y) {
                        matchResult[j].first=kp1[i];
                        matched[matchIndex].second=value;
                        break;
                    }
            }
        }
    }
}

void buildinHarris(Mat &src, Mat &dst, double neighbor, double smooth, double weight, double threshold)
{
    Mat gImg,zImg,zzImg,zzzImg;
    int i,j;
    if(src.channels()!=1)
        cvtColor(src, gImg, CV_BGR2GRAY);
    else
        gImg=src.clone();
    cornerHarris( gImg, zImg, neighbor, smooth, weight, BORDER_DEFAULT );
    normalize( zImg, zzImg, 0, 255, NORM_MINMAX, CV_32FC1, Mat() );
    convertScaleAbs( zzImg, zzzImg );
    dst=Mat::zeros(gImg.rows, gImg.cols, CV_8UC3);
    for( j = 0; j < gImg.rows ; j++ )   for( i = 0; i < gImg.cols; i++ )
        dst.at<Vec3b>(j,i)=Vec3b(gImg.at<uchar>(j,i),gImg.at<uchar>(j,i),gImg.at<uchar>(j,i));
    for( j = 0; j < zzImg.rows ; j++ )   for( i = 0; i < zzImg.cols; i++ ) {
        if( zzzImg.at<uchar>(j,i) > threshold )
            rectangle(dst, Point(i-floor(neighbor/2),j-floor(neighbor/2)),
                      Point(i+ceil(neighbor/2),j+ceil(neighbor/2)), Scalar(0,255,0), 1);
    }

}

void help(void)
{
    cout << endl;
    cout << "Usage: ass1 [<image-file>]" << endl;
    cout << endl;
    cout << "Key summary:" << endl;
    cout << "---------------------------------" << endl;
    cout << endl;
    cout << "<ESC>: quit" << endl;
    cout << "'q' - quit" << endl;
    cout << "'c' - Detect, localize and match the corners of the two image" << endl;
    cout << "'C' - Detect, localize and match the corners of the two image(Using OpenCV build-in Harris Corner Detection)" << endl;
    cout << "'a' - Detect the corner using OpenCV bulid-in function (cornerHarris)" << endl;
    cout << "'h' - Display a short description of the program, its command line arguments, and the keys it supports." << endl;
    cout << endl;
}

int main(int argc, char *argv[])
{
    string winName="win";
    string smoothString="Gaussian kernel Size";
    string neighborString="Neighbor Size";
    string weightString="Weight of Trace(/100)";
    string thresholdString="Corner Threshold";
    string matchString="Match Threshold(*10)";
    
    int i,j,k,key;
    int displayMode=1;   // the current display mode
    int smoothingSize=3;
    int neighborSize=5;
    int weightInt=4;
    int threshold=100;
    int matchThreshold=35;
    
    int neighbor;
    double weight;
    double smooth;
    double sigma;
    
    double zd;

    Mat gImg,zImg,zzImg,zzzImg;
    Mat cImg1,cImg2;
    Mat outImg1,outImg2;
    
    Mat featureDescriptions1, featureDescriptions2;
    vector<KeyPoint> kp1,kp2;
    vector< pair<KeyPoint,KeyPoint> > matchResult;
    BFMatcher matcher=BFMatcher(NORM_L2,true);
    vector<DMatch> matches;
    
    VideoCapture cap;
    bool ifUseCamera=false;
    
    bool ifMy=true;
    
    Timer timer;
    
    // capture an image from a camera or read it from a file
//    if(argc<3){
//        cap=VideoCapture(0);
//        if(!cap.isOpened())  {  // check if we succeeded
//            cout<<"Cannot open camera!"<<endl;
//            exit(0);
//        }
//        ifUseCamera=true;
//    }
//    else {
//        cImg1=imread(argv[1], CV_LOAD_IMAGE_COLOR);
//        if(cImg1.empty()){
//            cout << "Could not read image : " << argv[1] << endl;
//            exit(0);
//        }
//        cImg2=imread(argv[2], CV_LOAD_IMAGE_COLOR);
//        if(cImg2.empty()){
//            cout << "Could not read image2 : " << argv[2] << endl;
//            exit(0);
//        }
//    }
    
    if(argc<3){
//        cImg1=imread("/Users/t/Desktop/a1.jpg");
//        cImg2=imread("/Users/t/Desktop/a2.jpg");
        cout<<"Not enough parameters! Need to input two images."<<endl;
        exit(0);
    }
    else {
        cImg1=imread(argv[1], CV_LOAD_IMAGE_COLOR);
        if(cImg1.empty()){
            cout << "Could not read image1 : " << argv[1] << endl;
            exit(0);
        }
        cImg2=imread(argv[2], CV_LOAD_IMAGE_COLOR);
        if(cImg2.empty()){
            cout << "Could not read image2 : " << argv[2] << endl;
            exit(0);
        }
    }
    
    
//    cImg2=Mat(cImg1.rows,cImg1.cols,cImg1.type());
//    if(cImg1.channels()==1)
//    {
//        for(i=0;i<cImg1.rows;i++) for(j=0;j<cImg1.cols;j++)
//        {
//            cImg2.at<uchar>(i,j)=cImg1.at<uchar>(cImg1.rows-i-1,cImg1.cols-j-1);
//        }
//    }
//    else if(cImg1.channels()==3)
//    {
//        for(i=0;i<cImg1.rows;i++) for(j=0;j<cImg1.cols;j++)
//        {
//            cImg2.at<Vec3b>(i,j)=cImg1.at<Vec3b>(cImg1.rows-i-1,cImg1.cols-j-1);
//        }
//    }
    
    cout << "OpenCV version: " << CV_VERSION <<  endl;
    
    
    Size sz = Size(cImg1.size().width + cImg2.size().width, cImg1.size().height + cImg2.size().height);
    Mat matchingImage;
    
    // check the read image
    
    // create a window with three trackbars
    namedWindow(winName, CV_WINDOW_AUTOSIZE);
    moveWindow(winName, 100, 100);
    resizeWindow(winName, 1200, 200);
    
    //createTrackbar(angleString, winNam
    createTrackbar(thresholdString, winName, &threshold, 255, reflesh);
    createTrackbar(smoothString, winName, &smoothingSize, 31, reflesh);
    createTrackbar(weightString, winName, &weightInt, 50, reflesh);
    createTrackbar(neighborString, winName, &neighborSize, 50, reflesh);
    createTrackbar(matchString, winName, &matchThreshold, 100, reflesh);
    
    // create the image pyramid
    
    cout<<"-------My version-------"<<endl;
    // enter the keyboard event loop
    while(1){
        if(ifUseCamera) {
            cap>>cImg1;
            if (cImg1.empty()){
                cout << "Could not grab image" << endl;
                exit(0);
            }
            cvtColor(cImg1, gImg, CV_BGR2GRAY);
        }

        key=cvWaitKey(10); // wait 10 ms for a key
        if(key==27) break;
        switch(key){
            case 'c':
                displayMode=5;
                ifMy=true;
                cout<<"-------My version-------"<<endl;
                refreshFlag=1;
                break;
            case 'C':
                displayMode=5;
                ifMy=false;
                cout<<"-------OpenCV version-------"<<endl;
                refreshFlag=1;
                break;
            case 'a':
                displayMode=2;
                refreshFlag=1;
                break;
            case 'g':
                displayMode=6;
                refreshFlag=1;
                break;
//            case 'e':
//                displayMode=3;
//                refreshFlag=1;
//                break;
//            case 'd':
//                displayMode=4;
//                refreshFlag=1;
//                break;
            case 'h':
                help();
                break;
            case 'q':
                exit(0);
        }
        
        // update the display as necessary
        if(refreshFlag || ifUseCamera){
            refreshFlag=0;
            weight=weightInt/100.0;
            smooth=(smoothingSize==0)?(0):(smoothingSize/2*2+1);
            neighbor=(neighborSize<2)?(2):(neighborSize);
            sigma=smooth/5.0;
            matchThreshold=matchThreshold*10;
            switch(displayMode){
                case 1:
                    imshow("Image 1",cImg1);
                    imshow("Image 2",cImg2);
                    break;
                case 5:
//                    timer.start();
                    getFeatures(cImg1, outImg1, kp1, featureDescriptions1, weight, threshold, smooth, sigma, neighbor, ifMy);
//                    timer.stop();
//                    cout<<timer.getElapsedTimeInMilliSec()<<"ms"<<endl;
//                    timer.start();
                    getFeatures(cImg2, outImg2, kp2, featureDescriptions2, weight, threshold, smooth, sigma, neighbor, ifMy);
//                    timer.stop();
//                    cout<<timer.getElapsedTimeInMilliSec()<<"ms"<<endl;
                    imshow("Image 1",outImg1);
                    imshow("Image 2",outImg2);
                    
//                    timer.start();
                    doMatch(kp1, featureDescriptions1, kp2, featureDescriptions2, matchResult, matchThreshold);
//                    timer.stop();
//                    cout<<timer.getElapsedTimeInMilliSec()<<"ms"<<endl;
//                    cout<<featureDescriptions1.row(0)<<endl;
//                    cout<<featureDescriptions2.row(0)<<endl;
                    cout<<"matched: "<<matchResult.size()<<endl;
                    matchingImage = Mat::zeros(sz, CV_8UC3);
                    outImg1.copyTo(Mat(matchingImage,Rect(0, 0, outImg1.size().width, outImg1.size().height)));
                    outImg2.copyTo(Mat(matchingImage,Rect(outImg1.size().width, outImg1.size().height,
                                                          outImg1.size().width, outImg1.size().height)));
                    for(i=0;i<matchResult.size();i++) {
   //                     cout<<matchResult[i].first.pt<<"   "<<matchResult[i].second.pt<<endl;
                        line(matchingImage, matchResult[i].first.pt,
                             Point(outImg1.size().width+matchResult[i].second.pt.x,
                                   outImg1.size().height+matchResult[i].second.pt.y), Scalar(0, 255, 255));
                        putText(matchingImage, to_string(i), matchResult[i].first.pt, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,255,0));
                        putText(matchingImage, to_string(i),
                                Point(outImg1.size().width+matchResult[i].second.pt.x,
                                      outImg1.size().height+matchResult[i].second.pt.y), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,255,0));
                    }
                    imshow("result", matchingImage);
                    
//                    cout<<featureDescriptions1.row(0)<<endl;
//                    cout<<featureDescriptions2.row(0)<<endl;
//                    matches.clear();
//                    matcher.match(featureDescriptions1, featureDescriptions2, matches);
//                    drawMatches(outImg1, kp1, outImg2, kp2, matches, matchingImage);
//                    imshow("result", matchingImage);
//                    moveWindow("result", 330, 500);
                    break;
                case 2:
                    buildinHarris(cImg1, outImg1, neighbor, smooth, weight, threshold);
                    buildinHarris(cImg2, outImg2, neighbor, smooth, weight, threshold);
                    imshow("Image 1",outImg1);    //show image
                    imshow("Image 2",outImg2);    //show image
                    break;
                case 3:
                    cvtColor(cImg1, gImg, CV_BGR2GRAY);
                    GaussianBlur(gImg, gImg, Size(smooth,smooth), sigma);
                    Canny(gImg, outImg1, 0, 30);
                    imshow(winName,outImg1);
                    break;
                case 4:
                    cvtColor(cImg1, outImg1, CV_BGR2GRAY);
                    cvtColor(cImg2, outImg2, CV_BGR2GRAY);
                    gImg=Mat(outImg1.rows,outImg1.cols,outImg1.type());
                    zd=0;
                    for(int i=0;i<outImg2.rows;i++) for (int j=0;j<outImg2.cols;j++){
                        gImg.at<uchar>(i,j)=outImg2.at<uchar>(outImg2.rows-i-1,outImg2.cols-j-1);
                        zd+=(abs(gImg.at<uchar>(i,j)-outImg1.at<uchar>(i,j)));
                    }
                    cout<<"difference: "<<zd<<endl;
                    outImg2=abs(outImg1-gImg)*10;
                    imshow("difference",outImg2);
                    break;
                case 6:
                    cvtColor(cImg1, outImg1, CV_BGR2GRAY);
                    GaussianBlur(outImg1, outImg1, Size(smooth,smooth), sigma);
                    cvtColor(cImg2, outImg2, CV_BGR2GRAY);
                    GaussianBlur(outImg2, outImg2, Size(smooth,smooth), sigma);
                    imshow("Image 1",outImg1);
                    imshow("Image 2",outImg2);
                    break;
            }
        }
        
    }
    
    return 0;
}