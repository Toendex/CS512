////////////////////////////////////////////////////////////////////////
//
// cs512 assignment1 skeleton using opencv (C++) Spring 2014
//
////////////////////////////////////////////////////////////////////////
//#include "stdafx.h"  // for windows os, you may need this header.
#include <math.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

using namespace cv;
using namespace std;


//
// global variables
//

int refreshFlag=1; // indicate whether the display needs to be refreshed



// YOU MAY ADD MORE FUNCTIONS IF YOU NEED

void convertGrayscale(Mat &src,Mat &dst)
{
    dst=Mat::zeros(src.rows,src.cols,CV_8UC1);
    for(int i=0;i<dst.rows;i++) for(int j=0;j<dst.cols;j++)
        dst.at<uchar>(i,j)=0.5+src.at<Vec3b>(i,j).val[2]*0.299+src.at<Vec3b>(i,j).val[1]*0.587+src.at<Vec3b>(i,j).val[0]* 0.114;
}

void singleColor(Mat &src, Mat &dst, int c)
{
    dst=Mat::zeros(src.rows, src.cols, src.type());
    for(int i=0;i<src.rows;i++) for(int j=0;j<src.cols;j++)
        dst.at<Vec3b>(i,j).val[c]=src.at<Vec3b>(i,j).val[c];
}

void xDerivative(Mat &src, Mat &z)
{
    double x[9]={-1,0,1,-1,0,1,-1,0,1};
    Mat kx=Mat(3,3,CV_64FC1,x);
    for(int i=0;i<src.rows;i++) for(int j=0;j<src.cols;j++)
        z.at<double>(i,j)=src.at<uchar>(i,j);
    filter2D(z, z, -1, kx);
    z+=765;
    z/=6;
}

void xConvFilter(Mat &src, Mat &dst)
{
    dst=src.clone();
    Mat z=Mat(src.rows,src.cols,CV_64FC1);
    xDerivative(src,z);
    for(int i=0;i<src.rows;i++) for(int j=0;j<src.cols;j++)
        dst.at<uchar>(i,j)=z.at<double>(i,j)+0.5;
}

void yDerivative(Mat &src, Mat &z)
{
    double y[9]={-1,-1,-1,0,0,0,1,1,1};
    Mat ky=Mat(3,3,CV_64FC1,y);
    for(int i=0;i<src.rows;i++) for(int j=0;j<src.cols;j++)
        z.at<double>(i,j)=src.at<uchar>(i,j);
    filter2D(z, z, -1, ky);
    z+=765;
    z/=6;
}

void yConvFilter(Mat &src, Mat &dst)
{
    dst=src.clone();
    Mat z=Mat(src.rows,src.cols,CV_64FC1);
    yDerivative(src,z);
    for(int i=0;i<src.rows;i++) for(int j=0;j<src.cols;j++)
        dst.at<uchar>(i,j)=z.at<double>(i,j)+0.5;
}

void myMakeBorder(Mat &src,Mat &dst, int width)
{
    dst=Mat(src.rows+width*2,src.cols+width*2,CV_8UC1);
    for(int i=0;i<src.rows;i++) for(int j=0;j<src.cols;j++)
        dst.at<uchar>(i+width,j+width)=src.at<uchar>(i,j);
    for(int i=0;i<width;i++) for(int j=width;j<src.cols+width;j++)
        dst.at<uchar>(i,j)=dst.at<uchar>(width,j);
    for(int i=0;i<src.rows+width;i++) for(int j=src.cols+width;j<src.cols+2*width;j++)
        dst.at<uchar>(i,j)=dst.at<uchar>(i,src.cols+width-1);
    for(int i=src.rows+width;i<src.rows+width*2;i++) for(int j=width;j<src.cols+width*2;j++)
        dst.at<uchar>(i,j)=dst.at<uchar>(src.rows+width-1,j);
    for(int i=0;i<src.rows+2*width;i++) for(int j=0;j<width;j++)
        dst.at<uchar>(i,j)=dst.at<uchar>(i,width);
}


void myMakeBorder2(Mat &src,Mat &dst, int width)
{
    dst=Mat(src.rows+width*2,src.cols+width*2,CV_64FC1);
    for(int i=0;i<src.rows;i++) for(int j=0;j<src.cols;j++)
        dst.at<double>(i+width,j+width)=src.at<double>(i,j);
    for(int i=0;i<width;i++) for(int j=width;j<src.cols+width;j++)
        dst.at<double>(i,j)=dst.at<double>(width+width-i,j);
    for(int i=0;i<src.rows+width;i++) for(int j=src.cols+width;j<src.cols+2*width;j++)
        dst.at<double>(i,j)=dst.at<double>(i,2*(src.cols+width-1)-j);
    for(int i=src.rows+width;i<src.rows+width*2;i++) for(int j=width;j<src.cols+width*2;j++)
        dst.at<double>(i,j)=dst.at<double>(2*(src.rows+width-1)-i,j);
    for(int i=0;i<src.rows+2*width;i++) for(int j=0;j<width;j++)
        dst.at<double>(i,j)=dst.at<double>(i,2*width-j);
}

void gaussianSmoothing(Mat &src, Mat &dst, int amountSmoothing, double sigma)
{
    src.copyTo(dst);
    int mid=amountSmoothing/2;
    double *kernel=new double[amountSmoothing];
    double total=0;
    for(int i=0;i<amountSmoothing;i++) {
        kernel[i]=exp(-(pow((double)(i-mid),2.0)/(2*pow(sigma,2.0))));
        total+=kernel[i];
    }
    for(int i=0;i<amountSmoothing;i++)
        kernel[i]/=total;
    
    total=0;
    for(int i=0;i<amountSmoothing;i++)
        total+=kernel[i];
    
    
//    Mat kx=Mat(amountSmoothing,1,CV_64FC1,kernel);
//    Mat ky=Mat(1,amountSmoothing,CV_64FC1,kernel);
    Mat zmat;
    
    //  copyMakeBorder(src, zmat, mid, mid, mid, mid, BORDER_REPLICATE);
    
//    Mat zz=Mat(src.rows,src.cols,CV_64FC1);
//    for(int i=0;i<src.rows;i++) for(int j=0;j<src.cols;j++)
//        zz.at<double>(i,j)=src.at<uchar>(i,j);
//    filter2D(zz, zz, -1, kx);
//    filter2D(zz, zz, -1, ky);
//    for(int i=0;i<src.rows;i++) for(int j=0;j<src.cols;j++)
//        dst.at<uchar>(i,j)=zz.at<double>(i,j)+0.5;
    
    Mat zz=Mat(src.rows,src.cols,CV_64FC1);
    for(int i=0;i<src.rows;i++) for(int j=0;j<src.cols;j++)
        zz.at<double>(i,j)=src.at<uchar>(i,j);

    myMakeBorder2(zz,zmat,mid);
    for(int i=0;i<src.rows;i++) for(int j=0;j<src.cols;j++) {
        double z=0;
        for(int k=0;k<amountSmoothing;k++)
            z+=zmat.at<double>(i+mid,j+k)*kernel[k];
        zz.at<double>(i,j)=z;
    }
    myMakeBorder2(zz,zmat,mid);
    for(int i=0;i<src.rows;i++) for(int j=0;j<src.cols;j++) {
        double z=0;
        for(int k=0;k<amountSmoothing;k++)
            z+=zmat.at<double>(i+k,j+mid)*kernel[k];
        dst.at<uchar>(i,j)=z+0.5;
    }
    
    //    zmat=src.clone();
    //    for(int i=0;i<src.rows;i++) for(int j=0;j<src.cols;j++) {
    //        double z=0;
    //        for(int k=0;k<amountSmoothing;k++)
    //            z+=src.at<uchar>(i,min(max(j+k-mid,0),src.cols-1))*kernel[k];
    //        zmat.at<uchar>(i,j)=z;
    //    }
    //    for(int i=0;i<src.rows;i++) for(int j=0;j<src.cols;j++) {
    //        double z=0;
    //        for(int k=0;k<amountSmoothing;k++)
    //            z+=zmat.at<uchar>(min(max(i+k-mid,0),src.rows-1),j)*kernel[k];
    //        dst.at<uchar>(i,j)=z;
    //    }
    
    
    delete kernel;
}

void computeGradientVectors(Mat &src, Mat &dx, Mat &dy)
{
    dx=Mat(src.rows,src.cols,CV_64FC1);
    dy=Mat(src.rows,src.cols,CV_64FC1);
    xDerivative(src,dx);
    yDerivative(src,dy);
}

void computeGradientMagnitude(Mat &src, Mat &dst, Mat &dx, Mat &dy)
{
    dst=Mat(src.rows,src.cols,CV_8UC1);
    for(int i=0;i<dx.rows;i++)  for(int j=0;j<dx.cols;j++)
        dst.at<uchar>(i,j)=sqrt((pow(abs(dx.at<double>(i,j)-127.5),2.0)+pow(abs(dy.at<double>(i,j)-127.5),2.0))*2.0)+0.5;
}

void magnitude(Mat &src, Mat &dst)
{
    Mat dx,dy;
    computeGradientVectors(src,dx,dy);
    computeGradientMagnitude(src,dst,dx,dy);
}

void selectGradientsForDisplay(Mat &grad, Mat &gradMag, float prcnt)
{
    
}

void drawGradientVectors(Mat &src, Mat &dst, Mat &dx, Mat &dy, int N)
{
    dst=Mat::zeros(src.rows,src.cols,CV_8UC3);
 //   dst=src.clone();
    double d=127.5;
    Mat dlx=6*N*(dx-d)/d;
    Mat dly=6*N*(dy-d)/d;
    double lx,ly;
        
    for(int i=0;i<dst.cols;i+=N) for(int j=0;j<dst.rows;j+=N) {
        lx=dlx.at<double>(j,i);
        ly=dly.at<double>(j,i);
        lx+=i;
        ly+=j;
        if(lx<i) {
            if (ly<j)
                line(dst, Point(i,j), Point(lx+0.5,ly+0.5), Scalar(255,0,0));
            else
                line(dst, Point(i,j), Point(lx+0.5,ly+0.5), Scalar(0,255,0));
        }
        else {
            if (ly<j)
                line(dst, Point(i,j), Point(lx+0.5,ly+0.5), Scalar(0,0,255));
            else
                line(dst, Point(i,j), Point(lx+0.5,ly+0.5), Scalar(255,255,255));
        }
    }
}

void plotGradient(Mat &src, Mat &dst, int N)
{
    Mat dx,dy;
    computeGradientVectors(src,dx,dy);
    drawGradientVectors(src, dst, dx, dy, N);
}

void rotateImage(Mat &src, Mat &dst, int theta)
{
    src.copyTo(dst);
	double M_PI=3.1415926;
    double ttheta=2*M_PI*theta/360.0;
    Mat rotate=(Mat_<double>(3,3) << cos(ttheta), -sin(ttheta), 0, sin(ttheta), cos(ttheta), 0, 0, 0, 1);
    Mat trans=(Mat_<double>(3,3) << 1, 0, -dst.rows/2, 0, 1, -dst.cols/2, 0, 0, 1);
    Mat rtrans,rrotate;
    invert(trans, rtrans);
    invert(rotate, rrotate);
    int nn=dst.rows*dst.cols;
    Mat tp=Mat(3,nn,CV_64FC1);
    for(int i=0;i<nn;i++) {
        tp.at<double>(0,i)=(int)i/dst.cols;
        tp.at<double>(1,i)=i%dst.cols;
        tp.at<double>(2,i)=1;
    }
    Mat mtp=rtrans*rrotate*trans*tp;
    int r,c;
    for(int i=0;i<nn;i++) {
        r=mtp.at<double>(0,i)+0.5;
        c=mtp.at<double>(1,i)+0.5;
        if(r<0 || c<0 || r>=dst.rows || c>=dst.cols)
            dst.at<Vec3b>(i/dst.cols,i%dst.cols)=0;
        else
            dst.at<Vec3b>(i/dst.cols,i%dst.cols)=src.at<Vec3b>(r,c);
    }
}


void trackbarHandlerSM(int pos, void *data)
{
    refreshFlag=1;
}

void trackbarHandlerPP(int pos, void *data)
{
    refreshFlag=1;
}

void trackbarHandlerRA(int pos, void *data)
{
    refreshFlag=1;
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
    cout << "'i' - reload the original image (i.e. cancel any previous processing)" << endl;
    cout << "'w' - save the current (possibly processed) image into the file 'out.jpg'" << endl;
    cout << "'g' - convert the image to grayscale using the openCV conversion function." << endl;
    cout << "'G' - convert the image to grayscale using your implementation of conversion function." << endl;
    cout << "'c' - cycle through the color channels of the image showing a different channel every time the key is pressed." << endl;
    cout << "'s' - convert the image to grayscale and smooth it using the openCV function. Use a track bar to control the amount of smoothing." << endl;
    cout << "'S' - convert the image to grayscale and smooth it using your function which should perform convolution with a suitable filter. Use a track bar to control the amount of smoothing." << endl;
    cout << "'x' - convert the image to grayscale and perform convolution with an x derivative filter. Normalize the obtained values to the range [0,255]." << endl;
    cout << "'y' - convert the image to grayscale and perform convolution with a y derivative filter. Normalize the obtained values to the range [0,255]." << endl;
    cout << "'m' - show the magnitude of the gradient normalized to the range [0,255]. The gradient is computed based on the x and y derivatives of the image." << endl;
    cout << "'p' - convert the image to grayscale and plot the gradient vectors of the image every N pixels and let the plotted gradient vectors have a length of K. Use a track bar to control N. Plot the vectors as short line segments of length K." << endl;
    cout << "'r' - convert the image to grayscale and rotate it using an angle of Q degrees. Use a track bar to control the rotation angle. The rotation of the image should be performed using an inverse map so there are no holes in it." << endl;
    cout << "'h' - Display a short description of the program, its command line arguments, and the keys it supports." << endl;
    cout << endl;
}


int main(int argc, char *argv[])
{
    string winName="win";
    string smoothString="Smoothing";
    string angleString="Rotation angle Q";
    string pixelString="N pixels";
    
    int displayMode=1;   // the current display mode
    int amountSmoothing;
    int pixelPercentage; // the percentage of pixels in which the
    // image gradient should be displayed
    int rotAngle;
    double sigma=1.0;     // the standard deviation of the Gaussian filter
    
    Mat cimg;  // the original color image
    Mat gimg;  // the grayscale image
    Mat scimg; // single channel image: show R,G,B once a time
    Mat ximg;  // result of x convolution
    Mat yimg;  // result of y convolution
    
    Mat grad;      // the gradient map
    Mat gradMag;   // the gradient magnitude
    
    Mat rimg;  // rotated image
    
    Mat tmpImg;    // temporary image
    Mat outImg;
    int key;
    int nowChannel=2;
    VideoCapture cap;
    bool ifUseCamera=false;
    bool changeColor=false;
    double difference=0;
    
    vector<Mat> channels;
    
    // capture an image from a camera or read it from a file
    if(argc<2){
        cap=VideoCapture(0);
        if(!cap.isOpened())  {  // check if we succeeded
            cout<<"Cannot open camera!"<<endl;
            exit(0);
        }
        ifUseCamera=true;
    }
    else {
        cimg=imread(argv[1], CV_LOAD_IMAGE_COLOR);
        if(cimg.empty()){
            cout << "Could not read image" << endl;
            exit(0);
        }
    }
    
    
 //   cimg=imread("/Users/t/Desktop/a.jpg");
    cout << "OpenCV version: " << CV_VERSION <<  endl;
    
    
    
    // check the read image
    
    // create a window with three trackbars
    namedWindow(winName, CV_WINDOW_AUTOSIZE);
    moveWindow(winName, 100, 100);
    
    amountSmoothing=11;
    createTrackbar(smoothString, winName, &amountSmoothing, 100, trackbarHandlerSM);
    
    pixelPercentage=10;
    createTrackbar(pixelString, winName, &pixelPercentage ,20, trackbarHandlerPP);
    
    rotAngle=0;
    createTrackbar(angleString, winName, &rotAngle ,360, trackbarHandlerRA);
    
    // create the image pyramid
    
    
    // enter the keyboard event loop
    while(1){
        
        if(ifUseCamera) {
            cap>>cimg;
            if (cimg.empty()){
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
                displayMode=1;
                refreshFlag=1;
                break;
            case 'g':
                displayMode=2;
                refreshFlag=1;
                break;
            case 'G':
                displayMode=3;
                refreshFlag=1;
                break;
            case 'c':
                displayMode=4;
                refreshFlag=1;
                changeColor=true;
                break;
            case 's':
                displayMode=5;
                refreshFlag=1;
                break;
            case 'S':
                displayMode=6;
                refreshFlag=1;
                break;
            case 'x':
                displayMode=7;
                refreshFlag=1;
                break;
            case 'y':
                displayMode=8;
                refreshFlag=1;
                break;
            case 'm':
                displayMode=9;
                refreshFlag=1;
                break;
            case 'p':
                displayMode=10;
                refreshFlag=1;
                break;
            case 'r':
                displayMode=11;
                refreshFlag=1;
                break;
            case 'd':
                displayMode=12;
                refreshFlag=1;
                break;
            case 'w':
                system("pwd");
                if(!imwrite("./out.jpg",outImg)) cout << "file save error" << endl;
                break;
            case 'h':
                help();
                break;
            case 'q':
                exit(0);
                break;
        }
        
        // update the display as necessary
        if(refreshFlag || ifUseCamera){
            refreshFlag=0;
            //      cout<<cimg.channels()<<endl<<cimg.rows<<endl<<cimg.cols<<endl<<cimg.type()<<endl;
            switch(displayMode){
                case 1:
                    outImg=cimg.clone();    //clone current image to output image
                    imshow(winName,outImg);    //show image
                    break;
                case 2:
                    cvtColor(cimg, outImg, CV_BGR2GRAY);
                    imshow(winName,outImg);    //show grayscale image
                    break;
                case 3:
                    convertGrayscale(cimg, outImg);
                    imshow(winName,outImg);    //show grayscale image
                    cvtColor(cimg, ximg, CV_BGR2GRAY);
                    ximg=abs(ximg-outImg);
              /*      difference=0;
                    for(int i=0;i<ximg.rows;i++)    for(int j=0;j<ximg.cols;j++)
                        difference+=ximg.at<uchar>(i,j);
                    cout<<"difference:"<<difference<<endl;*/
                    break;
                case 4:
                    if(changeColor){
                        nowChannel=(nowChannel+1)%3;
                        changeColor=false;
                    }
                    singleColor(cimg,outImg,nowChannel);
                    imshow(winName,outImg);    //show grayscale image
                    break;
                case 5:
                    cvtColor(cimg, outImg, CV_BGR2GRAY);
                    //  medianBlur(outImg, outImg, amountSmoothing/2*2+1);
                    sigma=(amountSmoothing/2*2+1)/5.0;
                    GaussianBlur(outImg, outImg, Size(amountSmoothing/2*2+1,amountSmoothing/2*2+1),sigma);
                    imshow(winName,outImg);    //show image
                    cout<<"filter size:"<<amountSmoothing/2*2+1<<"\t sigma:"<<(amountSmoothing/2*2+1)/5.0<<endl;
                    break;
                case 6:
                    cvtColor(cimg, outImg, CV_BGR2GRAY);
                    sigma=(amountSmoothing/2*2+1)/5.0;
                    gaussianSmoothing(outImg, outImg, amountSmoothing/2*2+1,sigma);
                    imshow(winName,outImg);    //show image
                    cout<<"filter size:"<<amountSmoothing/2*2+1<<"\t sigma:"<<(amountSmoothing/2*2+1)/5.0<<endl;
                    break;
                case 12:
                    cvtColor(cimg, outImg, CV_BGR2GRAY);
                    ximg=outImg.clone();
                    gaussianSmoothing(outImg, outImg, amountSmoothing/2*2+1,(amountSmoothing/2*2+1)/5.0);
                    GaussianBlur(ximg, ximg, Size(amountSmoothing/2*2+1,amountSmoothing/2*2+1),(amountSmoothing/2*2+1)/5.0);
                    outImg=(ximg-outImg)*5+127.5;
                    imshow(winName,outImg);    //show image
                    cout<<"filter size:"<<amountSmoothing/2*2+1<<"\t sigma:"<<(amountSmoothing/2*2+1)/5.0<<endl;
                    break;
                case 7:
                    cvtColor(cimg, outImg, CV_BGR2GRAY);
                    xConvFilter(outImg, outImg);
                    imshow(winName,outImg);    //show image
                    break;
                case 8:
                    cvtColor(cimg, outImg, CV_BGR2GRAY);
                    yConvFilter(outImg, outImg);
                    imshow(winName,outImg);    //show image
                    break;
                case 9:
                    cvtColor(cimg, outImg, CV_BGR2GRAY);
                    //                    channels.clear();
                    //                    split(cimg,channels);
                    //                    for(int i=0;i<3;i++)
                    //                        magnitude(channels[i],channels[i]);
                    //                    merge(channels,outImg);
                    magnitude(outImg,outImg);
                    imshow(winName,outImg);    //show image
                    break;
                case 10:
                    cvtColor(cimg, outImg, CV_BGR2GRAY);
                    plotGradient(outImg,outImg,max(pixelPercentage,1));
                    imshow(winName,outImg);    //show image
                    cout<<"N:"<<pixelPercentage<<endl;
                    break;
                case 11:
                    outImg=Mat::zeros(cimg.rows, cimg.cols, cimg.type());
                    rotateImage(cimg,outImg,rotAngle);
                    imshow(winName,outImg);    //show image
                    cout<<"rotation:"<<rotAngle<<endl;
                    break;
            }
        }
        
    }
    
    // release the images
    cimg.release();
    return 0;
}



////////////////////////////////////////////////////////////////////////
// EOF
////////////////////////////////////////////////////////////////////////