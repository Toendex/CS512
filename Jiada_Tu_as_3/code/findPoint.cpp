#include <math.h>
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>

#define WIDTH 5
#define HEIGHT 8
#define WIDTH_DIAMETER 20
#define HEIGHT_DIAMETER 20

using namespace std;
using namespace cv;

int refreshFlag=1; // indicate whether the display needs to be refreshed

void help()
{
    cout<<"--------------HELP-------------"<<endl;
    cout<<"Press 'a' to take photo and generate an image point file.\n----No file will be generated if no calibration target is detected."<<endl;
    cout<<"Press 'q' to quit"<<endl;
    cout<<"-------------------------------"<<endl;
}

int main(int argc, char *argv[])
{
    string winName="win";
    
    int i,j,k,key;
    int displayMode=1;   // the current display mode
    
    VideoCapture cap;
    bool ifUseCamera=false;
    Mat cImg;
    vector<Mat> imgs;
    Mat gImg;
    
    string str;
    
    int imgNum=0;
    
    if(argc<4){
        cap=VideoCapture(0);
        if(!cap.isOpened())  {  // check if we succeeded
            cout<<"Cannot open camera!"<<endl;
            exit(0);
        }
        ifUseCamera=true;
    }
    else {
        Mat img;
        for(int i=1;i<4;i++) {
            img=imread(argv[i], CV_LOAD_IMAGE_COLOR);
            if(img.empty()){
                cout << "Could not read image"<<i<<" : " << argv[i] << endl;
                exit(0);
            }
            imgs.push_back(img);
        }
        ifUseCamera=false;
    }
    
    cout << "OpenCV version: " << CV_VERSION <<  endl;
    
    // check the read image
    
    // create a window with three trackbars
    namedWindow(winName, CV_WINDOW_AUTOSIZE);
    moveWindow(winName, 100, 100);
    resizeWindow(winName, 1200, 200);
    
    // create the image pyramid
    
    cout<<"-------My version-------"<<endl;
    // enter the keyboard event loop
    
    cv::Size size(WIDTH,HEIGHT);
    vector<Point2f> chessboardCorners;
    ofstream os;
    bool ifFound;
    
//    os.open("/Users/t/Desktop/test/world.txt",ios::out);
    os.open("./my_world.txt",ios::out);
    os<<WIDTH*HEIGHT<<endl;
    for(i=0;i<HEIGHT;i++)
        for(j=0;j<WIDTH;j++)
            os<<j*WIDTH_DIAMETER<<"\t"<<i*HEIGHT_DIAMETER<<"\t"<<0<<endl;
    os.close();
    cout<<"World file \"./my_world.txt\" is added"<<endl;
    if(!ifUseCamera) {
        for(i=0;i<imgs.size();i++) {
            chessboardCorners.clear();
            cImg=imgs[i];
            ifFound=findChessboardCorners(cImg,size,chessboardCorners);
            if(!ifFound) {
                cout<<"Didn't find chessboard corners("<<WIDTH<<"*"<<HEIGHT<<") in image "<<i<<"."<<endl;
                continue;
            }
            cornerSubPix(gImg,chessboardCorners,Size(11,11),Size(-1,-1),TermCriteria(TermCriteria::COUNT+TermCriteria::EPS,30,0.1));
            
            if(chessboardCorners.size()!=WIDTH*HEIGHT)
                cout<<"Error happened! ChessboardCorners size not match!"<<endl;
//            str=string("/Users/t/Desktop/test/image_")+to_string(i)+string(".txt");
            str=string("./my_image_")+to_string(i)+string(".txt");
            os.open(str.c_str(),ios::out);
            os<<WIDTH*HEIGHT<<endl;
            for(j=0;j<chessboardCorners.size();j++)
                os<<chessboardCorners[j].x<<"\t"<<chessboardCorners[j].y<<endl;
            os.close();
        }
        return 0;
    }
    
    
    
    imgNum=0;
    while(ifUseCamera /*&& imgNum<3*/){
        cap>>cImg;
        if (cImg.empty()){
            cout << "Could not grab image" << endl;
            exit(0);
        }
        
        key=cvWaitKey(10); // wait 10 ms for a key
        if(key==27) break;
        switch(key){
            case 'a':
                displayMode=2;
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
            switch(displayMode){
                case 1:
                    imshow(winName,cImg);    //show image
                    break;
                case 2:
                    displayMode=1;
                    imgs.push_back(cImg);
                    //                    imshow(winName,cImg);    //show image
                    //                    imshow(strs[imgNum-1],cImg);    //show image
                    chessboardCorners.clear();
                    ifFound=findChessboardCorners(cImg,size,chessboardCorners);
                    if(!ifFound) {
                        cout<<"No chessboard "<<WIDTH<<"*"<<HEIGHT<<" found!"<<endl;
                        break;
                    }
                    cvtColor(cImg, gImg, CV_BGR2GRAY);
                    cornerSubPix(gImg,chessboardCorners,Size(11,11),Size(-1,-1),TermCriteria(TermCriteria::COUNT+TermCriteria::EPS,30,0.1));
                    drawChessboardCorners(cImg,size,chessboardCorners,ifFound);
                    
                    imshow("corners",cImg);
                    
//                    str=string("/Users/t/Desktop/test/image_")+to_string(imgNum)+string(".txt");
                    str=string("./my_image_")+to_string(imgNum)+string(".txt");
                    os.open(str.c_str(),ios::out);
                    if(chessboardCorners.size()!=WIDTH*HEIGHT)
                        cout<<"Error happened! ChessboardCorners size not match!"<<endl;
                    os<<WIDTH*HEIGHT<<endl;
                    for(i=0;i<chessboardCorners.size();i++)
                        os<<chessboardCorners[i].x<<"\t"<<chessboardCorners[i].y<<endl;
                    os.close();
                    
                    cout<<"Chessboard "<<WIDTH<<"*"<<HEIGHT<<" found! A file \""<<str<<"\" is added"<<endl;
                    imgNum++;
                    break;
            }
        }
    }
    
    return 0;
}
