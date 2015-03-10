#include <iostream>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <math.h>
#include <cmath>
#include <string>
#include <fstream>

#define DESIRED_PROBABILITY 0.99
#define MAX_TRY 100000

using namespace std;
using namespace cv;

bool readFiles(const string &worldFile, vector<string> &imageFiles, vector<Point3f> &world, vector< vector<Point2f> > &images)
{
    world.clear();
    images.clear();
    fstream wldin(worldFile.c_str());
    for(int i=0;i<imageFiles.size();i++) {
        images.push_back(vector<Point2f>());
    }
    int n,m;
    wldin>>n;
    for(int i=0;i<n;i++) {
        Point3f pt3;
        wldin>>pt3.x>>pt3.y>>pt3.z;
        world.push_back(pt3);
    }
    for(int i=0;i<imageFiles.size();i++) {
        fstream imgsin(imageFiles[i].c_str());
        imgsin>>m;
        if(m!=n) {
            cout<<"# of points in world and images don't match!"<<endl;
            return false;
        }
        for(int j=0;j<m;j++) {
            Point2f pt2;
            imgsin>>pt2.x>>pt2.y;
            images[i].push_back(pt2);
        }
    }
    return true;
}

Mat findH(vector<Point3f> &world, vector<Point2f> &image, int n) {
    Mat A=Mat::zeros(2*n, 9, CV_32FC1);
    for (int i=0; i<n; i++) {
        int a=i*2;
        int b=a+1;
        A.at<float>(a,0)=world[i].x;
        A.at<float>(a,1)=world[i].y;
        A.at<float>(a,2)=1;
        A.at<float>(a,6)=world[i].x * image[i].x * (-1);
        A.at<float>(a,7)=world[i].y * image[i].x * (-1);
        A.at<float>(a,8)=image[i].x * (-1);
        A.at<float>(b,3)=world[i].x;
        A.at<float>(b,4)=world[i].y;
        A.at<float>(b,5)=1;
        A.at<float>(b,6)=world[i].x * image[i].y * (-1);
        A.at<float>(b,7)=world[i].y * image[i].y * (-1);
        A.at<float>(b,8)=image[i].y * (-1);
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

Mat getV(Mat &H, int i, int j) {
    Mat V=Mat::zeros(6,1,CV_32FC1);
    i--;
    j--;
    V.at<float>(0,0)=H.at<float>(0,i)*H.at<float>(0,j);
    V.at<float>(1,0)=H.at<float>(0,i)*H.at<float>(1,j)+H.at<float>(1,i)*H.at<float>(0,j);
    V.at<float>(2,0)=H.at<float>(1,i)*H.at<float>(1,j);
    V.at<float>(3,0)=H.at<float>(2,i)*H.at<float>(0,j)+H.at<float>(0,i)*H.at<float>(2,j);
    V.at<float>(4,0)=H.at<float>(2,i)*H.at<float>(1,j)+H.at<float>(1,i)*H.at<float>(2,j);
    V.at<float>(5,0)=H.at<float>(2,i)*H.at<float>(2,j);
    return V;
}

Mat getKstar(vector<Mat> &Hs) {
    int n=Hs.size();
    Mat vs=Mat::zeros(2*n,6,CV_32FC1);
    for(int i=0;i<n;i++) {
        Mat z=getV(Hs[i], 1, 2).t().row(0);
        z.row(0).copyTo(vs.row(2*i));
        z=(getV(Hs[i], 1, 1)-getV(Hs[i], 2, 2)).t().row(0);
        z.row(0).copyTo(vs.row(2*i+1));
    }
    Mat w,u,vt;
    SVD::compute(vs, w, u, vt, SVD::FULL_UV);
    //    cout<<"vs in getParameters:\n"<<vs<<endl;
    //    cout<<"w in getParameters:\n"<<w<<endl;
    //    cout<<"u in getParameters:\n"<<u<<endl;
    //    cout<<"vt in getParameters:\n"<<vt<<endl;
    float s11=vt.at<float>(5,0);
    float s12=vt.at<float>(5,1);
    float s22=vt.at<float>(5,2);
    float s13=vt.at<float>(5,3);
    float s23=vt.at<float>(5,4);
    float s33=vt.at<float>(5,5);
    float c1=s12*s13-s11*s23;
    float c2=s11*s22-s12*s12;
    float v0=c1/c2;
    float lambda=s33-(s13*s13+v0*c1)/s11;
    float au=sqrt(lambda/s11);
    float av=sqrt(lambda*s11/c2);
    float s=(-1)*s12*au*au*av/lambda;
    float u0=s*v0/au-s13*au*au/lambda;
    Mat Kstar=Mat::zeros(3, 3, CV_32FC1);
    Kstar.at<float>(0,0)=au;
    Kstar.at<float>(0,1)=s;
    Kstar.at<float>(0,2)=u0;
    Kstar.at<float>(1,1)=av;
    Kstar.at<float>(1,2)=v0;
    Kstar.at<float>(2,2)=1;
    //    cout<<"Kstar:\n"<<Kstar<<endl;
    return Kstar;
}

Mat getRstarTstar(Mat &Kstar, Mat &H)
{
    Mat h1=H.col(0);
    Mat h2=H.col(1);
    Mat h3=H.col(2);
    Mat kinv=Kstar.inv();
    Mat kh1=kinv*h1;
    Mat kh2=kinv*h2;
    Mat kh3=kinv*h3;
    Mat kh1sq=kh1.mul(kh1);
    float alpha=((kh3.at<float>(2,0)>=0)?1:(-1))*1/sqrt(kh1sq.at<float>(0,0)+kh1sq.at<float>(1,0)+kh1sq.at<float>(2,0));
    Mat RstarTstar=Mat::zeros(3,4,CV_32FC1);
    Mat temp1=alpha*kh1.col(0);
    temp1.col(0).copyTo(RstarTstar.col(0));
    Mat temp2=alpha*kh2.col(0);
    temp2.col(0).copyTo(RstarTstar.col(1));
    RstarTstar.col(0).cross(RstarTstar.col(1)).copyTo(RstarTstar.col(2));
    Mat temp4=alpha*kh3.col(0);
    temp4.col(0).copyTo(RstarTstar.col(3));
    //    cout<<"RstarTstar:\n"<<RstarTstar<<endl;
    return RstarTstar;
}

double evalue(vector<Point3f> &world, vector<Point2f> &image, Mat &Kstar, Mat &RstarTstar)
{
    int n=world.size();
    if(n!=image.size()) {
        cout<<"n!=m in evalue"<<endl;
        exit(0);
    }
    Mat M=Kstar*RstarTstar;
    Mat m1t=M.row(0);
    Mat m2t=M.row(1);
    Mat m3t=M.row(2);
    double E=0;
    for (int i=0; i<n; i++) {
        Mat p=Mat::zeros(4, 1, CV_32FC1);
        p.at<float>(0,0)=world[i].x;
        p.at<float>(1,0)=world[i].y;
        p.at<float>(2,0)=world[i].z;
        p.at<float>(3,0)=1;
        float a=Mat(m1t*p).at<float>(0,0);
        float b=Mat(m2t*p).at<float>(0,0);
        float c=Mat(m3t*p).at<float>(0,0);
        E+=sqrt(pow(image[i].x-a/c,2)+pow(image[i].y-b/c,2));
    }
    E/=n;
    return E;
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

int cmp(pair<int,float> &a, pair<int,float> &b)
{
    return a.second<b.second;
}

bool ifEqual(pair< pair<int, float> ,Mat> &a, pair< pair<int, float> ,Mat> &b)
{
    if(a.first.first==b.first.first && a.first.second==b.first.second)
        return true;
    return false;
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
    if(a_f/b_f>2 || b_f/a_f>2) {
        if(a_s/a_f<0.01)
            a_s=a_f*0.01;
        if(b_s/b_f<0.01)
            b_s=b_f*0.01;
    }
    float a_e=pow(a_f,3)/a_s;
    float b_e=pow(b_f,3)/b_s;
    if(a_e>b_e)
        return a;
    return b;
}

Mat ransacFindH(int n, int d, float p, double w, float kThreshold, vector<Point3f> &world, vector<Point2f> &image, float max_tolerance, float min_tolerance)
{
    double k=log(1-p)/log(1-pow(w, n));
    float t;
    srand((unsigned)time(0));
    srand(rand());
    srand(rand());
    pair< pair<int, float> ,Mat> better_H(pair<int, float>(-1,MAXFLOAT),Mat());
    for(int i=0;i<kThreshold && i<k;i++) {
        vector<int> rdm;
        getRandom(0, world.size(), n, rdm);
        vector<Point3f> rdmWorld;
        vector<Point2f> rdmImage;
        for(int j=0;j<n;j++) {
            rdmWorld.push_back(world[rdm[j]]);
            rdmImage.push_back(image[rdm[j]]);
        }
        Mat H;
        H=findH(rdmWorld, rdmImage, n);
        float ifNan=0;
        for(int a=0;a<H.rows;a++)
            for(int b=0;b<H.cols;b++)
                ifNan+=H.at<float>(a,b);
        if(isnan(ifNan)) {
            i--;
            cout<<"Nan Happened1!"<<endl;
            continue;
        }
        
        Mat pts=Mat::zeros(3, (int)world.size(), CV_32FC1);
        for(int j=0;j<world.size();j++) {
            pts.at<float>(0,j)=world[j].x;
            pts.at<float>(1,j)=world[j].y;
            pts.at<float>(2,j)=1;
        }
        Mat ppts=H*pts;
        //        cout<<H<<endl;
        vector< pair<int, float> > dists;
        dists.clear();
        for(int j=0;j<world.size();j++) {
            float x=ppts.at<float>(0,j)/ppts.at<float>(2,j);
            float y=ppts.at<float>(1,j)/ppts.at<float>(2,j);
            //            cout<<"("<<x<<", "<<y<<"), "<<"("<<image[j].x<<", "<<image[j].y<<")"<<endl;
            float z=sqrt(pow(x-image[j].x,2)+pow(y-image[j].y,2));
            dists.push_back(pair<int, float>(j,z));
        }
        sort(dists.begin(),dists.end(),cmp);
        if(dists.size()%2==0) {
            if(dists.size()!=0)
                t=1.5*(dists[dists.size()/2-1].second+dists[dists.size()].second)/2.0;
        }
        else {
            t=1.5*dists[(int)(dists.size()-1)/2].second;
        }
        t=t>max_tolerance?max_tolerance:t;
        t=t<min_tolerance?min_tolerance:t;
        
        
        //recompute
        rdmWorld.clear();
        rdmImage.clear();
        for(int j=0; j<dists.size() && dists[j].second<t; j++) {
            rdmWorld.push_back(world[dists[j].first]);
            rdmImage.push_back(image[dists[j].first]);
        }
        if(rdmImage.size()<4) {
//            cout<<"d<4! continue"<<endl;
//            cout<<"t="<<t<<"; d="<<rdmImage.size()<<endl;
            continue;
        }

            H=findH(rdmWorld, rdmImage, (int)rdmImage.size());
            ifNan=0;
            for(int a=0;a<H.rows;a++)
                for(int b=0;b<H.cols;b++)
                    ifNan+=H.at<float>(a,b);
            if(isnan(ifNan)) {
                i--;
                cout<<"Nan Happened2!"<<endl;
                continue;
            }
            pts=Mat::zeros(3, (int)world.size(), CV_32FC1);
            for(int j=0;j<world.size();j++) {
                pts.at<float>(0,j)=world[j].x;
                pts.at<float>(1,j)=world[j].y;
                pts.at<float>(2,j)=1;
            }
            ppts=H*pts;
//          cout<<H<<endl;
            dists.clear();
            for(int j=0;j<world.size();j++) {
                float x=ppts.at<float>(0,j)/ppts.at<float>(2,j);
                float y=ppts.at<float>(1,j)/ppts.at<float>(2,j);
//                  cout<<"("<<x<<", "<<y<<"), "<<"("<<image[j].x<<", "<<image[j].y<<")"<<endl;
                float z=sqrt(pow(x-image[j].x,2)+pow(y-image[j].y,2));
                dists.push_back(pair<int, float>(j,z));
            }
            sort(dists.begin(),dists.end(),cmp);
            if(dists.size()%2==0) {
                if(dists.size()!=0)
                    t=1.5*(dists[dists.size()/2-1].second+dists[dists.size()].second)/2.0;
            }
            else {
                t=1.5*dists[(int)(dists.size()-1)/2].second;
            }
            t=t>max_tolerance?max_tolerance:t;
            t=t<min_tolerance?min_tolerance:t;

        
        int inlines=0;
        float inlinesDist=0;
        while (inlines<dists.size() && dists[inlines].second<t) {
            inlinesDist+=dists[inlines].second;
            inlines++;
        }
        if(inlines<d)
            continue;
        
        pair< pair<int, float> ,Mat> candidate(pair<int,float>(inlines,inlinesDist), H);
        pair< pair<int, float> ,Mat> z=better_H;
        better_H=betterModel(better_H, candidate);
        w=better_H.first.first/(float)world.size();
        k=log(1-p)/log(1-pow(w, n));
    }
    cout<<"w="<<w<<", k="<<k<<", # of inliers="<<better_H.first.first<<", average inlier distance error="<<better_H.first.second/better_H.first.first<<endl;
    return better_H.second;
}

//void ransac(int n, int d, double p, double w, double kThreshold, vector<Point3f> &world, vector< vector<Point2f> > &images)
//{
//    double k=log(1-p)/log(1-pow(w, n));
//    double t;
//    srand((unsigned)time(0));
//    srand(rand());
//    srand(rand());
//    for(int i=0;i<kThreshold && i<k;i++) {
//        vector<int> rdm;
//        getRandom(0, world.size(), n, rdm);
//        vector<Point3f> rdmWorld;
//        vector< vector<Point2f> > rdmImages;
//        for(int j=0;j<images.size();j++)
//            rdmImages.push_back(vector<Point2f>());
//        for(int j=0;j<n;j++) {
//            rdmWorld.push_back(world[rdm[j]]);
//            for(int a=0;a<rdmImages.size();a++)
//                rdmImages[a].push_back(images[a][rdm[j]]);
//        }
//
//        vector<Mat> Hs;
//        for(int j=0;j<rdmImages.size();j++)
//            Hs.push_back(findH(rdmWorld, rdmImages[j], n));
//        Mat Kstar=getKstar(Hs);
//        vector<Mat> RstarTstars;
//        for(int j=0;j<rdmImages.size();j++)
//            RstarTstars.push_back(getRstarTstar(Kstar, Hs[j]));
//        vector<Mat> Ms;
//        float ifNan=0;
//        for(int j=0;j<rdmImages.size();j++) {
//            Ms.push_back(Kstar*RstarTstars[j]);
//            for(int a=0;a<Ms[j].rows;a++)
//                for(int b=0;b<Ms[j].cols;b++)
//                    ifNan+=Ms[j].at<float>(a,b);
//            if(isnan(ifNan))
//                break;
//        }
//        if(isnan(ifNan)) {
//            i--;
//            continue;
//        }
//
//        for(int j=0;j<images.size();j++) {
//            vector<float> dists;
//            dists.clear();
//            for(int a=0;a<world.size();a++) {
//                Mat p=Mat::zeros(4, 1, CV_32FC1);
//                p.at<float>(0,0)=world[a].x;
//                p.at<float>(1,0)=world[a].y;
//                p.at<float>(2,0)=world[a].z;
//                p.at<float>(3,0)=1;
//                Mat pp=Ms[j]*p;
//                dists.push_back(sqrt(pow(pp.at<float>(0,0)/pp.at<float>(2,0)-images[j][a].x,2)+pow(pp.at<float>(1,0)/pp.at<float>(2,0)-images[j][a].y,2)));
//            }
//            sort(dists.begin(),dists.end());
//        }
//
//        k=log(1-p)/log(1-pow(w, n));
//    }
//}

int main(int argc, char *argv[])
{
    cout.setf(ios::fixed);
    cout<<"----------Coplanar calibration------------"<<endl;
    cout<<"Execute this program as:"<<endl;
    cout<<"[program-name][world-points-file][image-points-file1][image-points-file2][image-points-file3]..."<<endl;
    cout<<"------------------------------------------"<<endl;
    if(argc<5) {
        cout<<"Error: not enough parameters, program will exit."<<endl;
        exit(0);
    }
    string worldFile(argv[1]);
    vector<string> imageFiles;
    for(int i=2;i<argc;i++)
        imageFiles.push_back(string(argv[i]));
    
//    string worldFile("/Users/t/Desktop/test/world.txt");
//    vector<string> imageFiles;
//    imageFiles.push_back("/Users/t/Desktop/test/image_0.txt");
//    imageFiles.push_back("/Users/t/Desktop/test/image_1.txt");
//    imageFiles.push_back("/Users/t/Desktop/test/image_2.txt");
    
    vector<Point3f> world;
    vector< vector<Point2f> > images;
    if(!readFiles(worldFile, imageFiles, world, images)) {
        cout<<"Error happenes when reading file, program will exit."<<endl;
        exit(0);
    }
    vector<Mat> Hs;
    //    Hs.push_back(ransacFindH(4, 4, DESIRED_PROBABILITY, 0.5, MAX_TRY, world, images[1], 10, 0.01));
    for(int i=0;i<images.size();i++) {
        cout<<"image "<<i<<":"<<endl;
        Hs.push_back(ransacFindH(4, 4, DESIRED_PROBABILITY, 0.5, MAX_TRY, world, images[i], 5, 0.01));
    //        Hs.push_back(findH(world, images[i], 4));
    }
    
    //    Mat H=Hs[0];
    //    double m;
    //    for(int i=0;i<world.size();i++) {
    //        Mat p=Mat(3, 1, CV_32FC1);
    //        p.at<float>(0,0)=world[i].x;
    //        p.at<float>(1,0)=world[i].y;
    //        p.at<float>(2,0)=1;
    //        Mat xyz=H*p;
    //        float x=xyz.at<float>(0,0)/xyz.at<float>(2,0);
    //        float y=xyz.at<float>(1,0)/xyz.at<float>(2,0);
    //        if(i==0)
    //            m=images[0][0].x/x;
    //        x*=m;
    //        y*=m;
    //        cout<<x<<"\t"<<y<<endl;
    //    }
    
    Mat Kstar=getKstar(Hs);
    vector<Mat> RstarTstars;
    for(int i=0;i<images.size();i++)
        RstarTstars.push_back(getRstarTstar(Kstar, Hs[i]));
    
    for (int i=0; i<images.size(); i++) {
        cout<<i<<"th H:"<<endl;
        cout<<Hs[i]<<endl;
    }
    cout<<"Kstar:"<<endl;
    cout<<Kstar<<endl;
    for (int i=0; i<images.size(); i++) {
        cout<<i<<"th RstarTstar:"<<endl;
        cout<<RstarTstars[i]<<endl;
    }
    
    for(int i=0;i<images.size();i++) {
        double E=evalue(world, images[i], Kstar, RstarTstars[i]);
        cout<<"for image"<<i<<", the error is "<<setprecision(4)<<E<<" pixels."<<endl;
    }
    
    return 0;
}