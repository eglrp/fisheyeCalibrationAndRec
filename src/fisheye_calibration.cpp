#include "stdio.h"
#include <iostream>
#include <fstream>
// #include <io.h>
#include <dirent.h>

#include "opencv2/opencv.hpp"
#include <opencv2/core/core.hpp>
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

//读文件
    void getFiles(string path, vector<string>& files){
            
            struct dirent *ptr;    
            DIR *dir;
            dir=opendir(path.c_str());
            int i = 0;
            while((ptr=readdir(dir))!=NULL)
            {
                string p;
            //跳过'.'和'..'两个目录
                if(ptr->d_name[0] == '.'){
                    
                    continue;}
                printf("%s is ready...\n",ptr->d_name);
                // sprintf(files[i],"./one/%s",ptr->d_name);
                files.push_back(p.assign(path).append("/").append(ptr->d_name));
            
             
          
            }
            closedir(dir);





    }



int main(int argc, char** argv)
{   
    string filePath = "/home/xiesc/picked/cam1";
    vector<string> files;

    ////获取该路径下的所有文件
    getFiles(filePath, files);

    const int board_w = 10;
    const int board_h = 6;
    const int NPoints = board_w * board_h;//棋盘格内角点总数
    const int boardSize = 100; //mm
    Mat image,grayimage;
    Size ChessBoardSize = cv::Size(board_w, board_h);
    vector<Point2f> tempcorners;
//设置
    int flag = 0;
    flag |= cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC;
    //flag |= cv::fisheye::CALIB_CHECK_COND;
    flag |= cv::fisheye::CALIB_FIX_SKEW;
    //flag |= cv::fisheye::CALIB_USE_INTRINSIC_GUESS;
    flag |= CALIB_FIX_K3 ;
    vector<Point3f> object;
    for (int j = 0; j < NPoints; j++)
    {
        object.push_back(Point3f((j % board_w) * boardSize, (j / board_w) * boardSize, 0)); 
    }

    cv::Matx33d intrinsics;//z:相机内参
    cv::Vec4d distortion_coeff;//z:相机畸变系数

    vector<vector<Point3f> > objectv;
    vector<vector<Point2f> > imagev;

    Size corrected_size(800, 600);
    Mat mapx, mapy;
    Mat corrected;

    ofstream intrinsicfile("intrinsics_front1103.txt");
    ofstream disfile("dis_coeff_front1103.txt");
    int num = 0;
    bool bCalib = false;
    while (num < files.size())
    {
        image = imread(files[num]);

        if (image.empty())
            break;
        imshow("corner_image", image);
        waitKey(1);
        cvtColor(image, grayimage, CV_BGR2GRAY);
        IplImage tempgray = grayimage;
        bool findchessboard = cvCheckChessboard(&tempgray, ChessBoardSize);

        if (findchessboard)
        {
            bool find_corners_result = findChessboardCorners(grayimage, ChessBoardSize, tempcorners, CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE +
            CALIB_CB_FAST_CHECK);
            if (find_corners_result)
            {
                cornerSubPix(grayimage, tempcorners, cvSize(5, 5), cvSize(-1, -1), cvTermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
                drawChessboardCorners(image, ChessBoardSize, tempcorners, find_corners_result);
                imshow("corner_image", image);
                cvWaitKey(100);

                objectv.push_back(object);
                imagev.push_back(tempcorners);
                cout << "capture " << num << " pictures" << endl;
            }
        }
        tempcorners.clear();
        num++;
    }

    cv::fisheye::calibrate(objectv, imagev, cv::Size(image.cols,image.rows), intrinsics, distortion_coeff, cv::noArray(), cv::noArray(), flag, cv::TermCriteria(3, 20, 1e-6));  
    fisheye::initUndistortRectifyMap(intrinsics, distortion_coeff, cv::Matx33d::eye(), getOptimalNewCameraMatrix(intrinsics, distortion_coeff, corrected_size, 1, corrected_size, 0), corrected_size, CV_16SC2, mapx, mapy);
    cv::Matx33d intrinsicsAftUndis = getOptimalNewCameraMatrix(intrinsics, distortion_coeff, corrected_size, 1, corrected_size, 0);
    for(int i=0; i<3; ++i)
    {
        for(int j=0; j<3; ++j)
        {
            intrinsicfile<<intrinsics(i,j)<<"\t";
            
        }
        intrinsicfile<<endl;
    }
    for(int i=0; i<3; ++i)
    {
        for(int j=0; j<3; ++j)
        {
            intrinsicfile<<intrinsicsAftUndis(i,j)<<"\t";
            
        }
        intrinsicfile<<endl;
    }
    for(int i=0; i<4; ++i)
    {
        disfile<<distortion_coeff(i)<<"\t";
    }
    intrinsicfile.close();
    disfile.close();

    num = 0;
    while (num < files.size())
    {
        image = imread(files[num]);

        if (image.empty())
            break;
        remap(image, corrected, mapx, mapy, INTER_LINEAR, BORDER_TRANSPARENT);

        // imshow("corner_image", image);
        // imshow("corrected", corrected);
        // // imwrite("");
        // cvWaitKey(5); 
        string imageFileName;
        std::stringstream StrStm;
        StrStm <<"/home/xiesc/picked/rec1/"<<num<<"_d.jpg";
        StrStm >> imageFileName;
        
        imwrite(imageFileName, corrected);
        num++;
    }

    // cv::destroyWindow("corner_image");
    // cv::destroyWindow("corrected");

    image.release();
    grayimage.release();
    corrected.release();
    mapx.release();
    mapy.release();

    return 0;
}
