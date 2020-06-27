#include "common.h"

void Common::SaveGrayImage(vector<cv::String> image_path ,const string gray_path, bool is_reverse_color)
{
    for(int i = 0;i<image_path.size();++i)
    {       
        Mat rgbImage = imread(image_path[i]);
        Mat grayImage, rev_image;

        //gray and reversed image
        if(!is_reverse_color)
        {
            cvtColor(rgbImage, grayImage, CV_BGR2GRAY);
            rev_image = grayImage;
        }
        else
        {
            cvtColor(rgbImage, grayImage, CV_BGR2GRAY);
            //reverse the gray scale
            rev_image = 255 - grayImage;
        }

        string save_path = gray_path + image_path[i].substr(image_path[i].find_last_of('/'));
        imwrite(save_path,rev_image);
    }

}

void Common::ReadBibocularCameraPara(string path, BinocularCameraPara & bino_cam)
{
    bool FSflag = false;
    FileStorage readfs;

    FSflag = readfs.open(path, FileStorage::READ);
    if (FSflag == false) cout << "Cannot open the file" << endl;
    readfs["M1"] >> bino_cam.m_cameraMatrix_l;
    readfs["D1"] >> bino_cam.m_distCoeffs_l;
    readfs["M2"] >> bino_cam.m_cameraMatrix_r;
    readfs["D2"] >> bino_cam.m_distCoeffs_r;
    readfs["R"] >> bino_cam.m_R_relative;
    readfs["T"] >> bino_cam.m_t_relative;
    readfs["E"] >> bino_cam.m_E;
    readfs["F"] >> bino_cam.m_F;

    readfs.release();

}

void Common::ImageDedistortion(Mat src, Mat & dst , BinocularCameraPara bino_cam, int flag)
{

    Mat output;
    Mat	cameraMatrix, distCoeffs;
    Size imageSize;
    Mat map1, map2;

    if(flag)
    {
        cameraMatrix = bino_cam.m_cameraMatrix_r.clone();
        distCoeffs = bino_cam.m_distCoeffs_r.clone();
    }
    else {
        cameraMatrix = bino_cam.m_cameraMatrix_l.clone();
        distCoeffs = bino_cam.m_distCoeffs_l.clone();
    }

    cout<<"cameraMatrix: "<<cameraMatrix<<endl;
    cout<<"distCoeffs: "<<distCoeffs<<endl;

    Mat newCameraMatrix = cv::getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 0);

    undistort(src, dst, cameraMatrix, distCoeffs);
}

void Common::SelectPairImage(vector<cv::String> & image_l, vector<cv::String> & image_r, Size boardSize)
{
    assert(image_l.size() == image_r.size());
    int iter_size = image_l.size();
    vector<cv::String> image_l_select;
    vector<cv::String>  image_r_select;
    vector<Point2f> corner1;
    vector<Point2f> corner2;
    for (int i = 0; i < iter_size; ++i)
    {
        cout<<"selecting "<<i+1<<" th image pair"<<endl;
        Mat grayImage1 = imread(image_l[i],0);
        Mat grayImage2 = imread(image_r[i],0);

        if (waitKey(10) == 'q')
        {
            break;
        }

        if (grayImage1.empty() || grayImage2.empty())
        {
            cout << "Could not load tImage..." << endl;
            continue;
        }

        bool isFind1 = findChessboardCorners(grayImage1, boardSize, corner1, CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE);
        bool isFind2 = findChessboardCorners(grayImage2, boardSize, corner2, CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE);

        if (isFind1 && isFind2)
        {
            image_l_select.emplace_back(image_l[i]);
            image_r_select.emplace_back(image_r[i]);            
        }
    }

    image_l.clear();
    image_l = image_l_select;

    image_r.clear();
    image_r = image_r_select;

}

Eigen::MatrixXd toEigenMatrixXd(const cv::Mat &cvMat)
{
    Eigen::MatrixXd eigenMat;
    eigenMat.resize(cvMat.rows, cvMat.cols);
    for (int i=0; i<cvMat.rows; i++)
        for (int j=0; j<cvMat.cols; j++)
            eigenMat(i,j) = cvMat.at<double>(i,j);

    return eigenMat;
}

vector<double> toStdVector(const cv::Mat &cvMat)
{
    vector<double> stdVec;
    for (int i=0; i<cvMat.rows; i++)
        for (int j=0; j<cvMat.cols; j++)
            stdVec.emplace_back(cvMat.at<double>(i,j));

    return stdVec;

}

