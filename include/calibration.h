#ifndef MONOCULARCALIBRATION_H
#define MONOCULARCALIBRATION_H

#include "head.h"

struct Config
{
    int m_boardWidth;          //横向的角点数目
    int m_boardHeight;        //纵向的角点数据
    int m_boardCorner;       //总的角点数据
    int m_squareSize;       //标定板黑白格子的大小 单位mm
    Size m_boardSize;      //总的内角点
    bool m_isReversecolor;

    Config() {}
    Config(int boardWidth, int boardHeight, int boardCorner, int squareSize, Size boardSize, bool isReversecolor)
    {
        this->m_boardWidth = boardWidth;
        this->m_boardHeight = boardHeight;
        this->m_boardCorner = boardCorner;
        this->m_squareSize = squareSize;
        this->m_boardSize = boardSize;
        this->m_isReversecolor = isReversecolor;
    }
};

struct MonocularCameraPara
{
    Mat m_cameraMatrix;
    Mat m_distCoeffs;

    MonocularCameraPara(){};

    MonocularCameraPara(Mat cameraMatrix , Mat distCoeffs)
    {
        m_cameraMatrix = cameraMatrix;
        m_distCoeffs = distCoeffs;
    }

};

struct BinocularCameraPara
{
    Mat m_cameraMatrix_l;
    Mat m_distCoeffs_l;
    Mat m_cameraMatrix_r;
    Mat m_distCoeffs_r;

    Mat m_R_relative;
    Mat m_t_relative;
    Mat m_E;
    Mat m_F;

    BinocularCameraPara(){};
    BinocularCameraPara(Mat cameraMatrix_l, Mat distCoeffs_l, Mat cameraMatrix_r, Mat distCoeffs_r, Mat R_relative, Mat t_relative, Mat E, Mat F)
    {
        m_cameraMatrix_l = cameraMatrix_l;
        m_distCoeffs_l = distCoeffs_l;
        m_cameraMatrix_r = cameraMatrix_r;
        m_distCoeffs_r = distCoeffs_r;

        m_R_relative = R_relative;
        m_t_relative = t_relative;
        m_E = E;
        m_F = F;
    }

};


struct Ceres_AdjustRt
{   
    const Eigen::Vector3d X_cam_l;
    const Eigen::Vector3d X_cam_r;

    Ceres_AdjustRt(Eigen::Vector3d X_cam_l_, Eigen::Vector3d X_cam_r_):X_cam_l(X_cam_l_),X_cam_r(X_cam_r_){}

    template<typename T>
    bool operator()(const T* const ceres_angleAxis, const T* const ceres_t, T* residual) const
    {
        T PXL[3];

        PXL[0] = T(X_cam_l[0]);
        PXL[1] = T(X_cam_l[1]);
        PXL[2] = T(X_cam_l[2]);

        T PXR[3];

        PXR[0] = T(X_cam_r[0]);
        PXR[1] = T(X_cam_r[1]);
        PXR[2] = T(X_cam_r[2]);

        T PX_r[3];
        ceres::AngleAxisRotatePoint(ceres_angleAxis, PXL, PX_r);

        residual[0] = PX_r[0] + ceres_t[0] - PXR[0];
        residual[1] = PX_r[1] + ceres_t[1] - PXR[1];
        residual[2] = PX_r[2] + ceres_t[2] - PXR[2];

        return true;
    }


};


class MonocularCalibration
{
public:
    void MonocularCali_lib(vector<cv::String> image_path, Config cfg, Mat &camera_matrix ,Mat &dist_coeffs, vector<Mat> & R_vec ,vector<Mat> & t_vec, vector<vector<Point3f>> & objRealPoint);
    void SaveMonocularCaliPara(string save_path, MonocularCameraPara monocam_para);
    void ReadMonocularCaliPara(string save_path, Mat & camera_matrix, Mat & dist_coeffs);
    void CaculateRelativeR_t_E_F(MonocularCameraPara monocam_left, MonocularCameraPara monocam_right, vector<vector<Point3f>> X_vec, vector<Mat> R_vec_l, vector<Mat> t_vec_l, vector<Mat> R_vec_r, vector<Mat> t_vec_r, string outputPath);
};

bool sort_by_idx(String s1 ,String s2);
void sortImagePath(vector<String> & image_path);

/*计算标定板上模块的实际物理坐标*/
void calRealPoint(vector<vector<Point3f>>& obj, int boardwidth, int boardheight, int imgNumber, int squaresize);
/*设置相机的初始参数 也可以不估计*/
void guessCameraParam(Mat & intrinsic, Mat & distortion_coeff);
void outputCameraParam(Mat intrinsic, Mat distortion_coeff, vector<Mat> rvecs, vector<Mat> tvecs);

#endif // MONOCULARCALIBRATION_H
