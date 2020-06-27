#ifndef COMMON_H
#define COMMON_H

#include "head.h"
#include "calibration.h"

struct View
{
    //poses
    Eigen::Matrix3d rotation;
    Eigen::Matrix3d K;
    Eigen::Matrix3d K_Inv;
    std::vector<double> distortionParams;
    Eigen::Matrix3d t_;
    Eigen::Vector3d t;

    double* rotation_array; //ÖáœÇ
    double* translatrion_array;
    double scale;
    View() {}
    View(Eigen::Matrix3d r, Eigen::Vector3d _t, Eigen::Matrix3d _K, std::vector<double> _distort)
    {
        this->rotation = r;
        this->t = _t;
        this->K = _K;
        this->distortionParams = _distort;

        this->K_Inv = this->K.inverse();

        t_ << 0, -t(2), t(1),
            t(2), 0, -t(0),
            -t(1), t(0), 0;

        this->rotation_array = new double[3];
        Eigen::AngleAxisd aa(this->rotation);
        Eigen::Vector3d v = aa.angle() * aa.axis();
        rotation_array[0] = v.x();
        rotation_array[1] = v.y();
        rotation_array[2] = v.z();

        this->translatrion_array = new double[3];
        translatrion_array[0] = t.x();
        translatrion_array[1] = t.y();
        translatrion_array[2] = t.z();
        this->scale = 1.0f;   //ºÍworldµÄ³ß¶È
    }
};

struct Observation
{
    Eigen::Vector2d pixel;
    Eigen::Vector2d match_pixel;
    int host_camera_id; //ÊôÓÚÄÄÒ»žöcamera
    int neighbor_camera_id; //ÓëÖ®Æ¥ÅäµÄcamera£¬ÓÃÀŽÑ°ÕÒÆ¥Åäµã


    Observation() {}
    Observation(Eigen::Vector2d p, Eigen::Vector2d m_p,int h_camera_id_ = -1, int n_camera_id_ = -1)
    {
        pixel = p;
        match_pixel = m_p;
        host_camera_id = h_camera_id_;
        neighbor_camera_id = n_camera_id_;
    }
};

struct Structure
{
    Eigen::Vector3d position;
    Eigen::Vector3d colors; //rgb
    uint structure_index;
    double* positions_array;
    bool isvalid_structure;

    Structure() {}
    Structure(Eigen::Vector3d _p , uint structure_index_,bool isvalid_structure_ = true)
    {
        position = _p;
        colors = Eigen::Vector3d::Zero();
        positions_array = new double[3];
        positions_array[0] = _p.x();
        positions_array[1] = _p.y();
        positions_array[2] = _p.z();

        structure_index = structure_index_;

        isvalid_structure = isvalid_structure_;
    }
};

//store the imformation of a feature in image
struct FeatureInImage
{
    KeyPoint m_feature;
    KeyPoint m_feature_match;
    Vector2d m_kp;
    Vector2d m_kp_match;
    Vector2d m_distance_from_edge;
    int m_which_temp;

    FeatureInImage(){};
    FeatureInImage(KeyPoint feature, KeyPoint feature_match, Vector2d kp, Vector2d kp_match, Vector2d distance_from_edge, int which_temp)
    {
        this->m_feature = feature;
        this->m_feature_match = feature_match;
        this->m_kp = kp;
        this->m_kp_match = kp_match;
        this->m_distance_from_edge = distance_from_edge;
        this->m_which_temp = which_temp;
    }
};


class Common
{
public:
    Common(void){};
    void SaveGrayImage(vector<cv::String> image_path ,const string gray_path, bool is_reverse_color);
    void ReadBibocularCameraPara(string path, BinocularCameraPara & bino_cam);
    void ImageDedistortion(Mat src, Mat & dst , BinocularCameraPara bino_cam, int flag);
    void SelectPairImage(vector<cv::String> & image_l, vector<cv::String> & image_r, Size boardSize);
    ~Common(void){};
};

Eigen::MatrixXd toEigenMatrixXd(const cv::Mat &cvMat);
vector<double> toStdVector(const cv::Mat &cvMat);

bool sort_by_vec_x(Point2d p1 ,Point2d p2);
bool sort_by_vec_y(Point2d p1 ,Point2d p2);
bool sort_by_vec_size(vector<Point2d> p1 ,vector<Point2d> p2);

#endif // COMMON_H
