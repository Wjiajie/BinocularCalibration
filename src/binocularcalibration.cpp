#include "binocularcalibration.h"

void calibrateBinocamera(const string path, Config cfg)
{
    Common common;
    MonocularCalibration mono;
    //calibrate the camera

    const string gray_path_l = path + "/left_gray";
    const string gray_path_r = path + "/right_gray";

    string pattern_left,pattern_right;
    vector<cv::String> image_path_l,image_path_r;
    pattern_left = path+"/camera_left";
    pattern_right = path+"/camera_right";
    glob(pattern_left, image_path_l);
    glob(pattern_right, image_path_r);

    common.SaveGrayImage(image_path_l ,gray_path_l, cfg.m_isReversecolor);
    common.SaveGrayImage(image_path_r ,gray_path_r, cfg.m_isReversecolor);

    vector<cv::String> gray_image_path_l,gray_image_path_r;
    glob(gray_path_l, gray_image_path_l);
    glob(gray_path_r, gray_image_path_r);

    cout << "selecting pair image..." << endl;

    common.SelectPairImage(gray_image_path_l, gray_image_path_r, cfg.m_boardSize);
    cout << gray_image_path_l.size() << " pair of images selected..." << endl;
    assert(gray_image_path_l.size() > 10 && gray_image_path_r.size() > 10);

    cout<<"Calibration left..."<<endl;
    Mat camera_matrix_l, dist_coeffs_l;
    vector<Mat> R_vec_l;
    vector<Mat> t_vec_l;
    vector<vector<Point3f>> X_vec;
    mono.MonocularCali_lib(gray_image_path_l, cfg, camera_matrix_l, dist_coeffs_l, R_vec_l, t_vec_l, X_vec);
    MonocularCameraPara monocam_left = MonocularCameraPara(camera_matrix_l ,dist_coeffs_l);
    mono.SaveMonocularCaliPara(path+"/intrinsics_l.yml",monocam_left);

    cout<<"Calibration right..."<<endl;
    Mat camera_matrix_r, dist_coeffs_r;
    vector<Mat> R_vec_r;
    vector<Mat> t_vec_r;
    vector<vector<Point3f>> temp;
    mono.MonocularCali_lib(gray_image_path_r, cfg, camera_matrix_r, dist_coeffs_r, R_vec_r, t_vec_r, temp);

    MonocularCameraPara monocam_right = MonocularCameraPara(camera_matrix_r ,dist_coeffs_r);
    mono.SaveMonocularCaliPara(path+"/intrinsics_r.yml", monocam_right);
    //caculate R,t,E,F and save the paras
    mono.CaculateRelativeR_t_E_F(monocam_left, monocam_right, X_vec, R_vec_l, t_vec_l, R_vec_r, t_vec_r, path);

}
