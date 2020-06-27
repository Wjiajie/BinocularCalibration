#include "calibration.h"
#include "common.h"

string OnputDataPath = "/home/jiajie/3d_reco/Binocular/BinocularCalibration1.1/HikivisionImage5-28";

void MonocularCalibration::SaveMonocularCaliPara(string save_path, MonocularCameraPara monocam_para)
{
    FileStorage fs(save_path, FileStorage::WRITE);
    assert(fs.isOpened());

    fs << "M" << monocam_para.m_cameraMatrix << "D" << monocam_para.m_distCoeffs;
    fs.release();
}

void MonocularCalibration::ReadMonocularCaliPara(string save_path, Mat & camera_matrix, Mat & dist_coeffs)
{
    bool FSflag = false;
    FileStorage readfs;

    FSflag = readfs.open(save_path, FileStorage::READ);
    if (FSflag == false) cout << "Cannot open the file" << endl;
    readfs["M"] >> camera_matrix;
    readfs["D"] >> dist_coeffs;

    readfs.release();
}


void  MonocularCalibration::MonocularCali_lib(vector<cv::String> image_path, Config cfg, Mat &camera_matrix ,Mat &dist_coeffs, vector<Mat> & R_vec ,vector<Mat> & t_vec, vector<vector<Point3f>> & objRealPoint)
{
    Mat intrinsic;                                                //相机内参数
    Mat distortion_coeff;                                   //相机畸变参数
    vector<Mat> rvecs;                                        //旋转向量                                         //平移向量
    vector<vector<Point2f>> corners;                        //各个图像找到的角点的集合 和objRealPoint 一一对应
    vector<Point2f> corner;

    int imageHeight;     //图像高度
    int imageWidth;      //图像宽度
    int goodFrameCount = 0;    //有效图像的数目

    //sort(image_path.begin(), image_path.end());
    sortImagePath(image_path);

    for(int i = 0;i<image_path.size();++i)
    {       
        //if input images are in rgb domain, set: Mat rgbImage = imread(image_path[i],1);
        Mat image = imread(image_path[i],0);
        Mat grayImage,rev_image;

        if(image.channels() == 1)
        {
            grayImage = imread(image_path[i],0);
            rev_image = grayImage.clone();
        }
        else
        {
            cvtColor(image, grayImage, CV_BGR2GRAY);
            rev_image = 255 - grayImage;
        }

        //imshow("rev_image: ",rev_image);

        if (waitKey(10) == 'q')
        {
            break;
        }

        if (rev_image.empty())
        {
            cout<<"Could not load tImage..."<<endl;
        }
        imageHeight = grayImage.rows;
        imageWidth = grayImage.cols;

        bool isFind = findChessboardCorners(rev_image, cfg.m_boardSize, corner, CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE);

        if (isFind == true) //所有角点都被找到 说明这幅图像是可行的
        {           
            //精确角点位置，亚像素角点检测
            cornerSubPix(rev_image, corner, Size(5, 5), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 20, 0.1));
            //绘制角点
            drawChessboardCorners(image, cfg.m_boardSize, corner, isFind);
            Mat rgbImage_r;
            double sf = 640. / MAX(image.rows, image.cols);
            resize(image, rgbImage_r, Size(), sf, sf, INTER_LINEAR_EXACT);
            imshow("chessboard", rgbImage_r);
            corners.push_back(corner);
            goodFrameCount++;            
        }
        else
        {
            cout<<"The image is bad please try again..."<<endl;
        }

    }

    waitKey(10);

    /*
    图像采集完毕 接下来开始摄像头的校正
    calibrateCamera()
    输入参数 objectPoints  角点的实际物理坐标
    imagePoints   角点的图像坐标
    imageSize     图像的大小
    输出参数
    cameraMatrix  相机的内参矩阵
    distCoeffs    相机的畸变参数
    rvecs         旋转矢量(外参数)
    tvecs         平移矢量(外参数）
    */

    cout<<"start para estimate ... "<<endl;


    /*设置实际初始参数 根据calibrateCamera来 如果flag = 0 也可以不进行设置*/
    guessCameraParam(intrinsic, distortion_coeff);
    cout<<"guessCameraParam ... "<<endl;
    /*计算实际的校正点的三维坐标*/
    const int frameNumber = int(image_path.size());
    cout<<"goodFrameCount: "<<goodFrameCount<<endl;
    calRealPoint(objRealPoint, cfg.m_boardWidth, cfg.m_boardHeight, goodFrameCount, cfg.m_squareSize);
    cout<<"calculate real successful.."<<endl;
    /*标定摄像头*/
    cout<<"imageWidth, imageHeight:"<<imageWidth<<" , "<< imageHeight<<endl;
    auto reproject_error = calibrateCamera(objRealPoint, corners, Size(imageWidth, imageHeight), intrinsic, distortion_coeff, rvecs, t_vec, 0); //CV_CALIB_RATIONAL_MODEL
    cout<<"calibration successful..."<<endl;
    cout<<"reproject_error: "<<reproject_error<<endl;

    for(auto rvec:rvecs)
    {
        Mat rotation_matrix = Mat(3, 3, CV_64FC1, Scalar::all(0));  // 图像的旋转矩阵
        Rodrigues(rvec, rotation_matrix);
        //cout<<"test R: "<<rotation_matrix.col(0).t() * rotation_matrix.col(1)<<endl;
        R_vec.emplace_back(rotation_matrix);
    }
    /*保存并输出参数*/
    camera_matrix = intrinsic.clone();
    dist_coeffs = distortion_coeff.clone();   
}

void MonocularCalibration::CaculateRelativeR_t_E_F(MonocularCameraPara monocam_left, MonocularCameraPara monocam_right, vector<vector<Point3f>> X_vec, vector<Mat> R_vec_l, vector<Mat> t_vec_l, vector<Mat> R_vec_r, vector<Mat> t_vec_r,string outputPath)
{
    //caculate relative R,t
    assert(R_vec_l.size() == X_vec.size() );
    assert(R_vec_l.size() == X_vec.size() );


    cout << "SATRT  BA" << endl;
    ceres::Problem problem;
    ceres::LossFunction* lossFunc = new ceres::HuberLoss(2.0f);

    Eigen::Matrix3d R_rel_eig_init = Eigen::Matrix3d::Zero();
    Eigen::Vector3d t_rel_eig_init = Eigen::Vector3d::Zero();

    for(uint i = 0;i<X_vec.size();++i)
    {
        //init R_relative_l_r and t_relative_l_r
        Mat R_relative_l_r = R_vec_r[i] * R_vec_l[i].inv();
        Mat t_relative_l_r = t_vec_r[i] - R_relative_l_r * t_vec_l[i];

        //double base_line = sqrt(pow(t_relative_l_r.at<double>(0,0),2) + pow(t_relative_l_r.at<double>(1,0),2) + pow(t_relative_l_r.at<double>(2,0),2));
        //cout<<"base_line : "<<base_line<<endl;

        Eigen::Matrix3d R_rel_eig_l_r = toEigenMatrixXd(R_relative_l_r);
        Eigen::Vector3d t_rel_eig_l_r = toEigenMatrixXd(t_relative_l_r);
        //cout<<"R_rel_eig_l_r: "<<R_rel_eig_l_r<<endl;
        //cout<<"t_rel_eig_l_r: "<<t_rel_eig_l_r<<endl;

        //select a reasonable t_relative_l_r
        R_rel_eig_init = R_rel_eig_init + R_rel_eig_l_r;
        t_rel_eig_init = t_rel_eig_init + t_rel_eig_l_r;

    }

    R_rel_eig_init /= int(X_vec.size());
    t_rel_eig_init /= int(X_vec.size());

    cout<<"R_rel_init: "<<R_rel_eig_init<<endl;
    cout<<"t_rel_init: "<<t_rel_eig_init<<endl;

    double* rotation_array_rel = new double[3];
    double* translatrion_array_rel = new double[3];

    for(uint i = 0;i<X_vec.size();++i)
    {

        //convert eigen to arrary[]
        Eigen::AngleAxisd aa_rel(R_rel_eig_init);
        Eigen::Vector3d v_rel = aa_rel.angle() * aa_rel.axis();

        rotation_array_rel[0] = v_rel.x();
        rotation_array_rel[1] = v_rel.y();
        rotation_array_rel[2] = v_rel.z();

        translatrion_array_rel[0] = t_rel_eig_init.x();
        translatrion_array_rel[1] = t_rel_eig_init.y();
        translatrion_array_rel[2] = t_rel_eig_init.z();

        Eigen::Matrix3d R_eig_l = toEigenMatrixXd(R_vec_l[i]);
        Eigen::Vector3d t_eig_l = toEigenMatrixXd(t_vec_l[i]);

        Eigen::Matrix3d R_eig_r = toEigenMatrixXd(R_vec_r[i]);
        Eigen::Vector3d t_eig_r = toEigenMatrixXd(t_vec_r[i]);


        for(uint j = 0;j<X_vec[i].size();++j)
        {
            Eigen::Vector3d X_world = Eigen::Vector3d(X_vec[i][j].x,X_vec[i][j].y,X_vec[i][j].z);
            Eigen::Vector3d X_cam_l  = R_eig_l * X_world + t_eig_l;
            Eigen::Vector3d X_cam_r  = R_eig_r * X_world + t_eig_r;

            ceres::CostFunction* cost_function =
                new ceres::AutoDiffCostFunction<Ceres_AdjustRt, 3, 3, 3>(new Ceres_AdjustRt(X_cam_l, X_cam_r));
            problem.AddResidualBlock(cost_function, lossFunc, rotation_array_rel, translatrion_array_rel);

        }

    }


    ceres::Solver::Options options;
    options.max_num_iterations = 20;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";

    cout << "FINISH CERES BA" << endl;

    Mat rvec = (Mat_<double>(3,1)<<rotation_array_rel[0],rotation_array_rel[1],rotation_array_rel[2]);
    Mat rotation_matrix = Mat(3, 3, CV_64FC1, Scalar::all(0));  // 图像的旋转矩阵
    Rodrigues(rvec, rotation_matrix);

    Mat tvec = (Mat_<double>(3,1)<<translatrion_array_rel[0],translatrion_array_rel[1],translatrion_array_rel[2]);

    cout<<"BA R: "<<rotation_matrix<<endl;
    cout<<"BA t: "<<tvec<<endl;

    Mat K_l = monocam_left.m_cameraMatrix;
    Mat distor_para_l = monocam_left.m_distCoeffs;

    Mat K_r = monocam_right.m_cameraMatrix;
    Mat distor_para_r = monocam_right.m_distCoeffs;

    //caculate E and F

    Mat t_rel_ant = (Mat_<double>(3,3)<<0, -translatrion_array_rel[2], translatrion_array_rel[1],
            translatrion_array_rel[2], 0, -translatrion_array_rel[0],
            -translatrion_array_rel[1], translatrion_array_rel[0], 0);

    Mat E = t_rel_ant * rotation_matrix;

    Mat F = K_r.t().inv() * E * K_l.inv();

    F= F / F.at<double>(2,2);

    // save intrinsic parameters
    FileStorage fs(outputPath + "/intrinsics.yml", FileStorage::WRITE);
    assert(fs.isOpened());
    if (fs.isOpened())
    {
        fs << "M1" << K_l << "D1" << distor_para_l <<
            "M2" << K_r << "D2" << distor_para_r << "R" << rotation_matrix << "T" << tvec << "E" << E << "F" << F;
        fs.release();
    }
    else
        cout << "Error: can not save the intrinsic parameters\n";



}

/*计算标定板上模块的实际物理坐标*/
void calRealPoint(vector<vector<Point3f>>& obj, int boardwidth, int boardheight, int imgNumber, int squaresize)
{
    vector<Point3f> imgpoint;
    for (int rowIndex = 0; rowIndex < boardheight; rowIndex++)
    {
        for (int colIndex = 0; colIndex < boardwidth; colIndex++)
        {
            imgpoint.emplace_back(Point3f(colIndex * squaresize, rowIndex * squaresize, 0));
        }
    }
    for (int imgIndex = 0; imgIndex < imgNumber; imgIndex++)
    {
        obj.emplace_back(imgpoint);
    }
}


/*设置相机的初始参数 也可以不估计*/
void guessCameraParam(Mat & intrinsic, Mat & distortion_coeff)
{

    /*分配内存*/
    intrinsic.create(3, 3, CV_64FC1);    //相机内参数
    distortion_coeff.create(5, 1, CV_64FC1);  //畸变参数

    /*
    fx 0 cx
    0 fy cy
    0 0  1     内参数
    */
    intrinsic.at<double>(0, 0) = 2091.8;   //fx
    intrinsic.at<double>(0, 2) = 2092.9;   //cx
    intrinsic.at<double>(1, 1) = 1878.8;   //fy
    intrinsic.at<double>(1, 2) = 1122.6;   //cy

    intrinsic.at<double>(0, 1) = 0;
    intrinsic.at<double>(1, 0) = 0;
    intrinsic.at<double>(2, 0) = 0;
    intrinsic.at<double>(2, 1) = 0;
    intrinsic.at<double>(2, 2) = 1;

    /*
    k1 k2 p1 p2 p3    畸变参数
    */
    distortion_coeff.at<double>(0, 0) = 0.0738;  //k1
    distortion_coeff.at<double>(1, 0) = -0.1686;  //k2
    distortion_coeff.at<double>(2, 0) = 0.0002;   //p1
    distortion_coeff.at<double>(3, 0) = 0.0050;   //p2
    distortion_coeff.at<double>(4, 0) = 0.2030;          //p3
}

void outputCameraParam(Mat intrinsic, Mat distortion_coeff, vector<Mat> rvecs, vector<Mat> tvecs)
{  
    /*保存数据*/
    cvSave((OnputDataPath+"/cameraMatrix.xml").c_str(), &intrinsic);
    cvSave((OnputDataPath+"/cameraDistoration.xml").c_str(), &distortion_coeff);
    cvSave((OnputDataPath+"/rotatoVector.xml").c_str(), &rvecs);
    cvSave((OnputDataPath+"/translationVector.xml").c_str(), &tvecs);
    /*输出数据*/
    cout << "fx :" << intrinsic.at<double>(0, 0) << endl << "fy :" << intrinsic.at<double>(1, 1) << endl;
    cout << "cx :" << intrinsic.at<double>(0, 2) << endl << "cy :" << intrinsic.at<double>(1, 2) << endl;//内参数

    cout << "k1 :" << distortion_coeff.at<double>(0, 0) << endl;
    cout << "k2 :" << distortion_coeff.at<double>(1, 0) << endl;
    cout << "p1 :" << distortion_coeff.at<double>(2, 0) << endl;
    cout << "p2 :" << distortion_coeff.at<double>(3, 0) << endl;
    cout << "p3 :" << distortion_coeff.at<double>(4, 0) << endl;   //畸变参数

}

bool sort_by_idx(String s1 ,String s2)
{
    //find_last_of("_"): find the last place of "_"
    //substr() find sub str
     string ext1 = s1.substr(s1.find_last_of('_') + 1,(s1.size()-s1.find_last_of('_') - 4 - 1));
     int num_s1 = atoi(ext1.c_str());
     string ext2 = s2.substr(s2.find_last_of('_') + 1,(s2.size()-s2.find_last_of('_') - 4 - 1));
     int num_s2 = atoi(ext2.c_str());
     return num_s1<num_s2;
}

void sortImagePath(vector<String> & image_path)
{
    //sort by index
    sort(image_path.begin(),image_path.end(),sort_by_idx);
}
