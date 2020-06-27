# include "binocularcalibration.h"

const string path = "/home/jiajie/3d_reco/Binocular/BinocularCalibration1.1/HikivisionImage6-24";  //working dir
const int boardWidth = 11;                               //横向的角点数目
const int boardHeight = 8;                              //纵向的角点数据
const int boardCorner = boardWidth * boardHeight;       //总的角点数据
const int squareSize = 30;                              //标定板黑白格子的大小 单位mm
const Size boardSize = Size(boardWidth, boardHeight);   //总的内角点
const bool isReversecolor = true;

int main(void)
{
    Config cfg = Config(boardWidth, boardHeight, boardCorner, squareSize, boardSize, isReversecolor);
    calibrateBinocamera(path, cfg);
    return 0;
}




