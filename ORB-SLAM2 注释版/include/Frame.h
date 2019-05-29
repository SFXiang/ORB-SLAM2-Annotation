/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef FRAME_H
#define FRAME_H

#include<vector>

#include "MapPoint.h"
#include "Thirdparty/DBoW2/DBoW2/BowVector.h"
#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"
#include "ORBVocabulary.h"
#include "KeyFrame.h"
#include "ORBextractor.h"

#include <opencv2/opencv.hpp>

namespace ORB_SLAM2
{
#define FRAME_GRID_ROWS 48
#define FRAME_GRID_COLS 64

class MapPoint;
class KeyFrame;

class Frame
{
public:
    Frame();

    // Copy constructor.   拷贝构造函数
    Frame(const Frame &frame);

    // Constructor for stereo cameras.  双目相机的构造函数
    Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, ORBextractor* extractorLeft, ORBextractor* extractorRight, ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth);

    // Constructor for RGB-D cameras.  RGBD相机的构造函数
 /****************************************************************************************
 *     函数属性：类Frame的构造函数（RGB-D 相机）
 *     函数功能：
 *                 1. 初始化该帧的ID
 *                 2. 初始化高斯金字塔的尺度参数
 *                 3. 提取图像的特征点
 *                 4. 对特征点进行失真矫正，并将相机的深度图特征点的深度值存储到容器中，便于调用
 *                 5.初始化该帧数据的地图点和局外点
 *                 6.如果为初始帧则将相机的相关参数重新加载进来
 *                 7.将特征点加入到网格中
 *     函数参数介绍：
 *                 imGray：是指该帧的rgb图对应的灰度图
 *                 imDepth：是指该帧的深度图
 *                 timeStamp：是获取该帧数据的时间
 *                 extractor：是指该帧数据的ORB提取器
 *                 voc：是存储字典的首地址
 *                 distCoef：是指相机的参数矩阵
 *                 bf：是指数据集图片尺寸可能是通过缩放得到的，这个就是缩放的尺寸
 *                 thDepth： 远近点的阈值
 *     备注：NULL
 *****************************************************************************************/
    Frame(const cv::Mat &imGray, const cv::Mat &imDepth, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth);

    // Constructor for Monocular cameras.  单目相机的构造函数
    Frame(const cv::Mat &imGray, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth);

 // Extract ORB on the image. 0 for left image and 1 for right image.  
 /*****************************************************************
 *     函数属性：类Frame的成员函数ExtractORB(int flag, const cv::Mat &im)
 *     函数功能：
 *                 调用ORB提取器的()运算符，将得到该帧图像的关键点和描述子
 *                 将该帧图像的关键点存储到mvKeys
 *                     该帧图像的描述子存储到mDescriptors
 *     函数参数介绍：
 *                 flag：提取图像的标志  0代表左提取   1代表右提取
 *                 im：待提取ORB特征的图像(灰度图)
 *     备注：NULL
 ******************************************************************/
    void ExtractORB(int flag, const cv::Mat &im);

    // Compute Bag of Words representation.计算词袋特征
 /*************************************************************************
 *     函数属性：类Frame的成员函数ComputeBoW()
 *     函数功能：
 *                 计算该帧数据描述子对应的BoW向量和Feature向量
 *     函数参数介绍：NULL
 *     备注：计算该帧数据描述子对应的BoW向量和Feature向量
 **************************************************************************/
    void ComputeBoW();

    // Set the camera pose.  
/*****************************************************************
 *     函数属性：类Frame的成员函数SetPose(cv::Mat Tcw)
 *     函数功能：
 *                 给该帧设置相机位姿（变换矩阵T）
 *     函数参数介绍：
 *                 Tcw：该帧数据的相机位姿
 *     备注：NULL
 ******************************************************************/
    void SetPose(cv::Mat Tcw);

    // Computes rotation, translation and camera center matrices from the camera pose.  
/*****************************************************************
 *     函数属性：类Frame的成员函数UpdatePoseMatrices()
 *     函数功能：
 *                 根据Tcw更新从当前帧相机坐标系到世界坐标系的旋转矩阵mRwc;
 *                                       从当前帧相机坐标系到世界坐标系的平移矩阵mOw;
 *                                       从世界坐标系到当前帧相机坐标系的旋转矩阵mRcw;
 *                                       从世界坐标系到当前帧相机坐标系的平移矩阵tcw;
 *                 计算旋转矩阵、平移矩阵的逆的时候方法根据：Tcw.inverse()=[R.t(),-R.t()*tcw;0.t(),1]矩阵，
 *                 注意平移矩阵的逆的计算方法！！！
 *     函数参数介绍：
 *                 Tcw：该帧数据的相机位姿
 *     备注：NULL
 ******************************************************************/
    void UpdatePoseMatrices();

    // Returns the camera center.    返回相机中心
    inline cv::Mat GetCameraCenter(){
        return mOw.clone();
    }

    // Returns inverse of rotation 返回旋转矩阵的逆
    inline cv::Mat GetRotationInverse(){
        return mRwc.clone();
    }

    // Check if a MapPoint is in the frustum of the camera
    // and fill variables of the MapPoint to be used by the tracking
/*****************************************************************
 *     函数属性：类Frame的成员函数isInFrustum(MapPoint *pMP, float viewingCosLimit)
 *     函数功能：
 *                 （1）根据地图点的世界坐标得到地图点的相机坐标，验证深度值
 *                 （2）根据地图点的相机坐标得到地图点的像素坐标，检测像素坐标是否在边界内
 *                 （3）计算地图点离相机中心的距离和角度是否符合设定的最大最小值
 *                 （4）如果都符合要求，就给地图点被用在追踪线程的数据赋值，标记该点将来要被投影
 *     函数参数介绍：
 *                pMP：待检测地图点
 *                viewingCosLimit：最大视角    也就是视锥中母线与中心线所呈的夹角
 *     备注：检查 一个地图点是否在相机的视锥内
 ******************************************************************/
    bool isInFrustum(MapPoint* pMP, float viewingCosLimit);

    // Compute the cell of a keypoint (return false if outside the grid) 
 /*****************************************************************
 *     函数属性：类Frame的成员函数PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY)
 *     函数功能：
 *                 计算特征点所在网格的位置
 *     函数参数介绍：
 *                 kp：特征点
 *                 posX、posY：第(posX,posY)个网格坐标
 *     备注：计算特征点所在网格的位置
 ******************************************************************/
    bool PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY);
/*****************************************************************
 *     函数属性：类Frame的成员函数GetFeaturesInArea(const float &x, const float  &y, const float  &r, const int minLevel, const int maxLevel) const
 *     函数功能：
 *                 （1）计算该区域所占据的最大最小网格点
 *                 （2）循环所有找到的网格点内的所有特征点
 *                              并剔除所有不符合要求的特征点（包括不在规定金字塔层数和不在范围内的）
 *                              返回满足条件特征点的序号
 *     函数参数介绍：
 *                x，y：区域的中心坐标（ x，y）
 *                r：边长的一半
 *                minLevel：所要提取特征点所在金字塔中的最小层数
 *                maxLevel：所要提取特征点所在金字塔中的最大层数
 *                返回满足条件特征点的序号
 *     备注：找到在 以x,y为中心,边长为2r的方形内且在[minLevel, maxLevel]的特征点
 ******************************************************************/
    vector<size_t> GetFeaturesInArea(const float &x, const float  &y, const float  &r, const int minLevel=-1, const int maxLevel=-1) const;

    // Search a match for each keypoint in the left image to a keypoint in the right image.
    // If there is a match, depth is computed and the right coordinate associated to the left keypoint is stored.
    // 左眼图片和右眼图片的匹配   如果匹配成功，被匹配的深度值和与左眼特征点相关的右眼坐标存储起来
    void ComputeStereoMatches();

    // Associate a "right" coordinate to a keypoint if there is valid depth in the depthmap.
 /*****************************************************************
 *     函数属性：类Frame的成员函数ComputeStereoFromRGBD(const cv::Mat &imDepth)
 *     函数功能：
 *                 将深度图中的深度值存储到mvDepth容器中
 *                 通过深度值和已知的矫正好的特征点x坐标，来计算右眼坐标，并将其存储到mvuRight容器中
 *                 计算右眼坐标基本原理介绍：mbf=f*b   有公式深度z = f*b/d  ,d=Ul-Ur 代表的视差，因此计算右眼的坐标就有，Ur = Ul-mbf/z（这里用d表示了深度）
 *     函数参数介绍：
 *                 imDepth：深度图
 *     备注：将RGBD相机的数据转到双目相机下
 ******************************************************************/
    void ComputeStereoFromRGBD(const cv::Mat &imDepth);

    // Backprojects a keypoint (if stereo/depth info available) into 3D world coordinates.
/*****************************************************************
 *     函数属性：类Frame的成员函数UnprojectStereo(const int &i)
 *     函数功能：
 *                 首先根据深度值容器得到深度值，然后根据特征点容器得到该特征点的像素坐标，通过相机内参得到相机坐标
 *           进而将相机坐标转换为世界坐标。
 *     函数参数介绍：
 *                 i：所要转换特征点的id
 *     备注：将特征点映射到3D世界坐标
 ******************************************************************/
    cv::Mat UnprojectStereo(const int &i);

public:
    // Vocabulary used for relocalization. 
    ORBVocabulary* mpORBvocabulary;

    // Feature extractor. The right is used only in the stereo case.
    ORBextractor* mpORBextractorLeft, *mpORBextractorRight;

    // Frame timestamp.此帧的时间戳
    double mTimeStamp;

    // Calibration matrix and OpenCV distortion parameters.    相机的相关参数（内参数矩阵以及相机矫正矩阵）
    cv::Mat mK;
    static float fx;
    static float fy;
    static float cx;
    static float cy;
    static float invfx;
    static float invfy;
    cv::Mat mDistCoef;

    // Stereo baseline multiplied by fx.     双目相机的基线*fx
    float mbf;

    // Stereo baseline in meters.     双目相机的基线
    float mb;

    // Threshold close/far points. Close points are inserted from 1 view.
    // Far points are inserted as in the monocular case from 2 views.  
    //远近点的阈值，近的点是从1帧中插入的，远的点是从2帧数据插入的
    float mThDepth;

    // Number of KeyPoints.     关键点的数量
    int N;

    // Vector of keypoints (original for visualization) and undistorted (actually used by the system).
    // In the stereo case, mvKeysUn is redundant as images must be rectified.
    // In the RGB-D case, RGB images can be distorted.在RGBD相机下，RGB图片必须被矫正
    // 矫正前的特征点
    std::vector<cv::KeyPoint> mvKeys, mvKeysRight;
    std::vector<cv::KeyPoint> mvKeysUn;  //放矫正后的关键点坐标

    // Corresponding stereo coordinate and depth for each keypoint.
    // "Monocular" keypoints have a negative value.
    std::vector<float> mvuRight;
    // 存储每一个关键点的深度值
    std::vector<float> mvDepth;

    // Bag of Words Vector structures.
    // BoW 向量
    DBoW2::BowVector mBowVec;
    DBoW2::FeatureVector mFeatVec;

    // ORB descriptor, each row associated to a keypoint.  ORB描述子，每一个描述子都关联一个特征点
    cv::Mat mDescriptors, mDescriptorsRight;

    // MapPoints associated to keypoints, NULL pointer if no association.   
    // 存储该帧数据的地图点，若该帧数据无地图点则返回空 
    std::vector<MapPoint*> mvpMapPoints;

    // Flag to identify outlier associations.
    // 存储该帧数据的局外点
    std::vector<bool> mvbOutlier;

    // Keypoints are assigned to cells in a grid to reduce matching complexity when projecting MapPoints.
    //每个网格宽度的逆
    static float mfGridElementWidthInv;
    //每个网格高度的逆
    static float mfGridElementHeightInv;
    //存储特征点的网格
    std::vector<std::size_t> mGrid[FRAME_GRID_COLS][FRAME_GRID_ROWS];

    // Camera pose.该帧的相机位姿
    cv::Mat mTcw;

    // Current and Next Frame id.  
    //  定义静态函数使得所有该类的对象都可以使用这个成员函数，给帧计数
    static long unsigned int nNextId;
    //  该帧对应的帧数
    long unsigned int mnId;

    // Reference Keyframe.   该帧对应的参考关键帧
    KeyFrame* mpReferenceKF;

    // Scale pyramid info.   尺度金字塔的信息
    int mnScaleLevels;
    float mfScaleFactor;
    float mfLogScaleFactor;
    vector<float> mvScaleFactors;
    vector<float> mvInvScaleFactors;
    vector<float> mvLevelSigma2;
    vector<float> mvInvLevelSigma2;

    // Undistorted Image Bounds (computed once).不失真图像的边界
    static float mnMinX;
    static float mnMaxX;
    static float mnMinY;
    static float mnMaxY;
    // 查看是否为初始帧，如果为初始帧那么就加载相机的相关参数（内参数矩阵等）
    static bool mbInitialComputations;


private:

    // Undistort keypoints given OpenCV distortion parameters.
    // Only for the RGB-D case. Stereo must be already rectified!
    // (called in the constructor).对关键点进行失真矫正（在构造函数中调用）
/*****************************************************************
 *     函数属性：类Frame的成员函数UndistortKeyPoints()
 *     函数功能：
 *                 首先检测是否需要失真矫正
 *                         如若需要利用opencv的函数cv::undistortPoints()对特征点进行矫正
 *                         并将矫正结果存入mvKeysUn容器中
 *                         此时就有：mvKeys容器存储矫正前的结果
 *                                            mvKeysUn容器存储矫正后的结果
 *     函数参数介绍：NULL
 *     备注：NULL
 ******************************************************************/
    void UndistortKeyPoints();
// Computes image bounds for the undistorted image (called in the constructor). 给不失真图像计算图像边界（构造函数中调用）
/*****************************************************************
 *     函数属性：类Frame的成员函数ComputeImageBounds(const cv::Mat &imLeft)
 *     函数功能：
 *                 函数分为两部分，一部分是当图片需要矫正时：图像的边界为矫正后的图像边界
 *                 第二部分是函数不需要矫正时 图像的边界就是原图像的边界
 *                 此函数的最终结果为：将图形的边界赋值即mnMinX、mnMaxX、mnMinY、mnMaxY
 *     函数参数介绍：
 *                         imLeft：图像彩色图对应的灰度图
 *     备注：计算图像边界（在初始化时调用）
 ******************************************************************/
    void ComputeImageBounds(const cv::Mat &imLeft);

    // Assign keypoints to the grid for speed up feature matching (called in the constructor).  
/*****************************************************************
 *     函数属性：类Frame的成员函数AssignFeaturesToGrid()
 *     函数功能：
 *                 将整张图片分为64×48的网格
 *                 并将每个特征点的id加入到该网格中，即mGrid容器存储的是特征点的id
 *     函数参数介绍：NULL
 *     备注：分配特征点到各个网格，加速特征匹配
 ******************************************************************/
    void AssignFeaturesToGrid();

    // Rotation, translation and camera center   相机的旋转阵和平移阵
    //从世界坐标系到当前帧相机坐标系的旋转矩阵
    cv::Mat mRcw;
    //从世界坐标系到当前帧相机坐标系的平移矩阵
    cv::Mat mtcw;
    //从当前帧相机坐标系到世界坐标系的旋转矩阵
    cv::Mat mRwc;
    //从当前帧相机坐标系到世界坐标系的平移矩阵
    cv::Mat mOw; //==mtwc
};

}// namespace ORB_SLAM

#endif // FRAME_H
