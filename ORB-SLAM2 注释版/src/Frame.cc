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

#include "Frame.h"
#include "Converter.h"
#include "ORBmatcher.h"
#include <thread>

namespace ORB_SLAM2
{

long unsigned int Frame::nNextId=0;
bool Frame::mbInitialComputations=true;
//下边这些都是静态成员变量，因此只能在类外附初值，并且属于整个类，每个对象所共有的
float Frame::cx, Frame::cy, Frame::fx, Frame::fy, Frame::invfx, Frame::invfy;
float Frame::mnMinX, Frame::mnMinY, Frame::mnMaxX, Frame::mnMaxY;
float Frame::mfGridElementWidthInv, Frame::mfGridElementHeightInv;

Frame::Frame()
{}

//Copy Constructor
Frame::Frame(const Frame &frame)
    :mpORBvocabulary(frame.mpORBvocabulary), mpORBextractorLeft(frame.mpORBextractorLeft), mpORBextractorRight(frame.mpORBextractorRight),
     mTimeStamp(frame.mTimeStamp), mK(frame.mK.clone()), mDistCoef(frame.mDistCoef.clone()),
     mbf(frame.mbf), mb(frame.mb), mThDepth(frame.mThDepth), N(frame.N), mvKeys(frame.mvKeys),
     mvKeysRight(frame.mvKeysRight), mvKeysUn(frame.mvKeysUn),  mvuRight(frame.mvuRight),
     mvDepth(frame.mvDepth), mBowVec(frame.mBowVec), mFeatVec(frame.mFeatVec),
     mDescriptors(frame.mDescriptors.clone()), mDescriptorsRight(frame.mDescriptorsRight.clone()),
     mvpMapPoints(frame.mvpMapPoints), mvbOutlier(frame.mvbOutlier), mnId(frame.mnId),
     mpReferenceKF(frame.mpReferenceKF), mnScaleLevels(frame.mnScaleLevels),
     mfScaleFactor(frame.mfScaleFactor), mfLogScaleFactor(frame.mfLogScaleFactor),
     mvScaleFactors(frame.mvScaleFactors), mvInvScaleFactors(frame.mvInvScaleFactors),
     mvLevelSigma2(frame.mvLevelSigma2), mvInvLevelSigma2(frame.mvInvLevelSigma2)
{
    for(int i=0;i<FRAME_GRID_COLS;i++)
        for(int j=0; j<FRAME_GRID_ROWS; j++)
            mGrid[i][j]=frame.mGrid[i][j];

    if(!frame.mTcw.empty())
        SetPose(frame.mTcw);
}


Frame::Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, ORBextractor* extractorLeft, ORBextractor* extractorRight, ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)
    :mpORBvocabulary(voc),mpORBextractorLeft(extractorLeft),mpORBextractorRight(extractorRight), mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
     mpReferenceKF(static_cast<KeyFrame*>(NULL))
{
    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
    thread threadLeft(&Frame::ExtractORB,this,0,imLeft);
    thread threadRight(&Frame::ExtractORB,this,1,imRight);
    threadLeft.join();
    threadRight.join();

    N = mvKeys.size();

    if(mvKeys.empty())
        return;

    UndistortKeyPoints();

    ComputeStereoMatches();

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));    
    mvbOutlier = vector<bool>(N,false);


    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imLeft);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf/fx;

    AssignFeaturesToGrid();
}
// Constructor for RGB-D cameras.  
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
Frame::Frame(const cv::Mat &imGray, const cv::Mat &imDepth, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, 
	     cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)
    :mpORBvocabulary(voc),mpORBextractorLeft(extractor),mpORBextractorRight(static_cast<ORBextractor*>(NULL)),
     mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth)
{
    // Frame ID
    // 定义该帧的ID
    mnId=nNextId++;

    // Scale Level Info  提取器高斯金字塔的相关尺度参数
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction  ORB提取器  提取该帧图像中的特征点以及计算描述子
    ExtractORB(0,imGray);
    // 特征点的数量
    N = mvKeys.size();

    if(mvKeys.empty())
        return;
    //对关键点进行失真矫正
    UndistortKeyPoints();
    //将RGB-D相机的图像数据映射到双目相机下
    ComputeStereoFromRGBD(imDepth);
    //初始化该帧数据中的地图点和局外点，默认无局外点，无地图点
    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
    mvbOutlier = vector<bool>(N,false);

    // This is done only for the first Frame (or after a change in the calibration)   
    // 当第一次初始化帧时，需要将相机的相关参数都加载进来，再有帧时就不需要加载了，提高运行速度
    if(mbInitialComputations)
    {
      //计算图像的边界
        ComputeImageBounds(imGray);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);  //每个网格宽度的倒数
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);  //每个网格高度的倒数

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }
    //计算基线
    mb = mbf/fx;
    //将特征点分配到各个网格，目的是加速特征匹配
    AssignFeaturesToGrid();
}


Frame::Frame(const cv::Mat &imGray, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)
    :mpORBvocabulary(voc),mpORBextractorLeft(extractor),mpORBextractorRight(static_cast<ORBextractor*>(NULL)),
     mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth)
{
    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
    ExtractORB(0,imGray);

    N = mvKeys.size();

    if(mvKeys.empty())
        return;
    //关键点的失真矫正
    UndistortKeyPoints();

    // Set no stereo information 设置非双目信息
    mvuRight = vector<float>(N,-1);
    mvDepth = vector<float>(N,-1);

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
    mvbOutlier = vector<bool>(N,false);

    // This is done only for the first Frame (or after a change in the calibration)   如果是第一帧数据或者在校准之后的发生了点改变
    if(mbInitialComputations)
    {
      //计算图片的边界
        ComputeImageBounds(imGray);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf/fx;
    // 将每个特征点分配到图片网格中
    AssignFeaturesToGrid();
}
/*****************************************************************
*      函数属性：类Frame的成员函数AssignFeaturesToGrid()
 *     函数功能：
 *                 将整张图片分为64×48的网格
 *                 并将每个特征点的id加入到该网格中，即mGrid容器存储的是特征点的id
 *     函数参数介绍：NULL
 *     备注：分配特征点到各个网格，加速特征匹配
 ******************************************************************/
void Frame::AssignFeaturesToGrid()
{
    int nReserve = 0.5f*N/(FRAME_GRID_COLS*FRAME_GRID_ROWS);
    for(unsigned int i=0; i<FRAME_GRID_COLS;i++)
        for (unsigned int j=0; j<FRAME_GRID_ROWS;j++)
            mGrid[i][j].reserve(nReserve);   //给每个网格预留下空间，为什么要预留这些？
    //将特征点分配到这些网格中
    for(int i=0;i<N;i++)
    {
        const cv::KeyPoint &kp = mvKeysUn[i];

        int nGridPosX, nGridPosY;  //存储网格位置，证明第(nGridPosX,nGridPosY)个网格
        if(PosInGrid(kp,nGridPosX,nGridPosY))    //如果第i个特征点位置在第(nGridPosX,nGridPosY)个网格中，就将该特征点的id存入该网格中
            mGrid[nGridPosX][nGridPosY].push_back(i);
    }
}
/*****************************************************************
*      函数属性：类Frame的成员函数ExtractORB(int flag, const cv::Mat &im)
 *     函数功能：
 *                 调用ORB提取器的()运算符，将得到该帧图像的关键点和描述子
 *                 将该帧图像的关键点存储到mvKeys
 *                     该帧图像的描述子存储到mDescriptors
 *     函数参数介绍：
 *                 flag：提取图像的标志  0代表左提取   1代表右提取
 *                 im：待提取ORB特征的图像(灰度图)
 *     备注：NULL
 ******************************************************************/
void Frame::ExtractORB(int flag, const cv::Mat &im)
{
  
    if(flag==0)
        (*mpORBextractorLeft)(im,cv::Mat(),mvKeys,mDescriptors);
    else
        (*mpORBextractorRight)(im,cv::Mat(),mvKeysRight,mDescriptorsRight);
}
/*****************************************************************
*      函数属性：类Frame的成员函数SetPose(cv::Mat Tcw)
 *     函数功能：
 *                 给该帧设置相机位姿（变换矩阵T）
 *     函数参数介绍：
 *                 Tcw：该帧数据的相机位姿
 *     备注：NULL
 ******************************************************************/
void Frame::SetPose(cv::Mat Tcw)
{
    mTcw = Tcw.clone();
    UpdatePoseMatrices();
}
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
void Frame::UpdatePoseMatrices()
{ 
  // mTcw表示从相机坐标系到世界坐标系的变换矩阵   P(世界)=mTcw*P(相机)
  // 计算相机光心时  我们通常将初始帧的相机位移向量为0      那么当前相机光新的位置应该是从世界坐标到当前相机坐标系的位移向量t,
  // 因此我们需要将mTcw求逆变换为从世界坐标系到相机坐标系的变换矩阵
    mRcw = mTcw.rowRange(0,3).colRange(0,3);
    mRwc = mRcw.t();
    mtcw = mTcw.rowRange(0,3).col(3);
    mOw = -mRcw.t()*mtcw;   //从世界坐标系到当前帧相机坐标系的平移矩阵!!!
    //  Tcw.inverse()=[R.t(),-R.t()*tcw;0.t(),1]
}
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
 *     备注：检查 一个地图点是否在该帧数据相机的视锥内
 ******************************************************************/
bool Frame::isInFrustum(MapPoint *pMP, float viewingCosLimit)
{
    pMP->mbTrackInView = false;

    // 3D in absolute coordinates
    //  得到地图点的绝对世界坐标
    cv::Mat P = pMP->GetWorldPos(); 

    // 3D in camera coordinates
    // 得到地图点的相机坐标
    const cv::Mat Pc = mRcw*P+mtcw;
    const float &PcX = Pc.at<float>(0);
    const float &PcY= Pc.at<float>(1);
    const float &PcZ = Pc.at<float>(2);

    // Check positive depth   检测位置深度值
    if(PcZ<0.0f)
        return false;

    // Project in image and check it is not outside   将地图点的相机坐标映射到像素坐标下
    const float invz = 1.0f/PcZ;
    const float u=fx*PcX*invz+cx;
    const float v=fy*PcY*invz+cy;
    //检测像素坐标是否在边界以外
    if(u<mnMinX || u>mnMaxX)
        return false;
    if(v<mnMinY || v>mnMaxY)
        return false;

    // Check distance is in the scale invariance region of the MapPoint
    // 计算世界坐标系下地图点离相机中心的距离，并判断是否在尺度变化的距离内
    const float maxDistance = pMP->GetMaxDistanceInvariance();
    const float minDistance = pMP->GetMinDistanceInvariance();
    const cv::Mat PO = P-mOw;
    const float dist = cv::norm(PO);

    if(dist<minDistance || dist>maxDistance)
        return false;

    // Check viewing angle    检查视角是否符合范围
    cv::Mat Pn = pMP->GetNormal();   //得到pMP地图点的视角

    const float viewCos = PO.dot(Pn)/dist;
    if(viewCos<viewingCosLimit)
        return false;

    // Predict scale in the image
    const int nPredictedLevel = pMP->PredictScale(dist,this);

    // Data used by the tracking   被用在追踪线程的数据   标记该点将来要被投影
    pMP->mbTrackInView = true;    //表示该地图点可以被观察到
    pMP->mTrackProjX = u;             //该地图点在该帧相机的投影像素x坐标
    pMP->mTrackProjXR = u - mbf*invz; //该3D点投影到双目右侧相机上的横坐标
    pMP->mTrackProjY = v;             //该地图点在该帧相机的投影像素y坐标
    pMP->mnTrackScaleLevel= nPredictedLevel;//该地图点在该帧被观察到时在高斯金字塔中的层数
    pMP->mTrackViewCos = viewCos;  //该地图点在该帧中被观察到时的角度

    return true;
}
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
 *     备注：找到在 以x,y为中心,边长为2r的方形内且在[minLevel, maxLevel]的特征点（通过网格查找的方式）
 ******************************************************************/
vector<size_t> Frame::GetFeaturesInArea(const float &x, const float  &y, const float  &r, const int minLevel, const int maxLevel) const
{
    vector<size_t> vIndices;
    vIndices.reserve(N);
    //计算该区域位于的最小网格横坐标
    const int nMinCellX = max(0,(int)floor((x-mnMinX-r)*mfGridElementWidthInv));
    if(nMinCellX>=FRAME_GRID_COLS)
        return vIndices;
    //计算该区域位于的最大网格横坐标
    const int nMaxCellX = min((int)FRAME_GRID_COLS-1,(int)ceil((x-mnMinX+r)*mfGridElementWidthInv));
    if(nMaxCellX<0)
        return vIndices;
    //计算该区域位于的最小网格纵坐标
    const int nMinCellY = max(0,(int)floor((y-mnMinY-r)*mfGridElementHeightInv));
    if(nMinCellY>=FRAME_GRID_ROWS)
        return vIndices;
    //计算该区域位于的最大网格纵坐标
    const int nMaxCellY = min((int)FRAME_GRID_ROWS-1,(int)ceil((y-mnMinY+r)*mfGridElementHeightInv));
    if(nMaxCellY<0)
        return vIndices;

    const bool bCheckLevels = (minLevel>0) || (maxLevel>=0);
    //查找这些（通过上述计算得到的）网格中的特征点，找出在层数范围内并且也在区域范围内的特征点
    for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
    {
        for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
        {  
            const vector<size_t> vCell = mGrid[ix][iy];    //第(ix,iy)个网格特征点序号的集合
            if(vCell.empty())
                continue;
            //遍历这个（第(ix,iy)）网格中所有特征点
            for(size_t j=0, jend=vCell.size(); j<jend; j++)
            {
                const cv::KeyPoint &kpUn = mvKeysUn[vCell[j]];   //得到具体的特征点
                if(bCheckLevels)
                {
		  //kpUn.octave表示这个特征点所在的层
                    if(kpUn.octave<minLevel)
                        continue;
                    if(maxLevel>=0)
                        if(kpUn.octave>maxLevel)
                            continue;
                }
                //剔除在区域范围外的点
                const float distx = kpUn.pt.x-x;
                const float disty = kpUn.pt.y-y;

                if(fabs(distx)<r && fabs(disty)<r)
                    vIndices.push_back(vCell[j]);
            }
        }
    }

    return vIndices;
}
 /*****************************************************************
 *     函数属性：类Frame的成员函数PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY)
 *     函数功能：
 *                 计算特征点所在网格的位置
 *     函数参数介绍：
 *                 kp：特征点
 *                 posX、posY：第(posX,posY)个网格坐标
 *     备注：计算特征点所在网格的位置
 ******************************************************************/
bool Frame::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY)
{
    //计算网格位置
    posX = round((kp.pt.x-mnMinX)*mfGridElementWidthInv);
    posY = round((kp.pt.y-mnMinY)*mfGridElementHeightInv);

    //Keypoint's coordinates are undistorted, which could cause to go out of the image
    //如果特征点的坐标超出边界则返回false
    if(posX<0 || posX>=FRAME_GRID_COLS || posY<0 || posY>=FRAME_GRID_ROWS)
        return false;

    return true;
}
 /*************************************************************************
 *     函数属性：类Frame的成员函数ComputeBoW()
 *     函数功能：
 *                 计算该帧数据描述子对应的BoW向量和Feature向量
 *     函数参数介绍：NULL
 *     备注：计算该帧数据描述子对应的BoW向量和Feature向量
 **************************************************************************/
void Frame::ComputeBoW()
{
    if(mBowVec.empty())  //判断BoW向量是否为空，防止重复运算
    {
        vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
        mpORBvocabulary->transform(vCurrentDesc,mBowVec,mFeatVec,4);
    }
}
/*****************************************************************
*      函数属性：类Frame的成员函数UndistortKeyPoints()
 *     函数功能：
 *                 首先检测是否需要失真矫正
 *                         如若需要利用opencv的函数cv::undistortPoints()对特征点进行矫正
 *                         并将矫正结果存入mvKeysUn容器中
 *                         此时就有：mvKeys容器存储矫正前的结果
 *                                            mvKeysUn容器存储矫正后的结果
 *     函数参数介绍：NULL
 *     备注：NULL
 ******************************************************************/
void Frame::UndistortKeyPoints()
{
    //检测是否需要矫正，如果传入的矫正矩阵第一个参数为0那么就说明传入的图像就是已经被矫正过的
    if(mDistCoef.at<float>(0)==0.0)
    {
        mvKeysUn=mvKeys;
        return;
    }

    // Fill matrix with points  将特征点填充进入mat矩阵中
    cv::Mat mat(N,2,CV_32F);
    for(int i=0; i<N; i++)
    {
        mat.at<float>(i,0)=mvKeys[i].pt.x;
        mat.at<float>(i,1)=mvKeys[i].pt.y;
    }

    // Undistort points   利用opencv的函数对关键点进行失真矫正
    mat=mat.reshape(2);
    cv::undistortPoints(mat,mat,mK,mDistCoef,cv::Mat(),mK);
    mat=mat.reshape(1);

    // Fill undistorted keypoint vector  将失真矫正后的的关键点向量填充进mvKeysUn
    mvKeysUn.resize(N);
    for(int i=0; i<N; i++)
    {
        cv::KeyPoint kp = mvKeys[i];
        kp.pt.x=mat.at<float>(i,0);
        kp.pt.y=mat.at<float>(i,1);
        mvKeysUn[i]=kp;
    }
}
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
void Frame::ComputeImageBounds(const cv::Mat &imLeft)
{
    if(mDistCoef.at<float>(0)!=0.0)  //如果图片需要失真矫正
    {
        // 矫正前四个边界点：(0,0) (cols,0) (0,rows) (cols,rows)
        cv::Mat mat(4,2,CV_32F);
        mat.at<float>(0,0)=0.0; mat.at<float>(0,1)=0.0;
        mat.at<float>(1,0)=imLeft.cols; mat.at<float>(1,1)=0.0;
        mat.at<float>(2,0)=0.0; mat.at<float>(2,1)=imLeft.rows;
        mat.at<float>(3,0)=imLeft.cols; mat.at<float>(3,1)=imLeft.rows;

        // Undistort corners  对rgb图进行失真矫正
        mat=mat.reshape(2);
        cv::undistortPoints(mat,mat,mK,mDistCoef,cv::Mat(),mK);   //将矫正后的图像放入mat中
        mat=mat.reshape(1);

        mnMinX = min(mat.at<float>(0,0),mat.at<float>(2,0));  //左上和左下横坐标最小的
        mnMaxX = max(mat.at<float>(1,0),mat.at<float>(3,0)); //右上和右下横坐标最大的
        mnMinY = min(mat.at<float>(0,1),mat.at<float>(1,1));  //左上和右上纵坐标最小的
        mnMaxY = max(mat.at<float>(2,1),mat.at<float>(3,1)); //左下和右下纵坐标最小的

    }
    else          //如果图片不需要失真矫正
    {
        mnMinX = 0.0f;
        mnMaxX = imLeft.cols;
        mnMinY = 0.0f;
        mnMaxY = imLeft.rows;
    }
}
/*****************************************************************
 *     函数属性：类Frame的成员函数ComputeStereoMatches()
 *     函数功能：
 *                 ？？？？
 *     函数参数介绍：NULL
 *     备注：双目匹配
 ******************************************************************/
void Frame::ComputeStereoMatches()
{
    mvuRight = vector<float>(N,-1.0f);
    mvDepth = vector<float>(N,-1.0f);

    const int thOrbDist = (ORBmatcher::TH_HIGH+ORBmatcher::TH_LOW)/2;

    const int nRows = mpORBextractorLeft->mvImagePyramid[0].rows;

    //Assign keypoints to row table
    vector<vector<size_t> > vRowIndices(nRows,vector<size_t>());

    for(int i=0; i<nRows; i++)
        vRowIndices[i].reserve(200);

    const int Nr = mvKeysRight.size();

    for(int iR=0; iR<Nr; iR++)
    {
        const cv::KeyPoint &kp = mvKeysRight[iR];
        const float &kpY = kp.pt.y;
        const float r = 2.0f*mvScaleFactors[mvKeysRight[iR].octave];
        const int maxr = ceil(kpY+r);
        const int minr = floor(kpY-r);

        for(int yi=minr;yi<=maxr;yi++)
            vRowIndices[yi].push_back(iR);
    }

    // Set limits for search
    const float minZ = mb;
    const float minD = 0;
    const float maxD = mbf/minZ;

    // For each left keypoint search a match in the right image
    vector<pair<int, int> > vDistIdx;
    vDistIdx.reserve(N);

    for(int iL=0; iL<N; iL++)
    {
        const cv::KeyPoint &kpL = mvKeys[iL];
        const int &levelL = kpL.octave;
        const float &vL = kpL.pt.y;
        const float &uL = kpL.pt.x;

        const vector<size_t> &vCandidates = vRowIndices[vL];

        if(vCandidates.empty())
            continue;

        const float minU = uL-maxD;
        const float maxU = uL-minD;

        if(maxU<0)
            continue;

        int bestDist = ORBmatcher::TH_HIGH;
        size_t bestIdxR = 0;

        const cv::Mat &dL = mDescriptors.row(iL);

        // Compare descriptor to right keypoints
        for(size_t iC=0; iC<vCandidates.size(); iC++)
        {
            const size_t iR = vCandidates[iC];
            const cv::KeyPoint &kpR = mvKeysRight[iR];

            if(kpR.octave<levelL-1 || kpR.octave>levelL+1)
                continue;

            const float &uR = kpR.pt.x;

            if(uR>=minU && uR<=maxU)
            {
                const cv::Mat &dR = mDescriptorsRight.row(iR);
                const int dist = ORBmatcher::DescriptorDistance(dL,dR);

                if(dist<bestDist)
                {
                    bestDist = dist;
                    bestIdxR = iR;
                }
            }
        }

        // Subpixel match by correlation
        if(bestDist<thOrbDist)
        {
            // coordinates in image pyramid at keypoint scale
            const float uR0 = mvKeysRight[bestIdxR].pt.x;
            const float scaleFactor = mvInvScaleFactors[kpL.octave];
            const float scaleduL = round(kpL.pt.x*scaleFactor);
            const float scaledvL = round(kpL.pt.y*scaleFactor);
            const float scaleduR0 = round(uR0*scaleFactor);

            // sliding window search
            const int w = 5;
            cv::Mat IL = mpORBextractorLeft->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduL-w,scaleduL+w+1);
            IL.convertTo(IL,CV_32F);
            IL = IL - IL.at<float>(w,w) *cv::Mat::ones(IL.rows,IL.cols,CV_32F);

            int bestDist = INT_MAX;
            int bestincR = 0;
            const int L = 5;
            vector<float> vDists;
            vDists.resize(2*L+1);

            const float iniu = scaleduR0+L-w;
            const float endu = scaleduR0+L+w+1;
            if(iniu<0 || endu >= mpORBextractorRight->mvImagePyramid[kpL.octave].cols)
                continue;

            for(int incR=-L; incR<=+L; incR++)
            {
                cv::Mat IR = mpORBextractorRight->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduR0+incR-w,scaleduR0+incR+w+1);
                IR.convertTo(IR,CV_32F);
                IR = IR - IR.at<float>(w,w) *cv::Mat::ones(IR.rows,IR.cols,CV_32F);

                float dist = cv::norm(IL,IR,cv::NORM_L1);
                if(dist<bestDist)
                {
                    bestDist =  dist;
                    bestincR = incR;
                }

                vDists[L+incR] = dist;
            }

            if(bestincR==-L || bestincR==L)
                continue;

            // Sub-pixel match (Parabola fitting)
            const float dist1 = vDists[L+bestincR-1];
            const float dist2 = vDists[L+bestincR];
            const float dist3 = vDists[L+bestincR+1];

            const float deltaR = (dist1-dist3)/(2.0f*(dist1+dist3-2.0f*dist2));

            if(deltaR<-1 || deltaR>1)
                continue;

            // Re-scaled coordinate
            float bestuR = mvScaleFactors[kpL.octave]*((float)scaleduR0+(float)bestincR+deltaR);

            float disparity = (uL-bestuR);

            if(disparity>=minD && disparity<maxD)
            {
                if(disparity<=0)
                {
                    disparity=0.01;
                    bestuR = uL-0.01;
                }
                mvDepth[iL]=mbf/disparity;
                mvuRight[iL] = bestuR;
                vDistIdx.push_back(pair<int,int>(bestDist,iL));
            }
        }
    }

    sort(vDistIdx.begin(),vDistIdx.end());
    const float median = vDistIdx[vDistIdx.size()/2].first;
    const float thDist = 1.5f*1.4f*median;

    for(int i=vDistIdx.size()-1;i>=0;i--)
    {
        if(vDistIdx[i].first<thDist)
            break;
        else
        {
            mvuRight[vDistIdx[i].second]=-1;
            mvDepth[vDistIdx[i].second]=-1;
        }
    }
}
 /*****************************************************************
 *     函数属性：类Frame的成员函数ComputeStereoFromRGBD(const cv::Mat &imDepth)
 *     函数功能：
 *                 将深度图中特征点的深度值存储到mvDepth容器中
 *                 通过深度值和已知的矫正好的特征点x坐标，来计算右眼坐标，并将其存储到mvuRight容器中
 *                 计算右眼坐标基本原理介绍：mbf=f*b   有公式深度z = f*b/d  ,d=Ul-Ur 代表的视差，因此计算右眼的坐标就有，Ur = Ul-mbf/z（这里用d表示了深度）
 *     函数参数介绍：
 *                 imDepth：深度图
 *     备注：将RGBD相机的数据转到双目相机下
 ******************************************************************/
void Frame::ComputeStereoFromRGBD(const cv::Mat &imDepth)
{
    mvuRight = vector<float>(N,-1);
    mvDepth = vector<float>(N,-1);

    for(int i=0; i<N; i++)
    {
        const cv::KeyPoint &kp = mvKeys[i];
        const cv::KeyPoint &kpU = mvKeysUn[i];

        const float &v = kp.pt.y;
        const float &u = kp.pt.x;

        const float d = imDepth.at<float>(v,u);

        if(d>0)
        {
            mvDepth[i] = d;
            mvuRight[i] = kpU.pt.x-mbf/d;  //计算右眼的坐标  mbf=f*b   有公式深度z = f*b/d  ,d=Ul-Ur 代表的视差，因此计算右眼的坐标就有，Ur = Ul-mbf/z（这里用d表示了深度）
        }
    }
}
/*****************************************************************
 *     函数属性：类Frame的成员函数UnprojectStereo(const int &i)
 *     函数功能：
 *                 首先根据深度值容器得到深度值，然后根据特征点容器得到该特征点的像素坐标，通过相机内参得到相机坐标
 *           进而将相机坐标转换为世界坐标。
 *     函数参数介绍：
 *                 i：所要转换特征点的id
 *     备注：将特征点映射到3D世界坐标
 ******************************************************************/
cv::Mat Frame::UnprojectStereo(const int &i)
{
    const float z = mvDepth[i];
    if(z>0)
    {
        const float u = mvKeysUn[i].pt.x;
        const float v = mvKeysUn[i].pt.y;
        const float x = (u-cx)*z*invfx;
        const float y = (v-cy)*z*invfy;
        cv::Mat x3Dc = (cv::Mat_<float>(3,1) << x, y, z);  //相机坐标
        return mRwc*x3Dc+mOw;    //世界坐标
    }
    else
        return cv::Mat();
}

} //namespace ORB_SLAM
