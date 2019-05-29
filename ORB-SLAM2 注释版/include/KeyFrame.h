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

#ifndef KEYFRAME_H
#define KEYFRAME_H

#include "MapPoint.h"
#include "Thirdparty/DBoW2/DBoW2/BowVector.h"
#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"
#include "ORBVocabulary.h"
#include "ORBextractor.h"
#include "Frame.h"
#include "KeyFrameDatabase.h"

#include <mutex>


namespace ORB_SLAM2
{

class Map;
class MapPoint;
class Frame;
class KeyFrameDatabase;

class KeyFrame
{
public:
/*******************************************************************************
 *     函数属性：类KeyFrame的构造函数
 *     函数功能：
 *                        构造关键帧
 *     函数参数介绍：
 *                         F：需要加入关键帧的帧
 *                         pMap：地图
 *                         pKFDB：关键帧数据集
 *     备注：NULL
 ******************************************************************************/
    KeyFrame(Frame &F, Map* pMap, KeyFrameDatabase* pKFDB);

    // Pose functions
    // 设置位姿，包括从世界坐标到相机坐标的变换矩阵Tcw 相机中心Ow 从相机坐标到世界坐标的变换矩阵Twc  ，双目相机的中心 Cw
    void SetPose(const cv::Mat &Tcw);
    //得到从世界坐标到相机坐标的变换矩阵Tcw
    cv::Mat GetPose();
    //得到从相机坐标到世界坐标的变换矩阵Twc
    cv::Mat GetPoseInverse();
    // 得到相机中心
    cv::Mat GetCameraCenter();
    //得到双目相机中心
    cv::Mat GetStereoCenter();
    // 得到旋转矩阵Rcw
    cv::Mat GetRotation();
    //得到平移矩阵tcw
    cv::Mat GetTranslation();

    // Bag of Words Representation //计算BoW向量和Feature向量
    void ComputeBoW();

    // Covisibility graph functions
    // 添加与该关键帧相关联的关键帧及其权重
    void AddConnection(KeyFrame* pKF, const int &weight);
    //擦除与该关键帧相关联的关键帧pKF
    void EraseConnection(KeyFrame* pKF);
/**********************************************************************************************************************
 * 函数属性：KeyFrame类成员函数UpdateConnections()
 * 函数功能：
 * 1. 首先获得该关键帧的所有MapPoint点，统计观测到这些3d点的每个关键与其它所有关键帧之间的共视程度
 *    对每一个找到的关键帧，建立一条边，边的权重是该关键帧与当前关键帧公共3d点的个数。
 * 2. 并且该权重必须大于一个阈值，如果没有超过该阈值的权重，那么就只保留权重最大的边（与其它关键帧的共视程度比较高）
 * 3. 对这些连接按照权重从大到小进行排序，以方便将来的处理
 *    更新完covisibility图之后，如果没有初始化过，则初始化为连接权重最大的边（与其它关键帧共视程度最高的那个关键帧），类似于最大生成树
 * 函数参数：NULL
 * 备注：NULL
 **********************************************************************************************************************/
    void UpdateConnections();
    // 将与该关键帧相关联的关键帧序列根据权重进行排序，将排序之后的关键帧和权重存储到mvpOrderedConnectedKeyFrames和mvOrderedWeights中
    void UpdateBestCovisibles();
    // 得到相关联的关键帧(关联关键帧是指权重大于15的共视关键帧,也就是有15个以上的共同地图点)
    std::set<KeyFrame *> GetConnectedKeyFrames();
    // 返回根据权重排序好的关键帧序列
    std::vector<KeyFrame* > GetVectorCovisibleKeyFrames();
    // 返回最好的（权重最大的）与该关键帧相关联的关键帧序列
    std::vector<KeyFrame*> GetBestCovisibilityKeyFrames(const int &N);
    // 返回权重大于w的关键帧
    std::vector<KeyFrame*> GetCovisiblesByWeight(const int &w);
    //得到帧pKF的权重
    int GetWeight(KeyFrame* pKF);

    // Spanning tree functions    给pKF添加孩子  孩子证明本关键点是pKF的父节点，即权重最大的关联关键帧
    void AddChild(KeyFrame* pKF);
    //给pKF删除孩子  孩子证明本关键点是pKF的父节点，即权重最大的关联关键帧
    void EraseChild(KeyFrame* pKF);
    // 将父节点改变为pKF并给pKF添加子节点为本关键帧   父节点是值与本节点最大关联关键帧
    void ChangeParent(KeyFrame* pKF);
    // 返回本关键帧的所有的孩子，也就是本关键帧为哪些关键帧的最大关联关键帧
    std::set<KeyFrame*> GetChilds();
    // 返回父关键帧 父关键帧为本关键帧的最大关联关键帧
    KeyFrame* GetParent();
    // 检查该关键帧是否有孩子，即该关键帧是否是其他关键帧的最大关联关键帧
    bool hasChild(KeyFrame* pKF);

    // Loop Edges  添加回环边  pKF与本关键帧形成回环
    void AddLoopEdge(KeyFrame* pKF);
    //返回该关键帧的回环关键帧
    std::set<KeyFrame*> GetLoopEdges();

    // MapPoint observation functions   添加地图点pMP及其索引idx
    void AddMapPoint(MapPoint* pMP, const size_t &idx);
    // 擦除索引为idx的地图点
    void EraseMapPointMatch(const size_t &idx);
    // 擦除地图点pMP及其在关键帧中的索引
    void EraseMapPointMatch(MapPoint* pMP);
    //替换该关键帧相关的地图点及其索引（有点儿bug）
    void ReplaceMapPointMatch(const size_t &idx, MapPoint* pMP);
    //得到与该关键帧相关联的地图点的集合
    std::set<MapPoint*> GetMapPoints();
    // 返回与该关键帧相关的地图点
    std::vector<MapPoint*> GetMapPointMatches();
    // 该关键帧相关的地图点中被观察到的次数大于minObs的地图点个数
    int TrackedMapPoints(const int &minObs);
    //得到在该关键帧中索引为idx的地图点
    MapPoint* GetMapPoint(const size_t &idx);

    // KeyPoint functions  在以(x,y)为中心,2r为边长的正方形区域内得到特征点的序列
    std::vector<size_t> GetFeaturesInArea(const float &x, const float  &y, const float  &r) const;
    // 将该关键帧的第i个特征点投影到世界坐标系下
    cv::Mat UnprojectStereo(int i);

    // Image  判断坐标为(x,y)的点是否在图片内
    bool IsInImage(const float &x, const float &y) const;

    // Enable/Disable bad flag changes  设置该关键帧不可被擦除
    void SetNotErase();
    // 设置该关键帧可擦除
    void SetErase();

    // Set/check bad flag
/*************************************************************************************
 *      函数属性：KeyFrame成员函数SetBadFlag()
 *      函数功能：
 *              （1）验证该帧是否可以被擦除
 *              （2）擦除所有本关键帧与关联关键帧之间的关联
 *              （3）擦除所有地图点与本关键帧之间的关联，标志本关键帧已经不能看到这些地图点,这些地图点也不会存在这些关键帧
 *              （4）清空存储与本关键帧关联的其他关键帧变量，清空排序之后的关联关键帧序列
 *              （5）清空子关键帧   并找每个子关键帧的新的父关键帧
 *              （6）在地图中和关键帧数据集中剔除本关键帧
 *      函数参数：NULL
 *      备注：删除该关键帧
 * 
 *****************************************************************************************/
    void SetBadFlag();
    // 检测该关键帧是否是好的
    bool isBad();

    // Compute Scene Depth (q=2 median). Used in monocular.   计算当前关键帧的场景深度  q=2代表中值 (该关键中所有地图点的中值)
    float ComputeSceneMedianDepth(const int q);
   //是否a>b
    static bool weightComp( int a, int b){
        return a>b;
    }
    //是否pKF1帧的ID小于pKF2的ID
    static bool lId(KeyFrame* pKF1, KeyFrame* pKF2){
        return pKF1->mnId<pKF2->mnId;
    }


    // The following variables are accesed from only 1 thread or never change (no mutex needed).
public:
    //下一关键帧的Id号
    static long unsigned int nNextId;
    // 此关键帧帧在关键帧中的Id
    long unsigned int mnId;
    // 此关键帧在原始帧的Id
    const long unsigned int mnFrameId;
    // 此帧的时间戳
    const double mTimeStamp;

    // Grid (to speed up feature matching)    网格（加速特征匹配）
    const int mnGridCols;    //网格列数
    const int mnGridRows;  //网格行数
    const float mfGridElementWidthInv;     //网格宽度的逆
    const float mfGridElementHeightInv;    //网格高度的逆

    // Variables used by the tracking
    long unsigned int mnTrackReferenceForFrame;
    long unsigned int mnFuseTargetForKF;

    // Variables used by the local mapping
    long unsigned int mnBALocalForKF;
    long unsigned int mnBAFixedForKF;

    // Variables used by the keyframe database
    // 回环帧的id(与当前帧产生回环的帧的id)
    long unsigned int mnLoopQuery;
    //  当前帧和回环帧的同属同一节点的单词数目
    int mnLoopWords;
    //  当前帧和回环帧的BOW得分
    float mLoopScore;
    //  待重定位帧的id
    long unsigned int mnRelocQuery ;
    //  当前关键帧与待重定位的帧相同节点数
    int mnRelocWords;
    //  重定位得分
    float mRelocScore;

    // Variables used by loop closing
    cv::Mat mTcwGBA;
    cv::Mat mTcwBefGBA;
    long unsigned int mnBAGlobalForKF;

    // Calibration parameters   相机的相关参数
    const float fx, fy, cx, cy, invfx, invfy, mbf, mb, mThDepth;

    // Number of KeyPoints   此关键帧中关键点的数量
    const int N;

    // KeyPoints, stereo coordinate and descriptors (all associated by an index)
    // 矫正前的关键点
    const std::vector<cv::KeyPoint> mvKeys;
    // 矫正后的关键点
    const std::vector<cv::KeyPoint> mvKeysUn;
    // 关键点的右眼坐标
    const std::vector<float> mvuRight; // negative value for monocular points
    // 关键点的深度
    const std::vector<float> mvDepth; // negative value for monocular points
    // 关键点的描述子
    const cv::Mat mDescriptors;

    //BoW  该关键点描述子对应的BoW向量
    DBoW2::BowVector mBowVec ;
    // 图片所有描述子对应的特征向量  两个参数,第一个参数是该特征点对应词典中的节点id  第二个参数是该特征点的特征索引
    DBoW2::FeatureVector mFeatVec;

    // Pose relative to parent (this is computed when bad flag is activated)  与父关键帧之间的变换矩阵
    cv::Mat mTcp;

    // Scale  高斯金字塔尺度相关的参数
    // 高斯金字塔的层数
    const int mnScaleLevels;
    //  高斯金字塔每层之间的缩放比例
    const float mfScaleFactor;
    const float mfLogScaleFactor;
    const std::vector<float> mvScaleFactors;
    const std::vector<float> mvLevelSigma2;
    const std::vector<float> mvInvLevelSigma2;

    // Image bounds and calibration   图片的边界
    const int mnMinX;
    const int mnMinY;
    const int mnMaxX;
    const int mnMaxY;
    // 相机内参数矩阵
    const cv::Mat mK;


    // The following variables need to be accessed trough a mutex to be thread safe.
protected:

    // SE3 Pose and camera center
    // 从世界坐标到相机坐标的变换矩阵
    cv::Mat Tcw;
    // 从相机坐标到世界坐标的变换矩阵
    cv::Mat Twc;
    // 相机中心
    cv::Mat Ow;

    cv::Mat Cw; // Stereo middel point. Only for visualization   双目相机的中心坐标

    // MapPoints associated to keypoints  和该关键帧相关的地图点
    std::vector<MapPoint*> mvpMapPoints;

    // BoW   关键帧数据集
    KeyFrameDatabase* mpKeyFrameDB;
    // ORB字典
    ORBVocabulary* mpORBvocabulary;

    // Grid over the image to speed up feature matching    加速特征提取的网格
    std::vector< std::vector <std::vector<size_t> > > mGrid;
    //与该关键帧链接的关键帧及其权重(权重大于15 的关键帧)   权重为其它关键帧与当前关键帧共视3d点的个数
    std::map<KeyFrame*,int> mConnectedKeyFrameWeights;
    // 与当前关键帧相关的根据权重排序之后关键帧序列
    std::vector<KeyFrame*> mvpOrderedConnectedKeyFrames;
    // 排序之后的权重序列
    std::vector<int> mvOrderedWeights;

    // Spanning Tree and Loop Edges
    //第一次建立链接
    bool mbFirstConnection;
    // 存储与该关键帧最相关的其他关键帧（权重最大，共视点最多）  在UpdateConnections()中更新
    KeyFrame* mpParent;
    std::set<KeyFrame*> mspChildrens;   // 存储该节点是哪几个关键帧的最大权重关联关键帧
    std::set<KeyFrame*> mspLoopEdges;  //回环边的关键帧容器，与本帧形成回环的边

    // Bad flags
    bool mbNotErase;   //该关键帧是否可被删除
    bool mbToBeErased;   //该关键帧是否将要被删除
    bool mbBad;    //该关键帧是否已经被删除
    //基线的一半
    float mHalfBaseline; // Only for visualization
   //地图
    Map* mpMap;

    std::mutex mMutexPose;   //关键帧位姿锁
    std::mutex mMutexConnections;  //保护mvpOrderedConnectedKeyFrames相关数据
    std::mutex mMutexFeatures;
};

} //namespace ORB_SLAM

#endif // KEYFRAME_H
