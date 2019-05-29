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

#ifndef MAPPOINT_H
#define MAPPOINT_H

#include"KeyFrame.h"
#include"Frame.h"
#include"Map.h"

#include<opencv2/core/core.hpp>
#include<mutex>

namespace ORB_SLAM2
{

class KeyFrame;
class Map;
class Frame;


class MapPoint
{
public:
/*******************************************************************************
 *     函数属性：类MapPoint的构造函数MapPoint(const cv::Mat &Pos, KeyFrame *pRefKF, Map* pMap):
 *     函数功能：
 *                 
 *     函数参数介绍：
 *                Pos：地图点的位置（世界坐标）
 *                pRefKF：参考关键帧
 *                pMap：地图
 *     备注：给定点的世界坐标和关键帧构造地图点
 * 
 ******************************************************************************/
    MapPoint(const cv::Mat &Pos, KeyFrame* pRefKF, Map* pMap);
/*******************************************************************************
 *     函数属性：类MapPoint的构造函数MapPoint(const cv::Mat &Pos, KeyFrame *pRefKF, Map* pMap):
 *     函数功能：
 *                 
 *     函数参数介绍：
 *                Pos：地图点的位置（世界坐标）
 *                pMap：地图
 *                pFrame：帧
 *                idxF：该地图点在该帧的索引id
 *     备注：给定点的世界坐标和帧构造地图点
 * 
 ******************************************************************************/
    MapPoint(const cv::Mat &Pos,  Map* pMap, Frame* pFrame, const int &idxF);
//设置该地图点的世界坐标
    void SetWorldPos(const cv::Mat &Pos);
    //得到该地图点的世界坐标
    cv::Mat GetWorldPos();
    //得到该地图点在该所有关键帧下的平均观测方向
    cv::Mat GetNormal();
    //得到参考关键帧
    KeyFrame* GetReferenceKeyFrame();
    //得到Observations  Observation存储的是可以看到该地图点的所有关键帧的集合
    std::map<KeyFrame*,size_t> GetObservations();
    //观察到该地图点的关键帧数量
    int Observations();
   //为该地图点的Observation添加关键帧  size_t idx是指该地图点在关键帧的索引 Observation存储的是可以看到该地图点的所有关键帧的集合
    // 在添加地图点到全局地图map时调用
    void AddObservation(KeyFrame* pKF,size_t idx);
    //为该地图点的Observation删除关键帧  Observation存储的是可以看到该地图点的所有关键帧的集合
    void EraseObservation(KeyFrame* pKF);
    //在关键帧pKF中该地图点的索引
    int GetIndexInKeyFrame(KeyFrame* pKF);
    // 判断地图点是否pKF关键帧中
    bool IsInKeyFrame(KeyFrame* pKF);
    // 如果mObservations擦除关键帧失败
    void SetBadFlag();
    //查看该地图点是否是有问题的
    bool isBad();
    // 用pMP地图点来替换本地图点，并消除本地图点
/*
 * 	将本地图点的被观察次数,被查找次数,以及观察到该地图点的关键帧都清空,坏地图点标志置位
 * 	将本地图点的被观察次数,被查找次数都加到替换地图点pMP中,并将当前地图点在关键帧中的位置用代替地图点代替
 * 	最后将本地图点从全局地图map中删除
 */
    void Replace(MapPoint* pMP);    
    //得到该地图点的替代地图点
    MapPoint* GetReplaced();
    // 增加该地图点被看到次数n
    void IncreaseVisible(int n=1);
    // 增加该地图点被查找次数n
    void IncreaseFound(int n=1);
    // 返回该地图点的查找率
    float GetFoundRatio();
    // 返回该地图点被查找到的次数
    inline int GetFound(){
        return mnFound;
    }
/*******************************************************************************
 *     函数属性：类MapPoint的成员函数ComputeDistinctiveDescriptors()
 *     函数功能：
 *                 1、检查该地图点在各个关键帧中的描述子
 *                 2、如果该关键帧没有问题，那么将该关键中该地图点的描述子存入vDescriptors容器中
 *                 3、计算所有被找到描述子之间的距离，并将其距离存入到Distances数组中
 *                 4、取第i个描述子与其他描述子距离的中值作为其均值参考，然后选出这N个中值中最小的，认为该描述子与其他描述子的距离和最近，认为该描述子可以代表本地图点
 *     函数参数介绍：NULL
 *     备注： 计算最优描述子
 * 
 ******************************************************************************/
    void ComputeDistinctiveDescriptors();
    //得到该地图点的最优描述子，所谓最优就是在观察到该地图点的所有关键帧中描述子距离的中值描述子 详情见函数ComputeDistinctiveDescriptors()
    cv::Mat GetDescriptor();
    //更新地图点被观察的平均方向和观测距离范围
    void UpdateNormalAndDepth();
    //返回最小距离
    float GetMinDistanceInvariance();
    //返回最大距离
    float GetMaxDistanceInvariance();
/*******************************************************************************
 *     函数属性：类MapPoint的成员函数PredictScale(const float &currentDist, KeyFrame*pKF);
 *     函数功能：
 *                 //                       ____
 *                 // Neare          /____\     level:n-1 --> dmin
 *                 //                    /______\                       d/dmin = 1.2^(n-1-m)
 *                 //                  /________\   level:m   --> d
 *                 //                /__________\                     dmax/d = 1.2^m
 *		   // Farther /____________\ level:0   --> dmax
 *		    //
 *		    //                 log(dmax/d)
 *		    // m = ceil(-------------------)
 *		    //                    log(1.2)
 *     函数参数介绍：
 *                currentDist：在该关键帧中的该地图点距离相机中心的距离
 *                pKF：关键帧
 *     备注： 预测该地图点位于该关键帧的层数
 * 
 ******************************************************************************/
    int PredictScale(const float &currentDist, KeyFrame*pKF);
    //同PredictScale(const float &currentDist, KeyFrame*pKF);
    int PredictScale(const float &currentDist, Frame* pF);

public:
    long unsigned int mnId;      //当前地图点的Id
    static long unsigned int nNextId;   //下一地图点的Id
    long int mnFirstKFid;           //第一关键帧Id（创建该地图点的关键帧的id）
    long int mnFirstFrame;       //第一帧的id（创建该地图点的帧的id）
    int nObs;    //observations中关键帧的数量 ，即可观察到该地图点的关键帧的数量

    // Variables used by the tracking    用来tracking的变量
    float mTrackProjX;    //该地图点在该帧相机的投影像素x坐标
    float mTrackProjY;    //该地图点在该帧相机的投影像素y坐标
    float mTrackProjXR;  //该3D点投影到双目右侧相机上的横坐标
    // mbTrackInView==false的点有几种：
    // a 已经和当前帧经过匹配（TrackReferenceKeyFrame，TrackWithMotionModel）但在优化过程中认为是外点
    // b 已经和当前帧经过匹配且为内点，这类点也不需要再进行投影
    // c 不在当前相机视野中的点（即未通过isInFrustum判断）
    bool mbTrackInView;//表示该地图点是否可以被观察到
    int mnTrackScaleLevel; //该地图点在该帧被观察到时在高斯金字塔中的层数
    float mTrackViewCos;   //该地图点在该帧中被观察到时的角度
    long unsigned int mnTrackReferenceForFrame;   
    // TrackLocalMap - SearchLocalPoints中决定是否进行isInFrustum判断的变量
    // mnLastFrameSeen==mCurrentFrame.mnId的点有几种：
    // a 已经和当前帧经过匹配（TrackReferenceKeyFrame，TrackWithMotionModel）但在优化过程中认为是外点
    // b 已经和当前帧经过匹配且为内点，这类点也不需要再进行投影
    long unsigned int mnLastFrameSeen;  // 上一被观察到该地图点的帧的id

    // Variables used by local mapping   用来局部地图构建的变量
    long unsigned int mnBALocalForKF;
    long unsigned int mnFuseCandidateForKF;

    // Variables used by loop closing   用来回环检测的变量
    long unsigned int mnLoopPointForKF;
    long unsigned int mnCorrectedByKF;
    long unsigned int mnCorrectedReference;    
    cv::Mat mPosGBA;
    long unsigned int mnBAGlobalForKF;


    static std::mutex mGlobalMutex;   //类的静态全局锁  对于该类的所有对象都有用

protected:    

     // Position in absolute coordinates 该地图点的世界坐标
     cv::Mat mWorldPos;

     // Keyframes observing the point and associated index in keyframe 可以观察到该地图点的关键帧以及该地图点在关键帧的索引  
     // 第一个参数是指能看到该地图点的关键帧，第二个参数是指该地图点在关键帧中的位置
     std::map<KeyFrame*,size_t> mObservations;

     // Mean viewing direction   观察该关键点时的平均观测方向
     cv::Mat mNormalVector;

     // Best descriptor to fast matching  该地图点在所有可观察到的关键帧中的最优描述子
     cv::Mat mDescriptor;

     // Reference KeyFrame  参考关键帧
     KeyFrame* mpRefKF;

     // Tracking counters 
     // 跟踪计数器 可以看到该地图点的次数
     int mnVisible;
     int mnFound;

     // Bad flag (we do not currently erase MapPoint from memory)从内存中擦出该地图点失败
     bool mbBad;
     //可代替该地图点的其他地图点
     MapPoint* mpReplaced;  

     // Scale invariance distances 尺度不变距离
     float mfMinDistance;
     float mfMaxDistance;

     Map* mpMap;
     
     std::mutex mMutexPos;   //位置锁
     std::mutex mMutexFeatures;  //保护mObservations相关的锁
};

} //namespace ORB_SLAM

#endif // MAPPOINT_H
