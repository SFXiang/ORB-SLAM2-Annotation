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

#ifndef LOCALMAPPING_H
#define LOCALMAPPING_H

#include "KeyFrame.h"
#include "Map.h"
#include "LoopClosing.h"
#include "Tracking.h"
#include "KeyFrameDatabase.h"

#include <mutex>


namespace ORB_SLAM2
{

class Tracking;
class LoopClosing;
class Map;

class LocalMapping
{
public:
    //构造函数
    LocalMapping(Map* pMap, const float bMonocular);
    
    void SetLoopCloser(LoopClosing* pLoopCloser);

    void SetTracker(Tracking* pTracker);

    // Main function
    void Run();

    void InsertKeyFrame(KeyFrame* pKF);

    // Thread Synch
    void RequestStop();
    void RequestReset();
    bool Stop();
    void Release();
    //检测是否局部建图进程被暂停,返回标志量mbStopped
    bool isStopped();
    bool stopRequested();
    bool AcceptKeyFrames();
    //设置此时不能接受新的关键帧  tracking线程不再接受新的关键帧
    void SetAcceptKeyFrames(bool flag);
    bool SetNotStop(bool flag);

    void InterruptBA();

    void RequestFinish();
    bool isFinished();

    int KeyframesInQueue(){
        unique_lock<std::mutex> lock(mMutexNewKFs);
        return mlNewKeyFrames.size();
    }

protected:
//检测是否存在新关键帧   返回mlNewKeyFrames队列是否为空
    bool CheckNewKeyFrames();
    //如果有新的关键帧,对于新关键帧的操作
    /***********************************************
    1 从新关键帧队列mlNewKeyFrames中取一新关键帧并将其从新关键帧列表中删除(避免重复操作新关键帧)
    2 计算新关键帧的BOW向量和Feature向量
    3 将该关键帧对应的地图点与该关键帧关联,并更新该地图点的平均观测方向和描述子
    4 更新Covisibility图
    5 将关键帧插入全局地图中

    ***********************************************/
    void ProcessNewKeyFrame();
//建立新的地图点
/*
 * 步骤:  1. 在当前关键帧的共视关键帧中找到共视程度最高的nn帧相邻帧vpNeighKFs
 * 		2. 遍历相邻关键帧vpNeighKFs,将当前关键帧和共视关键帧进行三角测量和对极约束
 * 		3. 对每次匹配得到的地图点进行筛选,看是否满足三角测量和对极约束  并根据对极约束计算三维点坐标
 * 			判断该匹配点是否是好的地图点  1> 两帧中的深度都是正的  2> 地图点在两帧的重投影误差 3> 检测地图点的尺度连续性
 * 		4. 如果三角化是成功的,那么建立新的地图点,并设置地图点的相关属性(a.观测到该MapPoint的关键帧  b.该MapPoint的描述子  c.该MapPoint的平均观测方向和深度范围),
 * 			然后将地图点加入当前关键帧和全局地图中
 * 备注: 注意对于单目相机的地图点三维坐标的确立需要根据三角测量来确定.
*/
    void CreateNewMapPoints();
// 检测新添加的局部地图点是否满足条件   筛选出好的地图点
// 筛选条件:地图点是否是好的,地图点的查找率小于0.25, 该地图点第一关键帧(第一次观察到改地图点的帧id)与当前帧id相隔距离, 该关键点的被观察到的次数
    void MapPointCulling();
/**
 * 检查并融合当前关键帧与相邻帧（两级相邻）重复的MapPoints
 * 1. 找到带融合的关键帧(当前关键帧的两级相邻关键帧)
 * 2. 在目标关键帧中寻找当前关键帧地图点的匹配  并在目标关键帧的地图点中中寻找当前关键帧所有地图点的融合点
 * 3. 在当前关键帧中寻找目标关键帧(当前关键帧的两级相邻)所有地图点的匹配  并在当前关键帧的地图点中中寻找目标关键帧所有地图点的融合点
 * 4. 更新特征点融合之后当前关键帧的地图点的最优描述子和该地图点被观察的平均方向以及深度范围
 * 5. 更新当前关键帧地图点融合之后的当前关键帧与关联关键帧的联系
 */
    void SearchInNeighbors();

    // 剔除冗余关键帧,检测的是当前关键帧(当前组关键帧没有新的关键帧)的共视关键帧
    // 判断方法:如果一个关键帧的地图点有90%被其他至少三个关键帧(相邻和相同尺度)看到  那么我们认为该关键帧是冗余的
    void KeyFrameCulling();
    //计算基础矩阵F=K.(-T)*t^R*K.(-1)
    cv::Mat ComputeF12(KeyFrame* &pKF1, KeyFrame* &pKF2);
    
// 将向量v转化为相应的反对称矩阵
    cv::Mat SkewSymmetricMatrix(const cv::Mat &v);
    //是否是单目相机
    bool mbMonocular;

    void ResetIfRequested();
    bool mbResetRequested;
    std::mutex mMutexReset;
    //返回变量mbFinishRequested   检测是否slam进程已经结束
    bool CheckFinish();
    void SetFinish();
    bool mbFinishRequested;
    bool mbFinished;
    std::mutex mMutexFinish;

    Map* mpMap;
    //与之相关联的回环检测线程
    LoopClosing* mpLoopCloser;
    // 与之相关联的追踪线程
    Tracking* mpTracker;
    //新关键帧列表
    std::list<KeyFrame*> mlNewKeyFrames;
    //  当前关键帧
    KeyFrame* mpCurrentKeyFrame;
    //最近添加的地图点
    std::list<MapPoint*> mlpRecentAddedMapPoints;
    //与新关键帧相关的锁
    std::mutex mMutexNewKFs;

    bool mbAbortBA;
    //局部建图被暂停变量
    bool mbStopped;
    //暂停请求标识
    bool mbStopRequested;
    //
    bool mbNotStop;
    //暂停请求锁
    std::mutex mMutexStop;

    //
    bool mbAcceptKeyFrames;
    //
    std::mutex mMutexAccept;
};

} //namespace ORB_SLAM

#endif // LOCALMAPPING_H
