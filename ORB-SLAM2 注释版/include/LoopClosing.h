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

#ifndef LOOPCLOSING_H
#define LOOPCLOSING_H

#include "KeyFrame.h"
#include "LocalMapping.h"
#include "Map.h"
#include "ORBVocabulary.h"
#include "Tracking.h"

#include "KeyFrameDatabase.h"

#include <thread>
#include <mutex>
#include "Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"

namespace ORB_SLAM2
{

class Tracking;
class LocalMapping;
class KeyFrameDatabase;


class LoopClosing
{
public:
    // set<KeyFrame*>指的是关键帧集构成的连续组    int指的是该连续组与其他连续组之间连续的数量
    typedef pair<set<KeyFrame*>,int> ConsistentGroup;    
    typedef map<KeyFrame*,g2o::Sim3,std::less<KeyFrame*>,
        Eigen::aligned_allocator<std::pair<const KeyFrame*, g2o::Sim3> > > KeyFrameAndPose;

public:

    LoopClosing(Map* pMap, KeyFrameDatabase* pDB, ORBVocabulary* pVoc,const bool bFixScale);

    void SetTracker(Tracking* pTracker);

    void SetLocalMapper(LocalMapping* pLocalMapper);

    // Main function
    void Run();

    void InsertKeyFrame(KeyFrame *pKF);

    void RequestReset();

    // This function will run in a separate thread
    void RunGlobalBundleAdjustment(unsigned long nLoopKF);

    bool isRunningGBA(){
        unique_lock<std::mutex> lock(mMutexGBA);
        return mbRunningGBA;
    }
    bool isFinishedGBA(){
        unique_lock<std::mutex> lock(mMutexGBA);
        return mbFinishedGBA;
    }   

    void RequestFinish();

    bool isFinished();

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

protected:
     //  检测回环关键帧
    bool CheckNewKeyFrames();
/*******************************************************************************
 *                 功能:检测是否产生了回环
 *                 检测回环的步骤:
 *                             1  检测上次回环发生是否离当前关键帧足够长时间    并且满足当前关键帧总数量大于10
 *                             2  找出当前关键帧的共视关键帧,并找出其中的最小得分
 *                             3  根据最小得分寻找回环候选帧   具体见含函数DetectLoopCandidates
 * 				4  在候选回环关键帧中寻找具有连续性的关键帧
 * 						这里将候选回环关键帧和他的共视关键帧组成一个候选组
 *                                             一个组和另一个组是连续的,是指他们至少存在一个共视关键帧
 *                                       如果两个组之间存在足够多的帧是共视关键帧,则证明两个组之间是完全连续组,则说明发生了回环
 * 
 ******************************************************************************/
    bool DetectLoop();
/********************************
 * 	计算每一个回环候选关键帧与当前关键帧之间的相似矩阵
 * 		1. 首先通过BOW向量将回环候选关键帧和当前关键帧进行匹配,得到匹配地图点,通过匹配地图点初始化相似矩阵求解器
 * 		2. 遍历所有的回环候选关键帧和当前关键帧计算sim矩阵,并优化sim矩阵,根据优化sim矩阵确定匹配内点数量,从而确定此sim矩阵的准确性,以及是否可以判定为一回环.
 * 		3. 在找到的回环关键帧周围查找共视关键帧   并匹配共视关键帧的地图点和当前关键帧  相当与是匹配covisual graph和当前关键帧  根据匹配点数量确定当前帧是否发生了回环
 **************************************/
    bool ComputeSim3();
// 根据矫正后的相机相似矩阵位姿匹配回环点和当前关键帧,并融合得到的关键帧中匹配点和回环地图点
    void SearchAndFuse(const KeyFrameAndPose &CorrectedPosesMap);
/******************************************
 * 	根据回环进行位姿矫正
 * 		1. 请求局部地图线程停止,并且中止现有的全局优化进程
 *		2. 根据当前帧求得的相机位姿(相似变换矩阵)来求解矫正前和矫正后的相邻帧位姿变换矩阵(相似变换矩阵)
 * 		3. 将相邻关键帧的所有地图点都根据更新后的相机位姿(相似变换矩阵)重新计算地图点世界坐标  
 * 		4. 进行地图点融合   将之前匹配的(在ComputeSim3()函数中计算局部地图点和当前帧的匹配)两地图点融合为同一地图点
 * 		5. 根据第3步中计算的地图点重新进行匹配,并融合匹配点和当前关键帧中的地图点
 * 		6. 在地图点融合之后,更新当前关键帧的共视图中各个关键帧的相连关键帧,更新连接之后,将这些相邻关键帧全部加入LoopConnections容器
 * 		7. 根据四种边(1 新检测到的回环边  2 父关键帧与子关键帧的边 3 历史回环关键帧  4 共视图边)对全局地图中的所有关键帧的位姿进行矫正
 * 		8. 根据地图点和关键帧位姿计算重投影误差对全局地图进行优化
 *************************************/
    void CorrectLoop();
    
    // 检测是否需要复位操作,如果需要则将回环关键帧序列清空  上次回环关键帧id清零  复位请求清零
    void ResetIfRequested();
    bool mbResetRequested;
    std::mutex mMutexReset;

    bool CheckFinish();
    void SetFinish();
    bool mbFinishRequested;
    bool mbFinished;
    std::mutex mMutexFinish;

    Map* mpMap;
    Tracking* mpTracker;

    KeyFrameDatabase* mpKeyFrameDB;
    ORBVocabulary* mpORBVocabulary;

    LocalMapping *mpLocalMapper;
    // 回环检测中的关键帧序列
    std::list<KeyFrame*> mlpLoopKeyFrameQueue;

    std::mutex mMutexLoopQueue;

    // Loop detector parameters
    float mnCovisibilityConsistencyTh;

    // Loop detector variables
    KeyFrame* mpCurrentKF;
    KeyFrame* mpMatchedKF;
    //之前所有的连续组集
    std::vector<ConsistentGroup> mvConsistentGroups;
    // 足够连续组集
    std::vector<KeyFrame*> mvpEnoughConsistentCandidates;
    std::vector<KeyFrame*> mvpCurrentConnectedKFs;
    std::vector<MapPoint*> mvpCurrentMatchedPoints;
    std::vector<MapPoint*> mvpLoopMapPoints;
    cv::Mat mScw;
    g2o::Sim3 mg2oScw;
    
    //发生回环关键帧的id
    long unsigned int mLastLoopKFid;

    // Variables related to Global Bundle Adjustment
    bool mbRunningGBA;
    bool mbFinishedGBA;
    bool mbStopGBA;
    std::mutex mMutexGBA;
    std::thread* mpThreadGBA;

    // Fix scale in the stereo/RGB-D case
    bool mbFixScale;


    bool mnFullBAIdx;
};

} //namespace ORB_SLAM

#endif // LOOPCLOSING_H
