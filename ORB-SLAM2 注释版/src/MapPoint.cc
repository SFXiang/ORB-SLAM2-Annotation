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

#include "MapPoint.h"
#include "ORBmatcher.h"

#include<mutex>

namespace ORB_SLAM2
{

long unsigned int MapPoint::nNextId=0;
mutex MapPoint::mGlobalMutex;
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
MapPoint::MapPoint(const cv::Mat &Pos, KeyFrame *pRefKF, Map* pMap):
    mnFirstKFid(pRefKF->mnId), mnFirstFrame(pRefKF->mnFrameId), nObs(0), mnTrackReferenceForFrame(0),
    mnLastFrameSeen(0), mnBALocalForKF(0), mnFuseCandidateForKF(0), mnLoopPointForKF(0), mnCorrectedByKF(0),
    mnCorrectedReference(0), mnBAGlobalForKF(0), mpRefKF(pRefKF), mnVisible(1), mnFound(1), mbBad(false),
    mpReplaced(static_cast<MapPoint*>(NULL)), mfMinDistance(0), mfMaxDistance(0), mpMap(pMap)
{
    Pos.copyTo(mWorldPos);
    mNormalVector = cv::Mat::zeros(3,1,CV_32F);

    // MapPoints can be created from Tracking and Local Mapping. This mutex avoid conflicts with id.
    unique_lock<mutex> lock(mpMap->mMutexPointCreation);   //地图点创建锁，防止冲突
    mnId=nNextId++;
}
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
MapPoint::MapPoint(const cv::Mat &Pos, Map* pMap, Frame* pFrame, const int &idxF):
    mnFirstKFid(-1), mnFirstFrame(pFrame->mnId), nObs(0), mnTrackReferenceForFrame(0), mnLastFrameSeen(0),
    mnBALocalForKF(0), mnFuseCandidateForKF(0),mnLoopPointForKF(0), mnCorrectedByKF(0),
    mnCorrectedReference(0), mnBAGlobalForKF(0), mpRefKF(static_cast<KeyFrame*>(NULL)), mnVisible(1),
    mnFound(1), mbBad(false), mpReplaced(NULL), mpMap(pMap)
{
    Pos.copyTo(mWorldPos);
    cv::Mat Ow = pFrame->GetCameraCenter();
    mNormalVector = mWorldPos - Ow;
    mNormalVector = mNormalVector/cv::norm(mNormalVector);  //对方向归一化

    cv::Mat PC = Pos - Ow;
    const float dist = cv::norm(PC);  //地图点距离相机中心的距离
    const int level = pFrame->mvKeysUn[idxF].octave;   //得到该关键点（地图点）所在的金字塔的层数
    const float levelScaleFactor =  pFrame->mvScaleFactors[level];   //得到该关键点（地图点）所在的金字塔层的缩放比例
    const int nLevels = pFrame->mnScaleLevels;            //得到该金字塔的总层数
    //最大距离和最小距离   具体详情请见PredictScale函数前的注释
    mfMaxDistance = dist*levelScaleFactor;
    mfMinDistance = mfMaxDistance/pFrame->mvScaleFactors[nLevels-1];

    pFrame->mDescriptors.row(idxF).copyTo(mDescriptor);   //得到该地图点在该帧下的描述子

    // MapPoints can be created from Tracking and Local Mapping. This mutex avoid conflicts with id.
    unique_lock<mutex> lock(mpMap->mMutexPointCreation);
    mnId=nNextId++;
}
//设置该地图点的世界坐标
void MapPoint::SetWorldPos(const cv::Mat &Pos)
{
    unique_lock<mutex> lock2(mGlobalMutex);  //全局锁，使用与该类的所有对象
    unique_lock<mutex> lock(mMutexPos);         //位置锁
    Pos.copyTo(mWorldPos);
}
//得到该地图点的世界坐标
cv::Mat MapPoint::GetWorldPos()
{
    unique_lock<mutex> lock(mMutexPos);
    return mWorldPos.clone();
}
//得到该地图点在该所有关键帧下的平均观测方向
cv::Mat MapPoint::GetNormal()
{
    unique_lock<mutex> lock(mMutexPos);
    return mNormalVector.clone();
}
//得到参考关键帧
KeyFrame* MapPoint::GetReferenceKeyFrame()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mpRefKF;
}
//为该地图点的Observation添加关键帧   idx是指该地图点在关键帧的索引 Observation存储的是可以看到该地图点的所有关键帧的集合
// 在添加地图点到全局地图map时调用
void MapPoint::AddObservation(KeyFrame* pKF, size_t idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    if(mObservations.count(pKF))  //如果在观察器中已经存在该关键帧，则不需要继续添加
        return;
    mObservations[pKF]=idx;//否则将该关键帧和该地图点在该关键帧的索引加入该map容器mObservations

    if(pKF->mvuRight[idx]>=0)
        nObs+=2;   //双目或RGBD
    else
        nObs++;   //单目
}
//为该地图点的Observation删除关键帧  Observation存储的是可以看到该地图点的所有关键帧的集合
void MapPoint::EraseObservation(KeyFrame* pKF)
{
    bool bBad=false;    //擦除失败标志量
    {
        unique_lock<mutex> lock(mMutexFeatures);
        if(mObservations.count(pKF))   //如果可以找到需要擦除的关键帧
        {
            int idx = mObservations[pKF];  //得到该关键帧下地图点索引
            if(pKF->mvuRight[idx]>=0)   //双目或RGBD
                nObs-=2;
            else   //单目
                nObs--;

            mObservations.erase(pKF);   //从mObservations中删除该关键帧

            if(mpRefKF==pKF)  //如果参考关键帧是该关键帧，那么将参考关键帧设为mObservations的第一帧
                mpRefKF=mObservations.begin()->first;

            // If only 2 observations or less, discard point    如果仅有两个或者更少的关键帧那么取消该节点，并标志为失败
            if(nObs<=2)
                bBad=true;
        }
    }

    if(bBad)  //如果擦除失败
        SetBadFlag();
}
//得到Observations  Observation存储的是可以看到该地图点的所有关键帧的集合
map<KeyFrame*, size_t> MapPoint::GetObservations()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mObservations;
}
// 观察到该地图点的关键帧数量
int MapPoint::Observations()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return nObs;
}
//  如果mObservations擦除关键帧失败
void MapPoint::SetBadFlag()
{
    map<KeyFrame*,size_t> obs;
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        mbBad=true;    //从内存中擦出关键帧失败
        obs = mObservations;//将此时可观察到该地图点的关键帧拷贝给obs
        mObservations.clear();
    }
    //遍历obs  从关键帧中擦除地图点
    for(map<KeyFrame*,size_t>::iterator mit=obs.begin(), mend=obs.end(); mit!=mend; mit++)
    {
        KeyFrame* pKF = mit->first;
        pKF->EraseMapPointMatch(mit->second);
    }

    mpMap->EraseMapPoint(this);   //从全局地图中擦除该地图点
}
//得到该地图点的替代地图点
MapPoint* MapPoint::GetReplaced()
{
    unique_lock<mutex> lock1(mMutexFeatures);
    unique_lock<mutex> lock2(mMutexPos);
    return mpReplaced;
}
//  用pMP地图点来替换本地图点，并消除本地图点
/*
 * 	将本地图点的被观察次数,被查找次数,以及观察到该地图点的关键帧都清空,坏地图点标志置位
 * 	将本地图点的被观察次数,被查找次数都加到替换地图点pMP中,并将当前地图点在关键帧中的位置用代替地图点代替
 * 	最后将本地图点从全局地图map中删除
 */
void MapPoint::Replace(MapPoint* pMP)
{
    if(pMP->mnId==this->mnId)
        return;

    int nvisible, nfound;
    map<KeyFrame*,size_t> obs;
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        obs=mObservations;
        mObservations.clear();
        mbBad=true;
        nvisible = mnVisible;
        nfound = mnFound;
        mpReplaced = pMP;
    }
    // 遍历所有可以看见该地图点的关键帧
    for(map<KeyFrame*,size_t>::iterator mit=obs.begin(), mend=obs.end(); mit!=mend; mit++)
    {
        // Replace measurement in keyframe
        KeyFrame* pKF = mit->first;

        if(!pMP->IsInKeyFrame(pKF)) //如果地图点pMP没在该帧中
        {
            pKF->ReplaceMapPointMatch(mit->second, pMP);   //则用pMP替换掉本地图点在该帧的位置
            pMP->AddObservation(pKF,mit->second);  //并在地图点pMP中增加可观察到该地图点的帧数
        }
        else
        {
            pKF->EraseMapPointMatch(mit->second);   //如果地图点pMP在该帧中，则从关键中擦除现在的地图点
        }
    }
    pMP->IncreaseFound(nfound);  //将原地图点的查找次数加入到代替地图点pMP中
    pMP->IncreaseVisible(nvisible); //将原地图点的看到次数加入到代替地图点pMP中
    pMP->ComputeDistinctiveDescriptors();

    mpMap->EraseMapPoint(this);
}
//查看该地图点是否是有问题的
bool MapPoint::isBad()
{
    unique_lock<mutex> lock(mMutexFeatures);
    unique_lock<mutex> lock2(mMutexPos);
    return mbBad;
}
// 增加该地图点被看到次数n
void MapPoint::IncreaseVisible(int n)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mnVisible+=n;
}
// 增加该地图点被查找次数n
void MapPoint::IncreaseFound(int n)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mnFound+=n;
}
// 返回该地图点的查找率
float MapPoint::GetFoundRatio()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return static_cast<float>(mnFound)/mnVisible;
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
void MapPoint::ComputeDistinctiveDescriptors()
{
    // Retrieve all observed descriptors   检查该地图点在各个关键帧中的描述子
    vector<cv::Mat> vDescriptors;

    map<KeyFrame*,size_t> observations;

    {
        unique_lock<mutex> lock1(mMutexFeatures);
        if(mbBad)
            return;
        observations=mObservations;
    }

    if(observations.empty())
        return;

    vDescriptors.reserve(observations.size());

    for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
    {
        KeyFrame* pKF = mit->first;

        if(!pKF->isBad())
            vDescriptors.push_back(pKF->mDescriptors.row(mit->second));//如果该关键帧没有问题，那么将该关键中该地图点的描述子存入vDescriptors容器中
    }

    if(vDescriptors.empty())
        return;

    // Compute distances between them   计算所有被找到描述子之间的距离，并将其距离存入到Distances数组中
    const size_t N = vDescriptors.size();

    float Distances[N][N];
    for(size_t i=0;i<N;i++)
    {
        Distances[i][i]=0;
        for(size_t j=i+1;j<N;j++)
        {
            int distij = ORBmatcher::DescriptorDistance(vDescriptors[i],vDescriptors[j]);
            Distances[i][j]=distij;
            Distances[j][i]=distij;
        }
    }

    // Take the descriptor with least median distance to the rest   取出距离中值所对应的描述子
    int BestMedian = INT_MAX;
    int BestIdx = 0;
    for(size_t i=0;i<N;i++)
    {
        vector<int> vDists(Distances[i],Distances[i]+N);
        sort(vDists.begin(),vDists.end());
        int median = vDists[0.5*(N-1)];  //取第i个描述子与其他描述子距离的中值作为其均值参考

        if(median<BestMedian)    //取这些中值中最小的，认为该描述子与其他描述子的距离和最近。
        {
            BestMedian = median;
            BestIdx = i;
        }
    }

    {
        unique_lock<mutex> lock(mMutexFeatures);
        mDescriptor = vDescriptors[BestIdx].clone();   //给最优描述子赋值
    }
}
//得到该地图点的最优描述子，所谓最优就是在观察到该地图点的所有关键帧中描述子距离的中值描述子 详情见函数ComputeDistinctiveDescriptors()
cv::Mat MapPoint::GetDescriptor()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mDescriptor.clone();
}
//在关键帧pKF中该地图点的索引
int MapPoint::GetIndexInKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexFeatures);
    if(mObservations.count(pKF))
        return mObservations[pKF];
    else
        return -1;
}
// 判断地图点是否pKF关键帧中
bool MapPoint::IsInKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexFeatures);
    return (mObservations.count(pKF));
}
//更新地图点被观察的平均方向和观测距离范围
void MapPoint::UpdateNormalAndDepth()
{
    map<KeyFrame*,size_t> observations;
    KeyFrame* pRefKF;
    cv::Mat Pos;
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        if(mbBad)
            return;
        observations=mObservations;      //获得观测到该3d点的所有关键帧
        pRefKF=mpRefKF;                           //观测到该点的参考关键帧
        Pos = mWorldPos.clone();              //3d点在世界坐标系中的位置
    }

    if(observations.empty())
        return;

    cv::Mat normal = cv::Mat::zeros(3,1,CV_32F);
    int n=0;
    for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
    {
        KeyFrame* pKF = mit->first;
        cv::Mat Owi = pKF->GetCameraCenter();
        cv::Mat normali = mWorldPos - Owi;
        normal = normal + normali/cv::norm(normali);  //对所有关键帧对该点的观测方向做归一化并求和
        n++;
    }

    cv::Mat PC = Pos - pRefKF->GetCameraCenter();    //从参考关键帧观察该地图点的向量
    const float dist = cv::norm(PC);                               //地图点到参考关键帧相机中心的距离
    const int level = pRefKF->mvKeysUn[observations[pRefKF]].octave;
    const float levelScaleFactor =  pRefKF->mvScaleFactors[level];
    const int nLevels = pRefKF->mnScaleLevels;                            //金字塔层数

    {
        unique_lock<mutex> lock3(mMutexPos);
	//详情请见PredictScale函数前的注释
        mfMaxDistance = dist*levelScaleFactor;    //计算最大尺度
        mfMinDistance = mfMaxDistance/pRefKF->mvScaleFactors[nLevels-1];    //计算最小尺度
        mNormalVector = normal/n;   //计算平均观测方向
    }
}
//返回最小距离
float MapPoint::GetMinDistanceInvariance()
{
    unique_lock<mutex> lock(mMutexPos);
    return 0.8f*mfMinDistance;
}
//返回最大距离
float MapPoint::GetMaxDistanceInvariance()
{
    unique_lock<mutex> lock(mMutexPos);
    return 1.2f*mfMaxDistance;
}
//                       ____
// Neare          /____\     level:n-1 --> dmin
//                    /______\                       d/dmin = 1.2^(n-1-m)
//                  /________\   level:m   --> d
//                /__________\                     dmax/d = 1.2^m
// Farther /____________\ level:0   --> dmax
//
//                 log(dmax/d)
// m = ceil(-------------------)
//                    log(1.2)
int MapPoint::PredictScale(const float &currentDist, KeyFrame* pKF)
{
    float ratio;
    {
        unique_lock<mutex> lock(mMutexPos);
        ratio = mfMaxDistance/currentDist;    //计算当前的缩放比例
    }

    int nScale = ceil(log(ratio)/pKF->mfLogScaleFactor);   //得到预测的当前金字塔层数
    if(nScale<0)
        nScale = 0;
    else if(nScale>=pKF->mnScaleLevels)
        nScale = pKF->mnScaleLevels-1;

    return nScale;
}

int MapPoint::PredictScale(const float &currentDist, Frame* pF)
{
    float ratio;
    {
        unique_lock<mutex> lock(mMutexPos);
        ratio = mfMaxDistance/currentDist;
    }

    int nScale = ceil(log(ratio)/pF->mfLogScaleFactor);
    if(nScale<0)
        nScale = 0;
    else if(nScale>=pF->mnScaleLevels)
        nScale = pF->mnScaleLevels-1;

    return nScale;
}



} //namespace ORB_SLAM
