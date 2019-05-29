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

#include "KeyFrame.h"
#include "Converter.h"
#include "ORBmatcher.h"
#include<mutex>

namespace ORB_SLAM2
{

long unsigned int KeyFrame::nNextId=0;
/*******************************************************************************
 *     函数属性：类KeyFrame的构造函数
 *     函数功能：
 *                        构造关键帧
 *     函数参数介绍：
 *                         F：需要加入关键帧的帧
 *                         pMap：地图
 *                         pKFDB：关键帧数据集
 *     备注：NULL
 * 
 ******************************************************************************/
KeyFrame::KeyFrame(Frame &F, Map *pMap, KeyFrameDatabase *pKFDB):
    mnFrameId(F.mnId),  mTimeStamp(F.mTimeStamp), mnGridCols(FRAME_GRID_COLS), mnGridRows(FRAME_GRID_ROWS),
    mfGridElementWidthInv(F.mfGridElementWidthInv), mfGridElementHeightInv(F.mfGridElementHeightInv),
    mnTrackReferenceForFrame(0), mnFuseTargetForKF(0), mnBALocalForKF(0), mnBAFixedForKF(0),
    mnLoopQuery(0), mnLoopWords(0), mnRelocQuery(0), mnRelocWords(0), mnBAGlobalForKF(0),
    fx(F.fx), fy(F.fy), cx(F.cx), cy(F.cy), invfx(F.invfx), invfy(F.invfy),
    mbf(F.mbf), mb(F.mb), mThDepth(F.mThDepth), N(F.N), mvKeys(F.mvKeys), mvKeysUn(F.mvKeysUn),
    mvuRight(F.mvuRight), mvDepth(F.mvDepth), mDescriptors(F.mDescriptors.clone()),
    mBowVec(F.mBowVec), mFeatVec(F.mFeatVec), mnScaleLevels(F.mnScaleLevels), mfScaleFactor(F.mfScaleFactor),
    mfLogScaleFactor(F.mfLogScaleFactor), mvScaleFactors(F.mvScaleFactors), mvLevelSigma2(F.mvLevelSigma2),
    mvInvLevelSigma2(F.mvInvLevelSigma2), mnMinX(F.mnMinX), mnMinY(F.mnMinY), mnMaxX(F.mnMaxX),
    mnMaxY(F.mnMaxY), mK(F.mK), mvpMapPoints(F.mvpMapPoints), mpKeyFrameDB(pKFDB),
    mpORBvocabulary(F.mpORBvocabulary), mbFirstConnection(true), mpParent(NULL), mbNotErase(false),
    mbToBeErased(false), mbBad(false), mHalfBaseline(F.mb/2), mpMap(pMap)
{
    mnId=nNextId++;

    mGrid.resize(mnGridCols);
    for(int i=0; i<mnGridCols;i++)
    {
        mGrid[i].resize(mnGridRows);
        for(int j=0; j<mnGridRows; j++)
            mGrid[i][j] = F.mGrid[i][j];
    }

    SetPose(F.mTcw);    
}
//计算BoW向量和Feature向量 在添加关键帧到地图点时调用
void KeyFrame::ComputeBoW()
{
    if(mBowVec.empty() || mFeatVec.empty())
    {
        vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
        // Feature vector associate features with nodes in the 4th level (from leaves up)
        // We assume the vocabulary tree has 6 levels, change the 4 otherwise
        mpORBvocabulary->transform(vCurrentDesc,mBowVec,mFeatVec,4);
    }
}
// 设置位姿，包括从世界坐标到相机坐标的变换矩阵Tcw 相机中心Ow 从相机坐标到世界坐标的变换矩阵Twc  ，双目相机的中心 Cw
void KeyFrame::SetPose(const cv::Mat &Tcw_)
{
    unique_lock<mutex> lock(mMutexPose);
    Tcw_.copyTo(Tcw);
    cv::Mat Rcw = Tcw.rowRange(0,3).colRange(0,3);
    cv::Mat tcw = Tcw.rowRange(0,3).col(3);
    cv::Mat Rwc = Rcw.t();
    Ow = -Rwc*tcw;

    Twc = cv::Mat::eye(4,4,Tcw.type());
    Rwc.copyTo(Twc.rowRange(0,3).colRange(0,3));
    Ow.copyTo(Twc.rowRange(0,3).col(3));
    cv::Mat center = (cv::Mat_<float>(4,1) << mHalfBaseline, 0 , 0, 1);
    Cw = Twc*center;
}
//得到从世界坐标到相机坐标的变换矩阵Tcw
cv::Mat KeyFrame::GetPose()
{
    unique_lock<mutex> lock(mMutexPose);
    return Tcw.clone();
}
//得到从相机坐标到世界坐标的变换矩阵Twc
cv::Mat KeyFrame::GetPoseInverse()
{
    unique_lock<mutex> lock(mMutexPose);
    return Twc.clone();
}
// 得到相机中心
cv::Mat KeyFrame::GetCameraCenter()
{
    unique_lock<mutex> lock(mMutexPose);
    return Ow.clone();
}
//得到双目相机中心
cv::Mat KeyFrame::GetStereoCenter()
{
    unique_lock<mutex> lock(mMutexPose);
    return Cw.clone();
}

// 得到旋转矩阵Rcw
cv::Mat KeyFrame::GetRotation()
{
    unique_lock<mutex> lock(mMutexPose);
    return Tcw.rowRange(0,3).colRange(0,3).clone();
}
//得到平移矩阵tcw
cv::Mat KeyFrame::GetTranslation()
{
    unique_lock<mutex> lock(mMutexPose);
    return Tcw.rowRange(0,3).col(3).clone();
}
// 添加与该关键帧相关联的关键帧及其权重  ，并存储到mConnectedKeyFrameWeights容器中
void KeyFrame::AddConnection(KeyFrame *pKF, const int &weight)
{
    {
        unique_lock<mutex> lock(mMutexConnections);
        if(!mConnectedKeyFrameWeights.count(pKF))
            mConnectedKeyFrameWeights[pKF]=weight;
        else if(mConnectedKeyFrameWeights[pKF]!=weight)
            mConnectedKeyFrameWeights[pKF]=weight;
        else
            return;
    }

    UpdateBestCovisibles();
}
// 将与该关键帧相关联的关键帧序列根据权重进行排序 ，将排序之后的关键帧和权重存储到mvpOrderedConnectedKeyFrames和mvOrderedWeights中
void KeyFrame::UpdateBestCovisibles()
{
    unique_lock<mutex> lock(mMutexConnections);
    vector<pair<int,KeyFrame*> > vPairs;
    vPairs.reserve(mConnectedKeyFrameWeights.size());
    for(map<KeyFrame*,int>::iterator mit=mConnectedKeyFrameWeights.begin(), mend=mConnectedKeyFrameWeights.end(); mit!=mend; mit++)
       vPairs.push_back(make_pair(mit->second,mit->first));

    sort(vPairs.begin(),vPairs.end());
    list<KeyFrame*> lKFs;
    list<int> lWs;
    for(size_t i=0, iend=vPairs.size(); i<iend;i++)
    {
        lKFs.push_front(vPairs[i].second);
        lWs.push_front(vPairs[i].first);
    }

    mvpOrderedConnectedKeyFrames = vector<KeyFrame*>(lKFs.begin(),lKFs.end());
    mvOrderedWeights = vector<int>(lWs.begin(), lWs.end());    
}
// 得到相关联的关键帧(关联关键帧是指权重大于15的共视关键帧,也就是有15个以上的共同地图点)
set<KeyFrame*> KeyFrame::GetConnectedKeyFrames()
{
    unique_lock<mutex> lock(mMutexConnections);
    set<KeyFrame*> s;
    for(map<KeyFrame*,int>::iterator mit=mConnectedKeyFrameWeights.begin();mit!=mConnectedKeyFrameWeights.end();mit++)
        s.insert(mit->first);
    return s;
}
// 返回根据权重排序好的关键帧序列
vector<KeyFrame*> KeyFrame::GetVectorCovisibleKeyFrames()
{
    unique_lock<mutex> lock(mMutexConnections);
    return mvpOrderedConnectedKeyFrames;
}
// 返回最好的（权重最大的）与该关键帧相关联的前N个关键帧序列
vector<KeyFrame*> KeyFrame::GetBestCovisibilityKeyFrames(const int &N)
{
    unique_lock<mutex> lock(mMutexConnections);
    if((int)mvpOrderedConnectedKeyFrames.size()<N)
        return mvpOrderedConnectedKeyFrames;
    else
        return vector<KeyFrame*>(mvpOrderedConnectedKeyFrames.begin(),mvpOrderedConnectedKeyFrames.begin()+N);

}
// 返回权重大于w的关键帧
vector<KeyFrame*> KeyFrame::GetCovisiblesByWeight(const int &w)
{
    unique_lock<mutex> lock(mMutexConnections);

    if(mvpOrderedConnectedKeyFrames.empty())
        return vector<KeyFrame*>();

    vector<int>::iterator it = upper_bound(mvOrderedWeights.begin(),mvOrderedWeights.end(),w,KeyFrame::weightComp);  //存储权重大于w的权重序列
    if(it==mvOrderedWeights.end())
        return vector<KeyFrame*>();
    else
    {
        int n = it-mvOrderedWeights.begin();
        return vector<KeyFrame*>(mvpOrderedConnectedKeyFrames.begin(), mvpOrderedConnectedKeyFrames.begin()+n);
    }
}
//得到帧pKF的权重
int KeyFrame::GetWeight(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexConnections);
    if(mConnectedKeyFrameWeights.count(pKF))
        return mConnectedKeyFrameWeights[pKF];
    else
        return 0;
}
// 添加地图点pMP及其索引idx  在加入关键帧时调用
void KeyFrame::AddMapPoint(MapPoint *pMP, const size_t &idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mvpMapPoints[idx]=pMP;
}
// 擦除索引为idx的地图点
void KeyFrame::EraseMapPointMatch(const size_t &idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mvpMapPoints[idx]=static_cast<MapPoint*>(NULL);
}
// 擦除地图点pMP及其在关键帧中的索引
void KeyFrame::EraseMapPointMatch(MapPoint* pMP)
{
    int idx = pMP->GetIndexInKeyFrame(this);
    if(idx>=0)
        mvpMapPoints[idx]=static_cast<MapPoint*>(NULL);
}

//替换该关键帧相关的地图点及其索引（有点儿bug）
void KeyFrame::ReplaceMapPointMatch(const size_t &idx, MapPoint* pMP)
{
    mvpMapPoints[idx]=pMP;
}
//得到与该关键帧相关联的地图点的集合
set<MapPoint*> KeyFrame::GetMapPoints()
{
    unique_lock<mutex> lock(mMutexFeatures);
    set<MapPoint*> s;
    for(size_t i=0, iend=mvpMapPoints.size(); i<iend; i++)
    {
        if(!mvpMapPoints[i])
            continue;
        MapPoint* pMP = mvpMapPoints[i];
        if(!pMP->isBad())
            s.insert(pMP);
    }
    return s;
}
// 该关键帧相关的地图点中被观察到的次数大于minObs的地图点个数
int KeyFrame::TrackedMapPoints(const int &minObs)
{
    unique_lock<mutex> lock(mMutexFeatures);

    int nPoints=0;
    const bool bCheckObs = minObs>0;
    for(int i=0; i<N; i++)
    {
        MapPoint* pMP = mvpMapPoints[i];
        if(pMP)
        {
            if(!pMP->isBad())
            {
                if(bCheckObs)
                {
                    if(mvpMapPoints[i]->Observations()>=minObs)   //地图点被观察次数大于minObs
                        nPoints++;
                }
                else
                    nPoints++;
            }
        }
    }

    return nPoints;
}
// 返回与该关键帧相关的地图点
vector<MapPoint*> KeyFrame::GetMapPointMatches()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mvpMapPoints;
}
//得到在该关键帧中索引为idx的地图点
MapPoint* KeyFrame::GetMapPoint(const size_t &idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mvpMapPoints[idx];
}
/**********************************************************************************************************************
 * 函数属性：KeyFrame类成员函数UpdateConnections()
 * 函数功能：
 * 1. 首先获得该关键帧的所有MapPoint点，统计观测到这些3d点的每个关键与其它所有关键帧之间的共视程度(共视程度是指两个帧之间存在共同的地图点数量)
 *    对每一个找到的关键帧(共视程度大于15)，建立一条边，边的权重是该关键帧与当前关键帧公共3d点的个数。
 * 2. 并且该权重必须大于一个阈值，如果没有超过该阈值的权重，那么就只保留权重最大的边（与其它关键帧的共视程度比较高）
 * 3. 对这些连接按照权重从大到小进行排序，以方便将来的处理
 *    更新完covisibility图之后，如果没有初始化过，则初始化为连接权重最大的边（与其它关键帧共视程度最高的那个关键帧），类似于最大生成树
 * 4. 更新关联关键帧及权重
 * 5. 更新父关键帧为关联关键帧权重最大帧
 * 函数参数：NULL
 * 备注：NULL
 **********************************************************************************************************************/
void KeyFrame::UpdateConnections()
{
    map<KeyFrame*,int> KFcounter;   //关键帧-权重，权重为其它关键帧与当前关键帧共视3d点的个数

    vector<MapPoint*> vpMP;//存储该关键帧的地图点

    {
        unique_lock<mutex> lockMPs(mMutexFeatures);
        vpMP = mvpMapPoints;
    }

    //For all map points in keyframe check in which other keyframes are they seen
    //Increase counter for those keyframes
    // 通过3D点间接统计可以观测到这些3D点的所有关键帧之间的共视程度
    // 即统计每一个关键帧都有多少关键帧与它存在共视关系，统计结果放在KFcounter
    for(vector<MapPoint*>::iterator vit=vpMP.begin(), vend=vpMP.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;

        if(!pMP)
            continue;

        if(pMP->isBad())
            continue;

        map<KeyFrame*,size_t> observations = pMP->GetObservations();//得到可以看到pMP地图点的所有关键帧以及地图点在这些关键帧中的索引

        for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            if(mit->first->mnId==mnId)  //如果查找到的关键帧就是本关键帧那么将跳过本次查找
                continue;
            KFcounter[mit->first]++;//否则被查找到的关键帧的权重+1
        }
    }

    // This should not happen
    if(KFcounter.empty())
        return;

    //If the counter is greater than threshold add connection
    //In case no keyframe counter is over threshold add the one with maximum counter
    int nmax=0;  //存储权重的最大值
    KeyFrame* pKFmax=NULL;//权重最大值所对应的关键帧
    int th = 15;

    vector<pair<int,KeyFrame*> > vPairs;//vPairs记录与其它关键帧共视帧数大于th的关键帧 权重-关键帧
    vPairs.reserve(KFcounter.size());
    for(map<KeyFrame*,int>::iterator mit=KFcounter.begin(), mend=KFcounter.end(); mit!=mend; mit++)
    {
        if(mit->second>nmax)
        {
            nmax=mit->second;
            pKFmax=mit->first;
        }
        if(mit->second>=th) //如果权重大于阈值
        {
            vPairs.push_back(make_pair(mit->second,mit->first));   //将权重大于th的关键帧存入vPairs中
            (mit->first)->AddConnection(this,mit->second);              //并增加本关键帧与该查找到的关键帧建立联系
        }
    }

    if(vPairs.empty())   //如果没有关键帧的权重大于阈值，则将权重最大的关键帧与本关键帧建立联系
    {
        vPairs.push_back(make_pair(nmax,pKFmax));
        pKFmax->AddConnection(this,nmax);
    }

    sort(vPairs.begin(),vPairs.end());
    list<KeyFrame*> lKFs;  //排序后的关联关键帧
    list<int> lWs;                 //排序后的权重
    for(size_t i=0; i<vPairs.size();i++)
    {
        lKFs.push_front(vPairs[i].second);
        lWs.push_front(vPairs[i].first);
    }

    {
        unique_lock<mutex> lockCon(mMutexConnections);

        // mspConnectedKeyFrames = spConnectedKeyFrames;
        mConnectedKeyFrameWeights = KFcounter;    //将关联关键帧及其权重存储
        mvpOrderedConnectedKeyFrames = vector<KeyFrame*>(lKFs.begin(),lKFs.end());   //存储排序后的关联关键帧
        mvOrderedWeights = vector<int>(lWs.begin(), lWs.end());    //存储排序后的权重

        if(mbFirstConnection && mnId!=0)
        {
            mpParent = mvpOrderedConnectedKeyFrames.front();   //将权重最大的关键帧存储到变量中
            mpParent->AddChild(this);   
            mbFirstConnection = false;   //证明不是第一次建立链接
        }

    }
}
//给pKF添加孩子  孩子证明本关键点是pKF的父节点，即权重最大的关联关键帧  在为当前关键帧添加父关键帧的同时为父关键帧添加子关键帧
void KeyFrame::AddChild(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mspChildrens.insert(pKF);
}
//给pKF删除孩子  孩子证明本关键点是pKF的父节点，即权重最大的关联关键帧
void KeyFrame::EraseChild(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mspChildrens.erase(pKF);
}
// 将父节点改变为pKF并给pKF添加子节点为本关键帧   父节点是值与本节点最大关联关键帧
void KeyFrame::ChangeParent(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mpParent = pKF;
    pKF->AddChild(this);
}
// 返回本关键帧的所有的孩子，也就是本关键帧为哪些关键帧的最大关联关键帧
set<KeyFrame*> KeyFrame::GetChilds()
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mspChildrens;
}
// 返回父关键帧 父关键帧为本关键帧的最大关联关键帧
KeyFrame* KeyFrame::GetParent()
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mpParent;
}
// 检查该关键帧是否有孩子，即该关键帧是否是其他关键帧的最大关联关键帧
bool KeyFrame::hasChild(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mspChildrens.count(pKF);
}
//添加回环边  pKF与本关键帧形成回环
void KeyFrame::AddLoopEdge(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mbNotErase = true;   //与其他关键帧形成回环后就不会被擦除
    mspLoopEdges.insert(pKF);
}
//返回该关键帧的回环关键帧
set<KeyFrame*> KeyFrame::GetLoopEdges()
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mspLoopEdges;
}
// 设置该关键帧不可被擦除
void KeyFrame::SetNotErase()
{
    unique_lock<mutex> lock(mMutexConnections);
    mbNotErase = true;
}
// 设置该关键帧可擦除
void KeyFrame::SetErase()
{
    {
        unique_lock<mutex> lock(mMutexConnections);
        if(mspLoopEdges.empty())  //如果没有其他关键帧与该关键帧形成回环，则将该关键帧设置为可擦除
        {
            mbNotErase = false;
        }
    }

    if(mbToBeErased)//如果将被擦除标志位为真则设置该关键帧为坏帧，并擦除该帧
    {
        SetBadFlag();
    }
}
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
void KeyFrame::SetBadFlag()
{   
    {
        unique_lock<mutex> lock(mMutexConnections);
        if(mnId==0)
            return;
        else if(mbNotErase)   //证明该帧不能被擦除
        {
            mbToBeErased = true;
            return;
        }
    }
    //擦除所有本关键帧与关联关键帧之间的关联
    for(map<KeyFrame*,int>::iterator mit = mConnectedKeyFrameWeights.begin(), mend=mConnectedKeyFrameWeights.end(); mit!=mend; mit++)
        mit->first->EraseConnection(this);
    //擦除所有地图点与本关键帧之间的关联，标志本关键帧已经不能看到这些地图点
    for(size_t i=0; i<mvpMapPoints.size(); i++)
        if(mvpMapPoints[i])
            mvpMapPoints[i]->EraseObservation(this);
    {
        unique_lock<mutex> lock(mMutexConnections);
        unique_lock<mutex> lock1(mMutexFeatures);

        mConnectedKeyFrameWeights.clear();   //清空存储与本关键帧关联的其他关键帧变量
        mvpOrderedConnectedKeyFrames.clear();  // 清空排序之后的关联关键帧序列

        // Update Spanning Tree
        set<KeyFrame*> sParentCandidates;
        sParentCandidates.insert(mpParent);   //将本关键帧的父关键帧插入sParentCandidates变量

        // Assign at each iteration one children with a parent (the pair with highest covisibility weight)
        // Include that children as new parent candidate for the rest
	// 清空子关键帧   有其他帧认为它是其父关键帧（同视话最高）
        while(!mspChildrens.empty())
        {
            bool bContinue = false;

            int max = -1;
            KeyFrame* pC;
            KeyFrame* pP;
            // 主要解决的问题是：如果将本关键帧消除的话，以本关键帧为父关键帧（共视化程度最高）的子关键帧中没有了父关键帧，需要重新给这些子关键帧找寻父关键帧
            for(set<KeyFrame*>::iterator sit=mspChildrens.begin(), send=mspChildrens.end(); sit!=send; sit++)  //遍历所有的子关键帧
            {
                KeyFrame* pKF = *sit;  //pKF存储的是子关键帧
                if(pKF->isBad())   //如果此子关键帧是坏的，则继续下一个子关键帧的检测
                    continue;

                // Check if a parent candidate is connected to the keyframe  
		// 检查与子关键帧相关关联的所有关键帧
		// 如果该帧的子节点和父节点（祖孙节点）之间存在连接关系（共视）
                    // 举例：B-->A（B的父节点是A） C-->B（C的父节点是B） D--C（D与C相连） E--C（E与C相连） F--C（F与C相连） D-->A（D的父节点是A） E-->A（E的父节点是A）
                    //      现在B挂了，于是C在与自己相连的D、E、F节点中找到父节点指向A的D
                    //      此过程就是为了找到可以替换B的那个节点。
                    // 上面例子中，B为当前要设置为SetBadFlag的关键帧
                    //           A为spcit，也即sParentCandidates
                    //           C为pKF,pC，也即mspChildrens中的一个
                    //           D、E、F为vpConnected中的变量，由于C与D间的权重 比 C与E间的权重大，因此D为pP
                vector<KeyFrame*> vpConnected = pKF->GetVectorCovisibleKeyFrames();
                for(size_t i=0, iend=vpConnected.size(); i<iend; i++)  //遍历每一个子关键帧相关联的其他关键帧
                {
                    for(set<KeyFrame*>::iterator spcit=sParentCandidates.begin(), spcend=sParentCandidates.end(); spcit!=spcend; spcit++)  //遍历本关键帧的父关键帧
                    {
                        if(vpConnected[i]->mnId == (*spcit)->mnId)  //如果该帧的子节点与父节点之间存在联系
                        {
                            int w = pKF->GetWeight(vpConnected[i]);   //找出与子关键帧关联关键帧中最大权重的关键帧，认为该关键帧为子关键帧的父关键帧
                            if(w>max)
                            {
                                pC = pKF;    //
                                pP = vpConnected[i];
                                max = w;
                                bContinue = true;
                            }
                        }
                    }
                }
            }

            if(bContinue)
            {
                pC->ChangeParent(pP);// 因为父节点死了，并且子节点找到了新的父节点，子节点更新自己的父节点
                sParentCandidates.insert(pC);// 因为子节点找到了新的父节点并更新了父节点，那么该子节点升级，作为其它子节点的备选父节点
                mspChildrens.erase(pC);// 该子节点处理完毕
            }
            else
                break;
        }

        // If a children has no covisibility links with any parent candidate, assign to the original parent of this KF
        // 如果子节点中没有找到新的父节点
        if(!mspChildrens.empty())
            for(set<KeyFrame*>::iterator sit=mspChildrens.begin(); sit!=mspChildrens.end(); sit++)
            {
                (*sit)->ChangeParent(mpParent);  //那么将本关键帧的父节点作为自己的父节点（父节点的父节点作为自己的父节点）
            }

        mpParent->EraseChild(this);  //在父节点中擦除该子节点（本关键帧）
        mTcp = Tcw*mpParent->GetPoseInverse();// 与父关键帧之间的变换矩阵
        mbBad = true;   //说明该关键帧为坏的，已经被剔除
    }


    mpMap->EraseKeyFrame(this);   //从地图中擦除该关键帧
    mpKeyFrameDB->erase(this);     //从关键帧数据集中剔除本关键帧
}
// 检测该关键帧是否是好的
bool KeyFrame::isBad()
{
    unique_lock<mutex> lock(mMutexConnections);
    return mbBad;
}
//擦除与该关键帧相关联的关键帧pKF
void KeyFrame::EraseConnection(KeyFrame* pKF)
{
    bool bUpdate = false;
    {
        unique_lock<mutex> lock(mMutexConnections);
        if(mConnectedKeyFrameWeights.count(pKF))   //查找与关键帧相关的序列关键帧中是否有pKF关键帧
        {
            mConnectedKeyFrameWeights.erase(pKF);
            bUpdate=true;
        }
    }

    if(bUpdate)   //如果擦除成功则更新关联关键帧序列
        UpdateBestCovisibles();
}
// 在以(x,y)为中心,2r为边长的正方形区域内得到特征点的序列
vector<size_t> KeyFrame::GetFeaturesInArea(const float &x, const float &y, const float &r) const
{
    vector<size_t> vIndices;
    vIndices.reserve(N);
    //最小网格x坐标
    const int nMinCellX = max(0,(int)floor((x-mnMinX-r)*mfGridElementWidthInv));
    if(nMinCellX>=mnGridCols)
        return vIndices;
    //最大网格x坐标
    const int nMaxCellX = min((int)mnGridCols-1,(int)ceil((x-mnMinX+r)*mfGridElementWidthInv));
    if(nMaxCellX<0)
        return vIndices;
    //最小网格y坐标
    const int nMinCellY = max(0,(int)floor((y-mnMinY-r)*mfGridElementHeightInv));
    if(nMinCellY>=mnGridRows)
        return vIndices;
    //最大网格y坐标
    const int nMaxCellY = min((int)mnGridRows-1,(int)ceil((y-mnMinY+r)*mfGridElementHeightInv));
    if(nMaxCellY<0)
        return vIndices;
    // 遍历区域内的所有网格
    for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
    {
        for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
        {
            const vector<size_t> vCell = mGrid[ix][iy];  //得到该网格内的特征点序列
            for(size_t j=0, jend=vCell.size(); j<jend; j++)//遍历每个网格内特征点序列
            {
                const cv::KeyPoint &kpUn = mvKeysUn[vCell[j]];
                const float distx = kpUn.pt.x-x;
                const float disty = kpUn.pt.y-y;

                if(fabs(distx)<r && fabs(disty)<r)     //除去在网格内却不在区域内的特征点，并将剩下的特征点存入到vIndices中
                    vIndices.push_back(vCell[j]);
            }
        }
    }

    return vIndices;
}
//判断坐标为(x,y)的点是否在图片内
bool KeyFrame::IsInImage(const float &x, const float &y) const
{
    return (x>=mnMinX && x<mnMaxX && y>=mnMinY && y<mnMaxY);
}
// 将该关键帧的第i个特征点投影到世界坐标系下
cv::Mat KeyFrame::UnprojectStereo(int i)
{
    const float z = mvDepth[i];
    if(z>0)
    {
        const float u = mvKeys[i].pt.x;
        const float v = mvKeys[i].pt.y;
        const float x = (u-cx)*z*invfx;
        const float y = (v-cy)*z*invfy;
        cv::Mat x3Dc = (cv::Mat_<float>(3,1) << x, y, z); //得到相机坐标系下的该特征点的坐标
        
        unique_lock<mutex> lock(mMutexPose);
        return Twc.rowRange(0,3).colRange(0,3)*x3Dc+Twc.rowRange(0,3).col(3);//得到世界坐标系下该特征点的坐标  R*x3Dc+t
    }
    else
        return cv::Mat();
}
//计算当前关键帧的场景深度  q=2代表中值(该关键中所有地图点的中值)
float KeyFrame::ComputeSceneMedianDepth(const int q)
{
    vector<MapPoint*> vpMapPoints;
    cv::Mat Tcw_;
    {
        unique_lock<mutex> lock(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPose);
        vpMapPoints = mvpMapPoints;
        Tcw_ = Tcw.clone();
    }

    vector<float> vDepths;
    vDepths.reserve(N);
    cv::Mat Rcw2 = Tcw_.row(2).colRange(0,3);    //R的最后一行（第三行）
    Rcw2 = Rcw2.t();
    float zcw = Tcw_.at<float>(2,3);   //t的最后一个元素（第三个元素）
    for(int i=0; i<N; i++)
    {
        if(mvpMapPoints[i])
        {
            MapPoint* pMP = mvpMapPoints[i];
            cv::Mat x3Dw = pMP->GetWorldPos();  //得到地图点pMP的世界坐标
            float z = Rcw2.dot(x3Dw)+zcw;// (R*x3Dw+t)的第三行，即z。      Rcw2.dot(x3Dw)代表Rcw2和x3Dw的数量积
            vDepths.push_back(z);
        }
    }

    sort(vDepths.begin(),vDepths.end());   //对该关键帧的所有地图点进行排序，并若q=2,返回中值

    return vDepths[(vDepths.size()-1)/q];
}

} //namespace ORB_SLAM
