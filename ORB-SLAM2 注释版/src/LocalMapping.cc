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

#include "LocalMapping.h"
#include "LoopClosing.h"
#include "ORBmatcher.h"
#include "Optimizer.h"

#include<mutex>

namespace ORB_SLAM2
{

LocalMapping::LocalMapping(Map *pMap, const float bMonocular):
    mbMonocular(bMonocular), mbResetRequested(false), mbFinishRequested(false), mbFinished(true), mpMap(pMap),
    mbAbortBA(false), mbStopped(false), mbStopRequested(false), mbNotStop(false), mbAcceptKeyFrames(true)
{
}
//设置对应的回环检测线程
void LocalMapping::SetLoopCloser(LoopClosing* pLoopCloser)
{
    mpLoopCloser = pLoopCloser;
}
// 设置相应的追踪线程
void LocalMapping::SetTracker(Tracking *pTracker)
{
    mpTracker=pTracker;
}
// 局部建图线程循环主函数
void LocalMapping::Run()
{

    mbFinished = false;
        // Loopclosing中的关键帧是LocalMapping发送过来的，LocalMapping是Tracking中发过来的
        // 在LocalMapping中通过InsertKeyFrame将关键帧插入闭环检测队列mlpLoopKeyFrameQueue
        // 闭环检测队列mlpLoopKeyFrameQueue中的关键帧不为空
    while(1)
    {
        // Tracking will see that Local Mapping is busy  设置此时不能接受新的关键帧
        
        SetAcceptKeyFrames(false);

        // Check if there are keyframes in the queue   检测在队列中是否存在新的关键帧,如果存在进行局部地图构建
        if(CheckNewKeyFrames())
        {
            // BoW conversion and insertion in Map
            ProcessNewKeyFrame();

            // Check recent MapPoints  检测上一关键帧进行时新添加的局部地图点
            MapPointCulling();

            // Triangulate new MapPoints  添加新的地图点
            CreateNewMapPoints();

            if(!CheckNewKeyFrames())    // 如果当前关键帧是当前关键帧集中的最后一个关键帧
            {
                // Find more matches in neighbor keyframes and fuse point duplications
                SearchInNeighbors();
            }

            mbAbortBA = false;

            if(!CheckNewKeyFrames() && !stopRequested())   // 如果当前关键帧为当前关键帧集中的最后一个关键帧,则进行局部BA优化  并检测是否存在冗余关键帧
            {
                // Local BA    在地图中存在关键帧数量大于2
                if(mpMap->KeyFramesInMap()>2)   
                    Optimizer::LocalBundleAdjustment(mpCurrentKeyFrame,&mbAbortBA, mpMap);

                // Check redundant local Keyframes   检测冗余关键帧
                KeyFrameCulling();
            }
	    //  将关键帧加入回环检测线程
            mpLoopCloser->InsertKeyFrame(mpCurrentKeyFrame);
        }
        else if(Stop())
        {
            // Safe area to stop
            while(isStopped() && !CheckFinish())
            {
                usleep(3000);//3ms
            }
            if(CheckFinish())
                break;
        }

        ResetIfRequested();

        // Tracking will see that Local Mapping is busy  设置此时可以进行新关键帧的插入
        SetAcceptKeyFrames(true);

        if(CheckFinish())
            break;

        usleep(3000);//3ms
    }

    SetFinish();
}
//在局部地图中插入新关键帧
void LocalMapping::InsertKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexNewKFs);
    mlNewKeyFrames.push_back(pKF);
    mbAbortBA=true;
}

//检测是否存在新关键帧   返回mlNewKeyFrames队列是否为空
bool LocalMapping::CheckNewKeyFrames()
{
    unique_lock<mutex> lock(mMutexNewKFs);
    return(!mlNewKeyFrames.empty());
}
//如果有新的关键帧,对于新关键帧的操作
/***********************************************
 1 从新关键帧队列mlNewKeyFrames中取一新关键帧并将其从新关键帧列表中删除(避免重复操作新关键帧)
 2 计算新关键帧的BOW向量和Feature向量
 3 将该关键帧对应的地图点与该关键帧关联,并更新该地图点的平均观测方向和描述子
 4 更新Covisibility图
 5 将关键帧插入全局地图中
 ***********************************************/
void LocalMapping::ProcessNewKeyFrame()
{
    {
      //取新关键帧列表中的一帧  并讲该帧从关键帧列表中删除
        unique_lock<mutex> lock(mMutexNewKFs);
        mpCurrentKeyFrame = mlNewKeyFrames.front();
        mlNewKeyFrames.pop_front();
    }

    // Compute Bags of Words structures  计算该关键帧的BOW向量和Feature向量
    mpCurrentKeyFrame->ComputeBoW();

    // Associate MapPoints to the new keyframe and update normal and descriptor  
    // 得到与该关键帧相关的地图点  将地图点和新关键帧关联,并更新normal(平均观测方向)和描述子
    const vector<MapPoint*> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();

    for(size_t i=0; i<vpMapPointMatches.size(); i++)
    {
        MapPoint* pMP = vpMapPointMatches[i];  //遍历每个地图点
        if(pMP)
        {
            if(!pMP->isBad())//地图点是好的
            {
	      //如果该地图点不在当前关键帧中,那么关联该关键帧 并更新改关键点的平均观测方向  计算最优描述子(单目)
                if(!pMP->IsInKeyFrame(mpCurrentKeyFrame))  
                {
                    pMP->AddObservation(mpCurrentKeyFrame, i);
                    pMP->UpdateNormalAndDepth();
                    pMP->ComputeDistinctiveDescriptors();
                }
                //将该地图点添加到新添加地图点的容器中 mlpRecentAddedMapPoints
                else // this can only happen for new stereo points inserted by the Tracking  (非单目)
                {
                    mlpRecentAddedMapPoints.push_back(pMP);
                }
            }
        }
    }    

    // Update links in the Covisibility Graph   更新 Covisibility 图 的边
    mpCurrentKeyFrame->UpdateConnections();

    // Insert Keyframe in Map  将该关键帧插入全局地图中
    mpMap->AddKeyFrame(mpCurrentKeyFrame);
}
// 检测新添加的局部地图点是否满足条件   筛选出好的地图点
// 筛选条件:地图点是否是好的,地图点的查找率大于0.25, 该地图点第一关键帧(第一次观察到改地图点的帧id)与当前帧id相隔距离, 该关键点的被观察到的次数
void LocalMapping::MapPointCulling()
{
    // Check Recent Added MapPoints  检测当前新添加的地图点
    list<MapPoint*>::iterator lit = mlpRecentAddedMapPoints.begin();
    const unsigned long int nCurrentKFid = mpCurrentKeyFrame->mnId;

    int nThObs;  //所有观察到该地图点的关键帧数量的阈值
    if(mbMonocular)
        nThObs = 2;
    else
        nThObs = 3;
    const int cnThObs = nThObs;

    while(lit!=mlpRecentAddedMapPoints.end())
    {
        MapPoint* pMP = *lit;
        if(pMP->isBad())  //如果该地图点是坏的,那么就擦除该地图点
        {
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if(pMP->GetFoundRatio()<0.25f )   //该地图点的查找率如果小于0.25 那么也将其删除,并将该地图点设置成坏的地图点
        {
            pMP->SetBadFlag();
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        //如果当前帧与该地图点第一观察关键帧相隔大于等于2并且观察到该地图点的关键帧数量小于3  则认为该地图点是坏的,擦除该局部地图点
        else if(((int)nCurrentKFid-(int)pMP->mnFirstKFid)>=2 && pMP->Observations()<=cnThObs)  
        {
            pMP->SetBadFlag();
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if(((int)nCurrentKFid-(int)pMP->mnFirstKFid)>=3)    //如果当前关键帧与第一次观察到该地图点的关键帧相隔大于等于3帧 则擦除该地图点
            lit = mlpRecentAddedMapPoints.erase(lit);
        else   //否则判定该地图点是好的
            lit++;
    }
}
//建立新的地图点
/*
 * 步骤:  1. 在当前关键帧的共视关键帧中找到共视程度最高的nn帧相邻帧vpNeighKFs
 * 		2. 遍历相邻关键帧vpNeighKFs,将当前关键帧和共视关键帧进行三角测量和对极约束
 * 		3. 对每次匹配得到的地图点(追踪线程未追踪到的地图点)进行筛选,看是否满足三角测量和对极约束  并根据对极约束计算三维点坐标(单目),对于双目和rgbd直接可以得到3D坐标
 * 			判断该匹配点是否是好的地图点  1> 两帧中的深度都是正的  2> 地图点在两帧的重投影误差 3> 检测地图点的尺度连续性
 * 		4. 如果三角化是成功的,那么建立新的地图点,并设置地图点的相关属性(a.观测到该MapPoint的关键帧  b.该MapPoint的描述子  c.该MapPoint的平均观测方向和深度范围),
 * 			然后将地图点加入当前关键帧和全局地图中
 * 备注: 注意对于单目相机的地图点三维坐标的确立需要根据三角测量来确定.
*/
void LocalMapping::CreateNewMapPoints()
{
    // Retrieve neighbor keyframes in covisibility graph
    int nn = 10;
    if(mbMonocular)//  单目相机
        nn=20;
    const vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);//取与该关键帧关联性最强的前nn个关键帧

    ORBmatcher matcher(0.6,false);

    cv::Mat Rcw1 = mpCurrentKeyFrame->GetRotation();   //当前关键帧的旋转矩阵
    cv::Mat Rwc1 = Rcw1.t();
    cv::Mat tcw1 = mpCurrentKeyFrame->GetTranslation();  //当前关键帧的平移矩阵
    cv::Mat Tcw1(3,4,CV_32F);
    Rcw1.copyTo(Tcw1.colRange(0,3));
    tcw1.copyTo(Tcw1.col(3));
    cv::Mat Ow1 = mpCurrentKeyFrame->GetCameraCenter();  //当前关键帧的相机中心点
    //当前关键帧的相机参数
    const float &fx1 = mpCurrentKeyFrame->fx;
    const float &fy1 = mpCurrentKeyFrame->fy;
    const float &cx1 = mpCurrentKeyFrame->cx;
    const float &cy1 = mpCurrentKeyFrame->cy;
    const float &invfx1 = mpCurrentKeyFrame->invfx;
    const float &invfy1 = mpCurrentKeyFrame->invfy;
    //高斯金字塔的缩放比例*1.5
    const float ratioFactor = 1.5f*mpCurrentKeyFrame->mfScaleFactor;  

    int nnew=0;  //新添加的地图点的数量

    // Search matches with epipolar restriction and triangulate     对极约束  和 三角测量
    for(size_t i=0; i<vpNeighKFs.size(); i++)
    {
        if(i>0 && CheckNewKeyFrames())
            return;

        KeyFrame* pKF2 = vpNeighKFs[i];

        // Check first that baseline is not too short   检测基线是否太短
        cv::Mat Ow2 = pKF2->GetCameraCenter();
        cv::Mat vBaseline = Ow2-Ow1;
        const float baseline = cv::norm(vBaseline);

        if(!mbMonocular)  //如果不是单目,检测基线的长度,如果基线满足要求则不需要进行三角测量计算地图点深度
        {
            if(baseline<pKF2->mb)
            continue;
        }
        else //如果是单目 则检测基线深度比  因为单目的深度是不确定的
        {
            const float medianDepthKF2 = pKF2->ComputeSceneMedianDepth(2);  //计算当前关键帧的场景深度
            const float ratioBaselineDepth = baseline/medianDepthKF2;   //基线/场景深度    称为基线深度比

            if(ratioBaselineDepth<0.01)  //如果基线深度比小于0.01 则搜索下一个关联最强关键帧
                continue;
        }

        // Compute Fundamental Matrix     计算当前帧和关联关键帧之间的基础矩阵Fundamental matrix  对极约束
        cv::Mat F12 = ComputeF12(mpCurrentKeyFrame,pKF2);

        // Search matches that fullfil epipolar constraint      两帧之间进行特征点匹配
	//  存储在追踪线程下没有被建立的地图点(没有从三种追踪方式追踪到的地图点)
        vector<pair<size_t,size_t> > vMatchedIndices;
	// 匹配两关键帧中未追踪到的特征点
        matcher.SearchForTriangulation(mpCurrentKeyFrame,pKF2,F12,vMatchedIndices,false);

        cv::Mat Rcw2 = pKF2->GetRotation();
        cv::Mat Rwc2 = Rcw2.t();
        cv::Mat tcw2 = pKF2->GetTranslation();
        cv::Mat Tcw2(3,4,CV_32F);
        Rcw2.copyTo(Tcw2.colRange(0,3));
        tcw2.copyTo(Tcw2.col(3));

        const float &fx2 = pKF2->fx;
        const float &fy2 = pKF2->fy;
        const float &cx2 = pKF2->cx;
        const float &cy2 = pKF2->cy;
        const float &invfx2 = pKF2->invfx;
        const float &invfy2 = pKF2->invfy;

        // Triangulate each match   对每对匹配通过三角化生成3D点,和 Triangulate函数差不多
        const int nmatches = vMatchedIndices.size();
        for(int ikp=0; ikp<nmatches; ikp++)
        {
            const int &idx1 = vMatchedIndices[ikp].first;
            const int &idx2 = vMatchedIndices[ikp].second;

            const cv::KeyPoint &kp1 = mpCurrentKeyFrame->mvKeysUn[idx1];
            const float kp1_ur=mpCurrentKeyFrame->mvuRight[idx1];
            bool bStereo1 = kp1_ur>=0;// 单目的右眼坐标为-1

            const cv::KeyPoint &kp2 = pKF2->mvKeysUn[idx2];
            const float kp2_ur = pKF2->mvuRight[idx2];
            bool bStereo2 = kp2_ur>=0;

            // Check parallax between rays  两匹配帧中特征点的归一化坐标系坐标
            cv::Mat xn1 = (cv::Mat_<float>(3,1) << (kp1.pt.x-cx1)*invfx1, (kp1.pt.y-cy1)*invfy1, 1.0);
            cv::Mat xn2 = (cv::Mat_<float>(3,1) << (kp2.pt.x-cx2)*invfx2, (kp2.pt.y-cy2)*invfy2, 1.0);
	    // 两特征点在世界坐标系下的坐标分别为  Rwc1 * xn1 + twc1  ,Rwc2 * xn2 + twc2    twc1和twc2分别为两相机中心
	    // 两特征点的夹角cos值为: (Rwc1 * xn1 + twc1 - twc1)(Rwc2 * xn2 + twc2 - twc2)/(norm(Rwc1 * xn1 + twc1 - twc1)norm(Rwc2 * xn2 + twc2 - twc2))
	    // 计算两特征点的方向夹角   
            cv::Mat ray1 = Rwc1*xn1;
            cv::Mat ray2 = Rwc2*xn2;
            const float cosParallaxRays = ray1.dot(ray2)/(cv::norm(ray1)*cv::norm(ray2));

            float cosParallaxStereo = cosParallaxRays+1;
            float cosParallaxStereo1 = cosParallaxStereo;
            float cosParallaxStereo2 = cosParallaxStereo;

            if(bStereo1)
                cosParallaxStereo1 = cos(2*atan2(mpCurrentKeyFrame->mb/2,mpCurrentKeyFrame->mvDepth[idx1]));
            else if(bStereo2)
                cosParallaxStereo2 = cos(2*atan2(pKF2->mb/2,pKF2->mvDepth[idx2]));

            cosParallaxStereo = min(cosParallaxStereo1,cosParallaxStereo2);
	    // 匹配点三维坐标
            cv::Mat x3D;
	    // 计算匹配点三维坐标,,即确定深度,如果为单目则用三角测量方式,对应第一个if   ,如果两帧中一帧为双目或rgbd帧,则直接可以得到三维点坐标
            if(cosParallaxRays<cosParallaxStereo && cosParallaxRays>0 && (bStereo1 || bStereo2 || cosParallaxRays<0.9998))   // 两帧都是单目帧  并且两相机下两特征点夹角不为平行
            {
                // Linear Triangulation Method  三角化线性方法
		// Trianularization: 已知匹配特征点对{x x'} 和 各自相机矩阵{P P'}, 估计三维点 X
		// x' = P'X  x = PX
		// 它们都属于 x = aPX模型
		//                                               |X|
		// |x|        |p1 p2  p3  p4 |    |Y|           |x|        |--p0--||.|
		// |y| = a |p5 p6  p7  p8 |    |Z| ===>|y| = a|--p1--||X|
		// |z|        |p9 p10 p11 p12||1|           |z|       |--p2--||.|
		// 采用DLT的方法：x叉乘PX = 0
		// |yp2 -  p1|        |0|
		// |p0 -  xp2| X = |0|
		// |xp1 - yp0|       |0|
		// 两个点:
		// |yp2   -  p1  |        |0|
		// |p0    -  xp2 | X = |0| ===> AX = 0
		// |y'p2' -  p1' |       |0|
		// |p0'   - x'p2'|       |0|
		// 变成程序中的形式：
		// |xp2  - p0 |        |0|
		// |yp2  - p1 | X = |0| ===> AX = 0
		// |x'p2'- p0'|        |0|
		// |y'p2'- p1'|        |0|
	      // xn1和xn2已经是相机归一化平面的坐标,所以这里的P就相当与T
                cv::Mat A(4,4,CV_32F);
                A.row(0) = xn1.at<float>(0)*Tcw1.row(2)-Tcw1.row(0);
                A.row(1) = xn1.at<float>(1)*Tcw1.row(2)-Tcw1.row(1);
                A.row(2) = xn2.at<float>(0)*Tcw2.row(2)-Tcw2.row(0);
                A.row(3) = xn2.at<float>(1)*Tcw2.row(2)-Tcw2.row(1);

                cv::Mat w,u,vt;
                cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);

                x3D = vt.row(3).t();

                if(x3D.at<float>(3)==0)
                    continue;

                // Euclidean coordinates
                x3D = x3D.rowRange(0,3)/x3D.at<float>(3);

            }
            else if(bStereo1 && cosParallaxStereo1<cosParallaxStereo2)  // 第一帧为双目或rgbd帧
            {
                x3D = mpCurrentKeyFrame->UnprojectStereo(idx1);                
            }
            else if(bStereo2 && cosParallaxStereo2<cosParallaxStereo1)  // 第二帧为双目或rgbd帧
            {
                x3D = pKF2->UnprojectStereo(idx2);
            }
            else
                continue; //No stereo and very low parallax

            cv::Mat x3Dt = x3D.t();

            //Check triangulation in front of cameras   检验该地图点在第一关键帧中的深度是否大于0
            float z1 = Rcw1.row(2).dot(x3Dt)+tcw1.at<float>(2);
            if(z1<=0)
                continue;
	    //检验该地图点在第一关键帧中的深度是否大于0
            float z2 = Rcw2.row(2).dot(x3Dt)+tcw2.at<float>(2);
            if(z2<=0)
                continue;

            //Check reprojection error in first keyframe  检测在第一帧中该第地图点的重投影误差
            const float &sigmaSquare1 = mpCurrentKeyFrame->mvLevelSigma2[kp1.octave];
            const float x1 = Rcw1.row(0).dot(x3Dt)+tcw1.at<float>(0);
            const float y1 = Rcw1.row(1).dot(x3Dt)+tcw1.at<float>(1);
            const float invz1 = 1.0/z1;

            if(!bStereo1)  // 第一帧为单目帧  https://baike.baidu.com/item/%E5%8D%A1%E6%96%B9%E5%88%86%E5%B8%83
            {
                float u1 = fx1*x1*invz1+cx1;
                float v1 = fy1*y1*invz1+cy1;
                float errX1 = u1 - kp1.pt.x;
                float errY1 = v1 - kp1.pt.y;
		// 卡方分布定义:有n个独立变量符合标准正态分布  那么这n个自由变量的平方和就符合卡方分布
                if((errX1*errX1+errY1*errY1)>5.991*sigmaSquare1)   // 概率95%  自由度为2 的情况下 要求误差小于5.991  参考卡方分布的相关知识
                    continue;
            }
            else  // 第一帧为双目帧
            {
                float u1 = fx1*x1*invz1+cx1;
                float u1_r = u1 - mpCurrentKeyFrame->mbf*invz1;
                float v1 = fy1*y1*invz1+cy1;
                float errX1 = u1 - kp1.pt.x;
                float errY1 = v1 - kp1.pt.y;
                float errX1_r = u1_r - kp1_ur;
                if((errX1*errX1+errY1*errY1+errX1_r*errX1_r)>7.8*sigmaSquare1)// 概率95%  自由度为3 的情况下 要求误差小于7.8 参考卡方分布的相关知识
                    continue;
            }

            //Check reprojection error in second keyframe  检测在第二帧中该第地图点的重投影误差
            const float sigmaSquare2 = pKF2->mvLevelSigma2[kp2.octave];
            const float x2 = Rcw2.row(0).dot(x3Dt)+tcw2.at<float>(0);
            const float y2 = Rcw2.row(1).dot(x3Dt)+tcw2.at<float>(1);
            const float invz2 = 1.0/z2;
            if(!bStereo2)
            {
                float u2 = fx2*x2*invz2+cx2;
                float v2 = fy2*y2*invz2+cy2;
                float errX2 = u2 - kp2.pt.x;
                float errY2 = v2 - kp2.pt.y;
                if((errX2*errX2+errY2*errY2)>5.991*sigmaSquare2)
                    continue;
            }
            else
            {
                float u2 = fx2*x2*invz2+cx2;
                float u2_r = u2 - mpCurrentKeyFrame->mbf*invz2;
                float v2 = fy2*y2*invz2+cy2;
                float errX2 = u2 - kp2.pt.x;
                float errY2 = v2 - kp2.pt.y;
                float errX2_r = u2_r - kp2_ur;
                if((errX2*errX2+errY2*errY2+errX2_r*errX2_r)>7.8*sigmaSquare2)
                    continue;
            }

            //Check scale consistency   检查尺度的连续性
            //检测该地图点离相机光心的距离
            cv::Mat normal1 = x3D-Ow1;
            float dist1 = cv::norm(normal1);

            cv::Mat normal2 = x3D-Ow2;
            float dist2 = cv::norm(normal2);

            if(dist1==0 || dist2==0)
                continue;
	    // ratioDist是不考虑金字塔尺度下的距离比例
            const float ratioDist = dist2/dist1;
	    // 金字塔尺度因子的比例
            const float ratioOctave = mpCurrentKeyFrame->mvScaleFactors[kp1.octave]/pKF2->mvScaleFactors[kp2.octave];

            /*if(fabs(ratioDist-ratioOctave)>ratioFactor)
                continue;*/
	    // ratioDist*ratioFactor < ratioOctave 或 ratioDist/ratioOctave > ratioFactor表明尺度变化是连续的
            if(ratioDist*ratioFactor<ratioOctave || ratioDist>ratioOctave*ratioFactor)
                continue;

            // Triangulation is succesfull  如果三角化是成功的,那么建立新的地图点,并设置地图点的相关属性,然后将地图点加入当前关键帧和全局地图中
	    // 地图点属性:
	    // a.观测到该MapPoint的关键帧
            // b.该MapPoint的描述子
            // c.该MapPoint的平均观测方向和深度范围
            MapPoint* pMP = new MapPoint(x3D,mpCurrentKeyFrame,mpMap);

            pMP->AddObservation(mpCurrentKeyFrame,idx1);            
            pMP->AddObservation(pKF2,idx2);

            mpCurrentKeyFrame->AddMapPoint(pMP,idx1);
            pKF2->AddMapPoint(pMP,idx2);

            pMP->ComputeDistinctiveDescriptors();

            pMP->UpdateNormalAndDepth();

            mpMap->AddMapPoint(pMP);
	    // 步骤6.8：将新产生的点放入检测队列
            // 这些MapPoints都会经过MapPointCulling函数的检验
            mlpRecentAddedMapPoints.push_back(pMP);

            nnew++;
        }
    }
}

/**
 * 检查并融合当前关键帧与相邻帧（两级相邻）重复的MapPoints
 * 1. 找到带融合的关键帧(当前关键帧的两级相邻关键帧)
 * 2. 在目标关键帧中寻找当前关键帧地图点的匹配  并在目标关键帧的地图点中寻找当前关键帧所有地图点的融合点
 * 3. 在当前关键帧中寻找目标关键帧(当前关键帧的两级相邻)所有地图点的匹配  并在当前关键帧的地图点中中寻找目标关键帧所有地图点的融合点
 * 4. 更新特征点融合之后当前关键帧的地图点的最优描述子和该地图点被观察的平均方向以及深度范围
 * 5. 更新当前关键帧地图点融合之后的当前关键帧与关联关键帧的联系
 */
void LocalMapping::SearchInNeighbors()
{
    //  Retrieve neighbor keyframes
    int nn = 10;
    if(mbMonocular)
        nn=20;
    const vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);
    //待融合的关键帧
    vector<KeyFrame*> vpTargetKFs;
    for(vector<KeyFrame*>::const_iterator vit=vpNeighKFs.begin(), vend=vpNeighKFs.end(); vit!=vend; vit++)
    {
        KeyFrame* pKFi = *vit;
        if(pKFi->isBad() || pKFi->mnFuseTargetForKF == mpCurrentKeyFrame->mnId)
            continue;
        vpTargetKFs.push_back(pKFi);  // 一级相邻
        pKFi->mnFuseTargetForKF = mpCurrentKeyFrame->mnId;

        // Extend to some second neighbors
        const vector<KeyFrame*> vpSecondNeighKFs = pKFi->GetBestCovisibilityKeyFrames(5);
        for(vector<KeyFrame*>::const_iterator vit2=vpSecondNeighKFs.begin(), vend2=vpSecondNeighKFs.end(); vit2!=vend2; vit2++)
        {
            KeyFrame* pKFi2 = *vit2;
            if(pKFi2->isBad() || pKFi2->mnFuseTargetForKF==mpCurrentKeyFrame->mnId || pKFi2->mnId==mpCurrentKeyFrame->mnId)
                continue;
            vpTargetKFs.push_back(pKFi2);  //二级相邻
        }
    }


    // Search matches by projection from current KF in target KFs    当前关键帧和目标关键帧集进行搜索匹配
    ORBmatcher matcher;
    // 当前帧的地图点
    vector<MapPoint*> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    // 在目标关键帧中寻找当前关键帧地图点的匹配  并在目标关键帧的地图点中中寻找当前关键帧所有地图点的融合点
    for(vector<KeyFrame*>::iterator vit=vpTargetKFs.begin(), vend=vpTargetKFs.end(); vit!=vend; vit++)
    {
        KeyFrame* pKFi = *vit;

        matcher.Fuse(pKFi,vpMapPointMatches);
    }

    // Search matches by projection from target KFs in current KF
    // 在当前关键帧中寻找目标关键帧(当前关键帧的两级相邻)所有地图点的匹配  并在当前关键帧的地图点中中寻找目标关键帧所有地图点的融合点
    vector<MapPoint*> vpFuseCandidates;
    vpFuseCandidates.reserve(vpTargetKFs.size()*vpMapPointMatches.size());

    for(vector<KeyFrame*>::iterator vitKF=vpTargetKFs.begin(), vendKF=vpTargetKFs.end(); vitKF!=vendKF; vitKF++)
    {
        KeyFrame* pKFi = *vitKF;

        vector<MapPoint*> vpMapPointsKFi = pKFi->GetMapPointMatches();

        for(vector<MapPoint*>::iterator vitMP=vpMapPointsKFi.begin(), vendMP=vpMapPointsKFi.end(); vitMP!=vendMP; vitMP++)// 遍历所有目标关键帧(当前关键帧的两级相邻)中的所有地图点
        {
            MapPoint* pMP = *vitMP;
            if(!pMP)
                continue;
            if(pMP->isBad() || pMP->mnFuseCandidateForKF == mpCurrentKeyFrame->mnId)
                continue;
            pMP->mnFuseCandidateForKF = mpCurrentKeyFrame->mnId;
            vpFuseCandidates.push_back(pMP);
        }
    }
    //在当前关键帧中寻找目标关键帧(当前关键帧的两级相邻)所有地图点的匹配  并在当前关键帧的地图点中中寻找目标关键帧所有地图点的融合点
    matcher.Fuse(mpCurrentKeyFrame,vpFuseCandidates);


    // Update points  更新特征点融合之后当前关键帧的地图点的最优描述子和该地图点被观察的平均方向以及深度范围
    vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    for(size_t i=0, iend=vpMapPointMatches.size(); i<iend; i++)
    {
        MapPoint* pMP=vpMapPointMatches[i];
        if(pMP)
        {
            if(!pMP->isBad())
            {
                pMP->ComputeDistinctiveDescriptors();
                pMP->UpdateNormalAndDepth();
            }
        }
    }

    // Update connections in covisibility graph   更新当前关键帧地图点融合之后的当前关键帧与关联关键帧的联系
    mpCurrentKeyFrame->UpdateConnections();
}
//计算基础矩阵F=K.(-T)*t^R*K.(-1)
cv::Mat LocalMapping::ComputeF12(KeyFrame *&pKF1, KeyFrame *&pKF2)
{
    cv::Mat R1w = pKF1->GetRotation();
    cv::Mat t1w = pKF1->GetTranslation();
    cv::Mat R2w = pKF2->GetRotation();
    cv::Mat t2w = pKF2->GetTranslation();
    // 由于旋转矩阵R是正交矩阵,因此旋转矩阵的逆等于旋转矩阵的转置
    cv::Mat R12 = R1w*R2w.t(); //从相机1-->相机2的旋转矩阵
    // 注意这里的平移矩阵的计算,详情见高翔slam14讲 第三章p8(电子版)
    cv::Mat t12 = -R1w*R2w.t()*t2w+t1w;  //从相机1-->相机2的平移矩阵

    cv::Mat t12x = SkewSymmetricMatrix(t12); //将平移向量转化为对应的反对称矩阵

    const cv::Mat &K1 = pKF1->mK;
    const cv::Mat &K2 = pKF2->mK;


    return K1.t().inv()*t12x*R12*K2.inv(); //计算基础矩阵F=K.(-T)*t^R*K.(-1)
}

void LocalMapping::RequestStop()
{
    unique_lock<mutex> lock(mMutexStop);
    mbStopRequested = true;
    unique_lock<mutex> lock2(mMutexNewKFs);
    mbAbortBA = true;
}
//暂停局部建图进程
bool LocalMapping::Stop()
{
    unique_lock<mutex> lock(mMutexStop);
    if(mbStopRequested && !mbNotStop)
    {
        mbStopped = true;
        cout << "Local Mapping STOP" << endl;
        return true;
    }

    return false;
}
//检测是否局部建图进程被暂停,返回标志量mbStopped
bool LocalMapping::isStopped()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopped;
}

bool LocalMapping::stopRequested()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopRequested;
}

void LocalMapping::Release()
{
    unique_lock<mutex> lock(mMutexStop);
    unique_lock<mutex> lock2(mMutexFinish);
    if(mbFinished)
        return;
    mbStopped = false;
    mbStopRequested = false;
    for(list<KeyFrame*>::iterator lit = mlNewKeyFrames.begin(), lend=mlNewKeyFrames.end(); lit!=lend; lit++)
        delete *lit;
    mlNewKeyFrames.clear();

    cout << "Local Mapping RELEASE" << endl;
}

bool LocalMapping::AcceptKeyFrames()
{
    unique_lock<mutex> lock(mMutexAccept);
    return mbAcceptKeyFrames;
}

void LocalMapping::SetAcceptKeyFrames(bool flag)
{
    unique_lock<mutex> lock(mMutexAccept);
    mbAcceptKeyFrames=flag;
}
// 未暂停标志变量赋值为flag
bool LocalMapping::SetNotStop(bool flag)
{
    unique_lock<mutex> lock(mMutexStop);

    if(flag && mbStopped)
        return false;

    mbNotStop = flag;

    return true;
}
//停止BA优化
void LocalMapping::InterruptBA()
{
    mbAbortBA = true;
}

// 剔除冗余关键帧,检测的是当前关键帧(当前组关键帧没有新的关键帧)的共视关键帧
// 判断方法:如果一个关键帧的地图点有90%被其他至少三个关键帧(相邻和相同尺度)看到  那么我们认为该关键帧是冗余的
void LocalMapping::KeyFrameCulling()
{
    // Check redundant keyframes (only local keyframes)   检测冗余关键帧  如果一个关键帧的地图点有90%被其他至少三个关键帧(相邻和相同尺度)看到  那么我们认为该关键帧是冗余的
    // A keyframe is considered redundant if the 90% of the MapPoints it sees, are seen
    // in at least other 3 keyframes (in the same or finer scale)
    // We only consider close stereo points
    vector<KeyFrame*> vpLocalKeyFrames = mpCurrentKeyFrame->GetVectorCovisibleKeyFrames();

    for(vector<KeyFrame*>::iterator vit=vpLocalKeyFrames.begin(), vend=vpLocalKeyFrames.end(); vit!=vend; vit++)
    {
        KeyFrame* pKF = *vit;
        if(pKF->mnId==0)
            continue;
        const vector<MapPoint*> vpMapPoints = pKF->GetMapPointMatches();

        int nObs = 3;
        const int thObs=nObs;
        int nRedundantObservations=0;
        int nMPs=0;
        for(size_t i=0, iend=vpMapPoints.size(); i<iend; i++)
        {
            MapPoint* pMP = vpMapPoints[i];
            if(pMP)
            {
                if(!pMP->isBad())
                {
                    if(!mbMonocular)
                    {
                        if(pKF->mvDepth[i]>pKF->mThDepth || pKF->mvDepth[i]<0)
                            continue;
                    }

                    nMPs++;
                    if(pMP->Observations()>thObs)
                    {
                        const int &scaleLevel = pKF->mvKeysUn[i].octave;
                        const map<KeyFrame*, size_t> observations = pMP->GetObservations();
                        int nObs=0;
			//计算当前地图点在其可视关键帧中所处的金字塔层数是否和当前关键帧下该地图点的高斯金字塔的层数相同或相邻 ,nObs计相同或相邻的次数
                        for(map<KeyFrame*, size_t>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
                        {
                            KeyFrame* pKFi = mit->first;
                            if(pKFi==pKF)
                                continue;
                            const int &scaleLeveli = pKFi->mvKeysUn[mit->second].octave;

                            if(scaleLeveli<=scaleLevel+1)
                            {
                                nObs++;
                                if(nObs>=thObs)
                                    break;
                            }
                        }
                        if(nObs>=thObs)  // 如果当前点在当前关键帧和其他3个以上的可视关键帧的高斯金字塔层数相同 则nRedundantObservations++
                        {
                            nRedundantObservations++;
                        }
                    }
                }
            }
        }  

        if(nRedundantObservations>0.9*nMPs) // 当前待检测冗余关键帧下有90%以上的点都可以在其他关键帧下找到,则认为该帧是冗余的
            pKF->SetBadFlag();
    }
}
// 将向量v转化为相应的反对称矩阵
cv::Mat LocalMapping::SkewSymmetricMatrix(const cv::Mat &v)
{
    return (cv::Mat_<float>(3,3) <<             0, -v.at<float>(2), v.at<float>(1),
            v.at<float>(2),               0,-v.at<float>(0),
            -v.at<float>(1),  v.at<float>(0),              0);
}

void LocalMapping::RequestReset()
{
    {
        unique_lock<mutex> lock(mMutexReset);
        mbResetRequested = true;
    }

    while(1)
    {
        {
            unique_lock<mutex> lock2(mMutexReset);
            if(!mbResetRequested)
                break;
        }
        usleep(3000);
    }
}

void LocalMapping::ResetIfRequested()
{
    unique_lock<mutex> lock(mMutexReset);
    if(mbResetRequested)
    {
        mlNewKeyFrames.clear();
        mlpRecentAddedMapPoints.clear();
        mbResetRequested=false;
    }
}

void LocalMapping::RequestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}
//返回变量mbFinishRequested
bool LocalMapping::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}
//局部建图线程完成
void LocalMapping::SetFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;    
    unique_lock<mutex> lock2(mMutexStop);
    mbStopped = true;
}

bool LocalMapping::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}

} //namespace ORB_SLAM
