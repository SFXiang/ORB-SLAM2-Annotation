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

#include "ORBmatcher.h"

#include<limits.h>

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"

#include<stdint-gcc.h>

using namespace std;

namespace ORB_SLAM2
{

const int ORBmatcher::TH_HIGH = 100;
const int ORBmatcher::TH_LOW = 50;
const int ORBmatcher::HISTO_LENGTH = 30;   //直方图宽度

ORBmatcher::ORBmatcher(float nnratio, bool checkOri): mfNNratio(nnratio), mbCheckOrientation(checkOri)
{
}

/*********************************************************
 通过投影的方式进行匹配     将地图点投影到对应帧的相应区域中,在对应区域中搜索匹配
 在帧F中搜索vpMapPoints地图点的匹配点
   th:  决定投影后搜索半径大大小
   筛选方法:
   在投影区域内遍历所有的特征点,计算描述子距离第一小和描述子距离第二小  当第一小明显小于第二小时证明该特征点具有最优性
   从而确定匹配
 **********************************************************/  
int ORBmatcher::SearchByProjection(Frame &F, const vector<MapPoint*> &vpMapPoints, const float th)
{
    //匹配点数量
    int nmatches=0;

    const bool bFactor = th!=1.0;  //是否需要尺度
    //给每一个地图点在帧F里寻找匹配点
    for(size_t iMP=0; iMP<vpMapPoints.size(); iMP++)   //  遍历所有地图点
    {
        MapPoint* pMP = vpMapPoints[iMP];
        if(!pMP->mbTrackInView)    // 该地图点是否在追踪线程中被追踪到   如果没有则舍弃该地图点
            continue;

        if(pMP->isBad())
            continue;
	// 该特征点被追踪到时在金字塔中的层数
        const int &nPredictedLevel = pMP->mnTrackScaleLevel;

        // The size of the window will depend on the viewing direction   窗口的大小决定观察的方向
        float r = RadiusByViewingCos(pMP->mTrackViewCos);

        if(bFactor)
            r*=th;
	//得到地图点所在投影区域的所有特征点索引
        const vector<size_t> vIndices =
                F.GetFeaturesInArea(pMP->mTrackProjX,pMP->mTrackProjY,r*F.mvScaleFactors[nPredictedLevel],nPredictedLevel-1,nPredictedLevel);

        if(vIndices.empty())
            continue;
	// 地图点的描述子
        const cv::Mat MPdescriptor = pMP->GetDescriptor();

        int bestDist=256;
        int bestLevel= -1;
        int bestDist2=256;
        int bestLevel2 = -1;
        int bestIdx =-1 ;

        // Get best and second matches with near keypoints   在关键点附近查找最好和第二匹配   
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)   //遍历区域内所有的特征点索引
        {
            const size_t idx = *vit;

            if(F.mvpMapPoints[idx])
                if(F.mvpMapPoints[idx]->Observations()>0)
                    continue;

            if(F.mvuRight[idx]>0)
            {
                const float er = fabs(pMP->mTrackProjXR-F.mvuRight[idx]);
                if(er>r*F.mvScaleFactors[nPredictedLevel])
                    continue;
            }
	    // 当前特征点的描述子
            const cv::Mat &d = F.mDescriptors.row(idx);
	    //  两描述子之间的距离
            const int dist = DescriptorDistance(MPdescriptor,d);

            if(dist<bestDist)
            {
                bestDist2=bestDist;
                bestDist=dist;
                bestLevel2 = bestLevel;
                bestLevel = F.mvKeysUn[idx].octave;
                bestIdx=idx;
            }
            else if(dist<bestDist2)
            {
                bestLevel2 = F.mvKeysUn[idx].octave;
                bestDist2=dist;
            }
        }

        // Apply ratio to second match (only if best and second are in the same scale level)   
        if(bestDist<=TH_HIGH)
        {
	  //此条件证明该特征点只与第一距离的描述子距离近,而与其他描述子距离远  
            if(bestLevel==bestLevel2 && bestDist>mfNNratio*bestDist2)  //第一最优描述子距离/第二描述子距离 > mfNNratio  
                continue;

            F.mvpMapPoints[bestIdx]=pMP;
            nmatches++;
        }
    }

    return nmatches;
}

float ORBmatcher::RadiusByViewingCos(const float &viewCos)
{
    if(viewCos>0.998)
        return 2.5;
    else
        return 4.0;
}

// 该点是否满足对极几何 (映射点到极线的距离)
// 参考multiple view geometry in computer vision 9.1 P240
bool ORBmatcher::CheckDistEpipolarLine(const cv::KeyPoint &kp1,const cv::KeyPoint &kp2,const cv::Mat &F12,const KeyFrame* pKF2)
{
    // Epipolar line in second image l = x1'F12 = [a b c]
    // 求出kp1在pKF2上对应的极线
    const float a = kp1.pt.x*F12.at<float>(0,0)+kp1.pt.y*F12.at<float>(1,0)+F12.at<float>(2,0);
    const float b = kp1.pt.x*F12.at<float>(0,1)+kp1.pt.y*F12.at<float>(1,1)+F12.at<float>(2,1);
    const float c = kp1.pt.x*F12.at<float>(0,2)+kp1.pt.y*F12.at<float>(1,2)+F12.at<float>(2,2);

    // 计算kp2特征点到极线的距离：
    // 极线l：ax + by + c = 0
    // (u,v)到l的距离为： |au+bv+c| / sqrt(a^2+b^2)

    const float num = a*kp2.pt.x+b*kp2.pt.y+c;

    const float den = a*a+b*b;

    if(den==0)
        return false;

    const float dsqr = num*num/den;

    // 尺度越大，范围应该越大。
    // 金字塔最底层一个像素就占一个像素，在倒数第二层，一个像素等于最底层1.2个像素（假设金字塔尺度为1.2）  卡方分布中自由度为1 概率为0.05  确定误差的取值为小于3.84
    return dsqr<3.84*pKF2->mvLevelSigma2[kp2.octave];
}
//  关键帧和当前帧通过词袋进行快速匹配   在同一词典节点上搜索匹配    
//  筛选方法:
//          在投影区域内遍历所有的特征点,计算描述子距离第一小和描述子距离第二小  当第一小明显小于第二小时证明该特征点具有最优性
//           从而确定匹配
int ORBmatcher::SearchByBoW(KeyFrame* pKF,Frame &F, vector<MapPoint*> &vpMapPointMatches)
{
    //关键帧特征点对应的地图点
    const vector<MapPoint*> vpMapPointsKF = pKF->GetMapPointMatches();
    //最终返回的地图点
    vpMapPointMatches = vector<MapPoint*>(F.N,static_cast<MapPoint*>(NULL)); 
    //关键帧的特征向量
    const DBoW2::FeatureVector &vFeatVecKF = pKF->mFeatVec;
    
    int nmatches=0;
    
    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);
    const float factor = 1.0f/HISTO_LENGTH;

    // We perform the matching over ORB that belong to the same vocabulary node (at a certain level)  执行匹配在属于同一词典节点（字）
    //关键帧特征向量的迭代器    first代表该特征向量的节点id  second代表该特征点在图像所有特征点集合中的索引
    DBoW2::FeatureVector::const_iterator KFit = vFeatVecKF.begin();
    //当前帧特征向量的迭代器    first代表该特征向量的节点id  second代表该特征点在图像所有特征点集合中的索引
    DBoW2::FeatureVector::const_iterator Fit = F.mFeatVec.begin();
    DBoW2::FeatureVector::const_iterator KFend = vFeatVecKF.end();
    DBoW2::FeatureVector::const_iterator Fend = F.mFeatVec.end();

    while(KFit != KFend && Fit != Fend)
    {
        if(KFit->first == Fit->first)  //如果当前帧和关键帧的当前特征向量属于同一节点
        {
            const vector<unsigned int> vIndicesKF = KFit->second;
            const vector<unsigned int> vIndicesF = Fit->second;

            for(size_t iKF=0; iKF<vIndicesKF.size(); iKF++)  //遍历关键帧所有特征点在图像所有特征点集合中的索引
            {
                const unsigned int realIdxKF = vIndicesKF[iKF];
		//关键帧特征点对应的地图点
                MapPoint* pMP = vpMapPointsKF[realIdxKF];

                if(!pMP)
                    continue;

                if(pMP->isBad())
                    continue;                
		//特征点在关键帧中的描述子
                const cv::Mat &dKF= pKF->mDescriptors.row(realIdxKF);
		//第一最优距离
                int bestDist1=256;
                int bestIdxF =-1 ;
		//第二最优距离
                int bestDist2=256;

                for(size_t iF=0; iF<vIndicesF.size(); iF++)  // 遍历当前帧所有特征点在图像所有特征点集合中的索引
                {
                    const unsigned int realIdxF = vIndicesF[iF];

                    if(vpMapPointMatches[realIdxF])
                        continue;
		    //特征点在当前帧的描述子
                    const cv::Mat &dF = F.mDescriptors.row(realIdxF);
		    //当前帧特征点描述子和关键帧特征点描述子的距离
                    const int dist =  DescriptorDistance(dKF,dF);

                    if(dist<bestDist1)  //如果两描述子距离小于最优距离  则更新第一最优距离和第二最优距离
                    {
                        bestDist2=bestDist1;
                        bestDist1=dist;
                        bestIdxF=realIdxF;
                    }
                    else if(dist<bestDist2)
                    {
                        bestDist2=dist;
                    }
                }

                if(bestDist1<=TH_LOW)   //第一最优距离小于阈值 
                {
		   // 第一最优距离/第二最优距离 < 距离比阈值     此条件证明该特征点只与第一距离的描述子距离近,而与其他描述子距离远  
		   // 提高匹配的正确率,如果不满足该条件,直接视为此次匹配失败
                    if(static_cast<float>(bestDist1)<mfNNratio*static_cast<float>(bestDist2))  
                    {
		        //将匹配到的地图点添加到vpMapPointMatches容器中
                        vpMapPointMatches[bestIdxF]=pMP;   
			// 提取此时匹配点的矫正关键点
                        const cv::KeyPoint &kp = pKF->mvKeysUn[realIdxKF];

                        if(mbCheckOrientation)  // 是否需要判别旋转 
                        {
                            float rot = kp.angle-F.mvKeys[bestIdxF].angle;  // 两特征点之间的旋转角度距离
                            if(rot<0.0)
                                rot+=360.0f;
                            int bin = round(rot*factor);
                            if(bin==HISTO_LENGTH)   //构建旋转角度差直方图
                                bin=0;
                            assert(bin>=0 && bin<HISTO_LENGTH);
                            rotHist[bin].push_back(bestIdxF);  // 将最优匹配点在当前帧的索引添加进rotHist[]容器
                        }
                        nmatches++; //匹配点数量增加1
                    }
                }

            }

            KFit++;//关键帧特征点迭代指针后移
            Fit++;   // 当前帧特征点指针后移
        }
        else if(KFit->first < Fit->first)//如果关键帧的当前特征向量节点id小于当前帧的特征点节点id
        {
            KFit = vFeatVecKF.lower_bound(Fit->first);
        }
        else
        {
            Fit = F.mFeatVec.lower_bound(KFit->first);
        }
    }


    if(mbCheckOrientation)   //如果需要检测旋转
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;
	//选出图像特征点的三个主方向
        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i==ind1 || i==ind2 || i==ind3)
                continue;
            for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)  // 如果匹配点的图像方向不具有区分性 则将该匹配点删除
            {
                vpMapPointMatches[rotHist[i][j]]=static_cast<MapPoint*>(NULL);
                nmatches--;
            }
        }
    }

    return nmatches;
}
/**********************************************************
 * 		通过相似矩阵Scw映射计算匹配点   将地图点向关键帧进行投影
 * 		pKF(in): 待检测关键帧
 * 		Scw(in): 当前关键帧的相似矩阵位姿
 * 		vpPoints(in):待匹配的地图点
 * 		vpMatched(out):得到的匹配点
 * 		th:阈值
 * 	主要思路:将地图点根据相似矩阵映射到当前关键帧中,在映射区域中查找地图点的匹配
 ************************************************/
int ORBmatcher::SearchByProjection(KeyFrame* pKF, cv::Mat Scw, const vector<MapPoint*> &vpPoints, vector<MapPoint*> &vpMatched, int th)
{
    // Get Calibration Parameters for later projection
    const float &fx = pKF->fx;
    const float &fy = pKF->fy;
    const float &cx = pKF->cx;
    const float &cy = pKF->cy;

    // Decompose Scw   将相似矩阵Scw分解为旋转平移和尺度的形式
    cv::Mat sRcw = Scw.rowRange(0,3).colRange(0,3);
    const float scw = sqrt(sRcw.row(0).dot(sRcw.row(0)));
    cv::Mat Rcw = sRcw/scw;
    cv::Mat tcw = Scw.rowRange(0,3).col(3)/scw;
    cv::Mat Ow = -Rcw.t()*tcw;

    // Set of MapPoints already found in the KeyFrame
    set<MapPoint*> spAlreadyFound(vpMatched.begin(), vpMatched.end());
    spAlreadyFound.erase(static_cast<MapPoint*>(NULL));

    int nmatches=0;

    // For each Candidate MapPoint Project and Match
    for(int iMP=0, iendMP=vpPoints.size(); iMP<iendMP; iMP++)
    {
        MapPoint* pMP = vpPoints[iMP];

        // Discard Bad MapPoints and already found
        if(pMP->isBad() || spAlreadyFound.count(pMP))
            continue;

        // Get 3D Coords.  得到当前地图点的世界坐标系坐标
        cv::Mat p3Dw = pMP->GetWorldPos();

        // Transform into Camera Coords.   当前地图点在当前关键帧的相机坐标系下的坐标
        cv::Mat p3Dc = Rcw*p3Dw+tcw;

        // Depth must be positive
        if(p3Dc.at<float>(2)<0.0)
            continue;

        // Project into Image   当前地图点在当前关键帧下的映射像素坐标
        const float invz = 1/p3Dc.at<float>(2);
        const float x = p3Dc.at<float>(0)*invz;
        const float y = p3Dc.at<float>(1)*invz;

        const float u = fx*x+cx;
        const float v = fy*y+cy;

        // Point must be inside the image
        if(!pKF->IsInImage(u,v))
            continue;

        // Depth must be inside the scale invariance region of the point
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        cv::Mat PO = p3Dw-Ow;
        const float dist = cv::norm(PO);

        if(dist<minDistance || dist>maxDistance)
            continue;

        // Viewing angle must be less than 60 deg
        cv::Mat Pn = pMP->GetNormal();

        if(PO.dot(Pn)<0.5*dist)
            continue;

        int nPredictedLevel = pMP->PredictScale(dist,pKF);

        // Search in a radius
        const float radius = th*pKF->mvScaleFactors[nPredictedLevel];

        const vector<size_t> vIndices = pKF->GetFeaturesInArea(u,v,radius);

        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = 256;
        int bestIdx = -1;
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            const size_t idx = *vit;
            if(vpMatched[idx])
                continue;

            const int &kpLevel= pKF->mvKeysUn[idx].octave;

            if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel)
                continue;

            const cv::Mat &dKF = pKF->mDescriptors.row(idx);

            const int dist = DescriptorDistance(dMP,dKF);

            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        if(bestDist<=TH_LOW)
        {
            vpMatched[bestIdx]=pMP;
            nmatches++;
        }

    }

    return nmatches;
}
/*********************************************************
 * 		为地图初始化寻找匹配点
 * 		F1(in):待匹配帧1
 * 		F2(in):待匹配帧2
 * 		vbPrevMatched(in):  F1的特征点位置坐标
 * 		vnMatches12(out):匹配帧2在匹配帧1的匹配索引,下标为匹配帧1的关键点索引,值为匹配帧2的关键点索引
 * 		windowSize(in):在F2帧中搜索特征点的窗口大小
 * 		步骤:
 * 			1. 构建搜索区域  假定匹配帧1和匹配帧2之间没有太大的位移,所以在匹配帧2的帧1特征点位置坐标处寻找对应的匹配点
 * 			2. 通过计算描述子距离来寻找最优匹配距离以及最优匹配距离对应帧2的特征点索引
 * 			3. 如果需要检查旋转,则构建旋转角度直方图来筛选匹配点
 **********************************************************************/
int ORBmatcher::SearchForInitialization(Frame &F1, Frame &F2, vector<cv::Point2f> &vbPrevMatched, vector<int> &vnMatches12, int windowSize)
{
  //匹配点数量
    int nmatches=0;
    vnMatches12 = vector<int>(F1.mvKeysUn.size(),-1);

    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);
    const float factor = 1.0f/HISTO_LENGTH;
    // 匹配点的最优距离
    vector<int> vMatchedDistance(F2.mvKeysUn.size(),INT_MAX);
    //匹配帧1在匹配帧2的匹配索引,下标为匹配帧2的关键点索引,值为匹配帧1的关键点索引
    vector<int> vnMatches21(F2.mvKeysUn.size(),-1);

    for(size_t i1=0, iend1=F1.mvKeysUn.size(); i1<iend1; i1++)
    {
      //F1帧i1索引特征点的关键点
        cv::KeyPoint kp1 = F1.mvKeysUn[i1];
        int level1 = kp1.octave;
        if(level1>0)
            continue;
	//得到F2帧对应区域的特征点索引
        vector<size_t> vIndices2 = F2.GetFeaturesInArea(vbPrevMatched[i1].x,vbPrevMatched[i1].y, windowSize,level1,level1);
	
        if(vIndices2.empty())
            continue;
	// F1帧i1索引特征点的描述子
        cv::Mat d1 = F1.mDescriptors.row(i1);
	// 最优特征距离
        int bestDist = INT_MAX;
	// 第二优特征描述子距离
        int bestDist2 = INT_MAX;
	// 最优匹配点在帧2中的特征索引
        int bestIdx2 = -1;
	// 遍历所有F2中对应区域的特征点 寻找最优特征描述子距离和第二优特征描述子距离
        for(vector<size_t>::iterator vit=vIndices2.begin(); vit!=vIndices2.end(); vit++) 
        {
            size_t i2 = *vit;

            cv::Mat d2 = F2.mDescriptors.row(i2);

            int dist = DescriptorDistance(d1,d2);

            if(vMatchedDistance[i2]<=dist)
                continue;

            if(dist<bestDist)
            {
                bestDist2=bestDist;
                bestDist=dist;
                bestIdx2=i2;
            }
            else if(dist<bestDist2)
            {
                bestDist2=dist;
            }
        }
	// 如果最优描述子距离小于TH_LOW阈值  并且bestDist/bestDist2<mfNNratio阈值,证明  最优距离远小于第二优距离
        if(bestDist<=TH_LOW)
        {
            if(bestDist<(float)bestDist2*mfNNratio)
            {
                if(vnMatches21[bestIdx2]>=0)
                {
                    vnMatches12[vnMatches21[bestIdx2]]=-1;
                    nmatches--;
                }
                // 将最优距离对应的匹配点在帧1和帧2中的索引存储
                vnMatches12[i1]=bestIdx2;
                vnMatches21[bestIdx2]=i1;
                vMatchedDistance[bestIdx2]=bestDist;
                nmatches++;

                if(mbCheckOrientation)  //如果需要旋转则构建旋转角度的直方图
                {
                    float rot = F1.mvKeysUn[i1].angle-F2.mvKeysUn[bestIdx2].angle;
                    if(rot<0.0)
                        rot+=360.0f;
                    int bin = round(rot*factor);
                    if(bin==HISTO_LENGTH)
                        bin=0;
                    assert(bin>=0 && bin<HISTO_LENGTH);
                    rotHist[bin].push_back(i1);
                }
            }
        }

    }

    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i==ind1 || i==ind2 || i==ind3)
                continue;
            for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
            {
                int idx1 = rotHist[i][j];
                if(vnMatches12[idx1]>=0)
                {
                    vnMatches12[idx1]=-1;
                    nmatches--;
                }
            }
        }

    }

    //Update prev matched  更新前一帧的匹配点对vbPrevMatched容器
    for(size_t i1=0, iend1=vnMatches12.size(); i1<iend1; i1++)
        if(vnMatches12[i1]>=0)
            vbPrevMatched[i1]=F2.mvKeysUn[vnMatches12[i1]].pt;

    return nmatches;
}
/*************************************************
 *     根据词袋模型来匹配两关键帧
	pKF1 : 关键帧1
	pKF2:关键帧2
	vpMatches12:匹配地图点
 *********************************************************/
int ORBmatcher::SearchByBoW(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint *> &vpMatches12)
{
    //关键帧1的关键点
    const vector<cv::KeyPoint> &vKeysUn1 = pKF1->mvKeysUn;
    //关键帧1的特征向量   迭代器的first是bow节点,对应的second是bow节点下该图像中的特征点
    const DBoW2::FeatureVector &vFeatVec1 = pKF1->mFeatVec;
    //关键帧1的地图点
    const vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();
    //关键帧1 的描述子
    const cv::Mat &Descriptors1 = pKF1->mDescriptors;

    //关键帧2的关键点
    const vector<cv::KeyPoint> &vKeysUn2 = pKF2->mvKeysUn;
    // 关键帧2的特征向量   迭代器的first是bow节点,对应的second是bow节点下该图像中的特征点
    const DBoW2::FeatureVector &vFeatVec2 = pKF2->mFeatVec;
    // 关键帧2 的地图点
    const vector<MapPoint*> vpMapPoints2 = pKF2->GetMapPointMatches();
    // 关键帧2的描述子
    const cv::Mat &Descriptors2 = pKF2->mDescriptors;
    //匹配到的地图点
    vpMatches12 = vector<MapPoint*>(vpMapPoints1.size(),static_cast<MapPoint*>(NULL));
    //关键帧2中的地图点是否是匹配点的关键帧标志
    vector<bool> vbMatched2(vpMapPoints2.size(),false);
    // 旋转角度差的直方图
    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);
    
    const float factor = 1.0f/HISTO_LENGTH;

    int nmatches = 0;
    //关键帧1的特征向量迭代器
    DBoW2::FeatureVector::const_iterator f1it = vFeatVec1.begin();
    // 关键帧2的特征向量迭代器
    DBoW2::FeatureVector::const_iterator f2it = vFeatVec2.begin();
    DBoW2::FeatureVector::const_iterator f1end = vFeatVec1.end();
    DBoW2::FeatureVector::const_iterator f2end = vFeatVec2.end();

    while(f1it != f1end && f2it != f2end)// 遍历所有的特征点向量
    {
        if(f1it->first == f2it->first) // 如果两特征点在同一词典节点上
        {
            for(size_t i1=0, iend1=f1it->second.size(); i1<iend1; i1++)  //遍历关键帧1的所有特征点
            {
                const size_t idx1 = f1it->second[i1];
		// 特征点1对应的地图点坐标
                MapPoint* pMP1 = vpMapPoints1[idx1];
                if(!pMP1)
                    continue;
                if(pMP1->isBad())
                    continue;
		//特征点1对应的描述子
                const cv::Mat &d1 = Descriptors1.row(idx1);
		// 第一最优描述子距离
                int bestDist1=256;
                int bestIdx2 =-1 ;
		// 第二最优描述子距离
                int bestDist2=256;

                for(size_t i2=0, iend2=f2it->second.size(); i2<iend2; i2++)//遍历关键帧2的所有特征点
                {
                    const size_t idx2 = f2it->second[i2];

                    MapPoint* pMP2 = vpMapPoints2[idx2];

                    if(vbMatched2[idx2] || !pMP2)
                        continue;

                    if(pMP2->isBad())
                        continue;

                    const cv::Mat &d2 = Descriptors2.row(idx2);
		    // 计算两描述子的距离
                    int dist = DescriptorDistance(d1,d2);
		    // 查找第一最优描述子距离和第二最优描述子距离
                    if(dist<bestDist1)
                    {
                        bestDist2=bestDist1;
                        bestDist1=dist;
                        bestIdx2=idx2;
                    }
                    else if(dist<bestDist2)
                    {
                        bestDist2=dist;
                    }
                }

                if(bestDist1<TH_LOW)
                {
                    if(static_cast<float>(bestDist1)<mfNNratio*static_cast<float>(bestDist2))   // 第一最优描述子距离/第二最优描述子距离<mfNNratio
                    {
                        vpMatches12[idx1]=vpMapPoints2[bestIdx2];
                        vbMatched2[bestIdx2]=true;

                        if(mbCheckOrientation) //如果需要检测旋转
                        {
                            float rot = vKeysUn1[idx1].angle-vKeysUn2[bestIdx2].angle;
                            if(rot<0.0)
                                rot+=360.0f;
                            int bin = round(rot*factor);
                            if(bin==HISTO_LENGTH)
                                bin=0;
                            assert(bin>=0 && bin<HISTO_LENGTH);
                            rotHist[bin].push_back(idx1);
                        }
                        nmatches++;
                    }
                }
            }

            f1it++;
            f2it++;
        }
        else if(f1it->first < f2it->first)  //如果特征点1的节点id小于节点2的节点id
        {
            f1it = vFeatVec1.lower_bound(f2it->first);
        }
        else
        {
            f2it = vFeatVec2.lower_bound(f1it->first);
        }
    }

    if(mbCheckOrientation)// 如果需要检测旋转
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i==ind1 || i==ind2 || i==ind3)
                continue;
            for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
            {
                vpMatches12[rotHist[i][j]]=static_cast<MapPoint*>(NULL);
                nmatches--;
            }
        }
    }

    return nmatches;
}
// 通过对极几何来计算匹配点   计算的是在追踪线程下没有计算三维点的特征点(没追踪到)
// 找出 pKF1中 特征点在pKF2中的匹配点  
// 根据BOW向量匹配在同一节点下的特征点  根据匹配点描述子距离最小的点并且满足对极几何的约束
int ORBmatcher::SearchForTriangulation(KeyFrame *pKF1, KeyFrame *pKF2, cv::Mat F12,
                                       vector<pair<size_t, size_t> > &vMatchedPairs, const bool bOnlyStereo)
{   
    //取图片1的特征向量
    const DBoW2::FeatureVector &vFeatVec1 = pKF1->mFeatVec;  
    //取图片2的特征向量
    const DBoW2::FeatureVector &vFeatVec2 = pKF2->mFeatVec; 

    //Compute epipole in second image   根据相机1的相机中心,以及相机2的旋转矩阵和平移矩阵计算相机二的相机中心
    cv::Mat Cw = pKF1->GetCameraCenter();
    cv::Mat R2w = pKF2->GetRotation();
    cv::Mat t2w = pKF2->GetTranslation();
    cv::Mat C2 = R2w*Cw+t2w;
    //步骤0：得到KF1的相机光心在KF2中的坐标（极点坐标）
    const float invz = 1.0f/C2.at<float>(2);
    const float ex =pKF2->fx*C2.at<float>(0)*invz+pKF2->cx;
    const float ey =pKF2->fy*C2.at<float>(1)*invz+pKF2->cy;

    // Find matches between not tracked keypoints   在没有追踪到的关键点之间寻找匹配
    // Matching speed-up by ORB Vocabulary    通过orb词典进行快速匹配
    // Compare only ORB that share the same node  仅仅比较在同一节点上的orb特征点

    int nmatches=0;
    vector<bool> vbMatched2(pKF2->N,false);
    vector<int> vMatches12(pKF1->N,-1);

    vector<int> rotHist[HISTO_LENGTH];   //直方图
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);

    const float factor = 1.0f/HISTO_LENGTH;
    //图像一中的特征点迭代器
    DBoW2::FeatureVector::const_iterator f1it = vFeatVec1.begin();
    //图像二中的特征点迭代器
    DBoW2::FeatureVector::const_iterator f2it = vFeatVec2.begin(); 
    DBoW2::FeatureVector::const_iterator f1end = vFeatVec1.end();
    DBoW2::FeatureVector::const_iterator f2end = vFeatVec2.end();
    // 将左图像的每个特征点与右图像同一node节点的所有特征点
    // 依次检测，判断是否满足对极几何约束，满足约束就是匹配的特征点
    while(f1it!=f1end && f2it!=f2end)  //循环每幅图像的每一个特征点
    {
        if(f1it->first == f2it->first)//如果两个特征点在同一个BOW词典节点上 则寻找匹配
        {
            for(size_t i1=0, iend1=f1it->second.size(); i1<iend1; i1++)//循环图片1 中所有特征点的特征索引
            {
	      //特征索引
                const size_t idx1 = f1it->second[i1];  
                //得到该特征点在关键帧中对应的地图点
                MapPoint* pMP1 = pKF1->GetMapPoint(idx1);  
                
                // If there is already a MapPoint skip
                if(pMP1)
                    continue;

                const bool bStereo1 = pKF1->mvuRight[idx1]>=0;  //该特征点对应的右眼坐标

                if(bOnlyStereo) //如果双目
                    if(!bStereo1)
                        continue;
                // 该关键帧1中该特征向量对应的关键点
                const cv::KeyPoint &kp1 = pKF1->mvKeysUn[idx1];
                //该关键帧1中对应于该特征向量的描述子向量
                const cv::Mat &d1 = pKF1->mDescriptors.row(idx1);
                
                int bestDist = TH_LOW;
                int bestIdx2 = -1;
                //循环图片2中所有特征点的特征索引 找到最小描述子距离的情况下满足对极几何约束条件的匹配点
                for(size_t i2=0, iend2=f2it->second.size(); i2<iend2; i2++)
                {
                    size_t idx2 = f2it->second[i2];
                    
                    MapPoint* pMP2 = pKF2->GetMapPoint(idx2);  
                    
                    // If we have already matched or there is a MapPoint skip
                    if(vbMatched2[idx2] || pMP2)
                        continue;

                    const bool bStereo2 = pKF2->mvuRight[idx2]>=0;  //该特征点对应的右眼坐标

                    if(bOnlyStereo)  //如果是双目   右眼坐标不能为空
                        if(!bStereo2)
                            continue;
                    //关键帧2中特征点2的描述子
                    const cv::Mat &d2 = pKF2->mDescriptors.row(idx2);
                    // 两个关键帧中两描述子的距离
                    const int dist = DescriptorDistance(d1,d2);
                    
                    if(dist>TH_LOW || dist>bestDist)  //如果两特征点之间的描述子距离大于阈值或者大于当前最优距离  ,则跳过该特征点
                        continue;

                    const cv::KeyPoint &kp2 = pKF2->mvKeysUn[idx2];

                    if(!bStereo1 && !bStereo2)//如果两个右眼坐标都不为空
                    {
                        const float distex = ex-kp2.pt.x;
                        const float distey = ey-kp2.pt.y;
                        if(distex*distex+distey*distey<100*pKF2->mvScaleFactors[kp2.octave])
                            continue;
                    }
		    // 检测匹配点是否满足对极几何的约束
                    if(CheckDistEpipolarLine(kp1,kp2,F12,pKF2))
                    {
                        bestIdx2 = idx2;
                        bestDist = dist;
                    }
                }
                
                if(bestIdx2>=0)
                {
                    const cv::KeyPoint &kp2 = pKF2->mvKeysUn[bestIdx2];
                    vMatches12[idx1]=bestIdx2;
                    nmatches++;

                    if(mbCheckOrientation)
                    {
                        float rot = kp1.angle-kp2.angle;
                        if(rot<0.0)
                            rot+=360.0f;
                        int bin = round(rot*factor);
                        if(bin==HISTO_LENGTH)
                            bin=0;
                        assert(bin>=0 && bin<HISTO_LENGTH);
                        rotHist[bin].push_back(idx1);
                    }
                }
            }

            f1it++;  //迭代器指针后移
            f2it++;  //迭代器指针后移
        }
        else if(f1it->first < f2it->first)  //如果图像1中的特征点节点id小于图像2中特征点节点id  则将图像1特征点的节点id跳转到图像2特征点节点id
        {
            f1it = vFeatVec1.lower_bound(f2it->first);
        }
        else    //如果图像2中的特征点节点id小于图像1中特征点节点id  则将图像2特征点的节点id跳转到图像1特征点节点id
        {
            f2it = vFeatVec2.lower_bound(f1it->first);
        }
    }

    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i==ind1 || i==ind2 || i==ind3)
                continue;
            for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
            {
                vMatches12[rotHist[i][j]]=-1;
                nmatches--;
            }
        }

    }

    vMatchedPairs.clear();
    vMatchedPairs.reserve(nmatches);

    for(size_t i=0, iend=vMatches12.size(); i<iend; i++)
    {
        if(vMatches12[i]<0)
            continue;
        vMatchedPairs.push_back(make_pair(i,vMatches12[i]));
    }

    return nmatches;
}
/**
 * @brief 将MapPoints投影到关键帧pKF中，并判断是否有重复的MapPoints
 * 1.如果MapPoint能匹配关键帧的特征点，并且该点有对应的MapPoint，那么将两个MapPoint合并（选择观测数多的）
 * 2.如果MapPoint能匹配关键帧的特征点，并且该点没有对应的MapPoint，那么为该点添加MapPoint
 * @param  pKF         相邻关键帧
 * @param  vpMapPoints 当前关键帧的MapPoints
 * @param  th          搜索半径的因子
 * @return             重复MapPoints的数量
 */
int ORBmatcher::Fuse(KeyFrame *pKF, const vector<MapPoint *> &vpMapPoints, const float th)
{
    cv::Mat Rcw = pKF->GetRotation();
    cv::Mat tcw = pKF->GetTranslation();

    const float &fx = pKF->fx;
    const float &fy = pKF->fy;
    const float &cx = pKF->cx;
    const float &cy = pKF->cy;
    const float &bf = pKF->mbf;

    cv::Mat Ow = pKF->GetCameraCenter();

    int nFused=0;

    const int nMPs = vpMapPoints.size();

    // 遍历所有的MapPoints
    for(int i=0; i<nMPs; i++)
    {
        MapPoint* pMP = vpMapPoints[i];

        if(!pMP)
            continue;

        if(pMP->isBad() || pMP->IsInKeyFrame(pKF))
            continue;

        cv::Mat p3Dw = pMP->GetWorldPos();
        cv::Mat p3Dc = Rcw*p3Dw + tcw;

        // Depth must be positive
        if(p3Dc.at<float>(2)<0.0f)
            continue;

        const float invz = 1/p3Dc.at<float>(2);
        const float x = p3Dc.at<float>(0)*invz;
        const float y = p3Dc.at<float>(1)*invz;

        const float u = fx*x+cx;
        const float v = fy*y+cy;// 步骤1：得到MapPoint在图像上的投影坐标

        // Point must be inside the image
        if(!pKF->IsInImage(u,v))
            continue;

        const float ur = u-bf*invz;

        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        cv::Mat PO = p3Dw-Ow;
        const float dist3D = cv::norm(PO);

        // Depth must be inside the scale pyramid of the image
        if(dist3D<minDistance || dist3D>maxDistance )
            continue;

        // Viewing angle must be less than 60 deg
        cv::Mat Pn = pMP->GetNormal();

        if(PO.dot(Pn)<0.5*dist3D)
            continue;

        int nPredictedLevel = pMP->PredictScale(dist3D,pKF);

        // Search in a radius
        const float radius = th*pKF->mvScaleFactors[nPredictedLevel];// 步骤2：根据MapPoint的深度确定尺度，从而确定搜索范围

        const vector<size_t> vIndices = pKF->GetFeaturesInArea(u,v,radius);

        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius

        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = 256;
        int bestIdx = -1;
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)// 步骤3：遍历搜索范围内的features
        {
            const size_t idx = *vit;

            const cv::KeyPoint &kp = pKF->mvKeysUn[idx];

            const int &kpLevel= kp.octave;

            if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel)
                continue;

            // 计算MapPoint投影的坐标与这个区域特征点的距离，如果偏差很大，直接跳过特征点匹配
            if(pKF->mvuRight[idx]>=0)
            {
                // Check reprojection error in stereo
                const float &kpx = kp.pt.x;
                const float &kpy = kp.pt.y;
                const float &kpr = pKF->mvuRight[idx];
                const float ex = u-kpx;
                const float ey = v-kpy;
                const float er = ur-kpr;
                const float e2 = ex*ex+ey*ey+er*er;

                if(e2*pKF->mvInvLevelSigma2[kpLevel]>7.8)
                    continue;
            }
            else
            {
                const float &kpx = kp.pt.x;
                const float &kpy = kp.pt.y;
                const float ex = u-kpx;
                const float ey = v-kpy;
                const float e2 = ex*ex+ey*ey;

                // 基于卡方检验计算出的阈值（假设测量有一个像素的偏差）
                if(e2*pKF->mvInvLevelSigma2[kpLevel]>5.99)
                    continue;
            }

            const cv::Mat &dKF = pKF->mDescriptors.row(idx);

            const int dist = DescriptorDistance(dMP,dKF);

            if(dist<bestDist)// 找MapPoint在该区域最佳匹配的特征点
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        // If there is already a MapPoint replace otherwise add new measurement
        if(bestDist<=TH_LOW)// 找到了MapPoint在该区域最佳匹配的特征点
        {
            MapPoint* pMPinKF = pKF->GetMapPoint(bestIdx);
            if(pMPinKF)// 如果这个点有对应的MapPoint
            {
                if(!pMPinKF->isBad())// 如果这个MapPoint不是bad，选择哪一个呢？  用被观察次数多的地图点代替被观察次数少的
                {
                    if(pMPinKF->Observations()>pMP->Observations())
                        pMP->Replace(pMPinKF);
                    else
                        pMPinKF->Replace(pMP);
                }
            }
            else// 如果这个点没有对应的MapPoint
            {
                pMP->AddObservation(pKF,bestIdx);
                pKF->AddMapPoint(pMP,bestIdx);
            }
            nFused++;
        }
    }

    return nFused;
}

/*****************************************
 * 根据相似矩阵Scw映射地图点到关键帧  匹配地图点为:vpReplacePoint   用于回环检测
 * 		pKF(in):映射关键帧
 * 		Scw(in):关键帧的相似矩阵位姿
 * 		vpPoints(in):待映射地图点
 * 		th(in):阈值
 * 		vpReplacePoint(out):匹配到的地图点
 * 	主要思路: 将回环地图点根据当前关键帧的位姿(相似矩阵)映射到当前关键帧,在当前关键帧中寻找回环地图点的代替点,存储进vpReplacePoint
 ************************************/
int ORBmatcher::Fuse(KeyFrame *pKF, cv::Mat Scw, const vector<MapPoint *> &vpPoints, float th, vector<MapPoint *> &vpReplacePoint)
{
    // Get Calibration Parameters for later projection
    const float &fx = pKF->fx;
    const float &fy = pKF->fy;
    const float &cx = pKF->cx;
    const float &cy = pKF->cy;

    // Decompose Scw
    cv::Mat sRcw = Scw.rowRange(0,3).colRange(0,3);
    const float scw = sqrt(sRcw.row(0).dot(sRcw.row(0)));
    cv::Mat Rcw = sRcw/scw;
    cv::Mat tcw = Scw.rowRange(0,3).col(3)/scw;
    cv::Mat Ow = -Rcw.t()*tcw;

    // Set of MapPoints already found in the KeyFrame
    const set<MapPoint*> spAlreadyFound = pKF->GetMapPoints();

    int nFused=0;

    const int nPoints = vpPoints.size();

    // For each candidate MapPoint project and match  将地图点映射为当前关键帧的像素坐标,根据三维点深度确定搜索半径从而确定搜索区域,在搜索区域内寻找匹配点
    for(int iMP=0; iMP<nPoints; iMP++)
    {
        MapPoint* pMP = vpPoints[iMP];

        // Discard Bad MapPoints and already found
        if(pMP->isBad() || spAlreadyFound.count(pMP))
            continue;

        // Get 3D Coords.
        cv::Mat p3Dw = pMP->GetWorldPos();

        // Transform into Camera Coords.
        cv::Mat p3Dc = Rcw*p3Dw+tcw;

        // Depth must be positive
        if(p3Dc.at<float>(2)<0.0f)
            continue;

        // Project into Image
        const float invz = 1.0/p3Dc.at<float>(2);
        const float x = p3Dc.at<float>(0)*invz;
        const float y = p3Dc.at<float>(1)*invz;

        const float u = fx*x+cx;
        const float v = fy*y+cy;

        // Point must be inside the image
        if(!pKF->IsInImage(u,v))
            continue;

        // Depth must be inside the scale pyramid of the image
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        cv::Mat PO = p3Dw-Ow;
        const float dist3D = cv::norm(PO);

        if(dist3D<minDistance || dist3D>maxDistance)
            continue;

        // Viewing angle must be less than 60 deg
        cv::Mat Pn = pMP->GetNormal();

        if(PO.dot(Pn)<0.5*dist3D)
            continue;

        // Compute predicted scale level
        const int nPredictedLevel = pMP->PredictScale(dist3D,pKF);

        // Search in a radius
        const float radius = th*pKF->mvScaleFactors[nPredictedLevel];

        const vector<size_t> vIndices = pKF->GetFeaturesInArea(u,v,radius);

        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius  在搜索区域内寻找匹配

        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = INT_MAX;
        int bestIdx = -1;
        for(vector<size_t>::const_iterator vit=vIndices.begin(); vit!=vIndices.end(); vit++)
        {
            const size_t idx = *vit;
            const int &kpLevel = pKF->mvKeysUn[idx].octave;

            if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel)
                continue;

            const cv::Mat &dKF = pKF->mDescriptors.row(idx);

            int dist = DescriptorDistance(dMP,dKF);

            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        // If there is already a MapPoint replace otherwise add new measurement
        if(bestDist<=TH_LOW)
        {
            MapPoint* pMPinKF = pKF->GetMapPoint(bestIdx);
            if(pMPinKF)
            {
                if(!pMPinKF->isBad())
                    vpReplacePoint[iMP] = pMPinKF;
            }
            else
            {
                pMP->AddObservation(pKF,bestIdx);
                pKF->AddMapPoint(pMP,bestIdx);
            }
            nFused++;
        }
    }

    return nFused;
}

/**************************************************
 * 通过相似矩阵计算匹配地图点
 * 		pKF1(in):关键帧1
 * 		pKF2(in):关键帧2
 * 		vpMatches12(out):匹配地图点
 * 		s12(in):关键帧1->关键帧2的相似矩阵中的尺度参数
 * 		R12(in):关键帧1->关键帧2的相似矩阵中的旋转矩阵参数
 * 		t12(in):关键帧1->关键帧2的相似矩阵中的位移矩阵参数
 * 		th(in):阈值
 * 步骤:
 * 		计算关键帧1 2的相机位姿以及相机1->相机2的位姿变换相似矩阵
 * 		双向匹配:
 * 		根据相似矩阵将关键帧1中的地图点向关键帧2中投影,确定投影区域,并在投影区域内寻找关键帧1中地图点的匹配
 * 		根据相似矩阵将关键帧2中的地图点向关键帧1中投影,确定投影区域,并在投影区域内寻找关键帧2中地图点的匹配
 * 		根据双向匹配结果,如果两次匹配都能成功,则确定该对匹配是有效的.将其存入vpMatches12容器   
 * 		最终返回匹配点对个数
 ****************************************/
int ORBmatcher::SearchBySim3(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint*> &vpMatches12,
                             const float &s12, const cv::Mat &R12, const cv::Mat &t12, const float th)
{
    const float &fx = pKF1->fx;
    const float &fy = pKF1->fy;
    const float &cx = pKF1->cx;
    const float &cy = pKF1->cy;

    // Camera 1 from world
    cv::Mat R1w = pKF1->GetRotation();
    cv::Mat t1w = pKF1->GetTranslation();

    //Camera 2 from world
    cv::Mat R2w = pKF2->GetRotation();
    cv::Mat t2w = pKF2->GetTranslation();

    //Transformation between cameras
    cv::Mat sR12 = s12*R12;
    cv::Mat sR21 = (1.0/s12)*R12.t();
    cv::Mat t21 = -sR21*t12;

    const vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();
    const int N1 = vpMapPoints1.size();

    const vector<MapPoint*> vpMapPoints2 = pKF2->GetMapPointMatches();
    const int N2 = vpMapPoints2.size();

    vector<bool> vbAlreadyMatched1(N1,false);
    vector<bool> vbAlreadyMatched2(N2,false);

    for(int i=0; i<N1; i++)
    {
        MapPoint* pMP = vpMatches12[i];
        if(pMP)
        {
            vbAlreadyMatched1[i]=true;
            int idx2 = pMP->GetIndexInKeyFrame(pKF2);
            if(idx2>=0 && idx2<N2)
                vbAlreadyMatched2[idx2]=true;
        }
    }

    vector<int> vnMatch1(N1,-1);
    vector<int> vnMatch2(N2,-1);

    // Transform from KF1 to KF2 and search   在KF2中寻找KF1各地图点的匹配
    for(int i1=0; i1<N1; i1++)
    {
        MapPoint* pMP = vpMapPoints1[i1];

        if(!pMP || vbAlreadyMatched1[i1])
            continue;

        if(pMP->isBad())
            continue;
	// 关键帧1中的地图点坐标
        cv::Mat p3Dw = pMP->GetWorldPos();
	// 关键帧1中的地图点坐标在相机1下的相机坐标系坐标
        cv::Mat p3Dc1 = R1w*p3Dw + t1w;
	// 关键帧1中的地图点映射到关键帧2的相机坐标系下的坐标
        cv::Mat p3Dc2 = sR21*p3Dc1 + t21;

        // Depth must be positive   检测相机坐标系下三维点坐标的深度
        if(p3Dc2.at<float>(2)<0.0)
            continue;
	// 将关键帧1中的地图点映射到相机2的像素坐标
        const float invz = 1.0/p3Dc2.at<float>(2);
        const float x = p3Dc2.at<float>(0)*invz;
        const float y = p3Dc2.at<float>(1)*invz;

        const float u = fx*x+cx;
        const float v = fy*y+cy;

        // Point must be inside the image   验证像素点坐标必须在图片内
        if(!pKF2->IsInImage(u,v))
            continue;

        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        const float dist3D = cv::norm(p3Dc2);

        // Depth must be inside the scale invariance region   地图点在关键帧2中的深度符合该地图点的最大观测深度和最小观测深度
        if(dist3D<minDistance || dist3D>maxDistance )
            continue;

        // Compute predicted octave    通过地图点在关键帧2的深度预测该地图点在关键帧2中图像金字塔的层数
        const int nPredictedLevel = pMP->PredictScale(dist3D,pKF2);

        // Search in a radius  通过高斯金字塔的层数预测投影半径
        const float radius = th*pKF2->mvScaleFactors[nPredictedLevel];
	// 在关键帧1的地图点投影到关键帧2区域内寻找关键帧1的地图点匹配
        const vector<size_t> vIndices = pKF2->GetFeaturesInArea(u,v,radius);

        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius   得到当前待匹配的关键帧1中的地图点描述子
        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = INT_MAX;
        int bestIdx = -1;
	// 在投影区域内寻找地图点的匹配
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            const size_t idx = *vit;

            const cv::KeyPoint &kp = pKF2->mvKeysUn[idx];

            if(kp.octave<nPredictedLevel-1 || kp.octave>nPredictedLevel)
                continue;

            const cv::Mat &dKF = pKF2->mDescriptors.row(idx);

            const int dist = DescriptorDistance(dMP,dKF);

            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        if(bestDist<=TH_HIGH)
        {
            vnMatch1[i1]=bestIdx;
        }
    }

    // Transform from KF2 to KF1 and search   双向匹配  在KF1中寻找KF2各地图点的匹配
    for(int i2=0; i2<N2; i2++)
    {
        MapPoint* pMP = vpMapPoints2[i2];

        if(!pMP || vbAlreadyMatched2[i2])
            continue;

        if(pMP->isBad())
            continue;

        cv::Mat p3Dw = pMP->GetWorldPos();
        cv::Mat p3Dc2 = R2w*p3Dw + t2w;
        cv::Mat p3Dc1 = sR12*p3Dc2 + t12;

        // Depth must be positive
        if(p3Dc1.at<float>(2)<0.0)
            continue;

        const float invz = 1.0/p3Dc1.at<float>(2);
        const float x = p3Dc1.at<float>(0)*invz;
        const float y = p3Dc1.at<float>(1)*invz;

        const float u = fx*x+cx;
        const float v = fy*y+cy;

        // Point must be inside the image
        if(!pKF1->IsInImage(u,v))
            continue;

        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        const float dist3D = cv::norm(p3Dc1);

        // Depth must be inside the scale pyramid of the image
        if(dist3D<minDistance || dist3D>maxDistance)
            continue;

        // Compute predicted octave
        const int nPredictedLevel = pMP->PredictScale(dist3D,pKF1);

        // Search in a radius of 2.5*sigma(ScaleLevel)
        const float radius = th*pKF1->mvScaleFactors[nPredictedLevel];

        const vector<size_t> vIndices = pKF1->GetFeaturesInArea(u,v,radius);

        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = INT_MAX;
        int bestIdx = -1;
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            const size_t idx = *vit;

            const cv::KeyPoint &kp = pKF1->mvKeysUn[idx];

            if(kp.octave<nPredictedLevel-1 || kp.octave>nPredictedLevel)
                continue;

            const cv::Mat &dKF = pKF1->mDescriptors.row(idx);

            const int dist = DescriptorDistance(dMP,dKF);

            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        if(bestDist<=TH_HIGH)
        {
            vnMatch2[i2]=bestIdx;
        }
    }

    // Check agreement   检测双向匹配的结果都成功,证明是匹配点
    int nFound = 0;

    for(int i1=0; i1<N1; i1++)
    {
        int idx2 = vnMatch1[i1];

        if(idx2>=0)
        {
            int idx1 = vnMatch2[idx2];
            if(idx1==i1)
            {
                vpMatches12[i1] = vpMapPoints2[idx2];
                nFound++;
            }
        }
    }

    return nFound;
}
//   从上一帧映射地图点与当前帧进行匹配，使用追踪线程的追踪上一帧的位姿
//   当前帧和上一帧之间进行匹配   
//   根据上一帧到当前帧的位姿,将上一帧的地图点投影到当前帧,然后将地图点反投影到像素坐标,在像素坐标一定范围内寻找最佳匹配点
 //   注意这里用到的当前帧的位姿是根据上一帧的位姿和上一帧的位姿变化速度来推算的相机位姿
int ORBmatcher::SearchByProjection(Frame &CurrentFrame, const Frame &LastFrame, const float th, const bool bMono)
{
    int nmatches = 0;

    // Rotation Histogram (to check rotation consistency)   旋转直方图   为了检测旋转方向的一致性
    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);
    const float factor = 1.0f/HISTO_LENGTH;
    //当前帧的相机位姿
    const cv::Mat Rcw = CurrentFrame.mTcw.rowRange(0,3).colRange(0,3);
    const cv::Mat tcw = CurrentFrame.mTcw.rowRange(0,3).col(3);
    //世界坐标系到当前相机坐标系的位移向量
    const cv::Mat twc = -Rcw.t()*tcw;
    //上一帧的相机位姿
    const cv::Mat Rlw = LastFrame.mTcw.rowRange(0,3).colRange(0,3);
    const cv::Mat tlw = LastFrame.mTcw.rowRange(0,3).col(3);
    // 上一帧相机坐标系到当前帧相机坐标系的位移向量
    const cv::Mat tlc = Rlw*twc+tlw;
    
    const bool bForward = tlc.at<float>(2)>CurrentFrame.mb && !bMono;
    const bool bBackward = -tlc.at<float>(2)>CurrentFrame.mb && !bMono;
    //追踪上一帧的特征点
    for(int i=0; i<LastFrame.N; i++)  //遍历上一帧所有的特征点
    {
        MapPoint* pMP = LastFrame.mvpMapPoints[i];

        if(pMP)
        {
            if(!LastFrame.mvbOutlier[i])
            {
                // Project    将上一帧的局内点(地图点)映射到当前帧
                cv::Mat x3Dw = pMP->GetWorldPos();
		//上一帧的局内地图点映射到当前帧的地图点坐标
                cv::Mat x3Dc = Rcw*x3Dw+tcw;

		
                const float xc = x3Dc.at<float>(0);
                const float yc = x3Dc.at<float>(1);
                const float invzc = 1.0/x3Dc.at<float>(2);
		//判断地图点深度是否是正值
                if(invzc<0)
                    continue;
		// 计算映射到当前帧的像素坐标
                float u = CurrentFrame.fx*xc*invzc+CurrentFrame.cx;
                float v = CurrentFrame.fy*yc*invzc+CurrentFrame.cy;
		
		//判断映射后的像素坐标是否在图像范围内
                if(u<CurrentFrame.mnMinX || u>CurrentFrame.mnMaxX)
                    continue;
                if(v<CurrentFrame.mnMinY || v>CurrentFrame.mnMaxY)
                    continue;
		//上一帧关键点所在金字塔的层数
                int nLastOctave = LastFrame.mvKeys[i].octave;

                // Search in a window. Size depends on scale   半径大小   特征点所在高斯金字塔的层数决定了半径大小
                float radius = th*CurrentFrame.mvScaleFactors[nLastOctave];
		//投影区域内的特征点索引
                vector<size_t> vIndices2;

                if(bForward)
                    vIndices2 = CurrentFrame.GetFeaturesInArea(u,v, radius, nLastOctave);
                else if(bBackward)
                    vIndices2 = CurrentFrame.GetFeaturesInArea(u,v, radius, 0, nLastOctave);
                else
                    vIndices2 = CurrentFrame.GetFeaturesInArea(u,v, radius, nLastOctave-1, nLastOctave+1);

                if(vIndices2.empty())
                    continue;

                const cv::Mat dMP = pMP->GetDescriptor();

                int bestDist = 256;
                int bestIdx2 = -1;
		// 遍历投影区域内所有的特征点
                for(vector<size_t>::const_iterator vit=vIndices2.begin(), vend=vIndices2.end(); vit!=vend; vit++)
                {
                    const size_t i2 = *vit;
                    if(CurrentFrame.mvpMapPoints[i2])
                        if(CurrentFrame.mvpMapPoints[i2]->Observations()>0)
                            continue;

                    if(CurrentFrame.mvuRight[i2]>0)
                    {
                        const float ur = u - CurrentFrame.mbf*invzc;
                        const float er = fabs(ur - CurrentFrame.mvuRight[i2]);
                        if(er>radius)
                            continue;
                    }

                    const cv::Mat &d = CurrentFrame.mDescriptors.row(i2);
		    //计算描述子距离
                    const int dist = DescriptorDistance(dMP,d);
		    //寻找最小描述子距离
                    if(dist<bestDist)
                    {
                        bestDist=dist;
                        bestIdx2=i2;
                    }
                }
		
                if(bestDist<=TH_HIGH)   //最小描述子距离小于阈值
                {
                    CurrentFrame.mvpMapPoints[bestIdx2]=pMP;
                    nmatches++;

                    if(mbCheckOrientation)  //如果需要检测旋转  则构建旋转方向直方图
                    {
                        float rot = LastFrame.mvKeysUn[i].angle-CurrentFrame.mvKeysUn[bestIdx2].angle;
                        if(rot<0.0)
                            rot+=360.0f;
                        int bin = round(rot*factor);
                        if(bin==HISTO_LENGTH)
                            bin=0;
                        assert(bin>=0 && bin<HISTO_LENGTH);
                        rotHist[bin].push_back(bestIdx2);
                    }
                }
            }
        }
    }

    //Apply rotation consistency   判断旋转方向的一致性
    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i!=ind1 && i!=ind2 && i!=ind3)
            {
                for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)   //只保留三个主方向上的特征点
                {
                    CurrentFrame.mvpMapPoints[rotHist[i][j]]=static_cast<MapPoint*>(NULL);
                    nmatches--;
                }
            }
        }
    }

    return nmatches;
}

int ORBmatcher::SearchByProjection(Frame &CurrentFrame, KeyFrame *pKF, const set<MapPoint*> &sAlreadyFound, const float th , const int ORBdist)
{
    int nmatches = 0;

    const cv::Mat Rcw = CurrentFrame.mTcw.rowRange(0,3).colRange(0,3);
    const cv::Mat tcw = CurrentFrame.mTcw.rowRange(0,3).col(3);
    const cv::Mat Ow = -Rcw.t()*tcw;

    // Rotation Histogram (to check rotation consistency)
    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);
    const float factor = 1.0f/HISTO_LENGTH;

    const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();

    for(size_t i=0, iend=vpMPs.size(); i<iend; i++)
    {
        MapPoint* pMP = vpMPs[i];

        if(pMP)
        {
            if(!pMP->isBad() && !sAlreadyFound.count(pMP))
            {
                //Project
                cv::Mat x3Dw = pMP->GetWorldPos();
                cv::Mat x3Dc = Rcw*x3Dw+tcw;

                const float xc = x3Dc.at<float>(0);
                const float yc = x3Dc.at<float>(1);
                const float invzc = 1.0/x3Dc.at<float>(2);

                const float u = CurrentFrame.fx*xc*invzc+CurrentFrame.cx;
                const float v = CurrentFrame.fy*yc*invzc+CurrentFrame.cy;

                if(u<CurrentFrame.mnMinX || u>CurrentFrame.mnMaxX)
                    continue;
                if(v<CurrentFrame.mnMinY || v>CurrentFrame.mnMaxY)
                    continue;

                // Compute predicted scale level   根据地图点到光心的距离来推算该特征点对应的高斯金字塔层数
                cv::Mat PO = x3Dw-Ow;
                float dist3D = cv::norm(PO);

                const float maxDistance = pMP->GetMaxDistanceInvariance();
                const float minDistance = pMP->GetMinDistanceInvariance();

                // Depth must be inside the scale pyramid of the image
                if(dist3D<minDistance || dist3D>maxDistance)
                    continue;

                int nPredictedLevel = pMP->PredictScale(dist3D,&CurrentFrame);

                // Search in a window
                const float radius = th*CurrentFrame.mvScaleFactors[nPredictedLevel];

                const vector<size_t> vIndices2 = CurrentFrame.GetFeaturesInArea(u, v, radius, nPredictedLevel-1, nPredictedLevel+1);

                if(vIndices2.empty())
                    continue;

                const cv::Mat dMP = pMP->GetDescriptor();

                int bestDist = 256;
                int bestIdx2 = -1;

                for(vector<size_t>::const_iterator vit=vIndices2.begin(); vit!=vIndices2.end(); vit++)
                {
                    const size_t i2 = *vit;
                    if(CurrentFrame.mvpMapPoints[i2])
                        continue;

                    const cv::Mat &d = CurrentFrame.mDescriptors.row(i2);

                    const int dist = DescriptorDistance(dMP,d);

                    if(dist<bestDist)
                    {
                        bestDist=dist;
                        bestIdx2=i2;
                    }
                }

                if(bestDist<=ORBdist)
                {
                    CurrentFrame.mvpMapPoints[bestIdx2]=pMP;
                    nmatches++;

                    if(mbCheckOrientation)
                    {
                        float rot = pKF->mvKeysUn[i].angle-CurrentFrame.mvKeysUn[bestIdx2].angle;
                        if(rot<0.0)
                            rot+=360.0f;
                        int bin = round(rot*factor);
                        if(bin==HISTO_LENGTH)
                            bin=0;
                        assert(bin>=0 && bin<HISTO_LENGTH);
                        rotHist[bin].push_back(bestIdx2);
                    }
                }

            }
        }
    }

    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i!=ind1 && i!=ind2 && i!=ind3)
            {
                for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
                {
                    CurrentFrame.mvpMapPoints[rotHist[i][j]]=NULL;
                    nmatches--;
                }
            }
        }
    }

    return nmatches;
}
/**************************************************
 * 
 * 功能:筛选出在直方图区间内特征点数量最多的三个特征点方向的索引
 * 
     histo  两描述子旋转距离的直方图
     L  直方图宽度
     ind1  第一最小旋转距离的索引
     ind2  第二最小旋转距离的索引
     ind3  第三最小旋转距离的索引
     
     如果max2/max1<0.1  那么证明第二第三方向不具有区分性,则将其索引置位初值
     如果max3/max1<0.1  那么证明第三方向不具有区分性,则将其索引置位初值
 **************************************************************/
void ORBmatcher::ComputeThreeMaxima(vector<int>* histo, const int L, int &ind1, int &ind2, int &ind3)
{
    int max1=0;  // 在直方图区间内特征点数量最大值
    int max2=0;  // 在直方图区间内特征点数量第二最大值
    int max3=0;  // 在直方图区间内特征点数量第三最大值

    for(int i=0; i<L; i++)
    {
      // 在该直方图区间内特征点数量
        const int s = histo[i].size();   
        if(s>max1)
        {
            max3=max2;
            max2=max1;
            max1=s;
            ind3=ind2;
            ind2=ind1;
            ind1=i;
        }
        else if(s>max2)
        {
            max3=max2;
            max2=s;
            ind3=ind2;
            ind2=i;
        }
        else if(s>max3)
        {
            max3=s;
            ind3=i;
        }
    }

    if(max2<0.1f*(float)max1)  // 如果max2/max1<0.1  那么证明第二第三方向不具有区分性,则将其索引置位初值
    {
        ind2=-1;
        ind3=-1;
    }
    else if(max3<0.1f*(float)max1)// 如果max3/max1<0.1  那么证明第三方向不具有区分性,则将其索引置位初值
    {
        ind3=-1;
    }
}


// Bit set count operation from
// http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
// 计算描述子的汉明距离  位运算 提高速度
int ORBmatcher::DescriptorDistance(const cv::Mat &a, const cv::Mat &b)
{
    const int *pa = a.ptr<int32_t>();
    const int *pb = b.ptr<int32_t>();

    int dist=0;

    for(int i=0; i<8; i++, pa++, pb++)
    {
        unsigned  int v = *pa ^ *pb;
        v = v - ((v >> 1) & 0x55555555);
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }

    return dist;
}

} //namespace ORB_SLAM
