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


#ifndef ORBMATCHER_H
#define ORBMATCHER_H

#include<vector>
#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include"MapPoint.h"
#include"KeyFrame.h"
#include"Frame.h"


namespace ORB_SLAM2
{

class ORBmatcher
{    
public:

    ORBmatcher(float nnratio=0.6, bool checkOri=true);

    // Computes the Hamming distance between two ORB descriptors
    // 计算两ORB描述子的汉明距离
    static int DescriptorDistance(const cv::Mat &a, const cv::Mat &b);

    // Search matches between Frame keypoints and projected MapPoints. Returns number of matches
    // Used to track the local map (Tracking)
    // 在帧的关键点和映射地图点之间寻找匹配，返回匹配点的数量，使用在局部地图的追踪线程中
 /*********************************************************
 通过投影的方式进行匹配     将地图点投影到对应帧的相应区域中,在对应区域中搜索匹配
 在帧F中搜索vpMapPoints地图点的匹配点
   th:  决定投影后搜索半径大大小
   筛选方法:
   在投影区域内遍历所有的特征点,计算描述子距离第一小和描述子距离第二小  当第一小明显小于第二小时证明该特征点具有最优性
   从而确定匹配
 **********************************************************/  
    int SearchByProjection(Frame &F, const std::vector<MapPoint*> &vpMapPoints, const float th=3);

    // Project MapPoints tracked in last frame into the current frame and search matches.
    // Used to track from previous frame (Tracking)
    //   从上一帧映射地图点与当前帧进行匹配，使用追踪线程的追踪上一帧的位姿
    //   当前帧和上一帧之间进行匹配   
    //   根据上一帧到当前帧的位姿,将上一帧的地图点投影到当前帧,然后将地图点反投影到像素坐标,在像素坐标一定范围内寻找最佳匹配点
    //   注意这里用到的当前帧的位姿是根据上一帧的位姿和上一帧的位姿变化速度来推算的相机位姿
    int SearchByProjection(Frame &CurrentFrame, const Frame &LastFrame, const float th, const bool bMono);

    // Project MapPoints seen in KeyFrame into the Frame and search matches.
    // Used in relocalisation (Tracking)
     // 映射在关键帧中观察到的地图点与当前帧进行匹配，使用在重定位中   在通过BOW匹配当前帧和重定位候选关键帧内点数量不足时,通过此方式增加匹配点
    int SearchByProjection(Frame &CurrentFrame, KeyFrame* pKF, const std::set<MapPoint*> &sAlreadyFound, const float th, const int ORBdist);

    // Project MapPoints using a Similarity Transformation and search matches.
    // Used in loop detection (Loop Closing)
    // 使用一个相似的矩阵映射地图点并查找匹配点，使用在回环检测中
/**********************************************************
 * 		通过相似矩阵Scw映射计算匹配点   将地图点向关键帧进行投影
 * 		pKF(in): 待检测关键帧
 * 		Scw(in): 当前关键帧的相似矩阵位姿
 * 		vpPoints(in):待匹配的地图点
 * 		vpMatched(out):得到的匹配点
 * 		th:阈值
 * 	主要思路:将地图点根据相似矩阵映射到当前关键帧中,在映射区域中查找地图点的匹配
 ************************************************/
     int SearchByProjection(KeyFrame* pKF, cv::Mat Scw, const std::vector<MapPoint*> &vpPoints, std::vector<MapPoint*> &vpMatched, int th);

    // Search matches between MapPoints in a KeyFrame and ORB in a Frame.    
    // Brute force constrained to ORB that belong to the same vocabulary node (at a certain level)
    // Used in Relocalisation and Loop Detection    
    // 关键帧的地图点和当前帧的ORB特征点之间进行匹配，使用在重定位和回环检测中  将匹配结果存储到vpMapPointMatches
     //  关键帧和当前帧通过词袋进行快速匹配   在同一词典节点上搜索匹配    
     //  筛选方法:
     //          在投影区域内遍历所有的特征点,计算描述子距离第一小和描述子距离第二小  当第一小明显小于第二小时证明该特征点具有最优性
     //           从而确定匹配
    int SearchByBoW(KeyFrame *pKF, Frame &F, std::vector<MapPoint*> &vpMapPointMatches);
    /*************************************************
     *     根据词袋模型来匹配两关键帧
	     pKF1 : 关键帧1
	     pKF2:关键帧2
	     vpMatches12:匹配地图点
      *********************************************************/
    int SearchByBoW(KeyFrame *pKF1, KeyFrame* pKF2, std::vector<MapPoint*> &vpMatches12);

    // Matching for the Map Initialization (only used in the monocular case)
/*********************************************************
 * 		为地图初始化寻找匹配点
 * 		F1:待匹配帧1
 * 		F2:待匹配帧2
 * 		vbPrevMatched:  F1的特征点位置坐标
 * 		vnMatches12:匹配帧2在匹配帧1的匹配索引,下标为匹配帧1的关键点索引,值为匹配帧2的关键点索引
 * 		windowSize:在F2帧中搜索特征点的窗口大小
 * 		步骤:
 * 			1. 构建搜索区域  假定匹配帧1和匹配帧2之间没有太大的位移,所以在匹配帧2的帧1特征点位置坐标处寻找对应的匹配点
 * 			2. 通过计算描述子距离来寻找最优匹配距离以及最优匹配距离对应帧2的特征点索引
 * 			3. 如果需要检查旋转,则构建旋转角度直方图来筛选匹配点
 **********************************************************************/
    int SearchForInitialization(Frame &F1, Frame &F2, std::vector<cv::Point2f> &vbPrevMatched, std::vector<int> &vnMatches12, int windowSize=10);

    // Matching to triangulate new MapPoints. Check Epipolar Constraint. 
// 通过对极几何来计算匹配点   计算的是在追踪线程下没有计算三维点的特征点(没追踪到)
// 找出 pKF1中 特征点在pKF2中的匹配点  
// 根据BOW向量匹配在同一节点下的特征点  根据匹配点描述子距离最小的点并且满足对极几何的约束
    int SearchForTriangulation(KeyFrame *pKF1, KeyFrame* pKF2, cv::Mat F12,
                               std::vector<pair<size_t, size_t> > &vMatchedPairs, const bool bOnlyStereo);

    // Search matches between MapPoints seen in KF1 and KF2 transforming by a Sim3 [s12*R12|t12]
    // In the stereo and RGB-D case, s12=1
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
    int SearchBySim3(KeyFrame* pKF1, KeyFrame* pKF2, std::vector<MapPoint *> &vpMatches12, const float &s12, const cv::Mat &R12, const cv::Mat &t12, const float th);

    // Project MapPoints into KeyFrame and search for duplicated MapPoints.
/**
 * @brief 将MapPoints投影到关键帧pKF中，并判断是否有重复的MapPoints
 * 1.如果MapPoint能匹配关键帧的特征点，并且该点有对应的MapPoint，那么将两个MapPoint合并（选择观测数多的）
 * 2.如果MapPoint能匹配关键帧的特征点，并且该点没有对应的MapPoint，那么为该点添加MapPoint
 * @param  pKF         相邻关键帧
 * @param  vpMapPoints 当前关键帧的MapPoints
 * @param  th          搜索半径的因子
 * @return             重复MapPoints的数量
 */
    int Fuse(KeyFrame* pKF, const vector<MapPoint *> &vpMapPoints, const float th=3.0);

    // Project MapPoints into KeyFrame using a given Sim3 and search for duplicated MapPoints.
/*****************************************
 * 根据相似矩阵Scw映射地图点到关键帧  匹配地图点为:vpReplacePoint   用于回环检测
 * 		pKF(in):映射关键帧
 * 		Scw(in):关键帧的相似矩阵位姿
 * 		vpPoints(in):待映射地图点
 * 		th(in):阈值
 * 		vpReplacePoint(out):匹配到的地图点
 * 	主要思路: 将回环地图点根据当前关键帧的位姿(相似矩阵)映射到当前关键帧,在当前关键帧中寻找回环地图点的代替点,存储进vpReplacePoint
 ************************************/
    int Fuse(KeyFrame* pKF, cv::Mat Scw, const std::vector<MapPoint*> &vpPoints, float th, vector<MapPoint *> &vpReplacePoint);

public:
    //两描述子的最优距离  最小阈值  小于该阈值证明可以认为是一次匹配
    static const int TH_LOW;
    static const int TH_HIGH;
    static const int HISTO_LENGTH;


protected:
// 该点是否满足对极几何 (特征点到极线的距离)
// 参考multiple view geometry in computer vision 9.1 P240
    bool CheckDistEpipolarLine(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &F12, const KeyFrame *pKF);

    float RadiusByViewingCos(const float &viewCos);
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
    void ComputeThreeMaxima(std::vector<int>* histo, const int L, int &ind1, int &ind2, int &ind3);
    // 第一最优距离/第二最优距离  阈值  此阈值判别该特征点只与第一距离的描述子距离近,而与其他描述子距离远  
    float mfNNratio;
    // 匹配过程中是否需要判别旋转的标志量
    bool mbCheckOrientation;
};

}// namespace ORB_SLAM

#endif // ORBMATCHER_H
