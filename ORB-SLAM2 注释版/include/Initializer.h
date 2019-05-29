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
#ifndef INITIALIZER_H
#define INITIALIZER_H

#include<opencv2/opencv.hpp>
#include "Frame.h"


namespace ORB_SLAM2
{

// THIS IS THE INITIALIZER FOR MONOCULAR SLAM. NOT USED IN THE STEREO OR RGBD CASE.
class Initializer
{
    typedef pair<int,int> Match;

public:

    // Fix the reference frame   设置参考帧(初始化函数)  ReferenceFrame : 单目初始化的参考帧   sigma:  标准差                   iterations:ransac迭代次数
    Initializer(const Frame &ReferenceFrame, float sigma = 1.0, int iterations = 200);

    // Computes in parallel a fundamental matrix and a homography
    // Selects a model and tries to recover the motion and the structure from motion
    // 并行计算基础矩阵和单应性矩阵  选择一个模型来恢复两帧之间的相对位姿
/************************************************************************
 * 	单目相机追踪线程的初始化
 * 		CurrentFrame(in):当前帧
 * 		vMatches12(in):在当前帧中与参考初始化帧之间所有的匹配点对的索引
 * 		R21(out):旋转矩阵
 * 		t21(out):平移矩阵
 * 		vP3D(out):匹配点的索引
 * 		vbTriangulated(out):该内点是否能被三角化
 * 	步骤:RANSAC随机选择8点  根据8点分别计算基础矩阵和单应性矩阵  最终根据得分获得得分较高的模型确定当前初始化的相机位姿
 *************************************************************************/
    bool Initialize(const Frame &CurrentFrame, const vector<int> &vMatches12,
                    cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated);


private:
/**************************************************
 * 		计算单应性矩阵
 * 		参考 : Multiple view geometry in computer vision  P109 算法4.4
 * 		1. 将匹配点像素坐标进行归一化处理
 * 		2. 用RANSAC方法随机选择8个点对,然后根据对极几何的方式计算单应性矩阵,并通过几何误差来计算每次RANSAC的得分score
 * 		3. 选择得分最大的RANSAC结果为当前的单应性矩阵,并根据单应性矩阵的几何误差得到像素点是否为内点,存储到vbMatchesInliers
 ************************************************************/
    void FindHomography(vector<bool> &vbMatchesInliers, float &score, cv::Mat &H21);
/**************************************************
 * 	计算基础矩阵F
 * 	将坐标点进行归一化处理
 * 	RANSAC迭代计算基础矩阵F并得到计算得分score(根据双向变换误差计算)
 * 	选择score最大的一组基础矩阵F为最终结果
 ************************************************************/
    void FindFundamental(vector<bool> &vbInliers, float &score, cv::Mat &F21);
// |x'|     | h1 h2 h3 ||x|
// |y'| = a | h4 h5 h6 ||y|  简写: x' = a H x, a为一个尺度因子
// |1 |     | h7 h8 h9 ||1|
// 使用DLT(direct linear tranform)求解该模型
// x' = a H x 
// ---> (x') 叉乘 (H x)  = 0
// ---> Ah = 0
// A = | 0  0  0 -x -y -1 xy' yy' y'|  h = | h1 h2 h3 h4 h5 h6 h7 h8 h9 |
//     |-x -y -1  0  0  0 xx' yx' x'|
// 计算单应性矩阵从1->2的单应性矩阵  参考高翔slam14讲 P153
// SVD分解求解线性方程组  Ah=0  ,h是单应性矩阵对应的列向量   A是2n*12的矩阵
    cv::Mat ComputeH21(const vector<cv::Point2f> &vP1, const vector<cv::Point2f> &vP2);
 /* 
 * 	计算本质矩阵E
 * 	根据SVD分解,解决8点法的问题  Ax=0  最小奇异值对应的奇异向量为方程的解
 * 	调整得到的本质矩阵E,使其满足内在要求  详情请见高翔slam P151
 */
    cv::Mat ComputeF21(const vector<cv::Point2f> &vP1, const vector<cv::Point2f> &vP2);
/***********************************************************
 * 		检测单应性矩阵   通过计算对称传输误差来计算该单应性矩阵的得分以及区分匹配点中的内点和外点
 * 			     循环所有的特征点,计算对称传输误差  1->2的几何误差   2->1的几何误差    得分是所有特征点的(阈值th-几何误差)和
 * 		参考高翔slam14讲 P153
 *****************************************************************/
    float CheckHomography(const cv::Mat &H21, const cv::Mat &H12, vector<bool> &vbMatchesInliers, float sigma);
/***************
 *	检测基础矩阵得分,并计算模型下匹配点是否为内点
 * 		根据对极几何知识,计算点到极线的距离从而确定此基础矩阵的得分
 * 		参考: multiple view geometry in computer vision  第九章 P245
 * ***************/
    float CheckFundamental(const cv::Mat &F21, vector<bool> &vbMatchesInliers, float sigma);

    bool ReconstructF(vector<bool> &vbMatchesInliers, cv::Mat &F21, cv::Mat &K,
                      cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated, float minParallax, int minTriangulated);
/***************
 * 	根据单应矩阵H重构相机位姿R和t
 * 	vbMatchesInliers(in): 当前匹配点是否为内点
 * 	H21(in):待分解的单应矩阵
 * 	K(out):相机内参数矩阵
 * 	R21,t21(out):分解后的旋转平移矩阵	
 * 	vP3D(out):匹配到的3D点坐标
 * 	vbTriangulated(out):是否满足三角化(视差是否平行)
 * 	minParallax(in):允许的最小视差
 * 	minTriangulated(in):经过筛选最小内点数量的阈值
 * 	步骤:  	将基础矩阵转换为本质矩阵
 * 			对本质矩阵利用SVD进行分解得到四组旋转平移矩阵
 * 			用三角法计算深度信息 从而筛选四组旋转平移矩阵
 * 	 H矩阵分解常见有两种方法：Faugeras SVD-based decomposition 和 Zhang SVD-based decomposition
 * 	参考文献：Faugeras et al, Motion and structure from motion in a piecewise planar environment. International Journal of Pattern Recognition and Artificial Intelligence, 1988.
 * 	这篇参考文献和下面的代码使用了Faugeras SVD-based decomposition算法
 * 
 *********/ 
    bool ReconstructH(vector<bool> &vbMatchesInliers, cv::Mat &H21, cv::Mat &K,
                      cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated, float minParallax, int minTriangulated);
// Trianularization: 已知匹配特征点对{x x'} 和 各自相机矩阵{P P'}, 估计三维点 X
// x' = P'X  x = PX
// 它们都属于 x = aPX模型
//                         |X|
// |x|     |p1 p2  p3  p4 ||Y|     |x|    |--p0--||.|
// |y| = a |p5 p6  p7  p8 ||Z| ===>|y| = a|--p1--||X|
// |z|     |p9 p10 p11 p12||1|     |z|    |--p2--||.|
// 采用DLT的方法：x叉乘PX = 0
// |yp2 -  p1|     |0|
// |p0 -  xp2| X = |0|
// |xp1 - yp0|     |0|
// 两个点:
// |yp2   -  p1  |     |0|
// |p0    -  xp2 | X = |0| ===> AX = 0
// |y'p2' -  p1' |     |0|
// |p0'   - x'p2'|     |0|
// 变成程序中的形式：
// |xp2  - p0 |     |0|
// |yp2  - p1 | X = |0| ===> AX = 0
// |x'p2'- p0'|     |0|
// |y'p2'- p1'|     |0|
/***
 * 三角测量计算匹配点的深度(匹配点三维坐标)
 * 	kp1:相机1的关键点
 * 	kp2:相机2的关键点
 * 	P1:相机1下的点像素坐标->世界坐标系下的坐标
 * 	P2:相机2下的点像素坐标->世界坐标系下的坐标
 * 	x3D:得到3D点
 **/
    void Triangulate(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &P1, const cv::Mat &P2, cv::Mat &x3D);
/**
 * ＠brief 归一化特征点到同一尺度（作为normalize DLT的输入）
 *
 * [x' y' 1]' = T * [x y 1]' \n
 * 归一化后x', y'的均值为0，sum(abs(x_i'-0))=1，sum(abs((y_i'-0))=1
 * 参考Multiple view geometry in computer vision  P109 算法4.4
 * @param vKeys             特征点在图像上的坐标
 * @param vNormalizedPoints 特征点归一化后的坐标
 * @param T                 将特征点归一化的矩阵
 * 归一化原因: 	在相似变换之后(点在不同的坐标系下),他们的单应性矩阵是不相同的
 * 				这就有个问题了,如果图像存在噪声,使得点的坐标发生了变化,那么它的单应性矩阵则会发生变化,
 * 				我们采取的方法是将点的坐标放到同一坐标系下,并将缩放尺度也进行统一 对同一幅图像的坐标进行相同的变换,不同图像进行不同变换
 * 				缩放尺度使得噪声对于图像的影响在一个数量级上
 */
    void Normalize(const vector<cv::KeyPoint> &vKeys, vector<cv::Point2f> &vNormalizedPoints, cv::Mat &T);
/*************************************************************************
 * 		检测得到的旋转和平移矩阵是否符合要求
 * 		R(in):旋转矩阵
 * 		t(in):平移矩阵
 * 		vKeys1(in):图像1中关键点的像素坐标
 * 		vKeys2(in):图像2中关键点的像素坐标
 * 		vMatches12(in):first 匹配点像素1坐标的索引   second 匹配点像素2坐标的索引  
 * 		vbMatchesInliers(in):是否是匹配内点 
 * 		K(in):相机内参数矩阵
 * 		vP3D(out):匹配点的三维坐标
 * 		th2(in):重投影误差阈值
 * 		vbGood(out):两相机的视差是否非平行
 * 		parallax(out):较大的视差角角度
 * 	返回经过筛选的匹配点的数量
 *    遴选标准:
 * 		1 . 是匹配内点
 * 		2.  三维点坐标都不是无穷值
 * 		3.  3D点在两个相机坐标系下的深度坐标都为正
 * 		4.  计算重投影误差,误差小于阈值
 * 		5.  3d点在两幅图像中的视角不是平行的
 ********************************************************/
    int CheckRT(const cv::Mat &R, const cv::Mat &t, const vector<cv::KeyPoint> &vKeys1, const vector<cv::KeyPoint> &vKeys2,
                       const vector<Match> &vMatches12, vector<bool> &vbInliers,
                       const cv::Mat &K, vector<cv::Point3f> &vP3D, float th2, vector<bool> &vbGood, float &parallax);
/**
 * @brief 从F恢复R t
 * 
 * 度量重构
 * 1. 由Fundamental矩阵结合相机内参K，得到Essential矩阵: \f$ E = k'^T F k \f$
 * 2. SVD分解得到R t
 * 3. 进行cheirality check, 从四个解中找出最合适的解
 * 
 * @see Multiple View Geometry in Computer Vision - Result 9.19 p259
 */
    void DecomposeE(const cv::Mat &E, cv::Mat &R1, cv::Mat &R2, cv::Mat &t);


    // Keypoints from Reference Frame (Frame 1)  参考帧的关键点
    vector<cv::KeyPoint> mvKeys1;

    // Keypoints from Current Frame (Frame 2)  当前帧的关键点
    vector<cv::KeyPoint> mvKeys2;

    // Current Matches from Reference to Current  从参考帧到当前帧的匹配点
    vector<Match> mvMatches12;
    // 是否为匹配点的标志
    vector<bool> mvbMatched1;

    // Calibration  相机相关参数
    cv::Mat mK;

    // Standard Deviation and Variance  标准差和方差   traking给的sigma=1
    float mSigma, mSigma2;

    // Ransac max iterations  ransac最大迭代次数
    int mMaxIterations;

    // Ransac sets  RANSAC的点索引集 第一层为RANSAC的迭代次数  第二层为当前迭代次数选择的8点中第几个点  值为点的索引
    vector<vector<size_t> > mvSets;   

};

} //namespace ORB_SLAM

#endif // INITIALIZER_H
