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


#ifndef SIM3SOLVER_H
#define SIM3SOLVER_H

#include <opencv2/opencv.hpp>
#include <vector>

#include "KeyFrame.h"



namespace ORB_SLAM2
{

class Sim3Solver
{
public:

    Sim3Solver(KeyFrame* pKF1, KeyFrame* pKF2, const std::vector<MapPoint*> &vpMatched12, const bool bFixScale = true);

    void SetRansacParameters(double probability = 0.99, int minInliers = 6 , int maxIterations = 300);

    cv::Mat find(std::vector<bool> &vbInliers12, int &nInliers);
/**************************
 * 		在待检测回环关键帧和回环候选关键帧之间RANSAC迭代求解相似矩阵sim
 * 			随机选取两关键帧的三对匹配地图点相机坐标系下的坐标,根据三对匹配点求解sim矩阵  
 * 			参考:Horn 1987, Closed-form solution of absolute orientataion using unit quaternions
 * 			求解sim矩阵之后根据sim矩阵检测内点情况
 * 				根据内点数量的多少来判定当前模型是否准确,当内点数达到一定阈值时停止迭代,返回sim矩阵
 ********************************/
    cv::Mat iterate(int nIterations, bool &bNoMore, std::vector<bool> &vbInliers, int &nInliers);

    cv::Mat GetEstimatedRotation();
    cv::Mat GetEstimatedTranslation();
    float GetEstimatedScale();


protected:

    void ComputeCentroid(cv::Mat &P, cv::Mat &Pr, cv::Mat &C);
    //参考: Horn 1987, Closed-form solution of absolute orientataion using unit quaternions
    void ComputeSim3(cv::Mat &P1, cv::Mat &P2);
    // 根据三维点的重投影误差(经过相似变换矩阵的映射)判断是否内点
    void CheckInliers();
    // 根据变换矩阵Tcw和K   将世界坐标系下的地图点vP3Dw映射成像素坐标
    void Project(const std::vector<cv::Mat> &vP3Dw, std::vector<cv::Mat> &vP2D, cv::Mat Tcw, cv::Mat K);
    // 从相机坐标系下的坐标转化为像素坐标  vP3Dc:三维相机坐标系下的坐标  vP2D:二维像素坐标  K:相机内参数矩阵
    void FromCameraToImage(const std::vector<cv::Mat> &vP3Dc, std::vector<cv::Mat> &vP2D, cv::Mat K);


protected:

    // KeyFrames and matches
    // 关键帧1
    KeyFrame* mpKF1;
    // 关键帧2
    KeyFrame* mpKF2;

    //关键帧1的特征点在相机坐标系下的坐标
    std::vector<cv::Mat> mvX3Dc1;
    // 关键帧2的特征点在相机坐标系下的坐标
    std::vector<cv::Mat> mvX3Dc2;
    // 关键帧1中 的地图点
    std::vector<MapPoint*> mvpMapPoints1;
    // 关键帧2中的地图点
    std::vector<MapPoint*> mvpMapPoints2;
    // 关键帧1和关键帧2的匹配地图点
    std::vector<MapPoint*> mvpMatches12;

    std::vector<size_t> mvnIndices1;
    std::vector<size_t> mvSigmaSquare1;
    std::vector<size_t> mvSigmaSquare2;
    std::vector<size_t> mvnMaxError1;
    std::vector<size_t> mvnMaxError2;
    // 关键帧1中地图点的数量(响应数)
    int N;
    int mN1;

    // Current Estimation
    cv::Mat mR12i;
    cv::Mat mt12i;
    float ms12i;
    cv::Mat mT12i;
    cv::Mat mT21i;
    std::vector<bool> mvbInliersi;
    // 当前RANSAC模型下的内点数量
    int mnInliersi;

    // Current Ransac State
    // 当前RANSAC迭代次数
    int mnIterations;
    // 当前最优RANSAC模型下内点状态
    std::vector<bool> mvbBestInliers;
    // 当前最优的RANSAC模型下内点数
    int mnBestInliers;
    // 当前RANSAC下最佳的变换矩阵模型
    cv::Mat mBestT12;
    // 当前RANSAC下最佳的旋转矩阵模型
    cv::Mat mBestRotation;
    // 当前RANSAC下最佳的平移矩阵模型
    cv::Mat mBestTranslation;
    float mBestScale;

    // Scale is fixed to 1 in the stereo/RGBD case
    bool mbFixScale;

    // Indices for random selection
    std::vector<size_t> mvAllIndices;

    // Projections
    std::vector<cv::Mat> mvP1im1;
    std::vector<cv::Mat> mvP2im2;

    // RANSAC probability
    double mRansacProb;

    // RANSAC min inliers
    // 最小内点数量阈值  当小于这一阈值时证明当下的RANSAC模型不符合要求
    int mRansacMinInliers;

    // RANSAC max iterations  RANSAC最大迭代次数
    int mRansacMaxIts;

    // Threshold inlier/outlier. e = dist(Pi,T_ij*Pj)^2 < 5.991*mSigma2
    float mTh;
    float mSigma2;

    // Calibration
    // 关键帧1的相机内参
    cv::Mat mK1;
    // 关键帧2的相机内参
    cv::Mat mK2;

};

} //namespace ORB_SLAM

#endif // SIM3SOLVER_H
