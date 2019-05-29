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

#include "Initializer.h"

#include "Thirdparty/DBoW2/DUtils/Random.h"

#include "Optimizer.h"
#include "ORBmatcher.h"

#include<thread>

namespace ORB_SLAM2
{

Initializer::Initializer(const Frame &ReferenceFrame, float sigma, int iterations)
{
    mK = ReferenceFrame.mK.clone();

    mvKeys1 = ReferenceFrame.mvKeysUn;

    mSigma = sigma;
    mSigma2 = sigma*sigma;
    mMaxIterations = iterations;
}
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
bool Initializer::Initialize(const Frame &CurrentFrame, const vector<int> &vMatches12, cv::Mat &R21, cv::Mat &t21,
                             vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated)
{
    // Fill structures with current keypoints and matches with reference frame  用当前帧的关键点和与参考帧的匹配填充成员变量mvKeys2和mvMatches12
    // Reference Frame: 1, Current Frame: 2
    mvKeys2 = CurrentFrame.mvKeysUn;

    mvMatches12.clear();
    mvMatches12.reserve(mvKeys2.size());
    mvbMatched1.resize(mvKeys1.size());
    for(size_t i=0, iend=vMatches12.size();i<iend; i++)
    {
        if(vMatches12[i]>=0)
        {
            mvMatches12.push_back(make_pair(i,vMatches12[i]));
            mvbMatched1[i]=true;  //证明当前索引下是匹配点
        }
        else
            mvbMatched1[i]=false;
    }

    const int N = mvMatches12.size();

    // Indices for minimum set selection	匹配点的所有索引
    vector<size_t> vAllIndices;
    vAllIndices.reserve(N);   
    // 当前RANSAC下能得到的匹配点索引
    vector<size_t> vAvailableIndices;

    for(int i=0; i<N; i++)
    {
        vAllIndices.push_back(i);
    }

    // Generate sets of 8 points for each RANSAC iteration   每一次RANSAC迭代都产生8个点
    mvSets = vector< vector<size_t> >(mMaxIterations,vector<size_t>(8,0));
    
    DUtils::Random::SeedRandOnce(0);
    
    for(int it=0; it<mMaxIterations; it++)
    {
        vAvailableIndices = vAllIndices;

        // Select a minimum set  随机选择8个点
        for(size_t j=0; j<8; j++)
        {
            int randi = DUtils::Random::RandomInt(0,vAvailableIndices.size()-1);
            int idx = vAvailableIndices[randi];
	    //RANSAC的点索引集 第一层为RANSAC的迭代次数  第二层为第几个点  值为点的索引
            mvSets[it][j] = idx;

            vAvailableIndices[randi] = vAvailableIndices.back();
            vAvailableIndices.pop_back();
        }
    }

    // Launch threads to compute in parallel a fundamental matrix and a homography   开辟两个线程计算单应性矩阵和基础矩阵
    vector<bool> vbMatchesInliersH, vbMatchesInliersF;
    float SH, SF;
    cv::Mat H, F;

    thread threadH(&Initializer::FindHomography,this,ref(vbMatchesInliersH), ref(SH), ref(H));
    thread threadF(&Initializer::FindFundamental,this,ref(vbMatchesInliersF), ref(SF), ref(F));

    // Wait until both threads have finished
    threadH.join();
    threadF.join();

    // Compute ratio of scores
    float RH = SH/(SH+SF);

    // Try to reconstruct from homography or fundamental depending on the ratio (0.40-0.45)
    if(RH>0.40)// RH大于0.4证明单应性矩阵更符合当前的模型
        return ReconstructH(vbMatchesInliersH,H,mK,R21,t21,vP3D,vbTriangulated,1.0,50);
    else //if(pF_HF>0.6)  否则说明F矩阵更符合当前的模型
        return ReconstructF(vbMatchesInliersF,F,mK,R21,t21,vP3D,vbTriangulated,1.0,50);

    return false;
}

/**************************************************
 * 		计算单应性矩阵
 * 		vbMatchesInliers(out): 当前匹配点索引下的匹配点是否为内点
 * 		score(out): 当前模型的评分
 * 		H21(out):计算得到的单应性矩阵
 * 		参考 : Multiple view geometry in computer vision  P109 算法4.4
 * 		1. 将匹配点像素坐标进行归一化处理
 * 		2. 用RANSAC方法随机选择8个点对,然后根据8点法计算单应性矩阵,并通过几何误差来计算每次RANSAC的得分score
 * 		3. 选择得分最大的RANSAC结果为当前的单应性矩阵,并根据单应性矩阵的几何误差得到像素点是否为内点,存储到vbMatchesInliers
 ************************************************************/
void Initializer::FindHomography(vector<bool> &vbMatchesInliers, float &score, cv::Mat &H21)
{
    // Number of putative matches
    const int N = mvMatches12.size();

    // Normalize coordinates  计算归一化坐标下匹配点的坐标 
    // 帧1 帧2 的归一化坐标
    vector<cv::Point2f> vPn1, vPn2;
    // 归一化坐标需要的归一化矩阵
    cv::Mat T1, T2;
    // vPn1 = T1 * mvKeys1
    Normalize(mvKeys1,vPn1, T1);
    // vPn2 = T2 * mvKeys2
    Normalize(mvKeys2,vPn2, T2);
    cv::Mat T2inv = T2.inv();

    // Best Results variables
    score = 0.0;
    vbMatchesInliers = vector<bool>(N,false);

    // Iteration variables
    // 当前迭代下选择的8个点对应的帧1归一化坐标
    vector<cv::Point2f> vPn1i(8);
    // 当前迭代下选择的8个点对应的帧2归一化坐标
    vector<cv::Point2f> vPn2i(8);
    cv::Mat H21i, H12i;
    vector<bool> vbCurrentInliers(N,false);
    float currentScore;

    // Perform all RANSAC iterations and save the solution with highest score  运行ransac迭代并保存最高分下的解决方案
    for(int it=0; it<mMaxIterations; it++)
    {
        // Select a minimum set  选择8个归一化之后的点对进行迭代
        for(size_t j=0; j<8; j++)
        {
            int idx = mvSets[it][j];
	    
            vPn1i[j] = vPn1[mvMatches12[idx].first];
            vPn2i[j] = vPn2[mvMatches12[idx].second];
        }
	//根据匹配点的归一化平面坐标求解出单应性矩阵H  归一化坐标系下1->2的单应性矩阵   vPn2i = Hn * vPn1i
        cv::Mat Hn = ComputeH21(vPn1i,vPn2i);
	// X2=H21*X1    结合vPn1 = T1 * mvKeys1, vPn2 = T2 * mvKeys2  得到:T2 * mvKeys2 =  Hn * T1 * mvKeys1   进一步得到:mvKeys2  = T2.inv * Hn * T1 * mvKeys1
        H21i = T2inv*Hn*T1;
	// X1=H12*X2
        H12i = H21i.inv();
	// 计算当前单应性矩阵的得分  ,并确定匹配内点
        currentScore = CheckHomography(H21i, H12i, vbCurrentInliers, mSigma);
	
        if(currentScore>score)  // 找到所有RANSAC迭代中最大得分
        {
            H21 = H21i.clone();
            vbMatchesInliers = vbCurrentInliers;
            score = currentScore;
        }
    }
}

/**************************************************
 * 	计算基础矩阵F
 * 	将坐标点进行归一化处理
 * 	RANSAC迭代计算基础矩阵F(对极几何)并得到计算得分score(根据双向变换误差计算)
 * 	选择score最大的一组基础矩阵F为最终结果
 ************************************************************/
void Initializer::FindFundamental(vector<bool> &vbMatchesInliers, float &score, cv::Mat &F21)
{
    // Number of putative matches
    const int N = vbMatchesInliers.size();

    // Normalize coordinates 归一化坐标
    vector<cv::Point2f> vPn1, vPn2;
    cv::Mat T1, T2;
    Normalize(mvKeys1,vPn1, T1);
    Normalize(mvKeys2,vPn2, T2);
    cv::Mat T2t = T2.t();

    // Best Results variables
    score = 0.0;
    vbMatchesInliers = vector<bool>(N,false);

    // Iteration variables
    vector<cv::Point2f> vPn1i(8);
    vector<cv::Point2f> vPn2i(8);
    cv::Mat F21i;
    vector<bool> vbCurrentInliers(N,false);
    float currentScore;

    // Perform all RANSAC iterations and save the solution with highest score  RANSAC迭代算法求解基础矩阵F
    for(int it=0; it<mMaxIterations; it++)
    {
        // Select a minimum set
        for(int j=0; j<8; j++)
        {
            int idx = mvSets[it][j];

            vPn1i[j] = vPn1[mvMatches12[idx].first];
            vPn2i[j] = vPn2[mvMatches12[idx].second];
        }
	// 计算本质矩阵E
        cv::Mat Fn = ComputeF21(vPn1i,vPn2i);
	//恢复基础矩阵F
        F21i = T2t*Fn*T1;
	// 检测基础矩阵得分
        currentScore = CheckFundamental(F21i, vbCurrentInliers, mSigma);

        if(currentScore>score)
        {
            F21 = F21i.clone();
            vbMatchesInliers = vbCurrentInliers;
            score = currentScore;
        }
    }
}
// |x'|        | h1 h2 h3 ||x|
// |y'| = a | h4 h5 h6 ||y|  简写: x' = a H x, a为一个尺度因子
// |1 |        | h7 h8 h9 ||1|
// 使用DLT(direct linear tranform)求解该模型
// x' = a H x   
// ---> (x') 叉乘 (H x)  = 0
// ---> Ah = 0
// A = | 0  0  0 -x -y -1 xy' yy' y'|         h = | h1 h2 h3 h4 h5 h6 h7 h8 h9 |
//        |-x -y -1  0  0  0 xx' yx' x'|
// 计算单应性矩阵从1->2的单应性矩阵  参考高翔slam14讲 P153
// SVD分解求解线性方程组  Ah=0  ,h是单应性矩阵对应的列向量   A是2n*12的矩阵
cv::Mat Initializer::ComputeH21(const vector<cv::Point2f> &vP1, const vector<cv::Point2f> &vP2)
{
    const int N = vP1.size();

    cv::Mat A(2*N,9,CV_32F);

    for(int i=0; i<N; i++)
    {
        const float u1 = vP1[i].x;
        const float v1 = vP1[i].y;
        const float u2 = vP2[i].x;
        const float v2 = vP2[i].y;

        A.at<float>(2*i,0) = 0.0;
        A.at<float>(2*i,1) = 0.0;
        A.at<float>(2*i,2) = 0.0;
        A.at<float>(2*i,3) = -u1;
        A.at<float>(2*i,4) = -v1;
        A.at<float>(2*i,5) = -1;
        A.at<float>(2*i,6) = v2*u1;
        A.at<float>(2*i,7) = v2*v1;
        A.at<float>(2*i,8) = v2;

        A.at<float>(2*i+1,0) = u1;
        A.at<float>(2*i+1,1) = v1;
        A.at<float>(2*i+1,2) = 1;
        A.at<float>(2*i+1,3) = 0.0;
        A.at<float>(2*i+1,4) = 0.0;
        A.at<float>(2*i+1,5) = 0.0;
        A.at<float>(2*i+1,6) = -u2*u1;
        A.at<float>(2*i+1,7) = -u2*v1;
        A.at<float>(2*i+1,8) = -u2;

    }

    cv::Mat u,w,vt;
    //参考http://blog.csdn.net/youngpan1101/article/details/54574130奇异值分解求解线性方程组  ,方程的解对应了最小奇异值对应的列向量
    cv::SVDecomp(A,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    return vt.row(8).reshape(0, 3);  // 将向量形式变为3*3的矩阵
}
/* 
 * 	计算本质矩阵E
 * 	根据SVD分解,解决8点法的问题  Ax=0  最小奇异值对应的奇异向量为方程的解
 * 	调整得到的本质矩阵E,使其满足内在要求  详情请见高翔slam P151
 */
cv::Mat Initializer::ComputeF21(const vector<cv::Point2f> &vP1,const vector<cv::Point2f> &vP2)
{
    const int N = vP1.size();

    cv::Mat A(N,9,CV_32F);

    for(int i=0; i<N; i++)
    {
        const float u1 = vP1[i].x;
        const float v1 = vP1[i].y;
        const float u2 = vP2[i].x;
        const float v2 = vP2[i].y;

        A.at<float>(i,0) = u2*u1;
        A.at<float>(i,1) = u2*v1;
        A.at<float>(i,2) = u2;
        A.at<float>(i,3) = v2*u1;
        A.at<float>(i,4) = v2*v1;
        A.at<float>(i,5) = v2;
        A.at<float>(i,6) = u1;
        A.at<float>(i,7) = v1;
        A.at<float>(i,8) = 1;
    }

    cv::Mat u,w,vt;

    cv::SVDecomp(A,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
    //F矩阵为A最小奇异值对应的奇异向量
    cv::Mat Fpre = vt.row(8).reshape(0, 3);

    cv::SVDecomp(Fpre,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
    // 将本质矩阵的奇异值最小值硬性设为0,使得本质矩阵满足内在要求  详情请见高翔slam P151
    w.at<float>(2)=0;

    return  u*cv::Mat::diag(w)*vt;
}

/***********************************************************
 * 		检测单应性矩阵   通过计算对称传输误差来计算该单应性矩阵的得分以及区分匹配点中的内点和外点
 * 			     循环所有的特征点,计算对称传输误差  1->2的几何误差   2->1的几何误差    得分是所有特征点的(阈值th-几何误差)和
 * 		参考高翔slam14讲 P153
 *****************************************************************/
float Initializer::CheckHomography(const cv::Mat &H21, const cv::Mat &H12, vector<bool> &vbMatchesInliers, float sigma)
{   
    const int N = mvMatches12.size();

    const float h11 = H21.at<float>(0,0);
    const float h12 = H21.at<float>(0,1);
    const float h13 = H21.at<float>(0,2);
    const float h21 = H21.at<float>(1,0);
    const float h22 = H21.at<float>(1,1);
    const float h23 = H21.at<float>(1,2);
    const float h31 = H21.at<float>(2,0);
    const float h32 = H21.at<float>(2,1);
    const float h33 = H21.at<float>(2,2);

    const float h11inv = H12.at<float>(0,0);
    const float h12inv = H12.at<float>(0,1);
    const float h13inv = H12.at<float>(0,2);
    const float h21inv = H12.at<float>(1,0);
    const float h22inv = H12.at<float>(1,1);
    const float h23inv = H12.at<float>(1,2);
    const float h31inv = H12.at<float>(2,0);
    const float h32inv = H12.at<float>(2,1);
    const float h33inv = H12.at<float>(2,2);

    vbMatchesInliers.resize(N);

    float score = 0;

    const float th = 5.991;

    const float invSigmaSquare = 1.0/(sigma*sigma);
     // 循环所有的特征点,计算对称传输误差 1->2的几何误差   2->1的几何误差  
    //  得分是所有特征点的(阈值th-几何误差)和
    for(int i=0; i<N; i++)
    {
        bool bIn = true;

        const cv::KeyPoint &kp1 = mvKeys1[mvMatches12[i].first];
        const cv::KeyPoint &kp2 = mvKeys2[mvMatches12[i].second];

        const float u1 = kp1.pt.x;
        const float v1 = kp1.pt.y;
        const float u2 = kp2.pt.x;
        const float v2 = kp2.pt.y;

        // Reprojection error in first image  计算第一幅图像的几何误差
        // x2in1 = H12*x2

        const float w2in1inv = 1.0/(h31inv*u2+h32inv*v2+h33inv);
        const float u2in1 = (h11inv*u2+h12inv*v2+h13inv)*w2in1inv;
        const float v2in1 = (h21inv*u2+h22inv*v2+h23inv)*w2in1inv;

        const float squareDist1 = (u1-u2in1)*(u1-u2in1)+(v1-v2in1)*(v1-v2in1);

        const float chiSquare1 = squareDist1*invSigmaSquare;

        if(chiSquare1>th)
            bIn = false;
        else
            score += th - chiSquare1;

        // Reprojection error in second image  计算第二幅图像的几何误差
        // x1in2 = H21*x1

        const float w1in2inv = 1.0/(h31*u1+h32*v1+h33);
        const float u1in2 = (h11*u1+h12*v1+h13)*w1in2inv;
        const float v1in2 = (h21*u1+h22*v1+h23)*w1in2inv;

        const float squareDist2 = (u2-u1in2)*(u2-u1in2)+(v2-v1in2)*(v2-v1in2);

        const float chiSquare2 = squareDist2*invSigmaSquare;

        if(chiSquare2>th)
            bIn = false;
        else
            score += th - chiSquare2;

        if(bIn)  // 如果重投影误差满足阈值 则是内点  否则为外点
            vbMatchesInliers[i]=true;
        else
            vbMatchesInliers[i]=false;
    }

    return score;
}
/***************
 *	检测基础矩阵得分,并计算模型下匹配点是否为内点
 * 		根据对极几何知识,计算点到极线的距离从而确定此基础矩阵的得分
 * 		参考: multiple view geometry in computer vision  第九章 P245
 * ***************/
float Initializer::CheckFundamental(const cv::Mat &F21, vector<bool> &vbMatchesInliers, float sigma)
{
    const int N = mvMatches12.size();

    const float f11 = F21.at<float>(0,0);
    const float f12 = F21.at<float>(0,1);
    const float f13 = F21.at<float>(0,2);
    const float f21 = F21.at<float>(1,0);
    const float f22 = F21.at<float>(1,1);
    const float f23 = F21.at<float>(1,2);
    const float f31 = F21.at<float>(2,0);
    const float f32 = F21.at<float>(2,1);
    const float f33 = F21.at<float>(2,2);

    vbMatchesInliers.resize(N);

    float score = 0;

    const float th = 3.841;
    const float thScore = 5.991;

    const float invSigmaSquare = 1.0/(sigma*sigma);

    for(int i=0; i<N; i++)
    {
        bool bIn = true;

        const cv::KeyPoint &kp1 = mvKeys1[mvMatches12[i].first];
        const cv::KeyPoint &kp2 = mvKeys2[mvMatches12[i].second];

        const float u1 = kp1.pt.x;
        const float v1 = kp1.pt.y;
        const float u2 = kp2.pt.x;
        const float v2 = kp2.pt.y;

        // Reprojection error in second image
        // l2=F21x1=(a2,b2,c2)    点到极线的距离

        const float a2 = f11*u1+f12*v1+f13;
        const float b2 = f21*u1+f22*v1+f23;
        const float c2 = f31*u1+f32*v1+f33;

        const float num2 = a2*u2+b2*v2+c2;

        const float squareDist1 = num2*num2/(a2*a2+b2*b2);

        const float chiSquare1 = squareDist1*invSigmaSquare;

        if(chiSquare1>th)
            bIn = false;
        else
            score += thScore - chiSquare1;

        // Reprojection error in second image  基础矩阵的性质:(P,P')的基础矩阵是(P',P)基础矩阵的转置  
	// 因此存在l1 =F21.t*x2=(a1,b1,c1)
        // l1 =x2tF21=(a1,b1,c1)

        const float a1 = f11*u2+f21*v2+f31;
        const float b1 = f12*u2+f22*v2+f32;
        const float c1 = f13*u2+f23*v2+f33;

        const float num1 = a1*u1+b1*v1+c1;

        const float squareDist2 = num1*num1/(a1*a1+b1*b1);

        const float chiSquare2 = squareDist2*invSigmaSquare;

        if(chiSquare2>th)
            bIn = false;
        else
            score += thScore - chiSquare2;

        if(bIn)
            vbMatchesInliers[i]=true;
        else
            vbMatchesInliers[i]=false;
    }

    return score;
}
/***************
 * 	根据基础矩阵F重构相机位姿R和t
 * 	vbMatchesInliers(in): 当前点是否为内点
 * 	F21(in):待分解的基础矩阵
 * 	K(in):相机内参数矩阵
 * 	R21,t21(out):分解后的旋转平移矩阵	
 * 	vP3D(out):匹配到的3D点坐标
 * 	vbTriangulated(out):是否满足三角化(视差是否平行)
 * 	minParallax(in):允许的最小视差
 * 	minTriangulated(in):经过筛选最小内点数量的阈值
 * 	步骤:  	将基础矩阵转换为本质矩阵
 * 			对本质矩阵利用SVD进行分解得到四组旋转平移矩阵
 * 			用三角法计算深度信息 从而筛选四组旋转平移矩阵
 *********/ 

bool Initializer::ReconstructF(vector<bool> &vbMatchesInliers, cv::Mat &F21, cv::Mat &K,
                            cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated, float minParallax, int minTriangulated)
{
    int N=0;
    for(size_t i=0, iend = vbMatchesInliers.size() ; i<iend; i++)
        if(vbMatchesInliers[i])
            N++;

    // Compute Essential Matrix from Fundamental Matrix   根据基础矩阵F计算本质矩阵E
    cv::Mat E21 = K.t()*F21*K;

    cv::Mat R1, R2, t;

    // Recover the 4 motion hypotheses   分解E 得到四组解
    DecomposeE(E21,R1,R2,t);  

    cv::Mat t1=t;
    cv::Mat t2=-t;

    // Reconstruct with the 4 hyphoteses and check  验证在这四组解情况下的点深度是否都大于0
    vector<cv::Point3f> vP3D1, vP3D2, vP3D3, vP3D4;
    // 是否满足三角化(视差是否平行)
    vector<bool> vbTriangulated1,vbTriangulated2,vbTriangulated3, vbTriangulated4;
    float parallax1,parallax2, parallax3, parallax4;
    // 检测得到的四个相机位姿是否符合要求
    // 返回的是经过筛选后内点的数量
    int nGood1 = CheckRT(R1,t1,mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K, vP3D1, 4.0*mSigma2, vbTriangulated1, parallax1);
    int nGood2 = CheckRT(R2,t1,mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K, vP3D2, 4.0*mSigma2, vbTriangulated2, parallax2);
    int nGood3 = CheckRT(R1,t2,mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K, vP3D3, 4.0*mSigma2, vbTriangulated3, parallax3);
    int nGood4 = CheckRT(R2,t2,mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K, vP3D4, 4.0*mSigma2, vbTriangulated4, parallax4);
    // 得到四个相机位姿中经过筛选内点数量最多的一组相机位姿
    int maxGood = max(nGood1,max(nGood2,max(nGood3,nGood4)));

    R21 = cv::Mat();
    t21 = cv::Mat();
    // 允许的经过筛选最小内点数量为0.9*N
    int nMinGood = max(static_cast<int>(0.9*N),minTriangulated);
    // 四个待选相机位姿中满足要求的个数
    int nsimilar = 0;
    if(nGood1>0.7*maxGood)
        nsimilar++;
    if(nGood2>0.7*maxGood)
        nsimilar++;
    if(nGood3>0.7*maxGood)
        nsimilar++;
    if(nGood4>0.7*maxGood)
        nsimilar++;

    // If there is not a clear winner or not enough triangulated points reject initialization  
    //  如果四个相机位姿中有多个相机位姿符合要求则存在不明确性,或者是最好的内点数量都小于内点阈值,那么就返回重构失败
    if(maxGood<nMinGood || nsimilar>1)
    {
        return false;
    }

    // If best reconstruction has enough parallax initialize  如果重构中存在足够多的视差初始化,那么说明当前重构旋转平移矩阵成功
    if(maxGood==nGood1)
    {
        if(parallax1>minParallax)
        {
            vP3D = vP3D1;
            vbTriangulated = vbTriangulated1;

            R1.copyTo(R21);
            t1.copyTo(t21);
            return true;
        }
    }else if(maxGood==nGood2)
    {
        if(parallax2>minParallax)
        {
            vP3D = vP3D2;
            vbTriangulated = vbTriangulated2;

            R2.copyTo(R21);
            t1.copyTo(t21);
            return true;
        }
    }else if(maxGood==nGood3)
    {
        if(parallax3>minParallax)
        {
            vP3D = vP3D3;
            vbTriangulated = vbTriangulated3;

            R1.copyTo(R21);
            t2.copyTo(t21);
            return true;
        }
    }else if(maxGood==nGood4)
    {
        if(parallax4>minParallax)
        {
            vP3D = vP3D4;
            vbTriangulated = vbTriangulated4;

            R2.copyTo(R21);
            t2.copyTo(t21);
            return true;
        }
    }

    return false;
}
/***************
 * 	根据单应矩阵H重构相机位姿R和t
 * 	vbMatchesInliers(in): 当前匹配点是否为内点
 * 	H21(in):待分解的单应矩阵   两相机的像素坐标变换矩阵   X2=H21*X1    ,X1.X2分别为像素坐标
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
 *********/ 
/**
 * @brief 从H恢复R t
 *
 * @see
 * - Faugeras et al, Motion and structure from motion in a piecewise planar environment. International Journal of Pattern Recognition and Artificial Intelligence, 1988.
 * - Deeper understanding of the homography decomposition for vision-based control
 */
bool Initializer::ReconstructH(vector<bool> &vbMatchesInliers, cv::Mat &H21, cv::Mat &K,
                      cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated, float minParallax, int minTriangulated)
{
    int N=0;
    for(size_t i=0, iend = vbMatchesInliers.size() ; i<iend; i++)
        if(vbMatchesInliers[i])
            N++;

    // We recover 8 motion hypotheses using the method of Faugeras et al.
    // Motion and structure from motion in a piecewise planar environment.
    // International Journal of Pattern Recognition and Artificial Intelligence, 1988

    // 因为特征点是图像坐标系，所以讲H矩阵由相机坐标系换算到图像坐标系    变换到相机归一化平面坐标的变换矩阵
    cv::Mat invK = K.inv();
    cv::Mat A = invK*H21*K;

    cv::Mat U,w,Vt,V;
    cv::SVD::compute(A,w,U,Vt,cv::SVD::FULL_UV);
    V=Vt.t();

    float s = cv::determinant(U)*cv::determinant(Vt);

    float d1 = w.at<float>(0);
    float d2 = w.at<float>(1);
    float d3 = w.at<float>(2);

    // SVD分解的正常情况是特征值降序排列
    if(d1/d2<1.00001 || d2/d3<1.00001)
    {
        return false;
    }

    vector<cv::Mat> vR, vt, vn;
    vR.reserve(8);
    vt.reserve(8);
    vn.reserve(8);

    //n'=[x1 0 x3] 4 posibilities e1=e3=1, e1=1 e3=-1, e1=-1 e3=1, e1=e3=-1
    // 法向量n'= [x1 0 x3] 对应ppt的公式17
    float aux1 = sqrt((d1*d1-d2*d2)/(d1*d1-d3*d3));
    float aux3 = sqrt((d2*d2-d3*d3)/(d1*d1-d3*d3));
    float x1[] = {aux1,aux1,-aux1,-aux1};
    float x3[] = {aux3,-aux3,aux3,-aux3};

    //case d'=d2
    // 计算ppt中公式19
    float aux_stheta = sqrt((d1*d1-d2*d2)*(d2*d2-d3*d3))/((d1+d3)*d2);

    float ctheta = (d2*d2+d1*d3)/((d1+d3)*d2);
    float stheta[] = {aux_stheta, -aux_stheta, -aux_stheta, aux_stheta};

    // 计算旋转矩阵 R‘，计算ppt中公式18
    //      | ctheta      0   -aux_stheta|       | aux1|
    // Rp = |    0        1       0      |  tp = |  0  |
    //      | aux_stheta  0    ctheta    |       |-aux3|

    //      | ctheta      0    aux_stheta|       | aux1|
    // Rp = |    0        1       0      |  tp = |  0  |
    //      |-aux_stheta  0    ctheta    |       | aux3|

    //      | ctheta      0    aux_stheta|       |-aux1|
    // Rp = |    0        1       0      |  tp = |  0  |
    //      |-aux_stheta  0    ctheta    |       |-aux3|

    //      | ctheta      0   -aux_stheta|       |-aux1|
    // Rp = |    0        1       0      |  tp = |  0  |
    //      | aux_stheta  0    ctheta    |       | aux3|
    for(int i=0; i<4; i++)
    {
        cv::Mat Rp=cv::Mat::eye(3,3,CV_32F);
        Rp.at<float>(0,0)=ctheta;
        Rp.at<float>(0,2)=-stheta[i];
        Rp.at<float>(2,0)=stheta[i];
        Rp.at<float>(2,2)=ctheta;

        cv::Mat R = s*U*Rp*Vt;
        vR.push_back(R);

        cv::Mat tp(3,1,CV_32F);
        tp.at<float>(0)=x1[i];
        tp.at<float>(1)=0;
        tp.at<float>(2)=-x3[i];
        tp*=d1-d3;

        // 对t进行归一化
        cv::Mat t = U*tp;
        vt.push_back(t/cv::norm(t));

        cv::Mat np(3,1,CV_32F);
        np.at<float>(0)=x1[i];
        np.at<float>(1)=0;
        np.at<float>(2)=x3[i];

        cv::Mat n = V*np;
        if(n.at<float>(2)<0)
            n=-n;
        vn.push_back(n);
    }

    //case d'=-d2
    // 计算ppt中公式22
    float aux_sphi = sqrt((d1*d1-d2*d2)*(d2*d2-d3*d3))/((d1-d3)*d2);

    float cphi = (d1*d3-d2*d2)/((d1-d3)*d2);
    float sphi[] = {aux_sphi, -aux_sphi, -aux_sphi, aux_sphi};

    // 计算旋转矩阵 R‘，计算ppt中公式21
    for(int i=0; i<4; i++)
    {
        cv::Mat Rp=cv::Mat::eye(3,3,CV_32F);
        Rp.at<float>(0,0)=cphi;
        Rp.at<float>(0,2)=sphi[i];
        Rp.at<float>(1,1)=-1;
        Rp.at<float>(2,0)=sphi[i];
        Rp.at<float>(2,2)=-cphi;

        cv::Mat R = s*U*Rp*Vt;
        vR.push_back(R);

        cv::Mat tp(3,1,CV_32F);
        tp.at<float>(0)=x1[i];
        tp.at<float>(1)=0;
        tp.at<float>(2)=x3[i];
        tp*=d1+d3;

        cv::Mat t = U*tp;
        vt.push_back(t/cv::norm(t));

        cv::Mat np(3,1,CV_32F);
        np.at<float>(0)=x1[i];
        np.at<float>(1)=0;
        np.at<float>(2)=x3[i];

        cv::Mat n = V*np;
        if(n.at<float>(2)<0)
            n=-n;
        vn.push_back(n);
    }


    int bestGood = 0;
    int secondBestGood = 0;    
    int bestSolutionIdx = -1;
    float bestParallax = -1;
    vector<cv::Point3f> bestP3D;
    vector<bool> bestTriangulated;

    // Instead of applying the visibility constraints proposed in the Faugeras' paper (which could fail for points seen with low parallax)
    // We reconstruct all hypotheses and check in terms of triangulated points and parallax
    // d'=d2和d'=-d2分别对应8组(R t)
    for(size_t i=0; i<8; i++)
    {
        float parallaxi;
        vector<cv::Point3f> vP3Di;
        vector<bool> vbTriangulatedi;
        int nGood = CheckRT(vR[i],vt[i],mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K,vP3Di, 4.0*mSigma2, vbTriangulatedi, parallaxi);

        // 保留最优的和次优的
        if(nGood>bestGood)
        {
            secondBestGood = bestGood;
            bestGood = nGood;
            bestSolutionIdx = i;
            bestParallax = parallaxi;
            bestP3D = vP3Di;
            bestTriangulated = vbTriangulatedi;
        }
        else if(nGood>secondBestGood)
        {
            secondBestGood = nGood;
        }
    }


    if(secondBestGood<0.75*bestGood && bestParallax>=minParallax && bestGood>minTriangulated && bestGood>0.9*N)
    {
        vR[bestSolutionIdx].copyTo(R21);
        vt[bestSolutionIdx].copyTo(t21);
        vP3D = bestP3D;
        vbTriangulated = bestTriangulated;

        return true;
    }

    return false;
}

// Trianularization: 已知匹配特征点对{x x'} 和 各自相机矩阵{P P'}, 估计三维点 X
// x' = P'X  x = PX    X是像素坐标系下的坐标,   x,x'是像素坐标,P,P'分别是两像素坐标到世界坐标系下的变换矩阵
// 它们都属于 x = aPX模型
//                                               |X|
// |x|        |p1 p2  p3  p4 |    |Y|           |x|       |--p0--||.|
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
/***
 * 三角测量计算匹配点的深度(匹配点三维坐标)
 * 	kp1:相机1的关键点
 * 	kp2:相机2的关键点
 * 	P1:相机1下的点像素坐标->世界坐标系下的坐标
 * 	P2:相机2下的点像素坐标->世界坐标系下的坐标
 * 	x3D:得到3D点
 **/
void Initializer::Triangulate(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &P1, const cv::Mat &P2, cv::Mat &x3D)
{
    cv::Mat A(4,4,CV_32F);

    A.row(0) = kp1.pt.x*P1.row(2)-P1.row(0);
    A.row(1) = kp1.pt.y*P1.row(2)-P1.row(1);
    A.row(2) = kp2.pt.x*P2.row(2)-P2.row(0);
    A.row(3) = kp2.pt.y*P2.row(2)-P2.row(1);

    cv::Mat u,w,vt;
    cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);
    x3D = vt.row(3).t();
    x3D = x3D.rowRange(0,3)/x3D.at<float>(3);
}
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
void Initializer::Normalize(const vector<cv::KeyPoint> &vKeys, vector<cv::Point2f> &vNormalizedPoints, cv::Mat &T)
{
    float meanX = 0;
    float meanY = 0;
    const int N = vKeys.size();

    vNormalizedPoints.resize(N);

    for(int i=0; i<N; i++)
    {
        meanX += vKeys[i].pt.x;
        meanY += vKeys[i].pt.y;
    }

    meanX = meanX/N;
    meanY = meanY/N;

    float meanDevX = 0;
    float meanDevY = 0;
    // 将所有vKeys点减去中心坐标，使x坐标和y坐标均值分别为0
    for(int i=0; i<N; i++)
    {
        vNormalizedPoints[i].x = vKeys[i].pt.x - meanX;
        vNormalizedPoints[i].y = vKeys[i].pt.y - meanY;

        meanDevX += fabs(vNormalizedPoints[i].x);
        meanDevY += fabs(vNormalizedPoints[i].y);
    }

    meanDevX = meanDevX/N;
    meanDevY = meanDevY/N;

    float sX = 1.0/meanDevX;
    float sY = 1.0/meanDevY;
    // 将x坐标和y坐标分别进行尺度缩放，使得x坐标和y坐标的一阶绝对矩分别为1
    for(int i=0; i<N; i++)
    {
        vNormalizedPoints[i].x = vNormalizedPoints[i].x * sX;
        vNormalizedPoints[i].y = vNormalizedPoints[i].y * sY;
    }
    // T=( sX ,0 ,-meanY*sY ;0 ,sY ,-meanX*sX ;0 ,0 ,1  ) T为坐标变换的矩阵   X'=T*X   ,X为归一化之前的像素坐标  X'为归一化之后的坐标
    T = cv::Mat::eye(3,3,CV_32F);
    T.at<float>(0,0) = sX;
    T.at<float>(1,1) = sY;
    T.at<float>(0,2) = -meanX*sX;
    T.at<float>(1,2) = -meanY*sY;
}

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
int Initializer::CheckRT(const cv::Mat &R, const cv::Mat &t, const vector<cv::KeyPoint> &vKeys1, const vector<cv::KeyPoint> &vKeys2,
                       const vector<Match> &vMatches12, vector<bool> &vbMatchesInliers,
                       const cv::Mat &K, vector<cv::Point3f> &vP3D, float th2, vector<bool> &vbGood, float &parallax)
{
    // Calibration parameters  相机内参
    const float fx = K.at<float>(0,0);
    const float fy = K.at<float>(1,1);
    const float cx = K.at<float>(0,2);
    const float cy = K.at<float>(1,2);

    vbGood = vector<bool>(vKeys1.size(),false);
    vP3D.resize(vKeys1.size());

    vector<float> vCosParallax;
    vCosParallax.reserve(vKeys1.size());

    // Camera 1 Projection Matrix K*[I|0]  相机1的映射矩阵  从像素坐标映射到相机坐标下的变换矩阵
    cv::Mat P1(3,4,CV_32F,cv::Scalar(0));
    K.copyTo(P1.rowRange(0,3).colRange(0,3));
    // 相机1 的相机中心在世界坐标系下的坐标
    cv::Mat O1 = cv::Mat::zeros(3,1,CV_32F);

    // Camera 2 Projection Matrix K*[R|t]  相机2的映射矩阵  从像素坐标映射到相机坐标下的变换矩阵
    cv::Mat P2(3,4,CV_32F);
    R.copyTo(P2.rowRange(0,3).colRange(0,3));
    t.copyTo(P2.rowRange(0,3).col(3));
    P2 = K*P2;
    // 第二个相机的光心在世界坐标系下的坐标
    // 注意计算相机光心的公式   三维点的变换矩阵是Tcw=[R|t]      而相机光心则做相对运动(T的逆) T'=[R.t | t*R.t]  而相机光心应该是从世界坐标到相机坐标,因此需要求逆之后的t
    cv::Mat O2 = -R.t()*t;   

    int nGood=0;
    // 针对每一个匹配点
    for(size_t i=0, iend=vMatches12.size();i<iend;i++)
    {
        if(!vbMatchesInliers[i])// 检测是否是内点
            continue;

        const cv::KeyPoint &kp1 = vKeys1[vMatches12[i].first];
        const cv::KeyPoint &kp2 = vKeys2[vMatches12[i].second];
        cv::Mat p3dC1;
	// 三角测量计算像素点的深度信息 
        Triangulate(kp1,kp2,P1,P2,p3dC1);

        if(!isfinite(p3dC1.at<float>(0)) || !isfinite(p3dC1.at<float>(1)) || !isfinite(p3dC1.at<float>(2)))
        {
            vbGood[vMatches12[i].first]=false;
            continue;
        }

        // Check parallax
        // 三维点在第一个相机中的视差
        cv::Mat normal1 = p3dC1 - O1;
        float dist1 = cv::norm(normal1);

	//  检测
	// 三维点在第二个相机中的视差
        cv::Mat normal2 = p3dC1 - O2;
        float dist2 = cv::norm(normal2);
	// normal1和normal2的夹角cos值
        float cosParallax = normal1.dot(normal2)/(dist1*dist2);

        // Check depth in front of first camera (only if enough parallax, as "infinite" points can easily go to negative depth)
	// 3D点在第一个相机坐标系下的坐标深度为负
        if(p3dC1.at<float>(2)<=0 && cosParallax<0.99998)
            continue;

        // Check depth in front of second camera (only if enough parallax, as "infinite" points can easily go to negative depth)
        cv::Mat p3dC2 = R*p3dC1+t;
	// 3D点在第二个相机坐标系下的坐标深度为负
        if(p3dC2.at<float>(2)<=0 && cosParallax<0.99998)
            continue;

        // Check reprojection error in first image  计算重投影误差
        float im1x, im1y;
        float invZ1 = 1.0/p3dC1.at<float>(2);
        im1x = fx*p3dC1.at<float>(0)*invZ1+cx;
        im1y = fy*p3dC1.at<float>(1)*invZ1+cy;

        float squareError1 = (im1x-kp1.pt.x)*(im1x-kp1.pt.x)+(im1y-kp1.pt.y)*(im1y-kp1.pt.y);

        if(squareError1>th2)  //重投影误差大于阈值
            continue;

        // Check reprojection error in second image   第二幅图像的重投影误差
        float im2x, im2y;
        float invZ2 = 1.0/p3dC2.at<float>(2);
        im2x = fx*p3dC2.at<float>(0)*invZ2+cx;
        im2y = fy*p3dC2.at<float>(1)*invZ2+cy;

        float squareError2 = (im2x-kp2.pt.x)*(im2x-kp2.pt.x)+(im2y-kp2.pt.y)*(im2y-kp2.pt.y);

        if(squareError2>th2)
            continue;
	//统计经过检验的3D点个数，记录3D点视差角
        vCosParallax.push_back(cosParallax);
        vP3D[vMatches12[i].first] = cv::Point3f(p3dC1.at<float>(0),p3dC1.at<float>(1),p3dC1.at<float>(2));
        nGood++;
	
        if(cosParallax<0.99998)
            vbGood[vMatches12[i].first]=true;
    }
    //  得到3D点中较大的视差角
    if(nGood>0)
    {
        sort(vCosParallax.begin(),vCosParallax.end());

        size_t idx = min(50,int(vCosParallax.size()-1));
        parallax = acos(vCosParallax[idx])*180/CV_PI;
    }
    else
        parallax=0;
    // 返回经过检验的内点数量
    return nGood;
}
//                                                |0 -1  0|
// E = U Sigma V'   let W = |1  0  0|
//                                                |0  0  1|
// 得到4个解 E = [R|t]
// R1 = UWV' R2 = UW'V' t1 = U3 t2 = -U3

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
void Initializer::DecomposeE(const cv::Mat &E, cv::Mat &R1, cv::Mat &R2, cv::Mat &t)
{
    cv::Mat u,w,vt;
    // 将E进行SVD分解
    cv::SVD::compute(E,w,u,vt);
    // 将特征向量附给t
    u.col(2).copyTo(t);
    // 将向量t进行归一化
    t=t/cv::norm(t);
    // w=Rz(pi/2)
    cv::Mat W(3,3,CV_32F,cv::Scalar(0));
    W.at<float>(0,1)=-1;
    W.at<float>(1,0)=1;
    W.at<float>(2,2)=1;

    R1 = u*W*vt;  //R1=u*W*vt
    if(cv::determinant(R1)<0)
        R1=-R1;

    R2 = u*W.t()*vt; //R2=u*W'*vt
    if(cv::determinant(R2)<0)
        R2=-R2;
}

} //namespace ORB_SLAM
