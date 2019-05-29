/**
* This file is part of ORB-SLAM2.
* This file is a modified version of EPnP <http://cvlab.epfl.ch/EPnP/index.php>, see FreeBSD license below.
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

/**
* Copyright (c) 2009, V. Lepetit, EPFL
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
* 1. Redistributions of source code must retain the above copyright notice, this
*    list of conditions and the following disclaimer.
* 2. Redistributions in binary form must reproduce the above copyright notice,
*    this list of conditions and the following disclaimer in the documentation
*    and/or other materials provided with the distribution.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
* ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
* ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*
* The views and conclusions contained in the software and documentation are those
* of the authors and should not be interpreted as representing official policies,
*   either expressed or implied, of the FreeBSD Project
*/

#ifndef PNPSOLVER_H
#define PNPSOLVER_H

#include <opencv2/core/core.hpp>
#include "MapPoint.h"
#include "Frame.h"

namespace ORB_SLAM2
{

class PnPsolver {
 public:
    //	 pnp求解器的初始化函数
  PnPsolver(const Frame &F, const vector<MapPoint*> &vpMapPointMatches);
    //	 pnp析构函数
  ~PnPsolver();
   //设置PNP相关参数
  void SetRansacParameters(double probability = 0.99, int minInliers = 8 , int maxIterations = 300, int minSet = 4, float epsilon = 0.4,
                           float th2 = 5.991);
  // 
  cv::Mat find(vector<bool> &vbInliers, int &nInliers);
  // 迭代求解
  cv::Mat iterate(int nIterations, bool &bNoMore, vector<bool> &vbInliers, int &nInliers);

 private:
  //检测内点（用重投影误差判断是否为内点）
  void CheckInliers();
  bool Refine();

  // Functions from the original EPnP code
  //根据maximum_number_of_correspondences给各变量开辟存储空间
  void set_maximum_number_of_correspondences(const int n);
  //重置相应点对（当前选择的内点）
  void reset_correspondences(void);
  //pws  地图点世界坐标   us 图像的像素坐标   的填充
  void add_correspondence(const double X, const double Y, const double Z,
              const double u, const double v);
  //计算相机位姿的核心代码（EPNP+ICP(SVD)方法）
  double compute_pose(double R[3][3], double T[3]);

  void relative_error(double & rot_err, double & transl_err,
              const double Rtrue[3][3], const double ttrue[3],
              const double Rest[3][3],  const double test[3]);
//打印位姿
  void print_pose(const double R[3][3], const double t[3]);
  //计算反投影误差   像素坐标的理论值和实际值之差的平方
  double reprojection_error(const double R[3][3], const double t[3]);
//选择控制点    中心点及主成分分析的方法得到四个控制点 cws1,2,3,4
  void choose_control_points(void);
// 求解四个控制点的系数alphas
// (a2 a3 a4)' = inverse(cws2-cws1 cws3-cws1 cws4-cws1)*(pws-cws1)，a1 = 1-a2-a3-a4
  // 每一个3D控制点，都有一组alphas与之对应
// cws1 cws2 cws3 cws4为四个控制点的坐标
  // pws为3D参考点的坐标   us该帧特征点对应的像素坐标
//该计算方法依据    ：
//      pws = sum(cws*alphas) = (z1 z2 z3 z4)*alphas  >>>alphas = inv(z1 z2 z3 z4)*pws
  void compute_barycentric_coordinates(void);
//相机坐标系下控制点的坐标   每一个控制点在相机坐标系下都表示为特征向量乘以beta的形式，EPnP论文的公式16
  void compute_ccs(const double * betas, const double * ut);
  
  //MX=0   填充M
  void fill_M(CvMat * M, const int row, const double * alphas, const double u, const double v);
  // 计算相机坐标系下匹配点的坐标
  void compute_pcs(void);
  
  
  void solve_for_sign(void);
// betas10        = [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
// betas_approx_1 = [B11 B12     B13         B14]
//当n=1时     betas_approx_1 = [B11 B12     B13         B14]
  void find_betas_approx_1(const CvMat * L_6x10, const CvMat * Rho, double * betas);
  // betas10        = [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
// betas_approx_2 = [B11 B12 B22                            ]
//当N=2时  betas_approx_2 = [B11 B12 B22                            ]
  void find_betas_approx_2(const CvMat * L_6x10, const CvMat * Rho, double * betas);
 // betas10        = [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
// betas_approx_3 = [B11 B12 B22 B13 B23                    ]
//当N=3时
  void find_betas_approx_3(const CvMat * L_6x10, const CvMat * Rho, double * betas);
  //用qr分解求解AX=b
  void qr_solve(CvMat * A, CvMat * b, CvMat * X);
//求內积
  double dot(const double * v1, const double * v2);
//求两向量p1 p2的距离
  double dist2(const double * p1, const double * p2);
   //任意两点之间的距离 （世界坐标系下的两两控制点坐标）
  void compute_rho(double * rho);
  //计算L矩阵  Lbeta=rho
  void compute_L_6x10(const double * ut, double * l_6x10);

  void gauss_newton(const CvMat * L_6x10, const CvMat * Rho, double current_betas[4]);
  void compute_A_and_b_gauss_newton(const double * l_6x10, const double * rho,
				    double cb[4], CvMat * A, CvMat * b);
//根据控制点求解R和t
  double compute_R_and_t(const double * ut, const double * betas,
			 double R[3][3], double t[3]);
//ICP方法计算相机位姿    3D-3D 
  void estimate_R_and_t(double R[3][3], double t[3]);
//拷贝R和t
  void copy_R_and_t(const double R_dst[3][3], const double t_dst[3],
		    double R_src[3][3], double t_src[3]);
//根据旋转矩阵求解四元数
  void mat_to_quat(const double R[3][3], double q[4]);

//相机参数
  double uc, vc, fu, fv;
  //pws  3D参考点的首地址  us 该帧数据中的图像特征点  alphas 每个3D参考点都有四个控制点构成，alphas为系数的首地址   pcs  相机坐标系下地图点的首地址
  double * pws, * us, * alphas, * pcs;
  //每次RANSAC迭代需要的特征点数
  int maximum_number_of_correspondences;
  int number_of_correspondences;
   //cws 世界坐标系下四个控制点坐标（可以由地图点的中心点和由主成分分析得到的三个主方向的到）  ccs 相机坐标系下四个控制点的坐标 
  double cws[4][3], ccs[4][3];
  double cws_determinant;
  // 该帧与地图点的匹配对   该容器存储了地图点
  vector<MapPoint*> mvpMapPointMatches;

  // 2D Points  存储当前帧的关键点2D坐标
  vector<cv::Point2f> mvP2D;
  //该帧中每个关键点对应的空间尺度
  vector<float> mvSigma2;

  // 3D Points   该地图点的世界坐标
  vector<cv::Point3f> mvP3Dw;

  // Index in Frame  在该帧中的关键点索引
  vector<size_t> mvKeyPointIndices;

  // Current Estimation  当前估计状态
  //当前相机旋转矩阵的最优值
  double mRi[3][3];
  //当前相机平移矩阵的最优值
  double mti[3];
  //当前变换矩阵的最优值
  cv::Mat mTcwi;
  //当前最优模型（相机位姿）下所有匹配点是否为内点
  vector<bool> mvbInliersi;
  //内点数量
  int mnInliersi;

  // Current Ransac State
  // RANSAC的迭代次数
  int mnIterations;
  // 最优状态下各匹配点是否为内点
  vector<bool> mvbBestInliers;
  //最优状态下的内点数量
  int mnBestInliers;
  // 最优的相机位姿
  cv::Mat mBestTcw;

  // Refined
  cv::Mat mRefinedTcw;
  vector<bool> mvbRefinedInliers;
  int mnRefinedInliers;

  // Number of Correspondences   匹配点数量
  int N;

  // Indices for random selection [0 .. N-1]
  vector<size_t> mvAllIndices;

  // RANSAC probability
  double mRansacProb;

  // RANSAC min inliers  
  int mRansacMinInliers;

  // RANSAC max iterations   
  int mRansacMaxIts;

  // RANSAC expected inliers/total ratio   预期的内点数和比率和
  float mRansacEpsilon;

  // RANSAC Threshold inlier/outlier. Max error e = dist(P1,T_12*P2)^2   内点和外点阈值
  float mRansacTh;

  // RANSAC Minimun Set used at each iteration
  //// mRansacMinSet为每次RANSAC需要的特征点数，默认为4组3D-2D对应点
  int mRansacMinSet;

  // Max square error associated with scale level. Max error = th*th*sigma(level)*sigma(level)
  vector<float> mvMaxError;

};

} //namespace ORB_SLAM

#endif //PNPSOLVER_H
