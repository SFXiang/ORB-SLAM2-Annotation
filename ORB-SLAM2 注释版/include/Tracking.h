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


#ifndef TRACKING_H
#define TRACKING_H

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include"Viewer.h"
#include"FrameDrawer.h"
#include"Map.h"
#include"LocalMapping.h"
#include"LoopClosing.h"
#include"Frame.h"
#include "ORBVocabulary.h"
#include"KeyFrameDatabase.h"
#include"ORBextractor.h"
#include "Initializer.h"
#include "MapDrawer.h"
#include "System.h"

#include <mutex>

namespace ORB_SLAM2
{

class Viewer;
class FrameDrawer;
class Map;
class LocalMapping;
class LoopClosing;
class System;

class Tracking
{  

public:
  //pSys  指定本系统， pVoc  指定所用的词典，pFrameDrawer  每一帧的观测器，pMapDrawer  地图的观测器 ，pMap 指代整个地图
  //pKFDB  关键帧数据集， strSettingPath配置文件的路径（相机相关参数以及工程中所用到的参数） ，sensor 传感器类型
    Tracking(System* pSys, ORBVocabulary* pVoc, FrameDrawer* pFrameDrawer, MapDrawer* pMapDrawer, Map* pMap,
             KeyFrameDatabase* pKFDB, const string &strSettingPath, const int sensor);

    // Preprocess the input and call Track(). Extract features and performs stereo matching.  处理输入并调用Track()函数，提取特征点并进行特征匹配
    cv::Mat GrabImageStereo(const cv::Mat &imRectLeft,const cv::Mat &imRectRight, const double &timestamp);
    // Preprocess the input and call Track(). Extract features and performs RGBD image matching.  处理输入并调用Track()函数，提取特征点并进行特征匹配
    cv::Mat GrabImageRGBD(const cv::Mat &imRGB,const cv::Mat &imD, const double &timestamp);
    // Preprocess the input and call Track(). Extract features and performs monocular matching.  处理输入并调用Track()函数，提取特征点并进行特征匹配
    cv::Mat GrabImageMonocular(const cv::Mat &im, const double &timestamp);

    void SetLocalMapper(LocalMapping* pLocalMapper);
    void SetLoopClosing(LoopClosing* pLoopClosing);
    void SetViewer(Viewer* pViewer);

    // Load new settings  加载新的配置文件，如果配置文件临时有改动
    // The focal lenght should be similar or scale prediction will fail when projecting points
    // TODO: Modify MapPoint::PredictScale to take into account focal lenght
    void ChangeCalibration(const string &strSettingPath);

    // Use this function if you have deactivated local mapping and you only want to localize the camera.
    //使用该函数如果你已经关闭局部地图构建并且仅仅想进行相机的定位
    void InformOnlyTracking(const bool &flag);


public:

    // Tracking states    当前的追踪线程状态
    enum eTrackingState{
        SYSTEM_NOT_READY=-1,    //系统还未准备好
        NO_IMAGES_YET=0,            //未曾有图片
        NOT_INITIALIZED=1,          //未初始化
        OK=2,                                   //正常状态
        LOST=3                                //丢帧状态
    };

    eTrackingState mState;              //当前追踪状态
    eTrackingState mLastProcessedState;   //前一追踪状态

    // Input sensor
    int mSensor;

    // Current Frame        当前帧
    Frame mCurrentFrame;
    cv::Mat mImGray;

    // Initialization Variables (Monocular)   初始化变量（单目）
    // 在上一帧中所有匹配点的索引
    std::vector<int> mvIniLastMatches;
    // 在当前帧中与参考初始化帧之间所有的匹配点对的索引 下标为匹配点在帧1的索引  值为匹配点在帧2的索引
    std::vector<int> mvIniMatches;
    // 上一次的匹配特征点(关键点坐标)
    std::vector<cv::Point2f> mvbPrevMatched;
    std::vector<cv::Point3f> mvIniP3D;
    // 初始化参考帧
    Frame mInitialFrame;

    // Lists used to recover the full camera trajectory at the end of the execution.   
    // 这些变量是用来在整个SLAM结束后生成完整的相机轨迹
    // Basically we store the reference keyframe for each frame and its relative transformation
    // 基本上我们为每一帧存储他的参考关键帧和他们之间的变换矩阵T
    // 存储上一个状态 当前帧相对于参考帧的变换矩阵
    list<cv::Mat> mlRelativeFramePoses;
    // 存储参考关键帧
    list<KeyFrame*> mlpReferences;
    //存储该帧帧的时间戳
    list<double> mlFrameTimes;
    //存储该帧丢失的状态
    list<bool> mlbLost;

    // True if local mapping is deactivated and we are performing only localization
    bool mbOnlyTracking;

    void Reset();

protected:

    // Main tracking function. It is independent of the input sensor.   主要追踪线程的函数，它独立于外部输入传感器
/*******************************************************************************
 *     函数属性：类Tracking的成员函数Track()   跟踪线程有四个状态分别是 没有图片,未初始化(第一张图片),线程完好, 追踪线程失败
 *     函数功能：检测当前系统状态是处于NO_IMAGES_YET，NOT_INITIALIZED，还是OK or LOST
 * 			1. 计算相机位姿:
 *				当前帧为第一帧则调用StereoInitialization()函数进行地图初始化
 *  				如果当前帧不为第一帧则：
 * 					检测当前系统的模式是否处于仅线程追踪模式，或者是处于局部地图构建与线程追踪同时运行
 *  					如果处于仅追踪线程：
 *                                          如果线程帧丢失状态，那么进行重定位   
 *                                          如果线程帧未丢失：
 *                                                 如果此帧匹配了足够多的地图点,则计算相机位姿(根据上一帧映射来计算位姿或者根据参考关键帧的BOW向量来计算位姿)
 *                                          	    如果此帧没有匹配到足够多的地图点,初始化相机位姿估计使用移动模型和重定位分别获得两个相机位姿
 *   				 	 如果未处于仅追踪模式:
 *  						如果线程帧丢失状态，那么进行重定位
 * 						如果线程未丢失:
 * 							如果是第二帧数据或者是重定位之后的第二帧,则通过参考关键帧来计算位姿
 * 							否则,通过上一帧的位姿和位姿变化速度来计算位姿,如果计算失败再通过参考关键帧来计算位姿
 * 					当前帧的参考关键帧置为当前的参考关键帧
 * 					进行局部建图,如果建图成功则当前状态为正常,否则追踪状态为丢失
 * 					如果追踪线程成功,则更新mVelocity(当前帧的位姿变化速度),并更新当前帧的地图点(不包括局外点),清空临时地图点,检测是否需要给局部地图添加关键帧
 * 					
 * 			2. 如果计算相机位姿成功
 * 				则计算当前帧相对于参考帧的变换矩阵存入mlRelativeFramePoses
 * 				存储当前帧的参考关键帧,当前帧的时间,以及当前帧是否丢失追踪
 * 				
 *                       存储完整的帧姿态信息以获取完整的相机轨迹
 *     函数参数介绍：NULL
 *     备注：追踪线程的主要成员函数，实现视觉里程计
 * 
 ******************************************************************************/
    void Track();

    // Map initialization for stereo and RGB-D   为双目和RGB-D相机进行地图初始化
    void StereoInitialization();

    // Map initialization for monocular    为单目进行地图初始化
/****************************************************************************
 *       单目相机的初始化
 * 		单目相机初始化至少需要两帧
 * 		如果是第一帧初始化帧 则建立好初始化器
 * 		如果已经建立好初始化器并且有初始化的参考帧, 则
 * 			匹配当前帧和初始化帧
 * 			计算两帧之间的位姿(H矩阵和F矩阵)
 * 		如果初始化成功 ,则更新初始化地图,将地图点加入初始化地图中,并更新关键帧,将关键帧加入到局部建图中
 ***************************************************************************/
    void MonocularInitialization();
    void CreateInitialMapMonocular();
    
//检测上一帧中的所有地图点，看是否已有其他地图点可以代替该地图点，如果可以代替则将其替代掉
    void CheckReplacedInLastFrame();
    /*******************************************************************************
 *     函数属性：类Tracking的成员函数TrackReferenceKeyFrame()
 *     函数功能：
 *                 1. 计算当前帧的词包，将当前帧的特征点分到特定层的nodes上
 *                 2. 对属于同一node的描述子进行匹配
 *                 3. 根据匹配对估计当前帧的姿态
 *                 4. 根据位姿估计下的模型判断是否为内点来剔除误匹配
 *                 5. 返回当前优化位姿模型下内点数量是否大于10  大于10证明当前位姿估计模型是好的,否则是差的
 *     函数参数介绍：NULL
 *     备注：
 *               根据上一帧的位姿作为初始位姿进行位姿优化得到当前帧的位姿
 ******************************************************************************/
    bool TrackReferenceKeyFrame();
/***********************************************************
 *		更新上一帧
 * 		1  根据参考关键帧相对于初始帧的相机位姿和上一帧相对于参考关键帧的相机位姿来计算上一帧相对于初始帧的相机位姿
 * 		2  如果上一帧不为关键帧并且当前不是单目相机并且是在仅追踪模式下,则
 *			 1.创建视觉里程计的地图点  根据测量他们的地图点深度排序这些点
 * 			 2.创建视觉里程计的地图点  根据测量他们的地图点深度排序这些点
 ****************************************************************/
    void UpdateLastFrame();
/******************************************************************************
 *    根据运动模型来进行位姿计算  并完成追踪
 *    1  更新上一帧的相机位姿   根据上一帧相机位姿以及上一帧相机位姿变化的速度来估算当前位姿
 *    2  通过映射的方式匹配当前帧和上一帧,并追踪上一帧的地图点
 *    3  优化当前帧的相机位姿
 *    4  去除局外点,如果当前模型下的局内点数量大于10 则返回成功追踪
 *    5  仅追踪模式下地图内的匹配点数量小于10将mbVO置1  表明地图内匹配内点数量不足   返回局内点数量是否大于20
 ***********************************************************************/
    bool TrackWithMotionModel();
/*******************************************************************************
 *     函数属性：类Tracking的成员函数Relocalization()
 *     函数功能：（1）计算当前帧的BoW映射
 *                        （2）找到与当前帧相似的候选关键帧
 *                        （3）匹配当前帧与找到的候选关键帧，计算相机位姿Tcw
 *                        （4）优化相机位姿Tcw
 *                                         首先进行优化，将优化的结果存到nGood中，
 *                                         如果优化结果不理想进行映射匹配，然后再进行优化
 *                                         如果优化结果还不理想缩小映射窗口后在进行匹配并优化
 *                                         此时若还不理想就判定该候选关键帧不能与本帧形成匹配，继续进行下一关键帧的匹配;如果可以，则证明已经进行重定位或者找到回环，退出循环
 *     函数参数介绍：NULL
 *     备注：重定位函数，用于失帧重定位和回环检测函数中
 * 
 ******************************************************************************/
    bool Relocalization();               //重定位
/*  更新局部地图
  1  为全局地图设置参考地图点
  2  更新局部关键帧和局部地图点*/
    void UpdateLocalMap();           //更新局部地图
    //  更新局部地图点
//  将局部关键帧中的所有关键帧的地图点加入局部地图点中
    void UpdateLocalPoints();        //更新局部地图点
    /************
 *    更新局部关键帧
 *    添加的关键帧为:     1  当前帧的所有地图点还在哪些关键帧中看到   将这些关键帧加入局部关键帧
 *                                         2  将所有看到地图点的所有关键帧的共视关键帧添加到局部地图中
 *                                         3  将所有看到地图点的所有关键帧的父关键帧和子关键帧添加到局部地图中
 *    更新参考关键帧为局部关键帧中看到最多当前帧地图点的关键帧
 ********/
    void UpdateLocalKeyFrames(); 
/*****************************************************************************
 * 		追踪局部地图
 * 		1. 更新局部地图(更新局部关键帧和局部地图点)
 * 		2. 在局部地图点中寻找在当前帧的视野范围内的点,匹配视野范围内的点与当前帧,匹配点存储到当前帧的地图点容器中
 * 		3. 根据新得到的匹配点重新优化相机位姿并得到匹配内点, 通过匹配内点的数量来判断当前局部地图是否追踪成功
 ****************************************************************************/
    bool TrackLocalMap(); 
/**
 * @brief 对Local MapPoints进行跟踪
 * 
 * 在局部地图中查找在当前帧视野范围内的点，将视野范围内的点和当前帧的特征点进行投影匹配
 */
    void SearchLocalPoints(); 
// 判断是否添加关键帧   
//判断条件:   1 仅线程追踪模式不需要添加新的关键帧
//                      2 局部地图模式被暂停或者被发起暂停请求(回环检测冻结局部地图)    则不添加关键帧
//                      3 上次重定位后没有足够的帧通过   则不添加关键帧
//                      4  追踪线程是比较弱的
//                      5   此帧距离上次插入关键帧已经超过了最大的帧数(很久没插入关键帧了)
//                      6  局部地图处于空闲状态并且没有超过最大关键帧数
//                      7  局内点数量比较少  跟踪地图中地图点比较少
    bool NeedNewKeyFrame();       //需要新的关键帧
// 添加新的关键帧
// 建立关键帧中的地图点加入全局map中(非单目情况下),并且将关键帧传递给localmapping线程
    void CreateNewKeyFrame();  

    // In case of performing only localization, this flag is true when there are no matches to
    // points in the map. Still tracking will continue if there are enough matches with temporal points.
    // In that case we are doing visual odometry. The system will try to do relocalization to recover
    // "zero-drift" localization to the map.
    // 在只执行定位的情况下，当在地图点中没有匹配时这个标志为true，如果在这时有足够的匹配那么追踪仍会继续
    // 在这种情况下我们进行视觉里程计，这个系统将努力去进行重定位来恢复0定位到地图中
    // 在TrackWithMotionModel()中置位  false证明有足够的匹配点能在地图中找到,  true表明没有足够的匹配点在地图点中找到
    bool mbVO;

    //Other Thread Pointers 其他线程的指针
    //局部建图线程的指针
    LocalMapping* mpLocalMapper;
    //回环检测线程的指针
    LoopClosing* mpLoopClosing;

    //ORB  ORB特征点提取器
    ORBextractor* mpORBextractorLeft, *mpORBextractorRight;
    ORBextractor* mpIniORBextractor;

    //BoW   ORB词典
    ORBVocabulary* mpORBVocabulary;
    // 关键帧的数据集
    KeyFrameDatabase* mpKeyFrameDB;

    // Initalization (only for monocular)    初始化（仅用于单目）
    Initializer* mpInitializer;

    //Local Map   局部地图
    // 参考关键帧
    KeyFrame* mpReferenceKF;
    // 局部地图的关键帧
    std::vector<KeyFrame*> mvpLocalKeyFrames;
    // 局部地图的地图点
    std::vector<MapPoint*> mvpLocalMapPoints;
    
    // System
    // 指代该系统
    System* mpSystem;
    
    //Drawers   
    //观察器
    Viewer* mpViewer;
    //每一帧的观测器
    FrameDrawer* mpFrameDrawer;
    //整个地图的观测器
    MapDrawer* mpMapDrawer;

    //Map   指代整个地图
    Map* mpMap;

    //Calibration matrix
    // 相机内参矩阵K
    cv::Mat mK;
    // 相机的矫正矩阵
    cv::Mat mDistCoef;
    //base-distance*f
    float mbf;

    //New KeyFrame rules (according to fps)
    // 新关键帧插入的最小间隔帧数
    int mMinFrames;
   // 新关键帧插入的最大间隔帧数
    int mMaxFrames;

    // Threshold close/far points
    // Points seen as close by the stereo/RGBD sensor are considered reliable
    // and inserted from just one frame. Far points requiere a match in two keyframes.
    float mThDepth;

    // For RGB-D inputs only. For some datasets (e.g. TUM) the depthmap values are scaled.  //数据集中深度图像的缩放值
    float mDepthMapFactor;

    // Current matches in frame    在该帧中当前的匹配内点数
    int mnMatchesInliers;

    // Last Frame, KeyFrame and Relocalisation Info   上一帧，关键帧和重定位信息
    // 上一关键帧
    KeyFrame* mpLastKeyFrame;
    // 上一帧
    Frame mLastFrame;
    // 上一关键帧的id
    unsigned int mnLastKeyFrameId;
    // 上一重定位帧的id
    unsigned int mnLastRelocFrameId;

    //Motion Model 移动模型    存储Tcl    运动速度  即上上一帧到上一帧的位姿变换矩阵,用来推测上一帧到当前帧的位姿变换矩阵
    cv::Mat mVelocity;

    //Color order (true RGB, false BGR, ignored if grayscale) 颜色顺序，RGB 1 BGR 0  如果是灰度图那么就忽略该参数
    bool mbRGB;

    list<MapPoint*> mlpTemporalPoints;
};

} //namespace ORB_SLAM

#endif // TRACKING_H
