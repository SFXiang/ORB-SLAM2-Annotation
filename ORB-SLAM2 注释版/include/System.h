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


#ifndef SYSTEM_H
#define SYSTEM_H

#include<string>
#include<thread>
#include<opencv2/core/core.hpp>

#include "Tracking.h"
#include "FrameDrawer.h"
#include "MapDrawer.h"
#include "Map.h"
#include "LocalMapping.h"
#include "LoopClosing.h"
#include "KeyFrameDatabase.h"
#include "ORBVocabulary.h"
#include "Viewer.h"

namespace ORB_SLAM2
{
class Viewer;
class FrameDrawer;
class Map;
class Tracking;
class LocalMapping;
class LoopClosing;

class System
{
public:
    // Input sensor
    enum eSensor{
        MONOCULAR=0,
        STEREO=1,
        RGBD=2
    };

public:

    // Initialize the SLAM system. It launches the Local Mapping, Loop Closing and Viewer threads.
   //初始化SLAM系统 。启动当地建图，回环检测和可视化器线程
    System(const string &strVocFile, const string &strSettingsFile, const eSensor sensor, const bool bUseViewer = true);

    // Proccess the given stereo frame. Images must be synchronized and rectified.
    // Input images: RGB (CV_8UC3) or grayscale (CV_8U). RGB is converted to grayscale.
    // Returns the camera pose (empty if tracking fails).
    //处理被给的双目帧，两张图片必须是被同步和被纠正后的
    //输入图片是RGB三通道的unsigned char型数据或者灰度图像 unsigned int型数据 RGB被转化为灰度图像
    //返回的是相机位姿，如果线程失败则返回空
    cv::Mat TrackStereo(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timestamp);

    // Process the given rgbd frame. Depthmap must be registered to the RGB frame.
    // Input image: RGB (CV_8UC3) or grayscale (CV_8U). RGB is converted to grayscale.
    // Input depthmap: Float (CV_32F).
    // Returns the camera pose (empty if tracking fails).
    //处理的是RGBD帧，深度图必须是和RGB图进行标定后的
    //输入图片是RGB三通道的unsigned char型数据或者灰度图像 unsigned int型数据 RGB被转化为灰度图像
    //输入深度地图为float型
    //返回的是相机位姿，如果线程失败则返回空
    cv::Mat TrackRGBD(const cv::Mat &im, const cv::Mat &depthmap, const double &timestamp);

    // Proccess the given monocular frame
    // Input images: RGB (CV_8UC3) or grayscale (CV_8U). RGB is converted to grayscale.
    // Returns the camera pose (empty if tracking fails).
    //处理被给的单目帧
    //输入图片是RGB三通道的unsigned char型数据或者灰度图像 unsigned int型数据 RGB被转化为灰度图像
    //返回的是相机位姿，如果线程失败则返回空
    cv::Mat TrackMonocular(const cv::Mat &im, const double &timestamp);

    // This stops local mapping thread (map building) and performs only camera tracking.
    //这个函数停止局部建图线程并且仅仅执行相机捕捉线程
    void ActivateLocalizationMode();
    // This resumes local mapping thread and performs SLAM again.
    // 这个函数重新开始局部建图线程并执行SLAM
    void DeactivateLocalizationMode();

    // Returns true if there have been a big map change (loop closure, global BA)
    // since last call to this function
    //与上次调用该函数时想比较地图是否发生了巨大的改变（回环检测或者全局优化），如果发生了巨大的改变返回真，否则为假
    bool MapChanged();

    // Reset the system (clear map)
    //复位重启系统。。。清空地图
    void Reset();

    // All threads will be requested to finish.
    // It waits until all threads have finished.
    // This function must be called before saving the trajectory.
    //所有线程将被命令完成，他会等待直到所有线程都确认关闭
    //这个函数必须在保存轨迹之前调用
    void Shutdown();

    // Save camera trajectory in the TUM RGB-D dataset format.
    // Only for stereo and RGB-D. This method does not work for monocular.
    // Call first Shutdown()
    // See format details at: http://vision.in.tum.de/data/datasets/rgbd-dataset
    // 保存相机轨迹到TUM RGBD数据集中
    // 仅仅是RGBD和双目中使用，这个方法在单目中无效
    // 调用之前首先调用关闭程序
    // 具体详情请查看http://vision.in.tum.de/data/datasets/rgbd-dataset
    void SaveTrajectoryTUM(const string &filename);

    // Save keyframe poses in the TUM RGB-D dataset format.
    // This method works for all sensor input.
    // Call first Shutdown()
    // See format details at: http://vision.in.tum.de/data/datasets/rgbd-dataset
    // 保存关键帧的位姿到TUM RGBD数据集
    // 这种方法在所有传感器输入下都有效
    // 调用之前首先调用关闭程序
    // 具体详情请查看http://vision.in.tum.de/data/datasets/rgbd-dataset
    void SaveKeyFrameTrajectoryTUM(const string &filename);

    // Save camera trajectory in the KITTI dataset format.
    // Only for stereo and RGB-D. This method does not work for monocular.
    // Call first Shutdown()
    // See format details at: http://www.cvlibs.net/datasets/kitti/eval_odometry.php
    // 保存相机轨迹到TUM RGBD数据集中
    // 仅仅是RGBD和双目中使用，这个方法在单目中无效
    // 调用之前首先调用关闭程序
    // 具体详情请查看http://www.cvlibs.net/datasets/kitti/eval_odometry.php
    void SaveTrajectoryKITTI(const string &filename);

    // TODO: Save/Load functions
    // SaveMap(const string &filename);
    // LoadMap(const string &filename);

    // Information from most recent processed frame
    // You can call this right after TrackMonocular (or stereo or RGBD)
    // 得到最近进程帧的信息
    // 调用这个函数在TrackMonocular，TrackRGBD，TrackStereo函数之后
    int GetTrackingState();
    //得到追踪线程中的地图点
    std::vector<MapPoint*> GetTrackedMapPoints();
    //得到追踪线程中的关键点
    std::vector<cv::KeyPoint> GetTrackedKeyPointsUn();

private:

    // Input sensor
    //输入传感器
    eSensor mSensor;

    // ORB vocabulary used for place recognition and feature matching.
    //ORB字典 被用来识别和特征匹配
    ORBVocabulary* mpVocabulary;

    // KeyFrame database for place recognition (relocalization and loop detection).
    // 关键帧数据集被用来识别（重定位和回环检测）
    KeyFrameDatabase* mpKeyFrameDatabase;

    // Map structure that stores the pointers to all KeyFrames and MapPoints.
    // 地图构建，存储指针指向所有的关键帧和地图点
    Map* mpMap;

    // Tracker. It receives a frame and computes the associated camera pose.                  接受到一帧数据并且计算相机位姿之间的联系
    // It also decides when to insert a new keyframe, create some new MapPoints and   它也可以插入一个新帧，创建许多新的地图点
    // performs relocalization if tracking fails.                                                  如果数据跑丢会执行重定位
    Tracking* mpTracker;   //Tracking线程的对象

    // Local Mapper. It manages the local map and performs local bundle adjustment.
    //局部地图，管理局部地图并执行局部优化
    LocalMapping* mpLocalMapper;    //局部建图线程的对象

    // Loop Closer. It searches loops with every new keyframe. If there is a loop it performs
    // a pose graph optimization and full bundle adjustment (in a new thread) afterwards.
    //回环检测，对于每一个新的帧他都会查找是否有回环 如果存在回环他会执行一个位姿图优化并且在新的线程中会进行全局优化
    LoopClosing* mpLoopCloser;         //回环检测线程的对象

    // The viewer draws the map and the current camera pose. It uses Pangolin.
    // 这个观测器观测了地图和当前相机位姿，他使用了Pangolin库
    Viewer* mpViewer;                  //可视化线程的对象

    FrameDrawer* mpFrameDrawer;    //每一针的可视化
    MapDrawer* mpMapDrawer;           //整个地图的可视化

    // System threads: Local Mapping, Loop Closing, Viewer.
    // The Tracking thread "lives" in the main execution thread that creates the System object.
    //系统线程，局部地图，回环检测，观测器  追踪线程存在于系统对象的主线程中
    std::thread* mptLocalMapping;       //局部建图线程
    std::thread* mptLoopClosing;          //回环检测线程
    std::thread* mptViewer;                    //可视化线程

    // Reset flag
    // 复位标志位
    std::mutex mMutexReset;                 //复位锁
    bool mbReset;                                    //复位标志

    // Change mode flags
    //改变模式的模式 停止局部建图线程的标志   激活局部建图线程的标志
    std::mutex mMutexMode;               //改变模式的锁
    bool mbActivateLocalizationMode;//激活仅定位模式，暂停局部建图线程
    bool mbDeactivateLocalizationMode;//关闭仅定位模式，激活局部建图线程

    // Tracking state
    // 追踪线程运行状态
    int mTrackingState;
    //追踪线程中的地图点
    std::vector<MapPoint*> mTrackedMapPoints;
    //追踪线程中的关键点
    std::vector<cv::KeyPoint> mTrackedKeyPointsUn;
    // 线程状态锁
    std::mutex mMutexState;
};

}// namespace ORB_SLAM

#endif // SYSTEM_H
