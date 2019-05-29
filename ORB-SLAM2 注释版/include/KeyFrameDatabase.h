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

#ifndef KEYFRAMEDATABASE_H
#define KEYFRAMEDATABASE_H

#include <vector>
#include <list>
#include <set>

#include "KeyFrame.h"
#include "Frame.h"
#include "ORBVocabulary.h"

#include<mutex>


namespace ORB_SLAM2
{

class KeyFrame;
class Frame;


class KeyFrameDatabase
{
public:
    //构造函数
    KeyFrameDatabase(const ORBVocabulary &voc);
   //添加关键帧pKF
   void add(KeyFrame* pKF);
  //擦除关键帧pKF
   void erase(KeyFrame* pKF);
  //清空关键帧pKF
   void clear();

   // Loop Detection
   /************************************************************************
 *          功能:得到回环候选帧
 *          将所有与当前帧具有公共单词id的所有关键帧(不包括与当前关键帧链接共视的关键帧)都设为候选关键帧,然后进行筛选
 *           筛选条件:
 *                    1  根据共有单词数来筛选   筛选最大共有单词数0.8倍以上的所有关键帧为候选关键帧
 *                    2  根据候选关键帧和当前待回环关键帧之间的BOW得分来筛选候选关键帧(大于阈值minScore得分的关键帧)
 *                    3  根据候选关键帧的前10个共视关键帧的累积回环得分来筛选回环候选关键帧(大于0.75最大累积得分的所有回环候选帧,并将得分大于当
 *                               前候选关键帧的共视关键帧代替当前候选关键帧)
 **************************************************************************************/
   std::vector<KeyFrame *> DetectLoopCandidates(KeyFrame* pKF, float minScore);

/******************************************
          在所有关键帧中检测重定位候选关键帧
          候选关键帧的选择标准:
          1  首先查找与该帧存在相同词典节点的所有关键帧作为候选关键帧
          2  然后根据关键帧与待重定位帧的相同节点数来删除相同节点数小的关键帧
          3  之后计算每个关键帧的累计共视相似度得分  并且如果该关键帧的共视关键帧比该关键帧有更多的得分就用该共视关键帧代替该关键帧
			  累计共视相似度得分的计算方法:  根据两帧之间相同节点在词典中相应节点的得分来计算两帧的共视相似度
          如果共视相似度大的证明有更多的可能进行重定位
 *******************************************/
   std::vector<KeyFrame*> DetectRelocalizationCandidates(Frame* F);

protected:

  // Associated vocabulary   相关的ORB词典
  const ORBVocabulary* mpVoc;

  // Inverted file    记录每个叶子节点的反向索引（反向索引是指所有有该叶子节点的关键帧的集合）
  std::vector<list<KeyFrame*> > mvInvertedFile;

  // Mutex  线程锁
  std::mutex mMutex;
};

} //namespace ORB_SLAM

#endif
