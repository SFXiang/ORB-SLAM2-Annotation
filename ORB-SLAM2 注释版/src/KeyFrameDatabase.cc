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

#include "KeyFrameDatabase.h"

#include "KeyFrame.h"
#include "Thirdparty/DBoW2/DBoW2/BowVector.h"

#include<mutex>

using namespace std;

namespace ORB_SLAM2
{

KeyFrameDatabase::KeyFrameDatabase (const ORBVocabulary &voc):
    mpVoc(&voc)
{
    mvInvertedFile.resize(voc.size());
}


void KeyFrameDatabase::add(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutex);
    //BoW向量中存储着叶子节点的编号和叶子节点的权重
    for(DBoW2::BowVector::const_iterator vit= pKF->mBowVec.begin(), vend=pKF->mBowVec.end(); vit!=vend; vit++)
        mvInvertedFile[vit->first].push_back(pKF);    //将该关键帧加入该关键帧的所有BOW向量节点下
}

void KeyFrameDatabase::erase(KeyFrame* pKF)
{
    unique_lock<mutex> lock(mMutex);

    // Erase elements in the Inverse File for the entry
    for(DBoW2::BowVector::const_iterator vit=pKF->mBowVec.begin(), vend=pKF->mBowVec.end(); vit!=vend; vit++)
    {
        // List of keyframes that share the word
        list<KeyFrame*> &lKFs =   mvInvertedFile[vit->first];

        for(list<KeyFrame*>::iterator lit=lKFs.begin(), lend= lKFs.end(); lit!=lend; lit++)
        {
            if(pKF==*lit)
            {
                lKFs.erase(lit);
                break;
            }
        }
    }
}

void KeyFrameDatabase::clear()
{
    mvInvertedFile.clear();
    mvInvertedFile.resize(mpVoc->size());
}

/************************************************************************
 *          功能: 得到回环候选帧
 *          将所有与当前帧具有公共单词id的所有关键帧(不包括与当前关键帧链接共视的关键帧)都设为候选关键帧,然后进行筛选
 *           筛选条件:
 *                    1  根据共有单词数来筛选   筛选最大共有单词数0.8倍以上的所有关键帧为候选关键帧
 *                    2  根据候选关键帧和当前待回环关键帧之间的BOW得分来筛选候选关键帧(大于阈值minScore得分的关键帧)
 *                    3  根据候选关键帧的前10个共视关键帧的累积回环得分来筛选回环候选关键帧(大于0.75最大累积得分的所有回环候选帧,并将得分大于当
 *                               前候选关键帧的共视关键帧代替当前候选关键帧)
 * 
 **************************************************************************************/
vector<KeyFrame*> KeyFrameDatabase::DetectLoopCandidates(KeyFrame* pKF, float minScore)
{
  //  与本关键帧相关联的所有关键帧(相关联是指存在15个以上的共视地图点)
    set<KeyFrame*> spConnectedKeyFrames = pKF->GetConnectedKeyFrames();
    // 所有回环候选帧
    list<KeyFrame*> lKFsSharingWords;

    // Search all keyframes that share a word with current keyframes
    // Discard keyframes connected to the query keyframe
    {
        unique_lock<mutex> lock(mMutex);

        for(DBoW2::BowVector::const_iterator vit=pKF->mBowVec.begin(), vend=pKF->mBowVec.end(); vit != vend; vit++)  //  关键帧的所有BOW向量
        { 
	  //  寻找每一BOW向量所在词典节点中所有的关键帧序列
            list<KeyFrame*> &lKFs =   mvInvertedFile[vit->first];
	  //  遍历这些关键帧  查找这些关键帧的回环关键帧是否是本关键帧,如果不是本关键帧
            for(list<KeyFrame*>::iterator lit=lKFs.begin(), lend= lKFs.end(); lit!=lend; lit++)
            {
                KeyFrame* pKFi=*lit;
                if(pKFi->mnLoopQuery!=pKF->mnId) //pKFi还没有标记为pKF的候选帧  该关键帧还没有加入lKFsSharingWords容器
                {
                    pKFi->mnLoopWords=0;
		    //  找出和当前帧具有公共单词的所有关键帧（不包括与当前帧链接的关键帧(共视关键帧)）
                    if(!spConnectedKeyFrames.count(pKFi))  // 如果关联关键帧中不存在pKFi关键帧,则将pKFi关键帧的回环id设为当前待回环关键帧
                    {
                        pKFi->mnLoopQuery=pKF->mnId;
                        lKFsSharingWords.push_back(pKFi);   // 将本关键帧加入回环列表中
                    }
                }
                pKFi->mnLoopWords++;   //这个关键帧的回环单词数+1
            }
        }
    }

    if(lKFsSharingWords.empty())
        return vector<KeyFrame*>();

    list<pair<float,KeyFrame*> > lScoreAndMatch;

    // Only compare against those keyframes that share enough words   得到所有回环候选关键帧中最大的回环单词数
    int maxCommonWords=0;
    for(list<KeyFrame*>::iterator lit=lKFsSharingWords.begin(), lend= lKFsSharingWords.end(); lit!=lend; lit++)
    {
        if((*lit)->mnLoopWords>maxCommonWords)
            maxCommonWords=(*lit)->mnLoopWords;
    }
    // 回环候选帧最小的回环单词数
    int minCommonWords = maxCommonWords*0.8f;
    
    int nscores=0;

    // Compute similarity score. Retain the matches whose score is higher than minScore
    for(list<KeyFrame*>::iterator lit=lKFsSharingWords.begin(), lend= lKFsSharingWords.end(); lit!=lend; lit++)
    {
      //  循环回环候选帧中所有的帧
        KeyFrame* pKFi = *lit;
	// 如果回环候选帧中的帧回环单词数大于最小回环单词数
        if(pKFi->mnLoopWords>minCommonWords)
        {
            nscores++;
	    // 检测待回环关键帧与当前候选回环关键帧的BOW得分
            float si = mpVoc->score(pKF->mBowVec,pKFi->mBowVec);

            pKFi->mLoopScore = si;   //回环BOW得分
            if(si>=minScore)    // 将得分小于最小BOW阈值的候选回环关键帧删除
                lScoreAndMatch.push_back(make_pair(si,pKFi));
        }
    }

    if(lScoreAndMatch.empty())
        return vector<KeyFrame*>();

    list<pair<float,KeyFrame*> > lAccScoreAndMatch;
    float bestAccScore = minScore;

    // Lets now accumulate score by covisibility
    for(list<pair<float,KeyFrame*> >::iterator it=lScoreAndMatch.begin(), itend=lScoreAndMatch.end(); it!=itend; it++)
    {
        KeyFrame* pKFi = it->second;
        vector<KeyFrame*> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10);   // 检测候选回环关键帧的前10帧共视关键帧
	// 当前回环候选帧的最高分(与回环候选帧共视帧的前10帧中与当前待回环关键帧回环得分中的最高分)
        float bestScore = it->first;
	// 当前回环关键帧的累计得分(与回环候选帧共视帧的前10帧如果也与当前帧构成回环,则将它的得分累计进来)
        float accScore = it->first;
	// 最高回环得分的关键帧
        KeyFrame* pBestKF = pKFi;
        for(vector<KeyFrame*>::iterator vit=vpNeighs.begin(), vend=vpNeighs.end(); vit!=vend; vit++)// 检测候选回环关键帧的前10帧共视关键帧
        {
            KeyFrame* pKF2 = *vit;
            if(pKF2->mnLoopQuery==pKF->mnId && pKF2->mnLoopWords>minCommonWords)
            {
                accScore+=pKF2->mLoopScore;
                if(pKF2->mLoopScore>bestScore)
                {
                    pBestKF=pKF2;
                    bestScore = pKF2->mLoopScore;
                }
            }
        }
	// 按照关键帧的累积回环得分对候选关键帧进行排序
        lAccScoreAndMatch.push_back(make_pair(accScore,pBestKF));
        if(accScore>bestAccScore)
            bestAccScore=accScore;
    }

    // Return all those keyframes with a score higher than 0.75*bestScore
    float minScoreToRetain = 0.75f*bestAccScore;

    set<KeyFrame*> spAlreadyAddedKF;
    vector<KeyFrame*> vpLoopCandidates;
    vpLoopCandidates.reserve(lAccScoreAndMatch.size());
    //  根据累计得分对其进行筛选  只取前75%的关键帧
    for(list<pair<float,KeyFrame*> >::iterator it=lAccScoreAndMatch.begin(), itend=lAccScoreAndMatch.end(); it!=itend; it++)
    {
        if(it->first>minScoreToRetain)
        {
            KeyFrame* pKFi = it->second;
            if(!spAlreadyAddedKF.count(pKFi))
            {
                vpLoopCandidates.push_back(pKFi);
                spAlreadyAddedKF.insert(pKFi);
            }
        }
    }


    return vpLoopCandidates;
}
/******************************************
          在所有关键帧中检测重定位候选关键帧
          候选关键帧的选择标准:
          1  首先查找与该帧存在相同词典节点的所有关键帧作为候选关键帧
          2  然后根据关键帧与待重定位帧的相同节点数来删除相同节点数小的关键帧
          3  之后计算每个关键帧的累计共视相似度得分  并且如果该关键帧的共视关键帧比该关键帧有更多的得分就用该共视关键帧代替该关键帧
			  累计共视相似度得分的计算方法:  根据两帧之间相同节点在词典中相应节点的得分来计算两帧的共视相似度
          如果共视相似度大的证明有更多的可能进行重定位
          重定位候选关键帧的存储容器(经过层层筛选的)
          lKFsSharingWords--->lScoreAndMatch--->lAccScoreAndMatch--->vpRelocCandidates
 *******************************************/
vector<KeyFrame*> KeyFrameDatabase::DetectRelocalizationCandidates(Frame *F)
{
  //存储与该帧数据有相同节点的关键帧容器
    list<KeyFrame*> lKFsSharingWords;

    // Search all keyframes that share a word with current frame
    {
        unique_lock<mutex> lock(mMutex);

        for(DBoW2::BowVector::const_iterator vit=F->mBowVec.begin(), vend=F->mBowVec.end(); vit != vend; vit++)  //遍历当前帧的所有BOW向量(每个特征点对应一个)
        {
            list<KeyFrame*> &lKFs =   mvInvertedFile[vit->first];  //寻找与该特征点存在于同一词典节点上的其他关键帧

            for(list<KeyFrame*>::iterator lit=lKFs.begin(), lend= lKFs.end(); lit!=lend; lit++)   //遍历这所有的关键帧
            {
                KeyFrame* pKFi=*lit;
                if(pKFi->mnRelocQuery!=F->mnId) //将该关键帧的重定位单词数量初始化
                {
                    pKFi->mnRelocWords=0;
                    pKFi->mnRelocQuery=F->mnId;
                    lKFsSharingWords.push_back(pKFi);
                }
                pKFi->mnRelocWords++;    // 该关键帧与待重定位帧存在相同字典节点的数量
            }
        }
    }
    if(lKFsSharingWords.empty())    // 如果该帧没有与之存在相同节点关键帧  那么返回空
        return vector<KeyFrame*>();

    // Only compare against those keyframes that share enough words
    // 所有候选关键帧和当前关键帧在相同节点下的特征点数量中的最大值
    int maxCommonWords=0;
    for(list<KeyFrame*>::iterator lit=lKFsSharingWords.begin(), lend= lKFsSharingWords.end(); lit!=lend; lit++)   // 寻找最大共视单词数
    {
        if((*lit)->mnRelocWords>maxCommonWords)
            maxCommonWords=(*lit)->mnRelocWords;
    }
    // 根据比例计算最小共视单词数
    int minCommonWords = maxCommonWords*0.8f;   
    // 存储的是第一次筛选后的候选关键帧以及其重定位得分(候选关键帧与待重定位关键帧的BOW得分)
    list<pair<float,KeyFrame*> > lScoreAndMatch;

    int nscores=0;

    // Compute similarity score.   计算与当前待重定位帧有相同词典节点的所有关键帧的相似度得分(根据词典的相同节点数来筛选)
    for(list<KeyFrame*>::iterator lit=lKFsSharingWords.begin(), lend= lKFsSharingWords.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;

        if(pKFi->mnRelocWords>minCommonWords)   // 如果该关键帧与待重定位帧的相同节点数大于最小相同节点数
        {
            nscores++;
	    // si存储的是在词典中待重定位帧与查询到的关键帧之间的得分
            float si = mpVoc->score(F->mBowVec,pKFi->mBowVec);
            pKFi->mRelocScore=si;   //  将关键帧的重定位得分赋值为si
            lScoreAndMatch.push_back(make_pair(si,pKFi));
        }
    }

    if(lScoreAndMatch.empty())    
        return vector<KeyFrame*>();
    //累计重定位得分(候选关键帧和当前待重定位关键帧的BOW得分) 及 最优的重定位关键帧
    list<pair<float,KeyFrame*> > lAccScoreAndMatch;
    float bestAccScore = 0;

    // Lets now accumulate score by covisibility   通过共视相似度来获得得分
    for(list<pair<float,KeyFrame*> >::iterator it=lScoreAndMatch.begin(), itend=lScoreAndMatch.end(); it!=itend; it++)  //遍历lScoreAndMatch中所有关键帧
    {
        KeyFrame* pKFi = it->second;
	// 存储与改关键帧中共视程度最好的10帧关键帧
        vector<KeyFrame*> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10);

	//存储关键帧的最优得分
        float bestScore = it->first;
	// 所有共视关键帧的累计重定位得分
        float accScore = bestScore;
	// 存储最优重定位关键帧
        KeyFrame* pBestKF = pKFi;
        for(vector<KeyFrame*>::iterator vit=vpNeighs.begin(), vend=vpNeighs.end(); vit!=vend; vit++)  //遍历这些共视关键帧
        {
            KeyFrame* pKF2 = *vit;
            if(pKF2->mnRelocQuery!=F->mnId)   //如果它的待重定位帧与该帧不同,则跳过该帧
                continue;

            accScore+=pKF2->mRelocScore;   //如果相同  则将共视关键帧的得分也加到本次重定位得分中
            if(pKF2->mRelocScore>bestScore)  // 如果共视关键帧的重定位得分大于当前关键帧的重定位得分
            {
                pBestKF=pKF2;    //最优重定位关键帧
                bestScore = pKF2->mRelocScore;    // 则将当前关键帧的最优重定位得分赋值为共视关键帧的重定位得分
            }

        }
        lAccScoreAndMatch.push_back(make_pair(accScore,pBestKF));
        if(accScore>bestAccScore)
            bestAccScore=accScore;
    }

    // Return all those keyframes with a score higher than 0.75*bestScore    返回所有重定位累计得分大于0.75倍最优重定位累计得分的所有关键帧
    float minScoreToRetain = 0.75f*bestAccScore;
    // 存储所有已经插入vpRelocCandidates的关键帧
    set<KeyFrame*> spAlreadyAddedKF;
    // 重定位候选帧
    vector<KeyFrame*> vpRelocCandidates;   
    vpRelocCandidates.reserve(lAccScoreAndMatch.size());
    for(list<pair<float,KeyFrame*> >::iterator it=lAccScoreAndMatch.begin(), itend=lAccScoreAndMatch.end(); it!=itend; it++)
    {
        const float &si = it->first;
        if(si>minScoreToRetain)  // 如果该关键帧的累计重定位得分大于最优累计关键帧重定位得分的0.75倍
        {
            KeyFrame* pKFi = it->second;
            if(!spAlreadyAddedKF.count(pKFi))  // 如果没有插入过该关键帧则插入该关键帧
            {
                vpRelocCandidates.push_back(pKFi);
                spAlreadyAddedKF.insert(pKFi);
            }
        }
    }

    return vpRelocCandidates;
}

} //namespace ORB_SLAM
