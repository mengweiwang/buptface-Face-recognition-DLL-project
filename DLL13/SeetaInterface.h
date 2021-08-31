#pragma once

#include <map>
#include <iostream>  
#include <string>//为了可以cout输出string字符串
#include <opencv2/opencv.hpp>//读取图像，和shared_ptr模板的定义
#include <stdio.h>//系统时间

#include <seeta/FaceDetector.h>//人脸位置检测
#include <seeta/MaskDetector.h>//是否佩戴口罩检测
#include <seeta/FaceLandmarker.h>//特征点检测
#include <seeta/FaceRecognizer.h>//人脸识别：特征提取，相似度计算
#include <seeta/GenderPredictor.h>//性别预测
#include <seeta/AgePredictor.h>//年龄预测
#include <seeta/FaceAntiSpoofing.h>//活体检测
#include <seeta/EyeStateDetector.h>//眼睛状态
#include <seeta/QualityOfPoseEx.h>//深度学习的姿态评估器
#include <seeta/FaceTracker.h>//人脸追踪器


#ifdef _DEBUG
//opencv库
#pragma comment(lib,"opencv_world420d.lib") 

//SeetaFace库,11个
#pragma comment(lib,"SeetaFaceDetector600d.lib") //人脸位置检测
#pragma comment(lib,"SeetaMaskDetector200d.lib")//是否佩戴口罩检测
#pragma comment(lib,"SeetaFaceLandmarker600d.lib")//特征点检测
#pragma comment(lib,"SeetaFaceRecognizer610d.lib")//人脸识别：特征提取，相似度计算
#pragma comment(lib,"SeetaGenderPredictor600d.lib") //性别预测
#pragma comment(lib,"SeetaAgePredictor600d.lib")//年龄预测
#pragma comment(lib,"SeetaFaceAntiSpoofingX600d.lib") //活体检测
#pragma comment(lib,"SeetaEyeStateDetector200d.lib")//眼睛状态
#pragma comment(lib,"SeetaPoseEstimation600d.lib")//深度学习的姿态评估器
#pragma comment(lib,"SeetaQualityAssessor300d.lib")//图片质量评估器,任何和质量相关的都需要使用这个lib库
#pragma comment(lib,"SeetaFaceTracking600d.lib")//人脸追踪器

#else
//opencv库
#pragma comment(lib,"opencv_world420.lib") 

//release 库,11个
#pragma comment(lib,"SeetaFaceDetector600.lib") 
#pragma comment(lib,"SeetaFaceLandmarker600.lib")

#pragma comment(lib,"SeetaFaceRecognizer610.lib")
#pragma comment(lib,"SeetaGenderPredictor600.lib") 
#pragma comment(lib,"SeetaAgePredictor600.lib") 
#pragma comment(lib,"SeetaFaceAntiSpoofingX600.lib") 
#pragma comment(lib,"SeetaEyeStateDetector200.lib")

//这四个没用到
#pragma comment(lib,"SeetaMaskDetector200.lib")
#pragma comment(lib,"SeetaFaceTracking600.lib") 
#pragma comment(lib,"SeetaPoseEstimation600.lib")
#pragma comment(lib,"SeetaQualityAssessor300.lib")
#endif


#define _NORMAL_FEATURE_NUM 1024
#define _MASK_FEATURE_NUM 512
using namespace std;
class SeetaInterface
{
private:
	float threshold = 0.5;
	string ModelPath = "C:/MyCode/seetaface/sf3.0_models/sf3.0_models/";//模型存放路径
	seeta::FaceDetector *FD;//不戴口罩人脸识别器指针
	seeta::MaskDetector *FD_mask;//带口罩人脸识别器指针,只用来检测是否佩戴口罩
	seeta::FaceLandmarker *FL;//不戴口罩人脸关键点定位指针
	seeta::FaceLandmarker *FL_mask;//口罩人脸关键点定位指针
	seeta::FaceRecognizer *FR;//不带口罩人脸识别模型器指针
	seeta::FaceRecognizer *FR_mask;//戴口罩人脸识别模型器指针

	//把所有的数据放入这个数据集中;<身份证号，特征指针> //非戴口罩的
	map<string, float*>   dataSet;

	//把所有的数据放入这个数据集中;<身份证号，特征指针> //戴口罩的
	map<string, float*>   dataSetMask;

	

	

	//初始化数据集，把内存中数据清空
	void InitDataSet();

	//初始化模型
	void InitModel(int num);

	/*戴口罩的特征提取*/
	float* FeatureExtraction1(SeetaImageData &simg);

	/*不带口罩的特征提取*/
	float* FeatureExtraction2(SeetaImageData &simg);

	/*戴口罩的特征点提取*/
	void FeaturePoints1(const SeetaImageData &simg, const SeetaRect &pos, SeetaPointF *points);

	/*不带口罩的特征点提取,真实情况是带不带口罩的特征点都是相同的特征点位置，可能戴口罩的特征点位置是推演出来的*/
	void FeaturePoints2(const SeetaImageData &simg,const SeetaRect &pos,SeetaPointF *points);

	/*把cv::Mat类型变成SeetaImageData类型*/
	SeetaImageData Mat2SeetaImageData(cv::Mat &matImage);

	/*戴口罩的人脸比对*/
	string FaceCompare1(float* feature);
	/*不带口罩的人脸比对*/
	string FaceCompare2(float* feature);
	

	
public:
	~SeetaInterface();//如果私有化析构函数，外部就不能对该类的对象进行delete操作
	SeetaInterface();//如果是私有函数那么就无法实例化这个类的对象，只有放在public才能够进行外部实例化
	
	/*
	功能：特征向量的转化，char2float，float2char
	*/
	void Float2Char(float *val, char *buff, int num);
	void Char2Float(char *buff, float* xval, int num);
	
	/*
	功能：初始化函数
	输入：线程数
	会初始化模型、清空内存中数据
	默认4线程，一般情况下4/8/16线程未最优
	*/
	void Init(string path,int num=4, float threshold = 0.5);

	
	/*
	功能：人脸检测，检测位置信息
	输入：cv::Mat图像
	输出：检测到的人脸位置信息(可能有多个)
	*/
	SeetaFaceInfoArray FaceDetection(SeetaImageData &simg);

	/*
	功能：特征提取
	输入：cv::Mat图像(限定图像只有一张人脸)
	输出：特征指针
	*/
	float* FeatureExtraction(SeetaImageData &simg,bool mask);

	/*
	功能：检测是否佩戴
	输入：cv::Mat图像(限定图像只有一张人脸)
	输出：是否佩戴口罩，1 佩戴，0 未佩戴
	*/
	bool DetectionMask(SeetaImageData &simg);

	/*
	功能：比对数据库中是否含有该人脸
	输入：人脸特征
	是否佩戴口罩
	输出：匹配的id，如果没有找到就返回""
	*/
	string FaceCompare(float *feature, bool mask);

	/*
	功能：添加特征到数据集
	输入：身份ID
		  特征向量，类型为shared_ptr<float>
		  是否戴口罩
	dataSet.insert(pair<string, shared_ptr<float>>(id, feature));
	*/
	bool Add(string id, float* feature,bool mask);

	/*
	功能：根据指定的id删除特征(戴口罩和不戴口罩的都会删除)
	输入：身份ID
	*/
	bool Delete(string id, bool mask);

	/*
	功能：根据指定的id删除特征(戴口罩和不戴口罩的都会删除)
	输入：身份ID
	*/
	bool Update(string id, float* feature, bool mask);

	/*
	功能：图片裁剪出人脸区域
	输入：整张图片
	输出：裁剪的人脸图片
	*/
	cv::Mat CropFace(SeetaImageData &simg);

};

