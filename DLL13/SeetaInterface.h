#pragma once

#include <map>
#include <iostream>  
#include <string>//Ϊ�˿���cout���string�ַ���
#include <opencv2/opencv.hpp>//��ȡͼ�񣬺�shared_ptrģ��Ķ���
#include <stdio.h>//ϵͳʱ��

#include <seeta/FaceDetector.h>//����λ�ü��
#include <seeta/MaskDetector.h>//�Ƿ�������ּ��
#include <seeta/FaceLandmarker.h>//��������
#include <seeta/FaceRecognizer.h>//����ʶ��������ȡ�����ƶȼ���
#include <seeta/GenderPredictor.h>//�Ա�Ԥ��
#include <seeta/AgePredictor.h>//����Ԥ��
#include <seeta/FaceAntiSpoofing.h>//������
#include <seeta/EyeStateDetector.h>//�۾�״̬
#include <seeta/QualityOfPoseEx.h>//���ѧϰ����̬������
#include <seeta/FaceTracker.h>//����׷����


#ifdef _DEBUG
//opencv��
#pragma comment(lib,"opencv_world420d.lib") 

//SeetaFace��,11��
#pragma comment(lib,"SeetaFaceDetector600d.lib") //����λ�ü��
#pragma comment(lib,"SeetaMaskDetector200d.lib")//�Ƿ�������ּ��
#pragma comment(lib,"SeetaFaceLandmarker600d.lib")//��������
#pragma comment(lib,"SeetaFaceRecognizer610d.lib")//����ʶ��������ȡ�����ƶȼ���
#pragma comment(lib,"SeetaGenderPredictor600d.lib") //�Ա�Ԥ��
#pragma comment(lib,"SeetaAgePredictor600d.lib")//����Ԥ��
#pragma comment(lib,"SeetaFaceAntiSpoofingX600d.lib") //������
#pragma comment(lib,"SeetaEyeStateDetector200d.lib")//�۾�״̬
#pragma comment(lib,"SeetaPoseEstimation600d.lib")//���ѧϰ����̬������
#pragma comment(lib,"SeetaQualityAssessor300d.lib")//ͼƬ����������,�κκ�������صĶ���Ҫʹ�����lib��
#pragma comment(lib,"SeetaFaceTracking600d.lib")//����׷����

#else
//opencv��
#pragma comment(lib,"opencv_world420.lib") 

//release ��,11��
#pragma comment(lib,"SeetaFaceDetector600.lib") 
#pragma comment(lib,"SeetaFaceLandmarker600.lib")

#pragma comment(lib,"SeetaFaceRecognizer610.lib")
#pragma comment(lib,"SeetaGenderPredictor600.lib") 
#pragma comment(lib,"SeetaAgePredictor600.lib") 
#pragma comment(lib,"SeetaFaceAntiSpoofingX600.lib") 
#pragma comment(lib,"SeetaEyeStateDetector200.lib")

//���ĸ�û�õ�
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
	string ModelPath = "C:/MyCode/seetaface/sf3.0_models/sf3.0_models/";//ģ�ʹ��·��
	seeta::FaceDetector *FD;//������������ʶ����ָ��
	seeta::MaskDetector *FD_mask;//����������ʶ����ָ��,ֻ��������Ƿ��������
	seeta::FaceLandmarker *FL;//�������������ؼ��㶨λָ��
	seeta::FaceLandmarker *FL_mask;//���������ؼ��㶨λָ��
	seeta::FaceRecognizer *FR;//������������ʶ��ģ����ָ��
	seeta::FaceRecognizer *FR_mask;//����������ʶ��ģ����ָ��

	//�����е����ݷ���������ݼ���;<���֤�ţ�����ָ��> //�Ǵ����ֵ�
	map<string, float*>   dataSet;

	//�����е����ݷ���������ݼ���;<���֤�ţ�����ָ��> //�����ֵ�
	map<string, float*>   dataSetMask;

	

	

	//��ʼ�����ݼ������ڴ����������
	void InitDataSet();

	//��ʼ��ģ��
	void InitModel(int num);

	/*�����ֵ�������ȡ*/
	float* FeatureExtraction1(SeetaImageData &simg);

	/*�������ֵ�������ȡ*/
	float* FeatureExtraction2(SeetaImageData &simg);

	/*�����ֵ���������ȡ*/
	void FeaturePoints1(const SeetaImageData &simg, const SeetaRect &pos, SeetaPointF *points);

	/*�������ֵ���������ȡ,��ʵ����Ǵ��������ֵ������㶼����ͬ��������λ�ã����ܴ����ֵ�������λ�������ݳ�����*/
	void FeaturePoints2(const SeetaImageData &simg,const SeetaRect &pos,SeetaPointF *points);

	/*��cv::Mat���ͱ��SeetaImageData����*/
	SeetaImageData Mat2SeetaImageData(cv::Mat &matImage);

	/*�����ֵ������ȶ�*/
	string FaceCompare1(float* feature);
	/*�������ֵ������ȶ�*/
	string FaceCompare2(float* feature);
	

	
public:
	~SeetaInterface();//���˽�л������������ⲿ�Ͳ��ܶԸ���Ķ������delete����
	SeetaInterface();//�����˽�к�����ô���޷�ʵ���������Ķ���ֻ�з���public���ܹ������ⲿʵ����
	
	/*
	���ܣ�����������ת����char2float��float2char
	*/
	void Float2Char(float *val, char *buff, int num);
	void Char2Float(char *buff, float* xval, int num);
	
	/*
	���ܣ���ʼ������
	���룺�߳���
	���ʼ��ģ�͡�����ڴ�������
	Ĭ��4�̣߳�һ�������4/8/16�߳�δ����
	*/
	void Init(string path,int num=4, float threshold = 0.5);

	
	/*
	���ܣ�������⣬���λ����Ϣ
	���룺cv::Matͼ��
	�������⵽������λ����Ϣ(�����ж��)
	*/
	SeetaFaceInfoArray FaceDetection(SeetaImageData &simg);

	/*
	���ܣ�������ȡ
	���룺cv::Matͼ��(�޶�ͼ��ֻ��һ������)
	���������ָ��
	*/
	float* FeatureExtraction(SeetaImageData &simg,bool mask);

	/*
	���ܣ�����Ƿ����
	���룺cv::Matͼ��(�޶�ͼ��ֻ��һ������)
	������Ƿ�������֣�1 �����0 δ���
	*/
	bool DetectionMask(SeetaImageData &simg);

	/*
	���ܣ��ȶ����ݿ����Ƿ��и�����
	���룺��������
	�Ƿ��������
	�����ƥ���id�����û���ҵ��ͷ���""
	*/
	string FaceCompare(float *feature, bool mask);

	/*
	���ܣ�������������ݼ�
	���룺���ID
		  ��������������Ϊshared_ptr<float>
		  �Ƿ������
	dataSet.insert(pair<string, shared_ptr<float>>(id, feature));
	*/
	bool Add(string id, float* feature,bool mask);

	/*
	���ܣ�����ָ����idɾ������(�����ֺͲ������ֵĶ���ɾ��)
	���룺���ID
	*/
	bool Delete(string id, bool mask);

	/*
	���ܣ�����ָ����idɾ������(�����ֺͲ������ֵĶ���ɾ��)
	���룺���ID
	*/
	bool Update(string id, float* feature, bool mask);

	/*
	���ܣ�ͼƬ�ü�����������
	���룺����ͼƬ
	������ü�������ͼƬ
	*/
	cv::Mat CropFace(SeetaImageData &simg);

};

