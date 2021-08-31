#pragma once
#include "SeetaInterface.h"
#define MYLIBAPI //extern "C" __declspec( dllexport )

#define NORMAL_FEATURE_NUM 1024 //���������������1024��float��4096��char

#define MASK_FEATURE_NUM 512 //�����ֵ�����������512��float��2048��char

#define CROP_IMAG_SIZE 256 //�ü����ͼ��ߴ�

/*
���ܣ���ʼ��dll��̬�⣬ϵͳ����ʱ��ʼ��1�μ��ɣ����ʼ��ģ�͡�����ڴ�������
���룺
char* path;����ʶ��ģ��·����ʾ����"C:/MyCode/seetaface/sf3.0_models/sf3.0_models/"
int num = 4;��Ҫָ�����߳�����Ĭ��4�̣߳�4�̻߳����������߳���
int threshold=50;�������ƶ���ֵ��������ƶȴ��������ֵ��Ϊ����ͬ�ˣ����С����Ϊ�ǲ�ͬ�ˣ�Ĭ����50
50Ϊ�ٷ��ƣ���ʾ���ƶȴ���0.5��Ϊ��ͬһ����
*/
MYLIBAPI void __stdcall buptface_Init(char* path, int num = 4, int threshold = 50);

/*
���ܣ�������⣬�������λ����Ϣ
���룺ͼƬ����
int in_width;ͼƬ���
int in_height;ͼƬ�߶�
int channels;ͼƬ��ͨ����
char * data;ͼƬ����������
�������⵽������λ����Ϣ���޶�ͼƬ��ֻ����һ��������
int *out_width;����������
int *out_height;��������߶�
int *x;�����������ϽǺ�����
int *y;�����������Ͻ�������
int *score;�������ŷ�������ʾ��⵽���Ƿ�Ϊ���������ŷ���;�ٷ��ƣ�50��ʾ�������ŷ���Ϊ0.5��95��ʾ�������ŷ���Ϊ0.95
int *num;��⵽��������
return�����룬-1˵��ͼ������ָ��Ϊ�գ�0˵��û�г�ʼ��ģ�ͣ�1˵���������
*/
MYLIBAPI int __stdcall buptface_FaceDetection(int in_width, int in_height, int channels, char * data, int *out_width, int *out_height, int *x, int *y, int *score, int *num);

/*
���ܣ�����Ƿ����
���룺ͼ������(�޶�ͼ��ֻ��һ������)
int in_width;ͼƬ���
int in_height;ͼƬ�߶�
int channels;ͼƬ��ͨ����
char * data;ͼƬ����������
������Ƿ��������
bool;�Ƿ�������֣�true������֣�falseδ�������
*/
MYLIBAPI bool __stdcall buptface_DetectionMask(int in_width, int in_height, int channels, char * data);

/*
���ܣ�������ȡ
���룺ͼ�����ݺ��Ƿ��������(�޶�ͼ��ֻ��һ������)
int in_width;ͼƬ���
int in_height;ͼƬ�߶�
int channels;ͼƬ��ͨ����
char * data;ͼƬ����������
bool mask;�Ƿ�������֣���true����false
�������������
char*;ָ���������ݵ�charָ��,��floatת��Ϊchar��ÿ��float���ݱ�Ϊ4��char
*/
MYLIBAPI char* __stdcall buptface_FeatureExtraction(int in_width, int in_height, int channels, char * data, bool mask);

/*
���ܣ�����������������ݼ���/�ڴ���
���룺���ID�������������Ƿ��������
char * id;��ӵ����id
char* feature;��ӵ���������
bool mask;�Ƿ������֣�true������֣�falseδ�������
������Ƿ���ӳɹ�
bool;true��ӳɹ���false���ʧ��;��һ�����ʧ������Ϊ��ӵ�id������id�ظ���
*/
MYLIBAPI bool __stdcall buptface_Add(char * id, char* feature, bool mask);

/*
���ܣ��ȶ����ݿ����Ƿ��и�����
���룺���������������Ƿ��������
char *feature;������������
bool mask;�Ƿ�������֣�true������֣�falseδ�������
�����ƥ���Ƿ�ɹ���ƥ�䵽��id
char* id;ƥ�䵽��id
bool;ƥ���Ƿ�ɹ�,trueƥ��ɹ���falseƥ��ʧ��
*/
MYLIBAPI bool __stdcall buptface_FaceCompare(char *feature, bool mask, char* id);

/*
���ܣ�����ָ����idɾ���ڴ��е����������������ֺͲ������ֵĶ���ɾ����
���룺���ID��ѡ��ɾ��������ֻ���δ������ֵ�����
char*  id;Ҫɾ�������ID
bool mask;ѡ��ɾ��������ֻ���δ������ֵ����ݣ�true�����falseδ���
�����ɾ���Ƿ�ɹ�
bool;trueɾ���ɹ���falseɾ��ʧ��
*/
MYLIBAPI bool __stdcall buptface_Delete(char*  id, bool mask);

/*
���ܣ�����ָ����id�޸��ڴ��б������������
���룺���ID�������������Ƿ��������
char* id;Ҫ�޸�����������id
char* feature;�µ���������
bool mask;�Ƿ���������ֵ�����������true������֣�falseδ�������
������޸��Ƿ�ɹ�
bool;����޸ĳɹ�����true������޸�ʧ�ܷ���false���޸�ʧ�ܣ�һ������Ϊ�޸ĵ�id���ڴ��в����ڡ�
*/
MYLIBAPI bool __stdcall buptface_Update(char* id, char* feature, bool mask);

/*
���ܣ�����ͼƬ�ü�����������
���룺����ͼƬ
int in_width;ͼƬ���
int in_height;ͼƬ�߶�
int in_channels;ͼƬ��ͨ����
char * in_data;ͼƬ����������
������ü�������ͼƬ
int *out_width;�ü�����ͼ���ȣ�Ϊ256
int *out_height;�ü�����ͼ��߶ȣ�Ϊ256
int *out_channels;�ü�����ͼ���ͨ����
char *out_data;�ü�����ͼ�����������
*/
MYLIBAPI void __stdcall buptface_CropFace(int in_width, int in_height, int in_channels, char *in_data, int *out_width, int *out_height, int *out_channels, char *out_data);

/*
���ܣ��������dll�ڲ�ά�����ڴ�������ڲ�ʹ������ʶ��ģ�ͺ���Ե��ã����ú������ڴ棬�ٴ�ʹ��ʱ��Ҫ���µ���Init�������г�ʼ��
*/
MYLIBAPI void __stdcall buptface_End();
