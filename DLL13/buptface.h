#pragma once
#include "SeetaInterface.h"
#define MYLIBAPI //extern "C" __declspec( dllexport )

#define NORMAL_FEATURE_NUM 1024 //常规的特征向量是1024个float，4096个char

#define MASK_FEATURE_NUM 512 //戴口罩的特征向量是512个float，2048个char

#define CROP_IMAG_SIZE 256 //裁剪后的图像尺寸

/*
功能：初始化dll动态库，系统启动时初始化1次即可，会初始化模型、清空内存中数据
输入：
char* path;人脸识别模型路径，示例："C:/MyCode/seetaface/sf3.0_models/sf3.0_models/"
int num = 4;想要指定的线程数，默认4线程，4线程基本的最优线程数
int threshold=50;人脸相似度阈值，如果相似度大于这个阈值认为是相同人，如果小于认为是不同人，默认是50
50为百分制，表示相似度大于0.5认为是同一个人
*/
MYLIBAPI void __stdcall buptface_Init(char* path, int num = 4, int threshold = 50);

/*
功能：人脸检测，检测人脸位置信息
输入：图片数据
int in_width;图片宽度
int in_height;图片高度
int channels;图片的通道数
char * data;图片的像素数据
输出：检测到的人脸位置信息【限定图片中只能有一张人脸】
int *out_width;人脸区域宽度
int *out_height;人脸区域高度
int *x;人脸区域左上角横坐标
int *y;人脸区域左上角纵坐标
int *score;人脸置信分数，表示检测到的是否为人脸的置信分数;百分制，50表示人脸置信分数为0.5；95表示人脸置信分数为0.95
int *num;检测到人脸个数
return返回码，-1说明图像数据指针为空，0说明没有初始化模型，1说明任务完成
*/
MYLIBAPI int __stdcall buptface_FaceDetection(int in_width, int in_height, int channels, char * data, int *out_width, int *out_height, int *x, int *y, int *score, int *num);

/*
功能：检测是否佩戴
输入：图像数据(限定图像只有一张人脸)
int in_width;图片宽度
int in_height;图片高度
int channels;图片的通道数
char * data;图片的像素数据
输出：是否佩戴口罩
bool;是否佩戴口罩，true佩戴口罩，false未佩戴口罩
*/
MYLIBAPI bool __stdcall buptface_DetectionMask(int in_width, int in_height, int channels, char * data);

/*
功能：特征提取
输入：图像数据和是否佩戴口罩(限定图像只有一张人脸)
int in_width;图片宽度
int in_height;图片高度
int channels;图片的通道数
char * data;图片的像素数据
bool mask;是否佩戴口罩，是true，否false
输出：人脸特征
char*;指向特征数据的char指针,由float转化为char，每个float数据变为4个char
*/
MYLIBAPI char* __stdcall buptface_FeatureExtraction(int in_width, int in_height, int channels, char * data, bool mask);

/*
功能：添加特征向量到数据集中/内存中
输入：身份ID，特征向量，是否佩戴口罩
char * id;添加的身份id
char* feature;添加的特征向量
bool mask;是否配电口罩，true佩戴口罩，false未佩戴口罩
输出：是否添加成功
bool;true添加成功，false添加失败;【一般添加失败是因为添加的id和已有id重复】
*/
MYLIBAPI bool __stdcall buptface_Add(char * id, char* feature, bool mask);

/*
功能：比对数据库中是否含有该人脸
输入：人脸特征向量，是否佩戴口罩
char *feature;人脸特征向量
bool mask;是否佩戴口罩，true佩戴口罩，false未佩戴口罩
输出：匹配是否成功，匹配到的id
char* id;匹配到的id
bool;匹配是否成功,true匹配成功，false匹配失败
*/
MYLIBAPI bool __stdcall buptface_FaceCompare(char *feature, bool mask, char* id);

/*
功能：根据指定的id删除内存中的特征向量【戴口罩和不戴口罩的都会删除】
输入：身份ID，选择删除佩戴口罩还是未佩戴口罩的数据
char*  id;要删除的身份ID
bool mask;选择删除佩戴口罩还是未佩戴口罩的数据，true佩戴，false未佩戴
输出：删除是否成功
bool;true删除成功，false删除失败
*/
MYLIBAPI bool __stdcall buptface_Delete(char*  id, bool mask);

/*
功能：根据指定的id修改内存中保存的特征向量
输入：身份ID，特征向量，是否佩戴口罩
char* id;要修改特征向量的id
char* feature;新的特征向量
bool mask;是否是佩戴口罩的特征向量，true佩戴口罩，false未佩戴口罩
输出：修改是否成功
bool;如果修改成功返回true，如果修改失败返回false【修改失败，一般是因为修改的id在内存中不存在】
*/
MYLIBAPI bool __stdcall buptface_Update(char* id, char* feature, bool mask);

/*
功能：完整图片裁剪出人脸区域
输入：整张图片
int in_width;图片宽度
int in_height;图片高度
int in_channels;图片的通道数
char * in_data;图片的像素数据
输出：裁剪的人脸图片
int *out_width;裁剪出的图像宽度，为256
int *out_height;裁剪出的图像高度，为256
int *out_channels;裁剪出的图像的通道数
char *out_data;裁剪出的图像的像素数据
*/
MYLIBAPI void __stdcall buptface_CropFace(int in_width, int in_height, int in_channels, char *in_data, int *out_width, int *out_height, int *out_channels, char *out_data);

/*
功能：用来清除dll内部维护的内存变量，在不使用人脸识别模型后可以调用，调用后会清空内存，再次使用时需要重新调用Init函数进行初始化
*/
MYLIBAPI void __stdcall buptface_End();
