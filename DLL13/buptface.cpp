#include "buptface.h"

SeetaInterface *INSTANCE = NULL;
void __stdcall buptface_Init(char * path, int num, int threshold)
{
	//cout << "开始初始化系统..." << endl;
	INSTANCE = new SeetaInterface();
	float _threshold = (float)(threshold*1.0 / 100);
	if (_threshold >= 1 || _threshold < 0) {
		return;
	}
	if (path == NULL) {
		return;
	}
	INSTANCE->Init(path, num, _threshold);
}

int __stdcall buptface_FaceDetection(int in_width, int in_height, int channels, char * data, int *out_width, int *out_height, int *x, int *y, int *score, int* num)
{
	if (data == NULL) {
		return -1;
	}
	//cout << "进入FaceDetection" << endl;

	//if (data == NULL) {
	//	cout << "data == NULL" << endl;
	//}
	//else {
	//	cout << "data != NULL" << endl;
	//}
	if (INSTANCE == NULL) {
		//没有初始化
		/*cout << "没有初始化模型" << endl;*/
		return 0;
	}
	unsigned char* u_data;
	u_data = reinterpret_cast<unsigned char*> (data);

	SeetaImageData simg;
	simg.height = in_height;
	simg.width = in_width;
	simg.channels = channels;
	simg.data = u_data;
	SeetaFaceInfoArray faceInfoArray = INSTANCE->FaceDetection(simg);
	if (faceInfoArray.data != NULL) {
		if (out_height != NULL) {
			*(out_height) = faceInfoArray.data[0].pos.height;
		}
		if (out_width != NULL) {
			*(out_width) = faceInfoArray.data[0].pos.width;
		}
		if (x != NULL) {
			*(x) = faceInfoArray.data[0].pos.x;
		}
		if (y != NULL) {
			*(y) = faceInfoArray.data[0].pos.y;
		}
		if (score != NULL) {
			*(score) = (int)((faceInfoArray.data[0].score) * 100);
		}
	}

	////cout << "转换前置信分数："<< faceInfoArray.data[0].score << endl;
	//float _score = faceInfoArray.data[0].score;
	//*(score) = (int)(_score*100);
	////cout << "转换后置信分数：" << *(score) << endl;
	if (num != NULL) {
		*(num) = faceInfoArray.size;
	}

	//return ((float)(_score * 100));
	////输出检测到的人脸位置信息
	//for (int i = 0; i < faceInfoArray.size; i++) {//每个人脸的位置
	//	cout << "人脸位置：" << endl;
	//	cout << "人脸区域左上角横坐标:" << faceInfoArray.data[i].pos.x << endl;
	//	cout << "人脸区域左上角纵坐标:" << faceInfoArray.data[i].pos.y << endl;
	//	cout << "人脸区域宽度:" << faceInfoArray.data[i].pos.width << endl;
	//	cout << "人脸区域高度:" << faceInfoArray.data[i].pos.height << endl;
	//	cout << "人脸置信分数" << faceInfoArray.data[i].score << endl;
	//	cout << endl;
	//}
	////输出检测到的人脸位置信息
	//for (int i = 0; i < (*num); i++) {//每个人脸的位置
	//	cout << "人脸位置：" << endl;
	//	cout << "人脸区域左上角横坐标:" << *x << endl;
	//	cout << "人脸区域左上角纵坐标:" << *y << endl;
	//	cout << "人脸区域宽度:" << *out_width << endl;
	//	cout << "人脸区域高度:" << *out_height << endl;
	//	cout << "人脸置信分数" << *score << endl;
	//	cout << endl;
	//}
	return 1;
}

bool __stdcall buptface_DetectionMask(int in_width, int in_height, int channels, char * data)
{
	if (INSTANCE == NULL) {
		//没有初始化
		/*cout << "没有初始化模型" << endl;*/
		return false;
	}
	if (data == NULL) {
		return false;
	}
	unsigned char* u_data;
	u_data = reinterpret_cast<unsigned char*> (data);

	SeetaImageData simg;
	simg.height = in_height;
	simg.width = in_width;
	simg.channels = channels;
	simg.data = u_data;

	return INSTANCE->DetectionMask(simg);

}

char * __stdcall buptface_FeatureExtraction(int in_width, int in_height, int channels, char * data, bool mask)
{
	if (INSTANCE == NULL) {
		//没有初始化
		/*cout << "没有初始化模型" << endl;*/
		return NULL;
	}
	if (data == NULL) {
		return NULL;
	}
	unsigned char* u_data;
	u_data = reinterpret_cast<unsigned char*> (data);

	SeetaImageData simg;
	simg.height = in_height;
	simg.width = in_width;
	simg.channels = channels;
	simg.data = u_data;
	float *f = INSTANCE->FeatureExtraction(simg, mask);
	//申请内存
	char * c;
	if (mask) {
		c = new char[4 * MASK_FEATURE_NUM];
	}
	else {
		c = new char[4 * NORMAL_FEATURE_NUM];
	}
	//把float变成char
	if (mask) {
		INSTANCE->Float2Char(f, c, MASK_FEATURE_NUM);
	}
	else {
		INSTANCE->Float2Char(f, c, NORMAL_FEATURE_NUM);
	}
	//释放float内存
	delete f;
	//返回char
	return c;
}

bool __stdcall buptface_Add(char * id, char * feature, bool mask)
{
	if (id == NULL) {
		return false;
	}
	if (feature == NULL) {
		return false;
	}

	if (INSTANCE == NULL) {
		//没有初始化
		/*cout << "没有初始化模型" << endl;*/
		return false;
	}
	/*if (feature == NULL) {
	cout << "feature特征数据为NULL" << endl;
	}*/
	//申请float空间
	float *f;
	if (mask) {
		f = new float[MASK_FEATURE_NUM];
	}
	else {
		f = new float[NORMAL_FEATURE_NUM];
	}
	//把char变成float
	if (mask) {
		INSTANCE->Char2Float(feature, f, MASK_FEATURE_NUM);
	}
	else {
		INSTANCE->Char2Float(feature, f, NORMAL_FEATURE_NUM);
	}

	return INSTANCE->Add(id, f, mask);
}

bool __stdcall buptface_FaceCompare(char * feature, bool mask, char * id)
{
	if (feature == NULL) {
		return false;
	}
	if (INSTANCE == NULL) {
		//没有初始化
		/*cout << "没有初始化模型" << endl;*/
		return false;
	}
	//申请float空间
	float *f;
	if (mask) {
		f = new float[MASK_FEATURE_NUM];
	}
	else {
		f = new float[NORMAL_FEATURE_NUM];
	}
	//把char变成float
	if (mask) {
		INSTANCE->Char2Float(feature, f, MASK_FEATURE_NUM);
	}
	else {
		INSTANCE->Char2Float(feature, f, NORMAL_FEATURE_NUM);
	}
	string str = INSTANCE->FaceCompare(f, mask);
	//释放float空间
	delete f;
	// 把string变成char
	if (str == "") {
		return false;
	}
	else {
		int i;
		for (i = 0; i < str.length(); ++i)
		{
			id[i] = str[i];
		}

		id[i] = '\0'; //这一步比较重要
		return true;
	}
	//string str = INSTANCE->FaceCompare(feature, mask);
	//cout << str << endl;
	//const char *p = str.data(); //加const  或用 char *p = (char*)str.data(); 的形式
	//cout <<"1"<< *p<<"2" << endl;
	//return p;
	/*const char *p = INSTANCE->FaceCompare(feature, mask).c_str();
	cout << "1" << *p << "2" << endl;
	return p;*/
}

bool __stdcall buptface_Delete(char * id, bool mask)
{
	if (id == NULL) {
		return false;
	}
	if (INSTANCE == NULL) {
		//没有初始化
		/*cout << "没有初始化模型" << endl;*/
		return false;
	}
	string str;
	str = id;
	return INSTANCE->Delete(str, mask);
}

bool __stdcall buptface_Update(char * id, char * feature, bool mask)
{
	if (id == NULL || feature == NULL) {
		return false;
	}
	if (INSTANCE == NULL) {
		//没有初始化
		/*cout << "没有初始化模型" << endl;*/
		return false;
	}
	//申请float空间
	float *f;
	if (mask) {
		f = new float[MASK_FEATURE_NUM];
	}
	else {
		f = new float[NORMAL_FEATURE_NUM];
	}
	//把char转化float
	if (mask) {
		INSTANCE->Char2Float(feature, f, MASK_FEATURE_NUM);
	}
	else {
		INSTANCE->Char2Float(feature, f, NORMAL_FEATURE_NUM);
	}
	string str;
	str = id;
	return INSTANCE->Update(str, f, mask);
}

void __stdcall buptface_CropFace(int in_width, int in_height, int in_channels, char * in_data, int * out_width, int * out_height, int * out_channels, char * out_data)
{
	if (in_data == NULL) {
		return;
	}
	if (INSTANCE == NULL) {
		//没有初始化
		/*cout << "没有初始化模型" << endl;*/
		return;
	}
	unsigned char* u_data;
	u_data = reinterpret_cast<unsigned char*> (in_data);

	SeetaImageData simg;
	simg.height = in_height;
	simg.width = in_width;
	simg.channels = in_channels;
	simg.data = u_data;

	cv::Mat simg2 = INSTANCE->CropFace(simg);
	*out_channels = simg2.channels();
	*out_width = simg2.cols;
	*out_height = simg2.rows;

	char * t_data = reinterpret_cast<char*>(simg2.data);
	for (long i = 0; i < (*out_width)*(*out_height)**out_channels; i++) {
		out_data[i] = t_data[i];
	}
	//out_data = simg2.data;


	//cv::Mat image(256, 256, CV_8UC(3), (simg2.data));
	/*cv::imshow("1", simg2);
	cv::waitKey();*/
	//cv::imwrite("C:/MyCode/seetaface/test_seetaface_models/images/2_2crop.jpg", image);

	//*out_width = 256;
	//*out_height = 256;
	//*out_channels = in_channels;

	//char* _data;
	//_data = reinterpret_cast<char*> (simg2.data);


	////数据转储
	//for (int i = 0; i < (*out_width)*(*out_height); i++) {
	//	out_data[i] = _data[i];
	//}
	/*cv::Mat image(256, 256, 16, (simg2.data));
	cv::imwrite("C:/MyCode/seetaface/test_seetaface_models/images/2_2crop.jpg", image);*/
}

void __stdcall buptface_End()
{
	//释放SeetaInterface对象
	delete INSTANCE;
	INSTANCE = NULL;
}
