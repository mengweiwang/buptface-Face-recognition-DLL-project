#include "buptface.h"

SeetaInterface *INSTANCE = NULL;
void __stdcall buptface_Init(char * path, int num, int threshold)
{
	//cout << "��ʼ��ʼ��ϵͳ..." << endl;
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
	//cout << "����FaceDetection" << endl;

	//if (data == NULL) {
	//	cout << "data == NULL" << endl;
	//}
	//else {
	//	cout << "data != NULL" << endl;
	//}
	if (INSTANCE == NULL) {
		//û�г�ʼ��
		/*cout << "û�г�ʼ��ģ��" << endl;*/
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

	////cout << "ת��ǰ���ŷ�����"<< faceInfoArray.data[0].score << endl;
	//float _score = faceInfoArray.data[0].score;
	//*(score) = (int)(_score*100);
	////cout << "ת�������ŷ�����" << *(score) << endl;
	if (num != NULL) {
		*(num) = faceInfoArray.size;
	}

	//return ((float)(_score * 100));
	////�����⵽������λ����Ϣ
	//for (int i = 0; i < faceInfoArray.size; i++) {//ÿ��������λ��
	//	cout << "����λ�ã�" << endl;
	//	cout << "�����������ϽǺ�����:" << faceInfoArray.data[i].pos.x << endl;
	//	cout << "�����������Ͻ�������:" << faceInfoArray.data[i].pos.y << endl;
	//	cout << "����������:" << faceInfoArray.data[i].pos.width << endl;
	//	cout << "��������߶�:" << faceInfoArray.data[i].pos.height << endl;
	//	cout << "�������ŷ���" << faceInfoArray.data[i].score << endl;
	//	cout << endl;
	//}
	////�����⵽������λ����Ϣ
	//for (int i = 0; i < (*num); i++) {//ÿ��������λ��
	//	cout << "����λ�ã�" << endl;
	//	cout << "�����������ϽǺ�����:" << *x << endl;
	//	cout << "�����������Ͻ�������:" << *y << endl;
	//	cout << "����������:" << *out_width << endl;
	//	cout << "��������߶�:" << *out_height << endl;
	//	cout << "�������ŷ���" << *score << endl;
	//	cout << endl;
	//}
	return 1;
}

bool __stdcall buptface_DetectionMask(int in_width, int in_height, int channels, char * data)
{
	if (INSTANCE == NULL) {
		//û�г�ʼ��
		/*cout << "û�г�ʼ��ģ��" << endl;*/
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
		//û�г�ʼ��
		/*cout << "û�г�ʼ��ģ��" << endl;*/
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
	//�����ڴ�
	char * c;
	if (mask) {
		c = new char[4 * MASK_FEATURE_NUM];
	}
	else {
		c = new char[4 * NORMAL_FEATURE_NUM];
	}
	//��float���char
	if (mask) {
		INSTANCE->Float2Char(f, c, MASK_FEATURE_NUM);
	}
	else {
		INSTANCE->Float2Char(f, c, NORMAL_FEATURE_NUM);
	}
	//�ͷ�float�ڴ�
	delete f;
	//����char
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
		//û�г�ʼ��
		/*cout << "û�г�ʼ��ģ��" << endl;*/
		return false;
	}
	/*if (feature == NULL) {
	cout << "feature��������ΪNULL" << endl;
	}*/
	//����float�ռ�
	float *f;
	if (mask) {
		f = new float[MASK_FEATURE_NUM];
	}
	else {
		f = new float[NORMAL_FEATURE_NUM];
	}
	//��char���float
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
		//û�г�ʼ��
		/*cout << "û�г�ʼ��ģ��" << endl;*/
		return false;
	}
	//����float�ռ�
	float *f;
	if (mask) {
		f = new float[MASK_FEATURE_NUM];
	}
	else {
		f = new float[NORMAL_FEATURE_NUM];
	}
	//��char���float
	if (mask) {
		INSTANCE->Char2Float(feature, f, MASK_FEATURE_NUM);
	}
	else {
		INSTANCE->Char2Float(feature, f, NORMAL_FEATURE_NUM);
	}
	string str = INSTANCE->FaceCompare(f, mask);
	//�ͷ�float�ռ�
	delete f;
	// ��string���char
	if (str == "") {
		return false;
	}
	else {
		int i;
		for (i = 0; i < str.length(); ++i)
		{
			id[i] = str[i];
		}

		id[i] = '\0'; //��һ���Ƚ���Ҫ
		return true;
	}
	//string str = INSTANCE->FaceCompare(feature, mask);
	//cout << str << endl;
	//const char *p = str.data(); //��const  ���� char *p = (char*)str.data(); ����ʽ
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
		//û�г�ʼ��
		/*cout << "û�г�ʼ��ģ��" << endl;*/
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
		//û�г�ʼ��
		/*cout << "û�г�ʼ��ģ��" << endl;*/
		return false;
	}
	//����float�ռ�
	float *f;
	if (mask) {
		f = new float[MASK_FEATURE_NUM];
	}
	else {
		f = new float[NORMAL_FEATURE_NUM];
	}
	//��charת��float
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
		//û�г�ʼ��
		/*cout << "û�г�ʼ��ģ��" << endl;*/
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


	////����ת��
	//for (int i = 0; i < (*out_width)*(*out_height); i++) {
	//	out_data[i] = _data[i];
	//}
	/*cv::Mat image(256, 256, 16, (simg2.data));
	cv::imwrite("C:/MyCode/seetaface/test_seetaface_models/images/2_2crop.jpg", image);*/
}

void __stdcall buptface_End()
{
	//�ͷ�SeetaInterface����
	delete INSTANCE;
	INSTANCE = NULL;
}
