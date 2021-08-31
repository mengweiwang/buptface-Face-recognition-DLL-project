
#include "SeetaInterface.h"

//
//map<string, float*>   SeetaInterface::dataSet;//��.h�ж���ľ�̬������Ҫ��������г�ʼ�������ò��ܹ�ʹ��
//map<string, float*>   SeetaInterface::dataSetMask;//��.h�ж���ľ�̬������Ҫ��������г�ʼ�������ò��ܹ�ʹ��



//��ʼ�����ݼ������ڴ����������
void SeetaInterface::InitDataSet() {
	dataSet.clear();//�ڱ�����ʹ�ò���Ҫ���SeetaInterface::�����ڣ�����ֱ��ʹ��
}


//��ʼ��ģ��
void SeetaInterface::InitModel(int num) {
	//���������������ģ�ͳ�ʼ��
	seeta::ModelSetting FD_setting;
	FD_setting.append(ModelPath + "face_detector.csta");
	FD_setting.set_device(seeta::ModelSetting::CPU);
	FD_setting.set_id(0);
	FD = new seeta::FaceDetector(FD_setting);
	FD->set(seeta::v6::FaceDetector::PROPERTY_NUMBER_THREADS, num);

	//�����������ģ�ͳ�ʼ��
	seeta::ModelSetting FD_setting_mask;
	FD_setting_mask.append(ModelPath + "mask_detector.csta");
	FD_setting_mask.set_device(seeta::ModelSetting::CPU);
	FD_setting_mask.set_id(0);
	FD_mask = new seeta::MaskDetector(FD_setting_mask);



	//�������������ؼ���ģ�ͳ�ʼ��
	seeta::ModelSetting PD_setting;
	PD_setting.append(ModelPath + "face_landmarker_pts5.csta");
	FL = new seeta::FaceLandmarker(PD_setting);

	//�����������ؼ���ģ�ͳ�ʼ��
	seeta::ModelSetting PD_setting_mask;
	PD_setting_mask.append(ModelPath + "face_landmarker_mask_pts5.csta");
	FL_mask = new seeta::FaceLandmarker(PD_setting_mask);

	//������������ʶ��ģ�ͳ�ʼ��
	seeta::ModelSetting fr_setting;
	fr_setting.append(ModelPath + "face_recognizer.csta");
	FR = new seeta::FaceRecognizer(fr_setting);
	FR->set(seeta::v6::FaceRecognizer::PROPERTY_NUMBER_THREADS, num);


	//����������ʶ��ģ�ͳ�ʼ��
	seeta::ModelSetting fr_setting_mask;
	fr_setting_mask.append(ModelPath + "face_recognizer_mask.csta");
	FR_mask = new seeta::FaceRecognizer(fr_setting_mask);
	FR_mask->set(seeta::v6::FaceRecognizer::PROPERTY_NUMBER_THREADS, num);


}

/*
���ܣ���ʼ������
���룺�߳���
���ʼ��ģ�͡�����ڴ�������
Ĭ��4�̣߳�һ�������4/8/16�߳�δ����
*/
void SeetaInterface::Init(string path, int num, float _threshold) {
	ModelPath = path;
	InitDataSet();
	InitModel(num);
	threshold = _threshold;
}

/*
������⺯��
���룺cv::Matͼ��
�������⵽������λ����Ϣ(�����ж��)
*/
SeetaFaceInfoArray SeetaInterface::FaceDetection(SeetaImageData &simg) {
	//SeetaImageData simg;
	//simg.height = matImage.rows;
	//simg.width = matImage.cols;
	//simg.channels = matImage.channels();
	//simg.data = matImage.data;

	return FD->detect(simg);//���ص���Ϣ�ǣ�����λ�ú����ŷ���

}


/*
���ܣ�����Ƿ����
���룺cv::Matͼ��(�޶�ͼ��ֻ��һ������)
������Ƿ�������֣�1 �����0 δ���
*/
bool SeetaInterface::DetectionMask(SeetaImageData &simg) {
	//������⣬���λ����Ϣ
	auto faces = FaceDetection(simg);//���ص���Ϣ�ǣ�����λ�ú����ŷ���
										 //if (faces.size <= 0) {
										 //	cout << "û�м�⵽����" << endl;
										 //}
										 //else
										 //{
										 //	cout << "��⵽��������:" << faces.size << endl << endl;
										 //}

										 //����Ƿ������

	float score;//�Ƿ�����ֵ����ŷ���
	bool mask = FD_mask->detect(simg, faces.data[0].pos, &score);
	//if (mask) {
	//	cout << "�����ֵ����Ŷ�:" << score << endl;
	//	cout << "����˿���" << endl;
	//}
	//else
	//{
	//	cout << "�����ֵ����Ŷ�:" << score << endl;
	//	cout << "δ�������" << endl;
	//}
	return mask;
}



SeetaInterface::SeetaInterface()
{

}


SeetaInterface::~SeetaInterface()
{
	//�ͷ�map�е��ڴ�
	for (auto it = dataSetMask.begin(); it != dataSetMask.end(); it = dataSetMask.begin()) {
		float* f = it->second;
		//cout << "dataSetMask size:" <<dataSetMask.size()<< endl;
		//cout << "dataSetMaskɾ��" << endl;
		//cout << "dataSetMaskɾ��" << it->first << endl;
		dataSetMask.erase(it);
		delete f;
	}
	for (auto it = dataSet.begin(); it != dataSet.end(); it = dataSet.begin()) {
		float* f = it->second;
		//cout << "dataSetMaskɾ��" << it->first << endl;
		dataSet.erase(it);
		delete f;
	}

	//�ͷ�ģ���ڴ�
	delete FD;
	delete FD_mask;
	delete FL;
	delete FL_mask;
	delete FR;
	delete FR_mask;
}

/*�������ֵ���������ȡ*/
void SeetaInterface::FeaturePoints2(const SeetaImageData &simg, const SeetaRect &pos, SeetaPointF *points) {
	FL->mark(simg, pos, points);
	//cout << "�������ֵ���������ȡ" << endl;
	//for (int i = 0; i < 5; i++) {//ÿ��������������
	//	cout << "����������" << i + 1 << "��λ��:" << endl;
	//	cout << "���������������:" << points[i].x << endl;
	//	cout << "����������������:" << points[i].y << endl;
	//	cout << endl;
	//}
}

/*�����ֵ���������ȡ*/
void SeetaInterface::FeaturePoints1(const SeetaImageData &simg, const SeetaRect &pos, SeetaPointF *points) {
	FL_mask->mark(simg, pos, points);
	//cout <<"�����ֵ���������ȡ" <<endl;
	//for (int i = 0; i < 5; i++) {//ÿ��������������
	//	cout << "����������" << i + 1 << "��λ��:" << endl;
	//	cout << "���������������:" << points[i].x << endl;
	//	cout << "����������������:" << points[i].y << endl;
	//	cout << endl;
	//}
}
/*�����ֵ�������ȡ*/
float* SeetaInterface::FeatureExtraction1(SeetaImageData &simg) {
	// ����λ�ü��
	auto faces = FaceDetection(simg);//���ص���Ϣ�ǣ�����λ�ú����ŷ���

	// ��������ȡ
	SeetaPointF points[5];
	FeaturePoints1(simg, faces.data[0].pos, points);
	//������ȡ
	float *features = new float[_MASK_FEATURE_NUM];
	memset(features, 0, _MASK_FEATURE_NUM * sizeof(float));
	//shared_ptr<float> feature(new float[FR_mask->GetExtractFeatureSize()]);//����һ��float�������鳤�ȵ��ڴ�ռ䣬һ�����������ȡ����������
	FR_mask->Extract(simg, points, features);//feature1.get()������ָ���ﷵ��һ������ָ�룬���ܹ�delete�������ָ��
	/*cout << "��������float����" << endl;
	for (int i = 0; i < 50; i++) {
		cout << "features[" << i << "0]" << features[i * 10] << endl;;
	}*/
	return  features;
}
/*�������ֵ�������ȡ*/
float* SeetaInterface::FeatureExtraction2(SeetaImageData &simg) {
	// ����λ�ü��
	auto faces = FaceDetection(simg);//���ص���Ϣ�ǣ�����λ�ú����ŷ���

	// ��������ȡ
	SeetaPointF points[5];
	FeaturePoints2(simg, faces.data[0].pos, points);

	//������ȡ
	float *features=new float[_NORMAL_FEATURE_NUM];
	memset(features, 0, _NORMAL_FEATURE_NUM * sizeof(float));
	//shared_ptr<float> feature(new float[FR->GetExtractFeatureSize()]);//����һ��float�������鳤�ȵ��ڴ�ռ䣬һ�����������ȡ����������
	FR->Extract(simg, points, features);//feature1.get()������ָ���ﷵ��һ������ָ�룬���ܹ�delete�������ָ��

	/*cout << "��������float����" << endl;
	for (int i = 0; i < 100; i++) {
		cout << "features[" << i << "0]" << features[i * 10] << endl;;
	}*/
	return  features;
}
/*
������ȡ����
���룺cv::Matͼ��(�޶�ͼ��ֻ��һ������)
�����
*/
float* SeetaInterface::FeatureExtraction(SeetaImageData &simg, bool mask) {
	// �������
	if (mask) {
		return FeatureExtraction1(simg);
		//return NULL;
	}
	else {
		//δ�������
		return FeatureExtraction2(simg);
	}
}


/*
���ܣ�������������ݼ�
���룺���ID
��������������Ϊshared_ptr<float>
�Ƿ������
dataSet.insert(pair<string, shared_ptr<float>>(id, feature));
*/
bool SeetaInterface::Add(string id, float* feature, bool mask) {
	//cout << "dataSetMask size:" << dataSetMask.size()<<endl;
	//cout << "dataSet size:" << dataSet.size() << endl;

	if (mask) {
		//���֮ǰ�Ȳ�����û��
		auto it = dataSetMask.find(id);
		//������ھ����ʧ��
		if (it != dataSetMask.end()) {
			return false;
		}

		dataSetMask.insert(pair<string, float*>(id, feature));
		return true;
		//cout << "����dataSetMask�к���" << dataSetMask.size() << endl;
	}
	else {
		//���֮ǰ�Ȳ�����û��
		auto it = dataSet.find(id);
		//������ھ����ʧ��
		if (it != dataSet.end()) {
			return false;
		}

		dataSet.insert(pair<string, float*>(id, feature));
		return true;
		//cout << "����dataSet�к���" << dataSet.size() << endl;
	}
}


/*�����ֵ������ȶ�*/
string SeetaInterface::FaceCompare1(float* feature) {

	map<string, float*> ::iterator it;
	for (it = dataSetMask.begin(); it != dataSetMask.end(); it++) {
		float* feature2 = it->second;

		//float threshold = 0.60;//����ʶ����ֵ
		float similarity = FR_mask->CalculateSimilarity(feature, feature2);
		if (similarity > threshold) {
			/*cout << endl;
			cout << "�������ƶȣ�" << similarity << endl;
			cout << "��ͬһ����"  << endl;*/
			return it->first;
		}
		else {
			/*cout << "�������ƶȣ�" << similarity << endl;
			cout << "����ͬһ����" << endl;*/
		}
	}
	return "";
}
/*�������ֵ������ȶ�*/
string SeetaInterface::FaceCompare2(float* feature) {
	map<string, float*> ::iterator it;
	for (it = dataSet.begin(); it != dataSet.end(); it++) {
		float* feature2 = it->second;

		//float threshold = 0.60;//����ʶ����ֵ
		float similarity = FR->CalculateSimilarity(feature, feature2);
		if (similarity > threshold) {
			/*cout << endl;
			cout << "�������ƶȣ�" << similarity << endl;
			cout << "��ͬһ����" << endl;*/
			return it->first;
		}
		else {
			/*cout << "�������ƶȣ�" << similarity << endl;
			cout << "����ͬһ����" << endl;*/
		}
	}
	return "";
}

/*
���ܣ��ȶ����ݿ����Ƿ��и�����
���룺��������
�Ƿ��������
�����ƥ���id�����û���ҵ��ͷ���NULL
*/
string SeetaInterface::FaceCompare(float* feature, bool mask) {
	//cout << "--------------------------------" << endl;
	// ������
	if (mask) {
		return FaceCompare1(feature);
	}
	else {
		// û�п���
		return FaceCompare2(feature);
	}
}

/*
���ܣ�����ָ����idɾ������(�����ֺͲ������ֵĶ���ɾ��)
���룺���ID
*/
bool SeetaInterface::Delete(string id,bool mask) {
	// ɾ��������ֵ�
	if (mask) {
		auto at = dataSet.find(id);
		// ��������ڣ�ʲôҲ����
		if (at == dataSet.end()) {

		}
		else {
		// ������ڣ����ͷ��ڴ�
			float* f = at->second;
			delete f;//�ͷ��ڴ�
		}
		
		size_t a = dataSet.erase(id);//ͨ��keyɾ��
		
		//cout << a << endl;
		//cout << "dataSetʣ��" << dataSet.size() << endl;
		if (a == 0 ) {
			return false;
		}
		else {
			return true;
		}
	}
	else {
	// ɾ��δ������ֵ��沿����
		auto at = dataSetMask.find(id);
		// ��������ڣ�ʲôҲ����
		if (at == dataSetMask.end()) {

		}
		else {
			// ������ڣ����ͷ��ڴ�
			float* f = at->second;
			delete f;//�ͷ��ڴ�
		}
		size_t b = dataSetMask.erase(id);//ͨ��keyɾ��
		//cout << b << endl;
		//cout << "dataSetMaskʣ��" << dataSetMask.size() << endl;
		if (b == 0) {
			return false;
		}
		else {
			return true;
		}
	}
}


/*
���ܣ�����ָ����idɾ������(�����ֺͲ������ֵĶ���ɾ��)
���룺���ID
*/
bool SeetaInterface::Update(string id, float* feature, bool mask) {
	if (mask) {
		//����֮ǰ�Ȳ�����û��
		auto it = dataSetMask.find(id);
		//������ھͿ��Ը���
		if (it != dataSetMask.end()) {
			dataSetMask[id] = feature;
			return true;
		}
		else {
		// ��������ھ���Ҫ�ͷ��ڴ�
			delete feature;
			//�������ʧ��
			return false;
		}
	}
	else {
		//����֮ǰ�Ȳ�����û��
		auto it = dataSet.find(id);
		//������ھͿ��Ը���
		if (it != dataSet.end()) {
			dataSet[id] = feature;
			return true;
		}
		else {
		// ��������ھ���Ҫ�ͷ��ڴ�
			delete feature;
			//�������ʧ��
			return false;
		}
	}
}


/*
���ܣ�ͼƬ�ü�����������
���룺����ͼƬ
������ü�������ͼƬ
*/
cv::Mat SeetaInterface::CropFace(SeetaImageData &simg) {



	SeetaFaceInfoArray faces = FaceDetection(simg);
	SeetaPointF points[5];
	FeaturePoints1(simg, faces.data[0].pos, points);

	seeta::ImageData cropface = FR->CropFaceV2(simg, points);// ʶ����������������

	cv::Mat imgmat(cropface.height, cropface.width, CV_8UC(cropface.channels), cropface.data);//���ַ�ʽֻ������ͼƬ��Ϣ��û�п���������Ϣ��ֻ�ǰ����ص�ַ������
	cv::Mat cropMatImage = imgmat.clone();

	//SeetaImageData out_image;
	//out_image.channels = cropMatImage.channels();
	//out_image.data = cropMatImage.data;
	//out_image.height = cropMatImage.rows;
	//out_image.width = cropMatImage.cols;

	//cv::imwrite("C:/MyCode/seetaface/test_seetaface_models/images/crop_222.jpg", cropMatImage);

	return cropMatImage;
}


/*
ǰ�᣺1��float��4�ֽ�(char)
1 float������ת����32λ������ʽ
2 Ȼ��Ѳ���ĵ�8λ����char[0],��8λ����char[3]
ʾ����13.625��������0100 0001 0101 1010 0000 0000 0000 0000
16�����ǣ�0x41 5A 00 00
10�����ǣ�65,90,00,00
��char[0]��0��char[3]��65(�����0��65Ҳ���Ǹ��ݲ���õ���)
�������ܣ���float������char����
���룺
float *val;float�����׵�ַ
char *buff;char�����׵�ַ
int num;float���鹲�м���Ԫ��
�����
*/
void SeetaInterface::Float2Char(float *val, char *buff, int num)
{
	float *xval;
	long  i;
	char *S;
	xval = val;
	S = (char*)(xval);
	for (i = 0; i < num * 4; i++) {
		buff[i] = *(S + i);
		//cout << (int)buff[i] << endl;;
	}
	//cout << endl;
}
void SeetaInterface::Char2Float(char *buff, float* xval, int num)
{
	int i;
	char *S;
	S = (char*)(xval);
	for (i = 0; i < 4*num; i++) {
		*(S + i) = *(buff + i);
	}
}