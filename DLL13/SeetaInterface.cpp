
#include "SeetaInterface.h"

//
//map<string, float*>   SeetaInterface::dataSet;//在.h中定义的静态变量需要在这里进行初始化，不让不能够使用
//map<string, float*>   SeetaInterface::dataSetMask;//在.h中定义的静态变量需要在这里进行初始化，不让不能够使用



//初始化数据集，把内存中数据清空
void SeetaInterface::InitDataSet() {
	dataSet.clear();//在本类内使用不需要添加SeetaInterface::作用于，可以直接使用
}


//初始化模型
void SeetaInterface::InitModel(int num) {
	//不戴口罩人脸检测模型初始化
	seeta::ModelSetting FD_setting;
	FD_setting.append(ModelPath + "face_detector.csta");
	FD_setting.set_device(seeta::ModelSetting::CPU);
	FD_setting.set_id(0);
	FD = new seeta::FaceDetector(FD_setting);
	FD->set(seeta::v6::FaceDetector::PROPERTY_NUMBER_THREADS, num);

	//口罩人脸检测模型初始化
	seeta::ModelSetting FD_setting_mask;
	FD_setting_mask.append(ModelPath + "mask_detector.csta");
	FD_setting_mask.set_device(seeta::ModelSetting::CPU);
	FD_setting_mask.set_id(0);
	FD_mask = new seeta::MaskDetector(FD_setting_mask);



	//不戴口罩人脸关键点模型初始化
	seeta::ModelSetting PD_setting;
	PD_setting.append(ModelPath + "face_landmarker_pts5.csta");
	FL = new seeta::FaceLandmarker(PD_setting);

	//戴口罩人脸关键点模型初始化
	seeta::ModelSetting PD_setting_mask;
	PD_setting_mask.append(ModelPath + "face_landmarker_mask_pts5.csta");
	FL_mask = new seeta::FaceLandmarker(PD_setting_mask);

	//不戴口罩人脸识别模型初始化
	seeta::ModelSetting fr_setting;
	fr_setting.append(ModelPath + "face_recognizer.csta");
	FR = new seeta::FaceRecognizer(fr_setting);
	FR->set(seeta::v6::FaceRecognizer::PROPERTY_NUMBER_THREADS, num);


	//戴口罩人脸识别模型初始化
	seeta::ModelSetting fr_setting_mask;
	fr_setting_mask.append(ModelPath + "face_recognizer_mask.csta");
	FR_mask = new seeta::FaceRecognizer(fr_setting_mask);
	FR_mask->set(seeta::v6::FaceRecognizer::PROPERTY_NUMBER_THREADS, num);


}

/*
功能：初始化函数
输入：线程数
会初始化模型、清空内存中数据
默认4线程，一般情况下4/8/16线程未最优
*/
void SeetaInterface::Init(string path, int num, float _threshold) {
	ModelPath = path;
	InitDataSet();
	InitModel(num);
	threshold = _threshold;
}

/*
人脸检测函数
输入：cv::Mat图像
输出：检测到的人脸位置信息(可能有多个)
*/
SeetaFaceInfoArray SeetaInterface::FaceDetection(SeetaImageData &simg) {
	//SeetaImageData simg;
	//simg.height = matImage.rows;
	//simg.width = matImage.cols;
	//simg.channels = matImage.channels();
	//simg.data = matImage.data;

	return FD->detect(simg);//返回的信息是，人脸位置和置信分数

}


/*
功能：检测是否佩戴
输入：cv::Mat图像(限定图像只有一张人脸)
输出：是否佩戴口罩，1 佩戴，0 未佩戴
*/
bool SeetaInterface::DetectionMask(SeetaImageData &simg) {
	//人脸检测，检测位置信息
	auto faces = FaceDetection(simg);//返回的信息是，人脸位置和置信分数
										 //if (faces.size <= 0) {
										 //	cout << "没有检测到人脸" << endl;
										 //}
										 //else
										 //{
										 //	cout << "检测到人脸个数:" << faces.size << endl << endl;
										 //}

										 //检测是否戴口罩

	float score;//是否戴口罩的置信分数
	bool mask = FD_mask->detect(simg, faces.data[0].pos, &score);
	//if (mask) {
	//	cout << "戴口罩的置信度:" << score << endl;
	//	cout << "佩戴了口罩" << endl;
	//}
	//else
	//{
	//	cout << "戴口罩的置信度:" << score << endl;
	//	cout << "未佩戴口罩" << endl;
	//}
	return mask;
}



SeetaInterface::SeetaInterface()
{

}


SeetaInterface::~SeetaInterface()
{
	//释放map中的内存
	for (auto it = dataSetMask.begin(); it != dataSetMask.end(); it = dataSetMask.begin()) {
		float* f = it->second;
		//cout << "dataSetMask size:" <<dataSetMask.size()<< endl;
		//cout << "dataSetMask删除" << endl;
		//cout << "dataSetMask删除" << it->first << endl;
		dataSetMask.erase(it);
		delete f;
	}
	for (auto it = dataSet.begin(); it != dataSet.end(); it = dataSet.begin()) {
		float* f = it->second;
		//cout << "dataSetMask删除" << it->first << endl;
		dataSet.erase(it);
		delete f;
	}

	//释放模型内存
	delete FD;
	delete FD_mask;
	delete FL;
	delete FL_mask;
	delete FR;
	delete FR_mask;
}

/*不带口罩的特征点提取*/
void SeetaInterface::FeaturePoints2(const SeetaImageData &simg, const SeetaRect &pos, SeetaPointF *points) {
	FL->mark(simg, pos, points);
	//cout << "不带口罩的特征点提取" << endl;
	//for (int i = 0; i < 5; i++) {//每个人脸的特征点
	//	cout << "人脸特征点" << i + 1 << "的位置:" << endl;
	//	cout << "人脸特征点横坐标:" << points[i].x << endl;
	//	cout << "人脸特征点纵坐标:" << points[i].y << endl;
	//	cout << endl;
	//}
}

/*戴口罩的特征点提取*/
void SeetaInterface::FeaturePoints1(const SeetaImageData &simg, const SeetaRect &pos, SeetaPointF *points) {
	FL_mask->mark(simg, pos, points);
	//cout <<"戴口罩的特征点提取" <<endl;
	//for (int i = 0; i < 5; i++) {//每个人脸的特征点
	//	cout << "人脸特征点" << i + 1 << "的位置:" << endl;
	//	cout << "人脸特征点横坐标:" << points[i].x << endl;
	//	cout << "人脸特征点纵坐标:" << points[i].y << endl;
	//	cout << endl;
	//}
}
/*戴口罩的特征提取*/
float* SeetaInterface::FeatureExtraction1(SeetaImageData &simg) {
	// 人脸位置检测
	auto faces = FaceDetection(simg);//返回的信息是，人脸位置和置信分数

	// 特征点提取
	SeetaPointF points[5];
	FeaturePoints1(simg, faces.data[0].pos, points);
	//特征提取
	float *features = new float[_MASK_FEATURE_NUM];
	memset(features, 0, _MASK_FEATURE_NUM * sizeof(float));
	//shared_ptr<float> feature(new float[FR_mask->GetExtractFeatureSize()]);//申请一个float特征数组长度的内存空间，一会用来存放提取的特征数组
	FR_mask->Extract(simg, points, features);//feature1.get()从智能指针里返回一个内置指针，不能够delete这个内置指针
	/*cout << "人脸特征float类型" << endl;
	for (int i = 0; i < 50; i++) {
		cout << "features[" << i << "0]" << features[i * 10] << endl;;
	}*/
	return  features;
}
/*不戴口罩的特征提取*/
float* SeetaInterface::FeatureExtraction2(SeetaImageData &simg) {
	// 人脸位置检测
	auto faces = FaceDetection(simg);//返回的信息是，人脸位置和置信分数

	// 特征点提取
	SeetaPointF points[5];
	FeaturePoints2(simg, faces.data[0].pos, points);

	//特征提取
	float *features=new float[_NORMAL_FEATURE_NUM];
	memset(features, 0, _NORMAL_FEATURE_NUM * sizeof(float));
	//shared_ptr<float> feature(new float[FR->GetExtractFeatureSize()]);//申请一个float特征数组长度的内存空间，一会用来存放提取的特征数组
	FR->Extract(simg, points, features);//feature1.get()从智能指针里返回一个内置指针，不能够delete这个内置指针

	/*cout << "人脸特征float类型" << endl;
	for (int i = 0; i < 100; i++) {
		cout << "features[" << i << "0]" << features[i * 10] << endl;;
	}*/
	return  features;
}
/*
特征提取函数
输入：cv::Mat图像(限定图像只有一张人脸)
输出：
*/
float* SeetaInterface::FeatureExtraction(SeetaImageData &simg, bool mask) {
	// 佩戴口罩
	if (mask) {
		return FeatureExtraction1(simg);
		//return NULL;
	}
	else {
		//未佩戴口罩
		return FeatureExtraction2(simg);
	}
}


/*
功能：添加特征到数据集
输入：身份ID
特征向量，类型为shared_ptr<float>
是否戴口罩
dataSet.insert(pair<string, shared_ptr<float>>(id, feature));
*/
bool SeetaInterface::Add(string id, float* feature, bool mask) {
	//cout << "dataSetMask size:" << dataSetMask.size()<<endl;
	//cout << "dataSet size:" << dataSet.size() << endl;

	if (mask) {
		//添加之前先查找有没有
		auto it = dataSetMask.find(id);
		//如果存在就添加失败
		if (it != dataSetMask.end()) {
			return false;
		}

		dataSetMask.insert(pair<string, float*>(id, feature));
		return true;
		//cout << "现在dataSetMask中含有" << dataSetMask.size() << endl;
	}
	else {
		//添加之前先查找有没有
		auto it = dataSet.find(id);
		//如果存在就添加失败
		if (it != dataSet.end()) {
			return false;
		}

		dataSet.insert(pair<string, float*>(id, feature));
		return true;
		//cout << "现在dataSet中含有" << dataSet.size() << endl;
	}
}


/*戴口罩的人脸比对*/
string SeetaInterface::FaceCompare1(float* feature) {

	map<string, float*> ::iterator it;
	for (it = dataSetMask.begin(); it != dataSetMask.end(); it++) {
		float* feature2 = it->second;

		//float threshold = 0.60;//人脸识别阈值
		float similarity = FR_mask->CalculateSimilarity(feature, feature2);
		if (similarity > threshold) {
			/*cout << endl;
			cout << "人脸相似度：" << similarity << endl;
			cout << "是同一个人"  << endl;*/
			return it->first;
		}
		else {
			/*cout << "人脸相似度：" << similarity << endl;
			cout << "不是同一个人" << endl;*/
		}
	}
	return "";
}
/*不戴口罩的人脸比对*/
string SeetaInterface::FaceCompare2(float* feature) {
	map<string, float*> ::iterator it;
	for (it = dataSet.begin(); it != dataSet.end(); it++) {
		float* feature2 = it->second;

		//float threshold = 0.60;//人脸识别阈值
		float similarity = FR->CalculateSimilarity(feature, feature2);
		if (similarity > threshold) {
			/*cout << endl;
			cout << "人脸相似度：" << similarity << endl;
			cout << "是同一个人" << endl;*/
			return it->first;
		}
		else {
			/*cout << "人脸相似度：" << similarity << endl;
			cout << "不是同一个人" << endl;*/
		}
	}
	return "";
}

/*
功能：比对数据库中是否含有该人脸
输入：人脸特征
是否佩戴口罩
输出：匹配的id，如果没有找到就返回NULL
*/
string SeetaInterface::FaceCompare(float* feature, bool mask) {
	//cout << "--------------------------------" << endl;
	// 戴口罩
	if (mask) {
		return FaceCompare1(feature);
	}
	else {
		// 没有口罩
		return FaceCompare2(feature);
	}
}

/*
功能：根据指定的id删除特征(戴口罩和不戴口罩的都会删除)
输入：身份ID
*/
bool SeetaInterface::Delete(string id,bool mask) {
	// 删除佩戴口罩的
	if (mask) {
		auto at = dataSet.find(id);
		// 如果不存在，什么也不做
		if (at == dataSet.end()) {

		}
		else {
		// 如果存在，就释放内存
			float* f = at->second;
			delete f;//释放内存
		}
		
		size_t a = dataSet.erase(id);//通过key删除
		
		//cout << a << endl;
		//cout << "dataSet剩余" << dataSet.size() << endl;
		if (a == 0 ) {
			return false;
		}
		else {
			return true;
		}
	}
	else {
	// 删除未佩戴口罩的面部特征
		auto at = dataSetMask.find(id);
		// 如果不存在，什么也不做
		if (at == dataSetMask.end()) {

		}
		else {
			// 如果存在，就释放内存
			float* f = at->second;
			delete f;//释放内存
		}
		size_t b = dataSetMask.erase(id);//通过key删除
		//cout << b << endl;
		//cout << "dataSetMask剩余" << dataSetMask.size() << endl;
		if (b == 0) {
			return false;
		}
		else {
			return true;
		}
	}
}


/*
功能：根据指定的id删除特征(戴口罩和不戴口罩的都会删除)
输入：身份ID
*/
bool SeetaInterface::Update(string id, float* feature, bool mask) {
	if (mask) {
		//更新之前先查找有没有
		auto it = dataSetMask.find(id);
		//如果存在就可以更新
		if (it != dataSetMask.end()) {
			dataSetMask[id] = feature;
			return true;
		}
		else {
		// 如果不存在就需要释放内存
			delete feature;
			//否则更新失败
			return false;
		}
	}
	else {
		//更新之前先查找有没有
		auto it = dataSet.find(id);
		//如果存在就可以更新
		if (it != dataSet.end()) {
			dataSet[id] = feature;
			return true;
		}
		else {
		// 如果不存在就需要释放内存
			delete feature;
			//否则更新失败
			return false;
		}
	}
}


/*
功能：图片裁剪出人脸区域
输入：整张图片
输出：裁剪的人脸图片
*/
cv::Mat SeetaInterface::CropFace(SeetaImageData &simg) {



	SeetaFaceInfoArray faces = FaceDetection(simg);
	SeetaPointF points[5];
	FeaturePoints1(simg, faces.data[0].pos, points);

	seeta::ImageData cropface = FR->CropFaceV2(simg, points);// 识别器进行人脸裁切

	cv::Mat imgmat(cropface.height, cropface.width, CV_8UC(cropface.channels), cropface.data);//这种方式只拷贝了图片信息，没有拷贝像素信息，只是把像素地址拷贝了
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
前提：1个float是4字节(char)
1 float浮点数转化成32位补码形式
2 然后把补码的低8位放在char[0],高8位放在char[3]
示例：13.625，补码是0100 0001 0101 1010 0000 0000 0000 0000
16进制是，0x41 5A 00 00
10进制是，65,90,00,00
即char[0]是0，char[3]是65(这里的0和65也都是根据补码得到的)
函数功能：把float数组变成char数组
输入：
float *val;float数组首地址
char *buff;char数组首地址
int num;float数组共有几个元素
输出：
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