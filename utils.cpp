#include "utils.h"
//#include "facedetect-dll.h"
//#pragma comment(lib,"libfacedetect.lib")

// project the global shape coordinates to [-1, 1]x[-1, 1]
cv::Mat_<double> ProjectShape(const cv::Mat_<double>& shape, const BoundingBox& bbox){
	cv::Mat_<double> results(shape.rows, 2);
	for (int i = 0; i < shape.rows; i++){
		results(i, 0) = (shape(i, 0) - bbox.center_x) / (bbox.width / 2.0);
		results(i, 1) = (shape(i, 1) - bbox.center_y) / (bbox.height / 2.0);
	}
	return results;
}

// reproject the shape to global coordinates
cv::Mat_<double> ReProjection(const cv::Mat_<double>& shape, const BoundingBox& bbox){
	cv::Mat_<double> results(shape.rows, 2);
	for (int i = 0; i < shape.rows; i++){
		results(i, 0) = shape(i, 0)*bbox.width / 2.0 + bbox.center_x;
		results(i, 1) = shape(i, 1)*bbox.height / 2.0 + bbox.center_y;
	}
	return results;
}

// get the mean shape, [-1, 1]x[-1, 1]
cv::Mat_<double> GetMeanShape(const std::vector<cv::Mat_<double> >& all_shapes,
	const std::vector<BoundingBox>& all_bboxes) {

	cv::Mat_<double> mean_shape = cv::Mat::zeros(all_shapes[0].rows, 2, CV_32FC1);
	for (int i = 0; i < all_shapes.size(); i++)
	{
		mean_shape += ProjectShape(all_shapes[i], all_bboxes[i]);
	}
	mean_shape = 1.0 / all_shapes.size()*mean_shape;
	return mean_shape;
}

// get the rotation and scale parameters by transferring shape_from to shape_to, shape_to = M*shape_from
void getSimilarityTransform(const cv::Mat_<double>& shape_to,
	const cv::Mat_<double>& shape_from,
	cv::Mat_<double>& rotation, double& scale){
	rotation = cv::Mat(2, 2, 0.0);
	scale = 0;

	// center the data
	double center_x_1 = 0.0;
	double center_y_1 = 0.0;
	double center_x_2 = 0.0;
	double center_y_2 = 0.0;
	for (int i = 0; i < shape_to.rows; i++){
		center_x_1 += shape_to(i, 0);
		center_y_1 += shape_to(i, 1);
		center_x_2 += shape_from(i, 0);
		center_y_2 += shape_from(i, 1);
	}
	center_x_1 /= shape_to.rows;
	center_y_1 /= shape_to.rows;
	center_x_2 /= shape_from.rows;
	center_y_2 /= shape_from.rows;

	cv::Mat_<double> temp1 = shape_to.clone();
	cv::Mat_<double> temp2 = shape_from.clone();
	for (int i = 0; i < shape_to.rows; i++){
		temp1(i, 0) -= center_x_1;
		temp1(i, 1) -= center_y_1;
		temp2(i, 0) -= center_x_2;
		temp2(i, 1) -= center_y_2;
	}


	cv::Mat_<double> covariance1, covariance2;
	cv::Mat_<double> mean1, mean2;
	// calculate covariance matrix
    cv::calcCovarMatrix(temp1, covariance1, mean1, cv::COVAR_COLS, CV_64F); //CV_COVAR_COLS
    cv::calcCovarMatrix(temp2, covariance2, mean2, cv::COVAR_COLS, CV_64F);

	double s1 = sqrt(norm(covariance1));
	double s2 = sqrt(norm(covariance2));
	scale = s1 / s2;
	temp1 = 1.0 / s1 * temp1;
	temp2 = 1.0 / s2 * temp2;

	double num = 0.0;
	double den = 0.0;
	for (int i = 0; i < shape_to.rows; i++){
		num = num + temp1(i, 1) * temp2(i, 0) - temp1(i, 0) * temp2(i, 1);
		den = den + temp1(i, 0) * temp2(i, 0) + temp1(i, 1) * temp2(i, 1);
	}

	double norm = sqrt(num*num + den*den);
	double sin_theta = num / norm;
	double cos_theta = den / norm;
	rotation(0, 0) = cos_theta;
	rotation(0, 1) = -sin_theta;
	rotation(1, 0) = sin_theta;
	rotation(1, 1) = cos_theta;
}

cv::Mat_<double> LoadGroundTruthShape(const char* name){
	int landmarks = 0;
	std::ifstream fin;
	std::string temp;
	fin.open(name, std::fstream::in);
	getline(fin, temp);// read first line
	fin >> temp >> landmarks;
	cv::Mat_<double> shape(landmarks, 2);
	getline(fin, temp); // read '\n' of the second line
	getline(fin, temp); // read third line
	for (int i = 0; i<landmarks; i++){
		fin >> shape(i, 0) >> shape(i, 1);
	}
	fin.close();
	return shape;
}

bool ShapeInRect(cv::Mat_<double>& shape, cv::Rect& ret){
	double sum_x = 0.0, sum_y = 0.0;
	double max_x = 0, min_x = 10000, max_y = 0, min_y = 10000;
	for (int i = 0; i < shape.rows; i++){
		if (shape(i, 0)>max_x) max_x = shape(i, 0);
		if (shape(i, 0)<min_x) min_x = shape(i, 0);
		if (shape(i, 1)>max_y) max_y = shape(i, 1);
		if (shape(i, 1)<min_y) min_y = shape(i, 1);

		sum_x += shape(i, 0);
		sum_y += shape(i, 1);
	}
	sum_x /= shape.rows;
	sum_y /= shape.rows;

	if ((max_x - min_x) > ret.width * 1.5) return false;
	if ((max_y - min_y) > ret.height * 1.5) return false;
	if (abs(sum_x - (ret.x + ret.width / 2.0)) > ret.width / 2.0) return false;
	if (abs(sum_y - (ret.y + ret.height / 2.0)) > ret.height / 2.0) return false;
	return true;
}

std::vector<cv::Rect> DetectFaces(cv::Mat_<uchar>& image, cv::CascadeClassifier& classifier){
	std::vector<cv::Rect_<int> > faces;
	classifier.detectMultiScale(image, faces, 1.1, 2, 0, cv::Size(30, 30));
	return faces;
}


void LoadImages(std::vector<cv::Mat_<uchar> >& images,
	std::vector<cv::Mat_<double> >& ground_truth_shapes,
	//const std::vector<cv::Mat_<double> >& current_shapes,
	std::vector<BoundingBox>& bboxes,
	std::string file_names){
	
    std::string fn_haar = "./../haarcascade_frontalface_alt2.xml";
    cv::CascadeClassifier haar_cascade;
    bool yes = haar_cascade.load(fn_haar);
    std::cout << "detector: " << yes << std::endl;
	std::cout << "loading images\n";
	std::ifstream fin;
	fin.open(file_names.c_str(), std::ifstream::in);
	std::string name;
	int count = 0;
	//std::cout << name << std::endl;
	while (fin >> name){
		//std::cout << "reading file: " << name << std::endl;
		//std::cout << name << std::endl;

        cv::Mat_<uchar> image = cv::imread(("./../helen/trainset/" + name + ".jpg").c_str(), 0);
        cv::Mat_<double> ground_truth_shape = LoadGroundTruthShape(("./../helen/trainset/" + name + ".pts").c_str());
        BoundingBox bbox;
        std::ifstream fin;
        fin.open(("./../helen/trainset/" + name + ".box").c_str());
        fin >> bbox.start_x
            >> bbox.start_y
            >> bbox.width
            >> bbox.height
            >> bbox.center_x
            >> bbox.center_y;
        fin.close();

        //cv::Mat_<uchar> image = cv::imread((name + ".jpg").c_str(), 0);
        //cv::Mat_<double> ground_truth_shape = LoadGroundTruthShape((name + ".pts").c_str());
		if (image.cols > 2000){
			cv::resize(image, image, cv::Size(image.rows / 3, image.cols / 3), 0, 0, cv::INTER_LINEAR);
			ground_truth_shape /= 3.0;
		}
		else if (image.cols > 1400 && image.cols <= 2000){
			cv::resize(image, image, cv::Size(image.rows / 2, image.cols / 2), 0, 0, cv::INTER_LINEAR);
			ground_truth_shape /= 2.0;
		}

//		BoundingBox bbox;
//		bbox = GetBoundingBox(ground_truth_shape, image.cols, image.rows);

        images.push_back(image);
        ground_truth_shapes.push_back(ground_truth_shape);
        bboxes.push_back(bbox);
        count++;
        if (count%100 == 0){
            std::cout << count << " images loaded\n";
        }
        continue;

        std::vector<cv::Rect> faces;// = DetectFaces(image);
        haar_cascade.detectMultiScale(image, faces, 1.1, 2, 0, cv::Size(30, 30));
		
         for (int i = 0; i < faces.size(); i++){
            cv::Rect faceRec = faces[i];
            if (ShapeInRect(ground_truth_shape, faceRec)){
                images.push_back(image);
                ground_truth_shapes.push_back(ground_truth_shape);
                BoundingBox bbox;
                bbox.start_x = faceRec.x;
                bbox.start_y = faceRec.y;
                bbox.width = faceRec.width;
                bbox.height = faceRec.height;
                bbox.center_x = bbox.start_x + bbox.width / 2.0;
                bbox.center_y = bbox.start_y + bbox.height / 2.0;
                bboxes.push_back(bbox);
//		 		std::ofstream fout;
//		 		fout.open(name + ".box", std::fstream::in);
//		 		fout << bbox.start_x << " "
//		 			<< bbox.start_y << " "
//		 			<< bbox.width << " "
//		 			<< bbox.height << " "
//		 			<< bbox.center_x << " "
//		 			<< bbox.center_y << std::endl;
//		 		fout.close();
                count++;
                if (count%100 == 0){
                    std::cout << count << " images loaded\n";
                    return;
                }
                break;
            }
         }

	}
	std::cout << "get " << bboxes.size() << " faces\n";
	fin.close();
}


double CalculateError(cv::Mat_<double>& ground_truth_shape, cv::Mat_<double>& predicted_shape){
	cv::Mat_<double> temp;
	double sum = 0;
	for (int i = 0; i<ground_truth_shape.rows; i++){
		sum += norm(ground_truth_shape.row(i) - predicted_shape.row(i));
	}
    return sum / (ground_truth_shape.rows);
}

void DrawPredictImage(cv::Mat_<uchar> image, cv::Mat_<double>& shape){
	for (int i = 0; i < shape.rows; i++){
		cv::circle(image, cv::Point2f(shape(i, 0), shape(i, 1)), 2, (255));
	}
	cv::imshow("show image", image);
	cv::waitKey(0);
}

BoundingBox GetBoundingBox(cv::Mat_<double>& shape, int width, int height){
	double min_x = 100000.0, min_y = 100000.0;
	double max_x = -1.0, max_y = -1.0;
	for (int i = 0; i < shape.rows; i++){
		if (shape(i, 0)>max_x) max_x = shape(i, 0);
		if (shape(i, 0)<min_x) min_x = shape(i, 0);
		if (shape(i, 1)>max_y) max_y = shape(i, 1);
		if (shape(i, 1)<min_y) min_y = shape(i, 1);
	}
	BoundingBox bbox;
	double scale = 0.6;
	bbox.start_x = min_x - (max_x - min_x) * (scale - 0.5);
	if (bbox.start_x < 0.0)
	{
		bbox.start_x = 0.0;
	}
	bbox.start_y = min_y - (max_y - min_y) * (scale - 0.5);
	if (bbox.start_y < 0.0)
	{
		bbox.start_y = 0.0;
	}
	bbox.width = (max_x - min_x) * scale * 2.0;
	if (bbox.width >= width){
		bbox.width = width - 1.0;
	}
	bbox.height = (max_y - min_y) * scale * 2.0;
	if (bbox.height >= height){
		bbox.height = height - 1.0;
	}
	bbox.center_x = bbox.start_x + bbox.width / 2.0;
	bbox.center_y = bbox.start_y + bbox.height / 2.0;
	return bbox;
}
