#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <time.h>
//#include <Windows.h>
#include "headers.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <omp.h>

using namespace cv;
using namespace std;

void DrawPredictedImage(cv::Mat_<uchar> image, cv::Mat_<double>& shape){
	for (int i = 0; i < shape.rows; i++){
		cv::circle(image, cv::Point2f(shape(i, 0), shape(i, 1)), 2, (255));
	}
	cv::imshow("show image", image);
	cv::waitKey(0);
}

void Test(const char* config_file_path){
	cout << "parsing config_file: " << config_file_path << endl;

    ifstream fin;
    fin.open(config_file_path, ifstream::in);
	std::string model_name;
    fin >> model_name;
    cout << "model name is: " << model_name << endl;
	bool images_has_ground_truth = false;
	fin >> images_has_ground_truth;
	if (images_has_ground_truth) {
		cout << "the image lists must have ground_truth_shapes!\n" << endl;
	}
	else{
		cout << "the image lists does not have ground_truth_shapes!!!\n" << endl;
	}

	int path_num;
    fin >> path_num;
    cout << "reading testing images paths: " << endl;
	std::vector<std::string> image_path_prefixes;
    std::vector<std::string> image_lists;
    for (int i = 0; i < path_num; i++) {
        string s;
        fin >> s;
        cout << s << endl;
        image_path_prefixes.push_back(s);
        fin >> s;
        cout << s << endl;
        image_lists.push_back(s);
    }

	cout << "parsing config file done\n" << endl;
	CascadeRegressor cas_load;
	cas_load.LoadCascadeRegressor(model_name);
	cout << "load model done\n" << endl;
	std::vector<cv::Mat_<uchar> > images;
	std::vector<cv::Mat_<double> > ground_truth_shapes;
	std::vector<BoundingBox> bboxes;

	std::cout << "\nLoading test dataset..." << std::endl;
	if (images_has_ground_truth) {
		LoadImages(images, ground_truth_shapes, bboxes, image_path_prefixes, image_lists);
		double error = 0.0;
		for (int i = 0; i < images.size(); i++){
			cv::Mat_<double> current_shape = ReProjection(cas_load.params_.mean_shape_, bboxes[i]);
	        cv::Mat_<double> res = cas_load.Predict(images[i], current_shape, bboxes[i]);//, ground_truth_shapes[i]);
			double e = CalculateError(ground_truth_shapes[i], res);
			// std::cout << "error:" << e << std::endl;
			error += e;
	        // DrawPredictedImage(images[i], res);
		}
		std::cout << "error: " << error << ", mean error: " << error/images.size() << std::endl;
	}
	else{
		LoadImages(images, bboxes, image_path_prefixes, image_lists);
		for (int i = 0; i < images.size(); i++){
			cv::Mat_<double> current_shape = ReProjection(cas_load.params_.mean_shape_, bboxes[i]);
	        cv::Mat_<double> res = cas_load.Predict(images[i], current_shape, bboxes[i]);//, ground_truth_shapes[i]);
	        DrawPredictedImage(images[i], res);
		}
	}
}

void Train(const char* config_file_path){

	cout << "parsing config_file: " << config_file_path << endl;

    ifstream fin;
    fin.open(config_file_path, ifstream::in);
	std::string model_name;
    fin >> model_name;
    cout << "\nmodel name is: " << model_name << endl;
    Parameters params = Parameters();
    fin >> params.local_features_num_
        >> params.landmarks_num_per_face_
        >> params.regressor_stages_
        >> params.tree_depth_
        >> params.trees_num_per_forest_
        >> params.initial_guess_
		>> params.overlap_;

    std::vector<double> local_radius_by_stage;
    local_radius_by_stage.resize(params.regressor_stages_);
    for (int i = 0; i < params.regressor_stages_; i++){
            fin >> local_radius_by_stage[i];
    }
    params.local_radius_by_stage_ = local_radius_by_stage;
    params.output();

    int path_num;
    fin >> path_num;
    cout << "\nreading training images paths: " << endl;

	std::vector<std::string> image_path_prefixes;
    std::vector<std::string> image_lists;
    for (int i = 0; i < path_num; i++) {
        string s;
        fin >> s;
        cout << s << endl;
        image_path_prefixes.push_back(s);
        fin >> s;
        cout << s << endl;
        image_lists.push_back(s);
    }

    fin >> path_num;
    cout << "\nreading validation images paths: " << endl;
	std::vector<std::string> val_image_path_prefixes;
    std::vector<std::string> val_image_lists;
    for (int i = 0; i < path_num; i++) {
        string s;
        fin >> s;
        cout << s << endl;
        val_image_path_prefixes.push_back(s);
        fin >> s;
        cout << s << endl;
        val_image_lists.push_back(s);
    }

    cout << "parsing config file done\n" << endl;


	std::vector<cv::Mat_<uchar> > images;
	std::vector<cv::Mat_<double> > ground_truth_shapes;
	std::vector<BoundingBox> bboxes;

	std::vector<cv::Mat_<uchar> > val_images;
	std::vector<cv::Mat_<double> > val_ground_truth_shapes;
	std::vector<BoundingBox> val_bboxes;
	std::cout << "Loading training dataset..." << std::endl;
	LoadImages(images, ground_truth_shapes, bboxes, image_path_prefixes, image_lists);
	if (val_image_lists.size() > 0) {
		std::cout << "\nLoading validation dataset..." << std::endl;
		LoadImages(val_images, val_ground_truth_shapes, val_bboxes, val_image_path_prefixes, val_image_lists);
	}
	// else{
	// 	std::cout << "your validation dataset is 0" << std::endl;
	// }

	params.mean_shape_ = GetMeanShape(ground_truth_shapes, bboxes);
	CascadeRegressor cas_reg;
	cas_reg.val_bboxes_ = val_bboxes;
    cas_reg.val_images_ = val_images;
    cas_reg.val_ground_truth_shapes_ = val_ground_truth_shapes;

	cas_reg.Train(images, ground_truth_shapes, bboxes, params);
	std::cout << "finish training, start to saving the model..." << std::endl;
	std::cout << "model name: " << model_name << std::endl;
	cas_reg.SaveCascadeRegressor(model_name);
	std::cout << "save the model successfully\n" << std::endl;
}

int main(int argc, char* argv[])
{
	std::cout << "\nuse [./application train train_config_file] to train models" << std::endl;
	std::cout << "    [./application test test_config_file] to test images\n\n" << std::endl;

	if (argc == 3) {
		if (strcmp(argv[1], "train") == 0)
		{
			Train(argv[2]);
            return 0;
		}
		if (strcmp(argv[1], "test") == 0)
		{
			Test(argv[2]);
            return 0;
		}
	}
	else {
		std::cout << "\nWRONG!!!" << std::endl;
	}

	return 0;
}
