#ifndef REGRESSOR_H
#define REGRESSOR_H

#include "utils.h"
#include "randomforest.h"


class Regressor {
public:
	int stage_;
	Parameters params_;
	std::vector<RandomForest> rd_forests_;
    std::vector<struct model*> linear_model_x_;
    std::vector<struct model*> linear_model_y_;

    struct feature_node* tmp_binary_features;
    cv::Mat_<uchar> tmp_image;
    cv::Mat_<double> tmp_current_shape;
    BoundingBox tmp_bbox;
    cv::Mat_<double> tmp_rotation;
    double tmp_scale;
    int leaf_index_count[68];
    int feature_node_index[68];
    std::atomic<int> cur_landmark {0};

public:
	Regressor();
	~Regressor();
    Regressor(const Regressor&);
	std::vector<cv::Mat_<double> > Train(const std::vector<cv::Mat_<uchar> >& images,
		const std::vector<int>& augmented_images_index,
		const std::vector<cv::Mat_<double> >& augmented_ground_truth_shapes,
		const std::vector<BoundingBox>& augmented_bboxes,
		const std::vector<cv::Mat_<double> >& augmented_current_shapes,
		const Parameters& params,
		const int stage);
    struct feature_node* GetGlobalBinaryFeatures(cv::Mat_<uchar>& image, cv::Mat_<double>& current_shape, BoundingBox& bbox, cv::Mat_<double>& rotation, double scale);
	cv::Mat_<double> Predict(cv::Mat_<uchar>& image, cv::Mat_<double>& current_shape, 
		BoundingBox& bbox, cv::Mat_<double>& rotation, double scale);
	void LoadRegressor(std::string ModelName, int stage);
	void SaveRegressor(std::string ModelName, int stage);
    void ConstructLeafCount();
    struct feature_node* GetGlobalBinaryFeaturesThread(cv::Mat_<uchar>& image, cv::Mat_<double>& current_shape, BoundingBox& bbox, cv::Mat_<double>& rotation, double scale);
    struct feature_node* GetGlobalBinaryFeaturesMP(cv::Mat_<uchar>& image,
        cv::Mat_<double>& current_shape, BoundingBox& bbox, cv::Mat_<double>& rotation, double scale);
    void GetFeaThread();
};

class CascadeRegressor {
public:
	Parameters params_;
	std::vector<cv::Mat_<uchar> > images_;
	std::vector<cv::Mat_<double> > ground_truth_shapes_;
	
    //std::vector<struct model*> linear_model_x_;
    //std::vector<struct model*> linear_model_y_;
    //std::vector<cv::Mat_<double> > current_shapes_;
	std::vector<BoundingBox> bboxes_;
	//cv::Mat_<double> mean_shape_;
	std::vector<Regressor> regressors_;
public:
	CascadeRegressor();
	void Train(const std::vector<cv::Mat_<uchar> >& images,
		const std::vector<cv::Mat_<double> >& ground_truth_shapes,
		//const std::vector<cv::Mat_<double> >& current_shapes,
		const std::vector<BoundingBox>& bboxes,
		Parameters& params);
	cv::Mat_<double> Predict(cv::Mat_<uchar>& image, cv::Mat_<double>& current_shape, BoundingBox& bbox, cv::Mat_<double>& ground_truth_shape);
	cv::Mat_<double> Predict(cv::Mat_<uchar>& image, cv::Mat_<double>& current_shape, BoundingBox& bbox);
	void LoadCascadeRegressor(std::string ModelName);
	void SaveCascadeRegressor(std::string ModelName);

};

#endif
