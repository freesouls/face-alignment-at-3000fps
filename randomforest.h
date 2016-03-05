#ifndef RANDOMFOREST_H
#define RANDOMFOREST_H
#include "utils.h"
#include <set>
class Node {
public:
	int leaf_identity; // used only when it is leaf node, and is unique among the tree
	Node* left_child_;
	Node* right_child_;
	int samples_;
	bool is_leaf_;
	int depth_; // recording current depth
	double threshold_;
	bool thre_changed_;
	FeatureLocations feature_locations_;
	Node(Node* left, Node* right, double thres, bool leaf);
	Node(Node* left, Node* right, double thres);
	Node();
};

class RandomForest {
public:
	int stage_;
	int local_features_num_;
	int landmark_index_;
	int tree_depth_;
	int trees_num_per_forest_;
	double local_radius_;
	int all_leaf_nodes_;
	double overlap_;
	//cv::Mat_<double> mean_shape_;
	std::vector<Node*> trees_;
	std::vector<FeatureLocations> local_position_; // size = param_.local_features_num
	std::vector<cv::Mat_<double> >* regression_targets_;

	bool TrainForest(//std::vector<cv::Mat_<double> >& regression_targets,
		const std::vector<cv::Mat_<uchar> >& images,
		const std::vector<int>& augmented_images_index,
		//const std::vector<cv::Mat_<double> >& augmented_ground_truth_shapes,
		const std::vector<BoundingBox>& augmented_bboxes,
		const std::vector<cv::Mat_<double> >& augmented_current_shapes,
		const std::vector<cv::Mat_<double> >& rotations,
		const std::vector<double>& scales);
	Node* BuildTree(std::set<int>& selected_indexes, cv::Mat_<int>& pixel_differences, std::vector<int>& images_indexes, int current_depth);
	int FindSplitFeature(Node* node, std::set<int>& selected_indexes,
		cv::Mat_<int>& pixel_differences, std::vector<int>& images_indexes, std::vector<int>& left_indexes, std::vector<int>& right_indexes);
	cv::Mat_<double> GetBinaryFeatures(const cv::Mat_<double>& image,
		const BoundingBox& bbox, const cv::Mat_<double>& current_shape, const cv::Mat_<double>& rotation, const double& scale);
	int MarkLeafIdentity(Node* node, int count);
	int GetNodeOutput(Node* node, const cv::Mat_<double>& image,
		const BoundingBox& bbox, const cv::Mat_<double>& current_shape, const cv::Mat_<double>& rotation, const double& scale);
	//predict()
	int GetBinaryFeatureIndex(int tree_index, const cv::Mat_<double>& image,
	const BoundingBox& bbox, const cv::Mat_<double>& current_shape, const cv::Mat_<double>& rotation, const double& scale);
	RandomForest();
	RandomForest(Parameters& param, int landmark_index, int stage, std::vector<cv::Mat_<double> >& regression_targets);
	void WriteTree(Node* p, std::ofstream& fout);
	Node* ReadTree(std::ifstream& fin);
	void SaveRandomForest(std::ofstream& fout);
	void LoadRandomForest(std::ifstream& fin);
};

#endif
