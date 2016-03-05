#include "randomforest.h"
#include <time.h>
#include <algorithm>
#include <stack>
Node::Node(){
	left_child_ = NULL;
	right_child_ = NULL;
	is_leaf_ = false;
	threshold_ = 0.0;
	leaf_identity = -1;
	samples_ = -1;
	thre_changed_ = false;
}

Node::Node(Node* left, Node* right, double thres){
	Node(left, right, thres, false);
}

Node::Node(Node* left, Node* right, double thres, bool leaf){
	left_child_ = left;
	right_child_ = right;
	is_leaf_ = leaf;
	threshold_ = thres;
	//offset_ = cv::Point2f(0, 0);
}

bool RandomForest::TrainForest(//std::vector<cv::Mat_<double>>& regression_targets,
	const std::vector<cv::Mat_<uchar> >& images,
	const std::vector<int>& augmented_images_index,
	//const std::vector<cv::Mat_<double>>& augmented_ground_truth_shapes,
	const std::vector<BoundingBox>& augmented_bboxes,
	const std::vector<cv::Mat_<double> >& augmented_current_shapes,
	const std::vector<cv::Mat_<double> >& rotations,
	const std::vector<double>& scales){
    //std::cout << "build forest of landmark: " << landmark_index_ << " of stage: " << stage_ << std::endl;
	//regression_targets_ = &regression_targets;
	time_t current_time;
	current_time = time(0);
	cv::RNG rd(current_time);
	// random generate feature locations
	//std::cout << "generate feature locations" << std::endl;
	local_position_.clear();
	local_position_.resize(local_features_num_);
	for (int i = 0; i < local_features_num_; i++){
		double x, y;
		do{
			x = rd.uniform(-local_radius_, local_radius_);
			y = rd.uniform(-local_radius_, local_radius_);
		} while (x*x + y*y > local_radius_*local_radius_);
		cv::Point2f a(x, y);

		do{
			x = rd.uniform(-local_radius_, local_radius_);
			y = rd.uniform(-local_radius_, local_radius_);
		} while (x*x + y*y > local_radius_*local_radius_);
		cv::Point2f b(x, y);

		local_position_[i] = FeatureLocations(a, b);
	}
	//std::cout << "get pixel differences" << std::endl;
	cv::Mat_<int> pixel_differences(local_features_num_, augmented_images_index.size()); // matrix: features*images

	for (int i = 0; i < augmented_images_index.size(); i++){

		cv::Mat_<double> rotation = rotations[i];
		double scale = scales[i];
		//getSimilarityTransform(ProjectShape(augmented_current_shapes[i], augmented_bboxes[i]),mean_shape_, rotation, scale);

		for (int j = 0; j < local_features_num_; j++){
			FeatureLocations pos = local_position_[j];
			double delta_x = rotation(0, 0)*pos.start.x + rotation(0, 1)*pos.start.y;
			double delta_y = rotation(1, 0)*pos.start.x + rotation(1, 1)*pos.start.y;
			delta_x = scale*delta_x*augmented_bboxes[i].width / 2.0;
			delta_y = scale*delta_y*augmented_bboxes[i].height / 2.0;
			int real_x = delta_x + augmented_current_shapes[i](landmark_index_, 0);
			int real_y = delta_y + augmented_current_shapes[i](landmark_index_, 1);
			real_x = std::max(0, std::min(real_x, images[augmented_images_index[i]].cols - 1)); // which cols
			real_y = std::max(0, std::min(real_y, images[augmented_images_index[i]].rows - 1)); // which rows
			int tmp = (int)images[augmented_images_index[i]](real_y, real_x); //real_y at first

			delta_x = rotation(0, 0)*pos.end.x + rotation(0, 1)*pos.end.y;
			delta_y = rotation(1, 0)*pos.end.x + rotation(1, 1)*pos.end.y;
			delta_x = scale*delta_x*augmented_bboxes[i].width / 2.0;
			delta_y = scale*delta_y*augmented_bboxes[i].height / 2.0;
			real_x = delta_x + augmented_current_shapes[i](landmark_index_, 0);
			real_y = delta_y + augmented_current_shapes[i](landmark_index_, 1);
			real_x = std::max(0, std::min(real_x, images[augmented_images_index[i]].cols - 1)); // which cols
			real_y = std::max(0, std::min(real_y, images[augmented_images_index[i]].rows - 1)); // which rows
			pixel_differences(j, i) = tmp - (int)images[augmented_images_index[i]](real_y, real_x);
		}
	}
	// train Random Forest
	// construct each tree in the forest

	double overlap = 0.3;
	if (overlap_ >= 1.0 && overlap_ <= 0) {
		std::cout << "your setting of overlap is not in the range (0, 1)\n setting overlap to 0.3" << std::endl;
	}
	else {
		overlap = overlap_;
	}
	int step = floor(((double)augmented_images_index.size())*overlap / (trees_num_per_forest_ - 1));
	trees_.clear();
	all_leaf_nodes_ = 0;
	for (int i = 0; i < trees_num_per_forest_; i++){
		int start_index = i*step;
		int end_index = augmented_images_index.size() - (trees_num_per_forest_ - i - 1)*step;
		//cv::Mat_<int> data = pixel_differences(cv::Range(0, local_features_num_), cv::Range(start_index, end_index));
		//cv::Mat_<int> sorted_data;
		//cv::sortIdx(data, sorted_data, cv::SORT_EVERY_ROW + cv::SORT_ASCENDING);
		std::set<int> selected_indexes;
		std::vector<int> images_indexes;
		for (int j = start_index; j < end_index; j++){
			images_indexes.push_back(j);
		}
		Node* root = BuildTree(selected_indexes, pixel_differences, images_indexes, 0);
		trees_.push_back(root);
	}
	/*int count = 0;
	for (int i = 0; i < trees_num_per_forest_; i++){
		Node* root = trees_[i];
		count = MarkLeafIdentity(root, count);
	}
	all_leaf_nodes_ = count;*/
	return true;
}


Node* RandomForest::BuildTree(std::set<int>& selected_indexes, cv::Mat_<int>& pixel_differences, std::vector<int>& images_indexes, int current_depth){
	if (images_indexes.size() > 0){ // the node may not split under some cases
		Node* node = new Node();
		node->depth_ = current_depth;
		node->samples_ = images_indexes.size();
		std::vector<int> left_indexes, right_indexes;
		if (current_depth == tree_depth_){ // the node reaches max depth
			node->is_leaf_ = true;
			node->leaf_identity = all_leaf_nodes_;
			all_leaf_nodes_++;
			return node;
		}

		int ret = FindSplitFeature(node, selected_indexes, pixel_differences, images_indexes, left_indexes, right_indexes);
		// actually it won't enter the if block, when the random function is good enough
		if (ret == 1){ // the current node contain all sample when reaches max variance reduction, it is leaf node
			node->is_leaf_ = true;
			node->leaf_identity = all_leaf_nodes_;
			all_leaf_nodes_++;
			return node;
		}

		//if (current_depth + 1 < tree_depth_){
		node->left_child_ = BuildTree(selected_indexes, pixel_differences, left_indexes, current_depth + 1);
		node->right_child_ = BuildTree(selected_indexes, pixel_differences, right_indexes, current_depth + 1);
		//}
		return node;
	}
	else{ // this case is not possible in this data structure
		return NULL;
	}
}

int RandomForest::FindSplitFeature(Node* node, std::set<int>& selected_indexes,
	cv::Mat_<int>& pixel_differences, std::vector<int>& images_indexes, std::vector<int>& left_indexes, std::vector<int>& right_indexes){
	std::vector<int> val;
	//cv::Mat_<int> sorted_fea;
	time_t current_time;
	current_time = time(0);
	cv::RNG rd(current_time);
	int threshold;
	double var = -1000000000000.0; // use -DBL_MAX will be better
	int feature_index = -1;
	std::vector<int> tmp_left_indexes, tmp_right_indexes;
	//int j = 0, tmp_index;
	for (int j = 0; j < local_features_num_; j++){
		if (selected_indexes.find(j) == selected_indexes.end()){
			tmp_left_indexes.clear();
			tmp_right_indexes.clear();
			double var_lc = 0.0, var_rc = 0.0, var_red = 0.0;
			double Ex_2_lc = 0.0, Ex_lc = 0.0, Ey_2_lc = 0.0, Ey_lc = 0.0;
			double Ex_2_rc = 0.0, Ex_rc = 0.0, Ey_2_rc = 0.0, Ey_rc = 0.0;
			// random generate threshold
			std::vector<int> data;
			data.reserve(images_indexes.size());
			for (int i = 0; i < images_indexes.size(); i++){
				data.push_back(pixel_differences(j, images_indexes[i]));
			}
			std::sort(data.begin(), data.end());
			int tmp_index = floor((int)(images_indexes.size()*(0.5 + 0.9*(rd.uniform(0.0, 1.0) - 0.5))));
			int tmp_threshold = data[tmp_index];
			for (int i = 0; i < images_indexes.size(); i++){
				int index = images_indexes[i];
				if (pixel_differences(j, index) < tmp_threshold){
					tmp_left_indexes.push_back(index);
					// do with regression target
					double value = regression_targets_->at(index)(landmark_index_, 0);
					Ex_2_lc += pow(value, 2);
					Ex_lc += value;
					value = regression_targets_->at(index)(landmark_index_, 1);
					Ey_2_lc += pow(value, 2);
					Ey_lc += value;
				}
				else{
					tmp_right_indexes.push_back(index);
					double value = regression_targets_->at(index)(landmark_index_, 0);
					Ex_2_rc += pow(value, 2);
					Ex_rc += value;
					value = regression_targets_->at(index)(landmark_index_, 1);
					Ey_2_rc += pow(value, 2);
					Ey_rc += value;
				}
			}
			if (tmp_left_indexes.size() == 0){
				var_lc = 0.0;
			} else{
				var_lc = Ex_2_lc / tmp_left_indexes.size() - pow(Ex_lc / tmp_left_indexes.size(), 2)
					+ Ey_2_lc / tmp_left_indexes.size() - pow(Ey_lc / tmp_left_indexes.size(), 2);
			}
			if (tmp_right_indexes.size() == 0){
				var_rc = 0.0;
			} else{
				var_rc = Ex_2_rc / tmp_right_indexes.size() - pow(Ex_rc / tmp_right_indexes.size(), 2)
					+ Ey_2_rc / tmp_right_indexes.size() - pow(Ey_rc / tmp_right_indexes.size(), 2);
			}
			var_red = -var_lc*tmp_left_indexes.size() - var_rc*tmp_right_indexes.size();
			if (var_red > var){
				var = var_red;
				threshold = tmp_threshold;
				feature_index = j;
				left_indexes = tmp_left_indexes;
				right_indexes = tmp_right_indexes;
			}
		}
	}
	if (feature_index != -1) // actually feature_index will never be -1
	{
		if (left_indexes.size() == 0 || right_indexes.size() == 0){
			node->is_leaf_ = true; // the node can contain all the samples
			return 1;
		}
		node->threshold_ = threshold;
		node->thre_changed_ = true;
		node->feature_locations_ = local_position_[feature_index];
		selected_indexes.insert(feature_index);
		return 0;
	}

	return -1;
}

int RandomForest::MarkLeafIdentity(Node* node, int count){
	std::stack<Node*> s;
	Node* p_current = node;

	if (node == NULL){
		return count;
	}
	// the node in the tree is either leaf node or internal node that has both left and right children
	while (1)//p_current || !s.empty())
	{

		if (p_current->is_leaf_){
			p_current->leaf_identity = count;
			count++;
			if (s.empty()){
				return count;
			}
			p_current = s.top()->right_child_;
			s.pop();
		}
		else{
			s.push(p_current);
			p_current = p_current->left_child_;
		}

		/*while (!p_current && !s.empty()){
			p_current = s.top();
			s.pop();
			p_current = p_current->right_child_;
		}*/
	}

}

cv::Mat_<double> RandomForest::GetBinaryFeatures(const cv::Mat_<double>& image,
	const BoundingBox& bbox, const cv::Mat_<double>& current_shape, const cv::Mat_<double>& rotation, const double& scale){
	cv::Mat_<double> res(1, all_leaf_nodes_, 0.0);
	for (int i = 0; i < trees_num_per_forest_; i++){
		Node* node = trees_[i];
		while (!node->is_leaf_){
			int direction = GetNodeOutput(node, image, bbox, current_shape, rotation, scale);
			if (direction == -1){
				node = node->left_child_;
			}
			else{
				node = node->right_child_;
			}
		}
		res(0, node->leaf_identity) = 1.0;
	}
	return res;
}

int RandomForest::GetBinaryFeatureIndex(int tree_index, const cv::Mat_<double>& image,
	const BoundingBox& bbox, const cv::Mat_<double>& current_shape, const cv::Mat_<double>& rotation, const double& scale){
	Node* node = trees_[tree_index];
	while (!node->is_leaf_){
		FeatureLocations& pos = node->feature_locations_;
		double delta_x = rotation(0, 0)*pos.start.x + rotation(0, 1)*pos.start.y;
		double delta_y = rotation(1, 0)*pos.start.x + rotation(1, 1)*pos.start.y;
		delta_x = scale*delta_x*bbox.width / 2.0;
		delta_y = scale*delta_y*bbox.height / 2.0;
		int real_x = delta_x + current_shape(landmark_index_, 0);
		int real_y = delta_y + current_shape(landmark_index_, 1);
		real_x = std::max(0, std::min(real_x, image.cols - 1)); // which cols
		real_y = std::max(0, std::min(real_y, image.rows - 1)); // which rows
		int tmp = (int)image(real_y, real_x); //real_y at first

		delta_x = rotation(0, 0)*pos.end.x + rotation(0, 1)*pos.end.y;
		delta_y = rotation(1, 0)*pos.end.x + rotation(1, 1)*pos.end.y;
		delta_x = scale*delta_x*bbox.width / 2.0;
		delta_y = scale*delta_y*bbox.height / 2.0;
		real_x = delta_x + current_shape(landmark_index_, 0);
		real_y = delta_y + current_shape(landmark_index_, 1);
		real_x = std::max(0, std::min(real_x, image.cols - 1)); // which cols
		real_y = std::max(0, std::min(real_y, image.rows - 1)); // which rows
		if ((tmp - (int)image(real_y, real_x)) < node->threshold_){
			node = node->left_child_;// go left
		}
		else{
			node = node->right_child_;// go right
		}
	}
	return node->leaf_identity;
}


int RandomForest::GetNodeOutput(Node* node, const cv::Mat_<double>& image,
	const BoundingBox& bbox, const cv::Mat_<double>& current_shape, const cv::Mat_<double>& rotation, const double& scale){

	FeatureLocations& pos = node->feature_locations_;
	double delta_x = rotation(0, 0)*pos.start.x + rotation(0, 1)*pos.start.y;
	double delta_y = rotation(1, 0)*pos.start.x + rotation(1, 1)*pos.start.y;
	delta_x = scale*delta_x*bbox.width / 2.0;
	delta_y = scale*delta_y*bbox.height / 2.0;
	int real_x = delta_x + current_shape(landmark_index_, 0);
	int real_y = delta_y + current_shape(landmark_index_, 1);
	real_x = std::max(0, std::min(real_x, image.cols - 1)); // which cols
	real_y = std::max(0, std::min(real_y, image.rows - 1)); // which rows
	int tmp = (int)image(real_y, real_x); //real_y at first

	delta_x = rotation(0, 0)*pos.end.x + rotation(0, 1)*pos.end.y;
	delta_y = rotation(1, 0)*pos.end.x + rotation(1, 1)*pos.end.y;
	delta_x = scale*delta_x*bbox.width / 2.0;
	delta_y = scale*delta_y*bbox.height / 2.0;
	real_x = delta_x + current_shape(landmark_index_, 0);
	real_y = delta_y + current_shape(landmark_index_, 1);
	real_x = std::max(0, std::min(real_x, image.cols - 1)); // which cols
	real_y = std::max(0, std::min(real_y, image.rows - 1)); // which rows
	if ((tmp - (int)image(real_y, real_x)) < node->threshold_){
		return -1; // go left
	}
	else{
		return 1; // go right
	}

}

RandomForest::RandomForest(Parameters& param, int landmark_index, int stage, std::vector<cv::Mat_<double> >& regression_targets){
	stage_ = stage;
	local_features_num_ = param.local_features_num_;
	landmark_index_ = landmark_index;
	tree_depth_ = param.tree_depth_;
	trees_num_per_forest_ = param.trees_num_per_forest_;
	local_radius_ = param.local_radius_by_stage_[stage_];
	overlap_ = param.overlap_;
	//mean_shape_ = param.mean_shape_;
	regression_targets_ = &regression_targets; // get the address pointer, not reference
}

RandomForest::RandomForest(){

}

void RandomForest::SaveRandomForest(std::ofstream& fout){
	fout << stage_ << " "
		<< local_features_num_ << " "
		<< landmark_index_ << " "
		<< tree_depth_ << " "
		<< trees_num_per_forest_ << " "
		<< local_radius_ << " "
		<< all_leaf_nodes_ << " "
		<< trees_.size() << std::endl;
	for (int i = 0; i < trees_.size(); i++){
		Node* root = trees_[i];
		WriteTree(root, fout);
	}
}

void RandomForest::WriteTree(Node* p, std::ofstream& fout){
	if (!p){
		fout << "#" << std::endl;
	}
	else{
		fout <<"Y" << " "
			<< p->threshold_ << " "
			<< p->is_leaf_ << " "
			<< p->leaf_identity << " "
			<< p->depth_ << " "
			<< p->feature_locations_.start.x << " "
			<< p->feature_locations_.start.y << " "
			<< p->feature_locations_.end.x << " "
			<< p->feature_locations_.end.y << std::endl;
		WriteTree(p->left_child_, fout);
		WriteTree(p->right_child_, fout);
	}
}

Node* RandomForest::ReadTree(std::ifstream& fin){
	std::string flag;
	fin >> flag;
	if (flag == "Y"){
		Node* p = new Node();
		fin >> p->threshold_
			>> p->is_leaf_
			>> p->leaf_identity
			>> p->depth_
			>> p->feature_locations_.start.x
			>> p->feature_locations_.start.y
			>> p->feature_locations_.end.x
			>> p->feature_locations_.end.y;
		p->left_child_ = ReadTree(fin);
		p->right_child_ = ReadTree(fin);
		return p;
	}
	else{
		return NULL;
	}
}

void RandomForest::LoadRandomForest(std::ifstream& fin){

	int tree_size;
	fin >> stage_
		>> local_features_num_
		>> landmark_index_
		>> tree_depth_
		>> trees_num_per_forest_
		>> local_radius_
		>> all_leaf_nodes_
		>> tree_size;
	std::string start_flag;
	trees_.clear();
	for (int i = 0; i < tree_size; i++){
		Node* root = ReadTree(fin);
		trees_.push_back(root);
	}
}
