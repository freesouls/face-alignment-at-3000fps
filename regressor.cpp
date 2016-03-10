#include "regressor.h"
#include <time.h>
#include <assert.h>
//SYSTEM MACORS LISTS: http://sourceforge.net/p/predef/wiki/OperatingSystems/
#ifdef _WIN32 // can be used under 32 and 64 bits both
#include <direct.h>
#elif __linux__
#include <sys/types.h>
#include <sys/stat.h>
#endif
CascadeRegressor::CascadeRegressor(){

}

void CascadeRegressor::Train(const std::vector<cv::Mat_<uchar> >& images,
	const std::vector<cv::Mat_<double> >& ground_truth_shapes,
	const std::vector<BoundingBox>& bboxes,
	Parameters& params){

	std::cout << "Start training..." << std::endl;
	images_ = images;
	params_ = params;
	bboxes_ = bboxes;
	ground_truth_shapes_ = ground_truth_shapes;

	std::vector<int> augmented_images_index; // just index in images_
	std::vector<BoundingBox> augmented_bboxes;
	std::vector<cv::Mat_<double> > augmented_ground_truth_shapes;
	std::vector<cv::Mat_<double> > augmented_current_shapes; //

	time_t current_time;
	current_time = time(0);
	//cv::RNG *random_generator = new cv::RNG();
	std::cout << "augment data sets" << std::endl;
	cv::RNG random_generator(current_time);
	for (int i = 0; i < images_.size(); i++){
		for (int j = 0; j < params_.initial_guess_; j++)
		{
			int index = 0;
			do {
				index = random_generator.uniform(0, images_.size());
			}while(index == i);

			augmented_images_index.push_back(i);
			augmented_ground_truth_shapes.push_back(ground_truth_shapes_[i]);
			augmented_bboxes.push_back(bboxes_[i]);
			cv::Mat_<double> temp = ground_truth_shapes_[index];
			temp = ProjectShape(temp, bboxes_[index]);
			temp = ReProjection(temp, bboxes_[i]);
			augmented_current_shapes.push_back(temp);
		}
        augmented_images_index.push_back(i);
        augmented_ground_truth_shapes.push_back(ground_truth_shapes_[i]);
        augmented_bboxes.push_back(bboxes_[i]);
        augmented_current_shapes.push_back(ReProjection(params_.mean_shape_, bboxes_[i]));
	}
	
	std::cout << "augmented size: " << augmented_current_shapes.size() << std::endl;

	std::vector<cv::Mat_<double> > shape_increaments;
	regressors_.resize(params_.regressor_stages_);
	for (int i = 0; i < params_.regressor_stages_; i++){
		std::cout << "training stage: " << i << " of " << params_.regressor_stages_ << std::endl;
		shape_increaments = regressors_[i].Train(images_,
											augmented_images_index,
											augmented_ground_truth_shapes,
											augmented_bboxes,
											augmented_current_shapes,
											params_,
											i);
		std::cout << "update current shapes" << std::endl;
		double error = 0.0;
		for (int j = 0; j < shape_increaments.size(); j++){
			augmented_current_shapes[j] = shape_increaments[j] + ProjectShape(augmented_current_shapes[j], augmented_bboxes[j]);
			augmented_current_shapes[j] = ReProjection(augmented_current_shapes[j], augmented_bboxes[j]);
			error += CalculateError(augmented_ground_truth_shapes[j], augmented_current_shapes[j]);
		}

        std::cout << "train regression error: " <<  error << ", mean error: " << error/shape_increaments.size() << std::endl;
		if (val_images_.size() > 0) { // check if validation set is add
			Validation(i);
		}
	}
}

std::vector<cv::Mat_<double> > Regressor::Train(const std::vector<cv::Mat_<uchar> >& images,
	const std::vector<int>& augmented_images_index,
	const std::vector<cv::Mat_<double> >& augmented_ground_truth_shapes,
	const std::vector<BoundingBox>& augmented_bboxes,
	const std::vector<cv::Mat_<double> >& augmented_current_shapes,
	const Parameters& params,
	const int stage){

	stage_ = stage;
	params_ = params;

	std::vector<cv::Mat_<double> > regression_targets;
	std::vector<cv::Mat_<double> > rotations_;
	std::vector<double> scales_;
	regression_targets.resize(augmented_current_shapes.size());
	rotations_.resize(augmented_current_shapes.size());
	scales_.resize(augmented_current_shapes.size());

	// calculate the regression targets
	std::cout << "calculate regression targets" << std::endl;
    #pragma omp parallel for
	for (int i = 0; i < augmented_current_shapes.size(); i++){
		regression_targets[i] = ProjectShape(augmented_ground_truth_shapes[i], augmented_bboxes[i])
			- ProjectShape(augmented_current_shapes[i], augmented_bboxes[i]);
		cv::Mat_<double> rotation;
		double scale;
		getSimilarityTransform(params_.mean_shape_, ProjectShape(augmented_current_shapes[i], augmented_bboxes[i]), rotation, scale);
		cv::transpose(rotation, rotation);
		regression_targets[i] = scale * regression_targets[i] * rotation;
		getSimilarityTransform(ProjectShape(augmented_current_shapes[i], augmented_bboxes[i]), params_.mean_shape_, rotation, scale);
		rotations_[i] = rotation;
		scales_[i] = scale;
	}

	std::cout << "train forest of stage:" << stage_ << std::endl;
	std::cout << "it will take some time to build the Random Forest, please be patient!!!" << std::endl;
	rd_forests_.resize(params_.landmarks_num_per_face_);
    #pragma omp parallel for
	for (int i = 0; i < params_.landmarks_num_per_face_; ++i){
        // std::cout << "landmark: " << i << std::endl;
		rd_forests_[i] = RandomForest(params_, i, stage_, regression_targets);
        rd_forests_[i].TrainForest(
			images,augmented_images_index, augmented_bboxes, augmented_current_shapes,
			rotations_, scales_);
	}
	std::cout << "Get Global Binary Features" << std::endl;

    struct feature_node **global_binary_features;
    global_binary_features = new struct feature_node* [augmented_current_shapes.size()];

    for(int i = 0; i < augmented_current_shapes.size(); ++i){
        global_binary_features[i] = new feature_node[params_.trees_num_per_forest_*params_.landmarks_num_per_face_+1];
    }
    int num_feature = 0;
    for (int i=0; i < params_.landmarks_num_per_face_; ++i){
        num_feature += rd_forests_[i].all_leaf_nodes_;
    }
    #pragma omp parallel for
    for (int i = 0; i < augmented_current_shapes.size(); ++i){
        int index = 1;
        int ind = 0;
        const cv::Mat_<double>& rotation = rotations_[i];
        const double scale = scales_[i];
        const cv::Mat_<uchar>& image = images[augmented_images_index[i]];
        const BoundingBox& bbox = augmented_bboxes[i];
        const cv::Mat_<double>& current_shape = augmented_current_shapes[i];
    	for (int j = 0; j < params_.landmarks_num_per_face_; ++j){
    		for (int k = 0; k < params_.trees_num_per_forest_; ++k){

                Node* node = rd_forests_[j].trees_[k];
                while (!node->is_leaf_){
                    FeatureLocations& pos = node->feature_locations_;
                    double delta_x = rotation(0, 0)*pos.start.x + rotation(0, 1)*pos.start.y;
                    double delta_y = rotation(1, 0)*pos.start.x + rotation(1, 1)*pos.start.y;
                    delta_x = scale*delta_x*bbox.width / 2.0;
                    delta_y = scale*delta_y*bbox.height / 2.0;
                    int real_x = delta_x + current_shape(j, 0);
                    int real_y = delta_y + current_shape(j, 1);
                    real_x = std::max(0, std::min(real_x, image.cols - 1)); // which cols
                    real_y = std::max(0, std::min(real_y, image.rows - 1)); // which rows
                    int tmp = (int)image(real_y, real_x); //real_y at first

                    delta_x = rotation(0, 0)*pos.end.x + rotation(0, 1)*pos.end.y;
                    delta_y = rotation(1, 0)*pos.end.x + rotation(1, 1)*pos.end.y;
                    delta_x = scale*delta_x*bbox.width / 2.0;
                    delta_y = scale*delta_y*bbox.height / 2.0;
                    real_x = delta_x + current_shape(j, 0);
                    real_y = delta_y + current_shape(j, 1);
                    real_x = std::max(0, std::min(real_x, image.cols - 1)); // which cols
                    real_y = std::max(0, std::min(real_y, image.rows - 1)); // which rows
                    if ((tmp - (int)image(real_y, real_x)) < node->threshold_){
                        node = node->left_child_;// go left
                    }
                    else{
                        node = node->right_child_;// go right
                    }
                }
                global_binary_features[i][ind].index = index + node->leaf_identity;//rd_forests_[j].GetBinaryFeatureIndex(k, images[augmented_images_index[i]], augmented_bboxes[i], augmented_current_shapes[i], rotations_[i], scales_[i]);
    			global_binary_features[i][ind].value = 1.0;
                ind++;
                //std::cout << global_binary_features[i][ind].index << " ";
    		}
            index += rd_forests_[j].all_leaf_nodes_;
    	}
        // if (i%500 == 0 && i > 0){
        //     std::cout << "extracted " << i << " images" << std::endl;
        // }
        global_binary_features[i][params_.trees_num_per_forest_*params_.landmarks_num_per_face_].index = -1;
        global_binary_features[i][params_.trees_num_per_forest_*params_.landmarks_num_per_face_].value = -1.0;
    }
    std::cout << "\n";

	struct problem* prob = new struct problem;
	prob->l = augmented_current_shapes.size();
    prob->n = num_feature;
    prob->x = global_binary_features;
    prob->bias = -1;

    struct parameter* regression_params = new struct parameter;
    regression_params-> solver_type = L2R_L2LOSS_SVR_DUAL;
    regression_params->C = 1.0/augmented_current_shapes.size();
    regression_params->p = 0;

	std::cout << "Global Regression of stage " << stage_ << std::endl;
    linear_model_x_.resize(params_.landmarks_num_per_face_);
    linear_model_y_.resize(params_.landmarks_num_per_face_);
    double** targets = new double*[params_.landmarks_num_per_face_];
    for (int i = 0; i < params_.landmarks_num_per_face_; ++i){
        targets[i] = new double[augmented_current_shapes.size()];
    }
	std::cout << "it will take some time to do Linear Regression, please be patient!!!" << std::endl;
    #pragma omp parallel for
    for (int i = 0; i < params_.landmarks_num_per_face_; ++i){

        // std::cout << "regress landmark " << i << std::endl;
		if (i%8==0) {
			std::cout << "regressing ..." << i << std::endl;
		}
        for(int j = 0; j< augmented_current_shapes.size();j++){
            targets[i][j] = regression_targets[j](i, 0);
        }
        prob->y = targets[i];
        check_parameter(prob, regression_params);
        struct model* regression_model = train(prob, regression_params);
        linear_model_x_[i] = regression_model;
        for(int j = 0; j < augmented_current_shapes.size(); j++){
            targets[i][j] = regression_targets[j](i, 1);
        }
        prob->y = targets[i];
        check_parameter(prob, regression_params);
        regression_model = train(prob, regression_params);
        linear_model_y_[i] = regression_model;

    }
    for (int i = 0; i < params_.landmarks_num_per_face_; ++i){
        delete[] targets[i];// = new double[augmented_current_shapes.size()];
    }
    delete[] targets;
	std::cout << "predict regression targets" << std::endl;

    std::vector<cv::Mat_<double> > predict_regression_targets;
    predict_regression_targets.resize(augmented_current_shapes.size());
    #pragma omp parallel for
    for (int i = 0; i < augmented_current_shapes.size(); i++){
        cv::Mat_<double> a(params_.landmarks_num_per_face_, 2, 0.0);
        for (int j = 0; j < params_.landmarks_num_per_face_; j++){
            a(j, 0) = predict(linear_model_x_[j], global_binary_features[i]);
            a(j, 1) = predict(linear_model_y_[j], global_binary_features[i]);
        }
        cv::Mat_<double> rot;
        cv::transpose(rotations_[i], rot);
        predict_regression_targets[i] = scales_[i] * a * rot;
        // if (i%500 == 0 && i > 0){
        //      std::cout << "predict " << i << " images" << std::endl;
        // }
    }
    // std::cout << "\n";


    for (int i = 0; i< augmented_current_shapes.size(); i++){
        delete[] global_binary_features[i];
    }
    delete[] global_binary_features;

	return predict_regression_targets;
}


cv::Mat_<double> CascadeRegressor::Predict(cv::Mat_<uchar>& image,
	cv::Mat_<double>& current_shape, BoundingBox& bbox, cv::Mat_<double>& ground_truth_shape){

	cv::Mat_<uchar> tmp;
	image.copyTo(tmp);

	for (int j = 0; j < current_shape.rows; j++){
		cv::circle(tmp, cv::Point2f(current_shape(j, 0), current_shape(j, 1)), 2, (255));
	}
	cv::imshow("show image", tmp);
	cv::waitKey(0);

	for (int i = 0; i < params_.regressor_stages_; i++){

		cv::Mat_<double> rotation;
		double scale;
		// if(i==0){
			// getSimilarityTransform(ProjectShape(ground_truth_shape, bbox), params_.mean_shape_, rotation, scale);
		// }else{
			getSimilarityTransform(ProjectShape(current_shape, bbox), params_.mean_shape_, rotation, scale);
		// }

		cv::Mat_<double> shape_increaments = regressors_[i].Predict(image, current_shape, bbox, rotation, scale);
		current_shape = shape_increaments + ProjectShape(current_shape, bbox);
		current_shape = ReProjection(current_shape, bbox);
		image.copyTo(tmp);
		for (int j = 0; j < current_shape.rows; j++){
			cv::circle(tmp, cv::Point2f(current_shape(j, 0), current_shape(j, 1)), 2, (255));
		}
		cv::imshow("show image", tmp);
		cv::waitKey(0);
	}
	cv::Mat_<double> res = current_shape;
	return res;
}


cv::Mat_<double> CascadeRegressor::Predict(cv::Mat_<uchar>& image,
	cv::Mat_<double>& current_shape, BoundingBox& bbox, int stage, bool is_train){
    int stages = is_train ? stage+1 : params_.regressor_stages_;
	for (int i = 0; i < stages; i++){
        cv::Mat_<double> rotation;
		double scale;
		getSimilarityTransform(ProjectShape(current_shape, bbox), params_.mean_shape_, rotation, scale);
		cv::Mat_<double> shape_increaments = regressors_[i].Predict(image, current_shape, bbox, rotation, scale);
		current_shape = shape_increaments + ProjectShape(current_shape, bbox);
		current_shape = ReProjection(current_shape, bbox);
	}
	cv::Mat_<double> res = current_shape;
	return res;
}

void CascadeRegressor::Validation(int stage) {
    std::cout << "Validation at stage: " << stage << std::endl;
    double error = 0.0;
    for (int i = 0; i < val_images_.size(); i++) {
        cv::Mat_<double> current_shape = ReProjection(params_.mean_shape_, val_bboxes_[i]);
        cv::Mat_<double> res = Predict(val_images_[i], current_shape, val_bboxes_[i], stage, true);
        error += CalculateError(val_ground_truth_shapes_[i], res);
    }
    std::cout << "Validation error: " << error << ", mean error: " << error/val_images_.size() << std::endl;
}


Regressor::Regressor(){
}

Regressor::Regressor(const Regressor &a){
}

Regressor::~Regressor(){

}
/*
struct feature_node* Regressor::GetGlobalBinaryFeaturesThread(cv::Mat_<uchar>& image,
    cv::Mat_<double>& current_shape, BoundingBox& bbox, cv::Mat_<double>& rotation, double scale){
    struct feature_node* binary_features = new feature_node[params_.trees_num_per_forest_*params_.landmarks_num_per_face_+1];
    tmp_binary_features = binary_features;
    tmp_image = image;
    tmp_current_shape = current_shape;
    tmp_bbox = bbox;
    tmp_rotation = rotation;
    tmp_scale = scale;
    // cur_landmark.store(0);


    int num_threads = 2;
    std::thread t1, t2;
    std::vector<std::thread> pool;
    //struct timeval tt1, tt2;
    //gettimeofday(&tt1, NULL);
    for(int i = 0; i < num_threads; i++){
        //t1 = std::thread(&Regressor::GetFeaThread, this);
        pool.push_back(std::thread(&Regressor::GetFeaThread, this));
    }
    //gettimeofday(&tt2, NULL);
    //std::cout << "threads: " << tt2.tv_sec - tt1.tv_sec + (tt2.tv_usec - tt1.tv_usec)/1000000.0 << std::endl;

    for(int i = 0; i < num_threads; i++){
        pool[i].join();
    }

    binary_features[params_.trees_num_per_forest_*params_.landmarks_num_per_face_].index = -1;
    binary_features[params_.trees_num_per_forest_*params_.landmarks_num_per_face_].value = -1.0;

    return binary_features;
}
*/
/*
void Regressor::GetFeaThread(){
    int cur = -1;
    while(1){
        cur = cur_landmark.fetch_add(1);
        if(cur >= params_.landmarks_num_per_face_){
            return;
        }
        //std::cout << stage_ << ": " << cur << std::endl;
        int ind = cur*params_.trees_num_per_forest_;
        for (int k = 0; k < params_.trees_num_per_forest_; ++k)
        {
            Node* node = rd_forests_[cur].trees_[k];
            while (!node->is_leaf_){
                FeatureLocations& pos = node->feature_locations_;
                double delta_x = tmp_rotation(0, 0)*pos.start.x + tmp_rotation(0, 1)*pos.start.y;
                double delta_y = tmp_rotation(1, 0)*pos.start.x + tmp_rotation(1, 1)*pos.start.y;
                delta_x = tmp_scale*delta_x*tmp_bbox.width / 2.0;
                delta_y = tmp_scale*delta_y*tmp_bbox.height / 2.0;
                int real_x = delta_x + tmp_current_shape(cur, 0);
                int real_y = delta_y + tmp_current_shape(cur, 1);
                real_x = std::max(0, std::min(real_x, tmp_image.cols - 1)); // which cols
                real_y = std::max(0, std::min(real_y, tmp_image.rows - 1)); // which rows
                int tmp = (int)tmp_image(real_y, real_x); //real_y at first

                delta_x = tmp_rotation(0, 0)*pos.end.x + tmp_rotation(0, 1)*pos.end.y;
                delta_y = tmp_rotation(1, 0)*pos.end.x + tmp_rotation(1, 1)*pos.end.y;
                delta_x = tmp_scale*delta_x*tmp_bbox.width / 2.0;
                delta_y = tmp_scale*delta_y*tmp_bbox.height / 2.0;
                real_x = delta_x + tmp_current_shape(cur, 0);
                real_y = delta_y + tmp_current_shape(cur, 1);
                real_x = std::max(0, std::min(real_x, tmp_image.cols - 1)); // which cols
                real_y = std::max(0, std::min(real_y, tmp_image.rows - 1)); // which rows
                if ((tmp - (int)tmp_image(real_y, real_x)) < node->threshold_){
                    node = node->left_child_;// go left
                }
                else{
                    node = node->right_child_;// go right
                }
            }

            //int ind = j*params_.trees_num_per_forest_ + k;
            tmp_binary_features[ind].index = leaf_index_count[cur] + node->leaf_identity;//rd_forests_[j].GetBinaryFeatureIndex(k,image, bbox, current_shape, rotation, scale);
            tmp_binary_features[ind].value = 1.0;
            ind++;
            //std::cout << binary_features[ind].index << " ";
        }
    }
}
*/
struct feature_node* Regressor::GetGlobalBinaryFeaturesMP(cv::Mat_<uchar>& image,
    cv::Mat_<double>& current_shape, BoundingBox& bbox, cv::Mat_<double>& rotation, double scale){
    int index = 1;

    struct feature_node* binary_features = new feature_node[params_.trees_num_per_forest_*params_.landmarks_num_per_face_+1];
    //int ind = 0;
#pragma omp parallel for
    for (int j = 0; j < params_.landmarks_num_per_face_; ++j)
    {
        for (int k = 0; k < params_.trees_num_per_forest_; ++k)
        {
            Node* node = rd_forests_[j].trees_[k];
            while (!node->is_leaf_){
                FeatureLocations& pos = node->feature_locations_;
                double delta_x = rotation(0, 0)*pos.start.x + rotation(0, 1)*pos.start.y;
                double delta_y = rotation(1, 0)*pos.start.x + rotation(1, 1)*pos.start.y;
                delta_x = scale*delta_x*bbox.width / 2.0;
                delta_y = scale*delta_y*bbox.height / 2.0;
                int real_x = delta_x + current_shape(j, 0);
                int real_y = delta_y + current_shape(j, 1);
                real_x = std::max(0, std::min(real_x, image.cols - 1)); // which cols
                real_y = std::max(0, std::min(real_y, image.rows - 1)); // which rows
                int tmp = (int)image(real_y, real_x); //real_y at first

                delta_x = rotation(0, 0)*pos.end.x + rotation(0, 1)*pos.end.y;
                delta_y = rotation(1, 0)*pos.end.x + rotation(1, 1)*pos.end.y;
                delta_x = scale*delta_x*bbox.width / 2.0;
                delta_y = scale*delta_y*bbox.height / 2.0;
                real_x = delta_x + current_shape(j, 0);
                real_y = delta_y + current_shape(j, 1);
                real_x = std::max(0, std::min(real_x, image.cols - 1)); // which cols
                real_y = std::max(0, std::min(real_y, image.rows - 1)); // which rows
                if ((tmp - (int)image(real_y, real_x)) < node->threshold_){
                    node = node->left_child_;// go left
                }
                else{
                    node = node->right_child_;// go right
                }
            }

            //int ind = j*params_.trees_num_per_forest_ + k;
            int ind = feature_node_index[j] + k;
            binary_features[ind].index = leaf_index_count[j] + node->leaf_identity;
            //binary_features[ind].index = index + node->leaf_identity;//rd_forests_[j].GetBinaryFeatureIndex(k,image, bbox, current_shape, rotation, scale);
            binary_features[ind].value = 1.0;
            //ind++;
            //std::cout << binary_features[ind].index << " ";
        }

        //index += rd_forests_[j].all_leaf_nodes_;
    }
    //std::cout << "\n";
    //std::cout << index << ":" << params_.trees_num_per_forest_*params_.landmarks_num_per_face_ << std::endl;
    binary_features[params_.trees_num_per_forest_*params_.landmarks_num_per_face_].index = -1;
    binary_features[params_.trees_num_per_forest_*params_.landmarks_num_per_face_].value = -1.0;
    return binary_features;
}

struct feature_node* Regressor::GetGlobalBinaryFeatures(cv::Mat_<uchar>& image,
    cv::Mat_<double>& current_shape, BoundingBox& bbox, cv::Mat_<double>& rotation, double scale){
    int index = 1;

    struct feature_node* binary_features = new feature_node[params_.trees_num_per_forest_*params_.landmarks_num_per_face_+1];
    int ind = 0;
    for (int j = 0; j < params_.landmarks_num_per_face_; ++j)
    {
        for (int k = 0; k < params_.trees_num_per_forest_; ++k)
        {
            Node* node = rd_forests_[j].trees_[k];
            while (!node->is_leaf_){
                FeatureLocations& pos = node->feature_locations_;
                double delta_x = rotation(0, 0)*pos.start.x + rotation(0, 1)*pos.start.y;
                double delta_y = rotation(1, 0)*pos.start.x + rotation(1, 1)*pos.start.y;
                delta_x = scale*delta_x*bbox.width / 2.0;
                delta_y = scale*delta_y*bbox.height / 2.0;
                int real_x = delta_x + current_shape(j, 0);
                int real_y = delta_y + current_shape(j, 1);
                real_x = std::max(0, std::min(real_x, image.cols - 1)); // which cols
                real_y = std::max(0, std::min(real_y, image.rows - 1)); // which rows
                int tmp = (int)image(real_y, real_x); //real_y at first

                delta_x = rotation(0, 0)*pos.end.x + rotation(0, 1)*pos.end.y;
                delta_y = rotation(1, 0)*pos.end.x + rotation(1, 1)*pos.end.y;
                delta_x = scale*delta_x*bbox.width / 2.0;
                delta_y = scale*delta_y*bbox.height / 2.0;
                real_x = delta_x + current_shape(j, 0);
                real_y = delta_y + current_shape(j, 1);
                real_x = std::max(0, std::min(real_x, image.cols - 1)); // which cols
                real_y = std::max(0, std::min(real_y, image.rows - 1)); // which rows
                if ((tmp - (int)image(real_y, real_x)) < node->threshold_){
                    node = node->left_child_;// go left
                }
                else{
                    node = node->right_child_;// go right
                }
            }

            //int ind = j*params_.trees_num_per_forest_ + k;
            //int ind = feature_node_index[j] + k;
            //binary_features[ind].index = leaf_index_count[j] + node->leaf_identity;
            binary_features[ind].index = index + node->leaf_identity;//rd_forests_[j].GetBinaryFeatureIndex(k,image, bbox, current_shape, rotation, scale);
            binary_features[ind].value = 1.0;
            ind++;
            //std::cout << binary_features[ind].index << " ";
        }

        index += rd_forests_[j].all_leaf_nodes_;
    }
    //std::cout << "\n";
    //std::cout << index << ":" << params_.trees_num_per_forest_*params_.landmarks_num_per_face_ << std::endl;
    binary_features[params_.trees_num_per_forest_*params_.landmarks_num_per_face_].index = -1;
    binary_features[params_.trees_num_per_forest_*params_.landmarks_num_per_face_].value = -1.0;
    return binary_features;
}

cv::Mat_<double> Regressor::Predict(cv::Mat_<uchar>& image,
	cv::Mat_<double>& current_shape, BoundingBox& bbox, cv::Mat_<double>& rotation, double scale){

	cv::Mat_<double> predict_result(current_shape.rows, current_shape.cols, 0.0);

	// feature_node* binary_features = GetGlobalBinaryFeaturesThread(image, current_shape, bbox, rotation, scale);
    feature_node* binary_features = GetGlobalBinaryFeatures(image, current_shape, bbox, rotation, scale);
	// feature_node* tmp_binary_features = GetGlobalBinaryFeaturesMP(image, current_shape, bbox, rotation, scale);
    for (int i = 0; i < current_shape.rows; i++){
		predict_result(i, 0) = predict(linear_model_x_[i], binary_features);
        predict_result(i, 1) = predict(linear_model_y_[i], binary_features);
	}

	cv::Mat_<double> rot;
	cv::transpose(rotation, rot);
    delete[] binary_features;
    //delete[] tmp_binary_features;
	return scale*predict_result*rot;
}

void CascadeRegressor::LoadCascadeRegressor(std::string ModelName){
	std::ifstream fin;
    fin.open((ModelName + "_params.txt").c_str(), std::fstream::in);
	params_ = Parameters();
	fin >> params_.local_features_num_
		>> params_.landmarks_num_per_face_
		>> params_.regressor_stages_
		>> params_.tree_depth_
		>> params_.trees_num_per_forest_
		>> params_.initial_guess_
		>> params_.overlap_;

	std::vector<double> local_radius_by_stage;
	local_radius_by_stage.resize(params_.regressor_stages_);
	for (int i = 0; i < params_.regressor_stages_; i++){
		fin >> local_radius_by_stage[i];
	}
	params_.local_radius_by_stage_ = local_radius_by_stage;

	cv::Mat_<double> mean_shape(params_.landmarks_num_per_face_, 2, 0.0);
	for (int i = 0; i < params_.landmarks_num_per_face_; i++){
		fin >> mean_shape(i, 0) >> mean_shape(i, 1);
	}
	params_.mean_shape_ = mean_shape;
	regressors_.resize(params_.regressor_stages_);
	for (int i = 0; i < params_.regressor_stages_; i++){
        regressors_[i].params_ = params_;
		regressors_[i].LoadRegressor(ModelName, i);
        regressors_[i].ConstructLeafCount();
	}
}


void CascadeRegressor::SaveCascadeRegressor(std::string ModelName){
	std::ofstream fout;
    fout.open((ModelName + "_params.txt").c_str(), std::fstream::out);
	fout << params_.local_features_num_ << " "
		<< params_.landmarks_num_per_face_ << " "
		<< params_.regressor_stages_ << " "
		<< params_.tree_depth_ << " "
		<< params_.trees_num_per_forest_ << " "
		<< params_.initial_guess_ << " "
		<< params_.overlap_ << std::endl;
	for (int i = 0; i < params_.regressor_stages_; i++){
		fout << params_.local_radius_by_stage_[i] << std::endl;
	}
	for (int i = 0; i < params_.landmarks_num_per_face_; i++){
		fout << params_.mean_shape_(i, 0) << " " << params_.mean_shape_(i, 1) << std::endl;
	}

	fout.close();

    for (int i = 0; i < params_.regressor_stages_; i++){
		//regressors_[i].SaveRegressor(fout);
        regressors_[i].SaveRegressor(ModelName, i);
		//regressors_[i].params_ = params_;
	}

}


void Regressor::LoadRegressor(std::string ModelName, int stage){
	char buffer[500];
    sprintf(buffer, "%s_%d_regressor.txt", ModelName.c_str(), stage);
	std::ifstream fin;
	fin.open(buffer, std::fstream::in);
	int rd_size, linear_size;
	fin >> stage_ >> rd_size >> linear_size;
	rd_forests_.resize(rd_size);
	for (int i = 0; i < rd_size; i++){
		rd_forests_[i].LoadRandomForest(fin);
	}
	linear_model_x_.clear();
	linear_model_y_.clear();
	for (int i = 0; i < linear_size; i++){
        sprintf(buffer, "%s_%d/%d_linear_x.txt", ModelName.c_str(), stage_, i);
		linear_model_x_.push_back(load_model(buffer));

        sprintf(buffer, "%s_%d/%d_linear_y.txt", ModelName.c_str(), stage_, i);
		linear_model_y_.push_back(load_model(buffer));
	}
}

void Regressor::ConstructLeafCount(){
    int index = 1;
    int ind = params_.trees_num_per_forest_;
    for (int i = 0; i < params_.landmarks_num_per_face_; ++i){
        leaf_index_count[i] = index;
        index += rd_forests_[i].all_leaf_nodes_;
        feature_node_index[i] = ind*i;
    }
}

void Regressor::SaveRegressor(std::string ModelName, int stage){
	char buffer[500];
	//strcpy(buffer, ModelName.c_str());
	assert(stage == stage_);
    sprintf(buffer, "%s_%d_regressor.txt", ModelName.c_str(), stage);

	std::ofstream fout;
	fout.open(buffer, std::fstream::out);
	fout << stage_ << " "
		<< rd_forests_.size() << " "
        << linear_model_x_.size() << std::endl;

	for (int i = 0; i < rd_forests_.size(); i++){
		rd_forests_[i].SaveRandomForest(fout);
	}

    for (
         int i = 0; i < linear_model_x_.size(); i++){
        sprintf(buffer, "%s_%d", ModelName.c_str(), stage_);
#ifdef _WIN32 // can be used under 32 and 64 bits
        _mkdir(buffer);
#elif __linux__
        struct stat st = {0};
        if (stat(buffer, &st) == -1) {
            mkdir(buffer, 0777);
        }
#endif
		//_mkdir(buffer);
        sprintf(buffer, "%s_%d/%d_linear_x.txt", ModelName.c_str(), stage_, i);
		save_model(buffer, linear_model_x_[i]);

        sprintf(buffer, "%s_%d/%d_linear_y.txt", ModelName.c_str(), stage_, i);
		save_model(buffer, linear_model_y_[i]);
	}
}
