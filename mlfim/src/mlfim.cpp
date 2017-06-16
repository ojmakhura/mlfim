/*
 * mlfim.cpp
 *
 *  Created on: 16 Jun 2017
 *      Author: ojmakh
 */

//#include "hdbscan.hpp"
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <map>
#include <opencv/cv.hpp>
#include <opencv/cxcore.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <ctime>
#include <string>
#include "process_frame.hpp"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;
using namespace cv::ximgproc::segmentation;

int main(int argc, char** argv) {
	ocl::setUseOpenCL(true);
	framed f;
    Ptr<Feature2D> detector = SURF::create(500);

    String name = argv[1];
    f.frame = imread(name);
    detector->detectAndCompute(f.frame, Mat(), f.keypoints, f.descriptors);

    int m = atoi(argv[3]);
	hdbscan scan(f.descriptors, _EUCLIDEAN, m, m);
	scan.run();
	f.labels = scan.getClusterLabels();
	set<int> lset(f.labels.begin(), f.labels.end());
	mapKeyPoints(f);
	printf("lset has %lu\n", lset.size());
	printf("f.clusterKeypointIdx has %lu\n", f.clusterKeypointIdx.size());

	for(map<int, vector<KeyPoint>>::iterator it = f.clusterKeyPoints.begin(); it != f.clusterKeyPoints.end(); ++it){
		Mat m = drawKeyPoints(f.frame, it->second, Scalar(0, 0, 255), -1);
		String s = "frame_keypoints_";
		string s1 = to_string(it->first);

		s += s1.c_str();
		printImage(argv[2], 1, s, m);
	}


	return 0;
}





