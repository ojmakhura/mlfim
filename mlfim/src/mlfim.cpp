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
	Mat frame;

    Ptr<Feature2D> detector = SURF::create(1500);


	return 0;
}





