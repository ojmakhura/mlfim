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

static void help(){
	printf( "This is a programming for estimating the number of objects in the video.\n"
	        "Usage: vocount\n"
	        "     [-help]			         	   	# Print this message\n"
	        "     [-m=<mode: [1, 2]>]     		   	# Either using one image which gives exploratory data analysis and just prints the clusters or matches the points from the training image to the query image points. \n"
			"     [-i1=<image one>]      			# The first image. Must always be provided.\n"
			"     [-i2=<image two>] 	      		# The second image. Must be provided when using mode 2\n"
			"     [-minPts=<min points>]      		# minimum points for hdbscan\n"
			"     [-maxPts=<min points>]      		# maximum minimum points for hdbscan\n"
	        "\n" );
}

typedef map<int, vector<int>> map_t;
typedef set<int> set_t;

Mat getColourDataset(Mat f, vector<KeyPoint> pts){
	cout << "getting stdata" << endl;
	Mat m(pts.size(), 3, CV_32FC1);
	float* data = m.ptr<float>(0);
	for(size_t i = 0; i < pts.size(); i++){
		Point2f pt = pts[i].pt;
		Vec3b p = f.at<Vec3b>(pt);
		int idx = i * 3;
		data[idx] = p.val[0];
		data[idx + 1] = p.val[1];
		data[idx + 2] = p.val[2];
	}
	cout << "getting stdata done" << endl;
	return m;
}

Mat getSelected(Mat desc, vector<KeyPoint> kps, vector<int> indices){
	Mat m;
	for(size_t i = 0; i < indices.size(); i++){
		if(m.empty()){
			m = desc.row(indices[i]);
		} else{
			m.push_back(desc.row(indices[i]));
		}
	}

	return m;
}

int main(int argc, char** argv) {
	ocl::setUseOpenCL(true);
	String queryName, trainName, outFolder;
	int mode, ogsize, minPts, maxPts;
	Mat queryImage, trainImage, queryDesc, trainDesc, dataset;
	vector<KeyPoint> queryKp, trainKp, datasetKp;
	vector<int> labelscl, labelskp, labelskps;
    Ptr<Feature2D> detector = SURF::create(1500);
	vector<set_t> qsetcl, qsetkp, qsetkps; // set of query cluster labels
	vector<map_t> clustercl, clusterkp, clusterkps;
	map<int, int> cmaps;

	cv::CommandLineParser parser(argc, argv,
						"{help||}{m|s|}{i1||}"
						"{i2||}{o||}{minPts|3|}"
						"{maxPts||}");

	if(!parser.has("i1")){
		printf("The first image must be provided. \n");
		help();
		return 0;
	} else{
		cout << "Loading query image." << endl;
		queryName = parser.get<String>("i1");
		queryImage = imread(queryName);
	    detector->detectAndCompute(queryImage, Mat(), queryKp, queryDesc);
	    dataset = queryDesc.clone();
	    datasetKp = queryKp;
	    ogsize = queryDesc.rows;

	}

	if(!parser.has("m")){
		printf("Please specify the mode. \n");
		help();
		return 1;
	} else{
		mode = parser.get<int>("m");

		if(mode == 2){
			if(!parser.has("i2")){
				printf("The image image must be provided when using mode 2. \n");
				help();
				return 0;
			} else{
				cout << "Loading training image." << endl;
				trainName = parser.get<String>("i2");
				trainImage = imread(trainName);
			    detector->detectAndCompute(trainImage, Mat(), trainKp, trainDesc);
			    dataset.push_back(trainDesc);
			    datasetKp.insert(datasetKp.end(), trainKp.begin(), trainKp.end());
			}
		} else if(mode != 1){
			printf("Mode has to be either 1 or 2");
			help();
			return -1;
		}
	}

	if (parser.has("help")) {
		help();
		return 0;
	}

	if(parser.has("minPts")){
		minPts = parser.get<int>("minPts");
	}

	if(parser.has("maxPts")){
		maxPts = parser.get<int>("maxPts");
	} else{
		maxPts = minPts;
	}

	if(parser.has("o")){

	}

	cout << "<<<<<<< Starting the " << endl;
	for(int i = minPts; i <= maxPts; i++){
		cout << "**********************************************************" << endl;
		printf("Running hdbscan with %d minPts.\n", i);
		Mat x = getColourDataset(queryImage, queryKp);
		hdbscan scan(_EUCLIDEAN, i, i);
		scan.run(x);
		labelscl = scan.getClusterLabels();
		set<int> lset(labelscl.begin(), labelscl.end());

		hdbscan scan2(_EUCLIDEAN, i, i);
		scan2.run(dataset);
		labelskp = scan2.getClusterLabels();
		set<int> lset2(labelskp.begin(), labelskp.end());

		set_t stcl, stkp, stcm;
		if(mode == 2){
			cout << "Loading sets." << endl;
			stcl.insert(labelscl.begin()+ogsize, labelscl.end());
			stkp.insert(labelskp.begin()+ogsize, labelskp.end());
		}
		qsetcl.push_back(stcl);
		qsetkp.push_back(stkp);

		cout << "Loading cluster maps." << endl;
		map_t mpcl, mpkp;
		map<int, vector<KeyPoint>> kpmapcl, kpmapkp, kpmapkps;
		for(size_t i = 0; i < labelscl.size(); i++){
			int label = labelscl[i];
			mpcl[label].push_back(i);
			kpmapcl[label].push_back(datasetKp[i]);

			label = labelskp[i];
			mpkp[label].push_back(i);
			kpmapkp[label].push_back(datasetKp[i]);

		}
		clustercl.push_back(mpcl);
		clusterkp.push_back(mpkp);
		printf("Found %lu clusters in mpcl.\n", mpcl.size()-1);
		printf("Found %lu clusters in mpkp.\n", mpkp.size()-1);

		String socl, sokp, sokps;
		if(parser.has("o")){
			String fld = parser.get<String>("o");
			String co = fld, co2 = fld, co3 = fld;

			co += "/colour/";
			co2 += "/keypoints/";
			co3 += "/selected/";

			co += to_string(i);
			co2 += to_string(i);
			co3 += to_string(i);

			String command = "mkdir \'";
			command += co;
			command += "\'";
			printf(command.c_str());
			const int dir_err = system(command.c_str());
			if (-1 == dir_err)
			{
			    printf("Error creating directory!n");
			    exit(1);
			}

			command = "mkdir \'";
			command += co2;
			command += "\'";
			printf(command.c_str());
			const int dir_err2 = system(command.c_str());
			if (-1 == dir_err2) {
				printf("Error creating directory!n");
				exit(1);
			}

			command = "mkdir \'";
			command += co3;
			command += "\'";
			printf(command.c_str());
			const int dir_err3 = system(command.c_str());
			if (-1 == dir_err3) {
				printf("Error creating directory!n");
				exit(1);
			}

			socl = co;
			sokp = co2;
			sokps = co3;
		}

		cout << endl;
		Mat dset;
		vector<KeyPoint> dsetKp;
		for(map_t::iterator it = mpcl.begin(); it != mpcl.end(); ++it){
			printf("mpcl: Cluster %d has %lu elements\n", it->first, it->second.size());

			if(parser.has("o")){
				String imageName = "frame_keypoints_";
				imageName += to_string(it->first);
				String ofile = socl;
				socl += "/";
				Mat m = drawKeyPoints(queryImage, kpmapcl[it->first], Scalar(0, 0, 255), -1);
				display("choose", m);
				printImage(socl, 1, imageName, m);
			}

			// Listen for a key pressed
			char c = ' ';

			while(true){
				if (c == 'a') {

					Mat xx = getSelected(queryDesc, queryKp, it->second);
					dsetKp.insert(dsetKp.end(), kpmapcl[it->first].begin(), kpmapcl[it->first].end());
					if(dset.empty()){
						cout << "Clone adding new data for cluster " << it->first << endl;
						dset = xx.clone();
					} else{
						cout << "adding new data for cluster " << it->first << endl;
						dset.push_back(xx);
					}
					break;

				} else if (c == 'q'){
					break;
				}
				c = (char) waitKey(20);
			}
		}

		cout << endl;
		for(map_t::iterator it = mpkp.begin(); it != mpkp.end(); ++it){
			printf("mpkp: Cluster %d has %lu elements\n", it->first, it->second.size());

			if(parser.has("o")){
				String imageName = "frame_keypoints_";
				imageName += to_string(it->first);
				String ofile = sokp;
				sokp += "/";
				Mat m = drawKeyPoints(queryImage, kpmapkp[it->first], Scalar(0, 0, 255), -1);
				printImage(sokp, 1, imageName, m);
			}
		}

		/**
		 * Run hdbscan on selected colours
		 */
		hdbscan scans(_EUCLIDEAN, 3, 3);
		scans.run(dset);
		labelskps = scans.getClusterLabels();
		set<int> lsetkps(labelskps.begin(), labelskps.end());

		map_t mpkps;
		for(size_t i = 0; i < labelskps.size(); i++){
			int label = labelskps[i];
			mpkps[label].push_back(i);
			kpmapkps[label].push_back(dsetKp[i]);
		}
		clusterkps.push_back(mpkps);
		printf("Found %lu clusters in mpcl.\n", mpcl.size()-1);

		cout << endl;
		Mat x2;
		for(map_t::iterator it = mpkps.begin(); it != mpkps.end(); ++it){
			printf("mpkps: Cluster %d has %lu elements\n", it->first, it->second.size());

			if(parser.has("o")){
				String imageName = "frame_keypoints_";
				imageName += to_string(it->first);
				String ofile = sokps;
				sokps += "/";
				Mat m = drawKeyPoints(queryImage, kpmapkps[it->first], Scalar(0, 0, 255), -1);

				if(x2.empty() && it->first != 0){
					x2 = drawKeyPoints(queryImage, kpmapkps[it->first], Scalar(0, 0, 255), -1);
				} else if(it->first != 0){
					x2 = drawKeyPoints(x2, kpmapkps[it->first], Scalar(0, 0, 255), -1);
				}

				printImage(sokps, 1, imageName, m);
			}
		}
		Mat g = drawKeyPoints(queryImage, dsetKp, Scalar(0, 0, 255), -1);
		printImage(sokps, 1, "all_selected", g);
		printImage(sokps, 1, "all_other", x2);

		cout << "**********************************************************" << endl << endl;
	}

	return 0;
}





