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

Mat getSelected(Mat desc, vector<int> indices){
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

void printClusterNumbers(map<int, int> maps, String folder){
	printf("Printing Cluster numbers to %s.\n", folder.c_str());
	ofstream myfile;
	String name = "/clusternumbers.csv";
	String f = folder;
	f += name;
	myfile.open(f.c_str());

	myfile << "minPts, Num Clusters\n";

	for(map<int, int >::iterator it = maps.begin(); it != maps.end(); ++it){
		myfile << it->first << "," << it->second << ",";

		myfile << "\n";
	}

	myfile.close();
}

void printCoreDistances(map<int, float> cores, String folder){
	printf("Printing core distances to %s.\n", folder.c_str());
	ofstream myfile;
	String name = "/coredistances.csv";
	String f = folder;
	f += name;
	myfile.open(f.c_str());

	myfile << "Cluster, Core Distance\n";

	for(map<int, float >::iterator it = cores.begin(); it != cores.end(); ++it){
		myfile << it->first << "," << it->second << ",";

		myfile << "\n";
	}

	myfile.close();
}

void execute(Mat dataset){

}

map_t mapClusters(map<int, vector<KeyPoint>>& cmap, vector<int> labels, map<int, float>& coreDisMap, float* core, vector<KeyPoint> keypoints){

	map_t amap;
	for (size_t i = 0; i < labels.size(); i++) {
		int label;

		label = labels[i];
		amap[label].push_back(i);
		cmap[label].push_back(keypoints[i]);

		//printf("core at %d is %f.\n", i, core[i]);
		if (coreDisMap.find(label) == coreDisMap.end()) {
			coreDisMap[label] = core[i];
		} else {
			if (coreDisMap[label] < core[i]) {
				coreDisMap[label] = core[i];
			}
		}

	}
	printf("Found %lu clusters in mpkp.\n", amap.size());

	return amap;
}

String createOutpuDirs(CommandLineParser parser, String& mainFolder, String subfolder, int i){
	String sokp;
	if (parser.has("o")) {
		String fld = parser.get<String>("o");
		mainFolder = fld;

		mainFolder += subfolder;

		sokp = mainFolder;
		sokp += to_string(i);

		String command = "mkdir \'";
		command += sokp;
		command += "\'";
		printf(command.c_str());
		const int dir_err2 = system(command.c_str());
		if (-1 == dir_err2) {
			printf("Error creating directory!n");
			exit(1);
		}

	}

	return sokp;

}

void printMapImages(Mat image, map_t inMap, map<int, vector<KeyPoint>> kpmap, String folder, bool hasOut){

	for(map_t::iterator it = inMap.begin(); it != inMap.end(); ++it){
		printf("mpkp: Cluster %d has %lu elements\n", it->first, it->second.size());

		if(hasOut){
			String imageName = "frame_keypoints_";
			imageName += to_string(it->first);
			Mat m = drawKeyPoints(image, kpmap[it->first], Scalar(0, 0, 255), -1); //DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
			printImage(folder, 1, imageName, m);
		}
	}
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

	String keypointsFolder , selectedFolder;

	cv::CommandLineParser parser(argc, argv,
						"{help||}{m|1|}{i1||}"
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

	Mat selDset;
	vector<KeyPoint> selkp;
	bool load = true;
	cout << "<<<<<<< Starting the " << endl;
	for(int i = minPts; i <= maxPts; i++){
		map<int, float> coreDisMap, coreDisMap0, coreDisMapCl, coreDisMapSel;
		cout << "**********************************************************" << endl;
		printf("Running hdbscan with %d minPts.\n", i);
		//Mat x = getColourDataset(queryImage, queryKp);

		/******************************************************************************************************************/
		hdbscan<float> scan2(_EUCLIDEAN, i);
		scan2.run(dataset.ptr<float>(), dataset.rows, dataset.cols, true);
		labelskp = scan2.getClusterLabels();
		set<int> lset2(labelskp.begin(), labelskp.end());

		set_t stcl, stkp, stcm;
		if(mode == 2){
			cout << "Loading sets." << endl;
			stkp.insert(labelskp.begin()+ogsize, labelskp.end());
		}

		qsetkp.push_back(stkp);
		float* core = scan2.getCoreDistances();
		cout << "Loading cluster maps." << endl;
		map_t mpkp;
		map<int, vector<KeyPoint>> kpmapkp, kpmapkp0;

		mpkp = mapClusters(kpmapkp, labelskp, coreDisMap, core, datasetKp);
		clusterkp.push_back(mpkp);
		cmaps[i] = mpkp.size();

		String sokp = createOutpuDirs(parser, keypointsFolder, "/keypoints/", i);

		cout << endl;
		printMapImages(queryImage, mpkp, kpmapkp, sokp, parser.has("o"));
		printCoreDistances(coreDisMap, sokp);

		/******************************************************************************************************************/

		printf("\n>>>>>>>>>Re clustering for Cluster 0<<<<<<<<<<\n");
		vector<KeyPoint> newKp;
		newKp.insert(newKp.end(), kpmapkp[0].begin(), kpmapkp[0].end());
		Mat dset = getSelected(queryDesc, mpkp[0]).clone();
		if(dset.rows > 3){
			hdbscan<float> scans(_EUCLIDEAN, 3);
			scans.run(dset.ptr<float>(), dset.rows, dset.cols, true);
			labelskps = scans.getClusterLabels();
			set<int> lsetkps(labelskps.begin(), labelskps.end());
			float* core0 = scan2.getCoreDistances();

			map_t mpkp0 = mapClusters(kpmapkp0, labelskps, coreDisMap0, core0, newKp);
			sokp = createOutpuDirs(parser, keypointsFolder, "/keypoints/cluster0/", i);
			printCoreDistances(coreDisMap0, sokp);
			printMapImages(queryImage, mpkp0, kpmapkp0, sokp, parser.has("o"));
		}

		cout << "**********************************************************" << endl << endl;
		/******************************************************************************************************************/

		Mat colour = getColourDataset(queryImage, queryKp).clone();
		map<int, vector<KeyPoint>> clkpmap;
		hdbscan<float> scanc(_EUCLIDEAN, 6);
		scanc.run(colour.ptr<float>(), colour.rows, colour.cols, true);
		vector<int> labelsc = scanc.getClusterLabels();
		set<int> lsetkps(labelsc.begin(), labelsc.end());
		float* corecl = scanc.getCoreDistances();
		map_t clmap = mapClusters(clkpmap, labelsc, coreDisMapCl, corecl, datasetKp);
		String socl = createOutpuDirs(parser, keypointsFolder, "/colour/", i);
		printMapImages(queryImage, clmap, clkpmap, socl, parser.has("o"));


		for(map_t::iterator it = clmap.begin(); it != clmap.end() && load; ++it){
			if (parser.has("o")) {
				String imageName = "frame_keypoints_";
				imageName += to_string(it->first);
				String ofile = socl;
				socl += "/";
				Mat m = drawKeyPoints(queryImage, clkpmap[it->first], Scalar(0, 0, 255), -1);
				display("choose", m);
				//printImage(socl, 1, imageName, m);
			}

			// Listen for a key pressed
			char c = ' ';
			while(true){
				if (c == 'a') {
					Mat xx = getSelected(queryDesc, it->second);
					selkp.insert(selkp.end(), clkpmap[it->first].begin(), clkpmap[it->first].end());
					if(selDset.empty()){
						cout << "Clone adding new data for cluster " << it->first << endl;
						selDset = xx.clone();
					} else{
						cout << "adding new data for cluster " << it->first << endl;
						selDset.push_back(xx);
					}
					break;
				} else if (c == 'q'){
					break;
				}
				c = (char) waitKey(20);
			}
		}

		load = false;
		map<int, vector<KeyPoint>> selkpmap;
		hdbscan<float> scans(_EUCLIDEAN, i);
		scans.run(selDset.ptr<float>(), selDset.rows, selDset.cols, true);
		vector<int> labelskpsel = scans.getClusterLabels();
		set<int> lsetsel(labelskpsel.begin(), labelskpsel.end());
		float* coresel = scans.getCoreDistances();

		map_t selmap = mapClusters(selkpmap, labelskpsel, coreDisMapSel, coresel, selkp);
		String sosel = createOutpuDirs(parser, keypointsFolder, "/selected/", i);
		printMapImages(queryImage, selmap, selkpmap, sosel, parser.has("o"));

	}

	printClusterNumbers(cmaps, keypointsFolder);


	return 0;
}





