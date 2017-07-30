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
#include <gsl/gsl_statistics.h>
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

void printStatistics(map<int, map<String, double>> stats, String folder){
	printf("Printing statistics to %s.\n", folder.c_str());
	ofstream coreFile, disFile;
	String name = "/core_distance_statistics.csv";
	String f = folder;
	f += name;
	coreFile.open(f.c_str());

	f = folder;
	name = "/distance_statistics.csv";
	f += name;
	disFile.open(f.c_str());

	coreFile << "minPts, Mean, Variance, Standard Deviation, Kurtosis, Skewness, Count\n";
	disFile << "minPts, Mean, Variance, Standard Deviation, Kurtosis, Skewness, Count\n";

	for(map<int, map<String, double>>::iterator it = stats.begin(); it != stats.end(); ++it){

		map<String, double> mp = it->second;
		coreFile << it->first << ",";
		coreFile << mp["mean_cr"] << ",";
		coreFile << mp["variance_cr"] << ",";
		coreFile << mp["sd_cr"] << ",";
		if(mp["kurtosis_cr"] == std::numeric_limits<double>::max()){
			coreFile << "NaN" << ",";
		} else {
			coreFile << mp["kurtosis_cr"] << ",";
		}

		if(mp["skew_cr"] == std::numeric_limits<double>::max()){
			coreFile << "NaN" << ",";
		} else {
			coreFile << mp["skew_cr"] << ",";
		}
		coreFile << mp["count"] << "\n";

		disFile << it->first << ",";
		disFile << mp["mean_dr"] << ",";
		disFile << mp["variance_dr"] << ",";
		disFile << mp["sd_dr"] << ",";
		if(mp["kurtosis_dr"] == std::numeric_limits<double>::max()){
			disFile << "NaN" << ",";
		} else {
			disFile << mp["kurtosis_dr"] << ",";
		}

		if(mp["skew_dr"] == std::numeric_limits<double>::max()){
			disFile << "NaN" << ",";
		} else {
			disFile << mp["skew_dr"] << ",";
		}
		disFile << mp["count"] << "\n";

	}

	coreFile.close();
	disFile.close();

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

float* getPointDataset(vector<KeyPoint> point){
	float *data = (float*)malloc(point.size() * 2 * sizeof(float));

	for(size_t i = 0; i < point.size(); i++){
		int idx = i *2;
		data[idx] = point[i].pt.x;
		data[idx+1] = point[i].pt.y;
	}

	return data;
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

map<String, double> printDistances(map<int, vector<float>> distances, String folder){
	printf("Printing core distances to %s.\n", folder.c_str());
	ofstream myfile;
	String name = "/coredistances.csv";
	String f = folder;
	f += name;
	myfile.open(f.c_str());
	map<String, double> stats;
	double cr[distances.size()];
	double dr[distances.size()];

	myfile << "Cluster, Min Core Distance, Max Core Distance, Core Distance Ratio, Min Distance, Max Distance, Distance Ratio\n";
	int c = 0;
	for(map<int, vector<float> >::iterator it = distances.begin(); it != distances.end(); ++it){
		// print min core, max core and core ratio
		cr[c] = (double)it->second[1]/it->second[0];
		myfile << it->first << "," << it->second[0] << "," << it->second[1] << "," << cr[c] << ",";

		// print min distance, max distance and distance ratio
		dr[c] = (double)it->second[3]/it->second[2];
		myfile << it->second[2] << "," << it->second[3] << "," << dr[c];

		myfile << "\n";
		c++;
	}

	myfile.close();

	// Calculating core distance statistics
	stats["mean_cr"] = gsl_stats_mean(cr, 1, c);
	stats["sd_cr"] = gsl_stats_sd(cr, 1, c);
	stats["variance_cr"] = gsl_stats_variance(cr, 1, c);
	if(c > 3){
		stats["kurtosis_cr"] = gsl_stats_kurtosis(cr, 1, c);
	} else{
		stats["kurtosis_cr"] = std::numeric_limits<double>::max();
	}

	if(c > 2){
		stats["skew_cr"] = gsl_stats_skew(cr, 1, c);
	} else{
		stats["skew_cr"] = std::numeric_limits<double>::max();
	}

	// Calculating distance statistics
	stats["mean_dr"] = gsl_stats_mean(dr, 1, c);
	stats["sd_dr"] = gsl_stats_sd(dr, 1, c);
	stats["variance_dr"] = gsl_stats_variance(dr, 1, c);
	if(c > 3){
		stats["kurtosis_dr"] = gsl_stats_kurtosis(dr, 1, c);
	} else{
		stats["kurtosis_dr"] = std::numeric_limits<double>::max();
	}

	if(c > 2){
		stats["skew_dr"] = gsl_stats_skew(dr, 1, c);
	} else{
		stats["skew_dr"] = std::numeric_limits<double>::max();
	}

	stats["count"] = c;
	//free(cr);
	//free(dr);

	return stats;
}

map<int, vector<float>> getDistances(map_t mp, hdbscan<float>& sc, float* core){

	map<int, vector<float>> pm;

	for(map_t::iterator it = mp.begin(); it != mp.end(); ++it){
		vector<int> idc = it->second;

		for(size_t i = 0; i < idc.size(); i++){

			// min and max core distances
			if(pm[it->first].size() == 0){
				pm[it->first].push_back(core[idc[i]]);
				pm[it->first].push_back(core[idc[i]]);
			} else{
				// min core distance
				if(pm[it->first][0] > core[idc[i]]){
					pm[it->first][0] = core[idc[i]];
				}

				//max core distance
				if(pm[it->first][1] < core[idc[i]]){
					pm[it->first][1] = core[idc[i]];
				}
			}

			// Calculating min and max distances
			for(size_t j = i+1; j < idc.size(); j++){
				float d = sc.getDistance(i, j); // (float)norm(desc.row(i), desc.row(j));
				//printf("distance is %f, mi\n", d);

				if(pm[it->first].size() == 2){
					pm[it->first].push_back(d);
					pm[it->first].push_back(d);
				} else{
					// min distance
					if(pm[it->first][2] > d){

						//printf("min distance sitching %f and %f\n", pm[it->first][2], d);
						pm[it->first][2] = d;
					}

					// max distance
					if(pm[it->first][3] < d){
						//printf("max distance sitching %f and %f\n", pm[it->first][3], d);
						pm[it->first][3] = d;
					}

				}

			}

		}

	}

	return pm;
}

map_t mapClusters(map<int, vector<KeyPoint>>& cmap, vector<int> labels, vector<KeyPoint> keypoints){

	map_t amap;
	for (size_t i = 0; i < labels.size(); i++) {
		int label;

		label = labels[i];
		amap[label].push_back(i);
		cmap[label].push_back(keypoints[i]);

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
    Ptr<Feature2D> detector = SURF::create();
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
	    vector<KeyPoint> qp(queryKp);
	    KeyPointsFilter::removeDuplicated(qp);
	    printf("quer kp = %d, qp = %d\n", queryKp.size(), qp.size());

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
			printf("Mode has to be either 1 or 2\n");
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
	map<int, map<String, double>> stats_kp, stats_kp0, stats_cl, stats_idx, stats_sel;
	bool load = true;
	cout << "<<<<<<< Starting the " << endl;
	for(int i = minPts; i <= maxPts; i++){
		map<int, vector<float>> disMap, disMap0, disMapCl, disMapSel, disMapId;
		cout << "**********************************************************" << endl;
		printf("Running hdbscan with %d minPts.\n", i);
		//Mat x = getColourDataset(queryImage, queryKp);

		/******************************************************************************************************************/
		cout << "----------------------------------- Original descriptors" << endl;
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

		mpkp = mapClusters(kpmapkp, labelskp, datasetKp);
		disMap = getDistances(mpkp, scan2, core);
		clusterkp.push_back(mpkp);
		cmaps[i] = mpkp.size();

		String sokp = createOutpuDirs(parser, keypointsFolder, "/keypoints/", i);

		cout << endl;
		printMapImages(queryImage, mpkp, kpmapkp, sokp, parser.has("o"));
		stats_kp[i] = printDistances(disMap, sokp);

		/******************************************************************************************************************/

		printf("\n>>>>>>>>>Re clustering for Cluster 0<<<<<<<<<<\n");
		vector<KeyPoint> newKp;
		newKp.insert(newKp.end(), kpmapkp[0].begin(), kpmapkp[0].end());
		Mat dset = getSelected(queryDesc, mpkp[0]).clone();
		if(dset.rows > 3){
			cout << "----------------------------------- Cluster 0 desctiprors" << endl;
			hdbscan<float> scans(_EUCLIDEAN, 3);
			scans.run(dset.ptr<float>(), dset.rows, dset.cols, true);
			labelskps = scans.getClusterLabels();
			set<int> lsetkps(labelskps.begin(), labelskps.end());
			float* core0 = scans.getCoreDistances();

			map_t mpkp0 = mapClusters(kpmapkp0, labelskps, newKp);
			disMap0 = getDistances(mpkp0, scans, core0);
			sokp = createOutpuDirs(parser, keypointsFolder, "/keypoints/cluster0/", i);
			stats_kp0[i] = printDistances(disMap0, sokp);
			printMapImages(queryImage, mpkp0, kpmapkp0, sokp, parser.has("o"));
		}

		cout << "**********************************************************" << endl << endl;
		/******************************************************************************************************************/

		Mat colour = getColourDataset(queryImage, queryKp).clone();
		map<int, vector<KeyPoint>> clkpmap;
		cout << "----------------------------------- Colour descriptors" << endl;
		hdbscan<float> scanc(_EUCLIDEAN, 6);
		scanc.run(colour.ptr<float>(), colour.rows, colour.cols, true);
		vector<int> labelsc = scanc.getClusterLabels();
		set<int> lsetkps(labelsc.begin(), labelsc.end());
		float* corecl = scanc.getCoreDistances();
		map_t clmap = mapClusters(clkpmap, labelsc, datasetKp);
		disMapCl = getDistances(clmap, scanc, corecl);
		String socl = createOutpuDirs(parser, keypointsFolder, "/colour/", i);
		stats_cl[i] = printDistances(disMapCl, socl);
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
		cout << "----------------------------------- Selected descriptors" << endl;
		hdbscan<float> scans(_EUCLIDEAN, i);
		scans.run(selDset.ptr<float>(), selDset.rows, selDset.cols, true);
		vector<int> labelskpsel = scans.getClusterLabels();
		set<int> lsetsel(labelskpsel.begin(), labelskpsel.end());
		float* coresel = scans.getCoreDistances();

		map_t selmap = mapClusters(selkpmap, labelskpsel, selkp);
		disMapSel = getDistances(selmap, scans, coresel);
		String sosel = createOutpuDirs(parser, keypointsFolder, "/selected/", i);
		stats_sel[i] = printDistances(disMapSel, sosel);
		printMapImages(queryImage, selmap, selkpmap, sosel, parser.has("o"));

		map<int, vector<KeyPoint>> selidmap;
		float* data = getPointDataset(selkp);
		cout << "----------------------------------- Point dataset" << endl;
		hdbscan<float> idscan(_EUCLIDEAN, 3);
		idscan.run(data, selkp.size(), 2, true);
		vector<int> labeldid = idscan.getClusterLabels();
		set<int> lsetid(labeldid.begin(), labeldid.end());
		float* coreid = idscan.getCoreDistances();
		map_t idmap = mapClusters(selidmap, labeldid, selkp);
		//disMapId = getDistances(idmap, desc, core)
		String soid = createOutpuDirs(parser, keypointsFolder, "/index/", i);
		printMapImages(queryImage, idmap, selidmap, soid, parser.has("o"));

	}

	printClusterNumbers(cmaps, keypointsFolder);

	if(parser.has("o")){
		String mfolder = parser.get<String>("o");
		String ofolder = mfolder;
		ofolder += "/keypoints/";
		printStatistics(stats_kp, ofolder);

		ofolder = mfolder;
		ofolder += "/keypoints/cluster0/";
		printStatistics(stats_kp0, ofolder);

		ofolder = mfolder;
		ofolder += "/selected/";
		printStatistics(stats_sel, ofolder);

		ofolder = mfolder;
		ofolder += "/colour/";
		printStatistics(stats_cl, ofolder);
	}


	return 0;
}





