/*
 * process_frame.cpp
 *
 *  Created on: 3 May 2017
 *      Author: ojmakh
 */

#include "process_frame.hpp"
#include <fstream>
#include <opencv/cv.hpp>
#include <dirent.h>

using namespace std;

void display(char const* screen, const InputArray& m) {
	if (!m.empty()) {
		namedWindow(screen, WINDOW_AUTOSIZE);
		imshow(screen, m);
	}
}

Scalar hsv_to_rgb(Scalar c) {
    Mat in(1, 1, CV_32FC3);
    Mat out(1, 1, CV_32FC3);

    float * p = in.ptr<float>(0);

    p[0] = c[0] * 360;
    p[1] = c[1];
    p[2] = c[2];

    cvtColor(in, out, COLOR_HSV2RGB);

    Scalar t;

    Vec3f p2 = out.at<Vec3f>(0, 0);

    t[0] = (int)(p2[0] * 255);
    t[1] = (int)(p2[1] * 255);
    t[2] = (int)(p2[2] * 255);

    return t;

}

Scalar color_mapping(int segment_id) {

    double base = (double)(segment_id) * 0.618033988749895 + 0.24443434;

    return hsv_to_rgb(Scalar(fmod(base, 1.2), 0.95, 0.80));

}

Mat getSegmentImage(Mat& gs, map<uint, vector<Point> >& points){
	//map<uint, vector<Point> > points;
    Mat output_image = Mat::zeros(gs.rows, gs.cols, CV_8UC3);
    uint* p;
    uchar* p2;

    /**
     * Get the segmentation points that will be assigned to individual
     */
	for (int i = 0; i < gs.rows; i++) {
		p = gs.ptr<uint>(i);
		p2 = output_image.ptr<uchar>(i);
		for (int j = 0; j < gs.cols; j++) {
			pair<uint, vector<Point> > pr(p[j], vector<Point>());
			std::map<uint, vector<Point> >::iterator it;
			it = points.find(p[j]);

			if (it == points.end()) {
				pair<map<uint, vector<Point> >::iterator, bool> ret =
						points.insert(pr);
				ret.first->second.push_back(Point(j, i));
			} else {
				it->second.push_back(Point(j, i));
			}

			Scalar color = color_mapping(p[j]);
			p2[j * 3] = color[0];
			p2[j * 3 + 1] = color[1];
			p2[j * 3 + 2] = color[2];
		}
	}

	return output_image;
}

void printImage(String folder, int idx, String name, Mat img) {
	//string folder = "/home/ojmakh/programming/phd/data/";
	stringstream sstm;

	sstm << folder.c_str() << "/" << idx << " " << name.c_str() << ".jpg";
	//cout << "printing " << sstm.str() << endl;
	imwrite(sstm.str(), img);
}

/**
 * Find the possible segment of interest
 * This function works by looking at the segments on both
 * sides of the roi rectangle. If the same segment is on
 * both sides, then it is added to the list of segments to
 * ignore.
 */
set<int32_t> getIgnoreSegments(Rect roi, Mat segments){
	set<int32_t> span;
	int32_t t = 3;
	cout << roi << endl;

	for(int i = roi.x; i < roi.x+roi.width; ++i){
		Point p1(roi.x, roi.y-t);
		Point p2(roi.x, roi.y+t);

		int32_t i1 = segments.at<int32_t>(p1);
		int32_t i2 = segments.at<int32_t>(p2);

		if(i1 == i2){
			span.insert(i1);
		}

		///printf("(p1.x, p1.y) = (%d, %d) ::::::: (p2.x, p2.y) = (%d, %d)\n", p1.x, p2.y, i1, i2);

		int end = roi.y + roi.height;
		p1.y = end + t;
		p2.y = end - t;

		i1 = segments.at<int32_t>(p1);
		i2 = segments.at<int32_t>(p2);

		if(i1 == i2){
			span.insert(i1);
		}
		///printf("(p1.x, p1.y) = (%d, %d) ::::::: (p2.x, p2.y) = (%d, %d)\n", p1.x, p2.y, i1, i2);
	}

	for(int j = roi.y; j < roi.y + roi.height; ++j){
		Point p1(roi.x - t, roi.y);
		Point p2(roi.x + t, roi.y);

		int32_t i1 = segments.at<int32_t>(p1);
		int32_t i2 = segments.at<int32_t>(p2);

		if(i1 == i2){
			span.insert(i1);
		}

		int end = roi.x + roi.width;
		p1.x = end + t;
		p2.x = end - t;

		i1 = segments.at<int32_t>(p1);
		i2 = segments.at<int32_t>(p2);

		if(i1 == i2){
			span.insert(i1);
		}

	}


	return span;
}

Mat drawKeyPoints(Mat in, vector<KeyPoint> points, Scalar colour, int type){
	Mat x = in.clone();
	if(type == -1){
		for(vector<KeyPoint>::iterator it = points.begin(); it != points.end(); ++it){
			circle(x, Point(it->pt.x, it->pt.y), 4, colour, CV_FILLED, 8, 0);
		}
	} else{
		drawKeypoints( in, points, x, Scalar::all(-1), type );
	}

	return x;
}

vector<KeyPoint> getAllMatchedKeypoints(framed& f){
	vector<KeyPoint> kp;

	for(map<int, vector<KeyPoint> >::iterator it = f.finalPointClusters.begin(); it != f.finalPointClusters.end(); ++it){
		kp.insert(kp.end(), it->second.begin(), it->second.end());
	}
	return kp;
}

void countPrint(framed& f){
	for (map<int, vector<int>>::iterator it = f.roiClusterPoints.begin(); it != f.roiClusterPoints.end(); ++it) {
		if(it->first != 0){
			int32_t n = f.clusterKeyPoints[it->first].size() / it->second.size();
			f.cest.push_back(n);
			f.total += n;
			printf("%d has %lu and total is %lu :: Approx Num of objects: %d\n\n", it->first, it->second.size(),
					f.clusterKeyPoints[it->first].size(), n);
			f.selectedFeatures += f.clusterKeyPoints[it->first].size();

			if (f.clusterKeyPoints[it->first].size() > f.lsize) {
				f.largest = it->first;
				f.lsize = f.clusterKeyPoints[it->first].size();
			}
		}
	}
}

void generateFinalPointClusters(framed& f){
	framed f1;
	map<int, int> pointsMap = splitROIPoints(f, f1);

	for(map<int, vector<int>>::iterator it = f.roiClusterPoints.begin(); it != f.roiClusterPoints.end(); ++it) {
		if (it->first != 0 && it->second.size() == 1) {
			int ptIdx = it->second[0];
			f.finalPointClusters[ptIdx] = f.clusterKeyPoints[it->first];
		}
	}

	for(map<int, vector<int>>::iterator it = f1.roiClusterPoints.begin(); it != f1.roiClusterPoints.end(); ++it) {
		if(it->first != 0 && it->second.size() == 1){
			int nptIdx = it->second[0];
			int optIdx = pointsMap[nptIdx];
			f.finalPointClusters[optIdx] = f1.clusterKeyPoints[it->first];
		}
	}

	if(f.finalPointClusters.size() == 0){

		for(map<int, vector<int>>::iterator it = f.roiClusterPoints.begin(); it != f.roiClusterPoints.end(); ++it) {
			if (it->first != 0) {
				int ptIdx = it->second[0];
				f.finalPointClusters[ptIdx] = f.clusterKeyPoints[it->first];
			}
		}

		for(map<int, vector<int>>::iterator it = f1.roiClusterPoints.begin(); it != f1.roiClusterPoints.end(); ++it) {
			if(it->first != 0){
				int nptIdx = it->second[0];
				int optIdx = pointsMap[nptIdx];
				f.finalPointClusters[optIdx] = f1.clusterKeyPoints[it->first];
			}
		}
	}
	cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
	countPrint(f);
	cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
	countPrint(f1);
	cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
}

void mapKeyPoints(framed& f){

	// Map cluster labels to the indices of the points
	for(uint i = 0; i < f.labels.size(); i++){
		int l = f.labels[i];
		f.clusterKeyPoints[l].push_back(f.keypoints[i]);
		f.clusterKeypointIdx[l].push_back(i);
	}

	// get a cluster labels belonging to the sample features and map them the KeyPoint indexes
	for(vector<int>::iterator it = f.roiFeatures.begin(); it != f.roiFeatures.end(); ++it){
		int key = f.labels[*it];
		f.roiClusterPoints[key].push_back(*it);
	}

	/*for(map<int, vector<int>>::iterator it = f.roiClusterPoints.begin(); it != f.roiClusterPoints.end(); ++it) {
		cout << "mapKeyPoints(framed& f) roiClusterPoints " << it->first << " it->second.size() " << it->second.size() << endl;
	}*/

}


/**
 *
 */
void getCount(framed& f){

	vector<KeyPoint> kp;
	for(map<int, vector<KeyPoint> >::iterator it = f.finalPointClusters.begin(); it != f.finalPointClusters.end(); ++it){
		f.cest.push_back(it->second.size());
		f.total += it->second.size();
		printf("Point %d has %lu objects\n", it->first, it->second.size());
		f.selectedFeatures += it->second.size();

		if(it->second.size() > f.lsize){
			f.largest = it->first;
			f.lsize = it->second.size();
		}

		Mat kimg = drawKeyPoints(f.frame, it->second, Scalar(0, 0, 255), -1);

		String ss = "img_keypoints-";
		string s = to_string(f.keyPointImages.size());
		ss += s.c_str();
		f.keyPointImages[ss] = kimg;
		kp.insert(kp.end(), it->second.begin(), it->second.end());
	}

	Mat mm = drawKeyPoints(f.frame, kp, Scalar(0, 0, 255), -1);

	String ss = "img_allkps";
	f.keyPointImages[ss] = mm;
}

void maintaintHistory(vocount& voc, framed& f){
	voc.frameHistory.push_back(f);
	if(voc.frameHistory.size() > 10){
		voc.frameHistory.erase(voc.frameHistory.begin());
	}
}

void mergeFlowAndImage(Mat& flow, Mat& gray, Mat& out) {
	CV_Assert(gray.channels() == 1);
	if (!flow.empty()) {

		if(out.empty()){
			out = Mat(flow.rows, flow.cols, flow.type());
		}

		Mat flow_split[2];
		Mat magnitude, angle;
		Mat hsv_split[3], hsv;
		split(flow, flow_split);
		cartToPolar(flow_split[0], flow_split[1], magnitude, angle, true);
		normalize(magnitude, magnitude, 0, 255, NORM_MINMAX);
		normalize(angle, angle, 0, 255, NORM_MINMAX);

		hsv_split[0] = angle; // already in degrees - no normalization needed
		Mat x;
		if(gray.empty()){
			x = Mat::ones(angle.size(), angle.type());
		} else{
			gray.convertTo(x, angle.type());
		}
		hsv_split[1] = x.clone();
		hsv_split[2] = magnitude;
		merge(hsv_split, 3, hsv);
		cvtColor(hsv, out, COLOR_HSV2BGR);
		normalize(out, out, 0, 255, NORM_MINMAX); // Normalise the matrix in the 0 - 255 range

		Mat n;
		out.convertTo(n, CV_8UC3); // Convert to 3 channel uchar matrix
		n.copyTo(out);

		// Normalise the flow within the range 0 ... 1
		normalize(flow, flow, 0, 1, NORM_MINMAX);
	}
}

uint getDataset(vocount& vcount, framed& f){
	Mat dataset = f.descriptors.clone();

	if (!vcount.frameHistory.empty()) {
		for (int j = 1; j < vcount.step; ++j) {
			int ix = vcount.frameHistory.size() - j;
			if (ix >= 0) {
				framed fx = vcount.frameHistory[ix];
				dataset.push_back(fx.descriptors);
			}
		}

	}
	f.ogsize = dataset.rows;

	f.dataset = dataset;
	return f.ogsize;
}

void printStats(String folder, map<int32_t, vector<int32_t> > stats){
	ofstream myfile;
	String f = folder;
	String name = "/stats.csv";
	f += name;
	myfile.open(f.c_str());

	myfile << "Frame #,Sample Size,Selected Sample,Feature Size, Selected Features, # Clusters,Cluster Sum, Cluster Avg., Box Est.,Actual\n";

	for(map<int32_t, vector<int32_t> >::iterator it = stats.begin(); it != stats.end(); ++it){
		vector<int32_t> vv = it->second;
		myfile << it->first << ",";

		for(uint i = 0; i < vv.size(); ++i){
			myfile << vv[i] << ",";
		}
		myfile << "\n";
	}

	myfile.close();

}

void printClusterEstimates(String folder, map<int32_t, vector<int32_t> > cEstimates){
	ofstream myfile;
	String f = folder;
	String name = "/ClusterEstimates.csv";
	f += name;
	myfile.open(f.c_str());

	myfile << "Frame #,Cluster Sum, Cluster Avg., Box Est.\n";

	for(map<int32_t, vector<int32_t> >::iterator it = cEstimates.begin(); it != cEstimates.end(); ++it){
		vector<int32_t> vv = it->second;
		size_t sz = vv.size();
		myfile << it->first << "," << vv[sz-1] << "," << vv[sz-2] << "," << vv[sz-3] << ",";

		for(uint i = 0; i < vv.size()-2; ++i){
			myfile << vv[i] << ",";
		}
		myfile << "\n";
	}

	myfile.close();

}

double calcDistanceL1(Point2f f1, Point2f f2){
	double diff = f1.x - f2.x;
	double sum = diff * diff;

	diff = f1.y - f2.y;
	sum += diff * diff;

	return sum;
}

/**
 * Find the roi features and at the same time find the central feature.
 */
void findROIFeature(framed& f){
	Rect2d r = f.roi;

	Point2f p;

	p.x = (r.x + r.width)/2.0f;
	p.y = (r.y + r.height)/2.0f;
	double distance;

	for(uint i = 0; i < f.keypoints.size(); ++i){
		if(f.hasRoi && f.roi.contains(f.keypoints[i].pt)){
			f.roiFeatures.push_back(i);

			// find the center feature index
			if(f.centerFeature == -1){
				f.centerFeature = i;
				distance = calcDistanceL1(p, f.keypoints[i].pt);
			} else {
				double d1 = calcDistanceL1(p, f.keypoints[i].pt);

				if(d1 < distance){
					distance = d1;
					f.centerFeature = i;
				}

			}

			// create the roi descriptor
			if(f.roiDesc.empty()){
				f.roiDesc = f.descriptors.row(i);
			} else{
				f.roiDesc.push_back(f.descriptors.row(i));
			}
		}
	}
}

bool processOptions(vocount& vcount, CommandLineParser& parser, VideoCapture& cap){

	if (parser.has("o")) {
		vcount.destFolder = parser.get<String>("o");
		vcount.print = true;
		printf("Will print to %s\n", vcount.destFolder.c_str());
	}

	if (parser.has("v") || parser.has("video")) {

		vcount.inputPath =
				parser.has("v") ?
						parser.get<String>("v") : parser.get<String>("video");
		cap.open(vcount.inputPath);
	} else {
		printf("You did not provide the video stream to open.");
		return false;
	}

	if (parser.has("w")) {

		String s = parser.get<String>("w");
		vcount.step = atoi(s.c_str());
	} else {
		vcount.step = 1;
	}

	if (parser.has("n")) {
		String s = parser.get<String>("n");
		vcount.rsize = atoi(s.c_str());
	}
	return true;
}

void printData(vocount& vcount, framed& f){
	if (vcount.print && !f.roiClusterPoints.empty()) {

		printImage(vcount.destFolder, vcount.frameCount, "frame", f.frame);

		Mat ff = drawKeyPoints(f.frame, f.keypoints, Scalar(0, 0, 255), -1);
		printImage(vcount.destFolder, vcount.frameCount, "frame_kp", ff);

		for(map<String, Mat>::iterator it = f.keyPointImages.begin(); it != f.keyPointImages.end(); ++it){
			printImage(vcount.destFolder, vcount.frameCount, it->first, it->second);
		}

		f.odata.push_back(f.roiFeatures.size());

		int selSampleSize = 0;

		for (map<int, vector<int>>::iterator it = f.roiClusterPoints.begin();
				it != f.roiClusterPoints.end(); ++it) {
			selSampleSize += it->second.size();
		}

		f.odata.push_back(selSampleSize);
		f.odata.push_back(f.ogsize);
		f.odata.push_back(f.selectedFeatures);
		f.odata.push_back(f.keyPointImages.size());
		f.odata.push_back(f.total);
		int32_t avg = f.total / f.keyPointImages.size();
		f.odata.push_back(avg);
		f.odata.push_back(f.boxStructures.size());

		map<int, int>::iterator it = vcount.truth.find(f.i);

		if(it == vcount.truth.end()){
			f.odata.push_back(0);
		} else{
			f.odata.push_back(it->second);
		}
		pair<int32_t, vector<int32_t> > pp(vcount.frameCount, f.odata);
		vcount.stats.insert(pp);
		f.cest.push_back(f.boxStructures.size());
		f.cest.push_back(avg);
		f.cest.push_back(f.total);
		pair<int32_t, vector<int32_t> > cpp(vcount.frameCount, f.cest);
		vcount.clusterEstimates.insert(cpp);
	}
}

int rectExist(vector<box_structure> structures, Rect& r){

	double maxIntersect = 0.0;
	//Rect maxRec(0, 0, 0, 0);
	int maxIndex = -1;

	for(uint i = 0; i < structures.size(); i++){
		Rect r2 = r & structures[i].box;
		double sect = ((double)r2.area()/r.area()) * 100;
		if(sect > maxIntersect){
			maxIndex = i;
			//maxRec = r2;
			maxIntersect = sect;
		}
	}


	if(maxIntersect > 50.0){
		return maxIndex;
	}

	return -1;
}

void boxStructure(framed& f){
	box_structure mbs;
	mbs.box = f.roi;

	for(map<int, vector<KeyPoint>>::iterator it = f.finalPointClusters.begin(); it != f.finalPointClusters.end(); ++it){
		vector<KeyPoint> c_points = it->second;
		KeyPoint roi_p = f.keypoints[it->first];
		mbs.points.push_back(roi_p);

		for(uint j = 0; j < c_points.size(); j++){
			//int pointIndex = c_points[j];

			KeyPoint point = c_points[j];
			if(point.pt != roi_p.pt){ // roi points have their own structure "mbs"
				Point2f pshift;
				pshift.x = point.pt.x - roi_p.pt.x;
				pshift.y = point.pt.y - roi_p.pt.y;

				// shift the roi to get roi for a possible new object
				Rect nr = f.roi;

				Point pp = pshift;
				nr = nr + pp;
				// check that the rect does not already exist
				int idx =rectExist(f.boxStructures, nr);
				if(idx == -1){
					box_structure bst;
					bst.box = nr;
					bst.points.push_back(point);
					f.boxStructures.push_back(bst);
				} else{
					f.boxStructures[idx].points.push_back(point);
				}
			}
		}
	}

	f.boxStructures.push_back(mbs);
	printf("boxStructure found %lu objects\n\n", f.boxStructures.size());
	String ss = "img_bounds";

	Mat img_bounds = f.keyPointImages["img_allkps"].clone();
	for (uint i = 0; i < f.boxStructures.size(); i++) {
		box_structure b = f.boxStructures[i];
		RNG rng(12345);
		Scalar value = Scalar(rng.uniform(0, 255), rng.uniform(0, 255),
				rng.uniform(0, 255));
		rectangle(img_bounds, b.box, value, 2, 8, 0);
	}
	f.keyPointImages[ss] = img_bounds;
}

map<int, int> addDescriptors(framed& f, framed& f1, int cluster, vector<int> roipts){

	vector<int> crois = f.clusterKeypointIdx[cluster];
	map<int, int> newMap;

	for(uint i = 0; i < crois.size(); i++){
		int rp = crois[i];

		f1.dataset.push_back(f.descriptors.row(rp));
		f1.keypoints.push_back(f.keypoints[rp]);
		newMap[f1.dataset.rows-1] = rp;

		for(uint j = 0; j < roipts.size(); j++){
			if(roipts[j] == rp){
				f1.roiFeatures.push_back(i);
				if(f1.roiDesc.empty()){
					f1.roiDesc = f.descriptors.row(rp);
				} else{
					f1.roiDesc.push_back(f.descriptors.row(rp));
				}
			}
		}
	}

	return newMap;
}

map<int, int> splitROIPoints(framed& f, framed& f1){

	map<int, int> fToF1;
	f1.i = f.i;
	f1.frame = f.frame;
	f1.hasRoi = f.hasRoi;

	bool run = false;
	for(map<int, vector<int>>::iterator it = f.roiClusterPoints.begin(); it != f.roiClusterPoints.end(); ++it){
		if (it->second.size() > 1 && it->first != 0) {

			run = true;
			map<int, int> fm = addDescriptors(f, f1, it->first, it->second);
			fToF1.insert(fm.begin(), fm.end());
		}
	}

	cout << "f.roiClusterPoints[0].size() " << f.roiClusterPoints[0].size() << endl;
	/**
	 * If there code above did not need reclustering, we check if the noise cluster has
	 * more roi features, in which case we add it to the reclustering data.
	 */
	if(!run && f.roiClusterPoints[0].size() > 0 && f1.dataset.rows == 0){

		run = true;
		map<int, int> fm = addDescriptors(f, f1, 0, f.roiClusterPoints[0]);

		fToF1.insert(fm.begin(), fm.end());
	}

	if(run){
		f1.dataset = f1.dataset.clone();
		f1.descriptors = f1.dataset.clone();
		f1.ogsize = f1.dataset.rows;
		f1.hasRoi = f.hasRoi;

		hdbscan<float> sc(_EUCLIDEAN, 4);
		sc.run(f1.dataset.ptr<float>(), f1.dataset.rows, f1.dataset.cols, true);
		f1.labels = sc.getClusterLabels();
		mapKeyPoints(f1);

		return fToF1;
	}

	return map<int, int>();
}

vector<Point2f> reduceDescriptorDimensions(Mat descriptors){
	vector<Point2f> points;

	for(int i = 0; i < descriptors.rows; i++){
		Point2f p(0, 0);
		for(int j = 0; j < descriptors.cols; i++){
			float f = descriptors.at<float>(i, j);

			if(j %2 == 0){
				p.x += f;
			} else{
				p.y += f;
			}
		}
		points.push_back(p);
	}

	return points;
}

map<int, int> getFrameTruth(String truthFolder){
	map<int, int> trueCount;
	DIR*     dir;
	    dirent*  pdir;

	    dir = opendir(truthFolder.c_str());     // open current directory

	    while (pdir = readdir(dir)) {
	        String s = pdir->d_name;
	        if(s != "." && s != ".."){
	            String full = truthFolder + "/";
	            full = full + pdir->d_name;
	            Mat image = imread(full, CV_LOAD_IMAGE_GRAYSCALE);
				threshold(image, image, 200, 255, THRESH_BINARY);
				Mat labels;
				connectedComponents(image, labels, 8, CV_16U);
				double min, max;
				cv::minMaxLoc(labels, &min, &max);

				char* pch = strtok (pdir->d_name," ");
				int fnum = atoi(pch);
				trueCount[fnum] = int(max);
	        }
	    }

	return trueCount;
}
