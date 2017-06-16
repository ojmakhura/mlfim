/*
 * process_frame.hpp
 *
 *  Created on: 3 May 2017
 *      Author: ojmakh
 */

#ifndef PROCESS_FRAME_HPP_
#define PROCESS_FRAME_HPP_
#include "hdbscan.hpp"
#include <opencv/cv.hpp>
#include <opencv2/ximgproc/segmentation.hpp>
#include <vector>
#include <cstdlib>
#include <iostream>

using namespace cv;
using namespace std;
using namespace cv::ximgproc::segmentation;

typedef struct _box_structure{
	Rect box;
	vector<KeyPoint> points;
} box_structure;

typedef struct EDGE{
	float anglediff;
	float distance;
	int idxa, idxb;
} edge;

typedef struct GRAPH{
	vector<edge> edgeList;
} graph;

typedef struct FRAMED{
	int i, ogsize;
	Mat descriptors,  												/// Frame descriptors
		frame,														/// The frame
		gray,														/// gray scale image
		dataset;													/// hdbscan cluster
	map<String, Mat> keyPointImages;										/// images with cluster by cluster keypoints drawn
	vector<KeyPoint> keypoints; 									/// Frame keypoints
    map<int, vector<KeyPoint> > clusterKeyPoints;					/// maps labels to their keypoints
    map<int, vector<int> > clusterKeypointIdx; 						/// maps labels to the keypoint indices
    map<int, vector<int> > roiClusterPoints;						/// cluster labels for the region of interest mapped to the roi points in the cluster
    map<int, vector<KeyPoint> > finalPointClusters;					/// for clusters where there is more than one roi point in the cluster. Maps the roi point
    																/// index to all closest descriptor points indices
	vector<int32_t> odata;											/// Output data
    vector<int> labels;												/// hdbscan cluster labels
	uint largest = 0;
	float lsize = 0;
	float total = 0;
	int32_t selectedFeatures = 0;
	vector<int32_t> cest;
	Rect2d roi;														/// region of interest rectangle
	vector<int> roiFeatures;										/// indices of the features inside the roi
	int centerFeature = -1;											/// index of the roi central feature
	Mat roiDesc;													/// region of interest descriptors
	graph roiStructure;												/// structure of the
	vector<graph> objStructures;
	vector<box_structure> boxStructures;							/// Bounding boxes for the individual objects in the frame
	bool hasRoi = false;
} framed;

typedef struct VOCOUNT{
    int frameCount = 0;
	String destFolder, inputPath;
	int step, rsize;											/// How many frames to use in the dataset
	bool roiExtracted = false;
	bool interactive = false;
	bool print = false;
    map<int32_t, vector<int32_t> > stats;
    map<int32_t, vector<int32_t> > clusterEstimates;
	vector<framed> frameHistory;
    map<int, int> truth;
} vocount;


void display(char const* screen, const InputArray& m);

Scalar hsv_to_rgb(Scalar c);

Scalar color_mapping(int segment_id);

Mat getSegmentImage(Mat& gs, map<uint, vector<Point> >& points);

void printImage(String folder, int idx, String name, Mat img);

Mat drawKeyPoints(Mat in, vector<KeyPoint> points, Scalar colour, int type);

void maintaintHistory(vocount& voc, framed& f);
set<int32_t> getIgnoreSegments(Rect roi, Mat segments);
void mapKeyPoints(framed& f);
void getCount(framed& f);
void runSegmentation(vocount& vcount, framed& f, Ptr<GraphSegmentation> graphSegmenter, Ptr<DenseOpticalFlow> flowAlgorithm);
void mergeFlowAndImage(Mat& flow, Mat& gray, Mat& out);
uint getDataset(vocount& vcount, framed& f);

void printStats(String folder, map<int32_t, vector<int32_t> > stats);

void printClusterEstimates(String folder, map<int32_t, vector<int32_t> > cEstimates);
void matchByBruteForce(vocount& vcount, framed& f);
vector<KeyPoint> getAllMatchedKeypoints(framed& f);
void findROIFeature(framed& f);
bool processOptions(vocount& voc, CommandLineParser& parser, VideoCapture& cap);

/***
 *
 */
void printData(vocount& vcount, framed& f);

/**
 *
 */
void boxStructure(framed& f);

/**
 *
 */
void generateFinalPointClusters(framed& f);

/**
 *
 */
vector<Point2f> reduceDescriptorDimensions(Mat descriptors);

/**
 *
 */
map<int, int> splitROIPoints(framed& f, framed& f1);

/**
 * Get the true count of objects from the given folder. The
 */
map<int, int> getFrameTruth(String truthFolder);
#endif /* PROCESS_FRAME_HPP_ */
