/*
 * distance.cpp
 *
 *  Created on: 18 May 2016
 *      Author: junior
 */

#include "distance.hpp"
#include <omp.h>

using namespace std;

namespace clustering {
namespace distance {

DistanceCalculator::~DistanceCalculator(){

	if(distance != NULL){
		delete [] distance;
	}

	if (coreDistances != NULL) {
		delete [] coreDistances;
	}
}

DistanceCalculator::DistanceCalculator(calculator cal){
	this->cal = cal;
	distance = NULL;
	coreDistances = NULL;
	internalRows = internalCols = rows = cols = 0;

}

void DistanceCalculator::setCalculator(calculator cal){
	this->cal = cal;
}

void DistanceCalculator::computeDistance(vector<float>& dataset, int rows, int cols, bool rowwise, int minPoints) {
    printf("DistanceCalculator::computeDistance(vector<double>& dataset\n");
    setDimenstions(rows, cols);

	if (cal == _EUCLIDEAN) {

        do_euclidean(&dataset, minPoints, rowwise, true);
	} else if (cal == COSINE) {
		//do_cosine(dataset, minPoints);
	} else if (cal == _MANHATTAN) {
		//do_manhattan(dataset, minPoints);
	} else if (cal == PEARSON) {
		//do_pearson(dataset, minPoints);
	} else if (cal == SUPREMUM) {
		//do_supremum(dataset, minPoints);
	}

}

void DistanceCalculator::computeDistance(float* dataset, int rows, int cols, bool rowwise, int minPoints){
    printf("DistanceCalculator::computeDistance(double* dataset, int rows, int cols, bool rowwise, int minPoints) %i\n", rowwise);
    setDimenstions(rows, cols);

    if (cal == _EUCLIDEAN) {

        do_euclidean(&dataset, minPoints, rowwise, false);
    } else if (cal == COSINE) {
        //do_cosine(dataset, minPoints);
    } else if (cal == _MANHATTAN) {
        //do_manhattan(dataset, minPoints);
    } else if (cal == PEARSON) {
        //do_pearson(dataset, minPoints);
    } else if (cal == SUPREMUM) {
        //do_supremum(dataset, minPoints);
    }
}

void DistanceCalculator::setDimenstions(int rows, int cols){

    this->rows = rows;
    this->cols = cols;
    int sub = (rows * rows -rows)/2;
    this->distance = new float[sub];
    this->coreDistances = new float[rows];
}

double DistanceCalculator::getElement(void* dataset, int index, bool isVector){

    double d;
    if(isVector){
        vector<double>* v = (vector<double>*)dataset;
        d = (*v)[index];
    } else {
        double** v = (double**)dataset;
        d = (*v)[index];
    }

    return d;
}

void DistanceCalculator::do_euclidean(void* dataset, int minPoints, bool rowwise, bool isVector){
    //printf("DistanceCalculator::do_euclidean(vector<double>& dataset\n");
#ifdef USE_OPENMP
#pragma omp parallel
#endif
    {

    double sortedDistance[rows];

#ifdef USE_OPENMP
#pragma omp for private(sortedDistance)
#endif
    for (uint i = 0; i < rows; i++) {
        for (uint j = 0; j < rows; j++) {
            double sum, diff = 0.0;
            uint offset2, offset1;
			sum = 0;
			//int r2, c2;
            if(rowwise){
                for (uint k = 0; ((k < cols) && (i != j)); k++) {
                    double num1 = getElement(dataset, i * cols + k, isVector);
                    double num2 = getElement(dataset, j * cols + k, isVector);
                    //printf("is rowwise nums are %f and %f\n", num1, num2);
                    diff = num1 - num2;

                    sum += (diff * diff);
                }
            } else {

                double num1 = getElement(dataset, i, isVector);
                double num2 = getElement(dataset, j, isVector);
                //printf("is  not rowwise nums are %f and %f\n", num1, num2);
                diff = num1 - num2;

                sum += (diff * diff);
            }

			sum = sqrt(sum);

			//printf("sum = %f\n", sum);
			int c;
			if (j > i) {
				// Calculate the linearised upper triangular matrix offset
				offset1 = i * rows + j;
				c = offset1 - triangular(i + 1);
				//printf("c calculated %d from (%d, %d)\n", c, i, j);
				offset2 = j * rows + i;
				//printf("offset calculated at (%d, %d)\n", offset1, offset2);

				*(distance + c) = sum;
				//printf("*(distance + c) 1 alloc %f\n", *(distance + c));
				//*(distance + offset2) = sum;
				//printf("distance sum set\n");
			} else if (i == j) {
				c = -1;
			} else {
				offset1 = j * rows + i;
				c = offset1 - triangular(j + 1);
			}

            sortedDistance[j] = sum;
			//(sortedDistance)[offset2] = sum;
			//printf("%.2f \n", *(distance + offset1));

			//printf("Index : %d ------ > (%d, %d) = %f\n", c, i, j, sum);
		}
//#ifdef USE_OPENMP
//#pragma omp barrier
//#endif
		//#pragma omp barrier
		//roffset = i * rows;
		std::sort(sortedDistance, sortedDistance + rows);
        coreDistances[i] = sortedDistance[minPoints];
	}
    }
}

void DistanceCalculator::cvComputeDistance(vector<Mat>& dataset, int minPoints, bool indexed){
	CV_Assert(dataset.size() > 0);

	/**
	 * Need all Matrices to be continuous and have only one channel
	 */
#pragma omp parallel for
	for(uint x = 0; x < dataset.size(); ++x){
		CV_Assert(dataset[x].isContinuous());
		//CV_Assert(dataset[x].channels() == 1);
	}

	this->rows = dataset[0].rows * dataset[0].cols;
	int sub = (rows * rows -rows)/2;
	// dataset only contains the data matrices
	//
	this->cols = dataset.size() + 2;

	this->distance = new float[sub];
	//this->sortedDistance = (float *) malloc(rows * rows * sizeof(float));
	this->coreDistances = new float[rows];

	if (cal == _EUCLIDEAN) {
		//cvEuclidean(&dataset, minPoints);
		do_euclidean(&dataset, minPoints, indexed);
	} else if (cal == COSINE) {
		//do_cosine(dataset, minPoints);
	} else if (cal == _MANHATTAN) {
		//do_manhattan(dataset, minPoints);
	} else if (cal == PEARSON) {
		//do_pearson(dataset, minPoints);
	} else if (cal == SUPREMUM) {
		//do_supremum(dataset, minPoints);
	}

}

void DistanceCalculator::cvComputeDistance(Mat& dataset, int minPoints){


	this->rows = dataset.rows;
	int sub = (rows * rows -rows)/2;
	// dataset only contains the data matrices
	//
	this->cols =  dataset.cols;

	this->distance = new float[sub];
	//this->sortedDistance = (float *) malloc(rows * rows * sizeof(float));
	this->coreDistances = new float[rows];

	if (cal == _EUCLIDEAN) {
		//cvEuclidean(&dataset, minPoints);
		do_euclidean(dataset, minPoints);
	} else if (cal == COSINE) {
		//do_cosine(dataset, minPoints);
	} else if (cal == _MANHATTAN) {
		//do_manhattan(dataset, minPoints);
	} else if (cal == PEARSON) {
		//do_pearson(dataset, minPoints);
	} else if (cal == SUPREMUM) {
		//do_supremum(dataset, minPoints);
	}

}

void DistanceCalculator::computeDistance(float* dataset, int rows, int cols, int minPoints) {

	this->rows = rows;
	this->cols = cols;
	int sub = (rows * rows -rows)/2;
	this->distance = new float[sub];
	this->coreDistances = new float[rows];

	if (cal == _EUCLIDEAN) {
		do_euclidean(dataset, minPoints);
	} else if (cal == COSINE) {
		do_cosine(dataset, minPoints);
	} else if (cal == _MANHATTAN) {
		do_manhattan(dataset, minPoints);
	} else if (cal == PEARSON) {
		do_pearson(dataset, minPoints);
	} else if (cal == SUPREMUM) {
		do_supremum(dataset, minPoints);
	}

}

float* DistanceCalculator::getDistance(){ return distance; }
//float* DistanceCalculator::getSortedDistance(){ return sortedDistance; }
float* DistanceCalculator::getCoreDistances(){ return coreDistances; }

/**************************************************************************
 * Private methods that handle opencv data
 **************************************************************************/

void DistanceCalculator::do_euclidean(Mat& dataset, int numNeighbors){

#ifdef USE_OPENMP
#pragma omp parallel
#endif
	{

	float sortedDistance[rows];
#ifdef USE_OPENMP
#pragma omp for
#endif
	for (uint i = 0; i < rows; i++) {
		for (uint j = 0; j < rows; j++) {
			float sum;
			uint offset1;
			sum = 0;

			sum = cv::norm(dataset.row(i), dataset.row(j), NORM_L2);

			int c;
			if(j > i){
				// Calculate the linearised upper triangular matrix offset
				offset1 = i * rows + j;
				c = offset1 - triangular(i + 1);

				*(distance + c) = sum;
			} else if(i == j){
				c = -1;
			} else{
				offset1 = j * rows + i;
				c = offset1 - triangular(j + 1);
			}

			(sortedDistance)[j] = sum;

		}
//#pragma omp barrier
		//roffset = i * rows;
		std::sort(sortedDistance, sortedDistance + rows);
		coreDistances[i] = (sortedDistance)[numNeighbors];
	}
	}

}

/**************************************************************************
 * Private methods
 **************************************************************************/

void addDistance(int i, int j, float distance){

}

float convolution(Point p, Mat m, int kernelSize){

	float sum = 0.0;
	int red = kernelSize/2;

	for(int i = 0; i < kernelSize; i++){
		for(int j = 0; j < kernelSize; j++){
			int r = p.y + i - red;

			if(r < 0){
				r = p.y;
			}

			int c = p.x + j - red;

			if(c < 0){
				c = p.x;
			}

			if(m.depth() == CV_8U){
				sum += m.at<uchar>(r, c);
			} else if(m.depth() == CV_32F){
				sum += m.at<float>(r, c);
			}
		}
	}

	return sum/(kernelSize * kernelSize);
}

void DistanceCalculator::do_euclidean(vector<Mat>* dataset, int numNeighbors, bool includeIndex) {
	//distance = 0;
	//printf("calculating distance (%d, %d, %f)\n", rows, cols, *dataset);

#ifdef USE_OPENMP
#pragma omp parallel
#endif
	{

	float sortedDistance[rows];
	int r1 = 0, c1 = 0;
#ifdef USE_OPENMP
#pragma omp for
#endif
	for (uint i = 0; i < rows; i++) {

		if(includeIndex){
			r1 = i/(*dataset)[0].cols;
			c1 = i%(*dataset)[0].cols;
		}

		for (uint j = 0; j < rows; j++) {
			float sum, diff = 0.0;
			uint roffset, offset2, offset1;
			sum = 0;
			int r2, c2;
			if(includeIndex){
				r2 = j/(*dataset)[0].cols;
				c2 = j%(*dataset)[0].cols;
				//printf("(%d, %d)\t", r1, c1);
				//printf("(%d, %d)\t", r2, c2);
				float num1 = r1 - r2;// = *(dataset + i * cols + k);
				float num2 = c1 - c2;
				sum += (num1 * num1);
				sum += (num2 * num2);
				//printf("calculating for mul: (%d, %d)\n", i, j);
			}

			int colOffset = 2;

			if(includeIndex){
				colOffset = 2;
			}

			for (uint k = 0; ((k < cols - colOffset) && (i != j)); k++) {

				//for(uint x = 0; x < dataset->size(); ++x){

					Mat m = (*dataset)[k];

					if(m.depth() == CV_8U){
						//printf("I am 8U\n");

						//printf("CV_8U at %d %d and %d %d\n", r1, c1, r2, c2);

						uchar* data = m.data;
						float num1 = *(data + i);
						float num2 = *(data + j);
						diff = num1 - num2;
					} else if(m.depth() == CV_32F){
						//printf("I am 32S\n");
						float* data = (float*)m.ptr<float>(0);
						float num1 = *(data + i);
						float num2 = *(data + j);
						diff = num1 - num2;

					}

					//printf("inner peace %d: (%f - %f) \n", k, num1, num2);

					sum += (diff * diff);
				//}
			}

			sum = sqrt(sum);

			//printf("sum = %f\n", sum);
			int c;
			if(j > i){
				// Calculate the linearised upper triangular matrix offset
				offset1 = i * rows + j;
				c = offset1 - triangular(i + 1);
				//printf("c calculated %d from (%d, %d)\n", c, i, j);
				offset2 = j * rows + i;
				//printf("offset calculated at (%d, %d)\n", offset1, offset2);

				*(distance + c) = sum;
				//printf("*(distance + c) 1 alloc %f\n", *(distance + c));
				//*(distance + offset2) = sum;
				//printf("distance sum set\n");
			} else if(i == j){
				c = -1;
			} else{
				offset1 = j * rows + i;
				c = offset1 - triangular(j + 1);
			}

			(sortedDistance)[j] = sum;
			//(sortedDistance)[offset2] = sum;
			//printf("%.2f \n", *(distance + offset1));

			//printf("Index : %d ------ > (%d, %d) = %f\n", c, i, j, sum);
		}
//#pragma omp barrier
		//roffset = i * rows;
		std::sort(sortedDistance, sortedDistance + rows);
		coreDistances[i] = (sortedDistance)[numNeighbors];

		/*for(int s = 0; s < 20; ++s){
			printf("%f, ", sortedDistance[s]);
		}
		printf("\n");*/

		//printf("coreDistances[%f] = %d\n", i, coreDistances[i]);
		//delete sortedDistance;
	}
	}
	/*printf("\n*************** Printing Distance Matrix ******************\n");

	int n = (rows*rows - rows)/2;
	printf("n is %d\n", n);
	for (int i = 0; i < n; i++) {
		printf("%d	|	%.1f\t\n", i,  *(distance + i));
	}*/
	/*
	printf(
			"\n*************** Printing Sorted Distance Matrix ******************\n");

	for (int i = 0; i < rows; i++) {
		printf("%.1f\t", *(coreDistances + i));

		printf("\n");
	}*/

	//printf("done");

}

void DistanceCalculator::do_euclidean(float* dataset, int numNeighbors) {

	uint roffset, offset2, offset1;
	float sum;
	float sortedDistance[rows];

	for (uint i = 0; i < rows; i++) {
		for (uint j = 0/*i + 1*/; j < rows; j++) {
			sum = 0;

			for (uint k = 0; ((k < cols) && (i != j)); k++) {

				float num1 = *(dataset + i * cols + k);
				float num2 = *(dataset + j * cols + k);
				float diff = num1 - num2;

				sum += (diff * diff);
			}

			sum = sqrt(sum);

			int c;
			if(j > i){
				// Calculate the linearised upper triangular matrix offset
				offset1 = i * rows + j;
				c = offset1 - triangular(i + 1);
				offset2 = j * rows + i;

				*(distance + c) = sum;
			} else if(i == j){
				c = -1;
			}

			else{
				offset1 = j * rows + i;
				c = offset1 - triangular(j + 1);
			}

			(sortedDistance)[j] = sum;
		}

		std::sort(sortedDistance, sortedDistance + rows);
		coreDistances[i] = (sortedDistance)[numNeighbors];

	}
}

void DistanceCalculator::do_cosine(float* dataset, int minPoints) {
	/*Mat dotProduct;
	Mat magnitudeOne;
	Mat magnitudeTwo;

	dotProduct = attributesOne.dot(attributesTwo);*/

	/*for (unsigned int i = 0; i < attributesOne->size() && i < attributesTwo->size(); i++) {
		dotProduct += ((*attributesOne)[i] * (*attributesTwo)[i]);
		magnitudeOne += ((*attributesOne)[i] * (*attributesOne)[i]);
		magnitudeTwo += ((*attributesTwo)[i] * (*attributesTwo)[i]);
	}*/

	//distance = 1 - (dotProduct / (magnitudeOne * magnitudeTwo));
	//reduce(distance, distance, 1, REDUCE_SUM);

}

void DistanceCalculator::do_manhattan(float* dataset, int minPoints) {

	//absdiff(attributesOne, attributesTwo, distance);
	//reduce(distance, distance, 1, REDUCE_SUM);

	/*for (unsigned int i = 0; i < attributesOne->size() && i < attributesTwo->size(); i++) {
		distance += abs((*attributesOne)[i] - (*attributesTwo)[i]);
	}*/
}

void DistanceCalculator::do_pearson(float* dataset, int minPoints) {

/*

	for (unsigned int i = 0; i < attributesOne.size() && i < attributesTwo->size(); i++) {
		meanOne += (*attributesOne)[i];
		meanTwo += (*attributesTwo)[i];
	}
*/

	/*reduce(attributesOne, meanOne, 1, REDUCE_SUM);
	reduce(attributesTwo, meanTwo, 1, REDUCE_SUM);

	meanOne = meanOne / attributesOne.cols;
	meanTwo = meanTwo / attributesTwo.cols;

	UMat covariance;
	UMat standardDeviationOne;
	UMat standardDeviationTwo;*/



	/*for (unsigned int i = 0; i < attributesOne->size() && i < attributesTwo->size(); i++) {
		covariance += (((*attributesOne)[i] - meanOne)
				* ((*attributesTwo)[i] - meanTwo));
		standardDeviationOne += (((*attributesOne)[i] - meanOne)
				* ((*attributesOne)[i] - meanOne));
		standardDeviationTwo += (((*attributesTwo)[i] - meanTwo)
				* ((*attributesTwo)[i] - meanTwo));
	}*/

	//distance = (1
			//- (covariance / (standardDeviationOne * standardDeviationTwo)));
}

void DistanceCalculator::do_supremum(float* dataset, int minPoints) {

	/*for (unsigned int i = 0; i < attributesOne->size() && i < attributesTwo->size(); i++) {
		double difference = abs((*attributesOne)[i] - (*attributesTwo)[i]);
		if (difference > distance)
			distance = difference;
	}*/

}

/**
 *
 */
uint DistanceCalculator::triangular(uint n){
	return (n * n + n)/2;
}

/**
 *
 */
float DistanceCalculator::getDistance(uint row, uint col){

	uint idx;
	if (row < col) {
		//printf("row > col\n");
		idx = (rows * row + col) - triangular(row + 1);

	} else if (row == col) {
		//printf("row == col\n");
		return 0;
	} else {
		//printf("row < col\n");
		idx = (rows * col + row) - triangular(col + 1);
	}

	//printf("(rows * row + col) - triangular(row + 1) : %d\n", (rows * row + col) - triangular(row + 1));
	//printf("(rows * col + row) - triangular(col + 1) : %d\n", (rows * col + row) - triangular(col + 1));

	//printf("getDistance at index %d\n", idx);
	return distance[idx];
}

}
}
