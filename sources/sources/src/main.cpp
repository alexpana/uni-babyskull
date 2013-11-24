#include <iostream>
#include <map>
#include <string>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

static const double BINARIZATION_AREA_PERCENTAGE = 0.04;
static const string DEFAULT_IMAGE = "images/H_28weeks_01.bmp";
static const int ISLAND_COUNT = 6;

struct Island{
	int index;
	int area;

	Island(int _index, int _area) : index(_index), area(_area) {
	}

	Island() : index(0), area(0) {
	}
};

double* createHistogram(const Mat& source) {
	double* histogram = new double[255];
	memset(histogram, 0, 255 * sizeof(double));

	for (int i = 0; i < source.rows; ++i) {
		for (int j = 0; j < source.cols; ++j) {
			histogram[source.at<uchar>(i, j)] += 1.0 / (source.rows * source.cols);
		}
	}

	return histogram;
}

Mat convertToBinary(const Mat& source, const double areaPercent){
	double *histogram = createHistogram(source);

	double acc = 0.0;
	int threshold = 255;
	while (threshold > 0 && acc < BINARIZATION_AREA_PERCENTAGE) {
		acc += histogram[threshold];
		threshold -= 1;
	}

	Mat binaryImage = Mat(source.rows, source.cols, CV_8U);
	for (int i = 0; i < binaryImage.rows; ++i) {
		for (int j = 0; j < binaryImage.cols; ++j) {
			if (source.at<uchar>(i, j) > threshold) {
				binaryImage.at<uchar>(i, j) = 255;
			} else {
				binaryImage.at<uchar>(i, j) = 0;
			}
		}
	}

	return binaryImage;
}

Mat createIslandMask(const Mat& source) {
	// Disjoint Sets
	int dSets[2048];
	memset(dSets, 0, sizeof(dSets));

	int lastSet = 1;

	// Disjoint Set - Identify set
	auto dsParent = [&dSets](int set)->int {
		while (dSets[set] != 0){
			set = dSets[set];
		}
		return set;
	};

	// Disjoint Set - Merge two sets
	auto dsMerge = [&dSets, &dsParent](int left, int right) {
		int pLeft = dsParent(left);
		int pRight = dsParent(right);
		dSets[pLeft] = pRight;
	};
	
	Mat result = Mat(source.rows, source.cols, CV_32S);
	result *= 0;

	int dx[4] = { 1,  0, -1, -1};
	int dy[4] = {-1, -1, -1,  0};

	auto clamp = [](int a, int left, int right)->int {
		return max(min(a, right), left);
	};

	for (int i = 0; i < source.rows; ++i) {
		for (int j = 0; j < source.cols; ++j) {
			int srcVal = source.at<uchar>(i, j);
			if (srcVal != 0) {

				// search neighbours for islands
				for (int k = 0; k < 4; ++k) {
					int dstValNeighbour = result.at<int>(clamp(i + dy[k], 0, source.rows - 1), clamp(j + dx[k], 0, source.cols - 1));

					if (dstValNeighbour != 0){
						int dstVal = result.at<int>(i, j);

						if (result.at<int>(i, j) == 0) {
							result.at<int>(i, j) = dsParent(dstValNeighbour);
						} 
						else if (dsParent(dstVal) != dsParent(dstValNeighbour)) {
							dsMerge(dstVal, dstValNeighbour);
						}
					}
				}

				// begin a new island
				if (result.at<int>(i, j) == 0) {
					result.at<int>(i, j) = lastSet++;
				}
			}
		}
	}

	for (int i = 0; i < source.rows; ++i) {
		for (int j = 0; j < source.cols; ++j) {
			result.at<int>(i, j) = dsParent(result.at<int>(i, j));
		}
	}

	return result;
}

vector<Island> extractIslands(const Mat& source) {
	map<int, Island> islandMap;

	for (int i = 0; i < source.rows; ++i) {
		for (int j = 0; j < source.cols; ++j) {
			int srcValue = source.at<int>(i, j);
			if (srcValue != 0){
				if (islandMap.count(srcValue) == 0){
					islandMap[srcValue] = Island(srcValue, 0);
				}
				islandMap[srcValue].area += 1;
			}
		}
	}

	vector<Island> result;
	for (auto it = islandMap.begin(); it != islandMap.end(); ++it) {
		result.push_back(it->second);
	}

	std::sort(result.begin(), result.end(), [](const Island& left, const Island& right) { return left.area > right.area; });

	return result;
}

Mat filterImageByIslands(const Mat& islandMask, vector<Island> islands, int islandCount) {
	Mat result = Mat(islandMask.rows, islandMask.cols, CV_8U);
	result *= 0;

	for (int i = 0; i < islandMask.rows; ++i) {
		for (int j = 0; j < islandMask.cols; ++j) {
			if (islandMask.at<int>(i, j) == 0){
				continue;
			}

			for (int k = 0; k < min((int)islands.size(), islandCount); ++k) {
				if (islandMask.at<int>(i, j) == islands[k].index) {
					result.at<uchar>(i, j) = 255;
					break;
				}
			}
		}
	}

	return result;
}

int main(int argc, char* argv[]){
	string imageName = DEFAULT_IMAGE;
	if (argc >= 2) {
		imageName = argv[1];
	}

	cout << "Reading input image." << endl;
	Mat srcImage = imread(imageName, CV_LOAD_IMAGE_GRAYSCALE);

	cout << "Converting to binary." << endl;
	Mat binaryImage = convertToBinary(srcImage, BINARIZATION_AREA_PERCENTAGE);
	
	cout << "Creating pixel island mask." << endl;
	Mat islandMask = createIslandMask(binaryImage);

	cout << "Extracting pixel islands." << endl;
	vector<Island> islands = extractIslands(islandMask);

	cout << "Filtering image based on largest islands." << endl;
	Mat filteredImage = filterImageByIslands(islandMask, islands, ISLAND_COUNT);

	imshow("binary", binaryImage);
	//imshow("islandMask", islandMask);
	imshow("filtered", filteredImage);
	cvWaitKey(0);

	return 0;
}

