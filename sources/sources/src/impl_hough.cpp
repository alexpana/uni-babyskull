#include "impl_hough.h"

#include <iostream>

using namespace std;
using namespace cv;

#define TO_PI 0.01745329251

IplImage* process_hough(const Mat& src, int window_size, int peak_threshold){

	IplImage sourceImage = (IplImage)src;
	int peak_count = 8;

	// Initialize Hough Accumulator Space
	int h_width = 360;
	int h_height = (int)sqrt(src.rows * src.rows + src.cols * src.cols) + 2;
	double* hough_space[360];
	for (int i = 0; i < 360; ++i){
		hough_space[i] = new double[h_height];
		memset(hough_space[i], 0, h_height * sizeof(double));
	}
	IplImage *hough_img = cvCreateImage(cvSize(h_width, h_height), sourceImage.depth, sourceImage.nChannels);

	cout << "Hough space is: " << h_width << "x" << h_height << endl;

	// Fill the Hough Accumulator
	for (int i = 0; i < sourceImage.width; ++i) {
		for (int j = 0; j < sourceImage.height; ++j) {
			for (int theta = 0; theta < 360; ++theta) {
				int d = (int)((float)i * cos((float)theta * TO_PI) + (float)j * sin((float)theta * TO_PI));
				if (d > 0 && d < h_height - 1) {
					double intensity = cvGet2D(&sourceImage, i, j).val[0];
					hough_space[theta][d] += intensity;
				}
			}
		}
	}

	// Compute the maximum intensity
	double max_intensity = 0;
	for (int i = 0; i < hough_img->width; ++i) {
		for (int j = 0; j < hough_img->height; ++j) {
			max_intensity = max(max_intensity, hough_space[i][j]);
		}
	}

	// Normalize the Hough Space and build the image
	for (int i = 0; i < hough_img->width; ++i) {
		for (int j = 0; j < hough_img->height; ++j) {
			double acc = (hough_space[i][j] / max_intensity) * 255;
			hough_space[i][j] = acc;
			cvSet2D(hough_img, j, i, cvScalar(acc, acc, acc));
		}
	}

	// Display the Hough Accumulator
	faps_display_image("Result Image", Mat(hough_img));

	// Find the peaks
	vector<Point2i> peaks;
	int hws = window_size / 2 + 1;
	for (int i = hws; i < h_width - hws; ++i) {
		for (int j = hws; j < h_height - hws; ++j) {
			if (hough_space[i][j] > peak_threshold){
				bool is_peak = true;
				for (int wi = -hws+1; wi < hws-1; ++wi) {
					for (int wj = -hws+1; wj < hws-1; ++wj) {
						if (hough_space[i + wi][j + wj] > hough_space[i][j]) {
							is_peak = false;
						}
					}
				}
				if (is_peak){
					peaks.push_back(Point2i(i, j));
				}
			}
		}
	}

	cout << "Found " << peaks.size() << " peaks with intensity greater than " << peak_threshold << endl;
	
	sort(peaks.begin(), peaks.end(), [&hough_space](const Point2i& lhs, const Point2i& rhs){ return hough_space[lhs.x][lhs.y] > hough_space[rhs.x][rhs.y]; });

	// Draw the lines on the original image
	IplImage source_image_copy = IplImage(sourceImage);
	for (int i = 0; i < peaks.size() && i < peak_count; ++i) {
		double theta = peaks[i].x * TO_PI;
		double rho = peaks[i].y;

		int a = peaks[i].y / cos(3.1415 / 2.0 - peaks[i].x * TO_PI);
		int b = peaks[i].y / cos(peaks[i].x * TO_PI);

		int x = cos(theta) * rho;
		int y = sin(theta) * rho;

		int x1 = cos(- 3.1415 / 2.0 + theta) * 200;
		int x2 = cos(- 3.1415 / 2.0 + theta) * -200;

		int y1 = sin(- 3.1415 / 2.0 + theta) * 200;
		int y2 = sin(- 3.1415 / 2.0 + theta) * -200;


		cvLine(&source_image_copy, cvPoint(x, y), cvPoint(x + x1, y + y1), cvScalar(0, 0, 255), 1);
		cvLine(&source_image_copy, cvPoint(x, y), cvPoint(x + x2, y + y2), cvScalar(0, 0, 255), 1);
		//cvCircle(&source_image_copy, cvPoint(x, y), 1, cvScalar(0, 0, 255));
	}

	//cvLine(&source_image_copy, cvPoint(-100, 5), cvPoint(400, 5), cvScalar(0, 0, 255), 1);

	// Display the lines over the original image
	faps_display_image("Hough Lines", Mat(&source_image_copy));

	// Cleanup
	for (int i = 0; i < 360; ++i){
		delete[] hough_space[i];
	}

	return hough_img;
}
