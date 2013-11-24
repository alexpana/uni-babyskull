#pragma once

#include "faps.h"

IplImage* process_hough(const cv::Mat& src, int window_size, int peak_threshold = 180);