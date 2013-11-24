#pragma once

#include <string>
#include <vector>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"

void faps_init();

void faps_timer_start();

int faps_timer_stop();

cv::Mat faps_open_image(const std::string& image_path);

cv::Mat faps_open_image(int argc, char* argv[], const std::string& image_path);

void faps_display_image(const std::string& name, const cv::Mat& image);

std::vector<std::string> faps_string_split(const std::string& str_value, const std::string& split_value);

void faps_wait();

int faps_to_int(const std::string& string_value);

template<typename ProcessFunction>
cv::Mat faps_process_image(const std::string& processName, ProcessFunction processFunction, const cv::Mat& image){
	cout << "Running process " << processName << endl;

	faps_timer_start();
	IplImage *result = processFunction(image);
	cout << "Process " << processName << " finished in " << faps_timer_stop() << "ms." << endl;

	//faps_display_image("Source Image", src);
	faps_display_image("Result Image", cv::Mat(result));

	return cv::Mat(result);
}

template<typename T>
T faps_random(T min, T max){
	return ((T)(rand()) * max) / (T)RAND_MAX;
}