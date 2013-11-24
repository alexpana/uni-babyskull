#include "faps.h"

#include <chrono>
#include <sstream>

#undef FLANN_DEPRECATED

using namespace std;
using namespace cv;

static chrono::high_resolution_clock::time_point faps_time_point;

void faps_init(){
	// pass
}

void faps_timer_start(){
	faps_time_point = chrono::high_resolution_clock::now();
}

int faps_timer_stop(){
	using namespace chrono;
	high_resolution_clock::time_point now = high_resolution_clock::now();
	return (int)duration_cast<chrono::milliseconds>(now - faps_time_point).count();
}

Mat faps_open_image(int argc, char* argv[], const string& image_path) {
	string imageSourcePath = image_path;

	if( argc >= 2 ){
		imageSourcePath = argv[1];
	}

	return cvLoadImage(imageSourcePath.c_str());
}

Mat faps_open_image(const string& image_path) {
	return cvLoadImage(image_path.c_str());
}

void faps_display_image(const string& name, const Mat& image){
	namedWindow(name, CV_WINDOW_AUTOSIZE);
	imshow(name, image);
	waitKey(1);
}

void faps_wait(){
	waitKey(0);
}

std::vector<std::string> faps_string_split(const std::string& str_value, const std::string& split_value)
{
	vector<string> result;
	string token = "";
	int i = 0;
	while (i < str_value.length()) {
		while (i < str_value.length() && split_value.find(str_value[i]) != string::npos) {
			++i;
		}
		while (i < str_value.length() && split_value.find(str_value[i]) == string::npos) {
			token += str_value[i];
			++i;
		}
		result.push_back(token);
		token = "";
	}
	return result;
}

int faps_to_int(const string& string_value) {
	int int_value;
	istringstream(string_value) >> int_value;
	return int_value;
}