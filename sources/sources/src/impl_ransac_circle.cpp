#include "impl_ransac_circle.h"

#include <iostream>

#include "faps.h"

using namespace cv;
using namespace std;

Point2f find_center(const Point2f& p1, const Point2f& p2, const Point2f& p3){
	float ma = (p2.y - p1.y) / (p2.x - p1.x);
	float mb = (p3.y - p2.y) / (p3.x - p2.x);

	Point2f center;
	center.x = (float)(ma * mb * (p1.y - p3.y) + mb * (p1.x + p2.x) - ma * (p2.x + p3.x)) / (float)(2 * (mb - ma));
	center.y = -1.0f / ma * (center.x - (float)(p1.x + p2.x) / 2.0f) + (float)(p1.y + p2.y) / 2.0f;

	return center;
}

float distance(const Point2f& p1, const Point2f& p2){
	return sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));
}

float distance_from_circle(const Point2f& c, float radius, const Point2f& p){
	return abs(sqrt((c.x - p.x) * (c.x - p.x) + (c.y - p.y) * (c.y - p.y)) - radius);
}

bool collinear(const Point2f& p1, const Point2f& p2, const Point2f& p3, float e){
	return false;
}

IplImage* process_ransac_circle(const Mat& src){
	// parameters
	int t = 10;
	float p = 0.99f;
	float w = 0.5f;
	int s = 3;

	IplImage sourceImg = (IplImage)src;
	IplImage *dst = cvCreateImage(cvGetSize(&sourceImg), sourceImg.depth, sourceImg.nChannels);

	vector<Point2f> points;

	for (int i = 0; i < sourceImg.width; ++i){
		for (int j = 0; j < sourceImg.height; ++j){
			CvScalar color = cvGet2D(&sourceImg, j, i);
			cvSet2D(dst, j, i, color);
			if (color.val[0] == 0){
				points.push_back(Point2f(i, j));
			}
		}
	}

	int N = log(1.0 - p) / log(1-w * w *w);
	int T = w * points.size();
	cout << "N: " << N << endl << "T: " << T << endl;

	cout << "Found " << points.size() << " total points." << endl;

	int max_count = 3;
	CvPoint max_center;
	float max_radius = 0;

	for (int i = 0; i < N; ++i){
		int p1, p2, p3;
		do{
			p1 = faps_random(0, (int)points.size());
			p2 = faps_random(0, (int)points.size());
			p3 = faps_random(0, (int)points.size());
		} while (collinear(points[p1], points[p2], points[p3], 0.001));

		Point2f circle_center = find_center(points[p1], points[p2], points[p3]);
		float circle_radius = sqrt((circle_center.x - points[p1].x) * (circle_center.x - points[p1].x) + 
			(circle_center.y - points[p1].y) * (circle_center.y - points[p1].y));

		int count = 0;
		for( vector<Point2f>::iterator it = points.begin(); it != points.end(); ++it){
			float d = distance_from_circle(circle_center, circle_radius, *it);
			if (d < t){
				count += 1;
			}
		}

		if (count > max_count){
			max_count = count;
			max_center = cvPoint(circle_center.x, circle_center.y);
			max_radius = circle_radius;
		}

		if (max_count >= T){
			break;
		}
	}

	cout << "Found " << max_count << " matching points." << endl;

	cvCircle(dst, max_center, max_radius, cvScalar(255, 0, 0));

	return dst;
}
