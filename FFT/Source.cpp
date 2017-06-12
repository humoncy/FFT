#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
using namespace std;
using namespace cv;

const double £k = 2.0 * acos(0);
const int N = 8;
int x[N];

int main() {

	// Read input images
	// Fig3.tif is in openCV\bin\Release
	Mat SrcImg1 = imread("Origin Image/Q1.tif", CV_LOAD_IMAGE_GRAYSCALE);
	Mat SrcImg2 = imread("Origin Image/Q2.tif", CV_LOAD_IMAGE_GRAYSCALE);
	Mat SrcImg3 = imread("Origin Image/Q3.tif", CV_LOAD_IMAGE_GRAYSCALE);

	// Create a grayscale output image matrix
	Mat DstImg1 = Mat(SrcImg1.rows, SrcImg1.cols, CV_8UC1);
	Mat DstImg2 = Mat(SrcImg2.rows, SrcImg2.cols, CV_8UC1);
	Mat DstImg3 = Mat(SrcImg3.rows, SrcImg3.cols, CV_8UC1);

	// Copy each pixel of the source image to the output image
	/*for(int i=0; i<SrcImg1.rows; ++i) {
		for (int j = 0; j < SrcImg1.cols; ++j)
		{
			DstImg1.at<uchar>(i, j) = SrcImg1.at<uchar>(i, j);
		}
	}*/

	for (int i = 0; i < N; i++) {
		x[i] = i;
		//cout << x[i] << ' ';
	}
	//cout << endl;

	for (int i = 1, j = 0; i<N; ++i)
	{
		for (int k = N >> 1; !((j ^= k)&k); k >>= 1);
		//for (int k=N>>1; k>(j^=k); k>>=1) ;
		if (i>j) swap(x[i], x[j]);
		//if (i<j) swap(x[i], x[j]);
	}

	for (int i = 0; i < N; i++) {
		//cout << x[i] << ' ';
	}
	//cout << endl;

	// Show images
	imshow("Input Image", SrcImg1);
	imshow("Output Image", DstImg1);

	// Write output images
	imwrite("New Image/Q1.tif", DstImg1);
	//imwrite("New Image/Q2.tif", DstImg2);
	//imwrite("New Image/Q3.tif", DstImg3);

	waitKey(0);
	return 0;
}