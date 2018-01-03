#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <bitset>
using namespace std;
using namespace cv;

const double pi = 2.0 * acos(0);

void zeroPadding(Mat & SrcImg, Mat & DstImg);
void shiftToMiddle(Mat & magI);
void FFT1D(complex<double> x[], int N);
Mat FFT2D(Mat & SrcImg);

Mat dftOpencv(Mat & I)
{
	Mat padded;                            //expand input image to optimal size
	int m = getOptimalDFTSize(I.rows);
	int n = getOptimalDFTSize(I.cols); // on the border add zero values
	copyMakeBorder(I, padded, 0, m - I.rows, 0, n - I.cols, BORDER_CONSTANT, Scalar::all(0));
	// imshow("Padded", padded);

	// Just a Mat array with 2 elements
	// planes[0] = Mat_<float>(padded)
	// planes[1] = Mat::zeros(padded.size(), CV_32F)
	Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };

	Mat complexI;
	merge(planes, 2, complexI);         // Add to the expanded another plane with zeros

	dft(complexI, complexI);            // this way the result may fit in the source matrix

										// compute the magnitude and switch to logarithmic scale
										// => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
	split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
	magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
	Mat magI = planes[0];

	magI += Scalar::all(1);                    // switch to logarithmic scale
	log(magI, magI);

	// crop the spectrum, if it has an odd number of rows or columns
	// ( -2 = 111110 in binary )
	magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));

	shiftToMiddle(magI);
	normalize(magI, magI, 0, 1, CV_MINMAX); // Transform the matrix with float values into a
											// viewable image form (float between values 0 and 1).

	// imshow("Input Image", I);    // Show the result
	/*imshow("spectrum magnitude", magI);
	waitKey();

	cv::Mat inverseTransform;
	cv::dft(complexI, inverseTransform, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT);
	normalize(inverseTransform, inverseTransform, 0, 1, CV_MINMAX);
	imshow("Reconstructed", inverseTransform);
	waitKey();*/

	return magI;
}

int main() {
	// Read input images
	// Fig3.tif is in openCV\bin\Release
	Mat SrcImg1 = imread("Origin Image/Q1.tif", CV_LOAD_IMAGE_GRAYSCALE);
	Mat SrcImg2 = imread("Origin Image/Q2.tif", CV_LOAD_IMAGE_GRAYSCALE);
	Mat SrcImg3 = imread("Origin Image/Q3.tif", CV_LOAD_IMAGE_GRAYSCALE);
	Mat SrcImg4 = imread("Origin Image/Q4.tif", CV_LOAD_IMAGE_GRAYSCALE);

	if (SrcImg1.empty() || SrcImg2.empty() || SrcImg3.empty(), SrcImg4.empty()) {
		return -1;
	}
	imshow("Input Image", SrcImg2);

	//Mat DstImg1 = dftOpencv(SrcImg1);
	Mat DstImg1 = FFT2D(SrcImg1);
	Mat DstImg2 = FFT2D(SrcImg2);
	Mat DstImg3 = FFT2D(SrcImg3);
	Mat DstImg4 = FFT2D(SrcImg4);

	// Show images
	imshow("Output Image1", DstImg2);
	//imshow("Output Image2", DstImg2);
	//imshow("Output Image3", DstImg3);

	normalize(DstImg1, DstImg1, 0, 255, CV_MINMAX);
	normalize(DstImg2, DstImg2, 0, 255, CV_MINMAX);
	normalize(DstImg3, DstImg3, 0, 255, CV_MINMAX);
	normalize(DstImg4, DstImg4, 0, 255, CV_MINMAX);

	DstImg1.convertTo(DstImg1, CV_8U);
	DstImg2.convertTo(DstImg2, CV_8U);
	DstImg3.convertTo(DstImg3, CV_8U);
	DstImg4.convertTo(DstImg4, CV_8U);

	// Write output images
	imwrite("New Image/Q1.tif", DstImg1);
	imwrite("New Image/Q2.tif", DstImg2);
	imwrite("New Image/Q3.tif", DstImg3);
	imwrite("New Image/Q4.tif", DstImg3);

	waitKey(0);
	return 0;
}

void zeroPadding(Mat & SrcImg, Mat & DstImg)
{
	int new_cols = SrcImg.cols;
	int new_rows = SrcImg.rows;
	while (new_cols & (new_cols -1)) ++new_cols;
	while (new_rows & (new_rows - 1)) ++new_rows;

	cout << "Original image size: " << SrcImg.size() << endl;

	Mat PaddedImg(new_rows, new_cols, CV_8UC1, Scalar(0));
	SrcImg.copyTo(PaddedImg(Rect(0,0, SrcImg.cols, SrcImg.rows)));

	DstImg = PaddedImg;

	cout << "Padded image size: " << DstImg.size() << endl;

	// imshow("Padded", DstImg);
	//cout << SrcImg.size() << endl;
	//cout << DstImg.size() << endl;
}

// Cooley-Tukey Algorithm
void FFT1D(complex<double> x[], int N) {
	// N must be power of 2

	// bit-reversal permutation
	for (int i = 1, j = 0; i<N; ++i) {
		for (int k = N >> 1; !((j ^= k)&k); k >>= 1);
		if (i>j) swap(x[i], x[j]);
	}

	/* dynamic programming */
	for (int k = 2; k <= N; k <<= 1)
	{
		float theta = -2.0 * pi / k;
		// Here, let w represents omega
		complex<double> dw(cos(theta), sin(theta));

		for (int j = 0; j < N; j += k)
		{
			complex<double> w(1, 0);
			for (int i = j; i < j + k / 2; i++)
			{
				complex<double> a = x[i];
				complex<double> b = x[i + k / 2] * w;
				x[i] = a + b;
				x[i + k / 2] = a - b;
				w *= dw;
			}
		}
	}
}

Mat FFT2D(Mat & SrcImg)
{
	Mat padded;
	zeroPadding(SrcImg, padded);

	Mat planes[] = { Mat_<double>(padded), Mat_<double>::zeros(padded.size()) };

	Mat complexI;
	merge(planes, 2, complexI);

	for (int i = 0; i < padded.rows; i++)
	{
		complex<double> *tmp = new complex<double>[padded.cols];
		for (int j = 0; j < padded.cols; j++) {
			tmp[j] = complex<double>(complexI.at<Vec2d>(i, j)[0], complexI.at<Vec2d>(i, j)[1]);
		}
		FFT1D(tmp, padded.cols);
		for (int j = 0; j < padded.cols; j++) {
			complexI.at<Vec2d>(i, j)[0] = tmp[j].real();
			complexI.at<Vec2d>(i, j)[1] = tmp[j].imag();
		}
	}

	for (int j = 0; j < padded.cols; j++)
	{
		complex<double> *tmp = new complex<double>[padded.rows];
		for (int i = 0; i < padded.rows; i++) {
			tmp[i] = complex<double>(complexI.at<Vec2d>(i, j)[0], complexI.at<Vec2d>(i, j)[1]);
		}
		FFT1D(tmp, padded.rows);
		for (int i = 0; i < padded.rows; i++) {
			complexI.at<Vec2d>(i, j)[0] = tmp[i].real();
			complexI.at<Vec2d>(i, j)[1] = tmp[i].imag();
		}
	}

	split(complexI, planes);                      // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
	magnitude(planes[0], planes[1], planes[0]);   // planes[0] = magnitude
	Mat magI = planes[0];

	magI += Scalar::all(1);                       // switch to logarithmic scale
	log(magI, magI);

	// crop the spectrum, if it has an odd number of rows or columns
	// ( -2 = 111110 in binary )
	magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));

	shiftToMiddle(magI);
	normalize(magI, magI, 0, 1, CV_MINMAX); // Transform the matrix with float values into a
											// viewable image form (float between values 0 and 1).
	//imshow("spectrum magnitude", magI);
	//waitKey();
	return magI;
}

void shiftToMiddle(Mat & magI)
{
	// rearrange the quadrants of Fourier image  so that the origin is at the image center
	int cx = magI.cols / 2;
	int cy = magI.rows / 2;

	Mat q0(magI, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
	Mat q1(magI, Rect(cx, 0, cx, cy));  // Top-Right
	Mat q2(magI, Rect(0, cy, cx, cy));  // Bottom-Left
	Mat q3(magI, Rect(cx, cy, cx, cy)); // Bottom-Right

	Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
	q2.copyTo(q1);
	tmp.copyTo(q2);
}
