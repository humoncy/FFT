#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <bitset>
using namespace std;
using namespace cv;

const double pi = 2.0 * acos(0);

void bitReversal();
void zeroPadding(Mat & SrcImg, Mat & DstImg);
void shiftToMiddle(Mat & magI);
void FFT2D(Mat & SrcImg, Mat & DstImg);

void dftOpencv(Mat & I)
{
	Mat padded;                            //expand input image to optimal size
	int m = getOptimalDFTSize(I.rows);
	int n = getOptimalDFTSize(I.cols); // on the border add zero values
	copyMakeBorder(I, padded, 0, m - I.rows, 0, n - I.cols, BORDER_CONSTANT, Scalar::all(0));
	// imshow("Padded", padded);

	// just a array named planes with 2 elements
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

	// rearrange the quadrants of Fourier image so that the origin is at the image center
	int cx = magI.cols / 2;
	int cy = magI.rows / 2;

	// Here: col: x, row: y
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

	normalize(magI, magI, 0, 1, CV_MINMAX); // Transform the matrix with float values into a
											// viewable image form (float between values 0 and 1).

	imshow("Input Image", I);    // Show the result
	imshow("spectrum magnitude", magI);
	waitKey();

	cv::Mat inverseTransform;
	cv::dft(complexI, inverseTransform, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT);
	normalize(inverseTransform, inverseTransform, 0, 1, CV_MINMAX);
	imshow("Reconstructed", inverseTransform);
	waitKey();
}

int main() {
	// Read input images
	// Fig3.tif is in openCV\bin\Release
	Mat SrcImg1 = imread("Origin Image/Q1.tif", CV_LOAD_IMAGE_GRAYSCALE);
	Mat SrcImg2 = imread("Origin Image/Q2.tif", CV_LOAD_IMAGE_GRAYSCALE);
	Mat SrcImg3 = imread("Origin Image/Q3.tif", CV_LOAD_IMAGE_GRAYSCALE);

	if (SrcImg1.empty() || SrcImg2.empty() || SrcImg3.empty()) {
		return -1;
	}
	imshow("Input Image", SrcImg1);


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

	//dftOpencv(SrcImg1);

	FFT2D(SrcImg1, DstImg1);
	//FFT2D(SrcImg2, DstImg2);
	//FFT2D(SrcImg3, DstImg3);

	// Show images
	imshow("Output Image1", DstImg1);
	//imshow("Output Image2", DstImg2);
	//imshow("Output Image3", DstImg3);

	// Write output images
	//imwrite("New Image/Q1.tif", DstImg1);
	//imwrite("New Image/Q2.tif", DstImg2);
	//imwrite("New Image/Q3.tif", DstImg3);

	waitKey(0);
	return 0;
}

void zeroPadding(Mat & SrcImg, Mat & DstImg)
{
	int new_cols = SrcImg.cols;
	int new_rows = SrcImg.rows;
	while (new_cols & (new_cols -1)) ++new_cols;
	while (new_rows & (new_rows - 1)) ++new_rows;

	cout << "Original: " << SrcImg.rows << ',' << SrcImg.cols << endl;
	cout << "Padded: " << new_rows << ',' << new_cols << endl;

	Mat PaddedImg(new_rows, new_cols, CV_8UC1, Scalar(0));
	SrcImg.copyTo(PaddedImg(Rect(0,0, SrcImg.cols, SrcImg.rows)));

	DstImg = PaddedImg;
	imshow("Padded", DstImg);
}

vector<int> bitReversal(const int N)
{
	vector<int> bit_reverse_indices(N);

	for (int i = 0; i < N; i++) {
		bit_reverse_indices[i] = i;
		//cout << bit_reverse_indices[i] << ' ';
	}
	cout << endl << endl;

	// bit reversal: i adds from least significant bit, while j adds from most significant bit
	for (int i = 1, j = 0; i < N; ++i)
	{
		for (int k = N >> 1; !((j ^= k)&k); k >>= 1) {
		}
		//for (int k=N>>1; k>(j^=k); k>>=1) ;

		// i>j or i<j are both used for ensuring only swap once between corresponding bit-place
		if (i>j) swap(bit_reverse_indices[i], bit_reverse_indices[j]);
		//if (i<j) swap(x[i], x[j]);

	}

	/*for (int i = 0; i < N; i++) {
		cout << bit_reverse_indices[i] << ' ';
	}
	cout << endl;*/

	return bit_reverse_indices;
}

void logTransform(Mat& DstFloat, Mat& DstImg)
{
	for (int i = 0; i < DstImg.rows; i++) {
		for (int j = 0; j < DstImg.cols; j++) {
			DstImg.at<uchar>(i, j) = log(1 + DstFloat.at<float>(i, j)) / log(1 + DstFloat.at<float>(0, 0)) * 255;
		}
	}
}

void FFT(int N, complex<double> *x) {
	// N must be a power of 2 (2 ^ n)

	// bit-reversal permutation
	for (int i = 1, j = 0; i<N; ++i) {
		for (int k = N >> 1; !((j ^= k)&k); k >>= 1);
		if (i>j) swap(x[i], x[j]);

		// for (int k=N>>1; k>(j^=k); k>>=1);
		// if (i<j) swap(x[i], x[j]);
	}

	// dynamic programming 
	for (int k = 2; k <= N; k <<= 1) { // k *= 2
		double w = -2.0 * pi / k;
		complex<double> dsida(cos(w), sin(w));

		for (int j = 0; j < N; j += k) { // 每k個做一次FFT
			complex<double> sida(1, 0);

			for (int i = j; i < j + k / 2; i++) { // 前k/2個與後k/2的三角函數值恰好對稱，因此兩兩對稱的一起做
				complex<double> a = x[i];
				complex<double> b = x[i + k / 2] * sida;
				x[i] = a + b;
				x[i + k / 2] = a - b;
				sida *= dsida;
			}
		}
	}
}

void FFT2D(Mat & SrcImg, Mat & DstImg)
{
	Mat padded;
	zeroPadding(SrcImg, padded);

	complex<double> **DstTemp = new complex<double>*[padded.rows];
	for (int i = 0; i < padded.rows; i++) {
		DstTemp[i] = new complex<double>[padded.cols];
		for (int j = 0; j < padded.cols; j++) {
			DstTemp[i][j] = (complex<double>)(double)DstImg.at<uchar>(i, j);
		}
	}

	//Mat_ < complex<double>>  complexDst(DstImg.rows, DstImg.cols);
	//for (int i = 0; i < DstImg.rows; i++) {
	//	for (int j = 0; j < DstImg.cols; j++) {
	//		complexDst.at<complex<double>>(i, j) = (complex<double>)(double)DstImg.at<uchar>(i, j);
	//		//cout << complexDst.at<complex<double>>(i, j) << ' ';
	//	}
	//}

	vector<int> bit_reverse_indices_col = bitReversal(padded.cols);
	/*for (int i = 0; i < DstImg.cols; i++) {
		cout << bit_reverse_indices_col[i] << ' ';
	}
	cout << endl;*/

	for (int rowIndex = 0; rowIndex < padded.rows; rowIndex++) {
		for (int k = 2; k <= padded.cols; k <<= 1) {
			double w = -2.0 * pi / k;
			complex<double> dtheta(cos(w), sin(w));

			// do FFT every ks
			for (int j = 0; j < padded.cols; j += k) {
				complex<double> theta(1, 0);
				for (int i = j; i < j + k / 2; i++) {
					complex<double> a = DstTemp[rowIndex][bit_reverse_indices_col[i]];
					complex<double> b = DstTemp[rowIndex][bit_reverse_indices_col[i + k / 2]] * theta;
					DstTemp[rowIndex][bit_reverse_indices_col[i]] = a + b;
					DstTemp[rowIndex][bit_reverse_indices_col[i + k / 2]] = a - b;

					theta *= dtheta;
				}
			}
		}
	}
	
	vector<int> bit_reverse_indices_row = bitReversal(padded.rows);

	for (int colIndex = 0; colIndex < padded.cols; colIndex++) {
		for (int k = 2; k <= padded.rows; k <<= 1) {
			double w = -2.0 * pi / k;
			complex<double> dtheta(cos(w), sin(w));

			for (int j = 0; j < padded.rows; j += k) {
				complex<double> theta(1, 0);
				for (int i = j; i < j + k / 2; i++) {
					complex<double> a = DstTemp[bit_reverse_indices_row[i]][colIndex];
					complex<double> b = DstTemp[bit_reverse_indices_row[i + k / 2]][colIndex] * theta;
					DstTemp[bit_reverse_indices_row[i]][colIndex] = a + b;
					DstTemp[bit_reverse_indices_row[i + k / 2]][colIndex] = a - b;
					
					theta *= dtheta;
				}
			}
		}
	}

	Mat DstFloat(padded.rows, padded.cols, CV_32F, Scalar(0));

	for (int i = 0; i < padded.rows; i++) {
		for (int j = 0; j < padded.cols; j++) {
			DstFloat.at<float>(i, j) = abs(DstTemp[i][j]);
		}
	}

	//normalize(DstFloat, DstFloat, 0, 1, CV_MINMAX);

	logTransform(DstFloat, padded);

	cout << (int)padded.at<uchar>(0, 0);
	
	DstImg = padded;
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
