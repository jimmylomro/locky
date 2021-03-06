/*
    This file is part of LOCKY.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
    ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    
    Author: Jaime Lomeli-R.
*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include <ctime>

#include <locky.h>

using namespace std;


int main(int argc, char ** argv) {


	cv::Mat image = cv::imread("../res/im.jpg");
	cv::Mat accumMat;
	std::vector<cv::KeyPoint> keypoints;


//--------------- LOCKYS
	cv::Ptr<locky::LOCKYFeatureDetector> detector = locky::LOCKYFeatureDetector::create();
	
	clock_t begin = clock();
	detector->detect(image, keypoints);
	clock_t end = clock();
	
	double secs = double(end-begin)/CLOCKS_PER_SEC;
	cout << "Detection time BCTS = " << secs << endl;
	cout << "Number of keypoints = " << keypoints.size() << endl;

	// Draw keypoints on original image and show
	cv::Mat lockys;
	cv::drawKeypoints(image, keypoints, lockys);
	cv::namedWindow("LOCKYS");
	cv::imshow("LOCKYS", lockys);
	
	detector->getAccumMat(accumMat);
	cv::namedWindow("BCTS");
	cv::imshow("BCTS", accumMat);
	
//--------------- LOCKY
	detector = locky::LOCKYFeatureDetector::create(100000,7,3,30,false);
	
	begin = clock();
	detector->detect(image, keypoints);
	end = clock();
	
	secs = double(end-begin)/CLOCKS_PER_SEC;
	cout << "Detection time BCT  = " << secs << endl;
	cout << "Number of keypoints = " << keypoints.size() << endl;

	// Draw keypoints on original image and show
	cv::Mat locky;
	cv::drawKeypoints(image, keypoints, locky);
	cv::namedWindow("LOCKY");
	cv::imshow("LOCKY", locky);
	
	detector->getAccumMat(accumMat);
	cv::namedWindow("BCT");
	cv::imshow("BCT", accumMat);



	cv::waitKey(0);

	return 0;
}
