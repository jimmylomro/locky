/*
    Copyright (C) 2016 Jaime Lomeli-R. Univesity of Southampton

    This file is part of OWN.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
       * Redistributions of source code must retain the above copyright
         notice, this list of conditions and the following disclaimer.
       * Redistributions in binary form must reproduce the above copyright
         notice, this list of conditions and the following disclaimer in the
         documentation and/or other materials provided with the distribution.
       * Neither the name of the ASL nor the names of its contributors may be
         used to endorse or promote products derived from this software without
         specific prior written permission.

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

OwnFeatureMaps:
    There are two ways of creating the feature maps, you could either call detectKeypoints and send a valid image
    or you could first call createFeatureMaps. The first method will also detect the keypoints.
    If detectKeypoints is called without an image argument and createFeatureMaps has not been called (i.e. featureMaps is empty)
    the function will throw an error.
*/

#ifndef _LOCKY_H_
#define _LOCKY_H_


#include <opencv2/features2d/features2d.hpp>
#include <opencv2/core/core.hpp>


namespace locky{
	
	
	class CV_EXPORTS_W LOCKYFeatureDetector : public cv::Feature2D {
	public:

		static cv::Ptr<LOCKYFeatureDetector> create(int iter = 100000, int maxExp = 7, int minExp = 3, int thresh = 30);
		
		virtual void getAccumMat(cv::Mat& toReturn) = 0;

    	virtual void detect(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints) = 0;

	};

}

#endif // _LOCKY_H_