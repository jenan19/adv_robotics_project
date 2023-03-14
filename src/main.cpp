#include <vector>
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

void loadImages(std::string *path, std::string *seriesName,std::vector<cv::Mat> *images, int numberOfImages)
{
    for (int i = 0; i < numberOfImages; i++)
    {
        // pushback images
        images->push_back(cv::Mat(cv::imread(*path + *seriesName + std::to_string(i + 1) + ".png",cv::IMREAD_GRAYSCALE)));
    }
    
}



void applyMorphology(std::vector<cv::Mat> *images, cv::Size maskSize_open,  cv::Size maskSize_close)
{
  for (int i = 0; i < images->size(); i++)
  {
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, maskSize_open);
    cv::morphologyEx(images->at(i), images->at(i), cv::MORPH_OPEN,kernel);

    kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, maskSize_close);
    cv::morphologyEx(images->at(i), images->at(i), cv::MORPH_CLOSE,kernel);
  }
  
    
}



void removeBackground(std::vector<cv::Mat> *images, double threshold, double maxval)
{
    for (int i = 0; i < images->size(); i++)
    {
        cv::inRange(images->at(i),cv::Scalar(threshold, threshold, threshold), cv::Scalar(maxval, maxval, maxval),images->at(i));
    }
    


}

void findContours(std::vector<cv::Mat> *images, std::vector<cv::Mat> *contours, double threshold)
{
    cv::RNG rng(12345);
    for (int i = 0; i < images->size(); i++)
    {
        std::vector<std::vector< cv::Point>> current_contour;
        cv::Mat canny_output;
        std::vector<cv::Vec4i> hierarchy;

        cv::Canny(images->at(i), canny_output, threshold, threshold*2);


        cv::findContours(images->at(i), current_contour,  hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);



        cv::Mat drawing = cv::Mat::zeros( canny_output.size(), CV_8UC3 );
        
        for( size_t j = 0; j< current_contour.size(); j++ )
        {
            cv::Scalar color = cv::Scalar( rng.uniform(100, 256), rng.uniform(100, 256), rng.uniform(100, 256) );
            cv::drawContours( drawing, current_contour, (int)j, color, 2, cv::LINE_8, hierarchy, 0 );
        }
        contours->push_back(drawing);

        cv::imshow("Drawing of contours UwU", drawing);
        cv::waitKey(0);
    }
}


void makeBinary(std::vector<cv::Mat> *images)
{


}



void showImages(std::vector<cv::Mat> *images)
{
    for (auto i : *images)
    {
        cv::imshow("dinos", i);
        cv::waitKey(0);
    }
}


int main() {

    std::cout << "henlo u stinky\n";

    std::vector<cv::Mat> images;

    std::vector<cv::Mat> contours;


    std::string path = "../data/DinoSR/";
    std::string series = "dinoSR";
    int numberOfimages = 16;


    loadImages(&path, &series, &images, numberOfimages);

    //showImages(&images);



    removeBackground(&images,30, 256);
    
    //showImages(&images);

    applyMorphology(&images, cv::Size (5,5), cv::Size (13,13));

    showImages(&images);

    //findContours(&images, &contours, 10);
    
    //showImages(&images);

    //cv::imshow("dino UwU", img);
    cv::waitKey(0);

    return 1;
}