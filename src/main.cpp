#include <vector>
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

void loadImages(std::string *path, std::string *seriesName,std::vector<cv::Mat> *images, int numberOfImages)
{
    for (int i = 0; i < numberOfImages; i++)
    {
        // pushback images
        images->push_back(cv::Mat(cv::imread(*path + *seriesName + std::to_string(i + 1) + ".png")));
    }
    


}




void removeBackground(std::vector<cv::Mat> *images, double threshold, double maxval)
{
    for (int i = 0; i < images->size(); i++)
    {
        cv::threshold(images->at(i),images->at(i), threshold, maxval, cv::THRESH_TOZERO);
    }
    


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

    std::string path = "../data/DinoSR/";
    std::string series = "dinoSR";
    int numberOfimages = 16;


    loadImages(&path, &series, &images, numberOfimages);

    //showImages(&images);

    removeBackground(&images,25,200);
    
    showImages(&images);

    //cv::imshow("dino UwU", img);
    cv::waitKey(0);

    return 1;
}