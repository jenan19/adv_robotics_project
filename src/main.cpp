#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>



int main() {


    std::cout << "henlo u stinky\n";

    cv::Mat img = cv::imread("data/DinoSR/dinoSR0001.png");

    cv::imshow("dino UwU", img);
    cv::waitKey(0);

    return 1;
}