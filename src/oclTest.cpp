#include "opencv2/core.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

bool checkOpenCL()
{
  cv::ocl::Context ctx = cv::ocl::Context::getDefault();
  if (!ctx.ptr())
  {
      std::cerr << "OpenCL is not available" << std::endl;
      return 0;
  }

  return 1;
}

int main()
{

  int succes = 0;

  if(checkOpenCL())
  {
    succes = 1;
  }

  std::cout << "Succes = " << succes;
  return succes;

}