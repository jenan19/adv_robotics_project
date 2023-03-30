#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 200





#include <vector>
#include <iostream>
#include <string>
#include <chrono>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
//#include <opencv2/core/ocl.hpp>
#include <Eigen/Dense>



typedef cv::Point3_<double> Pixel;



void loadImages(std::string *path, std::string *seriesName,std::vector<cv::Mat> *images, int numberOfImages)
{
    for (int i = 0; i < numberOfImages; i++)
    {
        // pushback images
        images->push_back(cv::Mat(cv::imread(*path + *seriesName + std::to_string(i + 1) + ".png",cv::IMREAD_GRAYSCALE)));
    }
    
}

void writeImages(std::vector<cv::Mat> *images,std::string *path, std::string *seriesName)
{
    for (int i = 0; i < images->size(); i++)
    {
        // pushback images
        cv::imwrite(*path + *seriesName + std::to_string(i + 1) + ".png", images->at(i));
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
        cv::UMat canny_input, canny_output;
        std::vector<cv::Vec4i> hierarchy;

        images->at(i).copyTo(canny_input);

        cv::Canny(canny_input, canny_output, threshold, threshold*2);


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


void read_parameters(std::string file_path,std::vector<std::vector<std::string>> &parameters)
{ 
    std::ifstream file(file_path); 

    if(file.is_open())
    { 
        std::string text; 
    
        while(std::getline(file,text))
        { 
            std::string element;
            std::stringstream p_line(text); 
            parameters.push_back({});
            while(std::getline(p_line, element, ' '))
            {
                parameters[parameters.size()-1].push_back(element);
            }
        }
    } 
    file.close(); 
}

void get_pmatrix(std::vector<std::vector<std::string>> parameters, std::vector<cv::Mat> &projections)
{ 
    cv::Mat K(3,3, cv::DataType<double>::type);
    cv::Mat R(3,3, cv::DataType<double>::type);
    cv::Mat t(3,1, cv::DataType<double>::type);
    cv::Mat E(3,4, cv::DataType<double>::type); 
    for(int i = 0; i < parameters.size(); i++)
    { 
        int count = 1;
        for(int row = 0; row < 3; row++)
        { 
            for(int col = 0; col < 3; col++)
            {
                K.at<double>(row,col) = std::stod(parameters[i][count]); 
                count++; 
            }
        } 
        for(int row = 0; row < 3; row++) 
        { 
            for(int col = 0; col < 3; col++)
            {
                R.at<double>(row,col) = std::stod(parameters[i][count]); 
                count++; 
            }
        }
        for(int row = 0; row < 3; row++) 
        { 
            t.at<double>(row,0) = std::stod(parameters[i][count]); 
            count++; 
        }
        
        //std::cout << "\nR:\n" << R << '\n';
        //std::cout << "\nt:\n" << t << '\n';
        //std::cout << "\nK:\n" << K << '\n';
        //cv::Mat zeroMat = cv::Mat::zeros(3,1, cv::DataType<double>::type);
        //cv::Mat K_(3,4, cv::DataType<double>::type);
        //cv::Mat E_(4,4, cv::DataType<double>::type);
        //cv::hconcat(K,zeroMat, K_);
        cv::hconcat(R,t,E);
        //std::cout << "\nE:\n" << E << '\n';
        //zeroMat = cv::Mat::zeros(1,4, cv::DataType<double>::type);
        //zeroMat.at<double>(0,3) = 1;

        //cv::vconcat(E,zeroMat,E_);

        //std::cout << "\nE_:\n" << E_ << '\n';
        projections.push_back(K*E); 
    }
}



template<typename T>
std::vector<double> linspace(T start_in, T end_in, int num_in)
{

  std::vector<double> linspaced;

  double start = static_cast<double>(start_in);
  double end = static_cast<double>(end_in);
  double num = static_cast<double>(num_in);

  if (num == 0) { return linspaced; }
  if (num == 1) 
    {
      linspaced.push_back(start);
      return linspaced;
    }

  double delta = (end - start) / (num - 1);

  for(int i=0; i < num-1; ++i)
    {
      linspaced.push_back(start + delta * i);
    }
  linspaced.push_back(end); // I want to ensure that start and end
                            // are exactly the same as the input
  return linspaced;
}

cv::Mat voxels_number(std::vector<double> xlim, std::vector<double> ylim, std::vector<double> zlim, std::vector<double> voxel_size)
{
    cv::Mat v(1,3, cv::DataType<double>::type);
    v.at<double>(0) = std::abs(xlim[1]-xlim[0])/voxel_size[0];
    v.at<double>(1) = std::abs(ylim[1]-ylim[0])/voxel_size[1];
    v.at<double>(2) = std::abs(zlim[1]-zlim[0])/voxel_size[2];
    return v;
}




std::vector<std::array<double, 4>> init_voxels(std::vector<double> xlim,std::vector<double> ylim, std::vector<double> zlim, std::vector<double> voxel_size)
{
    cv::Mat v = voxels_number(xlim, ylim, zlim, voxel_size);

    cv::Mat v_act(1,3, cv::DataType<int>::type);
    v_act.at<int>(0) = (int)v.at<double>(0)+1;
    v_act.at<int>(1) = (int)v.at<double>(1)+1;
    v_act.at<int>(2) = (int)v.at<double>(2)+1;
    int total_number = v_act.at<int>(0)*v_act.at<int>(1)*v_act.at<int>(2); 

    //cv::Mat voxels(total_number, 4, cv::DataType<double>::type); 


    //get voxel bounds
    double startx  = xlim[0];
    double endx    = xlim[1];
    
    double starty  = ylim[0];
    double endy    = ylim[1];
    
    double startz  = zlim[0];
    double endz    = zlim[1];





    // make linspaces in X, Y, Z (devide range of axis in actual number of voxel steps)
    std::vector<double, std::allocator<double>> lin_x = linspace(startx, endx, v_act.at<int>(0));
    std::vector<double, std::allocator<double>> lin_y = linspace(starty, endy, v_act.at<int>(1));
    std::vector<double, std::allocator<double>> lin_z = linspace(startz, endz, v_act.at<int>(2));


    std::vector<std::array<double, 4>> voxel;


    for(int i = 0; i < v_act.at<int>(2); i++)
    {
        for (int j = 0; j < v_act.at<int>(0); j++)
        {
            for (int l = 0; l < v_act.at<int>(1); l++)
            {
                //std::cout <<  lin_x[l] << " " << lin_y[j] << " " << lin_z[i] << std::endl;
                voxel.push_back({lin_x[j],lin_y[l], lin_z[i], 1});
            }
            
        }

    }

    return voxel;
}

std::string type2str(int type) {
  std::string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}






std::vector<std::array<double, 4>> projectImagesOnvoxels(std::vector<std::array<double, 4>> &voxels, 
                                                        std::vector<std::vector<std::string>> &parameters,
                                                        std::vector<cv::Mat> *images)

{


   
    
    auto start = std::chrono::high_resolution_clock::now();

    

    //std::cout << "Making objects 3d\n";
    //cv::UMat object_points_3D(voxels.size(), 4,cv::DataType<double>::type);
    //cv::Mat object_points_3D_temp(voxels.size(), 4,cv::DataType<double>::type);
    
    cv::Mat object_points_3D(voxels.size(), 4,cv::DataType<double>::type);
    for (int i = 0; i < voxels.size(); i++)
    {

        cv::Vec4d* it = object_points_3D.ptr<cv::Vec4d>(i);
        
        *it = {voxels[i][0], voxels[i][1], voxels[i][2], 1};
        
        voxels[i][3] = 0;
    }
     //std::cout << "done...\n ";
    
    //std::cout << "\nobjects_3d\n" << object_points_3D << '\n';
    /*
    object_points_3D_temp.copyTo(object_points_3D);

    object_points_3D_temp.~Mat();*/

    auto point3Ddone  = std::chrono::high_resolution_clock::now();
    
    object_points_3D = object_points_3D.t();
    
    auto transposeDone  = std::chrono::high_resolution_clock::now();

    //CAMERA PARAMETERS
    std::vector<cv::Mat> projection;
    

        
    get_pmatrix(parameters, projection);
   
    
    
    for (int i = 0; i < projection.size(); i++)
    {
        cv::Mat points2Dint, dummy;
        cv::Mat* img = &images->at(i);


        //std::cout << "\nPROJ\n" << projection[i] << '\n';
        //cv::imshow("cur im", images->at(i));
        //cv::waitKey(0);
        //std::string ty =  type2str( images->at(i).type() );
        //printf("Matrix: %s %dx%d \n", ty.c_str(), images->at(i).cols, images->at(i).rows );
        
        //std::cout << "copying\n";
        //cv::UMat current_projection;
        //projection[i].copyTo(current_projection);
        
        //std::cout << "Multiplying\n";
        /*
        try {
            cv:gemm(projection[i], object_points_3D, 1.0, dummy, 0.0, points2Dint);
        } 
        catch (cv::Exception& e) {
            std::cerr << e.what();
        
        }*/
        
        //PROJECTION TO THE IMAGE PLANE
        
        cv::gemm(projection[i], object_points_3D, 1.0, dummy, 0, points2Dint);

        //points2Dint = projection[i] * object_points_3D;

        //std::cout << "2D points is continous: " << points2Dint.isContinuous() << '\n';

        //std::cout << "2d points" << points2Dint.t();
        //std::cout << points2Dint.rows << "  " << points2Dint.cols  <<'\n';
       
        
        int img_lim_x = images->at(i).cols;
        int img_lim_y = images->at(i).rows;


        //The efficient way but goes into seg. fault
        
        //points2Dint.getStdAllocator()

        //std::cout << "2D points is continous: " << points2Dint.isContinuous() << '\n';
     
        

        for (int l = 0; l < points2Dint.rows; l++)
        {
            cv::Point3d* it = points2Dint.ptr<cv::Point3d>(l);
            
            //approximate of a/b but faster
            int x = it->x * (1.0d / it->z);
            int y = it->y * (1.0d / it->z);
            
            //for all non zero and in-bounds cords check if pixel is true (255) and increment the vote for the given voxel if yes
            
            voxels[l][3] = (x && y && x < img_lim_x && y < img_lim_y && (int)img->at<uchar>(y,x)) ? ++voxels[l][3]: voxels[l][3];


            /////// equivalent to but slightly faster than:
            /*
            //if the point is out of bounds replace value with 0
            if (x < 0 || img_lim_x < x) 
            {
                x = 0;
            }

            if (y < 0 || img_lim_y < y) 
            {
                y = 0;
            }
           
            if (x && y && (int)images->at(i).at<uchar>(y,x))
            {                
                voxels[l][3] += 1;
            }*/
                
        }

    }
    
    auto point3D_duration = std::chrono::duration_cast<std::chrono::microseconds>( point3Ddone - start );
    auto transpose_duration = std::chrono::duration_cast<std::chrono::microseconds>( transposeDone - point3Ddone );
    auto point2D_duration = std::chrono::duration_cast<std::chrono::microseconds>( std::chrono::high_resolution_clock::now() - transposeDone );
    std::cout << "Time taken making 3d points  : " << point3D_duration.count() /1000000.0d << " seconds" << std::endl;
    std::cout << "Time taken transposing : " << transpose_duration.count() /1000000.0d << " seconds" << std::endl;
    std::cout << "Time taken making 2d points : " << point2D_duration.count() /1000000.0d << " seconds" << std::endl;
    //std::cout << images->at(0);
    return voxels;
}



// std::vector< std::vector <std::vector < double, std::allocator<double> > > > 

void voxelListToFile(std::vector<std::array<double, 4>> voxelList)
{
    /*
    std::vector< std::vector <std::vector < double, std::allocator<double> > > > voxel3D;
    std::vector<double> xlim = {voxelList[0][0], voxelList[-1][0]};
    std::vector<double> ylim = {voxelList[0][1], voxelList[-1][1]};
    std::vector<double> zlim = {voxelList[0][2], voxelList[-1][2]};
    cv::Mat v = voxels_number(xlim, ylim, zlim, voxel_size);
0.07, -0.04
    for (int l = 0; l < voxelList.size(); l++)
    {
        
        
    }
        */
    
    std::ofstream voxelFile;
    voxelFile.open("voxelFile.csv");
    voxelFile << "x,y,z,score\n";
    for (unsigned i = 0; i < voxelList.size(); i++)
    {
       
        voxelFile << voxelList[i][0] << ',';
        voxelFile << voxelList[i][1] << ',';
        voxelFile << voxelList[i][2] << ',';
        voxelFile << voxelList[i][3] << '\n';
    }

}



int main() 
{

    
    

    std::vector<cv::Mat> images;

    std::vector<cv::Mat> contours;


    std::string path = "../data/DinoSR/";
    std::string series = "dinoSR";
    std::string output = "bin_images";
    int numberOfimages = 16; 
    std::vector<cv::Mat> projections;
    std::vector<std::vector<std::string>> parameters;

    auto start = std::chrono::high_resolution_clock::now();

    read_parameters(path + "dinoSR_par.txt", parameters);
    get_pmatrix(parameters, projections); 
    
    loadImages(&path, &series, &images, numberOfimages);

    removeBackground(&images,30, 256);

    applyMorphology(&images, cv::Size (5,5), cv::Size (13,13));

    auto img_done_time = std::chrono::high_resolution_clock::now();



    std::vector<double> voxel_size = {0.001, 0.001, 0.001};

   
    std::vector<double> xlim = {-0.07, 0.02};
    std::vector<double> ylim = {-0.02, 0.07};
    std::vector<double> zlim = {-0.07, 0.02};
    
    std::vector<std::array<double, 4>>  voxels = init_voxels(xlim,ylim,zlim, voxel_size);
    

    voxels = projectImagesOnvoxels(voxels, parameters, &images);

    auto voxel_done_time = std::chrono::high_resolution_clock::now();

    auto img_duration = std::chrono::duration_cast<std::chrono::microseconds>(img_done_time - start);
    auto voxel_duration = std::chrono::duration_cast<std::chrono::microseconds>( voxel_done_time - img_done_time );

    std::cout << "Time taken by loading images  : " << img_duration.count() /1000000.0 << " seconds" << std::endl;
    std::cout << "Time taken by voxel generation: " << voxel_duration.count() /1000000.0 << " seconds" << std::endl;
    
    voxelListToFile(voxels);

    
    /*
    for (auto voxel : voxels)
    {
        for (int i = 0; i < 4; i++)
        {
            std::cout << voxel[i] << " ";
        }
        std::cout<< '\n';
    }*/
    

    //showImages(&images);



    //removeBackground(&images,30, 256);
    
    //showImages(&images);

    //applyMorphology(&images, cv::Size (5,5), cv::Size (13,13));

    //showImages(&images);

    //findContours(&images, &contours, 10);
    
    //showImages(&images);

    //cv::imshow("dino UwU", img);

    writeImages(&images, &path, &output);

    //cv::waitKey(1);

    return 1;
}
