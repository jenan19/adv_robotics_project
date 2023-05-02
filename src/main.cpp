#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 200

#include <vector>
#include <iostream>
#include <string>
#include <chrono>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <pcl/common/common.h>
#include <pcl/common/angles.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d.h>
#include <pcl/surface/gp3.h>
#include <pcl/io/vtk_io.h>
//#include <open3d/Open3D.h>      //Den her laver noget mærkeligt 


//#include <opencv2/core/ocl.hpp>
#include <Eigen/Dense>



//typedef cv::Point3_<double> Pixel;



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
                //std::cout << element;
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
                //std::cout << parameters[i][count] << " ";
                K.at<double>(row,col) = std::stod(parameters[i][count]);
                count++; 
            }
        } 
        //std::cout << "\n";
        for(int row = 0; row < 3; row++) 
        { 
            for(int col = 0; col < 3; col++)
            {
                R.at<double>(row,col) = std::stod(parameters[i][count]);
                //std::cout << parameters[i][count] << " ";
                count++; 
            }
        }
        //std::cout << "\n";
        for(int row = 0; row < 3; row++) 
        { 
            t.at<double>(row,0) = std::stod(parameters[i][count]);
            //std::cout << parameters[i][count] << " ";
            count++; 
        }
        //std::cout << "\n";
        

        cv::Mat z90deg = (cv::Mat_<double>(3,3) << 0, -1, 0, 1, 0, 0, 0, 0, 1);

        R = R * z90deg;

        cv::hconcat(R.t(),(-R.t()*t),E);

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

                voxel.push_back({lin_x[j],lin_y[l], lin_z[i], 0});
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


void voxelListToFile(std::vector<std::array<double, 4>> voxelList)
{

    
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


void updateVoxel(cv::Mat image, cv::Mat projection, std::vector<std::array<double, 4>> &voxels)
{
    //GENERATE LIST OF HOMOGENOUS 3D POINTS FROM VOXELS (the score is ignored and replaced with 1)
    cv::Mat object_points_3D(voxels.size(), 4,cv::DataType<double>::type);
    for (int i = 0; i < voxels.size(); i++)
    {

        cv::Vec4d* it = object_points_3D.ptr<cv::Vec4d>(i);
        
        *it = {voxels[i][0], voxels[i][1], voxels[i][2], 1};
        //std::cout << *it << '\n';
    }




    cv::Mat projected2Dpoints, dummy;


    //PROJECTION TO THE IMAGE PLANE
    // Note: its faster to save the transpose and then pass it to GEMM than just pasing the transposed points idfk...
    object_points_3D = object_points_3D.t();
    cv::gemm(projection, object_points_3D, 1.0, dummy, 0, projected2Dpoints);

    cv::Mat img = image.clone();
    
    //NORMALIZE POINTS
    int img_lim_x = img.cols;
    int img_lim_y = img.rows;
    cv::Mat normalized2Dpoints;
    projected2Dpoints = projected2Dpoints.t();
    cv::convertPointsFromHomogeneous(projected2Dpoints, normalized2Dpoints);
 
    //CHECK IF COORDINATES ARE VALID AND THAT OBJECT IS SEEN IN PIXEL
    for (int i = 0; i < normalized2Dpoints.rows; i++)
    {
        //std::cout << *normalized2Dpoints.ptr<cv::Point2d>(i) << '\n';
        int x = normalized2Dpoints.ptr<cv::Point2d>(i)->x;
        int y = normalized2Dpoints.ptr<cv::Point2d>(i)->y;
        //std:: cout <<  << " " << normalized2Dpoints.cols << '\n';

        // fast if statement for better handeling of context switch (++voxel is asumed)
        voxels[i][3] = (0 < x && 0 < y && x < img_lim_x && y < img_lim_y && (int)img.at<uchar>(y,x)) ? ++voxels[i][3]: voxels[i][3];
    }
    std::cout << "done... " << '\n';

}


bool compareScores(const std::array<double, 4> &pointA, const std::array<double, 4> &pointB)
{
    return pointA[3] > pointB[3];
}

pcl::PointCloud<pcl::PointXYZ> updatePointCloud(std::vector<std::array<double, 4>> voxels)
{
    pcl::PointCloud<pcl::PointXYZ> cloud;
    

    //sort from highest score to min
    std::sort(voxels.begin(), voxels.end(), compareScores);
    // Fill in the cloud data
    // cloud.width    = voxels.size();
    // cloud.height   = 1;
    // cloud.is_dense = false;
    // cloud.resize (cloud.width * cloud.height);
    
    double maxv = voxels[0][3];

    double iso_value = maxv - (int)((maxv / 100.0) * 20);
    std::cerr << "MAX " << maxv << " iso " << iso_value << std::endl;
    //int i = 0;
    for (int i = 0; i <voxels.size(); i++)
    {
        std::array<double, 4> voxel = voxels[i];
        
        if(voxel[3] >= iso_value)
        {
            pcl::PointXYZ pt(voxel[0], voxel[1], voxel[2]);

            cloud.push_back(pt);
            
        }
        else
        {
            break;
        }
    }

    pcl::io::savePCDFileASCII ("test_pcd.pcd", cloud);
    std::cerr << "Saved " << cloud.size () << " data points to test_pcd.pcd." << std::endl;
    
    return cloud;
}






pcl::visualization::PCLVisualizer::Ptr simpleVis (pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud)
{
  // --------------------------------------------
  // -----Open 3D viewer and add point cloud-----
  // --------------------------------------------
  pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  viewer->addPointCloud<pcl::PointXYZ> (cloud, "sample cloud");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
  viewer->addCoordinateSystem (1.0);
  viewer->initCameraParameters ();
  return (viewer);
}



void fastTriangulation(pcl::PointCloud<pcl::PointXYZ> cloud_)
{
    std::cout << "NORMAL ESTIMATION... " << '\n';

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud  (new pcl::PointCloud<pcl::PointXYZ>(cloud_));
    // Normal estimation*
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> n;
    pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud (cloud);
    n.setInputCloud (cloud);
    n.setSearchMethod (tree);
    n.setKSearch (20);
    n.compute (*normals);
    //* normals should not contain the point normals + surface curvatures
    std::cout << "Concatenate... " << '\n';
    // Concatenate the XYZ and normal fields*
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals (new pcl::PointCloud<pcl::PointNormal>);
    pcl::concatenateFields (*cloud, *normals, *cloud_with_normals);
    //* cloud_with_normals = cloud + normals

    std::cout << "Create search tree... " << '\n';
    // Create search tree*
    pcl::search::KdTree<pcl::PointNormal>::Ptr tree2 (new pcl::search::KdTree<pcl::PointNormal>);
    tree2->setInputCloud (cloud_with_normals);

    std::cout << "init objects... " << '\n';
    // Initialize objects
    pcl::GreedyProjectionTriangulation<pcl::PointNormal> gp3;
    pcl::PolygonMesh triangles;

    // Set the maximum distance between connected points (maximum edge length)
    gp3.setSearchRadius (0.025);

    // Set typical values for the parameters
    gp3.setMu (2.5);
    gp3.setMaximumNearestNeighbors (150);
    gp3.setMaximumSurfaceAngle(M_PI/4); // 45 degrees
    gp3.setMinimumAngle(M_PI/18); // 10 degrees
    gp3.setMaximumAngle(2*M_PI/3); // 120 degrees
    gp3.setNormalConsistency(false);

    std::cout << "Search... " << '\n';
    // Get result
    gp3.setInputCloud (cloud_with_normals);
    gp3.setSearchMethod (tree2);
    gp3.reconstruct (triangles);

    // Additional vertex information
    std::vector<int> parts = gp3.getPartIDs();
    std::vector<int> states = gp3.getPointStates();


    std::cout << "Save... " << '\n';
    pcl::io::saveVTKFile("mesh.vtk", triangles); 
    std::cout << "Done... " << '\n';
}




int main() 
{


    
    std::vector<cv::Mat> images;

    std::vector<cv::Mat> contours;


    std::string path = "../data/kiwi/";
    std::string series = "kiwi";
    std::string output = "bin_images";
    int numberOfimages = 37; 
    std::vector<cv::Mat> projections;
    std::vector<std::vector<std::string>> parameters;

    auto start = std::chrono::high_resolution_clock::now();

    std::cout << "read params... " << '\n';
    read_parameters(path + series + "_par.txt", parameters);
    std::cout << "get pmatrix... " << '\n';
    get_pmatrix(parameters, projections); 
    

    
    std::cout << "loading imgs... " << '\n';
    loadImages(&path, &series, &images, numberOfimages);
    std::cout << "removing background... " << '\n'; 
    //showImages(&images); 
    removeBackground(&images,0, 200);
    //showImages(&images);
    std::cout << "morphing.. " << '\n';
    applyMorphology(&images, cv::Size (5,5), cv::Size (13,13));

    writeImages(&images, &path, &output);

    //showImages(&images);

    auto img_done_time = std::chrono::high_resolution_clock::now();



    std::vector<double> voxel_size = {0.001, 0.001, 0.001};

   
    std::vector<double> xlim = {-0.2, 0.2};
    std::vector<double> ylim = {-0.2, 0.2};
    std::vector<double> zlim = {-0.5, 0.5};
    
    std::vector<std::array<double, 4>>  voxels = init_voxels(xlim,ylim,zlim, voxel_size);

    //pcl::PointCloud<pcl::PointXYZ>::Ptr vizualizer (new pcl::PointCloud<pcl::PointXYZ>);
    std::cout << "running.. " << '\n';
    pcl::visualization::PCLVisualizer::Ptr viewer;
    pcl::PointCloud<pcl::PointXYZ> cloud;
    for (int i = 0; i < numberOfimages; i++)
    {
        
        updateVoxel(images[i],projections[i],voxels);

        
        
        //vizualizer.reset( new pcl::PointCloud<pcl::PointXYZ>(cloud));

        //viewer = simpleVis(vizualizer);

        //viewer->spinOnce(1);

    }

    cloud = updatePointCloud(voxels);
    //fastTriangulation(cloud);
    //voxels = projectImagesOnvoxels(voxels, parameters, &images);

    auto voxel_done_time = std::chrono::high_resolution_clock::now();

    auto img_duration = std::chrono::duration_cast<std::chrono::microseconds>(img_done_time - start);
    auto voxel_duration = std::chrono::duration_cast<std::chrono::microseconds>( voxel_done_time - img_done_time );

    std::cout << "Time taken by loading images  : " << img_duration.count() /1000000.0 << " seconds" << std::endl;
    std::cout << "Time taken by voxel generation: " << voxel_duration.count() /1000000.0 << " seconds" << std::endl;
    
    std::cout << "Time taken per image (average): " << voxel_duration.count() /1000000.0 / 16.0 << " seconds" << std::endl;
    voxelListToFile(voxels);

/*
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudPTR(new pcl::PointCloud<pcl::PointXYZ>::Ptr );
    *cloudPTR = updatePointCloud(voxels);
    viewer = simpleVis(cloudPTR);
    while (!viewer->wasStopped ())
    {
        viewer->spinOnce (100);

    }
    
*/
    //cv::waitKey(1);

    return 1;
}
