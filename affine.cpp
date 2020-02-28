#ifndef AFFINE_AND_ICP_H
#define AFFINE_AND_ICP_H

#endif // AFFINE_AND_ICP_H

#include <opencv2/core.hpp>
#include <opencv2/surface_matching/ppf_helpers.hpp>
#include <opencv2/surface_matching/icp.hpp>
#include <iostream>
#include <fstream>
#include <opencv2/calib3d.hpp>
#define _USE_MATH_DEFINES
#include <cmath>
#include <algorithm>
#include <chrono>
#include <random>
#include <vector>
#include "pointmatcher/PointMatcher.h"
#include "boost/filesystem.hpp"
#include <cassert>
#include "common_functions.h"

using namespace std;
using namespace cv;
using namespace PointMatcherSupport;

typedef PointMatcher<float> PM;

PM::TransformationParameters parseTranslation(string& translation, const int cloudDimension);
PM::TransformationParameters parseRotation(string& rotation, const int cloudDimension);

/**
 *  Generate new point cloud of given 'n' points and add noise to 'm' points
 *  @param [in] n - no.of points to be generated
 *  @param [in] pointCloud - The point cloud to extract points from
 *  @param [in] transformedCloud - The point cloud to extract points from
 *  @param [in] nPoints1 - The point cloud to assign points to
 *  @param [in] nPoints2 - The point cloud to assign points to
 *  @param [in] m - no.of points to add noise
 *  @param [in] noise - Add noise or no - Boolean value
 *  @return void
*/
void randPC(int n, Mat pointCloud, Mat transformedCloud, Mat nPoints1, Mat nPoints2, int m, bool noise, int random_points_n[])
{

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    vector<int> rn;
    for(int i = 0; i < pointCloud.rows; i++)
    {
        rn.push_back(i);
    }
    std::shuffle(rn.begin(), rn.end(), std::default_random_engine(seed));
    for(int i = 0; i < n; i++)
    {
        random_points_n[i] = rn[i];
    }

    std::rand();
    for (int i =0; i < n; i++)//assign those points to new Point clouds
    {
        nPoints1.at<float>(i,0) = pointCloud.at<float>(random_points_n[i],0);
        nPoints1.at<float>(i,1) = pointCloud.at<float>(random_points_n[i],1);
        nPoints1.at<float>(i,2) = pointCloud.at<float>(random_points_n[i],2);

        nPoints2.at<float>(i,0) = transformedCloud.at<float>(random_points_n[i],0);
        nPoints2.at<float>(i,1) = transformedCloud.at<float>(random_points_n[i],1);
        nPoints2.at<float>(i,2) = transformedCloud.at<float>(random_points_n[i],2);
    }
    if(noise) //If the user wants noise to be added
    {
        double noise_scale = 1;
        Mat nPoints2_noise = ppf_match_3d::addNoisePC(nPoints2, noise_scale);//add noise to the (transformed n points) point cloud
        cout << "Noise scale = " << noise_scale << endl;
        cout << "no.of points selected = " << n << endl;
        cout << "no.of Noise points = " << m << endl;
        int random_points_m[m]; // m random points in the range 0 to n
        vector<int> rm;
        for(int i = 0; i < n; i++)
        {
            rm.push_back(i);
        }
        std::shuffle(rm.begin(), rm.end(), std::default_random_engine(seed));
        for(int i = 0; i < m; i++)
        {
            random_points_m[i] = rm[i];
        }
        for (int i =0; i < m; i++)//assign noise points to nPoints2
        {
            nPoints2.at<float>(random_points_m[i],0) = nPoints2_noise.at<float>(random_points_m[i],0);
            nPoints2.at<float>(random_points_m[i],1) = nPoints2_noise.at<float>(random_points_m[i],1);
            nPoints2.at<float>(random_points_m[i],2) = nPoints2_noise.at<float>(random_points_m[i],2);
        }

    }
}

/**
 *  Functionalites include:
        - libpointmatcher
            - Estimates ICP transformation between reference and reding point clouds
            - Contains ICP chain
            - Can configure ICP configurations with YAML file and provide initial transformations
            - Save point clouds as .vtk files
 *  @param [in] affineArray - Transformation from Affine transformed PC to Sampled PC
 *  @return void
*/
void libPointMatcher(Mat affineArray)
{
    typedef PointMatcher<float> PM; //Contains objects and functions for ICP process
    typedef PM::DataPoints DP; //Represents a point cloud

    const DP reading(DP::load("/home/internship_computervision/nagendra/point_cloud_test/threshold/file.csv"));
    const DP reference(DP::load("/home/internship_computervision/nagendra/point_cloud_test/threshold/filetwo.csv"));
    const DP cv_icp(DP::load("/home/internship_computervision/nagendra/point_cloud_test/threshold/filethree.csv"));
    const DP libnPoints1(DP::load("/home/internship_computervision/nagendra/point_cloud_test/threshold/nPoints1.csv"));
    const DP libnPoints3(DP::load("/home/internship_computervision/nagendra/point_cloud_test/threshold/nPoints3.csv"));
    /*
     * reading = sampled point cloud
     * reference = Inverse affine transformed point cloud
     * libnPoints1 = n points from sampled point cloud
     * libnPoints3 = n points from Inverse affine Transformed point cloud
    **/

    PM::ICP icp; //ICP chain

    string configFile = "/home/internship_computervision/nagendra/point_cloud_test/threshold/icp_tutorial_cfg.yaml";
    ifstream ifs(configFile.c_str());
    icp.loadFromYaml(ifs); //To load ICP configuration from a YAML file
    string initTranslation("0,0,0"); //Initial transformations
    string initRotation("1,0,0;0,1,0;0,0,1");

    int cloudDimension = reference.getEuclideanDim();
    PM::TransformationParameters translation =
            parseTranslation(initTranslation, cloudDimension);
    PM::TransformationParameters rotation =
            parseRotation(initRotation, cloudDimension);
    PM::TransformationParameters initTransfo = translation*rotation;

    std::shared_ptr<PM::Transformation> rigidTrans;
    rigidTrans = PM::get().REG(Transformation).create("RigidTransformation");

    if (!rigidTrans->checkParameters(initTransfo)) {
        cerr << endl
             << "Initial transformation is not rigid, identiy will be used"
             << endl;
        initTransfo = PM::TransformationParameters::Identity(
                    cloudDimension+1,cloudDimension+1);
    }

    const DP initializedData = rigidTrans->compute(libnPoints1, initTransfo); //Applying initial transformations


    PM::TransformationParameters T = icp(libnPoints3, initializedData); //Estimate ICP translation

    DP data_out(reference);

    icp.transformations.apply(data_out, T); //Apply translation to Affine Transformed Point cloud

    cv::Mat icp_trans(4,4,CV_64F);
    for (int i =0; i < 4; i++)
    {
        for (int j =0; j < 4; j++)
        {
            icp_trans.at<double>(i,j) = T(i,j);
        }
    }
    cout << "\nPose from Affine Transformed PC and Sampled PC after libpointmatcher \"ICP\"\n" << endl << icp_trans << endl;
    cv::Mat temp_pose;
    cv::transpose(icp_trans, temp_pose);
    cout << "\nInput estimated libpointmatcher : \n" << (affineArray*temp_pose.t()).inv() << endl;

    //save the Point clouds to .vtk files.
    reading.save("/home/internship_computervision/nagendra/point_cloud_test/threshold/1sampled.vtk"); //Sampled Point Cloud
    reference.save("/home/internship_computervision/nagendra/point_cloud_test/threshold/2affine_transformed.vtk"); //Inverse affine PC
    cv_icp.save("/home/internship_computervision/nagendra/point_cloud_test/threshold/3icp_opencv.vtk"); //PC after opencv icp
    data_out.save("/home/internship_computervision/nagendra/point_cloud_test/threshold/4icp_lib.vtk"); //PC after libpointmatcher icp
}

/**
 *  Functionalites include:
        - Down sample the input point cloud
        - Select any random 10 points from the sampled point cloud
        - Transform the Down sampled point cloud with a predefined Pose
        - Noise
            - Add noise to any 3 points of the 10 points from the transformed point cloud
            - Estimate the Pose between the 10 ground points and the same points on the transformed point cloud along with 3 noise points
        - Estimate the Pose between the 10 ground points and the same points on the transformed point cloud
 *  @param [in] input_pointCloud - Input point cloud
 *  @return void
*/
void downSampledCloud(Mat input_pointCloud)
{
    Mat affineArray;
    Mat sampled_PC;
    int sampleStep = 6;
    sampled_PC = SamplePCUniform(input_pointCloud, sampleStep); //Down samples the input point cloud
    cout << sampled_PC.rows << " points out of " << input_pointCloud.rows << " points are in the Sampled point cloud." << endl;

    cv::Mat scaling = (cv::Mat_<double>(4,4) << 3.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 1.0);

    //pre-defined pose transformation
    Mat Pose = scaling * (cv::Mat_<double>(4,4) << -0.2603982, 0.5110604,  0.8191521, -5.0, 0.4629855,  -0.6784166, 0.5704343, 20.0, 0.8472527,  0.5277957,  -0.0599551, 10.0, 0.0, 0.0, 0.0, 1.0);

    cv::Vec3f viewpoint(0, 0, 0);
    Mat sampled_PC_normal;
    cv::ppf_match_3d::computeNormalsPC3d(sampled_PC, sampled_PC_normal, 100, false, viewpoint);
    Mat transformedCloud = transformCloud(Pose, sampled_PC_normal);

    int n = 50, m = 40;
    int random_points_n[n];// n = no.of ground points to select from sampled point cloud, m = no.of point to add noise to
    bool noise = true;
    Mat nPoints1(n, 3, CV_32F), nPoints2(n, 3, CV_32F), nPoints3(n, 3, CV_32F);
/*
 * nPoints1 = n Points from Sampled point cloud
 * nPoints2 = n Points from Transformed point cloud
 * nPoints3 = n points from Inverse affine Transformed point cloud
**/
    randPC(n, sampled_PC, transformedCloud, nPoints1, nPoints2, m, noise, random_points_n); //Generate new point cloud of given 'n' points and add noise to 'm' points

    Mat inversePose(4, 4, CV_64F);
    std::vector<uchar> inliers;
    int output;
    double ransacThreshold=0.01; double confidence=0.99999;

    output = estimateAffine3D(nPoints2, nPoints1, affineArray, inliers, ransacThreshold, confidence); // Estimate Pose transformation using estimateAffine3D()

    cv::vconcat(affineArray, cv::Mat::zeros(1,4,CV_64F), affineArray);
    affineArray.at<double>(3,3) = 1.0;

    cout << "\nPose from Sampled PC and Transformed PC after initial transformation\n" << Pose << endl;
    cout << "\nPose from Transformed PC and Sampled PC after estimateAffine3D()\n" << affineArray << endl;
    cout << "\nPose from Sampled PC to Transformed PC after estimateAffine3D()\n" << affineArray.inv() << endl;

    cv::Mat affineTransformedPC = transformCloud(affineArray, transformedCloud); // New transformed point cloud from the previously transformed point cloud by applying the inverse pose

    for (int i =0; i < n; i++)//assign those points to new Point clouds
    {
        nPoints3.at<float>(i,0) = affineTransformedPC.at<float>(random_points_n[i],0);
        nPoints3.at<float>(i,1) = affineTransformedPC.at<float>(random_points_n[i],1);
        nPoints3.at<float>(i,2) = affineTransformedPC.at<float>(random_points_n[i],2);
    }

    Matx44d finalICP_pose = poseICP(nPoints3, nPoints1); //Estimate pose using ICP
    cout << "\nPose from Affine Transformed PC and Sampled PC after OpenCV \"ICP\"\n" << finalICP_pose << endl;
    cv::Mat temp_pose;
    cv::transpose(finalICP_pose, temp_pose);
    cv::Mat temp_transformed = transformCloud(temp_pose.t(), affineTransformedPC);

    cout << "\nInput estimated OpenCV : \n" << (affineArray*temp_pose.t()).inv() << endl;
    toCSV(sampled_PC, affineTransformedPC, temp_transformed); // Pass the clouds to visualize
    toCSV2(nPoints1, nPoints3);

    libPointMatcher(affineArray);//ICP estimation using libpointmatcher library
}

