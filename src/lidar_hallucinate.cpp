#include <string>
#include <vector>

#include "ros/ros.h"
#include "tf2_ros/transform_listener.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.h"
#include "sensor_msgs/LaserScan.h"
#include "geometry_msgs/PointStamped.h"
#include "geometry_msgs/TransformStamped.h"
#include "lidar_hallucination/VirtualCircle.h"
#include "lidar_hallucination/VirtualCircles.h"

class LidarHallucination{
public:
    LidarHallucination() : tfListener_(tfBuffer_)
    {
        std::string inputScanTopic;
        std::string outputScanTopic;
        std::string addCircleTopic;

        // Get parameters from parameter server
        ros::param::get("amcl/global_frame_id", this->mapFrame_);
        ros::param::param<std::string>("add_circle_topic", addCircleTopic, "add_circle");
        this->nh_.param<std::string>("input_scan_topic", inputScanTopic, "scan_filtered");
        this->nh_.param<std::string>("output_scan_topic", outputScanTopic, "scan_hallucinated");

        // Define publisher and subscriber
        this->pubScan_  = this->nh_.advertise<sensor_msgs::LaserScan>(outputScanTopic, 100);
        this->subBody_  = this->nh_.subscribe(addCircleTopic,
                                              100,
                                              &LidarHallucination::bodyCallback,
                                              this);
        this->subLidar_ = this->nh_.subscribe(inputScanTopic,
                                              100,
                                              &LidarHallucination::lidarCallback,
                                              this);
    }

    void bodyCallback(const lidar_hallucination::VirtualCircles::ConstPtr& msg){
        // Read and store new virtual circles from the message
        for (int i=0; i < msg->circles.size(); i++){
            // Convert relative position to global position
            this->circles_.push_back(msg->circles[i]);
        }
        ROS_INFO_STREAM("New object added!");
    }

    void lidarCallback(const sensor_msgs::LaserScan::ConstPtr& msg){
        scan_ = std::move(*msg);

        // Remove all expired objects from the list
        ROS_INFO_STREAM("Number of circles: "<<this->circles_.size());
        this->discardExpired();
        
        // If no valid object exists, skip hallucination process 
        if (this->circles_.empty()){
            this->pubScan_.publish( scan_ );
            return;
        }
        // Prepare an array of virtual scan with inf values
        const int num_scan = scan_.ranges.size();
        std::vector<float> virtualScan;
        virtualScan.assign(num_scan, std::numeric_limits<float>::infinity());

        // Get transformation between global map frame to LiDAR sensor frame
        // Speed: 7.6 +- 5.5 us, maximum 50 us
        while(ros::ok()){
            try{
                this->tfStamped_ = this->tfBuffer_.lookupTransform(scan_.header.frame_id, this->mapFrame_, ros::Time(0));
                break;
            }catch(tf2::TransformException ex){
                ROS_WARN("%s", ex.what());
            }
        }

        // Create outline of all virtual circles
        for(auto circle : this->circles_){
            geometry_msgs::PointStamped global_point, transformed_point;
            global_point.header = scan_.header;
            global_point.point.x = circle.x;
            global_point.point.y = circle.y;
            tf2::doTransform( global_point, transformed_point, this->tfStamped_);

            // Check if circle is visible to the sensor
            float x = transformed_point.point.x;
            float y = transformed_point.point.y;
            float dist = std::hypotf(x,y);
            float th  = std::atan2(y,x);
            float dth = std::asin(circle.radius/dist);

            float th_min = std::max(th-dth, scan_.angle_min);
            float th_max = std::min(th+dth, scan_.angle_max);

            if (th_max > th_min){ // true if object is visible
                ROS_INFO_STREAM("Visible circle found!");
                int index = (th_max - scan_.angle_min) / scan_.angle_increment;
                float theta = index * scan_.angle_increment + scan_.angle_min;
                float c = std::pow(dist, 2) - std::pow(circle.radius, 2);
                while (theta > th_min){
                    float b = x*std::cos(theta) + y*std::sin(theta);
                    virtualScan[index] = std::min(virtualScan[index], b - std::sqrt(b*b - c) );
                    index--;
                    theta = theta - scan_.angle_increment;
                }
            }
        }

        // Only leave outline between actual and virtual scan
        for(int i=0; i<num_scan; i++){
            virtualScan[i] = std::min(scan_.ranges[i], virtualScan[i]);
        }

        // Publish hallucinated LiDAR scan
        scan_.ranges = virtualScan;
        this->pubScan_.publish( scan_ );
    }

    void show(){
        // Debug ONLY
        for (auto circle : this->circles_){
            ROS_INFO_STREAM("id: "<<circle.radius<<" life: "<<circle.life.toSec() );
        }
    }
    
private:
    // parameters
    ros::NodeHandle nh_;
    std::string mapFrame_;
    ros::Publisher pubScan_;
    ros::Subscriber subBody_;
    ros::Subscriber subLidar_;
    tf2_ros::Buffer tfBuffer_;
    tf2_ros::TransformListener tfListener_;

    sensor_msgs::LaserScan scan_;
    geometry_msgs::TransformStamped tfStamped_;
    std::vector<lidar_hallucination::VirtualCircle> circles_;

    // functions
    void discardExpired()
    {
        /*
        The pre-processing function which remove all the expired objects from the list.
        This entire process took ~2.5 us. At most 20 us with 100 objects.
        */
        std::vector<int> expired_idx;

        // Find index of expired virtual circles
        int idx = 0;
        ros::Time time = ros::Time::now();
        for (auto circle : this->circles_){
            if (time > circle.life){
                expired_idx.push_back(idx);
            }
            idx++;
        }
        // Remove expired objects from the list
        std::vector<int>::reverse_iterator rit;
        for (rit = expired_idx.rbegin(); rit != expired_idx.rend(); rit++){
            this->circles_.erase( this->circles_.begin() + *rit);
        }
    }
};

int main(int _argc, char** _argv)
{
    // Define ros and its handler
    ros::init(_argc, _argv, "lidar_hallucinator");
    LidarHallucination lh; // Construct class

    float HZ = 10.0;
    ros::Rate loop_rate(HZ); // debug

    // Run until interupted
    while(ros::ok()){
        // lh.show();
        ros::spinOnce();
        loop_rate.sleep();
    }

    return 0;
}