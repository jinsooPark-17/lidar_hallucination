#include <gazebo/transport/transport.hh>
#include <gazebo/msgs/msgs.hh>
#include <gazebo/gazebo_client.hh>
#include <ros/ros.h>
#include <std_msgs/Bool.h>

// Define global variable used in gazebo message callback
const std::string DELIMITER = "::";
std::vector<std::string> TARGETS;
std::string FLOOR_NAME = "plane1_model";
ros::Publisher pub_collision;

void contactCB(ConstContactsPtr &_msg){
    // Read Gazebo contact message
    for(int i=0; i<_msg->contact_size(); ++i){
        // Get entity name
        std::string entity_name = _msg->contact(i).collision1();
        entity_name = entity_name.substr(0, entity_name.find(DELIMITER));

        if(std::find(TARGETS.begin(), TARGETS.end(), entity_name) != TARGETS.end()){
            // If entity is target, check the name of contact object
            std::string contact = _msg->contact(i).collision2();
            contact = contact.substr(0, contact.find(DELIMITER));

            // If contact entity is not floor, publish collision alarm
            if(contact != FLOOR_NAME){
                std_msgs::Bool is_collide;
                is_collide.data = true;
                pub_collision.publish(is_collide);
                return; // Skip rest
            }
            else{ // Debug only
                ROS_INFO_STREAM(entity_name+":"+contact+" observed");
            }
        }
    }
}

// Main function
int main(int _argc, char** _argv){
    // Configure ROS node
    ros::init(_argc, _argv, "Collision_observer");
    ros::NodeHandle nh;
    pub_collision = nh.advertise<std_msgs::Bool>("/is_collide", 1000);
    // Configure Gazebo node
    gazebo::client::setup(_argc, _argv);
    gazebo::transport::NodePtr gznode(new gazebo::transport::Node());
    gznode->Init();
    // Define Gazebo contact message subscriber
    gazebo::transport::SubscriberPtr sub_contact = gznode->Subscribe("/gazebo/default/physics/contacts", contactCB);

    // Get name of target entities from argument
    for(int i=1; i<_argc; ++i){
        TARGETS.push_back(_argv[i]);
    }

    // Get parameters from ROS parameter server
    float hz;
    nh.param<float>("frequency", hz, 5.0);
    ros::Rate rate(hz);

    // Observe collision until system is down
    while(ros::ok()){
        ros::spinOnce();
        rate.sleep();
    }

    // Clear gazebo node
    gazebo::client::shutdown();
}