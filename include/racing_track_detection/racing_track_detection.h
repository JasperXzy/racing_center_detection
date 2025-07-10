#ifndef RACING_TRACK_DETECTION_H_
#define RACING_TRACK_DETECTION_H_

#include <opencv2/opencv.hpp>

#include "rclcpp/rclcpp.hpp"
#include "dnn_node/dnn_node.h"
#include "dnn_node/dnn_node_data.h"
#include "hbm_img_msgs/msg/hbm_msg1080_p.hpp"
#include "std_msgs/msg/int16_multi_array.hpp"
#include "ai_msgs/msg/perception_targets.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "std_msgs/msg/int32.hpp"
#include "geometry_msgs/msg/point32.hpp"

using rclcpp::NodeOptions;

using hobot::dnn_node::DNNInput;
using hobot::dnn_node::DnnNode;
using hobot::dnn_node::DnnNodeOutput;
using hobot::dnn_node::ModelTaskType;
using hobot::dnn_node::DNNTensor;

class LineCoordinateResult{
 public:
  float x;
  float y;
  float confidence; 
  void Reset() {x = -1.0; y = -1.0; confidence = 0.0;} // 初始化置信度
};

class LineCoordinateParser{
 public:
  LineCoordinateParser() {}
  ~LineCoordinateParser() {}
  int32_t Parse(
      std::shared_ptr<LineCoordinateResult>& output,
      std::shared_ptr<DNNTensor>& output_tensor) ; 
};

class TrackDetectionNode : public DnnNode {
 public:
  TrackDetectionNode(const std::string& node_name,
                        const NodeOptions &options = NodeOptions());
  ~TrackDetectionNode() override;

 protected:
  int SetNodePara() override;
  int PostProcess(const std::shared_ptr<DnnNodeOutput> &outputs) override;

 private:
  // 巡线状态变量
  bool enable_lane_following_{true};  // 默认为开启巡线功能
  // 置信度阈值参数
  double confidence_threshold_ = 0.0; // 默认值设为0，表示总是发布，在参数中设置
  
  int Predict(std::vector<std::shared_ptr<DNNInput>> &dnn_inputs,
              const std::shared_ptr<DnnNodeOutput> &output,
              const std::shared_ptr<std::vector<hbDNNRoi>> rois);
  void subscription_callback(
    const hbm_img_msgs::msg::HbmMsg1080P::SharedPtr msg);
  bool GetParams(); 
  bool AssignParams(const std::vector<rclcpp::Parameter> & parameters);
  ModelTaskType model_task_type_ = ModelTaskType::ModelInferType;

  //订阅/sign4return话题的订阅器
  rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr
    sign_subscriber_;
  // 处理/sign4return消息的回调函数
  void sign_callback(const std_msgs::msg::Int32::SharedPtr msg);

  rclcpp::Subscription<hbm_img_msgs::msg::HbmMsg1080P>::SharedPtr
    subscriber_hbmem_ = nullptr;
  rclcpp::Publisher<ai_msgs::msg::PerceptionTargets>::SharedPtr publisher_ =
      nullptr;
  cv::Mat image_bgr_; 
  std::string model_path_ = "config/race_track_detection.bin";
  std::string sub_img_topic_ = "/hbmem_img";
};

#endif  // RACING_TRACK_DETECTION_H_
