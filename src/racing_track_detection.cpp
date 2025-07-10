#include "racing_track_detection/racing_track_detection.h"

#include <fstream>
#include <string>
#include <cmath> // For std::exp (sigmoid inverse) or direct sigmoid computation

#include <opencv2/opencv.hpp>
#include "dnn_node/util/image_proc.h"
#include "hobot_cv/hobotcv_imgproc.h"
#include "rclcpp/qos.hpp"
void prepare_nv12_tensor_without_padding(const char *image_data,
                                         int image_height,
                                         int image_width,
                                         hbDNNTensor *tensor) {
  auto &properties = tensor->properties;
  properties.tensorType = HB_DNN_IMG_TYPE_NV12;
  properties.tensorLayout = HB_DNN_LAYOUT_NCHW;
  auto &valid_shape = properties.validShape;

  valid_shape.numDimensions = 4;
  valid_shape.dimensionSize[0] = 1;
  valid_shape.dimensionSize[1] = 3;
  valid_shape.dimensionSize[2] = image_height;
  valid_shape.dimensionSize[3] = image_width;

  auto &aligned_shape = properties.alignedShape;
  aligned_shape = valid_shape;

  int32_t image_length = image_height * image_width * 3 / 2;

  hbSysAllocCachedMem(&tensor->sysMem[0], image_length);
  memcpy(tensor->sysMem[0].virAddr, image_data, image_length);

  hbSysFlushMem(&(tensor->sysMem[0]), HB_SYS_MEM_CACHE_CLEAN);
}

void prepare_nv12_tensor_without_padding(int image_height,
                                         int image_width,
                                         hbDNNTensor *tensor) {
  auto &properties = tensor->properties;
  properties.tensorType = HB_DNN_IMG_TYPE_NV12;
  properties.tensorLayout = HB_DNN_LAYOUT_NCHW;

  auto &valid_shape = properties.validShape;
  valid_shape.numDimensions = 4;
  valid_shape.dimensionSize[0] = 1;
  valid_shape.dimensionSize[1] = 3;
  valid_shape.dimensionSize[2] = image_height;
  valid_shape.dimensionSize[3] = image_width;

  auto &aligned_shape = properties.alignedShape;
  int32_t w_stride = ALIGN_16(image_width);
  aligned_shape.numDimensions = 4;
  aligned_shape.dimensionSize[0] = 1;
  aligned_shape.dimensionSize[1] = 3;
  aligned_shape.dimensionSize[2] = image_height;
  aligned_shape.dimensionSize[3] = w_stride;

  int32_t image_length = image_height * w_stride * 3 / 2;
  hbSysAllocCachedMem(&tensor->sysMem[0], image_length);
}

// sigmoid 激活函数实现
float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

TrackDetectionNode::TrackDetectionNode(const std::string& node_name,
                      const NodeOptions& options)
  : DnnNode(node_name, options) {
  this->declare_parameter<std::string>("model_path", model_path_);
  this->declare_parameter<std::string>("sub_img_topic", sub_img_topic_);
  // 声明置信度阈值参数
  this->declare_parameter<double>("confidence_threshold", confidence_threshold_);

  this->get_parameter("model_path", model_path_);
  this->get_parameter("sub_img_topic", sub_img_topic_);
  // 获取置信度阈值参数
  this->get_parameter("confidence_threshold", confidence_threshold_);
  RCLCPP_INFO(rclcpp::get_logger("TrackDetectionNode"), "Confidence Threshold: %f", confidence_threshold_);

  sign_subscriber_ = this->create_subscription<std_msgs::msg::Int32>(
    "/sign4return",
    rclcpp::SensorDataQoS(),
    std::bind(&TrackDetectionNode::sign_callback, this, std::placeholders::_1));

  if (Init() != 0) {
    RCLCPP_ERROR(rclcpp::get_logger("TrackDetectionNode"), "Init failed!");
  }

  publisher_ =
    this->create_publisher<ai_msgs::msg::PerceptionTargets>("racing_track_center_detection", 5);
  subscriber_hbmem_ =
    this->create_subscription<hbm_img_msgs::msg::HbmMsg1080P>(
      sub_img_topic_,
      rclcpp::SensorDataQoS(),
      std::bind(&TrackDetectionNode::subscription_callback,
      this,
      std::placeholders::_1));
}

TrackDetectionNode::~TrackDetectionNode() {

}

void TrackDetectionNode::sign_callback(const std_msgs::msg::Int32::SharedPtr msg) {
  int sign_value = msg->data;
  RCLCPP_INFO(rclcpp::get_logger("TrackDetectionNode"), "Received sign4return: %d", sign_value);
  
  if (sign_value == 5) {
    // 收到5时，取消巡线功能
    enable_lane_following_ = false;
    RCLCPP_INFO(rclcpp::get_logger("TrackDetectionNode"), "Lane following disabled (manual control)");
  } else if (sign_value == 6) {
    // 收到6时，恢复巡线功能
    enable_lane_following_ = true;
    RCLCPP_INFO(rclcpp::get_logger("TrackDetectionNode"), "Lane following enabled");
  }
}

int TrackDetectionNode::SetNodePara() {
  if (!dnn_node_para_ptr_) {
    return -1;
  }
  RCLCPP_INFO(rclcpp::get_logger("TrackDetectionNode"), "path:%s\n", model_path_.c_str());
  dnn_node_para_ptr_->model_file = model_path_;
  dnn_node_para_ptr_->model_task_type = model_task_type_;
  dnn_node_para_ptr_->task_num = 1;
  dnn_node_para_ptr_->bpu_core_ids.push_back(HB_BPU_CORE_1);;
  return 0;
}

int TrackDetectionNode::PostProcess(
  const std::shared_ptr<DnnNodeOutput> &outputs) {
  std::shared_ptr<LineCoordinateParser> line_coordinate_parser =
      std::make_shared<LineCoordinateParser>();
  std::shared_ptr<LineCoordinateResult> result =
      std::make_shared<LineCoordinateResult>();
  
  // Parse现在会填充 x, y, confidence
  line_coordinate_parser->Parse(result, outputs->output_tensors[0]);
  
  float x = result->x;
  float y = (result->y) + 256;
  float confidence = result->confidence;

  RCLCPP_INFO(rclcpp::get_logger("TrackDetectionNode"),
               "PostProcess raw x: %f, raw y: %f, confidence: %f", result->x, result->y, confidence);
  // 在这里进行y轴的偏移处理，使其回到原始图片全尺寸的坐标系
  // 模型预测的y是基于224x224裁剪区域内的点，需要加上裁剪区域的起始Y
  // CROP_START_Y (480-224 = 256)

  RCLCPP_INFO(rclcpp::get_logger("TrackDetectionNode"),
               "PostProcess mapped x: %d, mapped y: %d, confidence: %f",
               static_cast<int>(x), static_cast<int>(y), confidence);

  // 检查巡线功能是否开启 及 置信度阈值
  if (!enable_lane_following_) {
    RCLCPP_DEBUG(rclcpp::get_logger("TrackDetectionNode"), "Lane following is disabled, skip publishing.");
    return 0;
  }

  ai_msgs::msg::PerceptionTargets::UniquePtr msg(
        new ai_msgs::msg::PerceptionTargets());
  msg->set__header(*outputs->msg_header);

  ai_msgs::msg::Target target;
  target.set__type("track_center");
  
  // 添加坐标点到 ai_msgs::msg::Point 的 point 数组中
  ai_msgs::msg::Point track_center_ai_point; // 注意Point是ai_msgs::msg::Point
  track_center_ai_point.set__type("midline_point"); // 你可以给这个Point一个类型名称

  geometry_msgs::msg::Point32 pt; // 实际的坐标数据是 geometry_msgs::msg::Point32
  pt.set__x(x); // x 坐标
  pt.set__y(y); // y 坐标
  pt.set__z(0.0); // Z轴设置为0，因为是2D坐标

  track_center_ai_point.point.emplace_back(pt); // 将 geometry_msgs::msg::Point32 添加到 ai_msgs::msg::Point 的 point 列表中

  // 将置信度添加到 ai_msgs::msg::Point 的 confidence 列表中
  // 因为只有一个点，所以 confidence 数组也只添加一个元素
  track_center_ai_point.confidence.emplace_back(confidence);

  std::vector<ai_msgs::msg::Point> tar_points;
  tar_points.push_back(track_center_ai_point); // 将 ai_msgs::msg::Point 对象添加到 Target 的 points 向量中
  target.set__points(tar_points);

  // 移除之前尝试的 target.properties.emplace_back(confidence_property);
  // 因为根据提供的消息定义，Target没有properties字段
  // 同时也不需要 ai_msgs::msg::Property confidence_property;

  msg->targets.emplace_back(target); // 将 Target 添加到 PerceptionTargets 的 targets 向量中
  publisher_->publish(std::move(msg)); // 发布消息
  return 0;
}

void TrackDetectionNode::subscription_callback(
    const hbm_img_msgs::msg::HbmMsg1080P::SharedPtr msg) {
  int ret = 0;
  if (!msg || !rclcpp::ok()) {
    return;
  }
  // 新增：若巡线功能关闭，直接返回，不处理图像
  if (!enable_lane_following_) {
    return;
  }
  
  std::stringstream ss;
  ss << "Recved img encoding: "
     << std::string(reinterpret_cast<const char*>(msg->encoding.data()))
     << ", h: " << msg->height << ", w: " << msg->width
     << ", step: " << msg->step << ", index: " << msg->index
     << ", stamp: " << msg->time_stamp.sec << "_"
     << msg->time_stamp.nanosec << ", data size: " << msg->data_size;
  RCLCPP_DEBUG(rclcpp::get_logger("TrackDetectionNode"), "%s", ss.str().c_str());

  auto model_manage = GetModel();
  if (!model_manage) {
    RCLCPP_ERROR(rclcpp::get_logger("TrackDetectionNode"), "Invalid model");
    return;
  }

  hbDNNRoi roi;
  roi.left = 0;
  roi.top = 480-224;
  roi.right = 640 - 1;
  roi.bottom = 480 - 1;
  // hbDNNTensor input_tensor;
  // resize
  cv::Mat img_mat(msg->height * 3 / 2, msg->width, CV_8UC1, (void*)(msg->data.data()));
  cv::Range rowRange(roi.top, 480);
  cv::Range colRange(roi.left, 640);
  cv::Mat crop_img_mat = hobot_cv::hobotcv_crop(img_mat, msg->height, msg->width, 224, 224, rowRange, colRange);

 std::shared_ptr<hobot::easy_dnn::NV12PyramidInput> pyramid = nullptr;
  pyramid = hobot::dnn_node::ImageProc::GetNV12PyramidFromNV12Img(
      reinterpret_cast<const char*>(crop_img_mat.data),
      224,
      224,
      224,
      224);
  
  if (!pyramid) {
    RCLCPP_ERROR(rclcpp::get_logger("TrackDetectionNode"), "Get Nv12 pyramid fail!");
    return;
  }
  
  std::vector<std::shared_ptr<DNNInput>> inputs;
  auto rois = std::make_shared<std::vector<hbDNNRoi>>();
  roi.left = 0;
  roi.top = 0;
  roi.right = 224;
  roi.bottom = 224;
  rois->push_back(roi);

  for (size_t i = 0; i < rois->size(); i++) {
    for (int32_t j = 0; j < model_manage->GetInputCount(); j++) {
      inputs.push_back(pyramid);
    }
  }

  auto dnn_output = std::make_shared<DnnNodeOutput>();
  dnn_output->msg_header = std::make_shared<std_msgs::msg::Header>();
  dnn_output->msg_header->set__frame_id(std::to_string(msg->index));
  dnn_output->msg_header->set__stamp(msg->time_stamp);
  ret = Predict(inputs, dnn_output, rois);
}

int TrackDetectionNode::Predict(
  std::vector<std::shared_ptr<DNNInput>> &dnn_inputs,
  const std::shared_ptr<DnnNodeOutput> &output,
  const std::shared_ptr<std::vector<hbDNNRoi>> rois) {
  RCLCPP_INFO(rclcpp::get_logger("TrackDetectionNode"), "input count:%zu roi count:%zu", dnn_inputs.size(), rois->size());
  return Run(dnn_inputs,
             output,
             rois,
             true);
}

int32_t LineCoordinateParser::Parse(
    std::shared_ptr<LineCoordinateResult> &output,
    std::shared_ptr<DNNTensor> &output_tensor) {
  if (!output_tensor) {
    RCLCPP_ERROR(rclcpp::get_logger("TrackDetectionNode"), "invalid out tensor");
    rclcpp::shutdown();
  }
  std::shared_ptr<LineCoordinateResult> result;
  if (!output) {
    result = std::make_shared<LineCoordinateResult>();
    output = result;
  } else {
    result = std::dynamic_pointer_cast<LineCoordinateResult>(output);
  }
  DNNTensor &tensor = *output_tensor;
  const int32_t *shape = tensor.properties.validShape.dimensionSize;
  RCLCPP_DEBUG(rclcpp::get_logger("TrackDetectionNode"),
               "Output tensor shape: batch=%d, channels=%d, height=%d, width=%d",
               shape[0], shape[1], shape[2], shape[3]); // 修正DnnNodeOutput的shape访问方式
  hbSysFlushMem(&(tensor.sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);
  
  float *output_data = reinterpret_cast<float *>(tensor.sysMem[0].virAddr);
  float x_norm = output_data[0];
  float y_norm = output_data[1];
  float confidence_logit = output_data[2]; // 读取第三个输出值

  result->confidence = sigmoid(confidence_logit); // 应用 Sigmoid

  // x 轴反归一化到 640x224 裁剪区域的像素坐标
  // 注意，这里的 224 是模型输入尺寸，640 是裁剪区域的宽度
  result->x = (x_norm * 112 + 112) * 640.0 / 224.0;
  
  // y 轴反归一化到 640x224 裁剪区域的像素坐标
  // 这里的 224 是模型输入尺寸和裁剪区域的高度
  result->y = 224 - (y_norm * 112 + 112);

  RCLCPP_INFO(rclcpp::get_logger("TrackDetectionNode"),
               "Parse -> x_norm: %f, y_norm: %f, conf_logit: %f",
               x_norm, y_norm, confidence_logit);
  RCLCPP_INFO(rclcpp::get_logger("TrackDetectionNode"),
               "Parse -> x_in_cropped_region: %f, y_in_cropped_region: %f, confidence: %f",
               result->x, result->y, result->confidence);
  return 0;
}

int main(int argc, char* argv[]) {

  rclcpp::init(argc, argv);

  rclcpp::spin(std::make_shared<TrackDetectionNode>("GetLineCoordinate"));

  rclcpp::shutdown();

  RCLCPP_WARN(rclcpp::get_logger("TrackDetectionNode"), "Pkg exit.");
  return 0;
}
