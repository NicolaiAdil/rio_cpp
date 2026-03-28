#include <cmath>
#include <string>
#include <vector>
#include <array>
#include <mutex>

#include "rclcpp/rclcpp.hpp"

#include "px4_msgs/msg/sensor_accel.hpp"
#include "px4_msgs/msg/sensor_gyro.hpp"

#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"

#include "nav_msgs/msg/odometry.hpp"
#include "geometry_msgs/msg/vector3_stamped.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"

#include "tf2/LinearMath/Quaternion.h"
#include "tf2_ros/transform_broadcaster.h"

#include <rio/rio_eskf.h>

namespace {

inline rio::Quat quatFromXYWZ(float x, float y, float z, float w) {
  rio::Quat q(w, x, y, z);
  return q.normalized();
}

inline bool finite3(const rio::Vec3& v) {
  return std::isfinite(v.x()) && std::isfinite(v.y()) && std::isfinite(v.z());
}

static inline bool hasField(const sensor_msgs::msg::PointCloud2& msg, const std::string& name) {
  for (const auto& f : msg.fields) {
    if (f.name == name) return true;
  }
  return false;
}

static inline std::string findFirstExistingField(
  const sensor_msgs::msg::PointCloud2& msg,
  const std::initializer_list<const char*>& candidates)
{
  for (const auto& c : candidates) {
    if (hasField(msg, c)) return std::string(c);
  }
  return std::string();
}

static inline std::array<double, 3> getVec3ParamOrThrow(
  rclcpp::Node* node, const std::string& name)
{
  const auto v = node->get_parameter(name).as_double_array();
  if (v.size() != 3) {
    throw std::runtime_error(name + " must have length 3");
  }
  return {v[0], v[1], v[2]};
}

static inline double deg2rad(double d) { return d * M_PI / 180.0; }

/// Convert PX4 microsecond timestamp to seconds (double)
inline double px4UsToSec(uint64_t us) {
  return static_cast<double>(us) * 1e-6;
}

/// Convert a double time (seconds) to a ROS stamp
inline builtin_interfaces::msg::Time secToStamp(double t) {
  builtin_interfaces::msg::Time stamp;
  stamp.sec = static_cast<int32_t>(std::floor(t));
  stamp.nanosec = static_cast<uint32_t>((t - std::floor(t)) * 1e9);
  return stamp;
}

}  // namespace

class RioNode final : public rclcpp::Node {
public:
  RioNode() : Node("state_estimator_cpp") {
    declareParams_();
    loadParamsOrThrow_();
    setupRosInterfaces_();

    RCLCPP_INFO(get_logger(),
      "RIO C++ node configured (PX4 accel+gyro mode):\n"
      "  accel_topic=%s\n"
      "  gyro_topic=%s\n"
      "  radar_topic=%s\n"
      "  state_topic=%s\n"
      "  T_acc=%.1f  T_ars=%.1f\n"
      "  gating_enable=%s  gate_nsigma=%.1f\n"
      "  p_IR=[%.4f, %.4f, %.4f]\n"
      "  q_IR=[%.4f, %.4f, %.4f, %.4f]\n"
      "  vr_sign=%d\n"
      "----------------------------------------------------------",
      accel_topic_.c_str(), gyro_topic_.c_str(),
      radar_topic_.c_str(), state_topic_.c_str(),
      static_cast<double>(params_rio_.tau_ba),
      static_cast<double>(params_rio_.tau_bg),
      params_rio_.gating_enable ? "true" : "false",
      static_cast<double>(params_rio_.gate_nsigma),
      static_cast<double>(params_rio_.p_IR.x()),
      static_cast<double>(params_rio_.p_IR.y()),
      static_cast<double>(params_rio_.p_IR.z()),
      static_cast<double>(params_rio_.q_IR.x()),
      static_cast<double>(params_rio_.q_IR.y()),
      static_cast<double>(params_rio_.q_IR.z()),
      static_cast<double>(params_rio_.q_IR.w()),
      static_cast<int>(params_rio_.vr_sign));
  }

private:
  // ---------------- Parameters (match Python names) ----------------
  void declareParams_() {
    // EKF parameters
    this->declare_parameter<std::vector<double>>("parameters.Q", std::vector<double>(12, 0.0));
    this->declare_parameter<std::vector<double>>("initial_sigma.position", {1e-6, 1e-6, 1e-6});
    this->declare_parameter<std::vector<double>>("initial_sigma.velocity", {1e-1, 1e-1, 1e-1});
    this->declare_parameter<std::vector<double>>("initial_sigma.accel_bias", {1e-2, 1e-2, 1e-2});
    this->declare_parameter<std::vector<double>>("initial_sigma.gyro_bias", {1e-4, 1e-4, 1e-4});
    this->declare_parameter<std::vector<double>>("initial_sigma.attitude_deg", {6.0, 6.0, 1e-6});
    this->declare_parameter<std::vector<double>>("initial_sigma.radar_position", {2e-3, 2e-3, 2e-3});
    this->declare_parameter<std::vector<double>>("initial_sigma.radar_attitude_deg", {0.5, 0.5, 0.5});

    this->declare_parameter<double>("parameters.radar_sigma_vr", 0.038);
    this->declare_parameter<double>("parameters.T_acc", 1000.0);
    this->declare_parameter<double>("parameters.T_ars", 500.0);

    // Gating parameters
    this->declare_parameter<bool>("parameters.radar_gating_enable", true);
    this->declare_parameter<double>("parameters.radar_gate_nsigma", 3.0);

    // Extrinsics: p_IR (IMU->radar in IMU frame), q_IR (rotation IMU->radar) [x y z w]
    this->declare_parameter<std::vector<double>>("parameters.p_IR", {0.0, 0.0, 0.0});
    this->declare_parameter<std::vector<double>>("parameters.q_IR", {0.0, 0.0, 0.0, 1.0}); // [x y z w]
    this->declare_parameter<int>("radar_vr_sign", 1);

    // Topics — now separate accel and gyro instead of a single IMU topic
    this->declare_parameter<std::string>("parameters.state_estimate_topic", "/rio/pose");
    this->declare_parameter<std::string>("parameters.accel_topic", "/fmu/out/sensor_accel");
    this->declare_parameter<std::string>("parameters.gyro_topic", "/fmu/out/sensor_gyro");
    this->declare_parameter<std::string>("parameters.radar_topic", "/radar/cloud");

    // Max age difference (seconds) between accel and gyro to consider them paired
    this->declare_parameter<double>("parameters.max_accel_gyro_dt", 0.005);

    // Extra (C++ node specific; safe defaults)
    this->declare_parameter<double>("parameters.max_dt", 0.05);
    this->declare_parameter<double>("parameters.min_dt", 1e-4);
  }

  void loadParamsOrThrow_() {
    // Topics
    state_topic_ = this->get_parameter("parameters.state_estimate_topic").as_string();
    accel_topic_ = this->get_parameter("parameters.accel_topic").as_string();
    gyro_topic_  = this->get_parameter("parameters.gyro_topic").as_string();
    radar_topic_ = this->get_parameter("parameters.radar_topic").as_string();
    max_accel_gyro_dt_ = this->get_parameter("parameters.max_accel_gyro_dt").as_double();

    // Q (12)
    const auto Qv = this->get_parameter("parameters.Q").as_double_array();
    if (Qv.size() != 12) {
      throw std::runtime_error("parameters.Q must have length 12");
    }

    // Configure rio-lib Params
    rio::Params p;

    p.g_W = rio::Vec3(0.0f, 0.0f, -9.81f);

    // Q diag (12) -> noise densities
    p.sigma_acc = static_cast<float>(std::sqrt(std::max(0.0, Qv[0])));
    p.sigma_ba  = static_cast<float>(std::sqrt(std::max(0.0, Qv[3])));
    p.sigma_gyr = static_cast<float>(std::sqrt(std::max(0.0, Qv[6])));
    p.sigma_bg  = static_cast<float>(std::sqrt(std::max(0.0, Qv[9])));

    // Bias time constants
    p.tau_ba = static_cast<float>(this->get_parameter("parameters.T_acc").as_double());
    p.tau_bg = static_cast<float>(this->get_parameter("parameters.T_ars").as_double());

    // dt clamps
    p.max_dt = static_cast<float>(this->get_parameter("parameters.max_dt").as_double());
    p.min_dt = static_cast<float>(this->get_parameter("parameters.min_dt").as_double());

    // Radar measurement params
    p.sigma_vr      = static_cast<float>(this->get_parameter("parameters.radar_sigma_vr").as_double());
    p.gating_enable = this->get_parameter("parameters.radar_gating_enable").as_bool();
    p.gate_nsigma   = static_cast<float>(this->get_parameter("parameters.radar_gate_nsigma").as_double());
    p.vr_sign       = static_cast<float>(this->get_parameter("radar_vr_sign").as_int());

    // Extrinsics: p_IR, q_IR
    const auto p_ir = this->get_parameter("parameters.p_IR").as_double_array();
    if (p_ir.size() != 3) {
      throw std::runtime_error("parameters.p_IR must have length 3");
    }
    p.p_IR = rio::Vec3(static_cast<float>(p_ir[0]),
                       static_cast<float>(p_ir[1]),
                       static_cast<float>(p_ir[2]));

    const auto q_ir = this->get_parameter("parameters.q_IR").as_double_array();
    if (q_ir.size() != 4) {
      throw std::runtime_error("parameters.q_IR must have length 4 [x y z w]");
    }
    p.q_IR = quatFromXYWZ(static_cast<float>(q_ir[0]),
                          static_cast<float>(q_ir[1]),
                          static_cast<float>(q_ir[2]),
                          static_cast<float>(q_ir[3]));

    params_rio_ = p;
    eskf_.setParams(params_rio_);

    // Initial covariance (21D): [dp dv dba dtheta dbg dp_IR dtheta_IR]
    const auto sig_p        = getVec3ParamOrThrow(this, "initial_sigma.position");
    const auto sig_v        = getVec3ParamOrThrow(this, "initial_sigma.velocity");
    const auto sig_ba       = getVec3ParamOrThrow(this, "initial_sigma.accel_bias");
    const auto sig_th_deg   = getVec3ParamOrThrow(this, "initial_sigma.attitude_deg");
    const auto sig_bg       = getVec3ParamOrThrow(this, "initial_sigma.gyro_bias");
    const auto sig_pir      = getVec3ParamOrThrow(this, "initial_sigma.radar_position");
    const auto sig_thir_deg = getVec3ParamOrThrow(this, "initial_sigma.radar_attitude_deg");

    const std::array<double, 3> sig_th = {
      deg2rad(sig_th_deg[0]), deg2rad(sig_th_deg[1]), deg2rad(sig_th_deg[2])
    };
    const std::array<double, 3> sig_thir = {
      deg2rad(sig_thir_deg[0]), deg2rad(sig_thir_deg[1]), deg2rad(sig_thir_deg[2])
    };

    auto fill3 = [&](int start, const std::array<double,3>& s){
      for (int i = 0; i < 3; ++i) {
        const double si = s[i];
        if (!std::isfinite(si) || si < 0.0) {
          throw std::runtime_error("initial_sigma contains invalid value");
        }
        P0_diag_[start + i] = static_cast<float>(si * si);
      }
    };

    fill3(0,  sig_p);
    fill3(3,  sig_v);
    fill3(6,  sig_ba);
    fill3(9,  sig_th);
    fill3(12, sig_bg);
    fill3(15, sig_pir);
    fill3(18, sig_thir);

    rio::NominalState x0;
    x0.p_IR = params_rio_.p_IR;
    x0.q_IR = params_rio_.q_IR;
    eskf_.reset(x0, P0_diag_.data(), 0.0f);

    initialized_att_ = false;
    initialized_time_ = false;
  }

  void setupRosInterfaces_() {
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>(state_topic_, 10);
    accel_bias_pub_ = this->create_publisher<geometry_msgs::msg::Vector3Stamped>("/rio/accel_bias", 10);
    gyro_bias_pub_  = this->create_publisher<geometry_msgs::msg::Vector3Stamped>("/rio/gyro_bias", 10);
    radar_extr_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("/radar/extrinsics", 10);

    // Subscribe to PX4 accel and gyro separately
    accel_sub_ = this->create_subscription<px4_msgs::msg::SensorAccel>(
      accel_topic_, rclcpp::SensorDataQoS(),
      std::bind(&RioNode::onAccel_, this, std::placeholders::_1));

    gyro_sub_ = this->create_subscription<px4_msgs::msg::SensorGyro>(
      gyro_topic_, rclcpp::SensorDataQoS(),
      std::bind(&RioNode::onGyro_, this, std::placeholders::_1));

    radar_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      radar_topic_, rclcpp::SensorDataQoS(),
      std::bind(&RioNode::onRadar_, this, std::placeholders::_1));
  }

  // ----------------------------------------------------------------
  // PX4 accel callback — store latest, then try to run the filter
  // ----------------------------------------------------------------
  void onAccel_(const px4_msgs::msg::SensorAccel::SharedPtr msg) {
    if (!std::isfinite(msg->x) || !std::isfinite(msg->y) || !std::isfinite(msg->z)) {
      return;
    }

    latest_accel_.x = msg->x;
    latest_accel_.y = msg->y;
    latest_accel_.z = msg->z;
    latest_accel_time_us_ = msg->timestamp_sample;
    accel_valid_ = true;

    tryProcessImu_();
  }

  // ----------------------------------------------------------------
  // PX4 gyro callback — store latest, then try to run the filter
  // ----------------------------------------------------------------
  void onGyro_(const px4_msgs::msg::SensorGyro::SharedPtr msg) {
    if (!std::isfinite(msg->x) || !std::isfinite(msg->y) || !std::isfinite(msg->z)) {
      return;
    }

    latest_gyro_.x = msg->x;
    latest_gyro_.y = msg->y;
    latest_gyro_.z = msg->z;
    latest_gyro_time_us_ = msg->timestamp_sample;
    gyro_valid_ = true;

    tryProcessImu_();
  }

  // ----------------------------------------------------------------
  // Combine accel + gyro and run the ESKF predict step.
  // Called from whichever callback arrives second.
  // Uses the newer of the two timestamps as the IMU timestamp.
  // ----------------------------------------------------------------
  void tryProcessImu_() {
    if (!accel_valid_ || !gyro_valid_) return;

    // Check that accel and gyro are close in time
    const double t_acc = px4UsToSec(latest_accel_time_us_);
    const double t_gyr = px4UsToSec(latest_gyro_time_us_);
    const double age_diff = std::abs(t_acc - t_gyr);

    if (age_diff > max_accel_gyro_dt_) {
      // Samples too far apart; wait for a fresher match
      return;
    }

    // Use the newer timestamp as the combined IMU time
    const double t = std::max(t_acc, t_gyr);
    // RCLCPP_INFO(get_logger(), "Processing IMU sample at t=%.3f (age diff=%.3f s)", t, age_diff);

    const rio::Vec3 f_b(latest_accel_.x, latest_accel_.y, -latest_accel_.z);
    const rio::Vec3 w_b(-latest_gyro_.x, -latest_gyro_.y, latest_gyro_.z);

    // Mark consumed so we don't re-process the same pair
    accel_valid_ = false;
    gyro_valid_  = false;

    // --- From here the logic mirrors the original onImu_ ---

    if (!initialized_time_) {
      last_imu_time_ = t;
      initialized_time_ = true;
      return;
    }

    // Initialize attitude from gravity
    if (!initialized_att_) {
      if (!eskf_.initAttitudeFromGravity(f_b, P0_diag_.data(), static_cast<float>(t))) {
        RCLCPP_WARN_THROTTLE(get_logger(), *this->get_clock(), 2000,
          "Waiting for level/stationary IMU for attitude init (|acc|=%.3f)", f_b.norm());
        return;
      }
      initialized_att_ = true;
      RCLCPP_INFO(get_logger(), "Initialized attitude from gravity (|acc|=%.3f)", f_b.norm());
      return;
    }

    // Propagate
    const float dt = static_cast<float>(t - last_imu_time_);
    last_imu_time_ = t;

    rio::ImuSample s;
    s.t   = static_cast<float>(t);
    s.acc = f_b;
    s.gyr = w_b;

    eskf_.predict(s, dt);
    eskf_.insPropagation(s, dt);

    if (!radar_buf_.empty()) {
      // RCLCPP_INFO(get_logger(), "Running correction with %zu radar returns", radar_buf_.size());
      const auto res = eskf_.correct(radar_buf_.data(), radar_buf_.size(), s);
      if (res.n_rejected > 0 || res.n_skipped > 0) {
        RCLCPP_INFO(get_logger(),
          "Radar correction: total=%zu accepted=%zu rejected=%zu skipped=%zu",
          res.n_total, res.n_accepted, res.n_rejected, res.n_skipped);
      }
      radar_buf_.clear();
    } else {
      eskf_.advancePriorToPosterior();
    }

    // Publish using a ROS stamp derived from the PX4 time
    publishState_(secToStamp(t));
  }

  void onRadar_(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    if (msg->width * msg->height == 0) return;

    if (!hasField(*msg, "x") || !hasField(*msg, "y") || !hasField(*msg, "z")) return;

    const std::string dop_field = findFirstExistingField(
      *msg, {"doppler", "vr", "v", "velocity", "radial_velocity"});
    if (dop_field.empty()) {
      RCLCPP_WARN_THROTTLE(get_logger(), *this->get_clock(), 2000,
        "Radar PointCloud2 missing Doppler field.");
      return;
    }

    sensor_msgs::PointCloud2ConstIterator<float> it_x(*msg, "x");
    sensor_msgs::PointCloud2ConstIterator<float> it_y(*msg, "y");
    sensor_msgs::PointCloud2ConstIterator<float> it_z(*msg, "z");
    sensor_msgs::PointCloud2ConstIterator<float> it_v(*msg, dop_field);

    radar_buf_.clear();
    radar_buf_.reserve(msg->width * msg->height);

    const size_t npts = static_cast<size_t>(msg->width) * static_cast<size_t>(msg->height);

    for (size_t i = 0; i < npts; ++i, ++it_x, ++it_y, ++it_z, ++it_v) {
      rio::Vec3 p_R(*it_x, *it_y, *it_z);
      if (!finite3(p_R)) continue;

      const float r = p_R.norm();
      if (r < 1e-3f) continue;

      rio::RadarDoppler m;
      m.u_R = p_R / r;
      m.vr = *it_v;
      m.sigma = params_rio_.sigma_vr;

      radar_buf_.push_back(m);
    }
  }

  // ---------------- Publishing ----------------
  void publishState_(const builtin_interfaces::msg::Time& stamp) {
    const auto& x = eskf_.getState();
    const auto& P = eskf_.getCovariance();

    // TF: ned -> body
    geometry_msgs::msg::TransformStamped tf;
    tf.header.stamp = stamp;
    tf.header.frame_id = "ned";
    tf.child_frame_id = "body";
    tf.transform.translation.x = x.p_WI.x();
    tf.transform.translation.y = x.p_WI.y();
    tf.transform.translation.z = x.p_WI.z();
    tf.transform.rotation.x = x.q_WI.x();
    tf.transform.rotation.y = x.q_WI.y();
    tf.transform.rotation.z = x.q_WI.z();
    tf.transform.rotation.w = x.q_WI.w();
    tf_broadcaster_->sendTransform(tf);

    // Odometry
    nav_msgs::msg::Odometry odom;
    odom.header.stamp = stamp;
    odom.header.frame_id = "ned";
    odom.child_frame_id = "body";

    odom.pose.pose.position.x = x.p_WI.x();
    odom.pose.pose.position.y = x.p_WI.y();
    odom.pose.pose.position.z = x.p_WI.z();
    odom.pose.pose.orientation.x = x.q_WI.x();
    odom.pose.pose.orientation.y = x.q_WI.y();
    odom.pose.pose.orientation.z = x.q_WI.z();
    odom.pose.pose.orientation.w = x.q_WI.w();

    for (double &c : odom.pose.covariance) c = 0.0;
    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 3; ++c)
        odom.pose.covariance[r * 6 + c] = static_cast<double>(P(r, c));

    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 3; ++c)
        odom.pose.covariance[(r + 3) * 6 + (c + 3)] = static_cast<double>(P(9 + r, 9 + c));

    odom.twist.twist.linear.x = x.v_WI.x();
    odom.twist.twist.linear.y = x.v_WI.y();
    odom.twist.twist.linear.z = x.v_WI.z();

    for (double &c : odom.twist.covariance) c = 0.0;
    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 3; ++c)
        odom.twist.covariance[r * 6 + c] = static_cast<double>(P(3 + r, 3 + c));

    odom_pub_->publish(odom);

    // Bias publishers
    geometry_msgs::msg::Vector3Stamped ba;
    ba.header = odom.header;
    ba.vector.x = x.b_a.x();
    ba.vector.y = x.b_a.y();
    ba.vector.z = x.b_a.z();
    accel_bias_pub_->publish(ba);

    geometry_msgs::msg::Vector3Stamped bg;
    bg.header = odom.header;
    bg.vector.x = x.b_g.x();
    bg.vector.y = x.b_g.y();
    bg.vector.z = x.b_g.z();
    gyro_bias_pub_->publish(bg);

    // Radar extrinsics publisher
    geometry_msgs::msg::PoseStamped extr;
    extr.header = odom.header;
    extr.pose.position.x = x.p_IR.x();
    extr.pose.position.y = x.p_IR.y();
    extr.pose.position.z = x.p_IR.z();
    extr.pose.orientation.x = x.q_IR.x();
    extr.pose.orientation.y = x.q_IR.y();
    extr.pose.orientation.z = x.q_IR.z();
    extr.pose.orientation.w = x.q_IR.w();
    radar_extr_pub_->publish(extr);
  }

private:
  // rio-lib filter
  rio::RioEskf eskf_;
  rio::Params params_rio_{};

  // ROS interfaces
  rclcpp::Subscription<px4_msgs::msg::SensorAccel>::SharedPtr accel_sub_;
  rclcpp::Subscription<px4_msgs::msg::SensorGyro>::SharedPtr gyro_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr radar_sub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  rclcpp::Publisher<geometry_msgs::msg::Vector3Stamped>::SharedPtr accel_bias_pub_;
  rclcpp::Publisher<geometry_msgs::msg::Vector3Stamped>::SharedPtr gyro_bias_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr radar_extr_pub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

  // Parameters
  std::string state_topic_;
  std::string accel_topic_;
  std::string gyro_topic_;
  std::string radar_topic_;
  double max_accel_gyro_dt_{0.005};

  // Latest buffered accel/gyro samples for pairing
  struct { float x{0}, y{0}, z{0}; } latest_accel_;
  struct { float x{0}, y{0}, z{0}; } latest_gyro_;
  uint64_t latest_accel_time_us_{0};
  uint64_t latest_gyro_time_us_{0};
  bool accel_valid_{false};
  bool gyro_valid_{false};

  // State
  bool initialized_att_{false};
  bool initialized_time_{false};
  double last_imu_time_{0.0};

  std::vector<rio::RadarDoppler> radar_buf_;
  std::array<float, 21> P0_diag_{};
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<RioNode>());
  rclcpp::shutdown();
  return 0;
}