#include <cmath>
#include <string>
#include <vector>
#include <array>

#include "rclcpp/rclcpp.hpp"

#include "sensor_msgs/msg/imu.hpp"
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

inline double stampToSec(const builtin_interfaces::msg::Time& t) {
  return static_cast<double>(t.sec) + 1e-9 * static_cast<double>(t.nanosec);
}

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

}  // namespace

class RioNode final : public rclcpp::Node {
public:
  RioNode() : Node("state_estimator_cpp") {
    declareParams_();
    loadParamsOrThrow_();
    setupRosInterfaces_();

    RCLCPP_INFO(get_logger(),
      "RIO C++ node configured:\n"
      "  imu_topic=%s\n"
      "  radar_topic=%s\n"
      "  state_topic=%s\n"
      "  T_acc=%.1f\n"
      "  T_ars=%.1f\n"
      "  gating_enable=%s\n"
      "  gate_nsigma=%.1f\n"
      "  extrinsics_l_BR_B=[%.2f, %.2f, %.2f]\n"
      "  extrinsics_q_R_B(body->radar)=[%.2f, %.2f, %.2f, %.2f]\n"
      "  initial_sigma=[%.5e, %.5e, %.5e, %.5e, %.5e, %.5e, %.5e, %.5e, %.5e, %.5e, %.5e, %.5e, %.5e, %.5e, %.5e]\n"
      "----------------------------------------------------------",
      imu_topic_.c_str(), radar_topic_.c_str(), state_topic_.c_str(),
      T_acc_, T_ars_, gating_enable_ ? "true" : "false", gate_nsigma_,
      params_rio_.p_BR_B.x(), params_rio_.p_BR_B.y(), params_rio_.p_BR_B.z(),
      params_rio_.q_RB.x(), params_rio_.q_RB.y(), params_rio_.q_RB.z(), params_rio_.q_RB.w(), 
      std::sqrt(P0_diag_[0]), std::sqrt(P0_diag_[1]), std::sqrt(P0_diag_[2]),
      std::sqrt(P0_diag_[3]), std::sqrt(P0_diag_[4]), std::sqrt(P0_diag_[5]),
      std::sqrt(P0_diag_[6]), std::sqrt(P0_diag_[7]), std::sqrt(P0_diag_[8]),
      std::sqrt(P0_diag_[9]), std::sqrt(P0_diag_[10]), std::sqrt(P0_diag_[11]),
      std::sqrt(P0_diag_[12]), std::sqrt(P0_diag_[13]), std::sqrt(P0_diag_[14]));
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

    // Extrinsics
    this->declare_parameter<std::vector<double>>("parameters.l_BR_B", {0.0, 0.0, 0.0});
    this->declare_parameter<std::vector<double>>("parameters.q_R_B", {0.0, 0.0, 0.0, 1.0}); // [x y z w]
    this->declare_parameter<int>("radar_vr_sign", 1);

    // Topics
    this->declare_parameter<std::string>("parameters.state_estimate_topic", "/rio/pose");
    this->declare_parameter<std::string>("parameters.imu_topic", "/imu/data");
    this->declare_parameter<std::string>("parameters.radar_topic", "/radar/cloud");

    // Extra (C++ node specific; safe defaults)
    this->declare_parameter<double>("parameters.max_dt", 0.05);   // matches rio-lib default
    this->declare_parameter<double>("parameters.min_dt", 1e-4);
  }

  void loadParamsOrThrow_() {
    // Topics
    state_topic_ = this->get_parameter("parameters.state_estimate_topic").as_string();
    imu_topic_   = this->get_parameter("parameters.imu_topic").as_string();
    radar_topic_ = this->get_parameter("parameters.radar_topic").as_string();

    // Q (12)
    const auto Qv = this->get_parameter("parameters.Q").as_double_array();
    if (Qv.size() != 12) {
      throw std::runtime_error("parameters.Q must have length 12");
    }

    radar_sigma_vr_ = this->get_parameter("parameters.radar_sigma_vr").as_double();
    if (!(radar_sigma_vr_ > 0.0)) {
      throw std::runtime_error("parameters.radar_sigma_vr must be > 0");
    }

    T_acc_ = this->get_parameter("parameters.T_acc").as_double();
    T_ars_ = this->get_parameter("parameters.T_ars").as_double();

    gating_enable_ = this->get_parameter("parameters.radar_gating_enable").as_bool();
    gate_nsigma_   = this->get_parameter("parameters.radar_gate_nsigma").as_double();

    radar_vr_sign_ = this->get_parameter("radar_vr_sign").as_int();

    const auto l = this->get_parameter("parameters.l_BR_B").as_double_array();
    if (l.size() != 3) {
      throw std::runtime_error("parameters.l_BR_B must have length 3");
    }

    const auto q = this->get_parameter("parameters.q_R_B").as_double_array();
    if (q.size() != 4) {
      throw std::runtime_error("parameters.q_R_B must have length 4 [x y z w]");
    }

    // Configure rio-lib Params:
    rio::Params p;

    // Gravity: Python labels frame "ned". Your YAML uses standard gravity magnitude.
    p.g_W = rio::Vec3(0.0f, 0.0f, -9.81f);

    // Convert Q diag (12) -> noise densities used by rio-lib
    // Python comment order: accel white (0..2), accel bias RW (3..5),
    // gyro white (6..8), gyro bias RW (9..11) :contentReference[oaicite:8]{index=8}
    p.sigma_acc = static_cast<float>(std::sqrt(std::max(0.0, Qv[0])));
    p.sigma_ba  = static_cast<float>(std::sqrt(std::max(0.0, Qv[3])));
    p.sigma_gyr = static_cast<float>(std::sqrt(std::max(0.0, Qv[6])));
    p.sigma_bg  = static_cast<float>(std::sqrt(std::max(0.0, Qv[9])));

    // Bias time constants
    p.tau_ba = static_cast<float>(T_acc_);
    p.tau_bg = static_cast<float>(T_ars_);

    // dt clamps (exposed)
    p.max_dt = static_cast<float>(this->get_parameter("parameters.max_dt").as_double());
    p.min_dt = static_cast<float>(this->get_parameter("parameters.min_dt").as_double());

    // Extrinsics:
    // Python parameter name q_R_B is radar->body. rio-lib stores q_RB as body->radar.
    const float qx = static_cast<float>(q[0]);
    const float qy = static_cast<float>(q[1]);
    const float qz = static_cast<float>(q[2]);
    const float qw = static_cast<float>(q[3]);
    const rio::Quat q_R_B = quatFromXYWZ(qx, qy, qz, qw);     // radar->body
    p.q_RB = q_R_B.conjugate();                               // body->radar

    p.p_BR_B = rio::Vec3(static_cast<float>(l[0]),
                         static_cast<float>(l[1]),
                         static_cast<float>(l[2]));           // IMU->radar in body

    params_rio_ = p;

    // Init filter
    eskf_.setParams(params_rio_);

    // Initial covariance from YAML nested keys (matches δx order: [dp dv dba dtheta dbg])
    const auto sig_p  = getVec3ParamOrThrow(this, "initial_sigma.position");
    const auto sig_v  = getVec3ParamOrThrow(this, "initial_sigma.velocity");
    const auto sig_ba = getVec3ParamOrThrow(this, "initial_sigma.accel_bias");
    const auto sig_th_deg = getVec3ParamOrThrow(this, "initial_sigma.attitude_deg");
    const auto sig_bg = getVec3ParamOrThrow(this, "initial_sigma.gyro_bias");

    // Convert attitude sigma from degrees -> radians
    std::array<double, 3> sig_th = {
      deg2rad(sig_th_deg[0]),
      deg2rad(sig_th_deg[1]),
      deg2rad(sig_th_deg[2])
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

    fill3(0,  sig_p);   // dp
    fill3(3,  sig_v);   // dv
    fill3(6,  sig_ba);  // dba
    fill3(9,  sig_th);  // dtheta
    fill3(12, sig_bg);  // dbg

    rio::NominalState x0;
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

    imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
      imu_topic_, rclcpp::SensorDataQoS(),
      std::bind(&RioNode::onImu_, this, std::placeholders::_1));

    radar_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      radar_topic_, rclcpp::SensorDataQoS(),
      std::bind(&RioNode::onRadar_, this, std::placeholders::_1));
  }

  // ---------------- Core callbacks ----------------
  void onImu_(const sensor_msgs::msg::Imu::SharedPtr msg) {
    // NaN guard similar to Python
    if (!std::isfinite(msg->orientation.x) || !std::isfinite(msg->orientation.y) ||
        !std::isfinite(msg->orientation.z) || !std::isfinite(msg->orientation.w) ||
        !std::isfinite(msg->angular_velocity.z)) {
      return;
    }

    const double t = stampToSec(msg->header.stamp);

    // First timestamp only seeds time
    if (!initialized_time_) {
      last_imu_time_ = t;
      initialized_time_ = true;
      return;
    }



    // Angular velocity
    last_omega_b_ = rio::Vec3(
      static_cast<float>(msg->angular_velocity.x),
      static_cast<float>(msg->angular_velocity.y),
      static_cast<float>(msg->angular_velocity.z));

    last_f_b_ = rio::Vec3(
      static_cast<float>(msg->linear_acceleration.x),
      static_cast<float>(msg->linear_acceleration.y),
      static_cast<float>(msg->linear_acceleration.z));

    // Initialize attitude from gravity
    if (!initialized_att_) {
      const rio::Vec3 f_b = last_f_b_;

      const float fn = f_b.norm();
      if (fn < 9.0f || fn > 10.5f) {
        RCLCPP_WARN_THROTTLE(get_logger(), *this->get_clock(), 2000,
          "Waiting for level/stationary IMU for attitude init (|acc|=%.3f)", fn);
        return;
      }
      RCLCPP_INFO(get_logger(), "Initializing attitude from gravity (|acc|=%.3f)", fn);

      const rio::Vec3 gb = f_b / std::max(1e-6f, fn);
      const float roll  = std::atan2(gb.y(), gb.z());
      const float pitch = std::atan2(-gb.x(), std::sqrt(gb.y()*gb.y() + gb.z()*gb.z()));
      const float yaw   = 0.0f; // arbitrary without compass

      tf2::Quaternion q;
      q.setRPY(roll, pitch, yaw);

      rio::NominalState x0 = eskf_.state();
      x0.q_WB = rio::Quat(static_cast<float>(q.w()),
                          static_cast<float>(q.x()),
                          static_cast<float>(q.y()),
                          static_cast<float>(q.z())).normalized();

      eskf_.reset(x0, P0_diag_.data(), static_cast<float>(t));

      initialized_att_ = true;

      RCLCPP_INFO(get_logger(), "Initialized attitude from gravity: roll=%.3f pitch=%.3f", roll, pitch);
      return;
    }

    // Propagate
    rio::ImuSample s;
    s.t = static_cast<float>(t);
    s.acc = rio::Vec3(
      static_cast<float>(msg->linear_acceleration.x),
      static_cast<float>(msg->linear_acceleration.y),
      static_cast<float>(msg->linear_acceleration.z));
    s.gyr = rio::Vec3(
      static_cast<float>(msg->angular_velocity.x),
      static_cast<float>(msg->angular_velocity.y),
      static_cast<float>(msg->angular_velocity.z));

    eskf_.propagate(s);

    // If we have radar measurements buffered from the last PointCloud2 callback, apply them now
    if (!radar_buf_.empty()) {
      if (gating_enable_) {
        // simple scalar gating on residual with predicted vr; to avoid duplicating Jacobians,
        // we gate using |r| < nsigma*sigma. This is weaker than full NIS but cheap.
        applyRadarUpdatesGated_();
      } else {
        eskf_.updateDopplerWithOmega(radar_buf_.data(), radar_buf_.size(), last_omega_b_);
      }
      radar_buf_.clear();
    }

    publishState_(msg->header.stamp);
  }

  void onRadar_(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    if (msg->width * msg->height == 0) return;

    // Require xyz
    if (!hasField(*msg, "x") || !hasField(*msg, "y") || !hasField(*msg, "z")) return;

    // Doppler field name: try common options
    const std::string dop_field = findFirstExistingField(
      *msg, {"doppler", "vr", "v", "velocity", "radial_velocity"});
    if (dop_field.empty()) {
      RCLCPP_WARN_THROTTLE(get_logger(), *this->get_clock(), 2000,
        "Radar PointCloud2 missing Doppler field (expected doppler/vr/v/velocity/radial_velocity).");
      return;
    }

    // Iterators require correct type. This assumes float32 fields.
    // If your Doppler field is float64 or int16, you must adjust.
    sensor_msgs::PointCloud2ConstIterator<float> it_x(*msg, "x");
    sensor_msgs::PointCloud2ConstIterator<float> it_y(*msg, "y");
    sensor_msgs::PointCloud2ConstIterator<float> it_z(*msg, "z");
    sensor_msgs::PointCloud2ConstIterator<float> it_v(*msg, dop_field);

    radar_buf_.clear();
    radar_buf_.reserve(msg->width * msg->height);

    const float sigma = static_cast<float>(radar_sigma_vr_);
    const size_t npts = static_cast<size_t>(msg->width) * static_cast<size_t>(msg->height);

    for (size_t i = 0; i < npts; ++i, ++it_x, ++it_y, ++it_z, ++it_v) {
      rio::Vec3 p_R(*it_x, *it_y, *it_z);
      if (!finite3(p_R)) continue;

      const float r = p_R.norm();
      if (r < 1e-3f) continue;

      rio::RadarDoppler m;
      m.u_R = p_R / r;
      m.vr = static_cast<float>(radar_vr_sign_) * (*it_v);
      m.sigma = sigma;

      radar_buf_.push_back(m);
    }
  }

  void applyRadarUpdatesGated_() {
    // Gate using predicted vr_hat computed from current state.
    // We reuse rio-lib measurement model implicitly by re-building residual per measurement:
    // We call update sequentially but skip if |residual| > nsigma*sigma.
    // For simplicity, we call rio-lib’s update one-by-one with omega.
    const float ns = static_cast<float>(gate_nsigma_);
    for (const auto& m : radar_buf_) {
      // Predict vr_hat approximately: u dot v_R (with lever-arm)
      // We compute v_R here to gate without touching covariance math.
      const auto& x = eskf_.state();

      const rio::Mat3 C_BW = x.q_WB.conjugate().toRotationMatrix();
      const rio::Mat3 C_RB = params_rio_.q_RB.toRotationMatrix();

      const rio::Vec3 v_B = C_BW * x.v_WB;
      const rio::Vec3 v_radar_B = v_B + last_omega_b_.cross(params_rio_.p_BR_B);
      const rio::Vec3 v_R = C_RB * v_radar_B;

      const float vr_hat = m.u_R.dot(v_R);
      const float r = m.vr - vr_hat;

      if (std::fabs(r) <= ns * m.sigma) {
        eskf_.updateDopplerWithOmega(&m, 1, last_omega_b_);
      }
    }
  }

  // ---------------- Publishing ----------------
  void publishState_(const builtin_interfaces::msg::Time& stamp) {
    const auto& x = eskf_.state();
    const auto& P = eskf_.covariance();

    // TF: ned -> body
    geometry_msgs::msg::TransformStamped tf;
    tf.header.stamp = stamp;
    tf.header.frame_id = "ned";
    tf.child_frame_id = "body";
    tf.transform.translation.x = x.p_WB.x();
    tf.transform.translation.y = x.p_WB.y();
    tf.transform.translation.z = x.p_WB.z();
    tf.transform.rotation.x = x.q_WB.x();
    tf.transform.rotation.y = x.q_WB.y();
    tf.transform.rotation.z = x.q_WB.z();
    tf.transform.rotation.w = x.q_WB.w();
    tf_broadcaster_->sendTransform(tf);

    // Odometry
    nav_msgs::msg::Odometry odom;
    odom.header.stamp = stamp;
    odom.header.frame_id = "ned";
    odom.child_frame_id = "body";

    odom.pose.pose.position.x = x.p_WB.x();
    odom.pose.pose.position.y = x.p_WB.y();
    odom.pose.pose.position.z = x.p_WB.z();
    odom.pose.pose.orientation.x = x.q_WB.x();
    odom.pose.pose.orientation.y = x.q_WB.y();
    odom.pose.pose.orientation.z = x.q_WB.z();
    odom.pose.pose.orientation.w = x.q_WB.w();

    // Map covariance like Python: pose position from P(0:3,0:3), pose attitude from P(9:12,9:12)
    // ROS expects 6x6 covariance packed row-major.
    for (double &c : odom.pose.covariance) c = 0.0;
    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 3; ++c)
        odom.pose.covariance[r * 6 + c] = static_cast<double>(P(r, c));

    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 3; ++c)
        odom.pose.covariance[(r + 3) * 6 + (c + 3)] = static_cast<double>(P(9 + r, 9 + c));

    odom.twist.twist.linear.x = x.v_WB.x();
    odom.twist.twist.linear.y = x.v_WB.y();
    odom.twist.twist.linear.z = x.v_WB.z();

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

    // Radar extrinsics publisher (configured, not estimated)
    geometry_msgs::msg::PoseStamped extr;
    extr.header = odom.header;
    extr.pose.position.x = params_rio_.p_BR_B.x();
    extr.pose.position.y = params_rio_.p_BR_B.y();
    extr.pose.position.z = params_rio_.p_BR_B.z();

    // Publish radar->body quaternion as in Python parameter naming:
    const rio::Quat q_R_B = params_rio_.q_RB.conjugate(); // invert body->radar
    extr.pose.orientation.x = q_R_B.x();
    extr.pose.orientation.y = q_R_B.y();
    extr.pose.orientation.z = q_R_B.z();
    extr.pose.orientation.w = q_R_B.w();
    radar_extr_pub_->publish(extr);
  }

private:
  // rio-lib filter
  rio::RioEskf eskf_;
  rio::Params params_rio_{};

  // ROS interfaces
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr radar_sub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  rclcpp::Publisher<geometry_msgs::msg::Vector3Stamped>::SharedPtr accel_bias_pub_;
  rclcpp::Publisher<geometry_msgs::msg::Vector3Stamped>::SharedPtr gyro_bias_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr radar_extr_pub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

  // Parameters
  std::string state_topic_;
  std::string imu_topic_;
  std::string radar_topic_;
  double radar_sigma_vr_{0.038};
  double T_acc_{1000.0};
  double T_ars_{500.0};
  bool gating_enable_{true};
  double gate_nsigma_{3.0};
  int radar_vr_sign_{1};

  // State
  bool initialized_att_{false};
  bool initialized_time_{false};
  double last_imu_time_{0.0};

  rio::Vec3 last_omega_b_{rio::Vec3::Zero()};
  rio::Vec3 last_f_b_{rio::Vec3::Zero()};
  std::vector<rio::RadarDoppler> radar_buf_;
  std::array<float, 15> P0_diag_{};
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<RioNode>());
  rclcpp::shutdown();
  return 0;
}