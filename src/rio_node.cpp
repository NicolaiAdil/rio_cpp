// Copyright 2016 Open Source Robotics Foundation, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include <rio/rio_eskf.h>

class MinimalSubscriber : public rclcpp::Node
{
public:
  MinimalSubscriber()
  : Node("rio_node")
  {
    RCLCPP_INFO(get_logger(), "Starting RIO smoke test...");

    // ---- Create filter ----
    rio::RioEskf eskf;

    // ---- Parameters (MUST be set) ----
    rio::Params p;
    p.g_W = rio::Vec3(0.0f, 0.0f, -9.81f);
    eskf.setParams(p);

    // ---- Initial state ----
    rio::NominalState x0;
    float P0[15];
    for (int i = 0; i < 15; ++i) P0[i] = 1e-3f;

    eskf.reset(x0, P0, 0.0f);

    // ---- Simulate IMU propagation ----
    rio::ImuSample imu;
    imu.t = 1.0f;                         // 1s step
    imu.acc = rio::Vec3(1.0f, 0.0f, 0.0f); // +1 m/s^2 in body x
    imu.gyr = rio::Vec3(0.0f, 0.0f, 0.0f);

    eskf.propagate(imu);

    const auto &x = eskf.state();

    RCLCPP_INFO(get_logger(),
      "After 1s propagate: p=[%.3f %.3f %.3f], v=[%.3f %.3f %.3f]",
      x.p_WB.x(), x.p_WB.y(), x.p_WB.z(),
      x.v_WB.x(), x.v_WB.y(), x.v_WB.z());

    auto topic_callback =
      [this](std_msgs::msg::String::UniquePtr msg) -> void {
        RCLCPP_INFO(this->get_logger(), "I heard: '%s'", msg->data.c_str());
      };
    subscription_ =
      this->create_subscription<std_msgs::msg::String>("topic", 10, topic_callback);
  }

private:
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr subscription_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MinimalSubscriber>());
  rclcpp::shutdown();
  return 0;
}
