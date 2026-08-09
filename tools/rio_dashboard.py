#!/usr/bin/env python3
"""Live debug dashboard for the rio_cpp radar-inertial odometry node.

Subscribes to the RIO node's published topics and plots the key health signals
live (works on live runs and on bag playback), overlaid against PX4 EKF2 ground
truth. Single matplotlib window, refreshed from a ROS timer (no threads).

    ros2 run rio_cpp rio_dashboard.py
"""
import math
from collections import deque

import numpy as np
import matplotlib.pyplot as plt

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, Vector3Stamped
from sensor_msgs.msg import PointCloud2
from px4_msgs.msg import VehicleLocalPosition, VehicleAttitude, SensorCombined

# Per-axis colours: x/N -> blue, y/E -> orange, z/D -> green.
AX = ("tab:blue", "tab:orange", "tab:green")
BUF = 20000  # max samples per series (~60 s at a few hundred Hz)


def quat_to_rpy(w, x, y, z):
    """Quaternion (w,x,y,z) -> roll, pitch, yaw in radians (ZYX)."""
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def rate_hz(times):
    """Message rate from buffered timestamps (seconds)."""
    if len(times) < 2:
        return 0.0
    span = times[-1] - times[0]
    return (len(times) - 1) / span if span > 1e-6 else 0.0


class RioDashboard(Node):
    def __init__(self):
        super().__init__("rio_dashboard")

        p = self.declare_parameter
        self.topic_state = p("topic_state", "/rio/pose").value
        self.topic_accel_bias = p("topic_accel_bias", "/rio/accel_bias").value
        self.topic_gyro_bias = p("topic_gyro_bias", "/rio/gyro_bias").value
        self.topic_extrinsics = p("topic_extrinsics", "/radar/extrinsics").value
        self.topic_truth_pos = p("topic_truth_local_pos", "/fmu/out/vehicle_local_position").value
        self.topic_truth_att = p("topic_truth_attitude", "/fmu/out/vehicle_attitude").value
        self.topic_imu = p("topic_imu", "/fmu/out/sensor_combined").value
        self.topic_radar = p("topic_radar", "/sr75_node/points").value
        self.window_secs = float(p("window_secs", 60.0).value)
        self.refresh_hz = float(p("refresh_hz", 10.0).value)

        # Buffers: one deque per plotted series, keyed by name.
        keys = (
            "pt px py pz vx vy vz roll pitch yaw sp sv sa "  # /rio/pose
            "abt abx aby abz gbt gbx gby gbz "               # biases
            "ext exr exp exy "                               # radar extrinsic q_IR (rpy)
            "tpt tpx tpy tpz tvt tvx tvy tvz "               # EKF2 position / velocity
            "tat tar tap tay "                               # EKF2 attitude (rpy)
            "imu_rt rad_rt"                                  # IMU / radar arrival times
        ).split()
        self.d = {k: deque(maxlen=BUF) for k in keys}
        self.radar_np = 0            # points in the last radar scan
        self.t0 = None               # shared time origin (PX4 seconds)
        self.t_latest = 0.0

        # Subscriptions. /rio/* are reliable; PX4 out-topics are best-effort.
        self.create_subscription(Odometry, self.topic_state, self.cb_state, 10)
        self.create_subscription(Vector3Stamped, self.topic_accel_bias, self.cb_accel_bias, 10)
        self.create_subscription(Vector3Stamped, self.topic_gyro_bias, self.cb_gyro_bias, 10)
        self.create_subscription(PoseStamped, self.topic_extrinsics, self.cb_extrinsics, 10)
        q = qos_profile_sensor_data
        self.create_subscription(VehicleLocalPosition, self.topic_truth_pos, self.cb_truth_pos, q)
        self.create_subscription(VehicleAttitude, self.topic_truth_att, self.cb_truth_att, q)
        self.create_subscription(SensorCombined, self.topic_imu, self.cb_imu, q)
        self.create_subscription(PointCloud2, self.topic_radar, self.cb_radar, q)

        self._build_figure()
        self.create_timer(1.0 / self.refresh_hz, self._refresh)
        self.get_logger().info("rio_dashboard up — waiting for data...")

    # ---------------- time helpers ----------------
    def _rel(self, t_abs):
        """Seconds since the first observed (PX4-based) timestamp."""
        if self.t0 is None:
            self.t0 = t_abs
        self.t_latest = max(self.t_latest, t_abs - self.t0)
        return t_abs - self.t0

    @staticmethod
    def _stamp_sec(header):
        return header.stamp.sec + header.stamp.nanosec * 1e-9

    # ---------------- RIO callbacks ----------------
    def cb_state(self, msg: Odometry):
        t = self._rel(self._stamp_sec(msg.header))
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        vel = msg.twist.twist.linear
        r, pch, y = quat_to_rpy(ori.w, ori.x, ori.y, ori.z)
        pc = msg.pose.covariance      # 6x6 row-major: pos(0-2), attitude(3-5)
        tc = msg.twist.covariance     # 6x6 row-major: velocity(0-2)
        sp = math.sqrt(max(0.0, (pc[0] + pc[7] + pc[14]) / 3.0))
        sa = math.sqrt(max(0.0, (pc[21] + pc[28] + pc[35]) / 3.0))
        sv = math.sqrt(max(0.0, (tc[0] + tc[7] + tc[14]) / 3.0))
        for k, v in (("pt", t), ("px", pos.x), ("py", pos.y), ("pz", pos.z),
                     ("vx", vel.x), ("vy", vel.y), ("vz", vel.z),
                     ("roll", r), ("pitch", pch), ("yaw", y),
                     ("sp", max(sp, 1e-9)), ("sv", max(sv, 1e-9)), ("sa", max(sa, 1e-9))):
            self.d[k].append(v)

    def cb_accel_bias(self, msg: Vector3Stamped):
        t = self._rel(self._stamp_sec(msg.header))
        for k, v in (("abt", t), ("abx", msg.vector.x), ("aby", msg.vector.y), ("abz", msg.vector.z)):
            self.d[k].append(v)

    def cb_gyro_bias(self, msg: Vector3Stamped):
        t = self._rel(self._stamp_sec(msg.header))
        for k, v in (("gbt", t), ("gbx", msg.vector.x), ("gby", msg.vector.y), ("gbz", msg.vector.z)):
            self.d[k].append(v)

    def cb_extrinsics(self, msg: PoseStamped):
        t = self._rel(self._stamp_sec(msg.header))
        o = msg.pose.orientation
        r, pch, y = quat_to_rpy(o.w, o.x, o.y, o.z)
        for k, v in (("ext", t), ("exr", r), ("exp", pch), ("exy", y)):
            self.d[k].append(v)

    # ---------------- EKF2 truth callbacks ----------------
    def cb_truth_pos(self, msg: VehicleLocalPosition):
        t = self._rel(msg.timestamp * 1e-6)
        if msg.xy_valid and msg.z_valid:
            for k, v in (("tpt", t), ("tpx", msg.x), ("tpy", msg.y), ("tpz", msg.z)):
                self.d[k].append(v)
        if msg.v_xy_valid and msg.v_z_valid:
            for k, v in (("tvt", t), ("tvx", msg.vx), ("tvy", msg.vy), ("tvz", msg.vz)):
                self.d[k].append(v)

    def cb_truth_att(self, msg: VehicleAttitude):
        t = self._rel(msg.timestamp * 1e-6)
        r, pch, y = quat_to_rpy(msg.q[0], msg.q[1], msg.q[2], msg.q[3])
        for k, v in (("tat", t), ("tar", r), ("tap", pch), ("tay", y)):
            self.d[k].append(v)

    # ---------------- rate-only callbacks ----------------
    def cb_imu(self, msg: SensorCombined):
        self.d["imu_rt"].append(msg.timestamp * 1e-6)

    def cb_radar(self, msg: PointCloud2):
        self.d["rad_rt"].append(self._stamp_sec(msg.header))
        self.radar_np = msg.width * msg.height

    # ---------------- figure ----------------
    def _line(self, ax, color, style="-", label=None, lw=1.4):
        (ln,) = ax.plot([], [], style, color=color, label=label, lw=lw)
        return ln

    def _triple(self, ax, labels, truth=False):
        """Three per-axis RIO lines (+ dotted truth twins) sharing an axis."""
        rio = [self._line(ax, AX[i], "-", labels[i]) for i in range(3)]
        tru = [self._line(ax, AX[i], ":", None, lw=1.2) for i in range(3)] if truth else None
        ax.legend(loc="upper left", fontsize=7, ncol=3)
        ax.grid(True, alpha=0.3)
        return rio, tru

    def _build_figure(self):
        plt.ion()
        self.fig, axes = plt.subplots(3, 3, figsize=(15, 9))
        self.fig.canvas.manager.set_window_title("RIO dashboard")
        (a_p, a_v, a_att), (a_traj, a_ab, a_gb), (a_ex, a_sig, a_txt) = axes
        self.axes = axes

        a_p.set_title("position p_WI [m] (NED)")
        self.l_p, self.l_tp = self._triple(a_p, ("N", "E", "D"), truth=True)
        a_v.set_title("velocity v_WI [m/s] (NED)")
        self.l_v, self.l_tv = self._triple(a_v, ("vN", "vE", "vD"), truth=True)
        a_att.set_title("attitude [deg]")
        self.l_att, self.l_tatt = self._triple(a_att, ("roll", "pitch", "yaw"), truth=True)

        a_traj.set_title("trajectory N–E [m]")
        self.l_traj = self._line(a_traj, "tab:blue", "-", "RIO")
        self.l_traj_t = self._line(a_traj, "0.5", ":", "EKF2")
        (self.l_traj_dot,) = a_traj.plot([], [], "o", color="tab:red", ms=6)
        a_traj.set_xlabel("E [m]"); a_traj.set_ylabel("N [m]")
        a_traj.set_aspect("equal", adjustable="datalim")
        a_traj.legend(loc="upper left", fontsize=7); a_traj.grid(True, alpha=0.3)

        a_ab.set_title("accel bias b_a [m/s²]")
        self.l_ab, _ = self._triple(a_ab, ("x", "y", "z"))
        a_gb.set_title("gyro bias b_g [rad/s]")
        self.l_gb, _ = self._triple(a_gb, ("x", "y", "z"))
        a_ex.set_title("radar extrinsic q_IR [deg] (gimbal)")
        self.l_ex, _ = self._triple(a_ex, ("roll", "pitch", "yaw"))

        a_sig.set_title("std-dev σ (log)")
        self.l_sp = self._line(a_sig, "tab:blue", "-", "σ pos [m]")
        self.l_sv = self._line(a_sig, "tab:orange", "-", "σ vel [m/s]")
        self.l_sa = self._line(a_sig, "tab:green", "-", "σ att [rad]")
        a_sig.set_yscale("log"); a_sig.legend(loc="upper right", fontsize=7); a_sig.grid(True, alpha=0.3)

        a_txt.axis("off")
        self.txt = a_txt.text(0.02, 0.98, "", va="top", ha="left",
                              family="monospace", fontsize=10, transform=a_txt.transAxes)

        self._time_axes = [a_p, a_v, a_att, a_ab, a_gb, a_ex, a_sig]
        self.fig.tight_layout()
        plt.show(block=False)

    # ---------------- refresh ----------------
    def _set(self, line, tkey, vkey, deg=False, unwrap=False):
        t = list(self.d[tkey])
        v = list(self.d[vkey])
        n = min(len(t), len(v))
        if n == 0:
            return
        v = np.array(v[:n])
        if unwrap:
            v = np.unwrap(v)
        if deg:
            v = np.degrees(v)
        line.set_data(t[:n], v)

    def _refresh(self):
        if not plt.fignum_exists(self.fig.number):
            rclpy.try_shutdown()
            return
        try:
            # position / velocity
            for i, k in enumerate(("px", "py", "pz")):
                self._set(self.l_p[i], "pt", k)
            for i, k in enumerate(("tpx", "tpy", "tpz")):
                self._set(self.l_tp[i], "tpt", k)
            for i, k in enumerate(("vx", "vy", "vz")):
                self._set(self.l_v[i], "pt", k)
            for i, k in enumerate(("tvx", "tvy", "tvz")):
                self._set(self.l_tv[i], "tvt", k)
            # attitude (deg, unwrapped)
            for i, k in enumerate(("roll", "pitch", "yaw")):
                self._set(self.l_att[i], "pt", k, deg=True, unwrap=True)
            for i, k in enumerate(("tar", "tap", "tay")):
                self._set(self.l_tatt[i], "tat", k, deg=True, unwrap=True)
            # biases / extrinsic
            for i, k in enumerate(("abx", "aby", "abz")):
                self._set(self.l_ab[i], "abt", k)
            for i, k in enumerate(("gbx", "gby", "gbz")):
                self._set(self.l_gb[i], "gbt", k)
            for i, k in enumerate(("exr", "exp", "exy")):
                self._set(self.l_ex[i], "ext", k, deg=True, unwrap=True)
            # sigma (log)
            self._set(self.l_sp, "pt", "sp")
            self._set(self.l_sv, "pt", "sv")
            self._set(self.l_sa, "pt", "sa")
            # trajectory (x=E, y=N)
            self.l_traj.set_data(list(self.d["py"]), list(self.d["px"]))
            self.l_traj_t.set_data(list(self.d["tpy"]), list(self.d["tpx"]))
            if self.d["px"]:
                self.l_traj_dot.set_data([self.d["py"][-1]], [self.d["px"][-1]])

            # rescale time-series x to the rolling window, y automatically
            tmin = max(0.0, self.t_latest - self.window_secs)
            for ax in self._time_axes:
                ax.set_xlim(tmin, max(self.t_latest, tmin + 1e-3))
                ax.relim(); ax.autoscale_view(scalex=False, scaley=True)
            for ax in (self.axes[1][0],):  # trajectory
                ax.relim(); ax.autoscale_view()

            self._update_text()
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
        except Exception as e:  # never let a draw hiccup kill the node
            self.get_logger().warn(f"refresh error: {e}")

    def _update_text(self):
        pose_hz = rate_hz(self.d["pt"])
        ekf2_hz = rate_hz(self.d["tpt"])
        imu_hz = rate_hz(self.d["imu_rt"])
        rad_hz = rate_hz(self.d["rad_rt"])
        age = (self.t_latest - self.d["pt"][-1]) if self.d["pt"] else float("nan")
        status = "live" if self.d["pt"] else "waiting for /rio/pose ..."
        self.txt.set_text(
            "rates [Hz]\n"
            f"  pose   {pose_hz:6.1f}\n"
            f"  imu    {imu_hz:6.1f}\n"
            f"  radar  {rad_hz:6.1f}  ({self.radar_np} pts/scan)\n"
            f"  ekf2   {ekf2_hz:6.1f}\n\n"
            f"pose age  {age:5.2f} s\n"
            f"status: {status}"
        )


def main():
    rclpy.init()
    node = RioDashboard()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
