#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/robot/b2/motion_switcher/motion_switcher_client.hpp>
#include <unitree/robot/b2/robot_state/robot_state_client.hpp>
#include <unitree/robot/r1/loco/r1_loco_client.hpp>
#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/idl/hg/BmsState_.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <deque>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace unitree::robot;
using namespace unitree_hg::msg::dds_;
using Clock = std::chrono::steady_clock;

namespace {

constexpr std::array<int, 10> ARM_JOINTS = {15, 16, 17, 18, 19, 22, 23, 24, 25, 26};
constexpr std::array<int, 3> AUX_JOINTS = {13, 29, 30}; // waist_yaw, head_pitch, head_yaw

// Compliant impedance gains for suspended/tabletop manipulation (gentle contact, zero jerk)
constexpr std::array<double, 10> ARM_KP = {26, 26, 20, 20, 14, 26, 26, 20, 20, 14};
constexpr std::array<double, 10> ARM_KD = {3.0, 3.0, 2.4, 2.4, 1.8, 3.0, 3.0, 2.4, 2.4, 1.8};

// Feather-soft compliant hold gains when a fault/contact freezes motion
constexpr std::array<double, 10> ARM_KP_SOFT = {12, 12, 9, 9, 6, 12, 12, 9, 9, 6};
constexpr std::array<double, 10> ARM_KD_SOFT = {3.2, 3.2, 2.5, 2.5, 1.8, 3.2, 3.2, 2.5, 2.5, 1.8};

// Tight torque thresholds so any unexpected contact/anomaly stops IMMEDIATELY before disturbing robot
constexpr std::array<double, 10> TORQUE_LIMIT = {7.8, 7.8, 5.8, 5.8, 3.5, 7.8, 7.8, 5.8, 5.8, 3.5};
constexpr double MAX_SAFE_MEASURED_DQ = 0.85; // rad/s (commanded max is 0.15 rad/s)

// Conservative URDF joint limits with safety margin
constexpr std::array<double, 10> ARM_MIN = {
    -2.60, -0.18, -1.80, -0.90, -1.85,
    -2.60, -2.35, -1.80, -0.90, -1.85
};
constexpr std::array<double, 10> ARM_MAX = {
     1.80,  2.35,  1.80,  2.05,  1.85,
     1.80,  0.18,  1.80,  2.05,  1.85
};
constexpr std::array<double, 3> AUX_MIN = {-0.65, -0.68, -2.05};
constexpr std::array<double, 3> AUX_MAX = { 0.65,  0.68,  2.05};

// Safety IMU thresholds (rad)
constexpr double IMU_FREEZE_ROLL  = 0.11; // ~6.3 deg -> freeze arm immediately
constexpr double IMU_FREEZE_PITCH = 0.13; // ~7.4 deg -> freeze arm immediately
constexpr double IMU_DAMP_ROLL    = 0.20; // ~11.5 deg -> invoke loco.Damp() (kill leg stepping!)
constexpr double IMU_DAMP_PITCH   = 0.22; // ~12.6 deg -> invoke loco.Damp()

constexpr double MAX_JOINT_SPEED_RAD_S = 0.15;

std::atomic<bool> g_running{true};
void on_signal(int) { g_running = false; }

uint32_t crc32_core(uint32_t* ptr, uint32_t len) {
    uint32_t xbit = 0;
    uint32_t data = 0;
    uint32_t CRC32 = 0xFFFFFFFF;
    const uint32_t dwPolynomial = 0x04c11db7;
    for (uint32_t i = 0; i < len; i++) {
        xbit = 1u << 31;
        data = ptr[i];
        for (uint32_t bits = 0; bits < 32; bits++) {
            if (CRC32 & 0x80000000) {
                CRC32 <<= 1;
                CRC32 ^= dwPolynomial;
            } else {
                CRC32 <<= 1;
            }
            if (data & xbit) CRC32 ^= dwPolynomial;
            xbit >>= 1;
        }
    }
    return CRC32;
}

// Minimal 3D FK for R1 5-DOF arm in waist_yaw_link frame (for 250Hz Cartesian Virtual Wall)
struct Vec3 { double x, y, z; };
struct Mat3 { double m[3][3]; };

Mat3 rpy_mat(double r, double p, double y) {
    double cr = std::cos(r), sr = std::sin(r);
    double cp = std::cos(p), sp = std::sin(p);
    double cy = std::cos(y), sy = std::sin(y);
    Mat3 R{};
    R.m[0][0] = cy * cp; R.m[0][1] = cy * sp * sr - sy * cr; R.m[0][2] = cy * sp * cr + sy * sr;
    R.m[1][0] = sy * cp; R.m[1][1] = sy * sp * sr + cy * cr; R.m[1][2] = sy * sp * cr - cy * sr;
    R.m[2][0] = -sp;     R.m[2][1] = cp * sr;                R.m[2][2] = cp * cr;
    return R;
}

Mat3 axis_rot(double ax, double ay, double az, double q) {
    double c = std::cos(q), s = std::sin(q), C = 1.0 - c;
    Mat3 R{};
    R.m[0][0] = c + ax*ax*C;      R.m[0][1] = ax*ay*C - az*s; R.m[0][2] = ax*az*C + ay*s;
    R.m[1][0] = ay*ax*C + az*s;   R.m[1][1] = c + ay*ay*C;    R.m[1][2] = ay*az*C - ax*s;
    R.m[2][0] = az*ax*C - ay*s;   R.m[2][1] = az*ay*C + ax*s; R.m[2][2] = c + az*az*C;
    return R;
}

Mat3 mul3(const Mat3& A, const Mat3& B) {
    Mat3 C{};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            C.m[i][j] = A.m[i][0]*B.m[0][j] + A.m[i][1]*B.m[1][j] + A.m[i][2]*B.m[2][j];
    return C;
}

Vec3 mul3v(const Mat3& A, const Vec3& v) {
    return {
        A.m[0][0]*v.x + A.m[0][1]*v.y + A.m[0][2]*v.z,
        A.m[1][0]*v.x + A.m[1][1]*v.y + A.m[1][2]*v.z,
        A.m[2][0]*v.x + A.m[2][1]*v.y + A.m[2][2]*v.z
    };
}

// Exact URDF link origins & axes from R1.urdf for Left (+1) and Right (-1) arms
void compute_arm_fk(bool is_left, const double q[5], Vec3& out_wrist, Vec3& out_tcp) {
    double s = is_left ? 1.0 : -1.0;
    struct LinkDef { Vec3 xyz; Vec3 rpy; Vec3 axis; };
    const LinkDef chain[5] = {
        {{ 0.00000,  s * 0.085688,  0.197490}, { s * 0.26187, 0.0, 0.0}, {0.0, 1.0, 0.0}},
        {{ 0.03445,  s * 0.047132, -0.025693}, {-s * 0.26187, 0.0, 0.0}, {1.0, 0.0, 0.0}},
        {{-0.03445,  s * 0.004300, -0.108350}, { 0.0,         0.0, 0.0}, {0.0, 0.0, 1.0}},
        {{ 0.016191, s * 0.026461, -0.082858}, { 0.0,         0.0, 0.0}, {0.0, 1.0, 0.0}},
        {{ 0.112180,-s * 0.030020, -0.011702}, { 0.0,         0.0, 0.0}, {1.0, 0.0, 0.0}},
    };
    Mat3 R = rpy_mat(0, 0, 0);
    Vec3 p{0.0, 0.0, 0.0};
    for (int i = 0; i < 5; ++i) {
        Vec3 dp = mul3v(R, chain[i].xyz);
        p = {p.x + dp.x, p.y + dp.y, p.z + dp.z};
        R = mul3(R, mul3(rpy_mat(chain[i].rpy.x, chain[i].rpy.y, chain[i].rpy.z),
                         axis_rot(chain[i].axis.x, chain[i].axis.y, chain[i].axis.z, q[i])));
    }
    out_wrist = p;
    Vec3 dtcp = mul3v(R, {0.165, 0.0, 0.0});
    out_tcp = {p.x + dtcp.x, p.y + dtcp.y, p.z + dtcp.z};
}

Vec3 cross3(const Vec3& a, const Vec3& b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

// Analytical 3D potential-energy gradient +dU/dq from R1.urdf + BrainCo Revo 2 hand mass
void compute_arm_gravity_tau_single(
    bool is_left,
    const double q[5],
    double imu_roll,
    double imu_pitch,
    double q_waist_roll,
    double q_waist_yaw,
    double extra_hand_mass_kg,
    double out_tau[5]
) {
    double s = is_left ? 1.0 : -1.0;
    struct DynLink { Vec3 xyz; Vec3 rpy; Vec3 axis; Vec3 com; double mass; };
    const DynLink chain[5] = {
        {{ 0.000000,  s * 0.085688,  0.197490}, { s * 0.26187, 0.0, 0.0}, {0.0, 1.0, 0.0}, { 0.00294386,  s * 0.04476868, -0.02286868}, 0.837950},
        {{ 0.034450,  s * 0.047132, -0.025693}, {-s * 0.26187, 0.0, 0.0}, {1.0, 0.0, 0.0}, {-0.03782800,  s * 0.00396700, -0.06649600}, 0.740212},
        {{-0.034450,  s * 0.004300, -0.108350}, { 0.0,         0.0, 0.0}, {0.0, 0.0, 1.0}, { 0.01314300, -s * 0.00435500, -0.07294800}, 0.738441},
        {{ 0.016191,  s * 0.026461, -0.082858}, { 0.0,         0.0, 0.0}, {0.0, 1.0, 0.0}, { 0.07237100, -s * 0.02461200, -0.01156600}, 0.759419},
        {{ 0.112180, -s * 0.030020, -0.011702}, { 0.0,         0.0, 0.0}, {1.0, 0.0, 0.0}, { 0.06804500, -s * 0.00168800,  0.00139200}, 0.324767},
    };
    Mat3 R = mul3(rpy_mat(imu_roll, imu_pitch, 0.0),
                  mul3(axis_rot(1.0, 0.0, 0.0, q_waist_roll),
                       axis_rot(0.0, 0.0, 1.0, q_waist_yaw)));
    Vec3 p{0.0, 0.0, 0.0};
    Vec3 j_pos[5]{};
    Vec3 j_axis[5]{};
    struct MassPt { Vec3 pos; double mass; int min_joint; };
    MassPt bodies[6]{};
    for (int i = 0; i < 5; ++i) {
        Vec3 dp = mul3v(R, chain[i].xyz);
        p = {p.x + dp.x, p.y + dp.y, p.z + dp.z};
        R = mul3(R, mul3(rpy_mat(chain[i].rpy.x, chain[i].rpy.y, chain[i].rpy.z),
                         axis_rot(chain[i].axis.x, chain[i].axis.y, chain[i].axis.z, q[i])));
        j_pos[i] = p;
        j_axis[i] = mul3v(R, chain[i].axis);
        Vec3 dc = mul3v(R, chain[i].com);
        bodies[i] = {{p.x + dc.x, p.y + dc.y, p.z + dc.z}, chain[i].mass, i};
        if (i == 4) {
            Vec3 dh = mul3v(R, {0.130, 0.0, 0.0});
            bodies[5] = {{p.x + dh.x, p.y + dh.y, p.z + dh.z}, std::max(0.0, extra_hand_mass_kg), 4};
        }
    }
    constexpr double G_ACC = 9.80665;
    for (int i = 0; i < 5; ++i) {
        double tau_i = 0.0;
        for (int b = 0; b < 6; ++b) {
            if (bodies[b].min_joint >= i) {
                Vec3 r_vec{bodies[b].pos.x - j_pos[i].x,
                           bodies[b].pos.y - j_pos[i].y,
                           bodies[b].pos.z - j_pos[i].z};
                Vec3 dp_dq = cross3(j_axis[i], r_vec); // d(pos)/dq_i
                tau_i += bodies[b].mass * G_ACC * dp_dq.z; // +dU/dq_i where U = sum m*g*z
            }
        }
        out_tau[i] = tau_i;
    }
}

void compute_arm_gravity_tau(
    const std::array<double, 10>& q10,
    double imu_roll,
    double imu_pitch,
    double q_waist_roll,
    double q_waist_yaw,
    double extra_hand_mass_kg,
    std::array<double, 10>& out_tau10
) {
    compute_arm_gravity_tau_single(true,  &q10[0], imu_roll, imu_pitch, q_waist_roll, q_waist_yaw, extra_hand_mass_kg, &out_tau10[0]);
    compute_arm_gravity_tau_single(false, &q10[5], imu_roll, imu_pitch, q_waist_roll, q_waist_yaw, extra_hand_mass_kg, &out_tau10[5]);
}

// Virtual wall check: forbids under-desk collision (X > 0.21m with Z < -0.055m) and body collision
bool check_virtual_wall(const std::array<double, 10>& q10, double table_z_floor, std::string& reason) {
    for (int side = 0; side < 2; ++side) {
        bool is_left = (side == 0);
        const double* q5 = is_left ? &q10[0] : &q10[5];
        Vec3 wr{}, tcp{};
        compute_arm_fk(is_left, q5, wr, tcp);
        // 1. Under-desk / desk-front-edge protection:
        //    If wrist or TCP extends past X = 0.215m, it MUST stay above table_z_floor (-0.052m)
        if ((tcp.x > 0.215 && tcp.z < table_z_floor) || (wr.x > 0.215 && wr.z < table_z_floor + 0.020)) {
            reason = std::string(is_left ? "Left" : "Right") + " arm virtual desk wall (X=" +
                     std::to_string(tcp.x).substr(0, 5) + ", Z=" + std::to_string(tcp.z).substr(0, 6) + ")";
            return false;
        }
        // 2. Torso self-collision cylinder: when X < 0.20m and Z > -0.22m, keep |Y| >= 0.08m on own side
        if (tcp.x > -0.08 && tcp.x < 0.20 && tcp.z > -0.22) {
            if ((is_left && tcp.y < -0.05) || (!is_left && tcp.y > 0.05)) {
                reason = std::string(is_left ? "Left" : "Right") + " arm torso self-collision guard";
                return false;
            }
        }
    }
    return true;
}

std::string json_escape(const std::string& s) {
    std::string out;
    for (char c : s) {
        if (c == '"') out += "\\\"";
        else if (c == '\\') out += "\\\\";
        else if (c == '\n') out += " ";
        else out += c;
    }
    return out;
}

bool parse_double_field(const std::string& line, const std::string& key, double& out) {
    std::string pat = "\"" + key + "\":";
    auto pos = line.find(pat);
    if (pos == std::string::npos) return false;
    pos += pat.size();
    while (pos < line.size() && line[pos] == ' ') pos++;
    char* endp = nullptr;
    out = std::strtod(line.c_str() + pos, &endp);
    return endp != (line.c_str() + pos);
}

std::string parse_string_field(const std::string& line, const std::string& key) {
    std::string pat = "\"" + key + "\":\"";
    auto pos = line.find(pat);
    if (pos == std::string::npos) return "";
    pos += pat.size();
    auto end = line.find('"', pos);
    if (end == std::string::npos) return "";
    return line.substr(pos, end - pos);
}

bool parse_double_array(const std::string& line, const std::string& key, std::vector<double>& out) {
    std::string pat = "\"" + key + "\":[";
    auto pos = line.find(pat);
    if (pos == std::string::npos) return false;
    pos += pat.size();
    auto end = line.find(']', pos);
    if (end == std::string::npos) return false;
    std::string body = line.substr(pos, end - pos);
    out.clear();
    const char* p = body.c_str();
    while (*p) {
        while (*p == ' ' || *p == ',') p++;
        if (!*p) break;
        char* endp = nullptr;
        double val = std::strtod(p, &endp);
        if (endp == p) break;
        out.push_back(val);
        p = endp;
    }
    return !out.empty();
}

double quintic(double u) {
    u = std::clamp(u, 0.0, 1.0);
    return u * u * u * (10.0 + u * (-15.0 + 6.0 * u));
}

struct Segment {
    std::string label = "move";
    std::array<double, 10> arm_to{};
    std::array<double, 3> aux_to{};
    std::vector<std::array<double, 10>> arm_path;
    std::vector<double> path_cum_s;
    bool has_arm = false;
    bool has_aux = false;
    double duration_s = 2.0;
};

} // namespace

int main(int argc, char** argv) {
    std::string iface = (argc > 1) ? argv[1] : "eth10";
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

    ChannelFactory::Instance()->Init(0, iface);

    std::mutex state_mtx;
    LowState_ latest_low{};
    BmsState_ latest_bms{};
    Clock::time_point low_rx_time{};
    bool has_low = false;
    bool has_bms = false;

    ChannelSubscriber<LowState_> sub_low("rt/lowstate");
    sub_low.InitChannel([&](const void* p) {
        LowState_ val = *static_cast<const LowState_*>(p);
        uint32_t c = crc32_core(reinterpret_cast<uint32_t*>(&val), (sizeof(val) >> 2) - 1);
        if (val.crc() != c) return;
        std::lock_guard<std::mutex> lk(state_mtx);
        latest_low = val;
        low_rx_time = Clock::now();
        has_low = true;
    }, 5);

    ChannelSubscriber<BmsState_> sub_bms("rt/lf/bmsstate");
    sub_bms.InitChannel([&](const void* p) {
        BmsState_ val = *static_cast<const BmsState_*>(p);
        std::lock_guard<std::mutex> lk(state_mtx);
        latest_bms = val;
        has_bms = true;
    }, 2);

    ChannelPublisher<LowCmd_> pub_arm("rt/arm_sdk");
    pub_arm.InitChannel();
    ChannelPublisher<LowCmd_> pub_low("rt/lowcmd");
    pub_low.InitChannel();

    std::atomic<unsigned> external_lowcmd_cnt{0};
    ChannelSubscriber<LowCmd_> sub_lowcmd_watch("rt/lowcmd");
    sub_lowcmd_watch.InitChannel([&](const void*) {
        external_lowcmd_cnt.fetch_add(1, std::memory_order_relaxed);
    }, 2);

    unitree::robot::b2::MotionSwitcherClient switcher;
    switcher.SetTimeout(2.0f);
    switcher.Init();

    unitree::robot::r1::LocoClient loco;
    loco.SetTimeout(1.5f);
    loco.Init();

    unitree::robot::b2::RobotStateClient rsc;
    rsc.SetTimeout(3.5f);
    rsc.Init();

    std::atomic<int> cached_fsm{-1};
    std::atomic<bool> want_enable_stereo{false};
    std::atomic<bool> want_emergency_damp{false};
    std::mutex mode_mtx;
    std::string cached_mode_name = "";

    std::mutex out_mtx;
    auto emit_json = [&](const std::string& js) {
        std::lock_guard<std::mutex> lk(out_mtx);
        std::cout << js << std::endl;
    };

    // Background observer for FSM ID, motion mode, stereo activation, and emergency Damp()
    std::thread mode_thread([&]() {
        while (g_running) {
            if (want_emergency_damp.exchange(false)) {
                int32_t rc = loco.Damp();
                emit_json("{\"event\":\"emergency_damp_executed\",\"rc\":" + std::to_string(rc) + "}");
            }
            if (want_enable_stereo.exchange(false)) {
                int32_t st1 = -1, st2 = -1;
                int32_t r1 = rsc.ServiceSwitch("video_hub", 0, st1);
                std::this_thread::sleep_for(std::chrono::milliseconds(300));
                int32_t r2 = rsc.ServiceSwitch("stereo_patch_pc1", 1, st2);
                emit_json("{\"event\":\"stereo_switched\",\"r_video_hub\":" + std::to_string(r1) +
                          ",\"r_stereo\":" + std::to_string(r2) + "}");
            }
            int fsm = -1;
            if (loco.GetFsmId(fsm) == 0) cached_fsm.store(fsm);
            std::string form, name;
            if (switcher.CheckMode(form, name) == 0) {
                std::lock_guard<std::mutex> lk(mode_mtx);
                cached_mode_name = name;
            }
            for (int k = 0; k < 10 && g_running && !want_emergency_damp.load(); ++k) {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
        }
    });

    std::mutex cmd_mtx;
    std::deque<std::string> cmd_queue;
    Clock::time_point last_heartbeat = Clock::now();

    std::thread stdin_thread([&]() {
        std::string line;
        while (g_running && std::getline(std::cin, line)) {
            if (line.empty()) continue;
            std::lock_guard<std::mutex> lk(cmd_mtx);
            last_heartbeat = Clock::now();
            if (line.find("\"cmd\":\"ping\"") != std::string::npos) continue;
            cmd_queue.push_back(line);
        }
        g_running = false;
    });

    emit_json("{\"event\":\"arm_daemon_ready\"}");

    // Controller state:
    //   SAFE_HOLD = compliant in-place freeze (keeps arm_sdk engaged so FSM 811 NEVER whips the arms!)
    //   RETURN_HOME_FOR_DISENGAGE = smoothly returns to nominal standing pose BEFORE ramping weight 1->0
    enum class Mode { IDLE, RAMP_UP, ACTIVE, SAFE_HOLD, RETURN_HOME_FOR_DISENGAGE, RAMP_DOWN, SUSPENDED_VISCOUS_STOP };
    Mode ctrl_mode = Mode::IDLE;
    bool suspended_mode = false; // true = Developer Mode rt/lowcmd (0 Nm legs, harness suspended)
    double weight = 0.0;
    double table_z_floor = -0.068; // virtual desk floor in waist_yaw_link [m] (dynamically updated from fitted plane)
    double grav_ff_scale = 0.85;   // URDF + Revo 2 gravity compensation feedforward scale [0..1]
    double extra_hand_mass_kg = 0.160; // extra BrainCo Revo 2 + adapter mass beyond R1.urdf wrist_roll_link
    bool grav_ff_enabled = true;
    std::array<double, 10> cur_tau_grav{};
    std::array<double, 10> cur_tau_ext{};
    std::array<double, 10> cmd_arm{};
    std::array<double, 3> cmd_aux{};
    std::array<double, 10> nominal_home_arm{};
    std::array<double, 3> nominal_home_aux{};
    std::array<double, 10> seg_arm_from{};
    std::array<double, 3> seg_aux_from{};
    Segment active_seg{};
    bool has_active_seg = false;
    Clock::time_point seg_start{};
    std::deque<Segment> seg_queue;
    std::string stage_label = "idle";
    std::string last_fault = "";
    Clock::time_point last_telemetry = Clock::now();
    std::array<Clock::time_point, 10> track_err_since{};

    // Compliant freeze: NEVER drop mode_pr=0 abruptly while arm is away from nominal pose!
    auto trigger_compliant_freeze = [&](const std::string& reason, const LowState_& st) {
        last_fault = reason;
        seg_queue.clear();
        has_active_seg = false;
        for (size_t i = 0; i < 10; ++i) cmd_arm[i] = st.motor_state()[ARM_JOINTS[i]].q();
        for (size_t i = 0; i < 3; ++i) cmd_aux[i] = st.motor_state()[AUX_JOINTS[i]].q();
        if (ctrl_mode != Mode::IDLE && ctrl_mode != Mode::RAMP_DOWN) {
            ctrl_mode = Mode::SAFE_HOLD;
            stage_label = "fault_hold";
        }
        emit_json("{\"event\":\"arm_fault_freeze\",\"reason\":\"" + json_escape(reason) + "\"}");
    };

    // Catastrophic balance / ESTOP kill-switch: immediately damp entire robot so legs NEVER thrash!
    auto trigger_emergency_damp = [&](const std::string& reason) {
        last_fault = reason;
        seg_queue.clear();
        has_active_seg = false;
        if (!suspended_mode) {
            ctrl_mode = Mode::IDLE;
            weight = 0.0;
            stage_label = "damped_estop";
            LowCmd_ zmsg{};
            zmsg.mode_machine() = 1;
            zmsg.mode_pr() = 0;
            zmsg.crc() = crc32_core(reinterpret_cast<uint32_t*>(&zmsg), (sizeof(zmsg) >> 2) - 1);
            pub_arm.Write(zmsg);
            want_emergency_damp.store(true);
        } else {
            // In suspended mode: pure viscous damping (kp=0, kd=3.2, legs=0Nm) so arms settle softly without dropping!
            weight = 0.0;
            ctrl_mode = Mode::SUSPENDED_VISCOUS_STOP;
            stage_label = "damped_estop";
        }
        emit_json("{\"event\":\"arm_emergency_damp\",\"reason\":\"" + json_escape(reason) + "\"}");
    };

    // Safe disengage: smoothly move back to nominal_home_arm/aux FIRST, then ramp weight down over 2.5s
    auto begin_safe_disengage = [&](const LowState_& st) {
        seg_queue.clear();
        has_active_seg = false;
        double max_diff = 0.0;
        for (size_t i = 0; i < 10; ++i) {
            cmd_arm[i] = st.motor_state()[ARM_JOINTS[i]].q();
            max_diff = std::max(max_diff, std::abs(cmd_arm[i] - nominal_home_arm[i]));
        }
        for (size_t i = 0; i < 3; ++i) {
            cmd_aux[i] = st.motor_state()[AUX_JOINTS[i]].q();
            max_diff = std::max(max_diff, std::abs(cmd_aux[i] - nominal_home_aux[i]));
        }
        if (max_diff > 0.10 && !suspended_mode) {
            seg_arm_from = cmd_arm;
            seg_aux_from = cmd_aux;
            active_seg = Segment{};
            active_seg.label = "safe_return_nominal";
            active_seg.arm_to = nominal_home_arm;
            active_seg.aux_to = nominal_home_aux;
            active_seg.has_arm = true;
            active_seg.has_aux = true;
            active_seg.duration_s = std::max(2.2, max_diff * 1.875 / 0.15);
            seg_start = Clock::now();
            has_active_seg = true;
            ctrl_mode = Mode::RETURN_HOME_FOR_DISENGAGE;
            stage_label = "returning_nominal";
        } else {
            ctrl_mode = Mode::RAMP_DOWN;
            stage_label = "releasing";
        }
    };

    const auto dt_sleep = std::chrono::milliseconds(4); // 250 Hz
    Clock::time_point prev_tick = Clock::now();

    while (g_running) {
        auto now = Clock::now();
        double dt = std::chrono::duration<double>(now - prev_tick).count();
        prev_tick = now;
        dt = std::clamp(dt, 0.001, 0.05);

        LowState_ st{};
        BmsState_ bms{};
        bool low_fresh = false;
        bool got_bms = false;
        {
            std::lock_guard<std::mutex> lk(state_mtx);
            if (has_low) {
                st = latest_low;
                low_fresh = std::chrono::duration<double>(now - low_rx_time).count() < 0.25;
            }
            if (has_bms) {
                bms = latest_bms;
                got_bms = true;
            }
        }

        std::deque<std::string> pending;
        double hb_age = 0.0;
        {
            std::lock_guard<std::mutex> lk(cmd_mtx);
            pending.swap(cmd_queue);
            hb_age = std::chrono::duration<double>(now - last_heartbeat).count();
        }

        if (low_fresh) {
            std::array<double, 10> q_meas{};
            for (size_t i = 0; i < 10; ++i) q_meas[i] = st.motor_state()[ARM_JOINTS[i]].q();
            compute_arm_gravity_tau(
                q_meas,
                st.imu_state().rpy()[0],
                st.imu_state().rpy()[1],
                st.motor_state()[12].q(),
                st.motor_state()[AUX_JOINTS[0]].q(),
                extra_hand_mass_kg,
                cur_tau_grav
            );
            for (size_t i = 0; i < 10; ++i) {
                cur_tau_ext[i] = st.motor_state()[ARM_JOINTS[i]].tau_est() - cur_tau_grav[i];
            }
        }

        for (const auto& line : pending) {
            std::string cmd = parse_string_field(line, "cmd");
            std::string req_id = parse_string_field(line, "id");
            if (cmd == "engage") {
                if (!low_fresh) {
                    emit_json("{\"event\":\"reply\",\"id\":\"" + req_id + "\",\"ok\":false,\"error\":\"LowState stale\"}");
                    continue;
                }
                double want_susp = 0.0;
                parse_double_field(line, "suspended", want_susp);
                parse_double_field(line, "table_z_floor", table_z_floor);
                if (parse_double_field(line, "grav_ff_scale", grav_ff_scale)) {
                    grav_ff_scale = std::clamp(grav_ff_scale, 0.0, 1.0);
                }
                if (parse_double_field(line, "extra_hand_mass_kg", extra_hand_mass_kg)) {
                    extra_hand_mass_kg = std::clamp(extra_hand_mass_kg, 0.0, 0.60);
                }
                grav_ff_enabled = (grav_ff_scale > 1e-4);

                if (want_susp > 0.5) {
                    // Harness-suspended Developer Mode: release built-in leg controller so legs have strictly 0 Nm!
                    std::string form, mname;
                    int chk_rc = switcher.CheckMode(form, mname);
                    if (chk_rc == 0 && !mname.empty()) {
                        int rel_rc = switcher.ReleaseMode();
                        if (rel_rc != 0) {
                            emit_json("{\"event\":\"reply\",\"id\":\"" + req_id + "\",\"ok\":false,\"error\":\"ReleaseMode failed rc=" + std::to_string(rel_rc) + "\"}");
                            continue;
                        }
                        for (int k = 0; k < 12; ++k) {
                            std::this_thread::sleep_for(std::chrono::milliseconds(150));
                            if (switcher.CheckMode(form, mname) == 0 && mname.empty()) break;
                        }
                    }
                    if (switcher.CheckMode(form, mname) != 0 || !mname.empty()) {
                        emit_json("{\"event\":\"reply\",\"id\":\"" + req_id + "\",\"ok\":false,\"error\":\"Built-in motion mode did not release (mode=" + mname + ")\"}");
                        continue;
                    }
                    // If currently IDLE, ensure no other process is publishing rt/lowcmd
                    if (ctrl_mode == Mode::IDLE) {
                        unsigned c0 = external_lowcmd_cnt.load(std::memory_order_relaxed);
                        std::this_thread::sleep_for(std::chrono::milliseconds(220));
                        unsigned c1 = external_lowcmd_cnt.load(std::memory_order_relaxed);
                        if (c1 != c0) {
                            emit_json("{\"event\":\"reply\",\"id\":\"" + req_id + "\",\"ok\":false,\"error\":\"Another rt/lowcmd publisher is active\"}");
                            continue;
                        }
                    }
                    // Refresh LowState AFTER ReleaseMode settle so initial cmd_arm has 0.000 rad step!
                    {
                        std::lock_guard<std::mutex> lk(state_mtx);
                        if (has_low) {
                            st = latest_low;
                            low_fresh = std::chrono::duration<double>(Clock::now() - low_rx_time).count() < 0.25;
                        }
                    }
                    if (!low_fresh) {
                        emit_json("{\"event\":\"reply\",\"id\":\"" + req_id + "\",\"ok\":false,\"error\":\"LowState stale after ReleaseMode\"}");
                        continue;
                    }
                    suspended_mode = true;
                } else {
                    int fsm = cached_fsm.load();
                    double roll = st.imu_state().rpy()[0];
                    double pitch = st.imu_state().rpy()[1];
                    if (st.mode_machine() != 1 || std::abs(roll) > IMU_FREEZE_ROLL || std::abs(pitch) > IMU_FREEZE_PITCH) {
                        emit_json("{\"event\":\"reply\",\"id\":\"" + req_id + "\",\"ok\":false,\"error\":\"Unsafe initial IMU tilt\"}");
                        continue;
                    }
                    if (fsm != 811 && fsm != 816) {
                        emit_json("{\"event\":\"reply\",\"id\":\"" + req_id + "\",\"ok\":false,\"error\":\"R1 must be in FSM 811 (or use suspended mode), current=" + std::to_string(fsm) + "\"}");
                        continue;
                    }
                    suspended_mode = false;
                }

                if (ctrl_mode == Mode::IDLE || ctrl_mode == Mode::SAFE_HOLD || ctrl_mode == Mode::SUSPENDED_VISCOUS_STOP) {
                    for (size_t i = 0; i < 10; ++i) cmd_arm[i] = st.motor_state()[ARM_JOINTS[i]].q();
                    for (size_t i = 0; i < 3; ++i) cmd_aux[i] = st.motor_state()[AUX_JOINTS[i]].q();
                    if (ctrl_mode == Mode::IDLE || ctrl_mode == Mode::SUSPENDED_VISCOUS_STOP) {
                        nominal_home_arm = cmd_arm;
                        nominal_home_aux = cmd_aux;
                        weight = 0.0;
                        ctrl_mode = Mode::RAMP_UP;
                        stage_label = "engaging";
                    } else {
                        ctrl_mode = Mode::ACTIVE;
                        stage_label = "holding";
                    }
                    last_fault.clear();
                    for (auto& tp : track_err_since) tp = Clock::time_point{};
                }
                emit_json("{\"event\":\"reply\",\"id\":\"" + req_id + "\",\"ok\":true}");
            } else if (cmd == "disengage") {
                if (low_fresh && ctrl_mode != Mode::IDLE && ctrl_mode != Mode::RAMP_DOWN) {
                    begin_safe_disengage(st);
                } else if (!low_fresh) {
                    ctrl_mode = Mode::IDLE;
                    weight = 0.0;
                    stage_label = "idle";
                }
                emit_json("{\"event\":\"reply\",\"id\":\"" + req_id + "\",\"ok\":true}");
            } else if (cmd == "estop") {
                // Immediate passive damping across whole robot (prevents any leg thrashing!)
                trigger_emergency_damp("ESTOP");
                emit_json("{\"event\":\"reply\",\"id\":\"" + req_id + "\",\"ok\":true}");
            } else if (cmd == "hold_here") {
                if (low_fresh && ctrl_mode != Mode::IDLE) {
                    trigger_compliant_freeze("hold_here", st);
                    last_fault.clear();
                    ctrl_mode = Mode::ACTIVE;
                    stage_label = "holding";
                }
                emit_json("{\"event\":\"reply\",\"id\":\"" + req_id + "\",\"ok\":true}");
            } else if (cmd == "set_params") {
                parse_double_field(line, "table_z_floor", table_z_floor);
                if (parse_double_field(line, "grav_ff_scale", grav_ff_scale)) {
                    grav_ff_scale = std::clamp(grav_ff_scale, 0.0, 1.0);
                    grav_ff_enabled = (grav_ff_scale > 1e-4);
                }
                if (parse_double_field(line, "extra_hand_mass_kg", extra_hand_mass_kg)) {
                    extra_hand_mass_kg = std::clamp(extra_hand_mass_kg, 0.0, 0.60);
                }
                emit_json("{\"event\":\"reply\",\"id\":\"" + req_id + "\",\"ok\":true}");
            } else if (cmd == "move") {
                if (ctrl_mode != Mode::ACTIVE && ctrl_mode != Mode::RAMP_UP) {
                    emit_json("{\"event\":\"reply\",\"id\":\"" + req_id + "\",\"ok\":false,\"error\":\"Arm daemon not active (stage=" + stage_label + ")\"}");
                    continue;
                }
                parse_double_field(line, "table_z_floor", table_z_floor);
                Segment seg{};
                std::vector<double> arm_v, aux_v;
                if (parse_double_array(line, "q", arm_v) && arm_v.size() == 10) {
                    for (size_t i = 0; i < 10; ++i) {
                        seg.arm_to[i] = std::clamp(arm_v[i], ARM_MIN[i], ARM_MAX[i]);
                    }
                    seg.has_arm = true;
                }
                if (parse_double_array(line, "aux", aux_v) && aux_v.size() == 3) {
                    for (size_t i = 0; i < 3; ++i) {
                        seg.aux_to[i] = std::clamp(aux_v[i], AUX_MIN[i], AUX_MAX[i]);
                    }
                    seg.has_aux = true;
                }
                if (seg.has_arm) {
                    std::string vw_err;
                    if (!check_virtual_wall(seg.arm_to, table_z_floor, vw_err)) {
                        emit_json("{\"event\":\"reply\",\"id\":\"" + req_id + "\",\"ok\":false,\"error\":\"Target rejected by virtual wall: " + json_escape(vw_err) + "\"}");
                        continue;
                    }
                }
                double dur = 2.0;
                parse_double_field(line, "duration", dur);
                seg.duration_s = std::clamp(dur, 0.40, 20.0);
                std::string lbl = parse_string_field(line, "label");
                if (!lbl.empty()) seg.label = lbl;

                double clear_prev = 0.0;
                if (parse_double_field(line, "replace", clear_prev) && clear_prev > 0.5) {
                    seg_queue.clear();
                    has_active_seg = false;
                }
                seg_queue.push_back(seg);
                emit_json("{\"event\":\"reply\",\"id\":\"" + req_id + "\",\"ok\":true}");
            } else if (cmd == "move_path") {
                if (ctrl_mode != Mode::ACTIVE && ctrl_mode != Mode::RAMP_UP) {
                    emit_json("{\"event\":\"reply\",\"id\":\"" + req_id + "\",\"ok\":false,\"error\":\"Arm daemon not active (stage=" + stage_label + ")\"}");
                    continue;
                }
                parse_double_field(line, "table_z_floor", table_z_floor);
                std::vector<double> flat_v, aux_v;
                if (!parse_double_array(line, "q_path", flat_v) || flat_v.size() < 10 || (flat_v.size() % 10) != 0) {
                    emit_json("{\"event\":\"reply\",\"id\":\"" + req_id + "\",\"ok\":false,\"error\":\"Invalid q_path array (must be multiple of 10)\"}");
                    continue;
                }
                size_t n_pts = flat_v.size() / 10;
                Segment seg{};
                seg.arm_path.reserve(n_pts);
                std::string vw_err;
                bool ok_path = true;
                for (size_t k = 0; k < n_pts; ++k) {
                    std::array<double, 10> wp{};
                    for (size_t i = 0; i < 10; ++i) {
                        wp[i] = std::clamp(flat_v[k * 10 + i], ARM_MIN[i], ARM_MAX[i]);
                    }
                    if (!check_virtual_wall(wp, table_z_floor, vw_err)) {
                        ok_path = false;
                        break;
                    }
                    seg.arm_path.push_back(wp);
                }
                if (!ok_path) {
                    emit_json("{\"event\":\"reply\",\"id\":\"" + req_id + "\",\"ok\":false,\"error\":\"Path waypoint rejected by virtual wall: " + json_escape(vw_err) + "\"}");
                    continue;
                }
                seg.arm_to = seg.arm_path.back();
                seg.has_arm = true;
                if (parse_double_array(line, "aux", aux_v) && aux_v.size() == 3) {
                    for (size_t i = 0; i < 3; ++i) {
                        seg.aux_to[i] = std::clamp(aux_v[i], AUX_MIN[i], AUX_MAX[i]);
                    }
                    seg.has_aux = true;
                }
                double dur = 2.0;
                parse_double_field(line, "duration", dur);
                seg.duration_s = std::clamp(dur, 0.35, 25.0);
                std::string lbl = parse_string_field(line, "label");
                if (!lbl.empty()) seg.label = lbl;
                else seg.label = "move_path";

                double clear_prev = 0.0;
                if (parse_double_field(line, "replace", clear_prev) && clear_prev > 0.5) {
                    seg_queue.clear();
                    has_active_seg = false;
                }
                seg_queue.push_back(seg);
                emit_json("{\"event\":\"reply\",\"id\":\"" + req_id + "\",\"ok\":true}");
            } else if (cmd == "enable_stereo") {
                want_enable_stereo.store(true);
                emit_json("{\"event\":\"reply\",\"id\":\"" + req_id + "\",\"ok\":true}");
            }
        }

        // Heartbeat watchdog: freeze gently in place (do NOT drop to FSM 811 abruptly!)
        if ((ctrl_mode == Mode::ACTIVE || ctrl_mode == Mode::RAMP_UP) && hb_age > 4.0 && low_fresh) {
            trigger_compliant_freeze("Bridge heartbeat timeout", st);
        }

        // Real-time 250 Hz control step
        if (ctrl_mode != Mode::IDLE) {
            if (!low_fresh) {
                ctrl_mode = Mode::IDLE;
                weight = 0.0;
                stage_label = "fault_stale_lowstate";
                emit_json("{\"event\":\"arm_fault\",\"error\":\"LowState stale during active control\"}");
            } else {
                const auto& rc = st.wireless_remote();
                unsigned keys = unsigned(rc[2]) | (unsigned(rc[3]) << 8);
                if ((keys & 0x220) == 0x220) {
                    trigger_emergency_damp("Wireless remote L2+B emergency damp");
                }

                // Stage-2 Catastrophic IMU tilt check (standing mode only) -> instant Damp() so legs NEVER thrash
                if (!suspended_mode && ctrl_mode != Mode::IDLE) {
                    double r = st.imu_state().rpy()[0], p = st.imu_state().rpy()[1];
                    if (!std::isfinite(r) || !std::isfinite(p) || std::abs(r) > IMU_DAMP_ROLL || std::abs(p) > IMU_DAMP_PITCH) {
                        trigger_emergency_damp("IMU tilt exceeded safe standing limit -> Damp()");
                    }
                }

                // Stage-1 Active motion guards -> compliant freeze in place (keeps arm_sdk engaged!)
                if (ctrl_mode == Mode::ACTIVE || ctrl_mode == Mode::RAMP_UP || ctrl_mode == Mode::RETURN_HOME_FOR_DISENGAGE) {
                    bool motion_safe = true;
                    std::string err_msg;
                    if (st.mode_machine() != 1) {
                        motion_safe = false; err_msg = "mode_machine != 1";
                    }
                    if (!suspended_mode) {
                        double r = st.imu_state().rpy()[0], p = st.imu_state().rpy()[1];
                        double gx = st.imu_state().gyroscope()[0], gy = st.imu_state().gyroscope()[1];
                        if (std::abs(r) > IMU_FREEZE_ROLL || std::abs(p) > IMU_FREEZE_PITCH || std::hypot(gx, gy) > 0.95) {
                            motion_safe = false; err_msg = "Early IMU balance disturbance -> compliant arm freeze";
                        }
                    }
                    for (size_t i = 0; i < 10 && motion_safe; ++i) {
                        const auto& m = st.motor_state()[ARM_JOINTS[i]];
                        if (!std::isfinite(m.q()) || !std::isfinite(m.dq()) || !std::isfinite(m.tau_est()) || m.motorstate() != 0) {
                            motion_safe = false; err_msg = "Arm motor state fault on joint " + std::to_string(ARM_JOINTS[i]);
                        } else if (m.temperature()[0] >= 60 || m.temperature()[1] >= 60) {
                            motion_safe = false; err_msg = "Arm motor temperature >= 60C";
                        } else if (std::abs(m.dq()) > MAX_SAFE_MEASURED_DQ) {
                            // Abnormal joint velocity spike -> immediate viscous damping in suspended mode!
                            if (suspended_mode) {
                                trigger_emergency_damp("Abnormal joint velocity dq=" + std::to_string(m.dq()).substr(0, 5) + " on joint " + std::to_string(ARM_JOINTS[i]));
                                motion_safe = false; err_msg = "";
                            } else {
                                motion_safe = false; err_msg = "Abnormal joint velocity on joint " + std::to_string(ARM_JOINTS[i]);
                            }
                        } else if (std::abs(m.tau_est()) > TORQUE_LIMIT[i]) {
                            motion_safe = false; err_msg = "Torque guard (" + std::to_string(m.tau_est()).substr(0, 5) + "Nm) on joint " + std::to_string(ARM_JOINTS[i]);
                        } else if (weight > 0.98 && std::abs(m.q() - cmd_arm[i]) > 0.38) {
                            motion_safe = false; err_msg = "Immediate position deviation guard on joint " + std::to_string(ARM_JOINTS[i]);
                        } else if (weight > 0.98 && std::abs(m.q() - cmd_arm[i]) > 0.24) {
                            if (track_err_since[i] == Clock::time_point{}) track_err_since[i] = now;
                            if (std::chrono::duration<double>(now - track_err_since[i]).count() > 0.25) {
                                motion_safe = false; err_msg = "Obstacle tracking guard on joint " + std::to_string(ARM_JOINTS[i]);
                            }
                        } else {
                            track_err_since[i] = Clock::time_point{};
                        }
                    }
                    if (!motion_safe && !err_msg.empty()) {
                        trigger_compliant_freeze(err_msg, st);
                    }
                }

                if (ctrl_mode == Mode::RAMP_UP) {
                    weight = std::min(1.0, weight + dt / 2.5);
                    if (grav_ff_enabled && weight > 0.55) {
                        for (size_t i = 0; i < 10; ++i) {
                            double dq_err = st.motor_state()[ARM_JOINTS[i]].q() - cmd_arm[i];
                            double ff_i = grav_ff_scale * weight * cur_tau_grav[i];
                            if (std::abs(dq_err) > 0.11 && (ff_i * dq_err) > 0.02) {
                                grav_ff_enabled = false;
                                emit_json("{\"event\":\"grav_ff_auto_disabled\",\"joint\":" + std::to_string(ARM_JOINTS[i]) + "}");
                                break;
                            }
                        }
                    }
                    if (weight >= 1.0) {
                        ctrl_mode = Mode::ACTIVE;
                        stage_label = "holding";
                        emit_json("{\"event\":\"arm_engaged\",\"grav_ff\":" + std::string(grav_ff_enabled ? "true" : "false") + "}");
                    }
                } else if (ctrl_mode == Mode::ACTIVE) {
                    weight = 1.0;
                    if (!has_active_seg && !seg_queue.empty()) {
                        active_seg = seg_queue.front();
                        seg_queue.pop_front();
                        seg_arm_from = cmd_arm;
                        seg_aux_from = cmd_aux;
                        if (!active_seg.has_arm) active_seg.arm_to = cmd_arm;
                        if (!active_seg.has_aux) active_seg.aux_to = cmd_aux;

                        double max_delta = 0.0;
                        if (!active_seg.arm_path.empty()) {
                            std::vector<std::array<double, 10>> pts;
                            pts.reserve(active_seg.arm_path.size() + 1);
                            pts.push_back(seg_arm_from);
                            size_t start_k = 0;
                            if (active_seg.arm_path.size() >= 2) {
                                double d0 = 0.0;
                                for (size_t i = 0; i < 10; ++i) d0 = std::max(d0, std::abs(active_seg.arm_path[0][i] - seg_arm_from[i]));
                                if (d0 < 0.006) start_k = 1;
                            }
                            for (size_t k = start_k; k < active_seg.arm_path.size(); ++k) {
                                pts.push_back(active_seg.arm_path[k]);
                            }
                            std::array<double, 10> joint_path{};
                            std::vector<double> cum(pts.size(), 0.0);
                            for (size_t k = 1; k < pts.size(); ++k) {
                                double sq = 0.0;
                                for (size_t i = 0; i < 10; ++i) {
                                    double d = pts[k][i] - pts[k - 1][i];
                                    sq += d * d;
                                    joint_path[i] += std::abs(d);
                                }
                                cum[k] = cum[k - 1] + std::max(1e-5, std::sqrt(sq));
                            }
                            double total_arc = std::max(1e-5, cum.back());
                            active_seg.path_cum_s.resize(pts.size());
                            for (size_t k = 0; k < pts.size(); ++k) active_seg.path_cum_s[k] = cum[k] / total_arc;
                            active_seg.arm_path = std::move(pts);
                            active_seg.arm_to = active_seg.arm_path.back();
                            for (size_t i = 0; i < 10; ++i) max_delta = std::max(max_delta, joint_path[i]);
                        } else {
                            for (size_t i = 0; i < 10; ++i) max_delta = std::max(max_delta, std::abs(active_seg.arm_to[i] - seg_arm_from[i]));
                        }
                        for (size_t i = 0; i < 3; ++i) max_delta = std::max(max_delta, std::abs(active_seg.aux_to[i] - seg_aux_from[i]));
                        double min_dur = max_delta * 1.875 / MAX_JOINT_SPEED_RAD_S;
                        double dur_floor = active_seg.arm_path.empty() ? 0.60 : 0.35;
                        active_seg.duration_s = std::max(active_seg.duration_s, std::max(dur_floor, min_dur));

                        seg_start = now;
                        has_active_seg = true;
                        stage_label = active_seg.label;
                        emit_json("{\"event\":\"segment_started\",\"label\":\"" + json_escape(active_seg.label) + "\",\"duration\":" + std::to_string(active_seg.duration_s) + "}");
                    }
                    if (has_active_seg) {
                        double elapsed = std::chrono::duration<double>(now - seg_start).count();
                        double u = std::min(1.0, elapsed / active_seg.duration_s);
                        double s = quintic(u);
                        std::array<double, 10> next_arm{};
                        if (!active_seg.arm_path.empty() && active_seg.path_cum_s.size() == active_seg.arm_path.size() && active_seg.arm_path.size() >= 2) {
                            auto it = std::lower_bound(active_seg.path_cum_s.begin() + 1, active_seg.path_cum_s.end(), s);
                            size_t k = std::min<size_t>(active_seg.arm_path.size() - 1, std::max<size_t>(1, size_t(it - active_seg.path_cum_s.begin())));
                            double s0 = active_seg.path_cum_s[k - 1];
                            double s1 = active_seg.path_cum_s[k];
                            double alpha = std::clamp((s - s0) / std::max(1e-7, s1 - s0), 0.0, 1.0);
                            for (size_t i = 0; i < 10; ++i) {
                                next_arm[i] = active_seg.arm_path[k - 1][i] + (active_seg.arm_path[k][i] - active_seg.arm_path[k - 1][i]) * alpha;
                            }
                        } else {
                            for (size_t i = 0; i < 10; ++i) {
                                next_arm[i] = seg_arm_from[i] + (active_seg.arm_to[i] - seg_arm_from[i]) * s;
                            }
                        }
                        // Real-time 250 Hz Cartesian virtual wall check on interpolated command!
                        std::string vw_err;
                        if (!check_virtual_wall(next_arm, table_z_floor, vw_err)) {
                            trigger_compliant_freeze(vw_err, st);
                        } else {
                            cmd_arm = next_arm;
                            for (size_t i = 0; i < 3; ++i) {
                                cmd_aux[i] = seg_aux_from[i] + (active_seg.aux_to[i] - seg_aux_from[i]) * s;
                            }
                            if (u >= 1.0) {
                                has_active_seg = false;
                                if (seg_queue.empty()) {
                                    stage_label = "holding";
                                    emit_json("{\"event\":\"motion_complete\",\"label\":\"" + json_escape(active_seg.label) + "\"}");
                                }
                            }
                        }
                    }
                } else if (ctrl_mode == Mode::RETURN_HOME_FOR_DISENGAGE) {
                    weight = 1.0;
                    double elapsed = std::chrono::duration<double>(now - seg_start).count();
                    double u = std::min(1.0, elapsed / active_seg.duration_s);
                    double s = quintic(u);
                    for (size_t i = 0; i < 10; ++i) cmd_arm[i] = seg_arm_from[i] + (active_seg.arm_to[i] - seg_arm_from[i]) * s;
                    for (size_t i = 0; i < 3; ++i) cmd_aux[i] = seg_aux_from[i] + (active_seg.aux_to[i] - seg_aux_from[i]) * s;
                    if (u >= 1.0) {
                        has_active_seg = false;
                        ctrl_mode = Mode::RAMP_DOWN;
                        stage_label = "releasing";
                    }
                } else if (ctrl_mode == Mode::SAFE_HOLD) {
                    weight = 1.0; // Stay engaged in feather-soft compliance so RL never whips the arms!
                } else if (ctrl_mode == Mode::SUSPENDED_VISCOUS_STOP) {
                    weight = 0.0; // kp=0, kd=3.2 pure passive viscous damping in suspended mode
                    for (size_t i = 0; i < 10; ++i) cmd_arm[i] = st.motor_state()[ARM_JOINTS[i]].q();
                    for (size_t i = 0; i < 3; ++i) cmd_aux[i] = st.motor_state()[AUX_JOINTS[i]].q();
                } else if (ctrl_mode == Mode::RAMP_DOWN) {
                    weight = std::max(0.0, weight - dt / 2.5);
                    if (weight <= 0.0) {
                        LowCmd_ msg{};
                        msg.mode_machine() = st.mode_machine();
                        msg.mode_pr() = 0;
                        msg.crc() = crc32_core(reinterpret_cast<uint32_t*>(&msg), (sizeof(msg) >> 2) - 1);
                        if (suspended_mode) pub_low.Write(msg);
                        else pub_arm.Write(msg);
                        ctrl_mode = Mode::IDLE;
                        stage_label = "idle";
                        emit_json("{\"event\":\"arm_disengaged\"}");
                    }
                }

                if (ctrl_mode != Mode::IDLE) {
                    LowCmd_ msg{};
                    msg.mode_machine() = st.mode_machine();
                    msg.mode_pr() = suspended_mode ? 0 : std::clamp(int(std::round(weight * 100.0)), 0, 100);
                    const auto& kp_arr = (ctrl_mode == Mode::SAFE_HOLD) ? ARM_KP_SOFT : ARM_KP;
                    const auto& kd_arr = (ctrl_mode == Mode::SAFE_HOLD || ctrl_mode == Mode::SUSPENDED_VISCOUS_STOP) ? ARM_KD_SOFT : ARM_KD;
                    double gain_scale = (ctrl_mode == Mode::SUSPENDED_VISCOUS_STOP) ? 0.0 : (suspended_mode ? weight : 1.0);
                    double eff_ff = (grav_ff_enabled && ctrl_mode != Mode::SUSPENDED_VISCOUS_STOP) ? (grav_ff_scale * weight) : 0.0;
                    for (size_t i = 0; i < 10; ++i) {
                        auto& m = msg.motor_cmd()[ARM_JOINTS[i]];
                        double ff_lim = ((i % 5) == 4) ? 1.8 : 4.8;
                        m.mode() = 1;
                        m.q() = cmd_arm[i];
                        m.dq() = 0.0;
                        m.tau() = std::clamp(eff_ff * cur_tau_grav[i], -ff_lim, ff_lim);
                        m.kp() = kp_arr[i] * gain_scale;
                        m.kd() = kd_arr[i];
                    }
                    for (size_t i = 0; i < 3; ++i) {
                        auto& m = msg.motor_cmd()[AUX_JOINTS[i]];
                        m.mode() = 1;
                        m.q() = cmd_aux[i];
                        m.dq() = 0.0;
                        m.tau() = 0.0;
                        m.kp() = ((i == 0) ? 35.0 : 12.0) * gain_scale;
                        m.kd() = (i == 0) ? 2.5 : 1.0;
                    }
                    msg.crc() = crc32_core(reinterpret_cast<uint32_t*>(&msg), (sizeof(msg) >> 2) - 1);
                    if (suspended_mode) pub_low.Write(msg);
                    else pub_arm.Write(msg);
                }
            }
        }

        // Emit telemetry at 10 Hz
        if (std::chrono::duration<double>(now - last_telemetry).count() >= 0.10) {
            last_telemetry = now;
            std::ostringstream oss;
            oss.precision(4);
            oss << std::fixed;
            bool is_eng = (ctrl_mode == Mode::ACTIVE || ctrl_mode == Mode::RAMP_UP ||
                           ctrl_mode == Mode::SAFE_HOLD || ctrl_mode == Mode::RETURN_HOME_FOR_DISENGAGE);
            oss << "{\"event\":\"robot_state\",\"low_fresh\":" << (low_fresh ? "true" : "false")
                << ",\"engaged\":" << (is_eng ? "true" : "false")
                << ",\"suspended\":" << (suspended_mode ? "true" : "false")
                << ",\"weight\":" << weight
                << ",\"grav_ff_gain\":" << (grav_ff_enabled ? grav_ff_scale : 0.0)
                << ",\"table_z_floor\":" << table_z_floor
                << ",\"stage\":\"" << json_escape(stage_label) << "\""
                << ",\"last_fault\":\"" << json_escape(last_fault) << "\""
                << ",\"fsm_id\":" << cached_fsm.load();
            {
                std::lock_guard<std::mutex> lk(mode_mtx);
                oss << ",\"mode_name\":\"" << json_escape(cached_mode_name) << "\"";
            }
            if (got_bms) {
                double v_sum = 0.0;
                for (auto v : bms.bmsvoltage()) v_sum += v;
                oss << ",\"bms\":{\"soc\":" << int(bms.soc())
                    << ",\"current_mA\":" << bms.current()
                    << ",\"voltage_V\":" << (v_sum / 1000.0) << "}";
            }
            if (low_fresh) {
                oss << ",\"mode_machine\":" << int(st.mode_machine())
                    << ",\"rpy\":[" << st.imu_state().rpy()[0] << "," << st.imu_state().rpy()[1] << "," << st.imu_state().rpy()[2] << "]"
                    << ",\"q_arm\":[";
                for (size_t i = 0; i < 10; ++i) {
                    if (i) oss << ",";
                    oss << st.motor_state()[ARM_JOINTS[i]].q();
                }
                oss << "],\"cmd_arm\":[";
                for (size_t i = 0; i < 10; ++i) {
                    if (i) oss << ",";
                    oss << cmd_arm[i];
                }
                oss << "],\"tau_arm\":[";
                for (size_t i = 0; i < 10; ++i) {
                    if (i) oss << ",";
                    oss << st.motor_state()[ARM_JOINTS[i]].tau_est();
                }
                oss << "],\"tau_grav\":[";
                for (size_t i = 0; i < 10; ++i) {
                    if (i) oss << ",";
                    oss << cur_tau_grav[i];
                }
                oss << "],\"tau_ext\":[";
                for (size_t i = 0; i < 10; ++i) {
                    if (i) oss << ",";
                    oss << cur_tau_ext[i];
                }
                oss << "],\"dq_arm\":[";
                int max_temp = 0;
                for (size_t i = 0; i < 10; ++i) {
                    if (i) oss << ",";
                    oss << st.motor_state()[ARM_JOINTS[i]].dq();
                    max_temp = std::max(max_temp, int(st.motor_state()[ARM_JOINTS[i]].temperature()[0]));
                    max_temp = std::max(max_temp, int(st.motor_state()[ARM_JOINTS[i]].temperature()[1]));
                }
                oss << "],\"max_temp\":" << max_temp << ",\"q_aux\":[";
                for (size_t i = 0; i < 3; ++i) {
                    if (i) oss << ",";
                    oss << st.motor_state()[AUX_JOINTS[i]].q();
                }
                oss << "],\"q_waist_roll\":" << st.motor_state()[12].q();
            }
            oss << "}";
            emit_json(oss.str());
        }

        std::this_thread::sleep_until(now + dt_sleep);
    }

    if (mode_thread.joinable()) mode_thread.join();
    if (stdin_thread.joinable()) stdin_thread.detach();
    return 0;
}
