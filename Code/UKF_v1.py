import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import io
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


# ---------------------------------------------------------------------------
# Quaternion helpers
# ---------------------------------------------------------------------------
def quat_normalize(q):
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return q / n


def quat_multiply(q1, q2):
    q1 = quat_normalize(q1)
    q2 = quat_normalize(q2)
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return quat_normalize(
        np.array(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dtype=float,
        )
    )


def quat_conjugate(q):
    q = quat_normalize(q)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


def quat_inv(q):
    return quat_conjugate(q)


def quat_from_rotvec(rv):
    rv = np.asarray(rv, dtype=float)
    if np.linalg.norm(rv) < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    r = R.from_rotvec(rv)
    x, y, z, w = r.as_quat()
    return np.array([w, x, y, z], dtype=float)


def rotvec_from_quat(q):
    q = quat_normalize(q)
    w, x, y, z = q
    r = R.from_quat([x, y, z, w])
    return r.as_rotvec()


def quat_to_matrix(q):
    q = quat_normalize(q)
    w, x, y, z = q
    return R.from_quat([x, y, z, w]).as_matrix()


def quat_to_euler_zyx(q):
    q = quat_normalize(q)
    w, x, y, z = q
    return R.from_quat([x, y, z, w]).as_euler("ZYX")


def normalize_vec(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n


# ---------------------------------------------------------------------------
# Data loading and synchronization
# ---------------------------------------------------------------------------
def project_to_so3(matrix):
    U, _, Vt = np.linalg.svd(matrix)
    Rm = U @ Vt
    if np.linalg.det(Rm) < 0:
        U[:, -1] *= -1.0
        Rm = U @ Vt
    return Rm


def load_calibrate_and_sync(mat_number):
    base = Path(__file__).resolve().parent
    imu_path = base / "Data" / "Train" / "IMU" / f"imuRaw{mat_number}.mat"
    vicon_path = base / "Data" / "Train" / "Vicon" / f"viconRot{mat_number}.mat"
    params_path = base / "IMUParams.mat"

    imu = io.loadmat(str(imu_path))
    vicon = io.loadmat(str(vicon_path))
    params = io.loadmat(str(params_path))

    vals = np.asarray(imu["vals"], dtype=float)
    if vals.shape[0] != 6 and vals.shape[1] == 6:
        vals = vals.T
    t_imu = np.asarray(imu["ts"], dtype=float).ravel()

    t_vicon = np.asarray(vicon["ts"], dtype=float).ravel()
    rots = np.asarray(vicon["rots"], dtype=float)
    if rots.ndim == 3 and rots.shape[:2] == (3, 3):
        rots = np.transpose(rots, (2, 0, 1))

    valid_vicon = np.isfinite(t_vicon) & np.isfinite(rots).all(axis=(1, 2))
    t_vicon = t_vicon[valid_vicon]
    rots = rots[valid_vicon]

    order = np.argsort(t_vicon)
    t_vicon = t_vicon[order]
    rots = rots[order]
    t_vicon, unique = np.unique(t_vicon, return_index=True)
    rots = rots[unique]
    rots = np.stack([project_to_so3(r) for r in rots])

    overlap = (np.isfinite(t_imu)) & (t_imu >= t_vicon[0]) & (t_imu <= t_vicon[-1])
    t_imu = t_imu[overlap]
    vals = vals[:, overlap]

    vicon_rotation = Slerp(t_vicon, R.from_matrix(rots))(t_imu)

    params_mat = np.asarray(params["IMUParams"], dtype=float)
    scale = params_mat[0]
    bias = params_mat[1]

    ax = (vals[0] * scale[0] + bias[0]) * 9.81
    ay = (vals[1] * scale[1] + bias[1]) * 9.81
    az = (vals[2] * scale[2] + bias[2]) * 9.81

    gyro_bias = np.mean(vals[3:6, : min(100, vals.shape[1])], axis=1)
    gyro_scale = (3300.0 / 1023.0) * (np.pi / 180.0) * 0.3
    # Dataset channel order is [ax, ay, az, wz, wx, wy].
    wz = gyro_scale * (vals[3] - gyro_bias[0])
    wx = gyro_scale * (vals[4] - gyro_bias[1])
    wy = gyro_scale * (vals[5] - gyro_bias[2])

    return t_imu, ax, ay, az, wx, wy, wz, vicon_rotation


# ---------------------------------------------------------------------------
# Gyro-only and accelerometer-only baselines
# ---------------------------------------------------------------------------
def integrate_gyro(ts, wx, wy, wz, R0_3x3):
    ts = np.asarray(ts, dtype=float).ravel()
    wx = np.asarray(wx, dtype=float).ravel()
    wy = np.asarray(wy, dtype=float).ravel()
    wz = np.asarray(wz, dtype=float).ravel()

    if not (wx.size == wy.size == wz.size == ts.size):
        raise ValueError("ts, wx, wy, wz must have the same length")

    current = R.from_matrix(R0_3x3)
    mats = np.empty((ts.size, 3, 3), dtype=float)
    mats[0] = current.as_matrix()

    for k in range(ts.size - 1):
        dt = ts[k + 1] - ts[k]
        if dt <= 0:
            raise ValueError("IMU timestamps must be strictly increasing")
        body_rate = np.array([wx[k], wy[k], wz[k]], dtype=float)
        dR = R.from_rotvec(body_rate * dt)
        current = current * dR
        mats[k + 1] = current.as_matrix()
    return R.from_matrix(mats)


def accel_tilt_from_calibrated(ax, ay, az):
    ax = np.asarray(ax, dtype=float)
    ay = np.asarray(ay, dtype=float)
    az = np.asarray(az, dtype=float)
    roll = np.arctan2(ay, np.sqrt(ax ** 2 + az ** 2))
    pitch = np.arctan2(-ax, np.sqrt(ay ** 2 + az ** 2))
    yaw_like = np.arctan2(np.sqrt(ax ** 2 + ay ** 2), az)
    return roll, pitch, yaw_like


def lowpass(values, alpha=0.8):
    values = np.asarray(values, dtype=float)
    out = np.empty_like(values)
    out[0] = values[0]
    for k in range(1, values.size):
        out[k] = (1.0 - alpha) * values[k] + alpha * out[k - 1]
    return out


def complementary_filter_angles(t_imu, wx, wy, wz, ax, ay, az, R0_3x3):
    """Complementary filter using gyro propagation and accelerometer tilt estimate."""
    ts = np.asarray(t_imu, dtype=float).ravel()
    wx = np.asarray(wx, dtype=float).ravel()
    wy = np.asarray(wy, dtype=float).ravel()
    wz = np.asarray(wz, dtype=float).ravel()
    N = ts.size

    roll_acc, pitch_acc, yaw_acc = accel_tilt_from_calibrated(ax, ay, az)
    current_R = R.from_matrix(R0_3x3)
    comp_rpy = np.empty((N, 3), dtype=float)
    comp_rpy[0] = current_R.as_euler("ZYX", degrees=False)

    for k in range(N - 1):
        dt = ts[k + 1] - ts[k]
        if dt <= 0:
            raise ValueError("IMU timestamps must be strictly increasing")
        gyro_update = current_R * R.from_rotvec(np.array([wx[k], wy[k], wz[k]]) * dt)
        gyro_rpy = gyro_update.as_euler("ZYX", degrees=False)
        accel_rpy = np.array([yaw_acc[k], pitch_acc[k], roll_acc[k]], dtype=float)

        # Report-style complementary fusion: heavier weight on gyro, lighter weight on accel.
        comp_rpy[k + 1] = 0.98 * gyro_rpy + 0.02 * accel_rpy
        current_R = R.from_euler("ZYX", comp_rpy[k + 1])

    return comp_rpy


def madgwick_filter(ts, wx, wy, wz, ax, ay, az, R0_3x3):
    """Madgwick quaternion filter using the standard gravity-based correction."""
    ts = np.asarray(ts, dtype=float).ravel()
    wx = np.asarray(wx, dtype=float).ravel()
    wy = np.asarray(wy, dtype=float).ravel()
    wz = np.asarray(wz, dtype=float).ravel()
    ax = np.asarray(ax, dtype=float).ravel()
    ay = np.asarray(ay, dtype=float).ravel()
    az = np.asarray(az, dtype=float).ravel()
    N = ts.size

    beta = 0.06
    q = R.from_matrix(R0_3x3).as_quat()
    q = np.array([q[3], q[0], q[1], q[2]], dtype=float)  # [w, x, y, z]
    q_hist = np.zeros((N, 4), dtype=float)
    q_hist[0] = q

    for k in range(1, N):
        dt = ts[k] - ts[k - 1]
        if dt <= 0:
            raise ValueError("IMU timestamps must be strictly increasing")

        qw, qx, qy, qz = q
        a = np.array([ax[k], ay[k], az[k]], dtype=float)
        a_norm = np.linalg.norm(a)
        if a_norm > 1e-12:
            a = a / a_norm

        # Gravity-based objective function used in the report-style Madgwick implementation.
        f = np.array(
            [
                2.0 * (qx * qz - qw * qy) - a[0],
                2.0 * (qw * qx + qy * qz) - a[1],
                2.0 * (0.5 - qx ** 2 - qy ** 2) - a[2],
            ],
            dtype=float,
        )

        J = np.array(
            [
                [-2.0 * qy, 2.0 * qz, -2.0 * qw, 2.0 * qx],
                [2.0 * qx, 2.0 * qw, 2.0 * qz, 2.0 * qy],
                [0.0, -4.0 * qx, -4.0 * qy, 0.0],
            ],
            dtype=float,
        )
        grad = J.T @ f
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-12:
            grad = grad / grad_norm

        # Gyro quaternion rate.
        omega = np.array([wx[k - 1], wy[k - 1], wz[k - 1]], dtype=float)
        q_gyro = 0.5 * np.array(
            [
                -qx * omega[0] - qy * omega[1] - qz * omega[2],
                qw * omega[0] + qy * omega[2] - qz * omega[1],
                qw * omega[1] - qx * omega[2] + qz * omega[0],
                qw * omega[2] + qx * omega[1] - qy * omega[0],
            ],
            dtype=float,
        )

        qdot = q_gyro - beta * grad
        q = q + qdot * dt
        q = quat_normalize(q)
        q_hist[k] = q

    madgwick_rpy = np.zeros((N, 3), dtype=float)
    for i in range(N):
        q_i = q_hist[i]
        R_i = R.from_quat([q_i[1], q_i[2], q_i[3], q_i[0]])
        madgwick_rpy[i] = R_i.as_euler("ZYX", degrees=False)
    return madgwick_rpy


# ---------------------------------------------------------------------------
# UKF state and sigma-point math
# ---------------------------------------------------------------------------
def sigma_points_from_state(x_prev, P, Q):
    """Create 2n sigma points around x_prev, using the 6D perturbation
    vector [rotation-vector, angular-rate] and a 7D state [q, omega].
    """
    n = 6
    P_plus_Q = 0.5 * (P + P.T) + Q
    jitter = 1e-12
    for _ in range(8):
        try:
            L = np.linalg.cholesky(P_plus_Q + jitter * np.eye(6))
            break
        except np.linalg.LinAlgError:
            jitter *= 10.0
    else:
        values, vectors = np.linalg.eigh(P_plus_Q)
        L = vectors @ np.diag(np.sqrt(np.maximum(values, 1e-12)))

    # For 2n equally weighted sigma points, each Cholesky column is scaled
    # by sqrt(n), so their sample covariance is P + Q.
    L = np.sqrt(n) * L
    W = np.zeros((2 * n, 6), dtype=float)
    for i in range(n):
        W[i] = L[:, i]
        W[i + n] = -L[:, i]

    sigma = np.zeros((2 * n, 7), dtype=float)
    for i in range(2 * n):
        q_pert = quat_from_rotvec(W[i, 0:3])
        q_new = quat_multiply(q_pert, x_prev[0:4])
        omega_new = x_prev[4:7] + W[i, 3:6]
        sigma[i, 0:4] = q_new
        sigma[i, 4:7] = omega_new
    return sigma


def propagate_sigma_points(sigma, dt):
    """Constant-rate process model applied independently to each sigma point."""
    propagated = np.zeros_like(sigma)
    for i in range(sigma.shape[0]):
        q_old = sigma[i, 0:4]
        omega_i = sigma[i, 4:7]
        omega_norm = np.linalg.norm(omega_i)
        if omega_norm > 1e-12:
            axis = omega_i / omega_norm
            q_delta = quat_from_rotvec(axis * omega_norm * dt)
            q_new = quat_multiply(q_old, q_delta)
        else:
            q_new = q_old
        propagated[i, 0:4] = q_new
        propagated[i, 4:7] = sigma[i, 4:7]
    return propagated


def mean_quaternion(q_list):
    q_mean = quat_normalize(q_list[0])
    for _ in range(12):
        avg_err = np.zeros(3, dtype=float)
        for q in q_list:
            rel = quat_multiply(q, quat_inv(q_mean))
            avg_err += rotvec_from_quat(rel)
        avg_err /= len(q_list)
        if np.linalg.norm(avg_err) < 1e-8:
            break
        q_mean = quat_multiply(quat_from_rotvec(avg_err), q_mean)
        q_mean = quat_normalize(q_mean)
    return q_mean


def mean_state_from_sigma(Y):
    q_list = [Y[i, 0:4] for i in range(Y.shape[0])]
    q_bar = mean_quaternion(q_list)
    w_bar = np.mean(Y[:, 4:7], axis=0)
    x_bar = np.zeros(7, dtype=float)
    x_bar[0:4] = q_bar
    x_bar[4:7] = w_bar
    return x_bar


def measurement_model_from_sigma(Y, g_world=np.array([0.0, 0.0, 9.81])):
    """Predict the 6-D IMU measurement [body gravity, angular velocity]."""
    Z = np.zeros((Y.shape[0], 6), dtype=float)
    for i in range(Y.shape[0]):
        q = quat_normalize(Y[i, 0:4])
        Rb = quat_to_matrix(q)
        Z[i, 0:3] = Rb.T @ g_world
        Z[i, 3:6] = Y[i, 4:7]
    return Z


def compute_covariances(Y, x_bar, Z):
    """Return prior covariance, measurement covariance, and cross-covariance."""
    M = Y.shape[0]
    E = np.zeros((M, 6), dtype=float)
    for i in range(M):
        q_i = quat_normalize(Y[i, 0:4])
        q_bar = quat_normalize(x_bar[0:4])
        rel = quat_multiply(q_i, quat_inv(q_bar))
        r_err = rotvec_from_quat(rel)
        w_err = Y[i, 4:7] - x_bar[4:7]
        E[i, :] = np.concatenate([r_err, w_err])

    P_pred = (E.T @ E) / float(M)
    P_pred = 0.5 * (P_pred + P_pred.T)

    z_bar = np.mean(Z, axis=0)
    Zc = Z - z_bar[None, :]
    Pzz = (Zc.T @ Zc) / float(M)
    Pzz = 0.5 * (Pzz + Pzz.T)

    Pxz = (E.T @ Zc) / float(M)
    return P_pred, Pzz, Pxz


def boxplus_state(x, delta):
    q_new = quat_multiply(quat_from_rotvec(delta[0:3]), x[0:4])
    omega_new = x[4:7] + delta[3:6]
    return np.concatenate([quat_normalize(q_new), omega_new])


def make_covariance_psd(P, minimum=1e-12):
    """Symmetrize P and remove tiny negative eigenvalues from roundoff."""
    P = 0.5 * (P + P.T)
    values, vectors = np.linalg.eigh(P)
    return vectors @ np.diag(np.maximum(values, minimum)) @ vectors.T


# ---------------------------------------------------------------------------
# UKF estimator
# ---------------------------------------------------------------------------
def ukf_attitude_estimate(t_imu, ax, ay, az, wx, wy, wz, R0_3x3):
    q0 = R.from_matrix(R0_3x3).as_quat()
    q0_wxyz = np.array([q0[3], q0[0], q0[1], q0[2]], dtype=float)

    x_prev = np.concatenate([q0_wxyz, np.array([wx[0], wy[0], wz[0]], dtype=float)])

    # Process noise and measurement noise from the project report.
    Q = np.diag([1e-5, 1e-5, 1e-5, 0.1, 0.1, 0.1])
    Rm = np.diag([11.2, 11.2, 11.2, 0.01, 0.01, 0.01])
    P_prev = np.eye(6) * 1e-3

    n_steps = t_imu.size
    q_est = np.zeros((n_steps, 4), dtype=float)
    q_est[0] = q0_wxyz

    for k in range(1, n_steps):
        dt = max(t_imu[k] - t_imu[k - 1], 1e-6)
        sigma = sigma_points_from_state(x_prev, P_prev, Q)
        Y = propagate_sigma_points(sigma, dt)

        x_pred = mean_state_from_sigma(Y)
        Z = measurement_model_from_sigma(Y)

        z_pred = np.mean(Z, axis=0)
        z_meas = np.array(
            [ax[k], ay[k], az[k], wx[k], wy[k], wz[k]], dtype=float
        )

        P_pred, Pzz, Pxz = compute_covariances(Y, x_pred, Z)
        S = Pzz + Rm
        # Solve K S = Pxz instead of explicitly forming S^{-1}.
        K = np.linalg.solve(S.T, Pxz.T).T

        innovation = z_meas - z_pred
        delta = K @ innovation
        x_post = boxplus_state(x_pred, delta)
        P_post = P_pred - K @ S @ K.T
        P_post = make_covariance_psd(P_post)

        x_prev = x_post
        P_prev = P_post
        q_est[k] = x_prev[0:4]

    return q_est


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_attitude_summary(mat_number, t_imu, ax, ay, az, wx, wy, wz, vicon_rotation, ukf_q):
    out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    vicon_zyx = vicon_rotation.as_euler("ZYX")
    vicon_yaw, vicon_pitch, vicon_roll = vicon_zyx.T

    gyro_rotation = integrate_gyro(t_imu, wx, wy, wz, vicon_rotation[0].as_matrix())
    gyro_zyx = gyro_rotation.as_euler("ZYX")
    gyro_yaw, gyro_pitch, gyro_roll = gyro_zyx.T
    gyro_roll = np.unwrap(gyro_roll)
    gyro_pitch = np.unwrap(gyro_pitch)
    gyro_yaw = np.unwrap(gyro_yaw)

    accel_roll, accel_pitch, accel_yaw = accel_tilt_from_calibrated(ax, ay, az)
    accel_roll = lowpass(accel_roll, alpha=0.8)
    accel_pitch = lowpass(accel_pitch, alpha=0.8)
    accel_yaw = lowpass(accel_yaw, alpha=0.8)

    comp_rpy = complementary_filter_angles(t_imu, wx, wy, wz, ax, ay, az, vicon_rotation[0].as_matrix())
    comp_yaw, comp_pitch, comp_roll = comp_rpy.T

    madgwick_rpy = madgwick_filter(t_imu, wx, wy, wz, ax, ay, az, vicon_rotation[0].as_matrix())
    mad_yaw, mad_pitch, mad_roll = madgwick_rpy.T

    ukf_zyx = np.zeros((t_imu.size, 3), dtype=float)
    for i, q in enumerate(ukf_q):
        ukf_zyx[i] = quat_to_euler_zyx(q)
    ukf_yaw, ukf_pitch, ukf_roll = ukf_zyx.T

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    plot_specs = [
        ("Roll", vicon_roll, gyro_roll, accel_roll, comp_roll, mad_roll, ukf_roll),
        ("Pitch", vicon_pitch, gyro_pitch, accel_pitch, comp_pitch, mad_pitch, ukf_pitch),
        ("Yaw", vicon_yaw, gyro_yaw, accel_yaw, comp_yaw, mad_yaw, ukf_yaw),
    ]

    for ax_i, (name, vicon_angle, gyro_angle, accel_angle, comp_angle, mad_angle, ukf_angle) in zip(axes, plot_specs):
        ax_i.plot(t_imu, vicon_angle, label="Vicon", linewidth=1.8)
        ax_i.plot(t_imu, gyro_angle, label="Gyro-only", linewidth=1.5)
        ax_i.plot(t_imu, accel_angle, label="Accel-only", linewidth=1.5)
        ax_i.plot(t_imu, comp_angle, label="Complementary", linewidth=1.5)
        ax_i.plot(t_imu, mad_angle, label="Madgwick", linewidth=1.5)
        ax_i.plot(t_imu, ukf_angle, label="UKF", linewidth=1.5)
        ax_i.set_ylabel(f"{name} (rad)")
        ax_i.grid(True, alpha=0.3)
        ax_i.legend(loc="upper right")

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"Attitude Comparison (mat {mat_number}): Vicon, Gyro, Accel, Complementary, Madgwick, and UKF")
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])

    out_path = out_dir / f"ukf_attitude_mat{mat_number}.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved UKF comparison plot to: {out_path}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="UKF attitude estimation from IMU and Vicon data.")
    parser.add_argument("--mat_number", type=int, default=1, help="Train-set index to process (1..6).")
    args = parser.parse_args()

    t_imu, ax, ay, az, wx, wy, wz, vicon_rotation = load_calibrate_and_sync(args.mat_number)
    ukf_q = ukf_attitude_estimate(t_imu, ax, ay, az, wx, wy, wz, vicon_rotation[0].as_matrix())
    plot_attitude_summary(args.mat_number, t_imu, ax, ay, az, wx, wy, wz, vicon_rotation, ukf_q)

    print(f"Processed mat {args.mat_number} with {len(t_imu)} IMU samples.")


if __name__ == "__main__":
    main()
