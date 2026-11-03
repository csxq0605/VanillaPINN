import math
import numpy as np
import torch
import torch.nn as nn

def autodiff(y, x, order=1):
    """
    Compute the n-th order derivative of y with respect to x using autograd.
    
    Parameters:
    y (torch.Tensor): The output tensor.
    x (torch.Tensor): The input tensor with respect to which the derivative is computed.
    order (int): The order of the derivative.
    
    Returns:
    torch.Tensor: The n-th order derivative of y with respect to x.
    """
    if order == 1:
        grads = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=torch.ones_like(y),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        return grads
    else:
        return autodiff(autodiff(y, x, order - 1), x, 1)

def read_result_data(file_path, time, filename, encoding="utf-8"):
    """
    逐行读取文本文件（可无后缀）。
    - parse_numbers=False: 返回每行的原始字符串（去除首尾空白、跳过空行和以 # 或 // 开头的注释行）
    - parse_numbers=True : 按分隔符解析为浮点数矩阵（delimiter=None 时自动把空白/逗号都当分隔）

    Parameters:
    file_path (str): 文本文件路径（可无扩展名）
    parse_numbers (bool): 是否解析为数值
    delimiter (str|None): 分隔符；None 表示自动（空白或逗号）
    encoding (str): 文件编码

    Returns:
    np.ndarray: 行字符串数组（默认）或二维浮点数组（parse_numbers=True）
    """
    lines = []
    path = f"{file_path}/{time}/{filename}"
    with open(path, "r", encoding=encoding) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#") or s.startswith("//"):
                continue
            lines.append(s)

    idx = None
    for i, line in enumerate(lines):
        try:
            num = int(line)
            idx = i + 2
            break
        except ValueError:
            pass
    
    data = []
    for i in range(num):
        line = lines[i + idx]
        line = line[1: -1] if line.startswith("(") and line.endswith(")") else line
        data.append([float(x) for x in line.split()][: 2])

    return np.array(data, dtype=np.float64)

def read_points_data(file_path, filename, encoding="utf-8"):
    """
    逐行读取文本文件（可无后缀）。
    - parse_numbers=False: 返回每行的原始字符串（去除首尾空白、跳过空行和以 # 或 // 开头的注释行）
    - parse_numbers=True : 按分隔符解析为浮点数矩阵（delimiter=None 时自动把空白/逗号都当分隔）

    Parameters:
    file_path (str): 文本文件路径（可无扩展名）
    parse_numbers (bool): 是否解析为数值
    delimiter (str|None): 分隔符；None 表示自动（空白或逗号）
    encoding (str): 文件编码

    Returns:
    np.ndarray: 行字符串数组（默认）或二维浮点数组（parse_numbers=True）
    """
    lines = []
    path = file_path if filename is None else f"{file_path}/{filename}"
    with open(path, "r", encoding=encoding) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#") or s.startswith("//"):
                continue
            lines.append(s)

    idx, num = [], []
    for i, line in enumerate(lines):
        try:
            num.append(int(line))
            idx.append(i + 2)
        except ValueError:
            pass
    
    points_dist = {}
    keys = ["internal", "inlet", "outlet", "bottom", "top", "obstacle"]
    for k, (n, i) in enumerate(zip(num, idx)):
        points = []
        for j in range(n):
            line = lines[i + j][1: -1]
            points.append([float(x) for x in line.split()][: 2])
        points_dist[keys[k]] = np.array(points, dtype=np.float64)
    return points_dist

def create_train_data(x_start, x_end, y_start, y_end, num_points):
    def y_lower(xs):
        slope = math.tan(math.degrees(15))
        return y_start + slope * xs + 1e-6
    
    x_internal1 = np.random.uniform(x_start + 1e-5, 0, int(num_points * 0.1))
    y_internal1 = np.random.uniform(y_start + 1e-6, y_end - 1e-6, int(num_points * 0.1))
    inputs_internal1 = np.stack((x_internal1, y_internal1), axis=-1)

    x_internal2 = np.random.uniform(0, x_end - 1e-5, int(num_points * 0.4))
    y_internal2 = np.random.uniform(y_lower(x_internal2), y_end - 1e-6, int(num_points * 0.4))
    inputs_internal2 = np.stack((x_internal2, y_internal2), axis=-1)

    x_inlet = np.random.uniform(x_start, x_start + 1e-5, int(num_points * 0.1))
    y_inlet = np.random.uniform(y_start + 1e-6, y_end - 1e-6, int(num_points * 0.1))
    inputs_inlet = np.stack((x_inlet, y_inlet), axis=-1)

    x_outlet = np.random.uniform(x_end - 1e-5, x_end, int(num_points * 0.1))
    y_outlet = np.random.uniform(y_lower(x_end), y_end - 1e-6, int(num_points * 0.1))
    inputs_outlet = np.stack((x_outlet, y_outlet), axis=-1)

    x_bottom = np.random.uniform(x_start + 1e-5, x_end - 1e-5, int(num_points * 0.1))
    y_bottom = np.random.uniform(y_start, y_start + 1e-6, int(num_points * 0.1))
    inputs_bottom = np.stack((x_bottom, y_bottom), axis=-1)

    x_top = np.random.uniform(x_start + 1e-5, x_end - 1e-5, int(num_points * 0.1))
    y_top = np.random.uniform(y_end - 1e-6, y_end, int(num_points * 0.1))
    inputs_top = np.stack((x_top, y_top), axis=-1)

    x_obstacle = np.random.uniform(0, x_end, int(num_points * 0.1))
    y_obstacle = x_obstacle * math.tan(math.radians(15))
    inputs_obstacle = np.stack((x_obstacle, y_obstacle), axis=-1)

    inputs = np.vstack((
        inputs_internal1, inputs_internal2,
        inputs_inlet, inputs_outlet,
        inputs_bottom, inputs_top,
        inputs_obstacle
    ))

    inputs = torch.tensor(inputs, dtype=torch.float64)
    conditions = ["internal"] * int(num_points * 0.5) + ["inlet"] * int(num_points * 0.1) + ["outlet"] * int(num_points * 0.1) + ["bottom"] * int(num_points * 0.1) + ["top"] * int(num_points * 0.1) + ["obstacle"] * int(num_points * 0.1)

    return inputs, conditions

GAMMA = 1.4
R_GAS = 289.0
C_V = R_GAS / (GAMMA - 1.0)
VEL_REF_X = 1736.0
VEL_REF_Y = 173.6
RHO_REF = 1.16838727127
P_REF = 101325.0
T_REF = 300.0
TIME_REF = 0.003048 * 1.5/ VEL_REF_X

def all_loss(inputs, outputs, conditions):
    u = outputs[:, 0]
    v = outputs[:, 1]
    rho = outputs[:, 2]
    p = outputs[:, 3]
    T = outputs[:, 4]

    vel_sq = u**2 + v**2
    energy = rho * (C_V * T + 0.5 * vel_sq)
    enthalpy = energy + p

    grad_rho = autodiff(rho, inputs)
    grad_rho_u = autodiff(rho * u, inputs)
    grad_rho_v = autodiff(rho * v, inputs)
    grad_mom_x = autodiff(rho * u * u + p, inputs)
    grad_mom_xy = autodiff(rho * u * v, inputs)
    grad_mom_y = autodiff(rho * v * v + p, inputs)
    grad_energy = autodiff(energy, inputs)
    grad_flux_x = autodiff(enthalpy * u, inputs)
    grad_flux_y = autodiff(enthalpy * v, inputs)

    continuity = grad_rho[:, 2] + grad_rho_u[:, 0] + grad_rho_v[:, 1]
    mom_x = grad_rho_u[:, 2] + grad_mom_x[:, 0] + grad_mom_xy[:, 1]
    mom_y = grad_rho_v[:, 2] + grad_mom_y[:, 0] + grad_mom_xy[:, 1]
    energy_res = grad_energy[:, 2] + grad_flux_x[:, 0] + grad_flux_y[:, 1]
    eos = p - rho * R_GAS * T

    residuals = torch.stack([continuity, mom_x, mom_y, energy_res, eos], dim=1)
    res_scales = outputs.new_tensor([1, 1, 1, 1, P_REF])
    residuals_dimless = residuals / res_scales

    losses = []
    for i, cond in enumerate(conditions):
        if cond == "internal":
            losses.append((residuals_dimless[i] ** 2).sum())
    if losses:
        return torch.stack(losses).mean()
    return outputs.new_zeros(1)

def initial_loss(inputs, outputs, conditions):
    output_scales = outputs.new_tensor([VEL_REF_X, VEL_REF_Y, RHO_REF, P_REF, T_REF])
    losses = []
    for i, cond in enumerate(conditions):
        if cond in {"internal", "obstacle", "outlet"}:
            target1 = outputs.new_tensor([0.0, 0.0, RHO_REF, P_REF, T_REF])
            diff1 = (outputs[i] - target1) / output_scales
            losses.append((diff1 ** 2).sum())
        if cond in {"inlet", "bottom", "top"}:
            target2 = outputs.new_tensor([VEL_REF_X, 0.0, RHO_REF, P_REF, T_REF])
            diff2 = (outputs[i] - target2) / output_scales
            losses.append((diff2 ** 2).sum())

    if losses:
        return torch.stack(losses).mean()
    return outputs.new_zeros(1, requires_grad=True)

def boundary_loss(inputs, outputs, conditions):
    inputs = inputs.requires_grad_(True)
    u = outputs[:, 0]
    v = outputs[:, 1]
    rho = outputs[:, 2]
    p = outputs[:, 3]
    T = outputs[:, 4]

    grad_u = autodiff(u, inputs)
    grad_v = autodiff(v, inputs)
    grad_rho = autodiff(rho, inputs)
    grad_p = autodiff(p, inputs)
    grad_T = autodiff(T, inputs)

    vel_scales = outputs.new_tensor([VEL_REF_X, VEL_REF_Y])
    vel_dimless = outputs[:, :2] / vel_scales
    output_scales = outputs.new_tensor([VEL_REF_X, VEL_REF_Y, RHO_REF, P_REF, T_REF])

    losses = []
    for i, cond in enumerate(conditions):
        if cond == "inlet":
            targets = outputs.new_tensor([VEL_REF_X, 0.0, RHO_REF, P_REF, T_REF])
            diff = (outputs[i] - targets) / output_scales
            losses.append((diff ** 2).sum())
        elif cond == "outlet":
            normal = outputs.new_tensor([1.0, 0.0])
            grad_space = grad_rho[i, :2] @ normal + grad_u[i, :2] @ normal + grad_v[i, :2] @ normal + grad_p[i, :2] @ normal + grad_T[i, :2] @ normal
            losses.append((grad_space ** 2).sum()) 
        elif cond in ("bottom", "top"):
            normal = outputs.new_tensor([0.0, -1.0 if cond == "bottom" else 1.0])
            vel = outputs[i, :2]
            u_n = vel @ normal
            inlet_target = outputs.new_tensor([VEL_REF_X, 0.0])
            diff_inlet = (vel - inlet_target) / vel_scales
            loss_inlet = (diff_inlet ** 2).sum()
            grad_vel = torch.stack([grad_u[i, :2], grad_v[i, :2]], dim=0)
            loss_outlet = (grad_vel @ normal).pow(2).sum()
            grad_scalars = (
                (grad_rho[i, :2] @ normal).pow(2)
                + (grad_p[i, :2] @ normal).pow(2)
                + (grad_T[i, :2] @ normal).pow(2)
            )
            weights = torch.sigmoid(-u_n / 1e-4)
            losses.append(weights * loss_inlet + (1 - weights) * loss_outlet + grad_scalars)
        elif cond == "obstacle":
            normal_vec = outputs.new_tensor([0.081670913853, -0.3048])
            normal = normal_vec / normal_vec.norm()
            v_normal = vel_dimless[i].dot(normal)
            grad_space1 = grad_rho[i, :2] @ normal + grad_p[i, :2] @ normal + grad_T[i, :2] @ normal
            grad_space2 = -normal[1] * grad_u[i, :2] @ normal + normal[0] * grad_v[i, :2] @ normal
            losses.append(v_normal.pow(2) + grad_space1.pow(2).sum() + grad_space2.pow(2).sum())

    if losses:
        return torch.stack(losses).mean()
    return outputs.new_zeros(1)

def MSE_loss(outputs, results):
    '''先根据上面的REF值对outputs和results做无量纲归一化, 再计算MSE'''
    output_scales = outputs.new_tensor([VEL_REF_X, VEL_REF_Y, RHO_REF, P_REF, T_REF])
    diff = (outputs - results) / output_scales
    return (diff ** 2).sum(dim = 1).mean()