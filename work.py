# 导入所需的库
import pandas as pd  # 用于数据处理和读取Excel文件
import numpy as np  # 用于数值计算
import matplotlib.pyplot as plt  # 用于绘图
from scipy.stats import chi2, shapiro, probplot, kstest  # 用于统计检验
import seaborn as sns  # 用于增强绘图效果

# 读取基站坐标数据
beacon_coords_path = "D:\\2.xlsx"  # 基站坐标文件路径
beacon_coords = pd.read_excel(beacon_coords_path, header=None, names=['x', 'y']).to_numpy()  # 读取基站坐标数据并转换为NumPy数组
distances_path = "D:\\1.xlsx"  # 距离矩阵文件路径
distances_matrix = pd.read_excel(distances_path, header=None).to_numpy()  # 读取距离矩阵数据并转换为NumPy数组

# 初始估计函数：根据距离矩阵的非零元素计算未知节点的初始坐标
def initial_estimate(distances_row, beacon_coords):
    indices = np.nonzero(distances_row)[0]  # 获取距离矩阵中非零元素的索引
    initial_x = np.mean(beacon_coords[indices, 0])  # 计算基站x坐标的平均值作为初始x坐标
    initial_y = np.mean(beacon_coords[indices, 1])  # 计算基站y坐标的平均值作为初始y坐标
    return np.array([initial_x, initial_y])  # 返回初始坐标

# 计算雅可比矩阵
def compute_jacobian(estimated_coords, beacon_indices, beacon_coords):
    jacobian = []  # 初始化雅可比矩阵列表
    for i in beacon_indices:  # 遍历基站索引
        d = np.linalg.norm(estimated_coords - beacon_coords[i])  # 计算估计坐标与基站坐标之间的欧几里得距离
        jacobian.append([(estimated_coords[0] - beacon_coords[i, 0]) / d,  # 计算雅可比矩阵的元素
                         (estimated_coords[1] - beacon_coords[i, 1]) / d])
    return np.array(jacobian)  # 返回雅可比矩阵

# 高斯-牛顿法优化函数
def gauss_newton(distances_row, beacon_coords, initial_coords, max_iterations=100, tolerance=1e-6):
    estimated_coords = initial_coords  # 初始化估计坐标
    beacon_indices = np.nonzero(distances_row)[0]  # 获取距离矩阵中非零元素的索引
    residual_history = []  # 初始化残差历史记录
    for _ in range(max_iterations):  # 迭代优化
        estimated_distances = np.array([np.linalg.norm(estimated_coords - beacon_coords[i]) for i in beacon_indices])  # 计算估计距离
        residuals = distances_row[beacon_indices] - estimated_distances  # 计算残差
        residual_history.append(np.linalg.norm(residuals))  # 记录残差
        jacobian = compute_jacobian(estimated_coords, beacon_indices, beacon_coords)  # 计算雅可比矩阵
        delta_coords = np.linalg.inv(jacobian.T @ jacobian) @ jacobian.T @ residuals  # 计算坐标增量
        estimated_coords += delta_coords  # 更新估计坐标
        if np.linalg.norm(delta_coords) < tolerance:  # 如果增量小于阈值，则停止迭代
            break
    sigma_squared = np.sum(residuals**2) / (len(beacon_indices) - 2)  # 计算方差
    cov_matrix = sigma_squared * np.linalg.inv(jacobian.T @ jacobian)  # 计算协方差矩阵
    return estimated_coords, residual_history, cov_matrix, residuals  # 返回优化结果

# 初始化结果列表
unknown_node_coords = []  # 未知节点坐标
all_residual_histories = []  # 所有残差历史记录
all_cov_matrices = []  # 所有协方差矩阵
all_residuals = []  # 所有残差

# 遍历距离矩阵的每一行，对应每个未知节点
for i, distances_row in enumerate(distances_matrix):
    initial_coords = initial_estimate(distances_row, beacon_coords)  # 计算初始坐标
    estimated_coords, residual_history, cov_matrix, residuals = gauss_newton(distances_row, beacon_coords, initial_coords)  # 运行高斯-牛顿法
    unknown_node_coords.append(estimated_coords)  # 保存未知节点坐标
    all_residual_histories.append(residual_history)  # 保存残差历史记录
    all_cov_matrices.append(cov_matrix)  # 保存协方差矩阵
    all_residuals.append(residuals)  # 保存残差

unknown_node_coords = np.array(unknown_node_coords)  # 将未知节点坐标列表转换为NumPy数组

# 打印未知节点的估计坐标
for i, coords in enumerate(unknown_node_coords):
    print(f"未知节点 {i+1} 的估计坐标: x = {coords[0]:.4f}, y = {coords[1]:.4f}")

# 绘制总残差收敛曲线
max_len = max(len(history) for history in all_residual_histories)  # 获取最长的残差历史记录长度
total_residuals = np.zeros(max_len)  # 初始化总残差数组
for history in all_residual_histories:  # 遍历所有残差历史记录
    padded_history = np.pad(history, (0, max_len - len(history)), 'constant')  # 对残差历史记录进行填充
    total_residuals += padded_history  # 累加总残差
plt.figure(figsize=(10, 6))  # 创建图形
plt.plot(total_residuals)  # 绘制总残差曲线
plt.xlabel('number of iterations')  # 设置x轴标签
plt.ylabel('total residual')  # 设置y轴标签
plt.title('residual convergence curve')  # 设置标题
plt.grid(True)  # 显示网格
plt.show()  # 显示图形

# 绘制每个未知节点的残差变化曲线
plt.figure(figsize=(10, 6))  # 创建图形
for i, history in enumerate(all_residual_histories):  # 遍历所有残差历史记录
    plt.plot(history, label=f'Node {i+1}')  # 绘制每个节点的残差曲线
plt.xlabel('iteration')  # 设置x轴标签
plt.ylabel('residual')  # 设置y轴标签
plt.title('residual change curve at each iteration')  # 设置标题
plt.legend()  # 显示图例
plt.grid(True)  # 显示网格
plt.show()  # 显示图形

# 绘制节点定位结果
plt.figure(figsize=(10, 6))  # 创建图形
plt.scatter(beacon_coords[:, 0], beacon_coords[:, 1], c='r', marker='o', label='Beacon Nodes')  # 绘制基站节点
plt.scatter(unknown_node_coords[:, 0], unknown_node_coords[:, 1], c='b', marker='x', label='Unknown Nodes')  # 绘制未知节点
plt.xlabel('X Coordinate')  # 设置x轴标签
plt.ylabel('Y Coordinate')  # 设置y轴标签
plt.title('Node Localization Result')  # 设置标题
plt.legend()  # 显示图例
plt.grid(True)  # 显示网格
plt.show()  # 显示图形

# 绘制每个未知节点的最终残差
final_residuals = [history[-1] for history in all_residual_histories]  # 提取每个节点的最终残差
plt.figure(figsize=(10, 6))  # 创建图形
plt.plot(final_residuals, 'o-', label='Final Residuals')  # 绘制最终残差曲线
plt.xlabel('Node Index')  # 设置x轴标签
plt.ylabel('Final Residual')  # 设置y轴标签
plt.title('Final Residuals for Each Node')  # 设置标题
plt.legend()  # 显示图例
plt.grid(True)  # 显示网格
plt.show()  # 显示图形

# 计算总残差矩阵的协方差矩阵
total_residuals_matrix = np.zeros((len(all_residual_histories), max_len))  # 初始化总残差矩阵
for i, residual_history in enumerate(all_residual_histories):  # 遍历所有残差历史记录
    padded_residuals = np.pad(residual_history, (0, max_len - len(residual_history)), 'constant')  # 对残差历史记录进行填充
    total_residuals_matrix[i, :] = padded_residuals  # 填充总残差矩阵
covariance_matrix = np.cov(total_residuals_matrix)  # 计算协方差矩阵
plt.figure(figsize=(10, 8))  # 创建图形
plt.imshow(covariance_matrix, cmap='hot', interpolation='nearest') 