# Wireless-Sensor-Network-Localization

问题描述（观测量、未知量、已知量）
无线传感器网络的定位问题
物联网是新一代信息技术的重要组成部分，它是通过各种信息传感设备，如传感器、射频识别（RFID）技术、全球定位系统等各种装置与技术，实时采集声、光、热、 电、力学、化学、生物、位置等各种信息，与互联网结合形成的一个巨大网络。作为物联网的重要组成部分，无线传感器网络（WSN, Wireless Sensor networks）就是由部署在监测区域内大量的廉价微型传感器节点组成，通过无线通信方式形成的一个多跳自组织网络。 
无线传感器网络的很多应用场合必须知道节点的位置，因此节点定位技术是 WSN 的关键技术和研究热点。然而，在所有节点上都配备GPS等定位设施成本很高。因此，一般只在部分节点通过 GPS 定位设备获得自身的精确位置，这些节点称为信标节点（beacon node）；而其它未知节点（unknown node）则通过网络连接信息和节点内部相互测距通过几何计算来估计其位置坐标。假设测距的随机误差服从正态分布，并且由于地形或设备原因，测距可能存在系统偏差。请在此假设条件下解决以下问题：
1. 对仿真算例中的未知节点进行定位（结果存储在附件表3中） 
2. 对定位结果精度进行评定 
3. 对定位结果进行总体模型检验

变量分析
观测量:每一个未知节点与最近六个信标节点的测量距离，使用 j 和 i 来标记未知节点和信标节点的编号，未知节点 j 和已知节点 i 之间的测量距离记为 d_{ij}；
未知量:每一个未知节点的位置坐标，使用 j 来标记未知节点的编号，记为 \left(x_j,y_j\right)  ；
已知量:1-15个信标节点的GPS坐标，记为 \left(x_i,y_i\right)，其中 i 是信标节点的编号；信标节点的位置是通过GPS测定的，因此当作误差为0的准确值来处理。

数学模型（函数模型和随机模型）
观测量（距离）与未知量（未知节点坐标）之间的关系可以通过欧几里得距离公式表示：
d_{ij}=\sqrt{\left(x_j-x_i\right)^2+\left(y_j-y_i\right)^2}+\varepsilon_{ij}
其中，  d_{ij}   是未知节点j与信标节点i之间的测量距离；
      \left(x_j,y_j\right) 是未知节点j的坐标；
      \left(x_i,y_i\right) 是信标节点i的坐标；
      \varepsilon_{ij}   是测量未知节点j与信标节点i之间距离的误差；
2.随机模型
在测量过程中，测量误差由系统偏差和随机误差组成，表示为：
\varepsilon_{ij}=\mu+\delta
其中，  \varepsilon_{ij}    是测量误差，与前面函数模型中的测量误差一致；
\mu     是系统误差，是由于地形和设备原因造成的；
\delta     是随机误差；
\delta  服从正态分布，可以定义其服从均值为0，方差为 \sigma^2 的正态分布，即 
\delta~N\left(0,\sigma^2\right)
因此可推得 \varepsilon_{ij} 服从均值为\mu\ ，方差为 \sigma^2 的正态分布，即
\varepsilon_{ij}~N\left(\mu,\sigma^2\right)

参数估计（非线性问题包括线性化、迭代求解、线性问题直接用BLUE估计的公式求解）

在无线传感器网络节点定位中，我们的目标是通过测量未知节点与若干信标节点之间的距离来估计未知节点的位置坐标 \left(x_j,y_j\right) 。由于距离公式是非线性的，因此需要对其进行线性化处理，并通过迭代方法求解。

表格预处理与数据读取:
要转化为代码可以处理的数据，需要对原数据进行预处理和读取工作。我们首先拆解了原excel文件中的不同表，将需要读取的两张表单独提取出来成为两个单独的excel文件方便处理，将存储相对距离表格的表头行和纵轴编号列都删掉了，只保留了原始的数据表格，记为1.xlsx，如图2；将存储信标节点坐标表格的表头行删掉了，也同样只保留了原始的数据表格，记为2.xlsx，如图3。由于这些数据都是标准的数据格式，所以不用进行数据格式转换。
接下来就是python中的转换代码：
beacon_coords_path = "2.xlsx" #这里需要替换为运行电脑上的相对路径
beacon_coords = pd.read_excel(beacon_coords_path, header=None, names=['x', 'y']).to_numpy()
distances_path = " \1.xlsx" #这里需要替换为运行电脑上的相对路径    
distances_matrix = pd.read_excel(distances_path, header=None).to_numpy()

初始估计:
为了开始迭代求解，我们首先需要为未知节点的坐标提供一个初始估计值。通常可以使用信标节点坐标的质心作为初始估计。这是因为质心可以作为未知节点可能位置的一个初始估计。
\left(x_j^{\left(0\right)},y_j^{\left(0\right)}\right)=\left(\frac{1}{6}\sum_{i=1}^{N}x_i,\frac{1}{6}\sum_{i=1}^{N}y_i\right)
其中， \left(x_i,y_i\right) 是第i 个信标节点的坐标；信标节点的数量为6；
代码展示：
def initial_estimate(distances_row, beacon_coords): 
indices = np.nonzero(distances_row)[0]     
initial_x = np.mean(beacon_coords[indices, 0])     
initial_y = np.mean(beacon_coords[indices, 1])     
return np.array([initial_x, initial_y])

雅可比矩阵:
引入雅可比矩阵J，其元素为距离对坐标的偏导数：
J_{ij}=\left[\begin{matrix}\frac{\partial d_{ij}}{\partial x_j}&\frac{\partial d_{ij}}{\partial y_j}\\\end{matrix}\right]
其中：
\frac{\partial d_{ij}}{\partial x_j}=\frac{x_j^{\left(0\right)}-x_i}{d_{ij}^{\left(0\right)}}，∂dij∂yj=yj0-yidij0
代码展示：
def compute_jacobian(estimated_coords, beacon_indices, beacon_coords):     jacobian = []     for i in beacon_indices:         d = np.linalg.norm(estimated_coords - beacon_coords[i])         jacobian.append([(estimated_coords[0] - beacon_coords[i, 0]) / d,                           (estimated_coords[1] - beacon_coords[i, 1]) / d])     return np.array(jacobian)

线性化处理和线性化模型的矩阵表示
在每次迭代中，我们使用泰勒展开对非线性距离公式进行线性化处理。对于每个距离测量d_{ij}，我们可以展开公式如下：
d_{ij}\approx d_{ij}^{\left(0\right)}+\left.\frac{\partiald_{ij}}{\partialx_j}\right|_{\left(x_j^{\left(0\right)},y_j^{\left(0\right)}\right)}\left(x_j-x_j^{\left(0\right)}\right)+\left.\frac{\partiald_{ij}}{\partialy_j}\right|_{\left(x_j^{\left(0\right)},y_j^{\left(0\right)}\right)}\left(y_j-y_j^{\left(0\right)}\right)
其中，d_{ij}^{\left(0\right)}=\sqrt{\left(x_j^{\left(0\right)}-x_i\right)^2+\left(y_j^{\left(0\right)}-y_i\right)^2} 是使用初始估计值计算的距离
将所有测量值和偏导数整合，可以得到线性化模型的矩阵形式：
d\approx d^{\left(0\right)}+J\left(P_j-P_j^{\left(0\right)}\right)
其中：
d=\left[\begin{matrix}d_{1j}\\\vdots\\d_{Nj}\\\end{matrix}\right]，d0=d1j0⋮dNj0，J=xj0-x1d1j0yj0-y1d1j0⋮⋮xj0-xNdNj0yj0-yNdNj0，Pj=xjyj
此处，d 是实际测量的距离向量，d^{\left(0\right)} 是使用初始估计值计算的距离向量，J 是雅可比矩阵， P_j 是未知节点的实际坐标向量， P_j^{\left(0\right)}是初始估计值向量。
这里的代码将与后面的迭代求解一起展示。

迭代求解
使用高斯-牛顿法进行迭代求解，每次迭代更新未知节点坐标估计值：
P_j^{\left(k+1\right)}=P_j^{\left(k\right)}+∆Pjk
其中增量 ∆Pjk 通过最小化残差平方和求解：
∆Pjk=JTJ-1JTd-d0
在此公式中，J^T 是雅可比矩阵的转置，\left(J^TJ\right)^{-1} 是 J^TJ\ 的逆矩阵，d-d^{\left(0\right)} 是测量值和初始估计值的残差向量。
迭代过程持续进行，直到增量 \ ∆Pjk 足够小未知，即达到预设的收敛标准。经过若干次迭代，我们可以获得最终的估计值：
{\hat{P}}_j=P_j^{\left(k+1\right)}
代码展示：
def gauss_newton(distances_row, beacon_coords, initial_coords, max_iterations=100, tolerance=1e-6):     estimated_coords = initial_coords     beacon_indices = np.nonzero(distances_row)[0]     residual_history = []     for _ in range(max_iterations):         estimated_distances = np.array([np.linalg.norm(estimated_coords - beacon_coords[i]) for i in beacon_indices])         residuals = distances_row[beacon_indices] - estimated_distances         residual_history.append(np.linalg.norm(residuals))         jacobian = compute_jacobian(estimated_coords, beacon_indices, beacon_coords)         delta_coords = np.linalg.inv(jacobian.T @ jacobian) @ jacobian.T @ residuals         estimated_coords += delta_coords         if np.linalg.norm(delta_coords) < tolerance:             break     sigma_squared = np.sum(residuals**2) / (len(beacon_indices) - 2)  
cov_matrix = sigma_squared * np.linalg.inv(jacobian.T @ jacobian)     return estimated_coords, residual_history, cov_matrix, residuals

unknown_node_coords = [] all_residual_histories = [] all_cov_matrices = [] all_residuals = [] for i, distances_row in enumerate(distances_matrix):     initial_coords = initial_estimate(distances_row, beacon_coords)     estimated_coords, residual_history, cov_matrix, residuals = gauss_newton(distances_row, beacon_coords, initial_coords)     unknown_node_coords.append(estimated_coords)     all_residual_histories.append(residual_history)     all_cov_matrices.append(cov_matrix)     all_residuals.append(residuals)  unknown_node_coords = np.array(unknown_node_coords) for i, coords in enumerate(unknown_node_coords):     print(f"未知节点 {i+1} 的估计坐标: x = {coords[0]:.4f}, y = {coords[1]:.4f}")

经过迭代求解之后，得到对应的未知点的坐标，将信标节点和未知节点求解结果一起表示在图上，如图所示，可以发现和算例的结果几乎差不多，粗略地说明了结果是正确的。下面，会根据求解结果进行更详细的参数精度评定与总体模型检验。

参数精度评定与总体模型检验:

迭代残差变化曲线和残差分布图:在每次迭代中，我们计算了估计距离与实际测量距离之间的差值，称为残差。把迭代求解过程中的总残差变化曲线求解出来，可以发现总残差在开始时呈现快速下降的趋势，随后出现了暂时的放缓，最后继续下降到几乎为0的水平，这种迭代残差变化曲线反映了计算结果是正确且有效的。此外，可以把迭代求解过程中的每一点的残差变化曲线求解出来，可以发现点与点虽然存在着一些变化的差异，但是整体上都是快速变小至接近0的，进一步验证了计算结果的正确性。

协方差矩阵计算
协方差矩阵反映了参数估计的不确定性和参数之间的相关性。通过分析协方差矩阵，可以了解参数估计的精度以及参数之间的线性相关性。
对于两个随机变量 X 和 Y，他们的平均值（期望值）分别记为 E[X] 和 E[Y] ，这两个随机变量的协方差定义为
Cov(\ X,Y\ )\ =\ E\ [(\ X\ -\ E[X] )( Y - E[Y] )]
协方差度量了 X 和 Y\  相对于它们各自的均值的联合变化趋势。如果两个变量的协方差为正，它们倾向于一起增加或减少；如果为负，一个变量的增加往往伴随着另一个变量的减少。
对于一个由 n 个随机变量 X_1,X_2,\ldots,X_n 组成的向量 ，协方差矩阵 \Sigma 是一个 n\times n 的方阵，其元素定义如下：
\Sigma=Cov\left(X_i,X_j\right)
协方差矩阵的对角线元素 \Sigma_{ii} 是变量 X_i 的方差，非对角线元素 \Sigma_{ij} 是 X_i 和 X_j 的协方差。
在可视化图中，协方差矩阵的对角线元素表示参数的方差，非对角线元素表示参数之间的协方差。方差越大，说明参数的不确定性越大；协方差越大，说明参数之间的相关性越强。
未知节点的残差在迭代变化过程中的协方差矩阵，可以看出未知节点和未知节点残差之间的相关性并不均衡。首先看对角线上的协方差分布，这里反映的是变量自身的方差，可以看出1点、2点、8-9点、20点、28点的方差都相当地大，这与前面迭代残差变化图相对应，这些点的迭代变化值显然比较大。再看其它区域的协方差分布，可以看出，坐标值的迭代是有明显倾向的，对于原本初始值就接近真值的未知点，往往方差较小，而一开始初始值远离真值的未知点，往往方差较大，因为需要迭代多次才能接近真值，而且结合之间的残差变化图，这些远点往往会在快速接近真值之后在真值附近停滞一两代，然后才继续缓慢逼近真值。

残差正态分布的检验和判断
之前在残差分布图中，我们发现残差的分布似乎没有什么规律，为了佐证这一点，我们进行残差正态分布的检验和判断，看其是否为正态分布。这里使用了检验正态分布的两种方法，一种是直观判断的Q-Q图表示，一种是属于假设检验的K-S检验方法。
Q-Q图是一种用于比较两组数据分布的图形工具，我们在这里进行原分布与正态分布的比较。横轴是表示理论分布分位数，是通过累计分布函数的逆函数 \Phi^{-1}\left(p\right) 来计算分位数，其中 p 是累计概率；纵轴是表示样本数据的有序值，样本中的第 i 个值对应于经验累计概率 \frac{i}{n+1} ，其中 n 是样本大小。一般来说，如果点显著偏移这条直线，表面样本分布与理论分布不一致。
根据Q-Q图来看，明显残差分布不符合正态分布情形。
K-S检验是一种基于最大距离的统计检验，用于比较一个样本分布与一个理论分布（或两个样本分布）之间的一致性。我们现在来比较原分布是否符合正态分布。其中定义了样本数据的理论累积分布函数，表示在该分布夏，随机变量取值小于或等于某个数值的概率，还有经验累计分布函数，是样本数据从小到大排序后，每个数据点所累积的观测值比例。对每一个数据点，计算经验累积分布函数和理论累计分布函数的绝对差值，根据D和样本大小确定p值，并根据p值来决定是否拒绝原假设。
根据结果，K-S检验值在0.39左右，表示样本分布与理论分布之间的最大偏差是0.39，判断显著性的概率值是一个非常接近0的数，再次说明了不是呈正态分布。
总的来看，残差值并不是呈现正态分布，但是这不代表着该算法是错误的。按Q-Q图和之前的残差分布图来看，残差分布更倾向于呈现一个双峰的融合正态分布，对于未知点而言，在迭代过程中，残差的大小与也与距离已知点的大小有关，在周边已知点都距离较远的情况下，可能使得残差也相应增加，另外，初始值的设定也有很强的关联，远点和近点的迭代快慢先后有着显著的差别，这使得残差有较大的不同。

