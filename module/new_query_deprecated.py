#coding: utf8
"""
两个样本集，投影到同一个特征空间中，测试以下几种任务来评估模型性能
1.query精度与召回率：使用余弦相似性进行相似度对每个query得到的结果进行排序，计算前k个加所结果的精度和召回率，k从1到所有可检索的样本的总数
"""
import numpy
import time
import sys
import pylab

def unify(X):
    """将设计矩阵X的每一行的范数归一"""
    assert isinstance(X, numpy.ndarray), X.ndim == 2 #参数检测
    norm = numpy.sqrt(numpy.sum(X**2, axis=1)) #样本集范数向量
    #若投影后为零向量，则范数设为1，在做除法时， 0/1仍等于0                                   
    for n, i in enumerate(norm):
        if i == 0.:
            norm[n] = 1.
    assert numpy.all(norm) # 检查分母不为零
    X_normalized = X / norm[:, numpy.newaxis]
    return X_normalized

def cos_dist(X_test, X_train):
    """两个矩阵中每一对样本（X_i, Y_j)的余弦距离组成的余弦距离矩阵"""
    assert isinstance(X_test, numpy.ndarray), X_test.ndim == 2
    assert isinstance(X_train, numpy.ndarray), X_train.ndim == 2
    #范数归一化
    X_test = unify(X_test)
    X_train = unify(X_train)
    cos_matrix = numpy.dot(X_test, X_train.T) #cos_matrix_ij代表第i个query与第j个训练样本的余弦相似度
    return cos_matrix

def euclidean_dist(X_test, X_train):
    """两个矩阵中每一对样本（X_i, Y_j)的欧氏距离组成的欧氏距离矩阵"""
    assert isinstance(X_test, numpy.ndarray), X_test.ndim == 2
    assert isinstance(X_train, numpy.ndarray), X_train.ndim == 2
    #范数归一化
    assert X_train.shape[1] == X_test.shape[1] #两个矩阵的样本所处的特征空间必须维数相同
    square_delta = ((X_train.reshape(X_train.shape[0], 1, X_train.shape[1]) - X_test.reshape(1, X_test.shape[0], X_test.shape[1]))**2).sum(axis=2)
    euc_matrix = numpy.sqrt(square_delta)
    return euc_matrix
    
def p_r(X_train, label_train, X_test, label_test, dist_func='cos', pic=True, name='', prefix='', suffix='', save_results=True):
    """计算精度和召回率, 要求类标是用向量形式，不能使用one-hot型"""
    #(1).预处理
    assert isinstance(label_train, numpy.ndarray), label_train.ndim == 1
    assert isinstance(label_test, numpy.ndarray), label_test.ndim == 1
    assert dist_func in ['cos', 'euc']
    #X_train = X_train[:800, :] #测试程序用的数据子集==================
    #X_test = X_test[:800, :] #测试程序用的数据子集=================
    #label_test = label_test[:800] #==================
    #label_train = label_train[:800] #======================
    t0 = time.clock()
    rows = X_test.shape[0]
    cols = X_train.shape[0]
    if dist_func is 'cos':
        distance_matrix = cos_dist(X_test=X_test, X_train=X_train) #这里使用了抽象的dict()函数，方便更换不同的度量
    elif dist_func is 'euc':
        distance_matrix = euclidean_dist(X_test=X_test, X_train=X_train)
    assert distance_matrix.shape == (rows, cols)
    t1 = time.clock()
    print '预处理时间:', t1 - t0
    sys.stdout.flush()
    
    #(2).主计算过程
    t2 = time.clock()
    #由于numpy.sort()没有参数reverse，所以采用先由小到大排序在反转列序的办法
    #sorted_dist_matrix = numpy.sort(distance_matrix, axis=1)[:, ::-1]
    #返回距离由达到小的下标矩阵
    argsort_matrix = numpy.argsort(distance_matrix, axis=1)[:, ::-1]
    #求出距离排序后每一行的对应位置样本的类标组成的矩阵
    label_matrix = numpy.ones(shape=(rows, cols)) * label_train
    sorted_label_matrix = numpy.zeros(shape=label_matrix.shape)
    #循环类标矩阵的每一行，按argsort_matrix的相应行作为下标进行调整
    for i in xrange(label_matrix.shape[0]):
        sorted_label_matrix[i] = label_matrix[i][argsort_matrix[i]]
    #将测试集X_test的类标向量label_test作为列向量平行复制X_train.shape[0]列，得到一个矩阵
    test_label_matrix = numpy.ones(shape=(rows, cols)) * label_test[:, numpy.newaxis]
    #通过比较类标是否相同，得到一个布尔矩阵
    bool_matrix = (test_label_matrix == sorted_label_matrix)
    accum_correct_matrix = numpy.zeros(shape=(rows, cols), dtype='float64')
    #TODO:下面的计算可能会比较慢。原因是每次切片是矩阵的一列而非一行，应该考虑先转置再按行切片赋值
    accum_correct_matrix[:, 0] = bool_matrix[:, 0]
    for i in xrange(1, bool_matrix.shape[1]):
        accum_correct_matrix[:, i] = accum_correct_matrix[:, i-1] + bool_matrix[:, i]
    t3 = time.clock()
    print '主机算过程用时:', t3 - t2
    sys.stdout.flush()
       
    #(3).获得精度矩阵和召回率矩阵
    t4 = time.clock()
    precision_denominator = numpy.ones(shape=(rows, cols), dtype='float64') * numpy.arange(1, cols+1)
    precision_matrix = accum_correct_matrix / precision_denominator #每个位置元素为（当前总正确数/当前总搜索数)
    recall_denominator = accum_correct_matrix[:, -1] #向量，索引为i的分量值表示训练样本集中与测试样本i同类的样本的总数
    recall_matrix = accum_correct_matrix / recall_denominator[:, numpy.newaxis]
    
    #(4).求测试集所有query在训练集上的平均召回率和平均精度
    mean_precision = numpy.mean(precision_matrix, axis=0)
    mean_recall = numpy.mean(recall_matrix, axis=0)
    t5 = time.clock()
    print '计算精度矩阵和召回率矩阵用时: ', t5 - t4
    sys.stdout.flush()
    
    #(5).保存计算结果
    if save_results:
        numpy.save(prefix + 'precision_matrix' + suffix + '.npy', precision_matrix)
        numpy.save(prefix + 'recall_matrix' + suffix + '.npy', recall_matrix)
        numpy.save(prefix + 'mean_precision' + suffix + '.npy', mean_precision)
        numpy.save(prefix + 'mean_recall' + suffix + '.npy', mean_recall)
    print 'model saved...'
    
    f = open(prefix+'mean_precision_record'+suffix, 'w')
    for num, i in enumerate(mean_precision):
        f.write(str(num) + ': ' + str(i) + '\n')
    f.close()
    
    f = open(prefix+'mean_recall_record'+suffix, 'w')
    for num, i in enumerate(mean_recall):
        f.write(str(num) + ': ' + str(i) + '\n')
    f.close()

    #(6).根据计算结果绘制p-r曲线
    if pic is True:
        print 'drawing curve...'
        pr_curve(name=name, mean_precision=mean_precision, mean_recall=mean_recall, prefix=prefix, suffix=suffix) #绘制召回率-精度曲线
    print 'all finished...'
    
def get_error_1124(X_train, label_train, X_test, label_test):
    """每个query只有一个正确答案，当答案位于检索结果的前t%时认为检索正确"""
    assert isinstance(label_train, numpy.ndarray), label_train.ndim == 1
    assert isinstance(label_test, numpy.ndarray), label_test.ndim == 1
    #X_train = X_train[:600, :] #测试程序用的数据子集===========
    #X_test = X_test[:600, :] #测试程序用的数据子集============
    #label_test = label_test[:800] #==================
    #label_train = label_train[:800] #======================
    t0 = time.clock()
    rows = X_test.shape[0]
    cols = X_test.shape[0]
    distance_matrix = cos_dist(X_test=X_test, X_train=X_test) #这里使用了抽象的dict()函数，方便更换不同的度量
    assert distance_matrix.shape == (rows, cols)
    t1 = time.clock()
    print '预处理时间:', t1 - t0
    sys.stdout.flush()
    
    t2 = time.clock()
    #由于numpy.sort()没有参数reverse，所以采用先由小到大排序在反转列序的办法
    #sorted_dist_matrix = numpy.sort(distance_matrix, axis=1)[:, ::-1]
    #返回距离由达到小的下标矩阵
    argsort_matrix = numpy.argsort(distance_matrix, axis=1)[:, ::-1]
    #record_matrix的ij元素表示当t=j+1时，第i个检索是否被判为正确
    record_matrix = numpy.zeros(shape=argsort_matrix.shape, dtype=bool)
    #使用动态规划的思想，先判断第一列的真值，再向后逐列计算
    record_matrix[:, 0] = (argsort_matrix[:, 0] == numpy.arange(rows)) #计算第一列
    for t in xrange(1, cols): #计算剩下的cols-1列
        cur = (argsort_matrix[:, t] == numpy.arange(rows))
        record_matrix[:, t] = cur + record_matrix[t-1] #每一个分量或运算，即两个向量的对应位置的两个元素只要有一个为真，结果为真
    t3 = time.clock()
    print '主计算过程时间: ', t3 - t2

    #计算所有检索在给定的t上的平均错误率
    average_error = 1. - record_matrix.mean(axis=0)
    #计算错误率曲线(以t为自变量)的AUC
    AUC = compute_AUC(average_error)
    return average_error

def compute_AUC(x, y=None):
    """计算曲线下方面积"""
    assert isinstance(x, numpy.ndarray), x.ndim == 1
    if y is not None:
        assert isinstance(y, numpy.ndarray), y.ndim == 1
    else:
        y = numpy.arange
    x = numpy.array(x, 'float64')
    y = numpy.array(y, 'float64')
    delta = [y[i] - y[i-1] for i in xrange(1, y.shape[0])]
    
    AUC = numpy.dot(x, delta)
    return AUC
    
#绘制p-r曲线
def pr_curve(mean_precision, mean_recall, name='20newsgroups', prefix='', suffix=''):
    """
    Parameters
    ----------

    mean_precision: numpy.ndarry，一维
        记录平均精度的向量.
    mean_recall: numpy.ndarray，一维
        记录平均召回率向量.
    prefix, duffix: 同rsm_query()
    """  
    #color_list = ['blue', 'red', 'green', 'cyan', 'yellow', 'black', 'magenta', (0.5,0.5,0.5)]
    y_vector = mean_precision * 100.
    x_vector = mean_recall * 100 #** (1./6.)

    pylab.figure(figsize=(8, 8))
    pylab.grid() #在做标系中显示网格
    pylab.plot(x_vector, y_vector, label='$r-p curve$', color='blue', linewidth=1)

    pylab.xlabel('recall(%)')
    pylab.ylabel('precision(%)')
    pylab.title(name)
    #pylab.xlim(0., 60) #x轴长度限制
    #pylab.ylim(0., 60) #y轴长度限制
    pylab.legend() #在图像中显示标记说明
    #pylab.show() # 显示图像
    pylab.savefig(prefix + 'r-p' + suffix + '.png', dpi=240) #保存图像，可以人为指定所保存的图像的分辨率
    print 'pic saved...'
