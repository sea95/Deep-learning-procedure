function statistics = f_statistics(response)
    %>>>> 每个质点（共n个，或n列）的均值-标准差,并将其从行向量转置为列向量
    mean_value = mean(response,1)';
    std_value = std(response,1)';
    
    % >>>> 计算均值加减一倍sigma的值
    envelop_left = mean_value - std_value;
    envelop_right = mean_value + std_value;
    
    % >>>> 提取基底的均值和标准差
    base_mean = mean_value(1);
    base_std = std_value(1);
    
    % ==== Summmary ====
    statistics.mean_value = mean_value;
    statistics.envelop_left = envelop_left;
    statistics.envelop_right = envelop_right;
    
    statistics.base_mean = base_mean;
    statistics.base_std = base_std;

end