function [data] = f_cell_to_num(data_cell )
    row = size(data_cell,1);
    
    if size(data_cell{1},2) > 50
        data = cell2mat(data_cell);                                  % 数据为时程数据时的转换形式
    else
        for i = 1:row
            data_temp = data_cell{i,:}(:,end) ;
            data(i,:) = data_temp';

        end
    end
    
end