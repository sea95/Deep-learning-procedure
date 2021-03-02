function [data_summary,input_scale,output_scale] = f_Split_normalized(data_input,data_output,options )
% Note:
% 1.Data_summary is composed of the training/ test/  predict subset,
% which is normalized with specified approach.
% 2.先对input数据进行处理，因为它受时程/IM format的影响；
% 后对output数据进行讨论处理，因为它受问题类型影响；

% ==== Determine the features ====
feature_selecte = options.feature_selecte;               % 'arti_type', 'white_type', 'all' (determine the input variables)
input_name = options.input_name; 
% ==== The para. for normlized ====
data_log = options.data_log;                             % 'true', 'false'  (whether trained with log format)
norm_input_type = options.norm_input_type;               % '0-1', 'MAX', 'mu-sigma'
norm_output_type = options.norm_output_type;             % '0-1', 'MAX', 'mu-sigma'

% ==== the para. for split dataset ====
develop_amount = options.amount;                         % the data for training/test process; while the remaining corre. to predict
train_ratio = options.train_ratio;                       % the ratio of training subset in dataset; while (1-ratio) corre. to test subset
threshold = options.threshold;                           % the value of yielding curvature /or the damage state threshold


    % ==== Step 1: speify the seleted features ====
%     feature_selecte = 'arti_type';  %'arti_type','white_type','all'
    if strcmp(options.input_format,'IM')
        if strcmp(feature_selecte,'arti_type')

            % ==== The discenpancies for different scenarios ====
            if strcmp(input_name,'white_input') 
                feature_index = [22	5	7	18	19	16	17	23	2	20	6	25];  % features of white noise
            elseif strcmp(input_name,'white_input_nonlinear') 
                feature_index = [22	5	7	18	19	16	17	23	2	20	6	25];  % features of white noise
            elseif strcmp(input_name,'arti_input')
                feature_index = [22	5	7	18	19	16	17	23	2	20	6 25 26];       % features of artificial
            elseif strcmp(input_name,'arti_input_4')
                feature_index = [22	5	7	18	19	16	17	23	2	20	6 ];       % features of single site excitations
            else
                feature_index = [22	5	7	18	19	16	17	23	2	20	6 25 26]; % features of near fault
            end

            data_input = data_input(:,feature_index);

        elseif strcmp(feature_selecte,'white_type')
            feature_index = [2	5	9	23	3	11	18	7	21	17	13 25 26];   % features of near_fault_F
            data_input = data_input(:,feature_index);
            % 待指定
        end
    end
    
    
    % ==== Step 2: Label and Normalized ====
    % >>>>根据不同问题类型，来决定是否 Label 和对 output 是否 Normalization。
    % >>>>regression 问题不需要label，但此处仍赋予一个全为1的向量，来保证格式的统一。
    % >>>>最终得到data_input, data_output, data_output_cata. 
    
    % ==== Step 2.1: Label process ====
    if strcmp(options.problem_type, 'regression')                          % catagorizing the data except regression tasks.
        data_output_cata = ones(size(data_output) );
        
    % >>>> Classification problem
    else
        [data_output_cata] = f_Label_process(data_output, threshold);
        
        if strcmp(options.ML_name, 'ANN')                                  % transform the data of one-hot format for ANN model                 
            data_output_cata = f_one_hot_format(data_output_cata);
        end
            
    end
    
    
    % ==== Step 2.2: Normalizing the dataset ====
    [data_input,input_scale] = f_Normalized_data( data_input,norm_input_type,data_log);
    
    if strcmp(options.problem_type, 'classification')                      % catagorizing the data except classification tasks.
        output_scale = 0;
        
    %>>>> Regression && Class-Reg problem
    else
        
        [data_output,output_scale] = f_Normalized_data( data_output,norm_output_type,data_log);

    end
    
    
    % ==== Step 3: Specified data format for specified model ====
    if strcmp (options.ML_name, 'LSTM')
        [data_input,data_output,data_output_cata]...
                             = f_LSTM_dataformat(data_input,data_output,data_output_cata,options );
                         
    end
    
    
    % ==== Step 4: Split ==== 
    data_summary = f_partition(data_input,data_output,data_output_cata,develop_amount,train_ratio,options);
    
end

%% ======================= sub-function =============================

function data_summary = f_partition(data_input,data_output,data_output_cata,develop_amount,train_ratio,options)

    % ==== data index for develop/predict process ====
    rng(options.seeds)
    develop_index = randperm(size(data_input,1), develop_amount);
    predict_index = setdiff(1:1:size(data_input,1), develop_index);
    
     % ==== data for develop process ====
    train_amount = develop_amount * train_ratio;
    test_amount = develop_amount - train_amount;
    
    train_index = randperm(develop_amount, train_amount);
    test_index = setdiff(1:1:develop_amount, train_index);
    
    % ==== split the dataset ====
    if strcmp (options.problem_type,'class_regression')
        data_input_temp = data_input(develop_index,:);                 % 该subset用于生成 Train-test subset
        data_output_temp = data_output(develop_index,:);
        data_output_cata_temp = data_output_cata(develop_index,:);
        
        train_input = data_input_temp(train_index,:);                            % Train/Test subset
        train_output = data_output_temp(train_index,:);
        train_output_cata = data_output_cata_temp(train_index,:);

        test_input = data_input_temp(test_index,:);
        test_output = data_output_temp(test_index,:);
        test_output_cata = data_output_cata_temp(test_index,:);
        
        predict_input = data_input(predict_index,:);                    % 用于分割得到 validate/predict subset
        predict_output = data_output(predict_index,:);
        predict_output_cata = data_output_cata(predict_index,:);
        
        % ==== summary ====
        data_summary.train_input = train_input;
        data_summary.train_output = train_output;
        data_summary.train_output_cata = train_output_cata;

        data_summary.test_input = test_input;
        data_summary.test_output = test_output;
        data_summary.test_output_cata = test_output_cata;
        
        data_summary.predict_input = predict_input;
        data_summary.predict_output = predict_output;
        data_summary.predict_output_cata = predict_output_cata;
        
    elseif strcmp (options.problem_type,'classification')
        data_input_temp = data_input(develop_index,:);                 % 该subset用于生成 Train-test subset
        data_output_cata_temp = data_output_cata(develop_index,:);
        
        train_input = data_input_temp(train_index,:);                  % Train/Test subset
        train_output_cata = data_output_cata_temp(train_index,:);

        test_input = data_input_temp(test_index,:);
        test_output_cata = data_output_cata_temp(test_index,:);

        predict_input = data_input(predict_index,:);                    % 用于分割得到 validate/predict subset
        predict_output_cata = data_output_cata(predict_index,:);
        
        % ==== summary ====
        data_summary.train_input = train_input;
        data_summary.train_output = train_output_cata;
        
        data_summary.test_input = test_input;
        data_summary.test_output = test_output_cata;
 
        data_summary.predict_input = predict_input;
        data_summary.predict_output = predict_output_cata;
        
    else  
        data_input_temp = data_input(develop_index,:);                  % 该subset用于生成 Train-test subset
        data_output_temp = data_output(develop_index,:);
       
        train_input = data_input_temp(train_index,:);                   % Train/Test subset
        train_output = data_output_temp(train_index,:);

        test_input = data_input_temp(test_index,:);
        test_output = data_output_temp(test_index,:);
        
        predict_input = data_input(predict_index,:);                    % 用于分割得到 validate/predict subset
        predict_output = data_output(predict_index,:);
        
        % ==== summary ====
        data_summary.train_input = train_input;
        data_summary.train_output = train_output;

        data_summary.test_input = test_input;
        data_summary.test_output = test_output;
        
        data_summary.predict_input = predict_input;
        data_summary.predict_output = predict_output;

        
    end
    
    % save('path/data_summary.mat','data_summary')
    
end


function  [input_norm,scale] = f_Normalized_data( input,norm_type,data_log)

    if data_log                    % ture or flase
        input = log(input);
        threshold  = log(1);
    else
        threshold = 1;
    end
    
    if size(input,2) == 1
       switch norm_type               % specify
        case '0_1_whole'      
            input_max = max(input(:) );
            input_min = min(input(:) );
            input_norm = (input - input_min)/(input_max - input_min);
            threshold = (threshold - input_min)/(input_max - input_min);
            scale.min = input_min;
            scale.max = input_max;
            scale.threshold = threshold;
            
        case '0_1_each_column'
            num=size(input,1);      
            input_max = max(input,[],1);
            input_min = min(input,[],1 );
            input_max_repmat = repmat(input_max,[num,1] );
            input_min_repmat = repmat(input_min,[num,1] );
            input_norm = (input- input_min_repmat)./(input_max_repmat-input_min_repmat);
            threshold = (threshold- input_min_repmat)./(input_max_repmat-input_min_repmat);
            
            scale.min = input_min_repmat;
            scale.max = input_max_repmat;
            scale.threshold = threshold;
            
        case 'MAX_whole'
            num=size(input,1);
            input_max = max(input(:)  );
            input_norm = input/input_max;
            threshold = threshold/input_max;
            
            scale.max = input_max;
            scale.threshold = threshold;
            
            
        case 'MAX_each_column'
            num=size(input,1);      
            input_max = max(input,[],1);
            input_max_repmat = repmat(input_max,[num,1] );
            input_norm = input./input_max_repmat;
            threshold = threshold./input_max_repmat;
            
            scale.max = input_max_repmat;
            scale.threshold = threshold;
            
            
            
        case 'mu_sigma_whole'
            mu = mean(input(:) );
            sigma = std(input(:) );
            input_norm = (input - mu )/sigma;
            threshold = (threshold - mu )/sigma;
            
            scale.mu = mu;
            scale.sigma = sigma;
            scale.threshold = threshold;
            
        case 'mu_sigma_each_column'
            num = size(input,1);
            mu = mean(input,1 );
            sigma = std(input,1 );
            mu_repmat = repmat(mu,[num,1] );
            sigma_repmat = repmat(sigma,[num,1]);
            input_norm = (input - mu_repmat)./sigma_repmat;
            threshold = (threshold - mu_repmat)./sigma_repmat;
            
            scale.mu = mu_repmat;
            scale.sigma = sigma_repmat;
            scale.threshold = threshold;
            
        end %swith 
        
    else
        
        switch norm_type               % specify
            case '0_1_whole'      
                input_max = max(input(:) );
                input_min = min(input(:) );
                input_norm = (input - input_min)/(input_max - input_min);
                scale.min = input_min;
                scale.max = input_max;

            case '0_1_each_column'
                num=size(input,1);      
                input_max = max(input,[],1);
                input_min = min(input,[],1 );
                input_max_repmat = repmat(input_max,[num,1] );
                input_min_repmat = repmat(input_min,[num,1] );
                input_norm = (input- input_min_repmat)./(input_max_repmat-input_min_repmat);

                scale.min = input_min_repmat;
                scale.max = input_max_repmat;

            case 'MAX_whole'
                num=size(input,1);
                input_max = max(input(:)  );
                input_norm = input/input_max;
                scale.max = input_max;

            case 'MAX_each_column'
                num=size(input,1);      
                input_max = max(input,[],1);
                input_max_repmat = repmat(input_max,[num,1] );
                input_norm = input./input_max_repmat;
                scale.max = input_max_repmat;

            case 'mu_sigma_whole'
                mu = mean(input(:) );
                sigma = std(input(:) );
                input_norm = (input - mu )/sigma;
                scale.mu = mu;
                scale.sigma = sigma;

            case 'mu_sigma_each_column'
                num = size(input,1);
                mu = mean(input,1 );
                sigma = std(input,1 );
                mu_repmat = repmat(mu,[num,1] );
                sigma_repmat = repmat(sigma,[num,1]);
                input_norm = (input - mu_repmat)./sigma_repmat;

                scale.mu = mu_repmat;
                scale.sigma = sigma_repmat;
        end %swith
    
    end % if size() == 1

end


function [data_output_cata] = f_Label_process(data_output, threshold)
    
    if length(threshold) == 1
        % ==== split and specify catagory on linear/yield ====
        data_output_cata = zeros(size(data_output,1),1 );
        linear_index = find(data_output < threshold);
        yield_index = find(data_output >= threshold);
        data_output_cata(linear_index) = 1;
        data_output_cata(yield_index) = 2;
    
    elseif length(threshold) == 4
        t1 = threshold(1); t2 = threshold(2); t3 =  threshold(3); t4 =  threshold(4); 
        
        slight = threshold(1); moderate = threshold(2); 
        extensive = threshold(3); collapse = threshold(4);
        
        data_output_cata = zeros(size(data_output,1),1 );
        
        % ==== index ====
        slight = find(data_output > t1 && data_output <t2 );
        moderate = find(data_output > t2 && data_output <t3 );
        extensive =  find(data_output > t3 && data_output <t4 );
        collapse = find(data_output > t4 && data_output <t4 );
        
        data_output_cata(slight) = 1;
        data_output_cata(moderate) = 2;
        data_output_cata(extensive) = 3;
        data_output_cata(collapse) = 4;
        
    end
    

end


function [data_input_cell,data_output_cell,data_output_cata_cell]...
            = f_LSTM_dataformat(data_input,data_output,data_output_cata,options )
        
    [row,column] = size(data_output);
    num = size(data_input,2);
    
    % ==== 1. transform the input of cell format ====
    data_input_cell =  num2cell(data_input,2);                      % 每行数据转为一个cell
    
    % ==== 2. transform the output of cell format ====
    if strcmp(options.problem_type, 'classification')
        data_output_cell = data_output;                             % 分类问题data_output 不用转换
        data_output_cata_cell = categorical(data_output_cata);           % 类别需转换为categories形式。       
    
    %>>>> Regression problem 
    else
        if column > 1 && column < 50                                % 此时output为envelope
            for i = 1:row
                data_temp = data_output(i,:)';
                data_output_cell{i,1} = repmat(data_temp,[1,num]);
                data_output_cata_cell = data_output_cata;           % data_output_cata不用转换
            end
        
        elseif  column > 100                                        % 此时output为timehis
            data_output_cell = num2cell(data_output,2);
            data_output_cata_cell = data_output_cata;
            
        elseif column == 1
            data_output_cell = data_output;
            data_output_cata_cell = data_output_cata;
        end
        assert(column < 50 || column > 100 , 'EDP为envelop时，点数过多')
    end
    

end


function data_one_hot = f_one_hot_format(data)
    row = size(data,1);
    max_value = max(data);
    data_one_hot = zeros(row,max_value);
    
    for i = 1:row
        data_one_hot(i,data(i)) = 1;
        
    end


end
