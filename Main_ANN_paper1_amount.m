clear all
time_1 = clock;
%% ================= Step 1.1 Load the dataset ======================
% 2 add the measures on time history into accuracy function.
%>>>> Note: change the name of arti/white/near_fault
excitation_type = 'artificial_seismic_wave';        % 'artificial_seismic_wave','white_noise', 'near_fault'
input_format = 'IM';                                % 'IM', 'time_history'
input_name = 'arti_input';                          % 'arti_input', 'white_input_timehis'
output_name = 'arti_F_envelop';                   %  'arti_F_envelop','white_F_timehis',...

                                                    %  'near_fault_M_envelop','near_fault_M_timehis',...
                                                    %  'near_fault_CD_base_max','near_fault_CD_base_timehis',near_fault_CD_envelop...
                                                    %  'near_fault_Dis_timehis')

[data_input,data_output] = f_data_source_selected(excitation_type,input_name,output_name);

% ==== save the results of 'training_process' ====
save_data = false;                                  % 'true' or 'false'  
document_name = [output_name '.mat' ]  ;
path = ['C:\Users\cxtj1\Desktop\matlab_fragility\Paper1_Nonlinear_response\A_ML_results\' 'ANN\' 'white_noise\' document_name];  


%% ================ Step 1.2 Pre-process for dataset ================
% ==== trail analysis of the dataset amount ====
A_amount_train_result = [];
A_amount_test_result = [];
A_amount_pre_result = [];

% block = [100 300 500 800 1200 1600 2000]
for block = [100 300 500 800 1200 1600 2000]
disp(['dataset amount is ' num2str(block)])
disp([])
    
AA_train_trail_analysis = [];
AA_test_trail_analysis = [];
AA_pre_trail_analysis = [];    

for block_1 = 1:5

% ==== The para. for split dataset ====
options.amount = block;                                % the data for training/test process; while the remaining corre. to predict
options.train_ratio = 0.7;                           % the ratio of training subset in dataset; while (1-ratio) corre. to test subset
options.seeds = round(rand()*1000 );
    
% ==== Determine the Model/ Features ====
options.ML_name = 'ANN';                            % 'ANN','LSTM'
options.problem_type = 'regression';                 % 'classification', 'regression', 'class_regression'
options.feature_selecte = 'arti_type';               % 'arti_type'/ 'white_type'/ 'all' (determine the input variables)
options.input_name = input_name;                     % '' 
options.input_format = input_format;                 % 'IM', 'time_history'

% ==== The para. for normlized ====
options.data_log = false;                             % 'true'/ 'false'  (whether trained with log format)
options.norm_input_type = '0_1_each_column';          % '0_1_whole', '0_1_each_column'/ 'MAX_whole', 'MAX_each_column'/ 'mu_sigma_whole', 'mu_sigma_each_column'
options.norm_output_type = 'MAX_whole';                
Restore_output_data = false;                         % 'false'/ 'true' (whether estimate performance on Restored output data)

% ==== Threshold for catagorizing (Label) process ====
options.threshold = 1;                               % the value of yielding curvature:1 /or the damage state threshold: [1 1.3 4.7 14]

if strcmp(options.problem_type,'classification')
    assert(size(data_output,2)==1 , 'output_name must be (arti/white/NF)_CD_base_max for the classification' )
    
end


[data_summary,input_scale,output_scale] = f_Split_normalized(data_input,data_output,options );

% ==== PCA or not ====
% [coeff,score,latent,~,explained,mean_data] = pca(data_input,'NumComponents',20);
% data_input = score; 
% scale.coeff = coeff;


%% ================ Step 1.3 Specify the current process =============
process_type = 'training_process';             % 'cross_validate_process', 'training_process'

% ==== Para. for K-fold (cross_validate_process) ====
K = 5; repeat_epoch = 3;                            % 'K-fold'

% ==== Para. for training/test process ====
model_amount = 3;                                   % the total amount of trained models, and the final results corre. their average values.


%% ======== Step 1.4 specify hyper-para. of machine learning =========
AA_two_trail_analysis = [];
AA_two_train_trail_analysis = [];
AA_two_test_trail_analysis = [];
AA_two_pre_trail_analysis = [];


numHiddenUnits = 30;                                           % specify the hidden neurons
maxEpochs = 300;
miniBatchSize = 20;

drop_rate = 0.3;
learning_rate = 0.01;

% numHiddenUnits = [5	10	15	20	25	30	40	50	70]

% ==== 1. Adjust layer struct ====
featureDimension = size(data_summary.train_input,2);
if iscell(data_summary.train_output)
    numResponses = size(data_summary.train_output{1},2);        % 当为cell格式时；
else
    numResponses = size(data_summary.train_output(1,:),2);      % 当为numeric格式时；
end

%>>>> Layers
if strcmp(options.problem_type,'regression')
layers = [ ...
    featureInputLayer(featureDimension)

    
    fullyConnectedLayer(numHiddenUnits)
    batchNormalizationLayer
    reluLayer
   
    
    dropoutLayer(drop_rate)
    fullyConnectedLayer(numResponses)
    regressionLayer];

% gruLayer
% batchNormalizationLayer
% bilstmLayer(numHiddenUnits,'OutputMode','sequence')
% fullyConnectedLayer(70)
%     tanhLayer
%     reluLayer
%     leakyReluLayer
%     clippedReluLayer
%     eluLayer
%     dropoutLayer(0.5)

% >>>> Classification
else
    layers = [ ...
    sequenceInputLayer(featureDimension)
    lstmLayer(numHiddenUnits,'OutputMode','last')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];
    
end

% if size(data_output,2) == 1
%     assert(strcmp(layers{3}.OutputMode,'last'), 'the OutputMode must be last' )
% end


% ==== 2. Speicfy the hyper-parameter for options ====
XValidation = data_summary.predict_input;
YValidation = data_summary.predict_output;

LSTM_options = trainingOptions('adam', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'Shuffle','every-epoch', ...
    'InitialLearnRate',learning_rate, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',1, ...
    'LearnRateDropFactor',0.99, ...
    'GradientThreshold',1, ...
    'Verbose',0);


% 'ValidationData',{XValidation,YValidation},...
% 'Shuffle','never', ...
% 'Plots','training-progress',...
%     'Momentum',0.95,...
% 'SquaredGradientDecayFactor',0.99,...

%% =============== Step 2.1 Cross validation process ================
if strcmp(process_type,'cross_validate_process')
    accuracy_all = [];
    accuracy_summary = [];
    prediction_epoch_all = cell(1,repeat_epoch);
    Model_all_save = cell(1,repeat_epoch);
    
    for epoch_num = 1:repeat_epoch
       index_cross = Cross_validate(data_summary.train_input,'K_fold',K);   % obtain cross_validate index
       for cross_num = 1:K
           disp([num2str(cross_num) 'th CV/' num2str(epoch_num) 'th repeate' ])
           % ==== Split the dataset ====
          train_index = find(index_cross ~= cross_num );
          test_index = find(index_cross == cross_num );

          train_x = data_summary.train_input(train_index,: );
          train_y = data_summary.train_output(train_index,: );
          test_x = data_summary.train_input(test_index,: );
          test_y = data_summary.train_output(test_index,: );
            
          % ==== Training process ====
          net = trainNetwork(train_x,train_y,layers,LSTM_options);
          YPred = predict(net,test_x,'MiniBatchSize',1);
          
          % ==== Cell to numerical ====
          if iscell(YPred) 
                [YPred] = f_cell_to_num(YPred );                          % 数据为envelop时的转换形式
          end
          
          if iscell(test_y)
                [test_y] = f_cell_to_num(test_y );                          % 数据为envelop时的转换形式
          end
          
          % ==== Whether Restore data format ====
          if Restore_output_data
%             [train_x] = f_Restore_input_format(train_x,options,input_scale)

            if ~strcmp(options.problem_type,'classification')
                [test_y,YPred] = f_Restore_output_format(test_y,YPred,options,output_scale);
            end
            
          end
          
          % ==== Estimate performace ====
          [accuracy_struct,accuracy_metrics] = Accuracy_estimate(test_y,YPred,options);
          
          if strcmp(options.problem_type,'regression')

               % ==== regresssion ====
               accuracy_all = [accuracy_all;accuracy_metrics];      % performance
               Model_all_save{epoch_num} = net;                % save the trained model
               label_epoch_all{epoch_num} = test_y;                 % save the data (actual/ prediction)
               prediction_epoch_all{epoch_num} = YPred;
               
           else
               % ==== classification ====
               Model_all_save{epoch_num} = net;                % save the trained model
               
               accuracy = accuracy_struct.accuracy;
               confusion_mat = accuracy_struct.confusion_mat;
               order = accuracy_struct.confusion_order;

               accuracy_summary = [accuracy_summary; accuracy];
               confusion_summary{epoch_num} = confusion_mat;

               accuracy_all.accuracy_summary = accuracy_summary;
               accuracy_all.confusion_summary = confusion_summary;
               accuracy_all.order = order;
              
           end % if strcmp(opts.problem_type)
   
       end % for cross_num = 1:K
       
    end % for epoch_num = 1:repeat_epoch
    
    
     % ==== compute the average results ====
    if strcmp(options.problem_type,'regression')
        accuracy_mean_all = mean(accuracy_all,1);
    
    else
        accuracy_mean_all = mean(accuracy_summary,1);
        
    end
    
    
end % if strcmp(type)


%% =================== Step 2.2 training process ====================
if strcmp(process_type,'training_process')
    accuracy_train_all = [];
    accuracy_vali_all = [];
    accuracy_test_all = [];
    accuracy_pre_all = [];
    accuracy_summary_train = [];
    accuracy_summary_vali = [];
    accuracy_summary_test = [];
    accuracy_summary_pre = [];
    
    for epoch_num = 1:model_amount
        disp([num2str(epoch_num) 'th model developing'])
        
        train_x = data_summary.train_input;
        train_y = data_summary.train_output;
        %          validate_x = data_summary.validate_input;
        %          validate_y = data_summary.validate_output;
        test_x = data_summary.test_input;
        test_y = data_summary.test_output;
        predict_x = data_summary.test_input;
        predict_y = data_summary.test_output;

        
        % ==== Training process ====
        net = trainNetwork(train_x,train_y,layers,LSTM_options);

        %>>>> 1. Performance on training subset
        prediction_training = predict(net,train_x,'MiniBatchSize',1);
        
        % ==== Cell to numerical ====
        if iscell(prediction_training) 
            [prediction_training] = f_cell_to_num(prediction_training );                          % 数据为envelop时的转换形式
        end

        if iscell(train_y)
            [train_y] = f_cell_to_num(train_y );                          % 数据为envelop时的转换形式
        end

        % ==== Whether Restore data format ====
        if Restore_output_data 
%             [train_x] = f_Restore_input_format(train_x,options,input_scale) ; 
            
            if ~strcmp(options.problem_type,'classification')                     % The process is only required for regression tasks 
                [train_y,prediction_training] = f_Restore_output_format(train_y,prediction_training,options,output_scale);
            end
            
        end
        
        [accuracy_struct_train,accuracy_metrics_train] = Accuracy_estimate(train_y,prediction_training,options);
        
        
        %>>>> 2. Performance on test subset
        prediction_test = predict(net,test_x,'MiniBatchSize',1);
         
        % ==== Cell to numerical ====
        if iscell(prediction_test) 
            [prediction_test] = f_cell_to_num(prediction_test );                          % 数据为envelop时的转换形式
        end

        if iscell(test_y)
            [test_y] = f_cell_to_num(test_y );                          % 数据为envelop时的转换形式
        end
        
        % ==== Whether Restore data format ====
          if Restore_output_data
%             [test_x] = f_Restore_input_format(test_x,options,input_scale);
            if ~strcmp(options.problem_type,'classification')
                [test_y,prediction_test] = f_Restore_output_format(test_y,prediction_test,options,output_scale);
            end
          end
         
        [accuracy_struct_test,accuracy_metrics_test] = Accuracy_estimate(test_y,prediction_test,options);
        
        
        %>>>> 3. Performance on predict subset
        prediction_pre = predict(net,predict_x,'MiniBatchSize',1);
        
        % ==== Cell to numerical ====
        if iscell(prediction_pre) 
            [prediction_pre] = f_cell_to_num(prediction_pre );                          % 数据为envelop时的转换形式
        end

        if iscell(predict_y)
            [predict_y] = f_cell_to_num(predict_y );                          % 数据为envelop时的转换形式
        end
        
        % ==== Whether Restore data format ====
          if Restore_output_data
%             [predict_x] = f_Restore_input_format(predict_x,options,input_scale);
            if ~strcmp(options.problem_type,'classification')
                [predict_y,prediction_pre] = f_Restore_output_format(predict_y,prediction_pre,options,output_scale);
            end
            
          end
        
          [accuracy_struct_pre,accuracy_metrics_pre] = Accuracy_estimate(predict_y,prediction_pre,options);
        
        
        
        % ==== summary of the result of each epoch ===
        Model_all_save{epoch_num} = net;                % save the trained model
        
        if strcmp(options.problem_type,'regression')           
         % ==== 1.regresssion ====
           accuracy_train_all = [accuracy_train_all;accuracy_metrics_train];      % performance
%            accuracy_vali_all = [accuracy_vali_all;accuracy_metrics_vali];      % performance
           accuracy_test_all = [accuracy_test_all;accuracy_metrics_test];      % performance
           accuracy_pre_all = [accuracy_pre_all;accuracy_metrics_pre];      % performance
           
%{          
%            prediction_train_all(:,:,epoch_num) = prediction_training;        % predict value   
% %            prediction_vali_all(:,:,epoch_num) = prediction_validate;
%            prediction_test_all(:,:,epoch_num) = prediction_test;
%            prediction_pre_all(:,:,epoch_num) = prediction_pre;
%}
           
           %>>>> mean value of above accuracy matrix
           accuracy_train_mean = mean(accuracy_train_all,1);
%            accuracy_vali_mean = mean(accuracy_vali_all,1);
           accuracy_test_mean = mean(accuracy_test_all,1);
           accuracy_pre_mean = mean(accuracy_pre_all,1);
           
%{           
%            pre_train_average = mean(prediction_train_all,3);
% %            pre_vali_average = mean(prediction_vali_all,3);
%            pre_test_average = mean(prediction_test_all,3);
%            pre_pre_average = mean(prediction_pre_all,3);
%}

         else
          % ==== 2. classification ====
            accuracy_train = accuracy_struct_train.accuracy;
            confusion_mat_train = accuracy_struct_train.confusion_mat;
            
%             accuracy_vali = accuracy_struct_vali.accuracy;
%             confusion_mat_vali = accuracy_struct_vali.confusion_mat;
            
            accuracy_test = accuracy_struct_test.accuracy;
            confusion_mat_test = accuracy_struct_test.confusion_mat;
            
            accuracy_pre = accuracy_struct_pre.accuracy;
            confusion_mat_pre = accuracy_struct_pre.confusion_mat;
            order = accuracy_struct_train.confusion_order;

            %>>>> summary
            accuracy_summary_train = [accuracy_summary_train; accuracy_train];
            confusion_summary_train(:,:,epoch_num) = confusion_mat_train;
            
%             accuracy_summary_vali = [accuracy_summary_vali; accuracy_vali];
%             confusion_summary_vali(:,:,epoch_num) = confusion_mat_vali;
            
            accuracy_summary_test = [accuracy_summary_test; accuracy_test];
            confusion_summary_test(:,:,epoch_num) = confusion_mat_test;
            
            accuracy_summary_pre = [accuracy_summary_pre; accuracy_pre];
            confusion_summary_pre(:,:,epoch_num) = confusion_mat_pre;
            
            %>>>> summary with struct format
            accuracy_all.accuracy_summary_train = accuracy_summary_train;
            accuracy_all.confusion_summary_train = confusion_summary_train;
%             accuracy_all.accuracy_summary_vali = accuracy_summary_vali;
%             accuracy_all.confusion_summary_vali = confusion_summary_vali;
            accuracy_all.accuracy_summary_test = accuracy_summary_test;
            accuracy_all.confusion_summary_test = confusion_summary_test;
            accuracy_all.accuracy_summary_pre = accuracy_summary_pre;
            accuracy_all.confusion_summary_pre = confusion_summary_pre;
            
            accuracy_all.order = order;
           
         end % if strcmp(problem_type)  

    end % for 1:Model_epoch

%{    
     % ==== summary with struct format ====
     % ====      save the model        ====
     if strcmp(options.problem_type,'regression')  
        A_label_prediction_value.train_input = train_x;
        A_label_prediction_value.train_label = train_y;                    % label
        A_label_prediction_value.train_predict = pre_train_average;        % average prediction
        
%         A_label_prediction_value.vali_label = validate_y;
%         A_label_prediction_value.vali_predict = pre_vali_average;

        A_label_prediction_value.test_input = test_x;
        A_label_prediction_value.test_label = test_y;
        A_label_prediction_value.test_predict = pre_test_average;
        
        A_label_prediction_value.predict_input = predict_x;
        A_label_prediction_value.predict_label = predict_y;
        A_label_prediction_value.pre_predict = pre_pre_average;
        
        A_accuracy_results.accuracy_train_all = accuracy_train_all;          % accuracy of each model
%         A_accuracy_results.accuracy_vali_all = accuracy_vali_all;
        A_accuracy_results.accuracy_test_all = accuracy_test_all;
        A_accuracy_results.accuracy_pre_all = accuracy_pre_all;
        
        A_accuracy_results.accuracy_train_mean = accuracy_train_mean;        % accuracy -> mean value
%         A_accuracy_results.accuracy_vali_mean = accuracy_vali_mean;
        A_accuracy_results.accuracy_test_mean = accuracy_test_mean;
        A_accuracy_results.accuracy_pre_mean = accuracy_pre_mean;
    
        %>>>> save the model
        if save_data
            save(path, 'Model_all_save','A_label_prediction_value','A_accuracy_results', 'data_summary','input_scale','output_scale')
        end
        
     else 
         % ==== classification ====
    %         save(path , 'ANN_all_save','accuracy_all', 'data_summary','input_scale','output_scale')

    
     end % if strcmp(problem_type)
%}
    
end


%% ============  Summary of the different output cases =============
    if strcmp(process_type,'cross_validate_process')
        AA_train_trail_analysis = [AA_train_trail_analysis; accuracy_mean_all];

    else
        AA_train_trail_analysis = [AA_train_trail_analysis; accuracy_train_mean];
        AA_test_trail_analysis = [AA_test_trail_analysis; accuracy_test_mean];
        AA_pre_trail_analysis = [AA_pre_trail_analysis; accuracy_pre_mean];
    end

    size(AA_train_trail_analysis)
    
end % the first 'for' round


%% ============== Summary of the different output cases =============
    A_amount_train_result = [A_amount_train_result; mean(AA_train_trail_analysis,1)];
    A_amount_test_result = [A_amount_test_result; mean(AA_test_trail_analysis,1)];
    A_amount_pre_result = [A_amount_pre_result; mean(AA_pre_trail_analysis,1)];
    
    disp(['summary size is: ' num2str(size(A_amount_train_result) )  ] )
    disp([ ])
end % the second 'for' repeate




%{
% %% ================== Step 3: Separate the results ==================
% if exist('AA_two_trail_analysis') && length(size(AA_two_trail_analysis) ) == 3     % 将cross-validation的每个结果（accuracy,R^2,RMSE等）分离至cell中的第i个位置，每一行为第二个for循环的变量，这里指epochs
%                                                                                   % 不同的列代表不同的神经元数。
%     amount_temp = size(AA_two_trail_analysis,2);
%     for i = 1:amount_temp 
%         AA_final_trail_results{i} = squeeze(AA_two_trail_analysis(:,i,:) );
%         
%     end
%     
%     
% elseif exist('AA_two_pre_trail_analysis') && length(size(AA_two_pre_trail_analysis) ) == 3
%     
%     amount_temp = size(AA_two_pre_trail_analysis,3);
%     for i = 1:amount_temp
%         AA_final_train_trail_results{i} = squeeze(AA_two_train_trail_analysis(:,i,:) );
%         AA_final_test_trail_results{i} = squeeze(AA_two_test_trail_analysis(:,i,:) );
%         AA_final_pre_trail_results{i} = squeeze(AA_two_pre_trail_analysis(:,i,:) );
% 
%     end
%     
% end
%}   


%%

time_2 = clock;
time_cost = etime(time_2,time_1)

load splat
sound(y,Fs)
% system('shutdown -s -t 3600')



