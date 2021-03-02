% clear all

%% ====================== Part 1: specific input cases ====================

% response_actual = rand(20,20);
% response_LSTM = rand(20,20);
% response_ANN = rand(20,20);
% 
% % ==== parameter ====
% n = 20;                                          % 质点数
% h = 1:1:n;
% 
% % ==== plot the dynamic response/demand : 20 in total ====
% rand_num = randperm(1500,20);
% 
% for i = 1: 20
%     subplot(4,5,i)
%     index = rand_num(i);
%     plot(response_actual(index,:),h,'k--' ); hold on
%     plot(response_LSTM(index,:),h,'r' ); 
%     plot(response_ANN(index,:),h,'b' );                       
%     
% end

%% >>>> 1.1： part1 中的一些快捷代码
% >>>> 随机选30条看看整体情况
% rand_num = randperm(1500,30);
% for i = 1: 30
%     index = rand_num(i);
%     plot(response_actual(index,:),h,'k--' ); hold on
%     plot(response_LSTM(index,:),h,'r' ); 
%     plot(response_ANN(index,:),h,'b' );
%     
% end

% >>>> 将选择的特定波赋给新变量
% response_selected = [];
% selected_num = 791;
% response_selected(:,1) =  response_actual(selected_num,:);
% response_selected(:,2) =  response_ANN(selected_num,:);
% response_selected(:,3) =  response_LSTM(selected_num,:);
% 
% % >>>> 通过画图确认新变量中的数据，用于绘制F，M
% plot(response_selected(:,1),h,'k--' ); hold on
% plot(response_selected(:,2),h,'r' ); 
% plot(response_selected(:,3),h,'b' );
% 
% % >>>> 用于绘制曲率
% plot(response_selected(:,1),response_selected(:,3),'*'); hold on;
% plot(-2:5,-2:5)
% axis([-2,5,-2,5])




%% ============ Part 2: statistical results (mu +- sigma) ===========
%>>>> single case：输入相应的响应 
response_actual = predict_y;
response_LSTM = pre_pre_average;
response_ANN = pre_pre_average;

% %>>>> main
% %>>>> statistics_actual会返回沿墩身每个质点的均值，及其对应的加减一倍 sigma 的 envelope
statistics_actual = f_statistics(response_actual);
statistics_LSTM = f_statistics(response_LSTM);
statistics_ANN = f_statistics(response_ANN);



