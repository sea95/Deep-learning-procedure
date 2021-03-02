function [data_input,data_output,path] = f_data_source_selected(excitation_type,input_name,output_name)

% Note: 数据均一行为一组数据；F，M，Displacement 的时程数据暂时只取base的数据。

% excitation_type = 'artificial_seismic_wave';        % 'white_noise', 'near_fault'
% input_name = 'arti_input';                          % 'arti_input', 'arti_input_timehis'
% output_name = 'arti_F_envelop';                     %  'near_fault_F_envelop','near_fault_F_timehis',...
                                                      %  'near_fault_M_envelop','near_fault_M_timehis',...
                                                      %  'near_fault_CD_base_max','near_fault_CD_base_timehis',...
                                                      %  'near_fault_Dis_timehis')

                                                      
% arti feature: {  Columns={'PGA','PGV','PGD','D90','AI','CAV','SED','SA_1','SA_2','SV_1',
                 %           'SV_2','SD_1','SD_2','SAmax/SA_1','SAmax/SA_2','SVmax/SV_1','SVmax/SV_2','SDmax/SD_1','SDmax/SD_2' 'SA_01s'
                 %            ,'EPA','EPV',H,L,Tg,Cs};

% white feature: {  Columns={'PGA','PGV','PGD','D90','AI','CAV','SED','SA_1','SA_2','SV_1',
                 %           'SV_2','SD_1','SD_2','SAmax/SA_1','SAmax/SA_2','SVmax/SV_1','SVmax/SV_2','SDmax/SD_1','SDmax/SD_2' 'SA_01s'
                 %            ,'EPA','EPV',H,L,S0};
                 
% near_fault feature: {  Columns={'PGA','PGV','PGD','D90','AI','CAV','SED','SA_1','SA_2','SV_1',
                 %           'SV_2','SD_1','SD_2','SAmax/SA_1','SAmax/SA_2','SVmax/SV_1','SVmax/SV_2','SDmax/SD_1','SDmax/SD_2' 'SA_01s'
                 %            ,'EPA','EPV',H,L,a,T};                 

                                                         
path_arti_data = 'C:\Users\cxtj1\Desktop\matlab_fragility\Paper1_Nonlinear_response\A_data_source\artificial_seismic_wave\arti_hybrid.mat';
path_white_data = 'C:\Users\cxtj1\Desktop\matlab_fragility\Paper1_Nonlinear_response\A_data_source\white_noise\white_noise_nonlinear.mat';
path_near_fault_data = 'C:\Users\cxtj1\Desktop\matlab_fragility\Paper1_Nonlinear_response\A_data_source\near_fault\near_fault';

curvature_yield = 0.000828;                         %rad/m
curvature_ultimate = 0.009988;

switch excitation_type
    
    % ==== artificial seismic wave ====
    case 'artificial_seismic_wave'
        load(path_arti_data)
        
        
        %>>>> input specified
        if strcmp(input_name,'arti_input')
            data_input = arti_input; 
            
        elseif strcmp(input_name,'arti_input_timehis')
            data_input = arti_input_timehis; 
        end  
             
        %>>>> output specified
        if strcmp(output_name,'arti_F_envelop')                %>> F 
            data_output = arti_F_envelop;
        elseif strcmp(output_name,'arti_F_timehis')
            data_output = arti_F_timehis;
            
        elseif strcmp(output_name,'arti_M_envelop')            %>> M
            data_output = arti_M_envelop;
        elseif strcmp(output_name,'arti_M_timehis')
            data_output = arti_M_timehis;
            
        elseif strcmp(output_name,'arti_Dis_timehis')           %>> Displacement
            data_output = arti_Dis_timehis;
           
        elseif strcmp(output_name,'arti_CD_base_max')          %>> Curvature ductility normlized with curvature_yield.
            data_output = arti_CD_base_max/curvature_yield;
        elseif strcmp(output_name,'arti_CD_envelop')
            data_output = arti_CD_envelop/curvature_yield;
        elseif strcmp(output_name,'arti_CD_base_timehis')
            data_output = arti_CD_base_timehis/curvature_yield;
   
        end
        
        
        index_4 = 1501 : 1 : 2000;
        % >>>> 4th site: input specified
        if strcmp(input_name,'arti_input_4')
             data_input = arti_input(index_4,1:(end-2) );
        elseif strcmp(input_name,'arti_input_timehis_4')
            data_input = arti_input_timehis(index_4,:);
        end
        
        % >>>> 4th site: output specified
        if strcmp(output_name,'arti_F_envelop_4')                %>> F 
            data_output = arti_F_envelop(index_4,:);
        elseif strcmp(output_name,'arti_F_timehis_4')
            data_output = arti_F_timehis(index_4,:);
            
        elseif strcmp(output_name,'arti_M_envelop_4')            %>> M
            data_output = arti_M_envelop(index_4,:);
        elseif strcmp(output_name,'arti_M_timehis_4')
            data_output = arti_M_timehis(index_4,:);
            
        elseif strcmp(output_name,'arti_Dis_timehis_4')           %>> Displacement
            data_output = arti_Dis_timehis(index_4,:);
           
        elseif strcmp(output_name,'arti_CD_base_max_4')          %>> Curvature ductility normlized with curvature_yield.
            data_output = arti_CD_base_max(index_4,:)/curvature_yield;
        elseif strcmp(output_name,'arti_CD_envelop_4')
            data_output = arti_CD_envelop(index_4,: )/curvature_yield;
        elseif strcmp(output_name,'arti_CD_base_timehis_4')
            data_output = arti_CD_base_timehis(index_4,:)/curvature_yield;
   
        end
       
        
        
        
        
        
    case 'white_noise'
        load(path_white_data)
        
        %>>>>1. 线弹性：屈服=1:1， 最大曲率 = 1.7
%         index_1 = find( white_CD_base_max/curvature_yield < 1 );
%         index_1_temp = index_1(randperm(length(index_1),500 ) );
%         
%         index_2 = find( white_CD_base_max/curvature_yield >= 1 & white_CD_base_max/curvature_yield <1.7 );
%         index_2_temp = index_2( randperm(length(index_2),500 )  );
%         
%         index_nonlinear = [index_1_temp index_2_temp];
        

        %>>>>2. 线弹性：屈服=2:1， 最大曲率 = 1.7
%         index_nonlinear = find(  white_CD_base_max/curvature_yield <= 0.8 );
        

         %>>>>3. 线弹性：屈服=2:1， 最大曲率 = 5
        index_1_temp = find( white_CD_base_max/curvature_yield < 1 );
        
        index_2 = find( white_CD_base_max/curvature_yield >= 1  );
        index_2_temp = index_2( randperm(length(index_2),500 )  );
        
        index_nonlinear = [index_1_temp; index_2_temp];

        
         %>>>> input specified
        if strcmp(input_name,'white_input')
            data_input = white_input; 
            
        elseif strcmp(input_name,'white_input_timehis')
            data_input = white_input_timehis; 
        
        elseif strcmp(input_name,'white_input_nonlinear')
            data_input = white_input(index_nonlinear,:); 
        end

         %>>>> output specified
        if strcmp(output_name,'white_F_envelop')                %>> F 
            data_output = white_F_envelop;
        elseif strcmp(output_name,'white_F_timehis')
            data_output = white_F_timehis;
        elseif strcmp(output_name,'white_F_envelop_nonlinear')
            data_output = white_F_envelop(index_nonlinear,:);
            
        elseif strcmp(output_name,'white_M_envelop')            %>> M
            data_output = white_M_envelop;
        elseif strcmp(output_name,'white_M_timehis')
            data_output = white_M_timehis;
            
        elseif strcmp(output_name,'white_Dis_timehis')           %>> Displacement
            data_output = white_Dis_timehis;
           
        elseif strcmp(output_name,'white_CD_base_max')          %>> Curvature ductility normlized with curvature_yield.
            data_output = white_CD_base_max/curvature_yield;
        elseif strcmp(output_name,'white_CD_envelop') 
            data_output = white_CD_envelop/curvature_yield;
        elseif strcmp(output_name,'white_CD_base_timehis')
            data_output = white_CD_base_timehis/curvature_yield;
   
        elseif strcmp(output_name,'white_CD_base_max_nonlinear')
            data_output = white_CD_base_max(index_nonlinear,:)/curvature_yield ;
            
        end
        
        
    case 'near_fault'
        load(path_near_fault_data);
        index_linear = find(near_fault_CD_base_max/curvature_yield <= 1 );
         %>>>> input specified
        if strcmp(input_name,'near_fault_input')
            data_input = near_fault_input; 
            
        elseif strcmp(input_name,'near_fault_input_timehis')
            data_input = near_fault_input_timehis; 
        
        elseif strcmp(input_name,'near_fault_input_linear')
            data_input = near_fault_input(index_linear,:); 
            
        end

         %>>>> output specified
        if strcmp(output_name,'near_fault_F_envelop')                %>> F 
            data_output = near_fault_F_envelop;
        elseif strcmp(output_name,'near_fault_F_linear_envelop')
            data_output = near_fault_F_envelop(index_linear,:);
        elseif strcmp(output_name,'near_fault_F_timehis')
            data_output = near_fault_F_timehis;
            
        elseif strcmp(output_name,'near_fault_M_envelop')            %>> M
            data_output = near_fault_M_envelop;
        elseif strcmp(output_name,'near_fault_M_timehis')
            data_output = near_fault_M_timehis;
            
        elseif strcmp(output_name,'near_fault_Dis_timehis')           %>> Displacement
            data_output = near_fault_Dis_timehis;
           
        elseif strcmp(output_name,'near_fault_CD_base_max')          %>> Curvature ductility normlized with curvature_yield.
            data_output = near_fault_CD_base_max/curvature_yield;
        elseif strcmp(output_name,'near_fault_CD_envelop')
            data_output = near_fault_CD_envelop/curvature_yield;
        elseif strcmp(output_name,'near_fault_CD_base_timehis')
            data_output = near_fault_CD_base_timehis/curvature_yield;
   
        end
        
        
        
        
        
end % switch case

%     path = ['C:\Users\cxtj1\Desktop\matlab_fragility\Paper1_Nonlinear_response\A_data_source\' excitation_type '\' input_name '_' output_name];



end % function

