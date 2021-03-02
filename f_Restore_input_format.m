function [input] = f_Restore_input_format(input,options,scale)
    data_log = options.data_log; 
    type = options.norm_input_type; 
    
switch type
    % ==== 0-1 ====
    case '0_1_whole'
        if data_log
           input = exp( input *(scale.max - scale.min ) + scale.min);
           
        else
           input = input * (scale.max - scale.min ) + scale.min ;
           
        end
        
    case '0_1_each_column'
        if data_log
           input = exp( input .*(scale.max - scale.min ) + scale.min);
           
        else
           input = input .* (scale.max - scale.min ) + scale.min ;
           
        end
        
    % ==== MAX ====
    case 'MAX_whole'
        if data_log
           input = exp( input * scale.max );
           
        else
           input = input * scale.max ;
        end
        
    case 'MAX_each_column'
        if data_log
           input = exp( input .* scale.max );
           
        else
           input = input .* scale.max ;
        end
        
        
    % ==== mu-sigma ====    
    case 'mu_sigma_whole'
        if data_log
            input = exp((input * scale.sigma) + scale.mu );
        else
            input = (input * scale.sigma) + scale.mu;
        end
        
    case 'mu_sigma_each_column'
        if data_log
            input = exp((input .* scale.sigma) + scale.mu );
        else
            input = (input .* scale.sigma) + scale.mu;
        end
        
end
        
        
end

