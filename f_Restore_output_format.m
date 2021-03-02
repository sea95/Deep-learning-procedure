function [actual,prediction] = f_Restore_output_format(actual,prediction,options,scale)

data_log = options.data_log; 
type = options.norm_output_type; 

switch type
    % ==== 0-1 ====
    case '0_1_whole'
        if data_log
            prediction = exp( prediction *(scale.max - scale.min ) + scale.min );
            actual = exp( actual *(scale.max - scale.min ) + scale.min );

        else
           prediction = prediction *(scale.max - scale.min ) + scale.min;
           actual = actual *(scale.max - scale.min ) + scale.min ;
            
        end
        
    case '0_1_each_column'
        if data_log
            prediction = exp( prediction .*(scale.max - scale.min ) + scale.min );
            actual = exp( actual .*(scale.max - scale.min ) + scale.min );
        else
            prediction = prediction .*(scale.max - scale.min ) + scale.min;
           actual = actual .*(scale.max - scale.min ) + scale.min ;
        end
        
    % ==== MAX ====
    case 'MAX_whole'
        if data_log
            prediction = exp( prediction * scale.max );
            actual = exp( actual * scale.max );
        
        else
            prediction = prediction * scale.max ;
            actual = actual * scale.max ;
            
        end
        
    case 'MAX_each_column' 
        if data_log
            prediction = exp( prediction .* scale.max );
            actual = exp( actual .* scale.max );
        
        else
            prediction = prediction .* scale.max ;
            actual = actual .* scale.max ;
            
        end
        
        
    % ==== mu-sigma ====
    case 'mu_sigma_whole'
        if data_log
            prediction = exp( prediction * scale.sigma +scale.mu );
            actual = exp( actual * scale.sigma +scale.mu );
        
        else
            prediction = prediction * scale.sigma +scale.mu ;
            actual = actual * scale.sigma +scale.mu ;
            
        end
        
    case 'mu_sigma_each_colnmn'
        if data_log
            prediction = exp( prediction .* scale.sigma +scale.mu );
            actual = exp( actual .* scale.sigma +scale.mu );
        
        else
            prediction = prediction .* scale.sigma +scale.mu ;
            actual = actual .* scale.sigma +scale.mu ;
            
        end
        
        
        
end 

end

