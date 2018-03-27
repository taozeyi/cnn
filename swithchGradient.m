function [modified_up_grad, modified_res_grad,modified_hiddenWeight]=swithchGradient(up_grad,res_grad,hiddenWeight, permutation, threshold)
    % Deep copy two modified array
    modified_up_grad(:) = up_grad(:);
    modified_res_grad(:) = res_grad(:);
    modified_hiddenWeight(:) = hiddenWeight(:);
    % determine the max value we want to replace with
    length = size(permutation(:));
    total_replace_num = length/2;
    % sub array for storage uses
    idx_list = zeros(size(permutation));
    value_list = zeros(size(permutation));
    hidden_list = zeros(size(permutation));
    candidate = [];
    %candidate = zeros(size(permutation));
    
    idx_list(:) = permutation(:);
    
    for idx = idx_list
        ele = up_grad(idx);
        value_list(idx) = ele;
        score = hiddenWeight(idx);
        hidden_list(idx) = score; 
        % disp(hidden_list)
    end
    % prepare the candidate array
    qualifed_list = find(abs(res_grad)>threshold);
    len_qual = size(qualifed_list(:));
    [~,co] = find(abs(res_grad)>threshold);
    % fprintf("---what is---");
    % disp(qualifed_list);
    if len_qual < total_replace_num
        cnt = 1;
        for k = qualifed_list
            ele = res_grad(k);
            candidate(cnt) = ele;
            cnt = cnt+1;
        end
    else
        qualifed_lis(:) = qualifed_list(1:total_replace_num);
        cnt = 1;
        for k = qualifed_lis
            ele = res_grad(k);
            candidate(cnt) = ele;
            cnt = cnt+1;
        end     
    end
       
    [~,c] = sort(hidden_list);
    
    for i = 1:total_replace_num
        idxx = c(i);
        idx = idx_list(idxx);
        switch_value = candidate(i);
        %switch_value = 0;
        xdi = co(i);
        
        modified_res_grad(idx) = 0;
        modified_up_grad(xdi) = switch_value;
        %fprintf("%d idx and xdi %d", idx, xdi);
        
        modified_hiddenWeight(idx) = modified_hiddenWeight(idx) -1;        
        modified_hiddenWeight(xdi) = modified_hiddenWeight(xdi) +1;
    end
    
end