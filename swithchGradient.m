function [modified_up_grad, modified_res_grad,modified_hiddenWeight]=swithchGradient(up_grad,res_grad,hiddenWeight, permutation, threshold)
    % Deep copy two modified array
    modified_up_grad(:) = up_grad(:);
    %fprintf('-----ele_up %d \n', size(up_grad));
    modified_res_grad(:) = res_grad(:);
    %fprintf('-----ele_res %d \n', size(res_grad));
    modified_hiddenWeight(:) = hiddenWeight(:);
    % determine the max value we want to replace with
    length = round(size(permutation(:),1));
    if (mod(length,2)~=0)
        length = round(length);
    end
    %fprintf('------length is %d \n', length);
    total_replace_num = length/2;
    %fprintf('------total replace number is %d \n', total_replace_num);
    % sub array for storage uses
    idx_list = zeros(size(permutation));
    value_list = zeros(size(permutation));
    hidden_list = zeros(size(permutation));
    candidate = [];
    %candidate = zeros(size(permutation));
    
    idx_list(:) = permutation(:);
    %fprintf('-----idx_list %d \n', size(idx_list));
    t = 1;
    for idx = idx_list
        ele = up_grad(idx);
        %fprintf('-----ele_up %d \n', size(ele));
        
        value_list(t) = ele;
        score = hiddenWeight(idx);
        hidden_list(t) = score; 
        % disp(hidden_list)
        t =t+1;
    end
    
    [~,co] = find(abs(res_grad)>threshold);
    % fprintf('-----After abs %d \n', size(res_grad));
    len_qual = size(co(:));
    % fprintf("---what is---");
    % disp(qualifed_list);
    if len_qual < total_replace_num
        cnt = 1;
        for k = co'
            ele = res_grad(k);      
            %disp(ele);
            candidate(cnt) = ele;
            cnt = cnt+1;
        end
    else
        co = co(1:total_replace_num);
        
        cnt = 1;
        %fprintf('-----cooooo---%d', size(co));     
        %disp(co);
        for k = co'
            %fprintf('what is k %d \n', size(k));
            %disp(k);
            ele = res_grad(k);
            %fprintf('-----ele_res %d \n', size(ele));
            %disp(ele);
            candidate(cnt) = ele;
            cnt = cnt+1;
        end     
    end
       
    [~,c] = sort(hidden_list);
    %fprintf('-----c 1 size is %d \n', size(c,1));
    %fprintf('-----c 2 size is %d \n', size(c,2));
    c = c(:);
    for i = 1:total_replace_num
        idxx = c(i);
        %disp("------------ci_list-----------");
        %disp(idxx);
        %disp("------------idx_list-----------");
        %disp(size(idx_list));       
        %disp(idx_list);
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