%------------------- Configurations ------------------------------------
imageDim = 28;      % input of MNIST images' size
numClasses = 10;    % MNIST images fall into 10 different classes
trainedNum = 60000;  % Real trained images
filterDim_1 = 5;    % filter size in first cnn layer
filterDim_2 = 5;
poolDim_1 = 2;
poolDim_2 = 2;
imageChannel = 1;
numFilters_1 = 10;
numFilters_2 = 10;

% Weight decay
lambda = 0.0001;

% Laoding MNIST Train Images
addpath ../common/;
images = loadMNISTImages('train-images-idx3-ubyte');
% dimensions = size(images) % [28*28, 6000]
images = reshape(images, imageDim, imageDim, 1, []);
% dimensions = size(images) % [28*28*1 6000]
labels = loadMNISTLabels('train-labels-idx1-ubyte');
labels(labels == 0) = 10;    % Modify 0 labels to 10

% Use 6000 images to train
images = images(:,:,:,1:60000);
labels = labels(1:60000);

% Initialization Weight and Bais
W_1 = 1e-1*randn(filterDim_1,filterDim_1,imageChannel,numFilters_1);
n1 = filterDim_1*filterDim_1*imageChannel*numFilters_1;
% W_1 = 5*5*1*8
b_1 = zeros(numFilters_1, 1);
W_2 = 1e-1*randn(filterDim_2,filterDim_2,numFilters_1,numFilters_2);
n2 = filterDim_2*filterDim_2*numFilters_1*numFilters_2;
% W_2 = 5*5*8*10
b_2 = zeros(numFilters_2, 1);

% 1st CNN Layer
out_Dim_1 = imageDim - filterDim_1 + 1;
out_Dim_1 = out_Dim_1 / poolDim_1;

% 2nd CNN Layer
out_Dim_2 = out_Dim_1 - filterDim_2 +1;
out_Dim_2 = out_Dim_2/ poolDim_2;

% After pooling convert to 2-D with hiddenSzie
% 4*4*10
hiddenSize = out_Dim_2^2*numFilters_2;

% Training Settings
epochs = 5;
miniBatch = 150;
alpha = 1e-1;

% Initialization of SoftMax Weight and Bais
% 10*(4*4*10)
Wd = rand(numClasses, hiddenSize);
nd = numClasses* hiddenSize;
% r  = sqrt(6) / sqrt(numClasses+hiddenSize+1);
% Wd = rand(numClasses, hiddenSize) * 2 * r - r;
% Wd = Good Initialization

bd = zeros(numClasses, 1);


m = length(labels);     % Training Size


% Traning
it = 0;
C = [];
dropRate = 0.1;
n_wd = round(nd*dropRate);
disp(n_wd);
n_w1 = round(n1*dropRate);
disp(n_w1);
n_w2 = round(n2*dropRate);
disp(n_w2);
for e = 1:epochs
    
    rp = randperm(m);   % Rnadom permutation this is an Array 1-6000
    
    Wd_hidden = ones(1,nd);
    W_1_hidden = ones(1,n1);
    W_2_hidden = ones(1,n2);
    
    Wd_res_grad = zeros(size(Wd));
    W_1_res_grad = zeros(size(W_1));
    W_2_res_grad = zeros(size(W_2));
    
    pre_cost = 0;
    
    p_wd = [];
    p_w1 = [];
    p_w2 = [];
    for s = 1:miniBatch:(m - miniBatch +1)
        it = it + 1;
        
        % momentum enable
        
        % Minibatch Picking
        mb_images = images(:,:,:,rp(s:s+miniBatch-1));
        mb_labels = labels(rp(s:s+miniBatch-1));
        
        numImages = size(mb_images,4); % the 4th dimension (28,28,1,150) = 150
        %%%%%%%%%%%%%
        Wd_grad = zeros(size(Wd));
        Wd_up_grad = zeros(size(Wd));
        bd_grad = zeros(size(bd));
        %%%%%%%
        W_1_grad = zeros(size(W_1));       
        W_1_up_grad = zeros(size(W_1));
        b_1_grad = zeros(size(b_1));
        %%%%%%%
        W_2_grad = zeros(size(W_2));        
        W_2_up_grad = zeros(size(W_2));        
        b_2_grad = zeros(size(b_2));
        %%%%%%%
        convDim_1 = imageDim-filterDim_1+ 1;    % 28-5+1 = 24
        out_conv_Dim_1 = (convDim_1)/poolDim_1; % 24/2 = 12
        convDim_2 = out_conv_Dim_1 - filterDim_2 + 1; %12-5+1 = 8
        out_conv_Dim_2 = (convDim_2)/poolDim_2; % 8/2 = 4
        
        %------------------- Feedfoward --------------------------------
        activations_1 = cnnConvolve4D(mb_images, W_1, b_1);
        activations_1_Pooled = cnnPool(poolDim_1, activations_1);
        
        activations_2 = cnnConvolve4D(activations_1_Pooled, W_2, b_2);
        activations_2_Pooled = cnnPool(poolDim_2, activations_2);
        
        % Reshape activations into 2-d matrix, hiddenSize x numImages,
        % 4*4*10 * 150 = 160*150
        activations_2_Pooled = reshape(activations_2_Pooled,[],numImages);
        
        %------------------- SoftMax Layer -----------------------------
        % Wd*activations_2_Pooled = 10*(4*4*10) x (4*4*10)*150 = 10*150
        probs = exp(bsxfun(@plus, Wd * activations_2_Pooled, bd));
        sum_probs = sum(probs,1);
        probs = bsxfun(@times, probs, 1 ./ sum_probs);
        % size(probs) = [10*150]
        
        %------------------- Calculate Cost ----------------------------
        % http://neuralnetworksanddeeplearning.com/chap3.html#the_cross-entropy_cost_function
        logp = log(probs);
        index = sub2ind(size(logp),mb_labels',1:size(probs,2));
        % Why cost only count this part?
        ceCost = -sum(logp(index));        
        % wCost = lambda/2 * (sum(Wd(:).^2)+sum(W_1(:).^2)+sum(W_2(:).^2));
        wCost = 0;
        cost = ceCost/numImages + wCost;
        
        %------------------- BackPropagation ---------------------------
        out_put = zeros(size(probs));
        out_put(index) = 1;
        Delta_Softmax = (probs - out_put);
        % t = - Delta_Softmax;
        
        % Error of second pooling layer
        Delta_Pool_2 = reshape(Wd' * Delta_Softmax,out_conv_Dim_2,out_conv_Dim_2,numFilters_2,numImages);
        
        Delta_Un_Pool_2 = zeros(convDim_2,convDim_2,numFilters_2,numImages);
        for im = 1:numImages
            for filter = 1: numFilters_2
                un_pool = Delta_Pool_2(:,:,filter,im);
                Delta_Un_Pool_2(:,:,filter,im) = kron(un_pool,ones(poolDim_2))./(poolDim_2 ^ 2);
            end
        end
        % Error of second convolutional layer
        Delta_Conv_2 = Delta_Un_Pool_2 .* activations_2 .* (1 - activations_2);
        
        % Error of first pooling layer
        Delta_Pool_1 = zeros(out_conv_Dim_1,out_conv_Dim_1,numFilters_1,numImages);
        for im = 1:numImages
            for f1 = 1: numFilters_1
                for f2 = 1:numFilters_2
                    % Not Understand HERE
                    Delta_Pool_1(:,:,f1,im) = Delta_Pool_1(:,:,f1,im)+convn(Delta_Conv_2(:,:,f2,im),W_2(:,:,f1,f2),'full');
                end
            end
        end
        
        % Error of first convolutional layer
        Delta_Un_Pool_1 = zeros(convDim_1,convDim_1,numFilters_1,numImages);
        for im = 1:numImages
            for filter = 1:numFilters_1
                un_pool = Delta_Pool_1(:,:,filter,im);
                Delta_Un_Pool_1(:,:,filter,im) = kron(un_pool,ones(poolDim_1))./(poolDim_1 ^ 2);
            end
        end
        
        Delta_Conv_1 = Delta_Un_Pool_1 .* activations_1 .* (1 - activations_1);
        
        % ------------------ Gradient Calculation -------------------------
        % SoftMax Layer
        Wd_grad = Delta_Softmax * activations_2_Pooled';
        bd_grad = sum(Delta_Softmax,2);
        % Second Layer
        for f2 = 1:numFilters_2
            for f1 = 1:numFilters_1
                for im = 1 : numImages
                    W_2_grad(:,:,f1,f2) = W_2_grad(:,:,f1,f2)+ conv2(activations_1_Pooled(:,:,f1,im),rot90(Delta_Conv_2(:,:,f2,im),2),'valid');
                end           
            end
            temp = Delta_Conv_2(:,:,f2,:);
            b_2_grad(f2) = sum(temp(:)); 
        end
        % First Layer
        for f1 = 1:numFilters_1
            for ch = 1:imageChannel
                for im = 1:numImages
                    W_1_grad(:,:,ch,f1) = W_1_grad(:,:,ch,f1)+ conv2(mb_images(:,:,ch,im),rot90(Delta_Conv_1(:,:,f1,im),2),'valid');
                end
            end
            temp = Delta_Conv_1(:,:,f1,:);
            b_1_grad(f1) = sum(temp(:));
        end
        
     
        if it == 1
            disp('first iteration')
            %r_wd = reshape(Wd, 1, []);
            %r_w1 = reshape(W_1, 1, []);
            %r_w2 = reshape(W_2, 1, []);
            p_wd = randperm(nd);
            p_w1 = randperm(n1);
            p_w2 = randperm(n2);
            %p_wd_u = p_wd(1:n_wd);
            %p_w1_u = p_w1(1:n_w1);
            %p_w2_u = p_w2(1:n_w2);
            Wd_res_grad(:) = Wd_grad(:);
            for idx = p_wd(1:n_wd)
                ele = Wd_grad(idx);
                % disp(ele)
                Wd_up_grad(idx) = Wd_up_grad(idx)+ ele;
                Wd_hidden(idx) = Wd_hidden(idx)+1;
                Wd_res_grad(idx) = 0;
            end
           
            % Wd_up_grad = reshape(Wd_up_grad, numClasses, hiddenSize);
            W_1_res_grad(:) = W_1_grad(:);
            for idx = p_w1(1:n_w1)
                ele = W_1_grad(idx);
                %disp(ele)
                W_1_up_grad(idx) = W_1_up_grad(idx)+ele;
                W_1_hidden(idx) =  W_1_hidden(idx)+1;
                W_1_res_grad(idx) = 0;
            end
            % W_1_up_grad(idx) = reshape(W_1_up_grad,filterDim_1,filterDim_1,imageChannel,numFilters_1);
            W_2_res_grad(:) = W_2_grad(:);
            for idx = p_w2(1:n_w2)
                ele = W_2_grad(idx);
                % disp(ele)
                W_2_up_grad(idx) = W_2_up_grad(idx)+ele;
                W_2_hidden(idx) =  W_2_hidden(idx)+1;
                W_2_res_grad(idx) = 0;
            end
            % W_2_up_grad = reshape(W_2_up_grad,filterDim_2,filterDim_2,numFilters_1,numFilters_2);            
            pre_cost = cost;
            
        else
            if pre_cost > cost
                
                for idx = p_wd(1:n_wd)
                    ele = Wd_grad(idx);
                    % disp(ele)
                    Wd_up_grad(idx) = Wd_up_grad(idx)+ ele;
                    Wd_hidden(idx) = Wd_hidden(idx)+1;
                    Wd_grad(idx) = 0;
                end
                Wd_res_grad(:) = Wd_res_grad(:) + Wd_grad(:);
                
                for idx = p_w1(1:n_w1)
                    ele = W_1_grad(idx);
                    % disp(ele)
                    W_1_up_grad(idx) = W_1_up_grad(idx)+ele;
                    W_1_hidden(idx) =  W_1_hidden(idx)+1;
                    W_1_grad(idx) = 0;
                end
                W_1_res_grad(:) =  W_1_res_grad(:)+ W_1_grad(:);
                
                for idx = p_w2(1:n_w2)
                    ele = W_2_grad(idx);
                    % disp(ele)
                    W_2_up_grad(idx) = W_2_up_grad(idx)+ele;
                    W_2_hidden(idx) =  W_2_hidden(idx)+1;
                    W_2_grad(idx) = 0;
                end
                W_2_res_grad(:) =  W_2_res_grad(:)+ W_2_grad(:);
                pre_cost = cost;
                
            else
                p_wd = randweightedpick(Wd_hidden, n_wd)';
                p_w1 = randweightedpick(W_1_hidden, n_w1)';
                p_w2 = randweightedpick(W_2_hidden, n_w2)';
                
                for idx = p_wd(1:n_wd)
                    ele = Wd_grad(idx);
                    % disp(ele)
                    Wd_up_grad(idx) = Wd_up_grad(idx)+ ele;
                    Wd_hidden(idx) = Wd_hidden(idx)+1;
                    Wd_grad(idx) = 0;
                end
                Wd_res_grad(:) = Wd_res_grad(:) + Wd_grad(:);
                
                for idx = p_w1(1:n_w1)
                    ele = W_1_grad(idx);
                    %disp(ele)
                    W_1_up_grad(idx) = W_1_up_grad(idx)+ele;
                    W_1_hidden(idx) =  W_1_hidden(idx)+1;
                    W_1_grad(idx) = 0;
                end
                W_1_res_grad(:) =  W_1_res_grad(:)+ W_1_grad(:);
                
                for idx = p_w2(1:n_w2)
                    ele = W_2_grad(idx);
                    %disp(ele)
                    W_2_up_grad(idx) = W_2_up_grad(idx)+ele;
                    W_2_hidden(idx) =  W_2_hidden(idx)+1;
                    W_2_grad(idx) = 0;
                end
                W_2_res_grad(:) =  W_2_res_grad(:)+ W_2_grad(:);
                pre_cost = cost;
                
            end
            %%%%%%%%%Threshold Section %%%%%%%%%%%%%
            [xw,yw,zw]=swithchGradient(Wd_up_grad,Wd_res_grad,Wd_hidden,p_wd(1:n_wd),sum(abs(Wd_grad(:)))/nd);
            Wd_up_grad(:) = xw(:);
            Wd_res_grad(:) = yw(:);
            Wd_hidde(:) = zw(:);
            
            [xw1,yw1,zw1]=swithchGradient(W_1_up_grad,W_1_res_grad,W_1_hidden,p_w1(1:n_w1),sum(abs(W_1_grad(:)))/n1);
            W_1_up_grad(:) = xw1(:);
            W_1_res_grad(:) = yw1(:);
            W_1_hidden(:) = zw1(:);
            
            [xw2,yw2,zw2]=swithchGradient(W_2_up_grad,W_2_res_grad,W_2_hidden,p_w2(1:n_w2),sum(abs(W_2_grad(:)))/n2);
            W_2_up_grad(:) = xw2(:);
            W_2_res_grad(:) = yw2(:);
            W_2_hidden(:) = zw2(:);
            
        end
        
        Wd = Wd - alpha*Wd_up_grad./miniBatch;
        bd = bd - alpha*bd_grad./miniBatch;
        W_1 = W_1 - alpha*W_1_up_grad./miniBatch;
        b_1 = b_1 - alpha*b_1_grad./miniBatch;
        W_2 = W_2 - alpha*W_2_up_grad./miniBatch;
        b_2 = b_2 - alpha*b_2_grad./miniBatch;
        
        
        fprintf('Epoch %d: Cost on iteration %d is %f\n',e,it,cost);
        C(length(C)+1) = cost;
    end
    alpha = alpha/(2.0*e);
    
    % -------------------------- TEST EVERY EPCHO----------------------
    testImages = loadMNISTImages('t10k-images-idx3-ubyte');
    testImages = reshape(testImages,imageDim,imageDim,1,[]);
    testLabels = loadMNISTLabels('t10k-labels-idx1-ubyte');
    testLabels(testLabels==0) = 10;
    
    activations1 = cnnConvolve4D(testImages, W_1, b_1);
    activationsPooled1 = cnnPool(poolDim_1, activations1);
    activations2 = cnnConvolve4D(activationsPooled1, W_2, b_2);
    activationsPooled2 = cnnPool(poolDim_2, activations2);
    
    activationsPooled2 = reshape(activationsPooled2,[],length(testImages));
    
    test_probs = exp(bsxfun(@plus, Wd * activationsPooled2, bd));
    test_sumProbs = sum(test_probs, 1);
    test_probs = bsxfun(@times, test_probs, 1 ./ test_sumProbs);
    
    % find the max weight in each column
    % return its(max) row number as predicated result
    [~,preds] = max(test_probs,[],1);
    preds = preds';
    acc = sum(preds==testLabels)/length(preds);
    fprintf('Accuracy is %f\n',acc);
    plot(C);
end
