function pooledFeatures = cnnPool(poolDim, convolvedFeatures)
    convolvedDim = size(convolvedFeatures,1);
    numFilters = size(convolvedFeatures,3);
    numImages = size(convolvedFeatures,4);
    
    d = convolvedDim / poolDim ;
    
    pooledFeatures = zeros(d,d,numFilters,numImages);
    
    for i = 1:numImages
        for f = 1: numFilters
            featuremap = squeeze(convolvedFeatures(:,:,f,i));
            pooledFeaturemap = conv2(featuremap,ones(poolDim)/(poolDim^2),'valid');
            pooledFeatures(:,:,f,i) = pooledFeaturemap(1:poolDim:end,1:poolDim:end);
        end
    end
    
end