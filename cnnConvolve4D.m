function convolvedFeatures = cnnConvolve4D(images,W,b)
    filterDim = size(W,1);
    numFilter_1 = size(W,3);
    numFilter_2 = size(W,4);
    numImages = size(images,4);
    imageDim = size(images,1);
    
    convDim = imageDim - filterDim +1;
    convolvedFeatures = zeros(convDim, convDim, numFilter_2, numImages);
    
    for i = 1:numImages
        for f2 = 1:numFilter_2
            convolvedImage = zeros(convDim, convDim);
            for f1 = 1:numFilter_1
                filter = squeeze(W(:,:,f1,f2)); %why?
                filter = rot90(squeeze(filter),2); %why?
                im = squeeze(images(:,:,f1,i));
                convolvedImage = convolvedImage + conv2(im, filter,'valid');
            end
            convolvedImage = bsxfun(@plus,convolvedImage,b(f2));
            convolvedImage = 1 ./ (1+exp(-convolvedImage));
            convolvedFeatures(:, :, f2, i) = convolvedImage;
        end
    end
    
    
end