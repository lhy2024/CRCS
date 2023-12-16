list = dir('E:\1_task\2021kfb\??_??????');   
for j =1:size(list,1)
    a = list(j).name;
    b = dir(['E:\\1_task\\2021kfb\\',a,'\\Block_HE_0.625\\?.bmp']);
    name = a;
    for i = 1:size(b,1)
        im = imread(sprintf(['E:\\1_task\\2021kfb\\',a,'\\Block_HE_0.625\\%d.bmp'],i));
        if ndims(im) == 2
                gray=im;
        else
            gray=rgb2gray(im);
        end
        
            
            std=stdfilt(gray,strel('disk',4).Neighborhood);
        
            
            sep=imbinarize(rescale(std));
            % imshow(labeloverlay(im,sep,'Transparency',0.8))
            % 去掉边缘
            sep(1:15,:)=false;
            sep(end-15:end,:)=false;
            sep(:,1:15)=false;
            sep(:,end-15:end)=false;
        
            
            sep = imclose(sep,strel('disk',3));
            
            % 计算背景值
            equal = mean(gray(~sep), 'all');
            if ~isfile(['E:\\1_task\\2021kfb\\',a,'\\equal.txt'])
                writematrix(equal, ['E:\\1_task\\2021kfb\\',a,'\\equal.txt']);
            end
    end
    
end