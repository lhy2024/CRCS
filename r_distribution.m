% 被try_fft.m调用
function ret = r_distribution(mat)
    % 生成距离场，把二维图像按半径平均到一维
    max_x = (size(mat,2)-1)/2;
    x=-max_x:max_x;
    max_y = (size(mat,1)-1)/2;
    y=-max_y:max_y;
    y=y';
    distances=sqrt(x.^2+y.^2);
    d=round(distances);

    for i=0:max(d,[],'all')
        avgs(i+1)=mean(mat(d==i));
    end
    ret = avgs;
end
