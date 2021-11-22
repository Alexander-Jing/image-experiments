A = 100*ones(15,15); % 对背景赋予无穷大，这里用100代替
A(5,6)=1; %目标点
A(8,9)=1;
A(12,3)=1;
A(14,11)=1;
for ii=2:size(A,1)
    for jj=2:size(A,2)-1
        temp0=A(ii,jj);
        temp1=min(A(ii,jj-1)+3,temp0);
        temp2=min(A(ii-1,jj-1)+4,temp1);
        temp3=min(A(ii-1,jj)+3,temp2);
        temp4=min(A(ii-1,jj+1)+4,temp3);
        A(ii,jj)=temp4;
    end
end
for ii=size(A,1)-1:-1:1
    for jj=size(A,2)-1:-1:2
        temp0=A(ii,jj);
        temp1=min(A(ii,jj+1)+3,temp0);
        temp2=min(A(ii+1,jj+1)+4,temp1);
        temp3=min(A(ii+1,jj)+3,temp2);
        temp4=min(A(ii+1,jj+1)+4,temp3);
        A(ii,jj)=temp4;
    end
end
D=round(A(:,2:end-1)/3);%第一列 和最后一列无法计算，或者补充以后也可以。
D
