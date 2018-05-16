function [Lefteye Righteye]= pclick(input2,Lefteye,Righteye,i)

imshow(input2(:,:),[])
hold on
plot(Lefteye(1,i),Lefteye(2,i),'o')
plot(Righteye(1,i),Righteye(2,i),'o')

prompt = 'What is the original value? ';
ind = input(prompt);
while ind~=3
    [Left_x,Left_y] = ginput(1)
    Lefteye(1,i)=Left_x;
    Lefteye(2,i)=Left_y;
    [Right_x,Right_y] = ginput(1)
    Righteye(1,i)=Right_x;
    Righteye(2,i)=Right_y;
    ind = input(prompt);
end

% prompt = 'What is the original value? ';
% ind = input(prompt);
% while ind~=3
%    if  ind == 2
%        [x,y] = ginput(1)
%     Lefteye(1,i)=x;
%     Lefteye(2,i)=y;
%    elseif  ind == 1
%        [x,y] = ginput(1)
%     Righteye(1,i)=x;
%     Righteye(2,i)=y;
%    end
%    ind = input(prompt);
% end