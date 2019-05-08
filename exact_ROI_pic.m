i=950;
i=num2str(i)
filename1='G:/test3_pic/';
filename2='.jpg';
filename=[filename1,i,filename2];
filename
%img=imread(filename)
Img=imread(filename);
[Img0,rect]=imcrop(Img);

%j=550;
j=j+1;
j=num2str(j)
filename3='G:/car/'
filename4='.jpg'
filename5=[filename3,j,filename4]
imwrite(Img0,filename5)
j=str2num(j)
