function [data, th] = read_Fundamental_data(data_file)

load(data_file);th = 0.03; 
data.X1 = data.matches.X1;
data.X2 = data.matches.X2;
data.im1 = data.matches.im1;
data.im2 = data.matches.im2;

end