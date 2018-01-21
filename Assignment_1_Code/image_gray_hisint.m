initial_graph = imread('/Users/tongge/self_driving_car_group/Assignment_1/sudoku-original.png');
gray_graph = rgb2gray(initial_graph);

subplot(1,2,1);
imshow(gray_graph);
subplot(1,2,2);
imhist(gray_graph);

