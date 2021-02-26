clear all;
close all;

%%

degree=2;

current_refinement_level=3;
vec_method_distortion={'randomly','regularly_1', 'regularly_2'};
id_method_distortion = 3;

method_distortion = string(vec_method_distortion(id_method_distortion));

filename_uniform = sprintf('../coords_of_uniform_dofs_of_degree_%d_refine_%d_sequenced.txt',degree,current_refinement_level);
filename_distorted = sprintf('../coords_of_%s_distorted_dofs_of_degree_%d_refine_%d_sequenced.txt',method_distortion,degree,current_refinement_level);

data_y_uniform=dlmread(filename_uniform);

data_y_distorted=dlmread(filename_distorted);

difference_absolute = data_y_distorted-data_y_uniform;
minimal_edge_length = data_y_uniform(degree+1)-data_y_uniform(1);
difference_relative = difference_absolute/minimal_edge_length;

vector_length = length(data_y_uniform);
vec_zeros=zeros(vector_length,1);

figure;
plot(data_y_uniform,vec_zeros,'ob');

hold on;

plot(data_y_distorted,vec_zeros,'*r');

hold on;

plot(data_y_uniform,difference_relative,'dk--');
hold off

legend({'DoFs on a uniform mesh','DoFs on a distorted mesh','l_{dist}/h_{min}'},'Location','southeast','NumColumns',1)

if id_method_distortion==1
    axis([0 1 -0.5 0.5])
end

filename_to_be_saved=sprintf('measure_distorting_%s_degree_%d_refine_%d.png',method_distortion,degree,current_refinement_level);
saveas(gcf,filename_to_be_saved)


