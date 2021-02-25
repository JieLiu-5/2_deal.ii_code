clear all;
close all;

%%

degree=2;
current_refinement_level=2;

data_x=dlmread('../local_dof_indices_temp.txt');
data_y=dlmread('../coords_of_uniform_dofs_of_refine_4_sequenced.txt');
data_y_second=dlmread('../coords_of_distorted_dofs_of_refine_4_sequenced.txt');

data_x_analytical = linspace(1,length(data_y),length(data_y));

% value_d = 1;
value_d = 1+data_x_analytical;

data_y_analytical = -value_d.*2.*(data_x_analytical-0.5);

difference_absolute = data_y-data_y_second;
difference_relative = difference_absolute/(data_y(degree+1)-data_y(1));

plot(data_y,difference_relative,'.r-');

hold on;
plot(data_y,0,'ob-');

% saveas(gcf,'solution_velocity_real.png')
% saveas(gcf,'solution_pressure_real.png')
% saveas(gcf,'solution_pressure.png')


