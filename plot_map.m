
function plot_map(result_folder)

addpath('VOCdevkit/')
addpath('VOCdevkit/VOCcode/')

VOCinit;

%validation test and map plot
%folder = dir([result_folder '/val*']);
folder = result_folder;
    VOCopts.detrespath = [result_folder '/comp3_det_test_%s.txt' ];
    VOCopts.imgsetpath = [result_folder  '/%s.txt'];
    VOCopts.testset = 'test';
VOCopts
tic;

        for i = 1:20

        class = VOCopts.classes{i};
        VOCevaldet(VOCopts, '/',class,true );
        viewdet(VOCopts,'/' ,class,false)
    end
end




%VOCopts%
% testset test
%eval_det('test/')

