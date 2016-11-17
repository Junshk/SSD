
function plot_map(result_folder)

addpath('VOCdevkit/')
addpath('VOCdevkit/VOCcode/')

VOCinit;

%validation test and map plot
folder = dir([result_folder '/val*']);
VOCopts
tic;
for j = 1:length(folder)
    folder_ = folder(j).name ;
    VOCopts.detrespath = [result_folder '/%s/comp3_det_test_%s.txt' ];
    VOCopts.imgsetpath = [result_folder '/' folder_ '/%s.txt'];
    VOCopts.testset = 'test';
        for i = 1:20

        class = VOCopts.classes{i};
        VOCevaldet(VOCopts, folder_,class,true );
        viewdet(VOCopts,folder_ ,class,false)
    end
end



end
%VOCopts%
% testset test
%eval_det('test/')

