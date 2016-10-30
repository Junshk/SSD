addpath('VOCdevkit/')
addpath('VOCdevkit/VOCcode/')

VOCinit;

%validation test and map plot
folder = dir('validation/val*');
tic;
for j = 1:length(folder)
    folder_ = folder(j).name ;
    VOCopts.detrespath = 'validation/%s/comp3_det_test_%s.txt' ;

    for i = 1:1

        class = VOCopts.classes{i};
        VOCevaldet(VOCopts, folder_,class,true )
    end
end




%VOCopts%
% testset test
%eval_det('test/')

