
function plot_map(result_folder)
result_folder
addpath('VOCdevkit/')
addpath('VOCdevkit/VOCcode/')

VOCinit;
if strcmp(result_folder , 'test2012')==1
  VOCopts.dataset = 'VOC2012_test'
end
if strcmp(result_folder , 'test2007')==1
  VOCopts.dataset = 'VOC2007'
end
if strcmp(result_folder , 'testValid')==1
    VOCopts.testset = 'val';
end

%validation test and map plot
%folder = dir([result_folder '/val*']);
folder = result_folder;
    VOCopts.detrespath = ['%s/comp3_det_test_%s.txt' ];
    VOCopts.imgsetpath = [result_folder  '/%s.txt'];
VOCopts
tic;

        for i = 1:20

        class = VOCopts.classes{i};
        VOCevaldet(VOCopts, result_folder,class,true );
        saveas(gcf,class,'jpg')
        %viewdet(VOCopts,result_folder ,class,false)
    end








end




%VOCopts%
% testset test
%eval_det('test/')

