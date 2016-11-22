require 'option'
require 'FetchData'
require 'donkey'


 img_Info_table = ImgTxt('VOCdevkit/VOC2012','trainval.txt')--ImgInfo()--trainInfo()--ImgInfo()
img_Info_table = ImgTxt('VOCdevkit/VOC2007','trainval.txt',img_Info_table)
img_Info_table = ImgTxt('VOCdevkit/VOC2007','test.txt',img_Info_table)

print('list num:',#img_Info_table)

local limit_num = 5e+4 
local iter = 0
local bsz = 1


if paths.dirp('data')==false then os.execute('mkdir data') end

while true do

save_name = 'data/SSDdata_'..iter..'.t7';
local inputs, targets

if paths.filep(save_name) == true then goto continue end

inputs, targets = patchFetch(bsz,img_Info_table)
torch.save(save_name,{input = inputs, target= targets})


::continue::
iter = iter + 1 
print('data_'..iter..' saved')
if iter > limit_num then break; end
end

