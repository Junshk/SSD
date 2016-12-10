require 'option'
require 'FetchData'
require 'donkey'


print('list num:',#img_Info_table)

local limit_num = data_num 
local iter = 0
local bsz = 1


if paths.dirp('data')==false then os.execute('mkdir data') end

while true do

save_name = 'data/SSDdata_'..iter..'.t7';
local inputs, targets

if paths.filep(save_name) == true then goto continue end

inputs, targets = patchFetch(bsz,img_Info_table)
torch.save(save_name,{input = inputs, target= targets})

print('data_'..iter..' saved')

::continue::
iter = iter + 1 
if iter > limit_num then break; end
end

