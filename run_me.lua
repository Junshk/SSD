
dofile('dataload.lua')


require 'pascal'
require 'training'

print('load datas')

local option =
{plot_iter =500,end_iter = 80*1000,print_iter=20,save_iter=500,
test_iter = 200,
batch_size = 14,
valid =true
}
training(option)



local result = test()

os.excute()
