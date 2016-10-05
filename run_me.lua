require 'pascal'
require 'training'

dofile('dataload.lua')
print('load datas')



training({plot_iter =1000,end_iter = 60*1000,print_iter=20,save_iter=500})

