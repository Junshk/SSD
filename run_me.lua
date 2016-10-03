require 'pascal'
require 'training'

dofile('dataload.lua')
print('load datas')



training({plot_iter =4000,end_iter = 60*1000,print_iter=10,save_iter=4000})

