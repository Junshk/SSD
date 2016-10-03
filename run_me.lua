require 'pascal'
require 'training'

dofile('dataload.lua')
print('load datas')



training({plot_iter =4000,end_iter = 60*1000,print_iter=1000,save_iter=4000})

