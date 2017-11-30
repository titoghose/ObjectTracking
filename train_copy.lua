--[[
DeepTracking: Seeing Beyond Seeing Using Recurrent Neural Networks.
Copyright (C) 2016  Peter Ondruska, Mobile Robotics Group, University of Oxford
email:   ondruska@robots.ox.ac.uk.
webpage: http://mrg.robots.ox.ac.uk/

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.
--]]

require('torch')
cmd = torch.CmdLine()
require('nn')
require('lfs')
require('nngraph')
require('optim')
require('image')
require('SensorData')
require('WeightedBCECriterion')
require('Recurrent')

cmd:option('-gpu', 0, 'use GPU')
cmd:option('-iter', 100000, 'the number of training iterations')
cmd:option('-N', 40, 'training sequence length')
cmd:option('-model', 'model_v3_LSTM', 'neural network model')
cmd:option('-data', 'Data/LidarData1.t7', 'training data')
cmd:option('-learningRate', 0.01, 'learning rate')
cmd:option('-initweights', '', 'initial weights')

params = cmd:parse(arg)

cmd:log('log_' .. params.model .. '.txt', params)

-- switch to GPU
if params.gpu > 0 then
	print('Using GPU ' .. params.gpu)
	require('cunn')
	require('cutorch')
	cutorch.setDevice(params.gpu)
	DEFAULT_TENSOR_TYPE = 'torch.CudaTensor'
else
	print('Using CPU')
	DEFAULT_TENSOR_TYPE = 'torch.FloatTensor'
end

torch.setdefaulttensortype(DEFAULT_TENSOR_TYPE)

-- load training data
print('Loading training data from file ' .. params.data)
--data = torch.load(params.data) -- load pre-processed 2D grid sensor input
data1 = LoadSensorData('./Data/LidarData1.t7', params)
data2 = LoadSensorData('./Data/LidarData2.t7', params)
data3 = LoadSensorData('./Data/LidarData3.t7', params)
data4 = LoadSensorData('./Data/LidarData4.t7', params)
data5 = LoadSensorData('./Data/LidarData5.t7', params)
width  = (#data1)[4] -- occupancy 2D grid width
height = (#data1)[3] -- occupancy 2D grid height
print('Occupancy grid has size ' .. width .. 'x' .. height)
M1 = math.floor((#data1)[1] / params.N) -- total number of training sequences
M2 = math.floor((#data2)[1] / params.N) -- total number of training sequences
M3 = math.floor((#data3)[1] / params.N) -- total number of training sequences
M4 = math.floor((#data4)[1] / params.N) -- total number of training sequences
M5 = math.floor((#data5)[1] / params.N) -- total number of training sequences
print('Number of sequences ' .. M1.." "..M2.." "..M3.." "..M4.." "..M5)

-- load neural network model
require(params.model)
-- initial hidden state
h0 = torch.zeros(48, height, width)
c0 = torch.zeros(48, height, width)
-- one step of RNN
step = getStepModule(width, height)

-- network weights + gradients
w, dw = step:getParameters()
print('Model has ' .. w:numel() .. ' parameters')

if #params.initweights > 0 then
	print('Loading weights ' .. params.initweights)
	w:copy(torch.load(params.initweights))
end

-- chain N steps into a recurrent neural network
model = Recurrent(step, params.N)

-- cost function
-- {y1, y2, ..., yN},{t1, t2, ..., tN} -> cost
criterion = nn.ParallelCriterion()
for i=1,params.N do
	criterion:add(WeightedBCECriterion(), 1/params.N)
end

-- return i-th training sequence
function getSequence(i)
	local input = {}
	for y = 1,params.N do
		input[y] = data[(i-1) * params.N + y]:type(DEFAULT_TENSOR_TYPE)
	end
	return input
end

-- filter and save model performance on a sample sequence
function evalModel(weights)
	input = getSequence(5, data4)
	table.insert(input, h0)
	table.insert(input, c0)
	w:copy(weights)
	local output = model:forward(input)
	-- temporarily switch to FloatTensor as image does not work otherwise.
	torch.setdefaulttensortype('torch.FloatTensor')
	for i = 1,#input-2 do
		image.save('video_' .. params.model .. '/input' .. i .. '.png',  input[i][2] / 2 + input[i][1])
		image.save('video_' .. params.model .. '/output' .. i .. '.png', input[i][2] / 2 + output[i])
	end
	torch.setdefaulttensortype(DEFAULT_TENSOR_TYPE)
end

-- blanks part of the sequence for predictive training
function dropoutInput(target)
	local input = {}
	for i=1,#target do
		input[i] = target[i]:clone()
		if (i-1) % 20 >= 10 then
		    input[i]:zero()
		end
	end
	return input
end

-- evaluates model on a random input
function trainModel(weights)
	-- input and target
	local target = getSequence(j)
	local input  = dropoutInput(target)
	table.insert(input, h0)
	table.insert(input, c0)
	-- forward pass
	w:copy(weights)
	local output = model:forward(input)
	local cost   = criterion:forward(output, target)
	-- backward pass
	dw:zero()
	model:backward(input, criterion:backward(output, target) )
	-- return cost and weight gradients
	return {cost}, dw
end

-- create directory to save weights and videos
lfs.mkdir('weights_' .. params.model)
lfs.mkdir('video_'   .. params.model)

local total_cost, config, state = 0, { learningRate = params.learningRate }, {}
collectgarbage()

--j = 1
for x = 1, params.iter do
	xlua.progress(x, params.iter)	
	-- train on first data file
	j = 1
	data = data1
	for k = 1, M1 do
		xlua.progress(k, M1)
		local _, cost = optim.adagrad(trainModel, w, config, state)
		total_cost = total_cost + cost[1][1]
		j = j+1
		-- not to run out of memory
		collectgarbage()
	end
	print("End of 1")
	-- train on second data file
	j = 1
	data = data2
	for k = 1, M2 do
		local _, cost = optim.adagrad(trainModel, w, config, state)
		total_cost = total_cost + cost[1][1]
		j = j+1
		-- not to run out of memory
		collectgarbage()
	end
	print("End of 2")
	-- train on third data file
	j = 1
	data = data3
	for k = 1, M3 do
		local _, cost = optim.adagrad(trainModel, w, config, state)
		total_cost = total_cost + cost[1][1]
		j = j+1
		-- not to run out of memory
		collectgarbage()
	end
	print("End of 3")
	-- train on fourth data file
	j = 1
	data = data4
	for k = 1, M4 do
		local _, cost = optim.adagrad(trainModel, w, config, state)
		total_cost = total_cost + cost[1][1]
		j = j+1
		-- not to run out of memory
		collectgarbage()
	end
	print("End of 4")
	-- train on fifth data file
	j = 1
	data = data5
	for k = 1, M5 do
		local _, cost = optim.adagrad(trainModel, w, config, state)
		total_cost = total_cost + cost[1][1]
		j = j+1
		-- not to run out of memory
		collectgarbage()
	end
	print("End of 5")
	-- evaluate after each epoch i.e one iteration through all five data files
	print('Iteration ' .. x .. ', cost: ' .. total_cost)
	total_cost = 0
	-- save weights
	torch.save('weights_' .. params.model .. '/' .. x .. '.dat', w:type('torch.FloatTensor'))
	-- visualise performance
	evalModel(w)
	collectgarbage()
end 
