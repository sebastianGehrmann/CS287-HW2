-- Only requirements allowed
require("hdf5")
require("nn")
require("optim")
require("xlua")

cmd = torch.CmdLine()

-- Cmd Args
cmd:option('-datafile', 'PTB.hdf5', 'data file')
cmd:option('-classifier', 'nb', 'classifier to use')
cmd:option('-dev', 'false', 'narrow training data for development')
cmd:option('-emb', 'false', 'Use Embeddings')

-- Hyperparameters
cmd:option('-alpha', 1, 'extra count for words in NB')
cmd:option('-batchsize', 32, 'Size of the Minibatch for SGD')
cmd:option('-learningrate', .02, 'Learning Rate')
cmd:option('-epochs', 10, 'Number of Epochs')
cmd:option('-hiddenUnits', 50, 'Number of hidden units')
cmd:option('-wordEmbeddingSize', 50, 'Size of the embedding for Words')
cmd:option('-capEmbeddingSize', 50, 'Size of the embedding for Capitalization')
cmd:option('-usePretrainedEmbedding', 'false', 'Use the pretrained vectors for embedding')
cmd:option('-useEmbeddings', 'false', 'Use pretrained Embeddings')
cmd:option('-predictTest', 'false', 'Output prediction file in NN')
-- Helper Functions

function transformLookUp(x, cap)
   for i=1,dwin do
      x:transpose(1,2)[i]:add(nfeatures*(i-1))
      cap:transpose(1,2)[i]:add(nfeatures*dwin+dc*(i-1))
   end
   return x,cap
end

function linearPredictor(LT, b, input, cap)
   sent, caps = transformLookUp(input, cap)
   X = torch.cat(sent, caps, 2)
   --built predictor
   Yhat = torch.IntTensor(input:size(1))
   for row=1,input:size(1) do
      h, i = LT:forward(X[row]):sum(1):add(b):max(2)
      Yhat[row] = i
   end
   return Yhat

end

-- Classifiers

function naiveBayes()
   --Step 1: Create W and b
   local b = torch.DoubleTensor(nclasses):fill(alpha)
   local LT = nn.LookupTable((nfeatures+dc)*dwin, nclasses)
   LT.weight:fill(0)
   --Step 2: Count all the words

   --2a: Count classes for division later
   local wordcount = torch.Tensor(nclasses):fill(alpha*nclasses)
   for row=1, tout:size()[1] do
      wordcount[tout[row]] = wordcount[tout[row]]+1
   end
   print(LT.weight:size()[1], "Number of Features")
   
   b:add(wordcount)
   b:div(tin:size()[1])
   b:log()

   --2b: Compute probabilities for each position in window
   for dw=1, dwin do
      print(dw, "Calculating this window for words")
      local W = torch.DoubleTensor(nfeatures, nclasses):fill(alpha)
      for row=1, tin:size()[1] do
         local w = tin[row][dw]
         local c = tout[row]
         W[w][c] = W[w][c] + 1
      end
      --transform to log-probabilities
      for row=1, W:size()[1] do
         W[row]:cdiv(wordcount)
      end

      W:log()
      --update LookupTable
      LT.weight[{{(dw-1)*nfeatures+1, dw*nfeatures}}] = W
   end
   --Step 2c: Count all the Caps, could do this with last step but in here for clarity
   for dw=1, dwin do
      print(dw, "Calculating this window for Caps")
      local W = torch.DoubleTensor(dc, nclasses):fill(alpha)
      for row=1, tcap:size()[1] do
         local w = tcap[row][dw]
         local c = tout[row]
         W[w][c] = W[w][c] + 1
      end
      --transform to log-probabilities
      for row=1, W:size()[1] do
         W[row]:cdiv(wordcount)
      end
      W:log()
      --update LookupTable
      LT.weight[{{1+ dwin * nfeatures + dc * (dw-1), dwin * nfeatures + dc*dw}}] = W
   end
   
   Yhat = linearPredictor(LT,b,tin,tcap)
   print(torch.eq(Yhat, tout):sum()/tout:size(1), "accuracy on training set")

   Yhat = linearPredictor(LT,b,vin,vcap)
   print(torch.eq(Yhat, vout):sum()/vout:size(1), "accuracy on validation set")

end

function logReg()
   --build model
   sparseLookup = nn.LookupTable(dwin*(nfeatures+dc), nclasses)
   --otherwise the vectors are stacked. We need them concatenated
   reshapeLookup = nn.Reshape(2*dwin*nclasses)
   linearLayer = nn.Linear(2*dwin*nclasses, nclasses)
   softMaxLayer = nn.LogSoftMax()

   model = nn.Sequential()
   model:add(sparseLookup)
   model:add(reshapeLookup)
   model:add(linearLayer)
   model:add(softMaxLayer)

   criterion = nn.ClassNLLCriterion()

   --transform inputs
   sent, caps = transformLookUp(vin, vcap)
   vX = torch.cat(sent, caps, 2)  
   sent, caps = transformLookUp(tin, tcap)
   X = torch.cat(sent, caps, 2)

   model = trainModel(model, criterion, X, vX)
   return model
end

function neuralNet(useEmbeddings)
   --print(vin[1])
   --Transform inputs for validation
   vsent, vcaps = transformLookUp(vin, vcap)
   vcaps:add(-dwin*nfeatures)


   sent, caps = transformLookUp(tin, tcap)
   caps:add(-dwin*nfeatures)

   par = nn.ParallelTable()
   seq1 = nn.Sequential()
   seq2 = nn.Sequential()
   wordEmbeddings = nn.LookupTable(dwin*nfeatures, opt.wordEmbeddingSize)
   capEmbeddings = nn.LookupTable(dwin*nfeatures, opt.capEmbeddingSize)

   --print(wordEmbeddings.weight[100])

   if opt.useEmbeddings then
      for w=1, dwin do
         wordEmbeddings.weight[{{(w-1)*nfeatures+1, w*nfeatures}}] = embeddings
      end
   end

   --print(wordEmbeddings.weight[100])
   seq1:add(wordEmbeddings)--:add(join)
   seq1:add(nn.Reshape(dwin*opt.wordEmbeddingSize))
   seq2:add(capEmbeddings)--:add(join)
   seq2:add(nn.Reshape(dwin*opt.capEmbeddingSize))

   par:add(seq1):add(seq2)
   --par:add(wordEmbeddings):add(capEmbeddings)
   join = nn.JoinTable(2, 2)
   linearLayer = nn.Linear(dwin*(opt.wordEmbeddingSize+opt.capEmbeddingSize), opt.hiddenUnits)
   hardTanhLayer = nn.HardTanh()
   scoringLayer = nn.Linear(opt.hiddenUnits, nclasses)

   mlp = nn.Sequential()
   mlp:add(par)
   mlp:add(join)
   mlp:add(linearLayer)
   mlp:add(hardTanhLayer)
   mlp:add(scoringLayer)
   mlp:add(nn.LogSoftMax())

   criterion = nn.ClassNLLCriterion()


   --preds = mlp:forward({sent, caps})
   --loss = criterion:forward(preds, tout)  


   trainedModel = trainNN(mlp, criterion, sent, caps, vsent, vcaps)


   --print(X:size())
   --print(mlp)
   --print(mlp:forward(X))
   --print(seq1:forward(sent))
   --print(mlp:forward({sent, caps}):size())

end

function trainNN(model, criterion, sent, caps, vsent, vcaps)  
   if opt.predictTest == "true" then

      tsent, tcaps = transformLookUp(testin, testcap)
      tcaps:add(-dwin*nfeatures)
   end   

   print(tsent:size(1), "size of the test set")
   --SGD after torch nn tutorial and https://github.com/torch/tutorials/blob/master/2_supervised/4_train.lua
   for i=1, epochs do
      --shuffle data
      shuffle = torch.randperm(sent:size(1))
      --mini batches, yay
      for t=1, sent:size(1), batchsize do
         xlua.progress(t, sent:size(1))

         local inputs = torch.Tensor(batchsize, sent:size(2))
         local icaps = torch.Tensor(batchsize, caps:size(2))
         local targets = torch.Tensor(batchsize)
         local k = 1
         for i = t,math.min(t+batchsize-1,sent:size(1)) do
            -- load new sample
            inputs[k] = sent[shuffle[i]]
            icaps[k] = caps[shuffle[i]]
            targets[k] = tout[shuffle[i]]
            k = k+1
         end
         k=k-1
         --in case the last batch is < batchsize
         if k < opt.batchsize then
           inputs = inputs:narrow(1, 1, k):clone()
           icaps = icaps:narrow(1, 1, k):clone()
           targets = targets:narrow(1, 1, k):clone()
         end
         --zero out
         model:zeroGradParameters()
         --predict and compute loss
         preds = model:forward({inputs, icaps})
         loss = criterion:forward(preds, targets)      
         dLdpreds = criterion:backward(preds, targets)
         model:backward({inputs, icaps}, dLdpreds)
         model:updateParameters(eta)
      end
      --predicting accuracy of epoch
      preds = model:forward({sent, caps})
      loss = criterion:forward(preds, tout)    
      print()
      --print(preds:max(2))
      print("epoch " .. i .. ", loss: " .. loss)

      _, yhat = preds:max(2)
      print(torch.eq(yhat:type('torch.IntTensor'), tout):sum()/tout:size(1), "accuracy on training set")      
      _, yhat = model:forward({vsent, vcaps}):max(2)
      validationAcc = torch.eq(yhat:type('torch.IntTensor'), vout):sum()/vout:size(1)
      print(validationAcc, "accuracy on validation set")
      print("")

      if opt.predictTest == "true" then
         preds = model:forward({tsent, tcaps})
         print(preds:size(1), "So many test predictions")
         _, classPred = preds:max(2)
         val = string.format("%.4f", validationAcc)
         l = string.format("%.4f", loss)
         filename = "predictions/" .. i .. "-" .. opt.capEmbeddingSize .. "-" .. val .. "-" .. l .. ".txt"

         torch.save(filename, classPred,'ascii')

      
      end
   end



   return model
end



function trainModel(model, criterion, X, vX)  

   --SGD after torch nn tutorial and https://github.com/torch/tutorials/blob/master/2_supervised/4_train.lua
   for i=1, epochs do
      --shuffle data
      shuffle = torch.randperm(X:size(1))
      --mini batches, yay
      for t=1, X:size(1), batchsize do
         xlua.progress(t, X:size(1))

         local inputs = torch.Tensor(batchsize, X:size(2))
         local targets = torch.Tensor(batchsize)
         local k = 1
         for i = t,math.min(t+batchsize-1,X:size(1)) do
            -- load new sample
            inputs[k] = X[shuffle[i]]
            targets[k] = tout[shuffle[i]]
            k = k+1
         end
         k=k-1
         --in case the last batch is < batchsize
         if k < opt.batchsize then
           inputs = inputs:narrow(1, 1, k):clone()
           targets = targets:narrow(1, 1, k):clone()
         end
         --zero out
         model:zeroGradParameters()
         --predict and compute loss
         preds = model:forward(inputs)
         loss = criterion:forward(preds, targets)      
         dLdpreds = criterion:backward(preds, targets)
         model:backward(X, dLdpreds)
         model:updateParameters(eta)
      end
      --predicting accuracy of epoch
      preds = model:forward(X)
      loss = criterion:forward(preds, tout)    
      print("epoch " .. i .. ", loss: " .. loss)
  
      _, yhat = preds:max(2)
      print(torch.eq(yhat:type('torch.IntTensor'), tout):sum()/tout:size(1), "accuracy on training set")      
      _, yhat = model:forward(vX):max(2)
      print(torch.eq(yhat:type('torch.IntTensor'), vout):sum()/vout:size(1), "accuracy on validation set")
   end

   return model
end


function main() 
   torch.setdefaulttensortype('torch.DoubleTensor')
   -- Parse input params
   opt = cmd:parse(arg)
   local f = hdf5.open(opt.datafile, 'r')
   nclasses = f:read('nclasses'):all():long()[1]
   nfeatures = f:read('nfeatures'):all():long()[1]
   alpha = opt.alpha
   dwin = f:read('dwin'):all():long()[1]
   epochs = opt.epochs
   eta = opt.learningrate
   batchsize = opt.batchsize

   print(nclasses, "classes in set")
   print(nfeatures, "features in set")

   -- Read the data here
   tin = f:read('train_input'):all()
   tcap = f:read('train_cap'):all()
   tout = f:read('train_output'):all()

   vin = f:read('valid_input'):all()
   vcap = f:read('valid_cap'):all()
   vout = f:read('valid_output'):all()

   testin = f:read('test_input'):all()
   testcap = f:read('test_cap'):all()

   embeddings = f:read('embeddings'):all()
   dc = 5

   if opt.dev == 'true' then
      print('Development mode')
      print('Narrowing the Training Data to 1000 Samples')
      tin = tin:narrow(1, 1, 1000):clone()
      tcap = tcap:narrow(1, 1, 1000):clone()
      tout = tout:narrow(1, 1, 1000):clone()
   end

   print(testin:size())
   print(testcap:size())

   -- Train.
   if opt.classifier == 'nb' then
      W,b = naiveBayes()
   elseif opt.classifier == 'lr' then
      model = logReg()
   elseif opt.classifier == 'nn' then
      neuralNet()
   end


   -- Test.
end

main()
