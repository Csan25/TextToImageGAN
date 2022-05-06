--[[ 

Generic training script for GAN, GAN-CLS, GAN-INT, GAN-CLS-INT.

--]]
require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'cunn'
require 'cudnn'

opt = {
   numCaption = 1,
   large = 0,
   save_every = 1,
   print_every = 1,
   dataset = 'coco',
   img_dir = '',
   cls_weight = 0,
   filenames = '',
   data_root = '',
   checkpoint_dir = '//home/research_centre_gpu/Keyboard/checkpoint',
   batchSize = 64,
   doc_length = 201,
   loadSize = 76,
   txtSize = 1024,         -- #  of dim for raw text.
   fineSize = 64,
   nt = 256,               -- #  of dim for text features.
   nz = 100,               -- #  of dim for Z
   ngf = 128,              -- #  of gen filters in first conv layer
   ndf = 64,               -- #  of discrim filters in first conv layer
   nThreads = 4,           -- #  of data loading threads to use
   niter = 1,             -- #  of iter at starting learning rate
   lr = 0.0002,            -- initial learning rate for adam
   lr_decay = 0.5,            -- initial learning rate for adam
   decay_every = 1,
   beta1 = 0.5,            -- momentum term of adam
   ntrain = math.huge,     -- #  of examples per epoch. math.huge for full dataset
   display = 1,            -- display samples while training. 0 = false
   display_id = 10,        -- display window id.
   gpu = 1,                -- gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
   name = 'coco',
   noise = 'normal',       -- uniform / normal
   init_g = '/home/research_centre_gpu/Keyboard/icml2016-master/coco_fast_t70_nc3_nt128_nz100_bs64_cls0.5_ngf196_ndf196_100_net_G.t7',
   init_d = '',
   use_cudnn = 1,
   nc = 3,
}

-- one-line argument parser. parses enviroment variables to override the defaults
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)
if opt.display == 0 then opt.display = false end

if opt.gpu > 0 then
   ok, cunn = pcall(require, 'cunn')
   ok2, cutorch = pcall(require, 'cutorch')
   cutorch.setDevice(opt.gpu)
end

opt.manualSeed = torch.random(1, 10000) -- fix seed
print("Random Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)
torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')

-- create data loader
local DataLoader = paths.dofile('data/data.lua')
local data = DataLoader.new(opt.nThreads, opt.dataset, opt)
print("Dataset: " .. opt.dataset, " Size: ", data:size())
----------------------------------------------------------------------------
local function weights_init(m)
   local name = torch.type(m)
   if name:find('Convolution') then
      m.weight:normal(0.0, 0.02)
      m.bias:fill(0)
   elseif name:find('BatchNormalization') then
      if m.weight then m.weight:normal(1.0, 0.02) end
      if m.bias then m.bias:fill(0) end
   end
end


local nc = 3
local nz = opt.nz
local ndf = opt.ndf
local ngf = opt.ngf
local real_label = 1
local fake_label = 0

local SpatialBatchNormalization = nn.SpatialBatchNormalization
local SpatialConvolution = nn.SpatialConvolution
local SpatialFullConvolution = nn.SpatialFullConvolution

-- Load stage1 gan
netG1 = torch.load(opt.init_g)
netG1:evaluate()
-- define encoder for 1st image
-- encoder = nn.Sequential()
-- encoder:add(SpatialConvolution(nc, ngf, 3, 3, 1, 1, 1 ,1)):add(nn.ReLU(true))
-- encoder:add(SpatialConvolution(ngf, ngf * 2, 4, 4, 2, 2, 1 ,1))
-- encoder:add(SpatialBatchNormalization(ngf * 2)):add(nn.ReLU(true))
-- encoder:add(SpatialConvolution(ngf*2, ngf * 4, 4, 4, 2, 2, 1 ,1))
-- encoder:add(SpatialBatchNormalization(ngf * 4)):add(nn.ReLU(true))

--formula for spatial convolution
--nn.SpatialConvolution(nInputPlane, nOutputPlane, kW, kH, [dW], [dH], [padW], [padH])
--owidth  = floor((width  + 2*padW - kW) / dW + 1)
--oheight = floor((height + 2*padH - kH) / dH + 1)

--formula for full convolution
--nn.SpatialFullConvolution(nInputPlane, nOutputPlane, kW, kH, [dW], [dH], [padW], [padH], [adjW], [adjH])
--owidth  = (width  - 1) * dW - 2*padW + kW + adjW
--oheight = (height - 1) * dH - 2*padH + kH + adjH

--old generator
convF = nn.Sequential()
  -- input is (nc) x 64 x 64 256
  convF:add(SpatialConvolution(nc, ngf, 4, 4, 2, 2, 1, 1))
  convF:add(nn.LeakyReLU(0.2, true))
  -- state size: (ndf) xx 32 x 32 128
  convF:add(SpatialConvolution(ngf, ngf * 2, 4, 4, 2, 2, 1, 1))
  convF:add(SpatialBatchNormalization(ngf * 2))
  convF:add(nn.LeakyReLU(0.2, true))

  -- state size: (ngf*2) x 16 x 16 64
  convF:add(SpatialConvolution(ngf * 2, ngf * 4, 4, 4, 2, 2, 1, 1))
  convF:add(SpatialBatchNormalization(ngf * 4))
  convF:add(nn.LeakyReLU(0.2, true))

  -- state size: (ngf*4) x 8 x 8 32
  convF:add(SpatialConvolution(ngf * 4, ngf * 8, 4, 4, 2, 2, 1, 1))
  convF:add(SpatialBatchNormalization(ngf * 8))

    -- state size: (ngf*8) x 4 x 4 16
	convF:add(SpatialConvolution(ngf * 8, ngf * 16, 4, 4, 2, 2, 1, 1))
	convF:add(SpatialBatchNormalization(ngf * 16))

	--8
	convF:add(SpatialConvolution(ngf * 16, ngf * 32, 4, 4, 2, 2, 1, 1))
	convF:add(SpatialBatchNormalization(ngf * 32))

	--4
    local conc = nn.ConcatTable()
    local conv = nn.Sequential()
    conv:add(SpatialConvolution(ngf * 32, ngf * 2, 1, 1, 1, 1, 0, 0))
    conv:add(SpatialBatchNormalization(ngf * 2)):add(nn.LeakyReLU(0.2, true))
    conv:add(SpatialConvolution(ngf * 2, ngf * 2, 3, 3, 1, 1, 1, 1))
    conv:add(SpatialBatchNormalization(ngf * 2)):add(nn.LeakyReLU(0.2, true))
    conv:add(SpatialConvolution(ngf * 2, ngf * 32, 3, 3, 1, 1, 1, 1))
    conv:add(SpatialBatchNormalization(ngf * 32))
    conc:add(nn.Identity())
    conc:add(conv)
    convF:add(conc)
    convF:add(nn.CAddTable())

  convF:add(nn.LeakyReLU(0.2, true))

  local fcG = nn.Sequential()
  fcG:add(nn.Linear(opt.txtSize,opt.nt))
  fcG:add(nn.LeakyReLU(0.2,true))
  fcG:add(nn.Replicate(4,3))
  fcG:add(nn.Replicate(4,4)) 
  netG = nn.Sequential()
  pt = nn.ParallelTable()
  pt:add(convF)
  pt:add(fcG)
  netG:add(pt)
  netG:add(nn.JoinTable(2))

  -- state size: (ngf*8 + 128) x 4 x 4
  netG:add(SpatialFullConvolution(ngf * 32 + opt.nt, ngf * 8, 4, 4, 2, 2, 1, 1))
  netG:add(SpatialBatchNormalization(ndf * 8)):add(nn.LeakyReLU(0.2, true))

  -- state size: (ngf*4) x 8 x 8
  netG:add(SpatialFullConvolution(ngf * 8, ngf * 4, 4, 4, 2, 2, 1, 1))
  netG:add(SpatialBatchNormalization(ngf * 4))
  netG:add(nn.ReLU(true))
  
  -- state size: (ngf*2) x 16 x 16
  netG:add(SpatialFullConvolution(ngf * 4, ngf, 4, 4, 2, 2, 1, 1))
  netG:add(SpatialBatchNormalization(ngf)):add(nn.ReLU(true))

  -- 32 x 32
  netG:add(SpatialFullConvolution(ngf, math.floor(ngf/2), 4, 4, 2, 2, 1, 1))
  netG:add(SpatialBatchNormalization(math.floor(ngf/2)):add(nn.ReLU(true))

  -- 64 x 64
  netG:add(SpatialFullConvolution(math.floor(ngf/2), math.floor(ngf/4), 4, 4, 2, 2, 1, 1))
  netG:add(SpatialBatchNormalization(math.floor(ngf/4))):add(nn.ReLU(true))

  -- state size: (ngf/4) x 128 x 128
  netG:add(SpatialFullConvolution(math.floor(ngf/4), nc, 4, 4, 2, 2, 1, 1))
  netG:add(nn.Tanh())

  -- state size: (nc) x 256 x 256
  netG:apply(weights_init)

--end


-- -- create stage 2 gan
-- convG = nn.Sequential()
--   -- input is (nc) x 64 x 64
--   convG:add(SpatialConvolution(nc, ngf, 4, 4, 2, 2, 1, 1))
--   convG:add(nn.LeakyReLU(0.2, true))
--   -- state size: (ndf) x 32 x 32
--   convG:add(SpatialConvolution(ngf, ngf * 2, 4, 4, 2, 2, 1, 1))
--   convG:add(SpatialBatchNormalization(ngf * 2))
--   convG:add(nn.LeakyReLU(0.2, true))

--   -- state size: (ndf*2) x 16 x 16
--   convG:add(SpatialConvolution(ngf * 2, ngf * 4, 4, 4, 2, 2, 1, 1))
--   convG:add(SpatialBatchNormalization(ngf * 4))
--   convG:add(nn.LeakyReLU(0.2, true))

--   -- state size: (ndf*4) x 8 x 8
--   convG:add(SpatialConvolution(ngf * 4, ngf * 8, 4, 4, 2, 2, 1, 1))
--   convG:add(SpatialBatchNormalization(ngf * 8))

--     -- state size: (ndf*8) x 4 x 4
--     local conc = nn.ConcatTable()
--     local conv = nn.Sequential()
--     conv:add(SpatialConvolution(ngf * 8, ngf * 2, 1, 1, 1, 1, 0, 0))
--     conv:add(SpatialBatchNormalization(ngf * 2)):add(nn.LeakyReLU(0.2, true))
--     conv:add(SpatialConvolution(ngf * 2, ngf * 2, 3, 3, 1, 1, 1, 1))
--     conv:add(SpatialBatchNormalization(ngf * 2)):add(nn.LeakyReLU(0.2, true))
--     conv:add(SpatialConvolution(ngf * 2, ngf * 8, 3, 3, 1, 1, 1, 1))
--     conv:add(SpatialBatchNormalization(ngf * 8))
--     conc:add(nn.Identity())
--     conc:add(conv)
--     convG:add(conc)
--     convG:add(nn.CAddTable())

--   convG:add(nn.LeakyReLU(0.2, true))

-- -- ca net
-- fcG = nn.Sequential()
-- fcG:add(nn.Linear(opt.txtSize,opt.nt))
-- fcG:add(nn.LeakyReLU(0.2,true))

-- netG = nn.Sequential()
-- -- concat Z and txt
-- ptg = nn.ParallelTable()
-- ptg:add(convG)
-- ptg:add(fcG)
-- netG:add(ptg)
-- netG:add(nn.JoinTable(2))

-- -- input is Z, going into a convolution
-- netG:add(SpatialFullConvolution(ngf * 8 + opt.nt, ngf * 4, 4, 4))
-- netG:add(SpatialBatchNormalization(ngf * 4)):add(nn.ReLU(true))

-- --   -- state size: (ngf*8) x 4 x 4
-- --   local conc = nn.ConcatTable()
-- --   local conv = nn.Sequential()
-- --   conv:add(SpatialConvolution(ngf * 8, ngf * 2, 1, 1, 1, 1, 0, 0))
-- --   conv:add(SpatialBatchNormalization(ngf * 2)):add(nn.ReLU(true))
-- --   conv:add(SpatialConvolution(ngf * 2, ngf * 2, 3, 3, 1, 1, 1, 1))
-- --   conv:add(SpatialBatchNormalization(ngf * 2)):add(nn.ReLU(true))
-- --   conv:add(SpatialConvolution(ngf * 2, ngf * 8, 3, 3, 1, 1, 1, 1))
-- --   conv:add(SpatialBatchNormalization(ngf * 8))
-- --   conc:add(nn.Identity())
-- --   conc:add(conv)
-- --   netG:add(conc)
-- --   netG:add(nn.CAddTable())

-- --   if opt.large == 1 then
-- --     -- state size: (ngf*8) x 4 x 4
-- --     local conc = nn.ConcatTable()
-- --     local conv = nn.Sequential()
-- --     conv:add(SpatialConvolution(ngf * 8, ngf * 2, 1, 1, 1, 1, 0, 0))
-- --     conv:add(SpatialBatchNormalization(ngf * 2)):add(nn.ReLU(true))
-- --     conv:add(SpatialConvolution(ngf * 2, ngf * 2, 3, 3, 1, 1, 1, 1))
-- --     conv:add(SpatialBatchNormalization(ngf * 2)):add(nn.ReLU(true))
-- --     conv:add(SpatialConvolution(ngf * 2, ngf * 8, 3, 3, 1, 1, 1, 1))
-- --     conv:add(SpatialBatchNormalization(ngf * 8))
-- --     conc:add(nn.Identity())
-- --     conc:add(conv)
-- --     netG:add(conc)
-- --     netG:add(nn.CAddTable())
-- --   end

-- --   netG:add(nn.ReLU(true))

-- -- -- state size: (ngf*8) x 4 x 4
-- -- netG:add(SpatialFullConvolution(ngf * 8, ngf * 4, 4, 4, 2, 2, 1, 1))
-- -- netG:add(SpatialBatchNormalization(ngf * 4))

--   -- resblock = nn.Sequential()
--   -- resblock:add(SpatialConvolution(ngf * 4, ngf * 4, 3, 3, 1, 1, 1, 1))
--   -- resblock:add(SpatialBatchNormalization(ngf * 4)):add(nn.ReLU(true))
--   -- resblock:add(SpatialConvolution(ngf * 4, ngf * 4, 3, 3, 1, 1, 1, 1))
--   -- resblock:add(SpatialBatchNormalization(ngf * 4))
--   -- resblock:add(SpatialConvolution(ngf * 4, ngf * 4, 3, 3, 1, 1, 1, 1))
--   -- resblock:add(SpatialBatchNormalization(ngf * 4)):add(nn.ReLU(true))
--   -- resblock:add(SpatialConvolution(ngf * 4, ngf * 4, 3, 3, 1, 1, 1, 1))
--   -- resblock:add(SpatialBatchNormalization(ngf * 4))
--   -- netG:add(resblock)

-- -- state size: (ngf*4) x 8 x 8
-- -- local conc = nn.ConcatTable()
-- -- local conv = nn.Sequential()
-- -- conv:add(SpatialConvolution(ngf * 4, ngf, 1, 1, 1, 1, 0, 0))
-- -- conv:add(SpatialBatchNormalization(ngf)):add(nn.ReLU(true))
-- -- conv:add(SpatialConvolution(ngf, ngf, 3, 3, 1, 1, 1, 1))
-- -- conv:add(SpatialBatchNormalization(ngf)):add(nn.ReLU(true))
-- -- conv:add(SpatialConvolution(ngf, ngf * 4, 3, 3, 1, 1, 1, 1))
-- -- conv:add(SpatialBatchNormalization(ngf * 4))
-- -- conc:add(nn.Identity())
-- -- conc:add(conv)
-- -- netG:add(conc)
-- -- netG:add(nn.CAddTable())

-- -- if opt.large == 1 then
-- --   -- state size: (ngf*4) x 8 x 8
-- --   local conc = nn.ConcatTable()
-- --   local conv = nn.Sequential()
-- --   conv:add(SpatialConvolution(ngf * 4, ngf, 1, 1, 1, 1, 0, 0))
-- --   conv:add(SpatialBatchNormalization(ngf)):add(nn.ReLU(true))
-- --   conv:add(SpatialConvolution(ngf, ngf, 3, 3, 1, 1, 1, 1))
-- --   conv:add(SpatialBatchNormalization(ngf)):add(nn.ReLU(true))
-- --   conv:add(SpatialConvolution(ngf, ngf * 4, 3, 3, 1, 1, 1, 1))
-- --   conv:add(SpatialBatchNormalization(ngf * 4))
-- --   conc:add(nn.Identity())
-- --   conc:add(conv)
-- --   netG:add(conc)
-- --   netG:add(nn.CAddTable())
-- -- end

-- -- netG:add(nn.ReLU(true))

-- -- try it out --
-- -- state size: (ngf*4) x 8 x 8
-- netG:add(SpatialFullConvolution(ngf * 4, math.floor(ngf/8), 4, 4, 2, 2, 1, 1))
-- netG:add(SpatialBatchNormalization(ngf * 2))
-- netG:add(nn.ReLU(true))

-- -- -- state size: (ngf*2) x 16 x 16
-- -- netG:add(SpatialFullConvolution(ngf * 2, ngf, 4, 4, 2, 2, 1, 1))
-- -- netG:add(SpatialBatchNormalization(ngf)):add(nn.ReLU(true))

-- -- netG:add(SpatialFullConvolution(ngf, math.floor(ngf/2), 4, 4, 2, 2, 1, 1))
-- -- netG:add(SpatialBatchNormalization(ngf)):add(nn.ReLU(true))

-- -- netG:add(SpatialFullConvolution(math.floor(ngf/2), math.floor(ngf/4), 4, 4, 2, 2, 1, 1))
-- -- netG:add(SpatialBatchNormalization(ngf)):add(nn.ReLU(true))

-- -- netG:add(SpatialFullConvolution(math.floor(ngf/4), math.floor(ngf/8), 4, 4, 2, 2, 1, 1))
-- -- netG:add(SpatialBatchNormalization(ngf)):add(nn.ReLU(true))

-- -- state size: (ngf) x 32 x 32
-- netG:add(SpatialFullConvolution(math.floor(ngf/8), nc, 4, 4, 2, 2, 1, 1))
-- netG:add(nn.Tanh())

-- -- state size: (nc) x 64 x 64
-- netG:apply(weights_init)

-- -- end --



-- -- state size: (ngf*4) x 8 x 8
-- netG:add(SpatialFullConvolution(ngf * 4, ngf * 2, 4, 4, 2, 2, 1, 1))
-- netG:add(SpatialBatchNormalization(ngf * 2))
-- netG:add(nn.ReLU(true))

-- -- state size: (ngf*2) x 16 x 16
-- netG:add(SpatialFullConvolution(ngf * 2, ngf, 4, 4, 2, 2, 1, 1))
-- netG:add(SpatialBatchNormalization(ngf)):add(nn.ReLU(true))

-- netG:add(SpatialFullConvolution(ngf, math.floor(ngf/2), 4, 4, 2, 2, 1, 1))
-- netG:add(SpatialBatchNormalization(ngf)):add(nn.ReLU(true))

-- netG:add(SpatialFullConvolution(math.floor(ngf/2), math.floor(ngf/4), 4, 4, 2, 2, 1, 1))
-- netG:add(SpatialBatchNormalization(ngf)):add(nn.ReLU(true))

-- netG:add(SpatialFullConvolution(math.floor(ngf/4), math.floor(ngf/8), 4, 4, 2, 2, 1, 1))
-- netG:add(SpatialBatchNormalization(ngf)):add(nn.ReLU(true))

-- -- state size: (ngf) x 32 x 32
-- netG:add(SpatialFullConvolution(math.floor(ngf/8), nc, 4, 4, 2, 2, 1, 1))
-- netG:add(nn.Tanh())

-- -- state size: (nc) x 64 x 64
-- netG:apply(weights_init)


-- -- Create stage 2 discriminator

-- convD = nn.Sequential()
-- -- input is (nc) x 64 x 64
-- convD:add(SpatialConvolution(nc, ndf, 4, 4, 2, 2, 1, 1))
-- convD:add(nn.LeakyReLU(0.2, true))
-- -- state size: (ndf) x 32 x 32
-- -- convD:add(SpatialConvolution(ndf, ndf * 2, 4, 4, 2, 2, 1, 1))
-- -- convD:add(SpatialBatchNormalization(ndf * 2))
-- -- convD:add(nn.LeakyReLU(0.2, true))

-- -- -- state size: (ndf*2) x 16 x 16
-- -- convD:add(SpatialConvolution(ndf * 2, ndf * 4, 4, 4, 2, 2, 1, 1))
-- -- convD:add(SpatialBatchNormalization(ndf * 4))
-- -- convD:add(nn.LeakyReLU(0.2, true))

-- -- -- state size: (ndf*4) x 8 x 8
-- -- convD:add(SpatialConvolution(ndf * 4, ndf * 8, 4, 4, 2, 2, 1, 1))
-- -- convD:add(SpatialBatchNormalization(ndf * 8))
-- -- convD:add(nn.LeakyReLU(0.2, true))

-- -- convD:add(SpatialConvolution(ndf * 8, ndf * 16, 4, 4, 2, 2, 1, 1))
-- -- convD:add(SpatialBatchNormalization(ndf * 16))
-- -- convD:add(nn.LeakyReLU(0.2, true))

-- convD:add(SpatialConvolution(ndf, ndf * 32, 4, 4, 2, 2, 1, 1))
-- convD:add(SpatialBatchNormalization(ndf * 32))
-- convD:add(nn.LeakyReLU(0.2, true))

-- -- convD:add(SpatialConvolution(ndf * 32, ndf * 16, 4, 4, 2, 2, 1, 1))
-- -- convD:add(SpatialBatchNormalization(ndf * 16))
-- -- convD:add(nn.LeakyReLU(0.2, true))

-- convD:add(SpatialConvolution(ndf * 32, ndf * 8, 4, 4, 2, 2, 1, 1))
-- convD:add(SpatialBatchNormalization(ndf * 8))
-- convD:add(nn.LeakyReLU(0.2, true))


--   -- state size: (ndf*8) x 4 x 4
--   local conc = nn.ConcatTable()
--   local conv = nn.Sequential()
--   conv:add(SpatialConvolution(ndf * 8, ndf * 2, 1, 1, 1, 1, 0, 0))
--   conv:add(SpatialBatchNormalization(ndf * 2)):add(nn.LeakyReLU(0.2, true))
--   conv:add(SpatialConvolution(ndf * 2, ndf * 2, 3, 3, 1, 1, 1, 1))
--   conv:add(SpatialBatchNormalization(ndf * 2)):add(nn.LeakyReLU(0.2, true))
--   conv:add(SpatialConvolution(ndf * 2, ndf * 8, 3, 3, 1, 1, 1, 1))
--   conv:add(SpatialBatchNormalization(ndf * 8))
--   conc:add(nn.Identity())
--   conc:add(conv)
--   convD:add(conc)
--   convD:add(nn.CAddTable())

-- convD:add(nn.LeakyReLU(0.2, true))

-- local fcD = nn.Sequential()
-- fcD:add(nn.Linear(opt.txtSize,opt.nt))
-- fcD:add(nn.LeakyReLU(0.2,true))
-- fcD:add(nn.Replicate(4,3))
-- fcD:add(nn.Replicate(4,4)) 
-- netD = nn.Sequential()
-- pt = nn.ParallelTable()
-- pt:add(convD)
-- pt:add(fcD)
-- netD:add(pt)
-- netD:add(nn.JoinTable(2))
-- -- state size: (ndf*8 + 128) x 4 x 4
-- netD:add(SpatialConvolution(ndf * 8 + opt.nt, ndf * 8, 1, 1))
-- netD:add(SpatialBatchNormalization(ndf * 8)):add(nn.LeakyReLU(0.2, true))
-- netD:add(SpatialConvolution(ndf * 8, 1, 4, 4))
-- netD:add(nn.Sigmoid())
-- -- state size: 1 x 1 x 1
-- netD:add(nn.View(1):setNumInputDims(3))
-- -- state size: 1
-- netD:apply(weights_init)

--start
convD = nn.Sequential()
  -- input is (nc) x 64 x 64
  convD:add(SpatialConvolution(nc, ndf, 4, 4, 2, 2, 1, 1))
  convD:add(nn.LeakyReLU(0.2, true))
  -- state size: (ndf) x 32 x 32
  convD:add(SpatialConvolution(ndf, ndf * 2, 4, 4, 2, 2, 1, 1))
  convD:add(SpatialBatchNormalization(ndf * 2))
  convD:add(nn.LeakyReLU(0.2, true))

  -- state size: (ndf*2) x 16 x 16
  convD:add(SpatialConvolution(ndf * 2, ndf * 4, 4, 4, 2, 2, 1, 1))
  convD:add(SpatialBatchNormalization(ndf * 4))
  convD:add(nn.LeakyReLU(0.2, true))

  -- state size: (ndf*4) x 8 x 8
  convD:add(SpatialConvolution(ndf * 4, ndf * 8, 4, 4, 2, 2, 1, 1))
  convD:add(SpatialBatchNormalization(ndf * 8))

    -- state size: (ndf*8) x 4 x 4
    local conc = nn.ConcatTable()
    local conv = nn.Sequential()
    conv:add(SpatialConvolution(ndf * 8, ndf * 2, 1, 1, 1, 1, 0, 0))
    conv:add(SpatialBatchNormalization(ndf * 2)):add(nn.LeakyReLU(0.2, true))
    conv:add(SpatialConvolution(ndf * 2, ndf * 2, 3, 3, 1, 1, 1, 1))
    conv:add(SpatialBatchNormalization(ndf * 2)):add(nn.LeakyReLU(0.2, true))
    conv:add(SpatialConvolution(ndf * 2, ndf * 8, 3, 3, 1, 1, 1, 1))
    conv:add(SpatialBatchNormalization(ndf * 8))
    conc:add(nn.Identity())
    conc:add(conv)
    convD:add(conc)
    convD:add(nn.CAddTable())

  convD:add(nn.LeakyReLU(0.2, true))

  local fcD = nn.Sequential()
  fcD:add(nn.Linear(opt.txtSize,opt.nt))
  fcD:add(nn.LeakyReLU(0.2,true))
  fcD:add(nn.Replicate(4,3))
  fcD:add(nn.Replicate(4,4)) 
  netD = nn.Sequential()
  pt = nn.ParallelTable()
  pt:add(convD)
  pt:add(fcD)
  netD:add(pt)
  netD:add(nn.JoinTable(2))
  -- state size: (ndf*8 + 128) x 4 x 4
  netD:add(SpatialConvolution(ndf * 8 + opt.nt, ndf * 8, 1, 1))
  netD:add(SpatialBatchNormalization(ndf * 8)):add(nn.LeakyReLU(0.2, true))
  netD:add(SpatialConvolution(ndf * 8, 1, 4, 4))
  netD:add(nn.Sigmoid())
  -- state size: 1 x 1 x 1
  netD:add(nn.View(1):setNumInputDims(3))
  -- state size: 1
  netD:apply(weights_init)
--end

local criterion = nn.BCECriterion()
---------------------------------------------------------------------------
optimStateG = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}
optimStateD = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}
----------------------------------------------------------------------------
local input_img = torch.Tensor(opt.batchSize, 3, opt.fineSize, opt.fineSize)
local input_img2 = torch.Tensor(opt.batchSize, 3, opt.fineSize, opt.fineSize)

local input_txt_emb1 = torch.Tensor(opt.batchSize, opt.txtSize)

local noise = torch.Tensor(opt.batchSize, nz, 1, 1)

local label = torch.Tensor(opt.batchSize)

local errD, errG, errW

local epoch_tm = torch.Timer()
local tm = torch.Timer()
local data_tm = torch.Timer()
----------------------------------------------------------------------------
if opt.gpu > 0 then
   input_img = input_img:cuda()
   input_img2 = input_img2:cuda()

   input_txt_emb1 = input_txt_emb1:cuda()

   noise = noise:cuda()
   label = label:cuda()

   netD:cuda()
   netG:cuda()
   netG1:cuda()
   criterion:cuda()
end

if opt.use_cudnn == 1 then
  cudnn = require('cudnn')
  netD = cudnn.convert(netD, cudnn)
  netG = cudnn.convert(netG, cudnn)
  netG1 =  cudnn.convert(netG1, cudnn)
end

local parametersD, gradParametersD = netD:getParameters()
local parametersG, gradParametersG = netG:getParameters()

if opt.display then disp = require 'display' end

local sample = function()
  data_tm:reset(); data_tm:resume()
  real_img, wrong_img, real_txt = data:getBatch()
  data_tm:stop()

  
  
  input_img:copy(real_img)
  input_img2:copy(wrong_img)
  input_txt_emb1:copy(real_txt)
end

-- create closure to evaluate f(X) and df/dX of discriminator
fake_score = 0.5

local fDx = function(x)
  netD:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
  netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)

  gradParametersD:zero()

  -- train with real
  label:fill(real_label)

  local output = netD:forward{input_img,input_txt_emb1}
  local errD_real = criterion:forward(output, label)
  local df_do = criterion:backward(output, label)
  netD:backward({input_img, input_txt_emb1}, df_do)

  errD_wrong = 0
  if opt.cls_weight > 0 then
    -- train with wrong 
    label:fill(fake_label)

    local output = netD:forward{input_img2, input_txt_emb1}
    errD_wrong = opt.cls_weight*criterion:forward(output, label)
    local df_do = criterion:backward(output, label)
    df_do:mul(opt.cls_weight)
    netD:backward({input_img2, input_txt_emb1}, df_do)
  end

  -- train with fake
  if opt.noise == 'uniform' then -- regenerate random noise
    noise:uniform(-1, 1)
  elseif opt.noise == 'normal' then
    noise:normal(0, 1)
  end
  local fake = netG1:forward{noise, input_txt_emb1}
  print(fake)
  -- print(noise)
  local newfake = netG:forward{fake,input_txt_emb1}
  input_img:copy(newfake)
  label:fill(fake_label)

  local output = netD:forward{input_img, input_txt_emb1}

  -- update fake score tracker
  local cur_score = output:mean()
  fake_score = 0.99 * fake_score + 0.01 * cur_score

  local errD_fake = criterion:forward(output, label)
  local df_do = criterion:backward(output, label)
  local fake_weight = 1 - opt.cls_weight
  errD_fake = errD_fake*fake_weight
  df_do:mul(fake_weight)
  netD:backward({input_img, input_txt_emb1}, df_do)

  errD = errD_real + errD_fake + errD_wrong
  errW = errD_wrong

  return errD, gradParametersD
end

-- create closure to evaluate f(X) and df/dX of generator
local fGx = function(x)
  netD:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
  netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)

  gradParametersG:zero()

  if opt.noise == 'uniform' then -- regenerate random noise
    noise:uniform(-1, 1)
  elseif opt.noise == 'normal' then
    noise:normal(0, 1)
  end
  local fake = netG1:forward{noise, input_txt_emb1}
  local newfake = netG:forward{fake, input_txt_emb1}
  input_img:copy(newfake)
  label:fill(real_label) -- fake labels are real for generator cost

  local output = netD:forward{input_img, input_txt_emb1}

  local cur_score = output:mean()
  fake_score = 0.99 * fake_score + 0.01 * cur_score

  errG = criterion:forward(output, label)
  local df_do = criterion:backward(output, label)
  local df_dg = netD:updateGradInput({input_img, input_txt_emb1}, df_do)

  netG:backward({fake, input_txt_emb1}, df_dg[1])
  return errG, gradParametersG
end


-- train
for epoch = 1, opt.niter do
  epoch_tm:reset()

  if epoch % opt.decay_every == 0 then
    optimStateG.learningRate = optimStateG.learningRate * opt.lr_decay
    optimStateD.learningRate = optimStateD.learningRate * opt.lr_decay
  end

  for i = 1, math.min(data:size(), opt.ntrain), opt.batchSize do
    tm:reset()

    sample()
    optim.adam(fDx, parametersD, optimStateD)
    optim.adam(fGx, parametersG, optimStateG)

    -- logging
    if ((i-1) / opt.batchSize) % opt.print_every == 0 then
      print(('[%d][%d/%d] T:%.3f  DT:%.3f lr: %.4g '
                .. ' G:%.3f  D:%.3f W:%.3f fs:%.2f'):format(
              epoch, ((i-1) / opt.batchSize),
              math.floor(math.min(data:size(), opt.ntrain) / opt.batchSize),
              tm:time().real, data_tm:time().real,
              optimStateG.learningRate,
              errG and errG or -1, errD and errD or -1,
              errW and errW or -1, fake_score))
      local fake = netG.output
      disp.image(fake:narrow(1,1,opt.batchSize), {win=opt.display_id, title=opt.name})
      disp.image(real_img, {win=opt.display_id * 3, title=opt.name})
    end
  end
  if epoch % opt.save_every == 0 then
    paths.mkdir(opt.checkpoint_dir)
    torch.save(opt.checkpoint_dir .. '/' .. opt.name .. '_' .. epoch .. '_net_G_stage2.t7', netG)
    torch.save(opt.checkpoint_dir .. '/' .. opt.name .. '_' .. epoch .. '_net_D_stage2.t7', netD)
    torch.save(opt.checkpoint_dir .. '/' .. opt.name .. '_' .. epoch .. '_opt_stage2.t7', opt)
    print(('End of epoch %d / %d \t Time Taken: %.3f'):format(
           epoch, opt.niter, epoch_tm:time().real))
  end
end

