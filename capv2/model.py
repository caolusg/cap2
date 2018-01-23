import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from capsule_layer import CapsuleLayer

class  CNN_Text(nn.Module):
    
    def __init__(self, args):
        super(CNN_Text,self).__init__()
        self.args = args
        
        V = args.embed_num
        D = args.embed_dim
        C = args.class_num
        Ci = 1
        Co = args.kernel_num
        Ks = args.kernel_sizes

        self.embed = nn.Embedding(V, D)
        #self.convs1 = [nn.Conv2d(Ci, Co, (K, D)) for K in Ks]
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        
        num_primary_unit = 8
        primary_unit_size = 1
        num_classes = 2
        output_unit_size = 100
        num_routing = 3
        cuda_enabled = False     
        
        # DigitCaps
        # Final layer: Capsule layer where the routing algorithm is.
        self.digits = CapsuleLayer(in_unit=num_primary_unit,
                                   in_channel=primary_unit_size,
                                   num_unit=num_classes,
                                   unit_size=output_unit_size, # 16D capsule per digit class
                                   use_routing=True,
                                   num_routing=num_routing,
                                   cuda_enabled=cuda_enabled)
        '''
        self.conv13 = nn.Conv2d(Ci, Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, Co, (5, D))
        '''
        #self.dropout = nn.Dropout(args.dropout)
        #self.fc1 = nn.Linear(len(Ks)*Co, C)
        #self.fc1 = nn.Linear(output_unit_size, 2)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3) #(N,Co,W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x
    
    

    def forward(self, x):
        x = self.embed(x) # (N,W,D)
        #print('---------------------------------------------------------------------------------')
        #print(type(x))
        #print(len(x))
        
        #print(x.size())
        #print(x[0].size())
        #print(x[1].size())
        
        if self.args.static:
            x = Variable(x)                
        x = x.unsqueeze(1) # (N,Ci,W,D)
        #print('000000000000000000000000000000000000000000000000000000000000000000000000000000000')
        #print(x[0].size())
        #print('111111111111111111111111111111111111111111111111111111111111111111111111111111111')
        t = [conv(x) for conv in self.convs1]
        #print(t[0].size())
        #print('222222222222222222222222222222222222222222222222222222222222222222222222222222222')
        x = [conv(x).squeeze(3) for conv in self.convs1] #[(N,Co,W), ...]*len(Ks)
        #x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1] #[(N,Co,W), ...]*len(Ks)       
        #print(len(x))
        #print(type(x))
        #print(x[0].size()) 
        #print(x)
        x = x[0]
        #x = utils.squash(x[0], dim=2)
        #print('==================START SQUASH===================================================')
        #print(type(x))
        #print(x)        
        #print('==================END SQUASH=====================================================')
        out_digit_caps = self.digits(x)
        print('1------------------------out_digit_caps')
        print(len(out_digit_caps))
        print(out_digit_caps[0])
        print('2------------------------out_digit_caps')
        #out_digit_caps = out_digit_caps.squeeze(3)
        x = out_digit_caps
        print(x)
        print('==================END digit caps=================================================')
        
        return out_digit_caps
        
        
        #x = [x_t.squeeze(2) for x_t in x]
        #print(type(x))
        
        #x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x] #[(N,Co), ...]*len(Ks)
        #print('333333333333333333333333333333333333333333333333333333333333333333333333333333333')
        #print(x[0].size())
        #x = torch.cat(x, 1)
        #print('444444444444444444444444444444444444444444444444444444444444444444444444444444444')
        #print(x[0].size())
        
        '''
        x1 = self.conv_and_pool(x,self.conv13) #(N,Co)
        x2 = self.conv_and_pool(x,self.conv14) #(N,Co)
        x3 = self.conv_and_pool(x,self.conv15) #(N,Co)
        x = torch.cat((x1, x2, x3), 1) # (N,len(Ks)*Co)
        '''
        #x = self.dropout(x) # (N,len(Ks)*Co)
        #logit = self.fc1(x) # (N,C)
        
        #print('555555555555555555555555555555555555555555555555555555555555555555555555555555555')
        #print(logit.size())
        #print(logit)
        #print('666666666666666666666666666666666666666666666666666666666666666666666666666666666')
        #return logit
