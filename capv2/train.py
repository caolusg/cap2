import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import utils
from torch.autograd import Variable


def train(train_iter, dev_iter, model, args):
    if args.cuda:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    steps = 0
    model.train()
    epc = 0
    print(args.epochs)
    epoch_tot_acc = 0
    for epoch in range(1, args.epochs+1):
        epc += 1
        c = 0
        for batch in train_iter:
            c+=1
            #print(c)
            #print('cccccccccccccccccccccccccccccccccccccccccccccccccccccccccc')
            #print (batch.text)           
            #print('kkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk')            
            #print(batch.text.size())
            #print('ddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd')            
            feature, target = batch.text, batch.label            
            feature.data.t_(), target.data.sub_(1)  # batch first, index align
            if args.cuda:
                feature, target = feature.cuda(), target.cuda()
            batch_size = len(feature)
            
            #print('-------------------target-----------------')
            #print(type(target))
            #print(target)
            target = target.cpu()
            d = target.data.numpy()
            d_t = torch.from_numpy(d)
            labels = d_t
            #print('-----------------target one hot-------------')
            target_one_hot = utils.one_hot_encode(d_t, length=2)
            assert target_one_hot.size() == torch.Size([batch_size, 2])
            #print(type(target_one_hot))            
            #print(target_one_hot)
            optimizer.zero_grad()
            logit = model(feature)
            
            out_digit_caps = logit
            #print('out_digit_caps')
            #print(out_digit_caps)
            target = Variable(target_one_hot)
            margin_loss = utils.margin_loss(out_digit_caps, target)
            loss = margin_loss
            #print(type(loss))
            #print('margin loss: ', margin_loss)
            loss = torch.mean(margin_loss, 0)
            print('mean loss: ', loss.data)
            #print(type(loss))
            
            
            #print('logit vector', logit.size(),logit)
            #print('target vector', target.size(),target)           
            #loss = F.cross_entropy(logit, target)
            #print('loss')
            #print(loss)
            loss.backward()
            optimizer.step()            
            
            acc = utils.accuracy(out_digit_caps, labels, False)
            epoch_tot_acc += acc
            epoch_avg_acc = epoch_tot_acc/epc
            print('epc: ', epc)
            print('c: ', c)
            print('acc: ', acc)
            print('epoch_avg_acc', epoch_avg_acc)
            
            '''
            steps += 1
            if steps % args.log_interval == 0:
                corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                accuracy = 100.0 * corrects/batch.batch_size
                sys.stdout.write(
                    '\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{}) epoch: {}'.format(steps, 
                                                                             loss.data[0], 
                                                                             accuracy,
                                                                             corrects,
                                                                             batch.batch_size,
                                                                             epc))
            if steps % args.test_interval == 0:
                eval(dev_iter, model, args)
            if steps % args.save_interval == 0:
                if not os.path.isdir(args.save_dir): os.makedirs(args.save_dir)
                save_prefix = os.path.join(args.save_dir, 'snapshot')
                save_path = '{}_steps{}.pt'.format(save_prefix, steps)
                torch.save(model, save_path)
            '''

def eval(data_iter, model, args):
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in data_iter:
        feature, target = batch.text, batch.label
        feature.data.t_(), target.data.sub_(1)  # batch first, index align
        if args.cuda:
            feature, target = feature.cuda(), target.cuda()

        logit = model(feature)
        loss = F.cross_entropy(logit, target, size_average=False)

        avg_loss += loss.data[0]
        corrects += (torch.max(logit, 1)
                     [1].view(target.size()).data == target.data).sum()

    size = len(data_iter.dataset)
    avg_loss = avg_loss/size
    accuracy = 100.0 * corrects/size
    model.train()
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss, 
                                                                       accuracy, 
                                                                       corrects, 
                                                                       size))


def predict(text, model, text_field, label_feild, cuda_flag):
    assert isinstance(text, str)
    model.eval()
    # text = text_field.tokenize(text)
    text = text_field.preprocess(text)
    text = [[text_field.vocab.stoi[x] for x in text]]
    x = text_field.tensor_type(text)
    x = autograd.Variable(x, volatile=True)
    if cuda_flag:
        x =x.cuda()
    print(x)
    output = model(x)
    _, predicted = torch.max(output, 1)
    return label_feild.vocab.itos[predicted.data[0][0]+1]
