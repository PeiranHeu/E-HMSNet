import torch
import pandas as pd

# 从.pt文件中读取数值
with open('./outdata/trainloss.pt', 'rb') as f:
    trainloss = torch.load(f)
with open('./outdata/testloss.pt', 'rb') as f:
    testloss = torch.load(f)
with open('./outdata/testmacc.pt', 'rb') as f:
    testmacc = torch.load(f)
with open('./outdata/testmiou.pt', 'rb') as f:
    testmiou = torch.load(f)
with open('./outdata/classacc.pt', 'rb') as f:
    cls_acc = torch.load(f)

# 将数值存入DataFrame对象中
df = pd.DataFrame({'train_loss': trainloss,
                   'test_loss': testloss,
                   'test_macc': testmacc,
                   'test_miou': testmiou,
                   '类别1':cls_acc[0],
                   '类别2':cls_acc[1],
                   '类别3':cls_acc[2],
                   '类别4':cls_acc[3],
                   '类别5':cls_acc[4],
                   '类别6':cls_acc[5],
                   '类别7':cls_acc[6],
                   '类别8':cls_acc[7],
                   '类别9':cls_acc[8]})

# 将DataFrame对象保存为Excel表格
df.to_excel('./outdata/output.xlsx', index=False)
