# To train on our own datasets 

### First, train on datasets32 (*BTImages.py line 113-118)
Config the PATH to datasets32 and name checkpoint folder:
```
args.data = './datasets32' # Đường dẫn tới dataset
args.arch = 'M1_datasets32_' + str(n_class) # Tên của thư mục checkpoint
```

### To run
Run the following command: `python BTImages.py`

---

### Second, train on datasets224 (*BTImages.py line 113-118)
Config the PATH to datasets224 and name checkpoint folder:
```
args.data = './datasets224' # Đường dẫn tới dataset
args.arch = 'M1_datasets224_' + str(n_class) # Tên của thư mục checkpoint
```

### To run
Run the following command: `python BTImages.py`

# To train on Cifar10 and Cifar100
### For Cifar10 dataset
- In *M1_CIFAR10.py name the checkpoint folder and set the suitable class
```
pathout = './checkpoints/CIFAR10_M1_Net/' # Tên checkpoint thích hợp
filenameLOG = pathout + '/' + 'FIE02' + '.txt'
if not os.path.exists(pathout):
    os.makedirs(pathout)
# get model

model = M1(num_classes = 10) # 10 classes với cifar10
model = model.to(device)

print(model)
```

### To run
Run the following command: `python M1_CIFAR10.py`

---

### For Cifar100 dataset
- In *M1_CIFAR10.py name the checkpoint folder and set the suitable class
```
pathout = './checkpoints/CIFAR100_M1_Net/' # Tên checkpoint thích hợp
filenameLOG = pathout + '/' + 'FIE02' + '.txt'
if not os.path.exists(pathout):
    os.makedirs(pathout)
# get model

model = M1(num_classes = 100) # 100 classes với cifar100
model = model.to(device)

print(model)
```

### To run
Run the following command: `python M1_CIFAR10.py -d cifar100`