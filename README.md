# MNIST_classification

### 3rd Semester [Deep Learning] assignment

#### LeNet5 모델과 Custum MLP를 이용해서 MNIST Classification 진행


1. dataset.py
  Pytorch DataLoader 구축 후 잘 설정되었는지 확인
<pre>
<code>
    if __name__ == '__main__':
      dir = "./data"
      transform = transforms.Compose([
          transforms.Resize((32, 32)),
          transforms.ToTensor()
      ])
      trainset = MNIST(data_dir=dir, mode='train', transform=transform)
      train_loader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=0)
      for img, label in train_loader:
          print(img)
          print(label)
          import sys;
          sys.exit(0)
</code>
</pre>
