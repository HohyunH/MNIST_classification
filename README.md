# MNIST_classification

### 3rd Semester [Deep Learning] assignment

#### LeNet5 모델과 Custom MLP를 이용해서 MNIST Classification 진행
#### LeNet5 모델과 Custom MLP 모델의 파라미터 갯수를 최대한 같게 구성한다.


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

2. model.py

- LeNet 5은 모든 convolutional layer에서 5x5 사이즈의 kernel을 가지고 학습이 진행된다.

![image](https://user-images.githubusercontent.com/46701548/138217219-0f408d42-6add-4ecd-8103-6f6cd3255f5c.png)

- Pytorch의 torchsummary 라이브러리를 이용해서 LeNet 5의 파라미터 갯수를 확인했다.

총 61,706개를 가지고있다.

![image](https://user-images.githubusercontent.com/46701548/138217415-defea63b-87a6-4f8c-813d-e051177c6aa7.png)

- Custom MLP

![image](https://user-images.githubusercontent.com/46701548/138217603-5267978c-7d01-4b8e-8f52-42c4ed05dbca.png)

- 두번째 hidden layer의 노드의 갯수를 59로 설정한다.

Pytorch의 torchsummary 라이브러리를 이용해서 Custom MLP의 파라미터 갯수를 확인했다.

총 61,075개를 가지고있다.

기존의 LeNet 5 네트워크와 약 600개 정도밖에 차이가 안나는것을 확인할 수 있다.

![image](https://user-images.githubusercontent.com/46701548/138217667-38bfcad9-0ef8-4c11-a8eb-7943416d6d39.png)


#### Performance in training data

- LeNet5

![image](https://user-images.githubusercontent.com/46701548/138217778-0a998edc-64b7-4520-91c6-8d31dfc3acc1.png)![image](https://user-images.githubusercontent.com/46701548/138217789-082ee70e-3276-4f38-a47c-a43182f03156.png)

- Custom MLP

![image](https://user-images.githubusercontent.com/46701548/138217815-7a124fb0-7d34-4f40-957e-48d4d7cef49d.png)![image](https://user-images.githubusercontent.com/46701548/138217825-7a57a419-c1b1-473f-b18c-29d9662d6cb1.png)


