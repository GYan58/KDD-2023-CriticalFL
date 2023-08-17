from Settings import *
from Models import AlexNet as Ax
from Models import ResNet as Re
from Models import VGG as Vg
from Models import LSTM as Lm


def load_Model(Type, Name):
    Model = None
    if Type == "alex":
        if Name == "fmnist":
            Model = Ax.alex_fmnist()

        if Name == "cifar10":
            Model = Ax.alex_cifar10()
                
    if Type == "vgg":         
        if Name == "fmnist":
            Model = Vg.vgg_fmnist()

        if Name == "cifar10":
            Model = Vg.vgg_cifar10()

    if Type == "resnet":
        if Name == "cifar100":
            Model = Re.resnet_cifar100()
                
    if Type == "lstm":
        if Name == "shake":
            Model = Lm.CharLSTM()

    return Model


class RandomGet:
    def __init__(self, Nclients=0):
        self.totalArms = OrderedDict()
        self.Clients = Nclients

    def register_client(self, clientId):
        if clientId not in self.totalArms:
            self.totalArms[clientId] = {}
            self.totalArms[clientId]['status'] = True
    
    def updateStatus(self, Id, Sta):
        self.totalArms[Id]['status'] = Sta

    def select_participant(self, num_of_clients):
        viable_clients = [x for x in self.totalArms.keys() if self.totalArms[x]['status']]
        return self.getTopK(num_of_clients, viable_clients)

    def getTopK(self, numOfSamples, feasible_clients):
        IDs = []
        for i in range(len(feasible_clients)):
            IDs.append(i)
        rd.shuffle(IDs)
        pickedClients = IDs[:numOfSamples]
        return pickedClients


class CPCheck:
    def __init__(self, clients, partens, window=10, alpha=0.5, threshold=0.01, dataname="cifar10"):
        self.Win = window
        self.Norms = [0,0,0,0,0,0,0,0,0,0,0,0]
        self.Round = 0
        self.Clients = clients
        self.Threshold = threshold
        Pert = 0.75
        self.CMaxLim = int(clients * Pert)
        self.CMinLim = int(partens * 1)
        self.Achieve = False
        self.MLim = 5

    def recvInfo(self,Norms):
        self.Round += 1
        AvgNorm = np.mean(Norms)
        self.Norms.append(AvgNorm)

    def WinCheck(self,CNum):
        if CNum == self.CMaxLim:
            self.Achieve = True
        OldNorm = max([np.mean(self.Norms[-self.Win-1:-1]),0.0000001])
        NewNorm = np.mean(self.Norms[-self.Win:])
        
        Is = 0
        if (NewNorm - OldNorm) / OldNorm > self.Threshold or self.Round <= self.MLim:
            Is = 1

        if Is == 1 and self.Achieve == False:
            CNum = min(self.CMaxLim, CNum * 2)
        
        if Is == 0:
            Reduce = max(int(CNum / 2),1)
            CNum = max(self.CMinLim, CNum - Reduce)
        return CNum, Is


from collections import defaultdict
import json

ALL_LETTERS = "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"
NUM_LETTERS = len(ALL_LETTERS)


def _one_hot(index, size):
    vec = [0 for _ in range(size)]
    vec[int(index)] = 1
    return vec


def letter_to_vec(letter):
    index = ALL_LETTERS.find(letter)
    return index


def word_to_indices(word):
    indices = []
    for c in word:
        indices.append(ALL_LETTERS.find(c))
    return indices


def batch_data(data, batch_size, seed):
    data_x = data['x']
    data_y = data['y']
    np.random.seed(seed)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)

    for i in range(0, len(data_x), batch_size):
        batched_x = data_x[i:i + batch_size]
        batched_y = data_y[i:i + batch_size]
        yield (batched_x, batched_y)


def read_dir(data_dir):
    clients = []
    groups = []
    data = defaultdict(lambda: None)

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        data.update(cdata['user_data'])
    
    clients = list(sorted(data.keys()))
    return clients, groups, data



def read_data(train_data_dir, test_data_dir):
    train_clients, train_groups, train_data = read_dir(train_data_dir)
    test_clients, test_groups, test_data = read_dir(test_data_dir)

    assert train_clients == test_clients
    assert train_groups == test_groups

    return train_clients, train_groups, train_data, test_data


class ShakeSpeare(Dataset):
    def __init__(self, train=True):
        super(ShakeSpeare, self).__init__()
        train_clients, train_groups, train_data_temp, test_data_temp = read_data(ShakeRoot + "train", ShakeRoot + "test")
        self.train = train
        if self.train:
            self.dic_users = {}
            train_data_x = []
            train_data_y = []
            for i in range(len(train_clients)):
                self.dic_users[i] = set()
                l = len(train_data_x)
                cur_x = train_data_temp[train_clients[i]]['x']
                cur_y = train_data_temp[train_clients[i]]['y']
                Length = len(cur_x)
                for j in range(Length):
                    self.dic_users[i].add(j + l)
                    train_data_x.append(cur_x[j])
                    train_data_y.append(cur_y[j])
            self.data = train_data_x
            self.label = train_data_y
        else:
            test_data_x = []
            test_data_y = []
            for i in range(len(train_clients)):
                cur_x = test_data_temp[train_clients[i]]['x']
                cur_y = test_data_temp[train_clients[i]]['y']
                Length = len(cur_x)
                for j in range(Length):
                    test_data_x.append(cur_x[j])
                    test_data_y.append(cur_y[j])
            
            self.data = test_data_x
            self.label = test_data_y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence, target = self.data[index], self.label[index]
        indices = word_to_indices(sentence)
        target = letter_to_vec(target)
        indices = torch.LongTensor(np.array(indices))
        return indices, target

    def get_client_dic(self):
        if self.train:
            return self.dic_users
        else:
            exit("The test dataset do not have dic_users!")

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

def get_sloaders(n_clients,dshuffle,batchsize):
    train_loader = ShakeSpeare(train=True)
    test_loader = ShakeSpeare(train=False)
    dict_users = train_loader.get_client_dic()
    dicts = []
    for ky in dict_users.keys():
        dicts += list(dict_users[ky])

    ELen = int(len(dicts) / n_clients)
    client_loaders = []
    for i in range(n_clients - 1):
        s_index = i * ELen
        e_index = (i + 1) * ELen
        new_dict = dicts[s_index:e_index]
        cloader = DataLoader(DatasetSplit(train_loader, new_dict), batch_size=batchsize, shuffle=dshuffle)
        client_loaders.append(cloader)

    cloader = DataLoader(DatasetSplit(train_loader, dicts[(n_clients - 1) * ELen:]), batch_size=batchsize, shuffle=dshuffle)
    client_loaders.append(cloader)
    
    train_loader = DataLoader(train_loader,batch_size=1000)
    test_loader = DataLoader(test_loader,batch_size=1000)

    return client_loaders, train_loader, test_loader, None


def get_shakeloader(n_clients,dshuffle,batchsize,partitions):
    train_loader = ShakeSpeare(train=True)
    test_loader = ShakeSpeare(train=False)
    
    client_loaders = []
    for i in range(n_clients):
        new_dict = partitions[i]
        cloader = DataLoader(DatasetSplit(train_loader, new_dict), batch_size=batchsize, shuffle=dshuffle)
        client_loaders.append(cloader)
    
    train_loader = DataLoader(train_loader,batch_size=1000)
    test_loader = DataLoader(test_loader,batch_size=1000)

    return client_loaders, train_loader, test_loader, None


def get_cifar10():
    data_train = torchvision.datasets.CIFAR10(root="./data", train=True, download=True)
    data_test = torchvision.datasets.CIFAR10(root="./data", train=False, download=True)
    TrainX, TrainY = data_train.data.transpose((0, 3, 1, 2)), np.array(data_train.targets)
    TestX, TestY = data_test.data.transpose((0, 3, 1, 2)), np.array(data_test.targets)

    return TrainX, TrainY, TestX, TestY


def get_cifar100():
    data_train = torchvision.datasets.CIFAR100(root="./data", train=True, download=True)
    data_test = torchvision.datasets.CIFAR100(root="./data", train=False, download=True)
    TrainX, TrainY = data_train.data.transpose((0, 3, 1, 2)), np.array(data_train.targets)
    TestX, TestY = data_test.data.transpose((0, 3, 1, 2)), np.array(data_test.targets)

    return TrainX, TrainY, TestX, TestY
    

def get_mnist():
    data_train = torchvision.datasets.MNIST(root="./data", train=True, download=True)
    data_test = torchvision.datasets.MNIST(root="./data", train=False, download=True)
    TrainX, TrainY = data_train.train_data.numpy().reshape(-1, 1, 28, 28) / 255, np.array(data_train.targets)
    TestX, TestY = data_test.test_data.numpy().reshape(-1, 1, 28, 28) / 255, np.array(data_test.targets)

    return TrainX, TrainY, TestX, TestY
    
    
def get_fmnist():
    data_train = torchvision.datasets.FashionMNIST(root="./data", train=True, download=True)
    data_test = torchvision.datasets.FashionMNIST(root="./data", train=False, download=True)
    TrainX, TrainY = data_train.train_data.numpy().reshape(-1, 1, 28, 28) / 255, np.array(data_train.targets)
    TestX, TestY = data_test.test_data.numpy().reshape(-1, 1, 28, 28) / 255, np.array(data_test.targets)

    return TrainX, TrainY, TestX, TestY


def get_image():
    train_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4802, 0.4481, 0.3975],std=[0.2302, 0.2265, 0.2262])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4802, 0.4481, 0.3975],std=[0.2302, 0.2265, 0.2262])
    ])

    TrainData = ImageFolder('', subdir='train',transform=train_transform)
    TrainLoader = data.DataLoader(TrainData, batch_size=1024, shuffle=False, num_workers=0)

    TestData = ImageFolder('', subdir='test',transform=test_transform)
    TestLoader = data.DataLoader(TestData, batch_size=1024, shuffle=False, num_workers=0)

    TrainX, TrainY, TestX, TestY = [],[],[],[]

    for bid, (inputs,outputs) in enumerate(TrainLoader):
        TrainX += list(inputs.cpu().detach().numpy())
        TrainY += list(outputs.cpu().detach().numpy())

    for bid, (inputs,outputs) in enumerate(TestLoader):
        TestX += list(inputs.cpu().detach().numpy())
        TestY += list(outputs.cpu().detach().numpy())

    return np.array(TrainX), np.array(TrainY), np.array(TestX), np.array(TestY)


class Addblur(object):

    def __init__(self, blur="Gaussian"):
        self.blur = blur

    def __call__(self, img):
        if self.blur == "normal":
            img = img.filter(ImageFilter.BLUR)
            return img
        if self.blur == "Gaussian":
            img = img.filter(ImageFilter.GaussianBlur)
            return img
        if self.blur == "mean":
            img = img.filter(ImageFilter.BoxBlur)
            return img


class AddNoise(object):
    def __init__(self, noise="Gaussian"):
        self.noise = noise
        self.density = 0.8
        self.mean = 0.0
        self.variance = 10.0
        self.amplitude = 10.0

    def __call__(self, img):

        img = np.array(img) 
        h, w, c = img.shape

        if self.noise == "pepper":
            Nd = self.density
            Sd = 1 - Nd
            mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[Nd / 2.0, Nd / 2.0, Sd])  
            mask = np.repeat(mask, c, axis=2)  
            img[mask == 2] = 0 
            img[mask == 1] = 255 

        if self.noise == "Gaussian":
            N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance, size=(h, w, 1))
            N = np.repeat(N, c, axis=2)
            img = N + img
            img[img > 255] = 255 

        img = Image.fromarray(img.astype('uint8')).convert('RGB')

        return img


FileNameEnd = ('.jpeg', '.JPEG', '.tif', '.jpg', '.png', '.bmp')
class ImageFolder(data.Dataset):
    def __init__(self, root, subdir='train', transform=None):
        super(ImageFolder,self).__init__()

        self.transform = transform 
        self.image = []    
        train_dir = join(root, 'train') 
        self.class_names = sorted(os.listdir(train_dir))

        self.names2index = {v: k for k, v in enumerate(self.class_names)}

        if subdir == 'train':
            for label in self.class_names:
                d = join(root, subdir, label)
                for directory, _, names in os.walk(d):
                    for name in names:
                        filename = join(directory, name)
                        if filename.endswith(FileNameEnd):
                            self.image.append((filename, self.names2index[label]))

        if subdir == 'test':
            test_dir = join(root, 'val')
            with open(join(test_dir, 'val_annotations.txt'), 'r') as f:
                infos = f.read().strip().split('\n')
                infos = [info.strip().split('\t')[:2] for info in infos]

                self.image = [(join(test_dir, 'images', info[0]), self.names2index[info[1]]) for info in infos]

    def __getitem__(self, item):
        path, label = self.image[item]
        with open(path, 'rb') as f: 
            img = Image.open(f).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.image)


class split_image_data(object):
    def __init__(self, dataset, labels, workers, balance=True, isIID=True, alpha=0.0, limit=False):
        seed = 1
        Perts = []
        self.Dataset = dataset 
        self.Labels = labels 
        self.workers = workers
        self.DirichRVs = []
        self.DirichCount = 0

        if alpha == 0 and not isIID:
            print("* Split Error...")

        if balance:
            for i in range(workers):
                Perts.append(1/workers)
        else:
            Sum = workers * (workers + 1) / 2
            SProb = 0
            for i in range(workers - 1):
                prob = int((i + 1) / Sum * 10000) / 10000
                SProb += prob
                Perts.append(prob)

            Left = 1 - SProb
            Perts.append(Left)
            bfrac = 0.1 / workers
            for i in range(len(Perts)):
                Perts[i] = Perts[i] * 0.9 + bfrac

        if not isIID and alpha > 0:
            self.partitions = self.__getDirichlet__(labels, Perts, seed, alpha, limit)
        if isIID:
            self.partitions = []
            rng = rd.Random()
            rng.seed(seed)
            data_len = len(labels)
            indexes = [x for x in range(0, data_len)]
            for frac in Perts:
                part_len = int(frac * data_len)
                self.partitions.append(indexes[0:part_len])
                indexes = indexes[part_len:]

    def __getDirichlet__(self, data, psizes, seed, alpha, limit):
        n_nets = len(psizes)
        K = len(np.unique(self.Labels))
        labelList = np.array(data)
        min_size = 0
        N = len(labelList)
        np.random.seed(seed)

        net_dataidx_map = {}
        idx_batch = []
        while min_size < K:
            idx_batch = [[] for _ in range(n_nets)]
            for k in range(K):
                idx_k = np.where(labelList == k)[0]
                proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
                proportions = np.array([ p *(len(idx_j) < N / n_nets) for p ,idx_j in zip(proportions ,idx_batch)]) 
                proportions = np.array(proportions)
                proportions = proportions /proportions.sum()
                proportions = (np.cumsum(proportions ) *len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j ,idx in zip(idx_batch ,np.split(idx_k ,proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

        net_cls_counts = {}

        for net_i, dataidx in net_dataidx_map.items():
            unq, unq_cnt = np.unique(labelList[dataidx], return_counts=True)
            tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
            net_cls_counts[net_i] = tmp

        local_sizes = []
        for i in range(n_nets):
            local_sizes.append(len(net_dataidx_map[i]))
        local_sizes = np.array(local_sizes)
        weights = local_sizes /np.sum(local_sizes)

        return idx_batch

    def get_splits(self):
        clients_split = []
        for i in range(self.workers):
            IDx = self.partitions[i]
            Ls = []
            Ds = []
            for ky in IDx:
                Ls.append(self.Labels[ky])
                Ds.append(self.Dataset[ky])

            Xs = []
            Ys = []
            Datas = {}
            for k in range(len(Ls)):
                L = Ls[k]
                D = Ds[k]
                if L not in Datas.keys():
                    Datas[L] = [D]
                else:
                    Datas[L].append(D)

            Kys = list(Datas.keys())
            Kl = len(Kys)
            CT = 0
            k = 0
            while CT < len(Ls):
                Id = Kys[k % Kl]
                k += 1
                if len(Datas[Id]) > 0:
                    Xs.append(Datas[Id][0])
                    Ys.append(Id)
                    Datas[Id] = Datas[Id][1:]
                    CT += 1

            clients_split += [(np.array(Xs), np.array(Ys))]
            del Xs, Ys
            gc.collect()

        return clients_split


def get_train_data_transforms(name, aug=False, blur=False, noise=False, normal=False):
    Ts = [transforms.ToPILImage()]
    if name == "mnist" or name == "fmnist":
        Ts.append(transforms.Resize((32, 32)))

    if aug == True and name == "cifar10":
        Ts.append(transforms.RandomCrop(32, padding=4))
        Ts.append(transforms.RandomHorizontalFlip())
        
    if aug == True and name == "cifar100":
        Ts.append(transforms.RandomCrop(32, padding=4))
        Ts.append(transforms.RandomHorizontalFlip())

    if blur == True:
        Ts.append(Addblur())

    if noise == True:
        Ts.append(AddNoise())

    Ts.append(transforms.ToTensor())

    if normal == True:
        if name == "mnist":
            Ts.append(transforms.Normalize((0.06078,), (0.1957,)))
        if name == "fmnist":
            Ts.append(transforms.Normalize((0.1307,), (0.3081,)))
        if name == "cifar10":
            Ts.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
        if name == "cifar100":
            Ts.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

    return transforms.Compose(Ts)


def get_test_data_transforms(name, normal=False):
    transforms_eval_F = {
        'mnist': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ]),
        'fmnist': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ]),
        'cifar10': transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ]),
        'cifar100': transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ]),
    }

    transforms_eval_T = {
        'mnist': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.06078,), (0.1957,))
        ]),
        'fmnist': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]),
        'cifar10': transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]),
        'cifar100': transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
    }

    if normal == False:
        return transforms_eval_F[name]
    else:
        return transforms_eval_T[name]


class CustomImageDataset(Dataset):
    def __init__(self, inputs, labels, transforms=None):
        self.inputs = torch.Tensor(inputs)
        self.labels = torch.Tensor(labels).long()
        self.transforms = transforms

    def __getitem__(self, index):
        img, label = self.inputs[index], self.labels[index]

        if self.transforms is not None:
            img = self.transforms(img)

        return (img, label)

    def __len__(self):
        return self.inputs.shape[0]


def get_loaders(Name, n_clients=10, isnoniid=False, alpha=0.0 ,aug=False, noise=False, blur=False, normal=False, dshuffle=True, batchsize=128):
    TrainX, TrainY, TestX, TestY = [], [], [], []
    LimitB = True
    if Name == "mnist":
        TrainX, TrainY, TestX, TestY = get_mnist()
    if Name == "fmnist":
        TrainX, TrainY, TestX, TestY = get_fmnist()
    if Name == "cifar10":
        TrainX, TrainY, TestX, TestY = get_cifar10()
    if Name == "cifar100":
        TrainX, TrainY, TestX, TestY = get_cifar100()
        LimitB = False
    if Name == "image":
        TrainX, TrainY, TestX, TestY = get_image()
    if Name == "shake":
        cloader, trloader, teloader, _ = get_sloaders(n_clients, False, batchsize)
        for batch_id, (inputs, targets) in enumerate(trloader):
            TrainX += list(inputs.detach().numpy())
            TrainY += list(targets.detach().numpy())
        for batch_id, (inputs, targets) in enumerate(teloader):
            TestX += list(inputs.detach().numpy())
            TestY += list(targets.detach().numpy())
        
        TrainY = np.array(TrainY)
        TestY = np.array(TestY)
        TrainX = np.array(TrainX)
        TestX = np.array(TestX)
        
        SPL = split_image_data(TrainX, TrainY, n_clients, True, isnoniid, alpha, LimitB)
        return get_shakeloader(n_clients,dshuffle,batchsize,SPL.partitions)
    
    transforms_train = None
    transforms_eval = None
    if Name != "image" and Name != "shake":
        transforms_train = get_train_data_transforms(Name, aug, blur, noise, normal)
        transforms_eval = get_test_data_transforms(Name, normal)

    splits = split_image_data(TrainX, TrainY, n_clients, True, isnoniid, alpha, LimitB).get_splits()

    client_loaders = []
    SumL = 0
    for x, y in splits:
        SumL += len(x)
        client_loaders.append(torch.utils.data.DataLoader(CustomImageDataset(x, y, transforms_train), batch_size=batchsize,shuffle=dshuffle))
    
    train_loader = torch.utils.data.DataLoader(CustomImageDataset(TrainX, TrainY, transforms_eval), batch_size=1000, shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(CustomImageDataset(TestX, TestY, transforms_eval), batch_size=1000, shuffle=False, num_workers=2)

    stats = {"split": [x.shape[0] for x, y in splits]}
    
    return client_loaders, train_loader, test_loader, stats


def minusParas(P1,P2,Fac=1):
    Res = cp.deepcopy(P1)
    for ky in P2.keys():
       Res[ky] = P1[ky] - P2[ky] * Fac
    return Res



