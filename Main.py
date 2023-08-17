from Sims import *
from Settings import *

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

class FL_Proc:
    def __init__(self, configs):
        self.DataName = configs["dname"]
        self.ModelName = configs["mname"]
        self.NClients = configs["nclients"] 
        self.PClients = configs["pclients"]  
        self.IsIID = configs["isIID"]
        self.Alpha = configs["alpha"]
        self.Aug = configs["aug"]
        self.MaxIter = configs["iters"]
        self.LR = configs["learning_rate"]
        self.Normal = configs["normal"]
        self.Algo = configs["algorithm"]  
        self.Optmzer = configs["optimizer"]  
        self.FixLR = configs["fixlr"]
        self.WDecay = configs["wdecay"]
        self.DShuffle = configs["data_shuffle"]
        self.BatchSize = configs["batch_size"]
        self.Epoch = configs["epoch"]
        self.GlobalLR = configs["global_lr"]
        self.UseCP = configs["critical"]
        self.FIM = configs["fim"]
        self.CThresh = configs["CThresh"]
        self.SOM = configs["server_optim"]
        self.Ratios = {}
        self.RandNum = configs["rand_num"]
        self.CPR = configs["compression"]
        self.GModel = load_Model(self.ModelName, self.DataName)
        self.Server = None
        self.Clients = {}
        self.ClientLoaders = None
        self.TrainLoader = None
        self.TestLoader = None
        self.LogStep = configs["log_step"]

        self.updateIDs = []
        for i in range(self.PClients):
            self.updateIDs.append(i)
        self.Detection = CPCheck(self.NClients, self.PClients, alpha=self.Alpha, threshold=self.CThresh, dataname=self.DataName)
        self.Selection = RandomGet(self.NClients)
        self.TrainRound = 0
    
    def get_train_datas(self):
        self.ClientLoaders, self.TrainLoader, self.TestLoader, Stat = get_loaders(self.DataName, self.NClients, self.IsIID,self.Alpha, self.Aug, False, False,self.Normal, self.DShuffle, self.BatchSize)

    def logging(self):
        teloss, teaccu = self.Server.evaluate(self.TestLoader)

    def main(self):
        self.get_train_datas()
        self.Server = Server_Sim(self.TrainLoader, self.GModel, self.LR, self.WDecay, self.FixLR, self.DataName)
        NumCount = 0
        for c in range(self.NClients):
            self.Clients[c] = Client_Sim(self.ClientLoaders[c], self.GModel, self.LR, self.WDecay, self.Epoch,self.FixLR, self.Optmzer)
            self.Selection.register_client(c)

        IDs = []
        for c in range(self.NClients):
            IDs.append(c)
        
        NumPartens = self.PClients
        DetStep = 2
        LStep = 0
        self.logging()
        CLP = 1
        
        for It in range(self.MaxIter):
            self.TrainRound = It + 1
            
            if (It + 1) % DetStep == 0:
                CLP = 0
                GetNorms = []
                for ky in self.updateIDs:
                    GetNorms.append(self.Clients[ky].gradnorm)

                if self.UseCP:
                    self.Detection.recvInfo(GetNorms)
                    NumPartens,CLP = self.Detection.WinCheck(len(self.updateIDs))
                    
            if self.UseCP == False and self.RandNum == True:
                P1 = 0.5
                P2 = 0.5 
                prob = np.random.rand()
                if prob <= P1:
                    NumPartens = self.PClients + int(self.PClients)
                if prob >= P2:
                    NumPartens = self.PClients - int(self.PClients / 2)

            updateIDs = self.Selection.select_participant(NumPartens)
            
            GlobalParms = self.Server.getParas()
            LrNow = self.Server.getLR()
            TransLens = []
            TransParas = []
            TransVecs = []
            for ky in updateIDs:
                if self.GlobalLR:
                    self.Clients[ky].updateLR(LrNow)
                self.Clients[ky].updateParas(GlobalParms)
                Nvec = self.Clients[ky].selftrain()
                ParasNow = self.Clients[ky].getParas()
                if self.CPR and CLP == 1:
                    ParasNow = self.Clients[ky].getKParas()
                LenNow = self.Clients[ky].DLen
                TransLens.append(LenNow)
                TransParas.append(ParasNow)
                TransVecs.append(Nvec)
            
            TauEffs = []
            SLen = np.sum(TransLens)
            for k in range(len(TransLens)):
                TauEffs.append(TransLens[k] / SLen * TransVecs[k])
            TauEff = np.sum(TauEffs)
            
            for k in range(len(TransLens)):
                GPara = TransParas[k]
                GLen = TransLens[k] / SLen
                GNvec = TauEff / TransVecs[k]
                self.Server.recvInfo(GPara, GLen, GNvec)
            self.Server.aggParas(self.SOM)
            
            if self.Optmzer == "VRL":
                GlobalParms = self.Server.getParas()
                for ky in updateIDs:
                    self.Clients[ky].updateParas(GlobalParms)
                    LSteps = self.Clients[ky].local_steps
                    self.Clients[ky].optimizer.update_delta(LSteps)
            self.updateIDs = updateIDs
            
            # Logging
            if (It + 1) % self.LogStep == 0:
                self.logging()


if __name__ == '__main__':
    Configs = {}
    Configs['dname'] = "cifar10"
    Configs["mname"] = "alex"
    Configs["algorithm"] = "CriticalFL"
    Configs['nclients'] = 128
    Configs['pclients'] = 16
    Configs["learning_rate"] = 0.01
    Configs["critical"] = True
    Configs["compression"] = True
    Configs["normal"] = True
    Configs["fixlr"] = False
    Configs["global_lr"] = True
    Configs["aug"] = False
    Configs["data_shuffle"] = True
    Configs["fim"] = False
    Configs['isIID'] = False
    Configs["rand_num"] = False
    Configs["epoch"] = 3
    Configs["batch_size"] = 16
    Configs["iters"] = 200
    Configs["log_step"] = 1
    Configs["wdecay"] = 1e-5
    Configs["CThresh"] = 0.01
    Configs["optimizer"] = "SGD" # "VRL","FedProx","FedNova"
    Configs["server_optim"] = None # "Adam","Adag","Yogi"
    Configs["alpha"] = 0.5
    
    FLSim = FL_Proc(Configs)
    FLSim.main()
                



