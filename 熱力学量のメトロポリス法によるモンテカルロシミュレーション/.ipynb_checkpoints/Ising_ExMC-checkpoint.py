#交換法

from random import random, randrange
import numpy as np
import matplotlib.pyplot as plt
import copy

class Ising_Exchange_MC_2D:
    def __init__(self, Nx = 20, Ny = 20, n = 100, steps = 60000, average = 5):
        self.Nx = Nx
        self.Ny = Ny
        self.Ntot = Nx*Ny
        self.steps = steps
        self.average = average
        self.n = n  #レプリカ数
        self.B = 0
        self.KBT_array = np.linspace(0.001, 6, n) #温度を0.001から6までn刻みで変動させる
        self.energy_array = np.zeros((n,2*self.steps+1)) #全てのレプリカの各温度におけるエネルギーの履歴
        self.J = 1
        self.replica_list = []
        self.energy_list = []
        self.ETplot = []
        self.CTplot = np.array([])
        self.E_dispersion = []


    def each_Ecalc(self, s, Nx, Ny):
        dum = 0

        #右側のスピンとの相互作用
        for i in range(0, Nx):
            for j in range(0, Ny):

                migi = i + 1

                if migi == Nx:
                    migi = 0
                dum += s[i,j]*s[migi,j]

        #上側のスピンとの相互作用
        for i in range(0, Nx):
            for j in range(0,Ny):

                ue = j+1

                if ue == Ny:
                    ue = 0

                dum += s[i,j]*s[i,ue]

        return dum

    #スピンフリップによる状態更新
    def spin_update(self, replica_list, KBT_array, energy_list):
        for t in range(len(replica_list)):
            s = replica_list[t]
            i = randrange(self.Nx)
            j = randrange(self.Ny)

            s_trial = s.copy()   #deep copyに相当
            s_trial[i,j] = -1*s[i,j]
            if i+1 != self.Nx and j+1 != self.Ny:
                delta_E = 2*s_trial[i,j]*(-1*self.J)*(s[i+1,j]+s[i-1,j]+s[i,j+1]+s[i,j-1])-self.B*(s_trial[i,j]-s[i,j])
            elif i+1 == self.Nx and j+1 != self.Ny:
                delta_E = 2*s_trial[i,j]*(-1*self.J)*(s[0,j]+s[i-1,j]+s[i,j+1]+s[i,j-1])-self.B*(s_trial[i,j]-s[i,j])
            elif i+1 != self.Nx and j+1 == self.Ny:
                delta_E = 2*s_trial[i,j]*(-1*self.J)*(s[i+1,j]+s[i-1,j]+s[i,0]+s[i,j-1])-self.B*(s_trial[i,j]-s[i,j])
            elif i+1 == self.Nx and j+1 == self.Ny:
                delta_E = 2*s_trial[i,j]*(-1*self.J)*(s[0,j]+s[i-1,j]+s[i,0]+s[i,j-1])-self.B*(s_trial[i,j]-s[i,j])

            #メトロポリス法による状態更新
            if delta_E < 0:
                replica_list[t] = s_trial
                energy_list[t] = energy_list[t] + delta_E
            else:
                if random() < np.exp(-delta_E/KBT_array[t]):
                    replica_list[t] = s_trial
                    energy_list[t] = energy_list[t] + delta_E

        return (replica_list, energy_list)

    #レプリカ間の交換1
    def replica_exchange_even(self, replica_list, KBT_array, energy_list):
        for j in range(int(self.n/2)):
            i = 2*j
            delta_ = (KBT_array[i+1]**(-1)-KBT_array[i]**(-1))*(energy_list[i] - energy_list[i+1])

            if delta_ < 0:
                x = replica_list[i]
                replica_list[i] = replica_list[i+1]
                replica_list[i+1] = x

                y = energy_list[i]
                energy_list[i] = energy_list[i+1]
                energy_list[i+1] = y
            else:
                if random() < np.exp(-delta_):
                    x = replica_list[i]
                    replica_list[i] = replica_list[i+1]
                    replica_list[i+1] = x

                    y = energy_list[i]
                    energy_list[i] = energy_list[i+1]
                    energy_list[i+1] = y

        return  (replica_list, energy_list)

    #レプリカ間の交換2
    def replica_exchange_odd(self, replica_list, KBT_array, energy_list):
        for j in range(int(self.n/2)-1):
            i = 2*j + 1
            delta_ = (KBT_array[i+1]**(-1)-KBT_array[i]**(-1))*(energy_list[i] - energy_list[i+1])

            if delta_ < 0:
                x = replica_list[i]
                replica_list[i] = replica_list[i+1]
                replica_list[i+1] = x

                y = energy_list[i]
                energy_list[i] = energy_list[i+1]
                energy_list[i+1] = y
            else:
                if random() < np.exp(-delta_):
                    x = replica_list[i]
                    replica_list[i] = replica_list[i+1]
                    replica_list[i+1] = x

                    y = energy_list[i]
                    energy_list[i] = energy_list[i+1]
                    energy_list[i+1] = y

        return  (replica_list, energy_list)

    def Initial_rand(self, Nx, Ny):
        s = np.random.randint(0,2,(Nx,Ny))
        for i in range(Nx):
            for j in range(Ny):
                if s[i,j] == 0:
                    s[i,j] = -1
        return s

    def main(self):
        avsteps = int(self.steps * 2/self.average)
        
        #ユーザーによるインスタンスを変更された場合の修正
        if self.steps != 60000 or self.n != 100:
            self.energy_array = np.zeros((self.n,2*self.steps+1))   #self.stepsか、self.nがデフォルト値と違った時にはenergy_arrayの形を変更
            
            if self.n != 100:
                self.KBT_array = np.linspace(0.001, 6, self.n) #self.nがデフォルト値と違った場合のみKBT_arrayの形も変更

        for i in range(self.n):
            self.replica_list.append(self.Initial_rand(self.Nx, self.Ny))

        #初期エネルギー計算
        for i in range(len(self.replica_list)):
            self.energy_list.append(-self.J*self.each_Ecalc(self.replica_list[i], self.Nx, self.Ny))

        self.energy_array[:,0] = self.energy_list

        #Exchange MC
        for j in range(self.steps):
            if j == int(self.steps/3):
                print("33%")
            elif j == int(2*self.steps/3):
                print("66%")

            i = 2*j
            if j % 2 == 1:
                #まずスピンフリップ
                self.replica_list, self.energy_list = self.spin_update(self.replica_list, self.KBT_array, self.energy_list)
                self.energy_array[:,i+1] = self.energy_list

                #次に温度交換
                self.replica_list, self.energy_list = self.replica_exchange_odd(self.replica_list, self.KBT_array, self.energy_list)
                self.energy_array[:,i+2] = self.energy_list
            if j % 2 == 0:
                #まずスピンフリップ
                self.replica_list, self.energy_list = self.spin_update(self.replica_list, self.KBT_array, self.energy_list)
                self.energy_array[:,i+1] = self.energy_list

                #次に温度交換
                self.replica_list, self.energy_list = self.replica_exchange_even(self.replica_list, self.KBT_array, self.energy_list)
                self.energy_array[:,i+2] = self.energy_list

        for i in range(self.n):
            self.ETplot.append(np.sum(self.energy_array[i, -2*avsteps:])/(2*avsteps)/self.Ntot)

        plt.plot(self.KBT_array, self.ETplot)
        plt.ylabel("Energy per spin")
        plt.xlabel("$k_{B}T$")
        plt.show()

        E2_expectation = []
        for i in range(self.n):
            E2_expectation.append(np.sum((self.energy_array[i,-2*avsteps:]/self.Ntot)**2)/(2*avsteps))
        E_expectation = np.array(self.ETplot)
        self.E_dispersion = (E2_expectation - E_expectation**2)**0.5
        self.CTplot = self.E_dispersion /(self.KBT_array**2)
        plt.plot(self.KBT_array,self.CTplot)
        plt.ylabel("Heat capacity per spin")
        plt.xlabel("$k_{B}T$")
        plt.show()


        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.matshow(self.replica_list[0])

if __name__ == "__main__":     #importされた時に実行されない
    tmp = Ising_Exchange_MC_2D()                   #Ising2Dクラスのオブジェクトを生成
    tmp.main()
