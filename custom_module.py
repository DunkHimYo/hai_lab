import numpy as np
from sklearn.preprocessing import StandardScaler,RobustScaler, MinMaxScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from itertools import chain


class DataWrangling():

    def __init__(self,data,label):

        #%% data attribute
        self.data=np.array(data) if not isinstance(data,np.ndarray) else data
        self.label = np.array(label) if not isinstance(data,np.ndarray) else label
        self.sacle_data = None

        #%% sacle attribute
        self.std=None
        self.robust=None
        self.min_max = None
        self.max_abs=None

        #%% train test splilt
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
    #%% scaler
    @property
    def std_scale_data(self):

        self.std = StandardScaler()
        self.std.fit(self.data.reshape(-1, self.data.shape[-1]))
        self.scaled_data=self.std.transform(self.data.reshape(-1, self.data.shape[-1])).reshape(self.data.shape)

    @property
    def robust_scale_data(self):

        self.robust = RobustScaler()
        self.robust.fit(self.data.reshape(-1, self.data.shape[-1]))
        self.sacle_data=self.robust.transform(self.data.reshape(-1, self.data.shape[-1])).reshape(self.data.shape)

    @property
    def min_max_scale_data(self):

        self.min_max = MinMaxScaler()
        self.min_max.fit(self.data.reshape(-1, self.data.shape[-1]))
        self.scaled_data=self.min_max.transform(self.data.reshape(-1, self.data.shape[-1])).reshape(self.data.shape)

    @property
    def max_abs_scale_data(self):

        self.max_abs = MaxAbsScaler()
        self.max_abs.fit(self.data.reshape(-1, self.data.shape[-1]))
        self.scaled_data=self.max_abs.transform(self.data.reshape(-1, self.data.shape[-1])).reshape(self.data.shape)

    #%% train_test_split
    def train_test_split(self,train_size=0.8, random_state=128):

        if self.scaled_data is None:
            raise Exception("Not Scaled data")
        else:
            x_train, x_test, y_train, y_test = train_test_split(self.scaled_data, self.label, train_size=train_size, stratify=self.label, random_state=random_state)
            self.x_train, self.x_test, self.y_train, self.y_test = x_train, x_test, y_train, y_test

#%%
class TFWrapper(DataWrangling):

    def __init__(self,data=None,label=None,model=None):
        super().__init__(data=data,label=label)
        self.model=model
        self.model_data=[]

        self.callback_list={"chk_point":[],"early_stop":[]}
        self.hist=[]

        self.k_fold_train = []
        self.k_fold_val = []

        self.chk_point_save_path = None
        self.chk_point_save_name = None

    def add_early_stopping(self, monitor='val_loss',patience=50):

        self.callback_list['early_stop']=[
            EarlyStopping(  # 성능 향상이 멈추면 훈련을 중지
                monitor=monitor,  # 모델 검증 정확도를 모니터링
                patience=patience  # 1 에포크 보다 더 길게(즉, 2에포크 동안 정확도가 향상되지 않으면 훈련 중지
            )]

    def set_chk_point(self, save_path='./', save_name='model'):
        self.chk_point_save_path = save_path
        self.chk_point_save_name = save_name

        self.callback_list['chk_point'] = [ModelCheckpoint(  # 에포크마다 현재 가중치를 저장
            filepath=f'{self.chk_point_save_path}/{self.chk_point_save_name}.h5',  # 모델 파일 경로
            monitor='val_loss',  # val_loss 가 좋아지지 않으면 모델 파일을 덮어쓰지 않음.
            save_best_only=True,
            mode='auto',
            verbose=True
        )]

    def add_chk_point(self):
        print(f'{self.chk_point_save_path}/{self.chk_point_save_name}.h5')
        try:
            self.callback_list['chk_point'] = [ModelCheckpoint(  # 에포크마다 현재 가중치를 저장
                filepath=f'{self.chk_point_save_path}/{self.chk_point_save_name}.h5',  # 모델 파일 경로
                monitor='val_loss',  # val_loss 가 좋아지지 않으면 모델 파일을 덮어쓰지 않음.
                save_best_only=True,
                mode='auto',
                verbose=True
            )]
        except:
            print("please set chk point hyperparameters")

    def KFold_fit_classification(self,k=5,epochs=100, batch_size=None):

        kf = StratifiedKFold(k, shuffle=True)

        for k,(train_index, validation_index) in enumerate(kf.split(self.x_train,self.y_train)):
            self.model_data.append(self.model())
            x_train = self.x_train[train_index]
            x_val = self.x_train[validation_index]
            y_train = self.y_train[train_index]
            y_val = self.y_train[validation_index]

            self.k_fold_train.append((x_train,y_train))
            self.k_fold_val.append((x_val,y_val))

            print(self.model_data[k].summary())

            if any(self.callback_list.values()):
                if any(self.callback_list['chk_point']):
                    self.set_chk_point(save_path=self.chk_point_save_path,
                                                  save_name=f"{self.chk_point_save_name}_{k}")
                    self.add_chk_point()
                callback=list(chain(*self.callback_list.values()))
                hist=self.model_data[k].fit(x_train,y_train,epochs=epochs, validation_data=(x_val,y_val),batch_size=batch_size, callbacks=callback)
            else:
                hist = self.model_data[k].fit(x_train, y_train, epochs=epochs, validation_data=(x_val, y_val),
                                      batch_size=batch_size)
            self.hist.append(hist)
            self.single_hist_plot(hist)
            if any(self.callback_list['chk_point']):
                self.chk_point_save_name=self.chk_point_save_name[:-2]


    def KFold_fit_regression(self,k=5,epochs=100, batch_size=None):

        kf = KFold(k, shuffle=True)

        for k,(train_index, validation_index) in enumerate(kf.split(self.x_train)):
            self.model_data.append(self.model())

            x_train = self.x_train[train_index]
            x_val = self.x_train[validation_index]
            y_train = self.y_train[train_index]
            y_val = self.y_train[validation_index]

            self.k_fold_train.append((x_train,y_train))
            self.k_fold_val.append((x_val,y_val))
            print(self.model_data[k].summary())

            if any(self.callback_list.values()):
                if any(self.callback_list['chk_point']):
                    self.set_chk_point(save_path=self.chk_point_save_path,save_name=f"{self.chk_point_save_name}_{k}")
                    self.add_chk_point()
                callback=list(chain(*self.callback_list.values()))
                print(callback)
                hist = self.model_data[k].fit(x_train, y_train, epochs=epochs, validation_data=(x_val, y_val),
                                              batch_size=batch_size, callbacks=callback)
            else:
                hist = self.model_data[k].fit(x_train, y_train, epochs=epochs, validation_data=(x_val, y_val),
                                              batch_size=batch_size)
            self.hist.append(hist)
            self.single_hist_plot(hist)

            self.hist.append(hist)
            self.single_hist_plot(hist)

            if any(self.callback_list['chk_point']):
                self.chk_point_save_name = self.chk_point_save_name[:-2]



    def single_hist_plot(self,history=None):

        fig,ax=plt.subplots(1,2)

        ax[0].plot(history.history['loss'])
        ax[0].plot(history.history['val_loss'])
        ax[0].set_title('loss', fontsize=15)
        ax[0].legend(['train', 'val'],loc=1)

        ax[1].plot(history.history['accuracy'])
        ax[1].plot(history.history['val_accuracy'])
        ax[1].legend(['train', 'val'],loc=4)
        ax[1].set_title('acc', fontsize=15)
        plt.show()

    def trained_hist_plot(self, shape=(2,2),save_path=''):

        fig, ax = plt.subplots(*shape, figsize=(7,5))
        ax=list(chain(*ax))

        for hist in self.hist:
            for idx, (title, values) in enumerate(hist.history.items()):

                ax[idx].plot(hist.history[title])
                ax[idx].set_title(title, fontsize=15)


        fig.legend([str(i)+' fold' for i in range(1,len(ax)+2)], loc='lower center',bbox_to_anchor=(0.5, -0.05),fancybox=True, shadow=True, ncol=5)
        fig.tight_layout()
        fig.show()
        if save_path!='':
            fig.savefig(save_path+'/hist.png',bbox_inches = 'tight')

    def save_model(self,path,name):
        for idx, model in enumerate(self.model_data):
            model.save(f'{path}/{name}_{idx}.h5')

#%% Example

if __name__=="__main__":

    from functools import partial
    from tensorflow.keras.models import Sequential
    from tensorflow.keras import layers
    from tensorflow.keras import datasets

    def Ann_seq(Nin, Nh, Nout):
        model=Sequential()
        model.add(layers.Dense(Nh,activation='relu', input_shape=(Nin,)))
        model.add(layers.Dense(Nout,activation='softmax'))
        model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
        return model

    Nin=784
    Nh=100
    number_of_class=10
    Nout=number_of_class

    (x_train, y_train), (x_test, y_test)=datasets.mnist.load_data()

    L,H,W=x_train.shape

    x_train=x_train.reshape(-1,W*H)
    x_test=x_test.reshape(-1,W*H)

    model=partial(Ann_seq,Nin,Nh,Nout)

    train_model=TFWrapper(model=model)

    train_model.x_train=x_train
    train_model.x_test=x_test
    train_model.y_train=y_train
    train_model.y_test=y_test

    train_model.set_chk_point()
    train_model.KFold_fit_classification(epochs=2,batch_size=100)

    train_model.save_model(path='./',name='rest')

    train_model.trained_hist_plot(save_path='./')
