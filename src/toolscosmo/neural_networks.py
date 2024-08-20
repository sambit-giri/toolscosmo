import numpy as np
import pickle, copy
from time import time, sleep
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

try: import tensorflow as tf
except: print('Install Tensorflow to use prob_nn.')

try: 
    import torch
    from torch import nn 
    import torch.optim as optim
except: 
    print('Install PyTorch.')

def moving_average(data, window_size):
    '''
    Function to compute moving average
    '''
    cumsum = np.cumsum(data)
    cumsum[window_size:] = cumsum[window_size:] - cumsum[:-window_size]
    return cumsum[window_size - 1:] / window_size


def save_model(model, filename):
    pickle.dump(model, open(filename,'wb'))
    print('Model parameters are saved as', filename)

def load_model(filename):
    model = pickle.load(open(filename,'rb'))
    return model


def metric_function(y_true, y_pred, metric='r2'):
    from sklearn import metrics

    if type(metric)==str:
        assert metric in ['explained_variance_score', 'max_error', 
                          'mean_squared_error', 'mean_squared_log_error',
                          'mean_absolute_error', 'median_absolute_error',
                          'r2_score', 'r2']

        if metric=='explained_variance_score': metric = metrics.explained_variance_score
        if metric=='max_error': metric = metrics.max_error
        if metric=='mean_squared_error': metric = metrics.mean_squared_error
        if metric=='mean_squared_log_error': metric = metrics.mean_squared_log_error
        if metric=='median_absolute_error': metric = metrics.median_absolute_error
        if metric in ['r2', 'r2_score']: metric = metrics.r2_score

    return metric(y_true, y_pred)

class NNRegressor:
    def __init__(self, model=None, layers=[3,32,64,128,250],
                 dropout_prob=0.5, weight_decay=0,
                 optimizer='Adam', loss_fn='MSE',
                 learning_rate=1e-4,
                 ):
        if model is None:
            layer_modules = []
            for i in range(len(layers) - 1):
                layer_modules.append(nn.Linear(layers[i], layers[i+1]))
                if i < len(layers) - 2:
                    layer_modules.append(nn.ReLU())
            model = nn.Sequential(*layer_modules)

        self.model = model 
        self.X_min = None
        self.X_max = None
        self.y_min = None
        self.y_max = None
        self.n_epochs = None
        self.batch_size = None
        self.lr = None
        self.loss_history = None
        self.dropout_prob = dropout_prob
        self.weight_decay = weight_decay
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.learning_rate = learning_rate

    def numpy_to_tensor(self, data):
        return torch.tensor(data, dtype=torch.float32)
    
    def tensor_to_numpy(self, data):
        return data.numpy()
    
    def normalise(self, data):
        data_min = data.min(axis=0)
        data_max = data.max(axis=0)
        data_normed = (data-data_min)/(data_max-data_min)
        return {'min': data_min, 'max': data_max, 'normed': data_normed}
    
    def apply_dropout(self):
        if self.dropout_prob > 0:
            for layer in self.model.modules():
                if isinstance(layer, nn.Dropout):
                    layer.p = self.dropout_prob

    def apply_weight_decay(self):
        if self.weight_decay > 0:
            for param in self.model.parameters():
                param.data -= self.weight_decay * param.data

    def fit(self, X, y, n_epochs=100, batch_size=10, learning_rate=None):
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        if learning_rate is not None:
            self.learning_rate = learning_rate 

        self.X_min, self.X_max = X.min(axis=0), X.max(axis=0)
        self.y_min, self.y_max = y.min(axis=0), y.max(axis=0)
        X_normed = (X-self.X_min)/(self.X_max-self.X_min)
        y_normed = (y-self.y_min)/(self.y_max-self.y_min)


        X_train, X_test, y_train, y_test = train_test_split(
                X_normed, y_normed, train_size=0.8, shuffle=True)
        X_train = self.numpy_to_tensor(X_train)
        y_train = self.numpy_to_tensor(y_train)
        X_test = self.numpy_to_tensor(X_test)
        y_test = self.numpy_to_tensor(y_test)

        model = self.model
        batch_start = torch.arange(0, len(X_train), batch_size)

        # loss function and optimizer
        if self.loss_fn.lower()=='mse':
            loss_fn = nn.MSELoss()  # mean square error
        else:
            loss_fn = self.loss_fn

        if self.optimizer=='Adam':
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        elif self.optimizer=='RMSprop':
            optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate)
        elif self.optimizer=='Adagrad':
            optimizer = optim.Adagrad(self.model.parameters(), lr=self.learning_rate)
        else:
            optimizer = self.optimizer

        # Hold the best model
        best_mse = np.inf   # init to infinity
        best_weights = None
        history = []
        
        # training loop
        for epoch in range(n_epochs):
            model.train()
            with tqdm(batch_start, unit="batch", mininterval=0) as bar:
                bar.set_description(f"Epoch {epoch}")
                for start in bar:
                    # take a batch
                    X_batch = X_train[start:start+batch_size]
                    y_batch = y_train[start:start+batch_size]
                    # forward pass
                    y_pred = model(X_batch)
                    loss = loss_fn(y_pred, y_batch)
                    # backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    # update weights
                    optimizer.step()
                    # print progress
                    bar.set_postfix(mse=float(loss))
            # evaluate accuracy at end of each epoch
            model.eval()
            y_pred = model(X_test)
            mse = loss_fn(y_pred, y_test)
            mse = float(mse)
            history.append(mse)
            if mse < best_mse:
                best_mse = mse
                best_weights = copy.deepcopy(model.state_dict())
        
        # restore model and return best accuracy
        model.load_state_dict(best_weights)
        self.model = model
        self.loss_history = history

    def predict(self, X):
        X_normed = (X-self.X_min)/(self.X_max-self.X_min)
        # Convert input data to tensor
        X_tensor = self.numpy_to_tensor(X_normed)
        # Set model to evaluation mode
        self.model.eval()
        # Make predictions
        with torch.no_grad():
            y_pred = self.model(X_tensor)
        # Convert predictions to numpy array
        y_pred_numpy = self.tensor_to_numpy(y_pred)
        return y_pred_numpy*(self.y_max-self.y_min)+self.y_min
    
    def append_extra_data(self, extra_data):
        self.extra_data = extra_data

    def save_model(self, filepath):
        save_data = {
                'model_state_dict': self.model.state_dict(),
                'X_min': self.X_min,
                'X_max': self.X_max,
                'y_min': self.y_min,
                'y_max': self.y_max,
                'n_epochs': self.n_epochs,
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'loss_history': self.loss_history,
                'extra_data': self.extra_data,
            }
        torch.save(save_data, filepath)

    def load_model(self, filepath):
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.X_min = checkpoint['X_min']
        self.X_max = checkpoint['X_max']
        self.y_min = checkpoint['y_min']
        self.y_max = checkpoint['y_max']
        self.n_epochs = checkpoint['n_epochs']
        self.batch_size = checkpoint['batch_size']
        self.learning_rate = checkpoint['learning_rate']
        self.loss_history = checkpoint['loss_history']
        self.extra_data = checkpoint['extra_data']

    def PCA_fit(self, X, n_components=32):
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_components)
        pca.fit(X)
        X_transformed = pca.transform(X)
        print(X.shape, X_transformed.shape, pca.score(X))
        return pca 
    
    def PCA_transform_data(self, data, pca):
        out = pca.transform(data)
        return out

    def PCA_inverse_transform_data(self, data, pca):
        out = pca.inverse_transform(data)
        return out 
    

# SIREN layer definition
class SirenLayer(nn.Module):
    def __init__(self, in_features, out_features, w0=30.0, dropout_prob=0.5):
        super(SirenLayer, self).__init__()
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout_prob)
        self.w0 = w0

    def forward(self, x):
        y = self.linear(x)
        y = self.dropout(y)  # Apply dropout
        return torch.sin(self.w0 * y)
