# Support for math
import numpy as np
import math

# Plotting tools
from matplotlib import pyplot as plt
import matplotlib
from matplotlib import cm
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import warnings
warnings.filterwarnings('ignore')

# File Tools for local
import pandas as pd
import sys

# Random seed for reproducibility
import random
from scipy.spatial import distance
import torch
# from botorch.models.gp_regression import HeteroskedasticSingleTaskGP - This was removed in path #2616.
from botorch.models.gp_regression import SingleTaskGP
from botorch.models import MixedSingleTaskGP
from botorch.acquisition import qExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf
from botorch.optim import optimize_acqf_mixed
from botorch.utils.transforms import normalize, unnormalize

from ipywidgets import interact, FloatSlider


#LHS sampling
#from pyDOE import lhs
from smt.sampling_methods import LHS
import random

# Cluster 
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

class PoolObjectives:
    def __init__(self, x_inputs, y_output, yvar_output, columns, bounds):
        self.x_inputs = x_inputs
        self.y_output = y_output
        self.yvar_output = yvar_output
        self.columns = columns #['time', 'temp', 'sulf', 'anly']
        self.bounds = bounds
        self.x_all = torch.tensor(x_inputs.to_numpy(),dtype=dtype)
        self.y_all = torch.tensor(y_output.to_numpy(),dtype=dtype).reshape(-1,1)
        self.yvar_all = torch.tensor(yvar_output.to_numpy(),dtype=dtype).reshape(-1,1)
        self.ground_truth = self._ground_truth_model()


    def _ground_truth_model(self):
        nx_all = normalize(self.x_all, self.bounds)
        model = SingleTaskGP(nx_all, self.y_all,self.yvar_all)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        return model
    
    def query_gp(self, new_candidates):
        self.ground_truth.eval()
        with torch.no_grad():
            posterior = self.ground_truth.posterior(new_candidates)

        mean = posterior.mean.squeeze().numpy()
        var = posterior.variance.squeeze().numpy()
        return torch.tensor(mean,dtype=dtype).reshape(-1,1), torch.tensor(var, dtype=dtype).reshape(-1,1)
    
    def get_objectives(self, new_candidates):
        new_candidates = new_candidates
        return  self.find_nearest() # return: x_values, y_means, y_vars

    def set_objectives(self, x_inputs, y_all):
        self.x_inputs = x_inputs
        self.y_all = y_all
    

## Build Model A class 
class Models:
    def __init__(self, x_train, y_train, y_train_var,bounds, batch_size ,objective:PoolObjectives,random_id = 0,):
        self.x_train = x_train
        self.y_train = y_train
        self.y_train_var = y_train_var
        self.bounds = bounds
        self.model = self._fit_gp_model()
        self.model_mixed = self._fit_gp_mixed_model()
        self.dtype = dtype
        self.columns = objective.columns #['time', 'temp', 'sulf', 'anly']
        self.batch_size = batch_size
        self.objective = objective
        self.x_all_candidates = objective.x_inputs
        self.y_all_candidates = objective.y_output
        self.yvar_all_candidates = objective.yvar_output
        self.random_id = random_id
        
    
    def _fit_gp_model(self):
        model = SingleTaskGP(self.x_train, self.y_train,self.y_train_var)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        return model
    
    def _fit_gp_mixed_model(self):
        model_mixed = MixedSingleTaskGP(self.x_train, self.y_train,cat_dims=[1],train_Yvar=self.y_train_var)
        mll = ExactMarginalLogLikelihood(model_mixed.likelihood, model_mixed)
        fit_gpytorch_mll(mll)
        return model_mixed

    def gp_evaluate(self, test_x, model_version):
        self.model.eval()
        self.model_mixed.eval()
        if model_version == 'ModelA':
            with torch.no_grad():
                posterior = self.model.posterior(test_x)
        elif model_version == 'ModelB' or model_version == 'ModelC':
            with torch.no_grad():
                posterior = self.model_mixed.posterior(test_x)
        else:
            raise ValueError("Invalid model version. Use 'ModelA', 'ModelB', or 'ModelC'.")

        mean = posterior.mean.squeeze().numpy()
        var = posterior.variance.squeeze().numpy()

        return mean, var

    def optimize_regular(self,batch_size):
        torch.manual_seed(self.random_id)
        np.random.seed(self.random_id)
        random.seed(self.random_id)
        self.best_f = self.y_train.max()
        qEI = qExpectedImprovement(model=self.model, best_f=self.best_f)
        candidate, _ = optimize_acqf(
            acq_function=qEI,
            bounds=torch.tensor([[0., 0., 0. , 0.], [1., 1., 1.,1.]], dtype=self.x_train.dtype),
            q=batch_size,
            num_restarts=15,
            raw_samples=100,
        )
        return unnormalize(candidate, self.bounds)
    
    def optimize_mixed(self, fixed_feature_list):
        torch.manual_seed(self.random_id)
        np.random.seed(self.random_id)
        random.seed(self.random_id)
        self.best_f = self.y_train.max()
        qEI = qExpectedImprovement(model=self.model_mixed , best_f=self.best_f)

        candidate_mixed, _ = optimize_acqf_mixed(
        acq_function=qEI,
        bounds=torch.tensor([[0., 0., 0. , 0.], [1., 1., 1.,1.]], dtype=dtype, device=device),
        q=self.batch_size,
        fixed_features_list=fixed_feature_list,
        num_restarts=10,
        raw_samples=15,
        )
        return unnormalize(candidate_mixed, self.bounds)
    
    def optimize_from_data(self, X_candidates, batch_size):
        torch.manual_seed(self.random_id)
        np.random.seed(self.random_id)
        random.seed(self.random_id)
        
        self.best_f = self.y_train.max()
        X_candidates = normalize(X_candidates, self.bounds)
        qEI = qExpectedImprovement(model=self.model, best_f=self.best_f)
        with torch.no_grad():
            acq_values = qEI(X_candidates.unsqueeze(1))  # shape (N, 1)

        top_indices = torch.topk(acq_values.squeeze(-1), batch_size,largest=True).indices

        return top_indices
    
    def get_top_indices(self):
        return self.top_indices
    
    def regular_candidates(self, x_candidates, batch_size=1):
        self.x_candidates = torch.tensor(x_candidates.to_numpy(), dtype=dtype) if isinstance(x_candidates, pd.DataFrame) else x_candidates
        #candidates_4D = self.optimize_regular(batch_size=1).cpu().numpy()
        candidates_id = self.optimize_from_data(self.x_candidates, batch_size).cpu().numpy()

        # Get the candidates form the pool
        candidates_4D = self.x_all_candidates.to_numpy()[candidates_id]
        y_means = self.y_all_candidates.to_numpy()[candidates_id]
        y_vars = self.yvar_all_candidates.to_numpy()[candidates_id]
        
        # # Remove candidates from the pool
        # self.x_candidates  = self.objective.remove_from_pool(candidates_id)

        return torch.tensor(candidates_4D, dtype=dtype),torch.tensor(y_means, dtype=dtype).reshape(-1, 1),torch.tensor(y_vars, dtype=dtype).reshape(-1, 1), candidates_id
    
    def batch_candidates(self, batch_size):
        candidates_4D = self.optimize_regular(batch_size=batch_size).cpu().numpy()
        #candidates_4D = self.optimize_from_data(x_candidates, self.batch_size).cpu().numpy()
        data = {
            self.columns[0]: candidates_4D [:, 0],
            self.columns[1]: candidates_4D [:, 1],
            self.columns[2]: candidates_4D [:, 2],
            self.columns[3]: candidates_4D [:, 3],
        }
        data_df = pd.DataFrame(data)
        return data_df.round(2)
    

    def ModelA_candidates(self, feature='temp', num_clusters=3):
        candidates_4D = self.optimize_regular(batch_size=self.batch_size).cpu().numpy()  # <-- fix is here
        data = {
            self.columns[0]: candidates_4D [:, 0],
            self.columns[1]: candidates_4D [:, 1],
            self.columns[2]: candidates_4D [:, 2],
            self.columns[3]: candidates_4D [:, 3],
        }
        data_df = pd.DataFrame(data)
        constrained = data_df[[feature]]
        scaler = StandardScaler()
        temp_data_scaled = scaler.fit_transform(constrained)

        kmeans = KMeans(n_clusters=num_clusters, n_init=100, random_state=42)
        data_df['cluster'] = kmeans.fit_predict(temp_data_scaled)
        self.centroids = scaler.inverse_transform(kmeans.cluster_centers_)

        for cluster in range(num_clusters):
            data_df.loc[data_df['cluster'] == cluster, feature] = self.centroids[cluster][0]
        
        return data_df.round(2)
    
    
    def ModelB_candidates(self,feature='temp', num_clusters=3):
        candidates_4D = self.optimize_regular(batch_size=self.batch_size).cpu().numpy()  # <-- fix is here
        
        # Create a DataFrame with the candidates
        data = {
            self.columns[0]: candidates_4D [:, 0],
            self.columns[1]: candidates_4D [:, 1], # Temperature... only three temperatures allowed...
            self.columns[2]: candidates_4D [:, 2],
            self.columns[3]: candidates_4D [:, 3],
        }
        data_df = pd.DataFrame(data)
        constrained = data_df[[feature]]
        scaler = StandardScaler()
        temp_data_scaled = scaler.fit_transform(constrained)

        kmeans = KMeans(n_clusters=num_clusters, n_init=100, random_state=42)
        data_df['cluster'] = kmeans.fit_predict(temp_data_scaled)
        centroids = scaler.inverse_transform(kmeans.cluster_centers_)
        for cluster in range(num_clusters):
            mask = data_df['cluster'] == cluster
            data_df.loc[mask, feature] = centroids[cluster][0]
        
        #print('centroids:', centroids)

        discrete_choices = normalize(torch.tensor(centroids),self.bounds[:,1])
        fixed_features_list = [{1: float(discrete_choices[0])},{1: float(discrete_choices[1])},{1: float(discrete_choices[2])}]

        candidates_mixed = self.optimize_mixed(fixed_feature_list=fixed_features_list)
 
        # Create a DataFrame with the candidates
        data_mix = {
            self.columns[0]: candidates_mixed [:, 0],
            self.columns[1]: candidates_mixed [:, 1],
            self.columns[2]: candidates_mixed [:, 2],
            self.columns[3]: candidates_mixed [:, 3],
        }
        data_mix_df = pd.DataFrame(data_mix)
        
        return data_mix_df.round(2)
    
    def ModelC_candidates(self, feature='temp'):
        candidates_4D = self.optimize_regular(batch_size=3).cpu().numpy()  # <-- fix is here
        # Create a DataFrame with the candidates
        data = {
            self.columns[0]: candidates_4D [:, 0],
            self.columns[1]: candidates_4D [:, 1], # Temperature... only three temperatures allowed...
            self.columns[2]: candidates_4D [:, 2],
            self.columns[3]: candidates_4D [:, 3],
        }
        data_df = pd.DataFrame(data)
        constraints = data_df[[feature]]
        #print("Constraints:", constraints)

        temp_values = data_df[feature].values
        constraints_array = constraints.values  # convert to NumPy

        assigned_temps = np.array([constraints_array[np.abs(constraints_array - val).argmin()] for val in temp_values])
        data_df[feature] = assigned_temps
        discrete_choices = normalize(torch.tensor(constraints.to_numpy()),self.bounds[:,1])
        fixed_features_list = [{1: float(discrete_choices[0])},{1: float(discrete_choices[1])},{1: float(discrete_choices[2])}]

        candidates_mixed = self.optimize_mixed(fixed_feature_list=fixed_features_list)
 
        # Create a DataFrame with the candidates
        data_mix = {
            self.columns[0]: candidates_mixed [:, 0],
            self.columns[1]: candidates_mixed [:, 1],
            self.columns[2]: candidates_mixed [:, 2],
            self.columns[3]: candidates_mixed [:, 3],
        }
        data_mix_df = pd.DataFrame(data_mix)
        
        return data_mix_df.round(2)
    
class Plotting:
    def __init__(self, gp_model, variable_combinations):
        self.models = gp_model
        self.variable_combinations = variable_combinations
        self.dtype = dtype

    def generate_input_data(self, A, B, c, d, combination):
        if combination == ('time', 'sulf', 'anly'):
            return torch.tensor(np.array([[A[i, j], d, B[i, j], c] for i in range(A.shape[0]) for j in range(A.shape[1])]), dtype=self.dtype)
        elif combination == ('time', 'anly', 'sulf'):
            return torch.tensor(np.array([[d, A[i, j], B[i, j], c] for i in range(A.shape[0]) for j in range(A.shape[1])]), dtype=self.dtype)
        elif combination == ('sulf', 'anly', 'time'):
            return torch.tensor(np.array([[A[i, j], c, B[i, j], d] for i in range(A.shape[0]) for j in range(A.shape[1])]), dtype=self.dtype)
    
    def gp_evaluate(self, test_x):
        self.models.eval()
        with torch.no_grad():
            posterior = self.models.posterior(test_x)

        mean = posterior.mean.squeeze().numpy()
        var = posterior.variance.squeeze().numpy()

        return mean, var
    def create_slices(self, c_slices, d_fixed, combination):
        num_points = 20
        a = np.linspace(0, 1, num_points)
        b = np.linspace(0, 1, num_points)
        A, B = np.meshgrid(a, b)

        store_mean = []
        for d in d_fixed:
            mean_values = []
            for c in c_slices:
                input_data = self.generate_input_data(A, B, c, d, combination)
                mean, _ = self.gp_evaluate(input_data)
                mean_values.append(mean.reshape(A.shape))  # Reshape to grid
            store_mean.append(mean_values)

        return A, B, store_mean

    def sliced_plotting(self, model_version, combination, minmax, colormap='Viridis'):
        self.model_version = model_version
        # Create slices for the fixed variable
        c_slices = np.linspace(0, 1, 12)
        d_fixed = [0, 0.25, 0.5, 0.75, 1.0]  # Fixed value for the other variable

        # Create a new figure with subplots for each combination
        fig = make_subplots(rows=1, cols=5, subplot_titles=('temp: 0', 'temp: 0.25','temp: 0.5', 'temp: 0.75','temp: 1.0'),
                        specs=[[{'type': 'surface'}, {'type': 'surface'},{'type': 'surface'}, {'type': 'surface'}, {'type': 'surface'}]])
        global_min = minmax[0].item()
        global_max = minmax[1].item()

        # Create slices and get mean values
        A, B, store_mean = self.create_slices(c_slices, d_fixed, combination)
        # Unpack the mean values for each slice
        mean_vals1, mean_vals2, mean_vals3, mean_vals4, mean_vals5 = store_mean[0], store_mean[1], store_mean[2], store_mean[3], store_mean[4]  

        for i, (c, y_grid1, y_grid2, y_grid3, y_grid4, y_grid5) in enumerate(zip(c_slices, mean_vals1,mean_vals2,mean_vals3,mean_vals4,mean_vals5), start=1):
            fig.add_trace(go.Surface(
                x=A,
                y=B,
                z=c * np.ones_like(A),  # Z-coordinate for slicing
                surfacecolor=y_grid1,  # Use predicted `y` as contour
                colorscale=colormap,
                cmin=global_min,
                cmax=global_max,
                showscale=True if i == 1 else False,  # Show color scale only on the first slice
                #   colorbar_x=0.45,
                opacity=0.7
            ), row=1, col=1)
    
            fig.add_trace(go.Surface(
                x=A,
                y=B,
                z=c * np.ones_like(A),  # Z-coordinate for slicing
                surfacecolor=y_grid2,  # Use predictediance as contour
                cmin=global_min,
                cmax=global_max,
                colorscale=colormap,
                showscale=True if i == 1 else False,  # Show color scale only on the first slice
                #colorbar_x=0.45,
                opacity=0.7
            ), row=1, col=2)

            fig.add_trace(go.Surface(
                x=A,
                y=B,
                z=c * np.ones_like(A),  # Z-coordinate for slicing
                surfacecolor=y_grid3,  # Use predictediance as contour
                cmin=global_min,
                cmax=global_max,
                colorscale=colormap,
                showscale=True if i == 1 else False,  # Show color scale only on the first slice
                opacity=0.7
            ), row=1, col=3)

            fig.add_trace(go.Surface(
                x=A,
                y=B,
                z=c * np.ones_like(A),  # Z-coordinate for slicing
                surfacecolor=y_grid4,  # Use predictediance as contour
                cmin=global_min,
                cmax=global_max,
                colorscale=colormap,
                showscale=True if i == 1 else False,  # Show color scale only on the first slice
                opacity=0.7
            ), row=1, col=4)

            fig.add_trace(go.Surface(
                x=A,
                y=B,
                z=c * np.ones_like(A),  # Z-coordinate for slicing
                surfacecolor=y_grid5,  # Use predictediance as contour
                cmin=global_min,
                cmax=global_max,
                colorscale=colormap,
                showscale=True if i == 1 else False,  # Show color scale only on the first slice
                opacity=0.7
            ), row=1, col=5)
            
        fig.update_layout(
            height=400,
            width=1300,
            margin=dict(l=50, r=50, b=50, t=50),
            scene=dict(
                xaxis_title=combination[0],
                yaxis_title=combination[1],
                zaxis_title=combination[2]
            ),
            scene2=dict(
                xaxis_title=combination[0],
                yaxis_title=combination[1],
                zaxis_title=combination[2]
            ),
            scene3=dict(
                xaxis_title=combination[0],
                yaxis_title=combination[1],
                zaxis_title=combination[2]
            ),
            scene4=dict(
                xaxis_title=combination[0],
                yaxis_title=combination[1],
                zaxis_title=combination[2]
            ),
            scene5=dict(
                xaxis_title=combination[0],
                yaxis_title=combination[1],
                zaxis_title=combination[2]
            )
        )

        fig.show()




