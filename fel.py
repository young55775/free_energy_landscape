import os
import matplotlib.pyplot as plt
import re
import seaborn as sns
import pandas as pd
from plotly import express as ex
import matplotlib
from matplotlib import cm
from sklearn.neighbors import KernelDensity
import numpy as np
import mpl_toolkits.axisartist as axisartist
from math import e
from math import log10 as lg
import scipy.constants as C
from sklearn.mixture import GaussianMixture as gm
from tqdm import tqdm
from Bio.PDB import *
from scipy.spatial import distance_matrix
from sklearn.decomposition import PCA

def parse_xvg(path):
    with open(path,'r') as f:
        wt = f.readlines()
    wt = [n for n in wt if n[0] not in ['@','#','&']]
    wt = [n.strip() for n in wt]
    wt = [n.split(' ')[-1] for n in wt]
    wt = [float(n) for n in wt]
    return wt

pc1 = parse_xvg(r"C:\Users\guozh\Desktop\MD\kras\wt\gtp\pc1.xvg")
pc2 = parse_xvg(r"C:\Users\guozh\Desktop\MD\kras\wt\gtp\pc2.xvg")
min_len = min(len(pc1),len(pc2))
pc1 = pc1[:min_len]
pc2 = pc2[:min_len]
t = list(range(len(pc1)))
'''
scatter plot
'''
# fig = plt.figure(constrained_layout=True,figsize=(6,7))
# gs = fig.add_gridspec(6,7)
# ax1 = axisartist.Subplot(fig, gs[:,1:])
# fig.add_axes(ax1)
# ax1.scatter(pc1[::2],pc2[::2],s=2,alpha=0.5,c=t[::2],cmap='gist_rainbow')
# ax2 = axisartist.Subplot(fig, gs[:,0])
# fig.add_axes(ax2)
# cmap = cm.get_cmap('gist_rainbow')
# norm = matplotlib.colors.Normalize(vmin=1,vmax=max(t)*10)
# plt.colorbar(cm.ScalarMappable(norm=norm,cmap=cmap),cax=ax2,orientation='vertical',label='Time (μs)')
# plt.show()
'''
kde plot for free energy
'''
# dt = pd.DataFrame({'PC1':pc1,'PC2':pc2,'t':t})
# sns.jointplot(data=dt,x='PC1',y='PC2',kind='kde') # free energy landscape
# plt.show()
'''
Scatter plot with annotation
'''
# dt = pd.DataFrame({'PC1':pc1,'PC2':pc2,'t':t})
# fig = ex.scatter(dt, x="PC1", y="PC2", color='t')
# fig.show()
'''
KDE calculation and Free energy estimation
'''
def invert_bolzmann(p,T):
    k = C.k
    E = -k*T*(p/lg(e))*C.N_A/1000
    return E # kJ/mol
def normalize_energy(energy):
    return energy-min(energy)
def remove_outlier(p):
    ave = np.mean(p)
    std = np.std(p)
    up = ave + 3*std
    down = ave - 3*std
    p[p>up] = up
    p[p<down] = down
    return p
data = np.vstack([pc1,pc2]).T
kde = KernelDensity(bandwidth=0.1).fit(data)
p = kde.score_samples(data)
p = remove_outlier(p)
energy = normalize_energy(invert_bolzmann(p,300))
'''
Choose parameter (based on AIC and BIC)
'''
aic = []
bic = []
for cp in tqdm(range(2,11)):
    pred = gm(n_components=cp,init_params='kmeans',max_iter=1000)
    pred.fit(data)
    aic.append(pred.aic(data))
    bic.append(pred.bic(data))
plt.plot(aic,label='AIC')
plt.plot(bic,label='BIC')
plt.legend()
plt.xticks(list(range(9)),list(range(2,11)))
plt.xlabel('N components') #choose number of state
'''
Assign label
'''
pred = gm(n_components=5,init_params='kmeans',max_iter=1000)
label = pred.fit_predict(data)
tau = 50 # time step set to 500 ps
pc1_t = pc1[::tau]
pc2_t = pc2[::tau]
label_t = label[::tau]
t_t = t[::tau]
e_t = energy[::tau]
data_t = pd.DataFrame({'PC1':pc1_t,'PC2':pc2_t,'t':t_t,'e':e_t,'state':label_t})
mat = np.array([[0]*len(label)]*len(list(set(label))))
for i in range(len(label)):
    mat[label[i],i] += 1
sns.heatmap(mat,cmap='Reds')
plt.xticks([0,10000,20000,30000,40000,50000,60000,70000,80000,90000,100000],[0,100,200,300,400,500,600,700,800,900,1000])
plt.xlabel('Time (ns)')
plt.ylabel('State')
'''
MSM transition matrix
'''
def get_transition_matrix(state_seq,tau):
    state_seq = state_seq[::tau]
    state = len(list(set(state_seq)))
    mat = np.array([[0]*state]*state)
    for i in range(1,len(state_seq)):
        current = state_seq[i]
        prev = state_seq[i-1]
        mat[current,prev] += 1
    return (mat.T/np.sum(mat,axis=1)).T

msm = get_transition_matrix(label,1)
sns.heatmap(msm,cmap='Reds',annot=True)
plt.xlabel('To')
plt.ylabel('From')
plt.title('MSM transition matrix')
'''
seek for the stable state
'''
init = np.array([0.1,0.1,0.2,0.3,0.3])
for i in tqdm(range(150)):
    init = np.dot(init.T,msm)
    plt.bar(range(len(init)),init)
    plt.xlabel('state')
    plt.ylabel('probability')
    plt.title(f'Step {i}')
    plt.savefig(rf"C:\Users\guozh\Desktop\MD\kras\msm\{i}.jpg")
    plt.close()
'''
Or you want to manually find PCA states between different simulations (especially for inner coordinate PCA or iPCA)
'''
from Bio import BiopythonWarning
import warnings
warnings.simplefilter('ignore', BiopythonWarning)
def extract_coordinates_from_pdb(pdb_file): # -> N*3 mat
    parser = PDBParser()
    structure = parser.get_structure('structure', pdb_file)
    coordinates = []
    sequence = []
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    coordinates.append(atom.get_coord())
                    sequence.append(residue.resname)
    coordinates = np.array(coordinates)
    return coordinates,sequence


def generate_pdb_from_coordinates(coordinates, output_pdb_file,sequence): # -> PDB file IO
    structure = Structure.Structure('new_structure')
    model = Model.Model(0)
    structure.add(model)
    chain = Chain.Chain('A')
    model.add(chain)
    for i, coord in enumerate(coordinates):
        residue_id = (' ', i + 1, ' ')
        residue = Residue.Residue(residue_id, sequence[i], ' ')
        atom = Atom.Atom(name='CA', coord=coord, bfactor=0.0, occupancy=1.0, altloc=' ', fullname=' CA ',
                         serial_number=i + 1, element='C')
        residue.add(atom)
        chain.add(residue)
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_pdb_file)

def get_distance_matrix(folder):
    files = os.listdir(folder)
    mat_lst = []
    for i in tqdm(files):
       mat,_ = extract_coordinates_from_pdb(os.path.join(folder,i))
       # mat = distance_matrix(mat,mat)
       mat = mat.flatten()
       mat_lst.append(mat)
    return np.vstack(mat_lst)

def pca(X, k):  # k is the components you want
    n_samples, n_features = X.shape
    mean = np.array([np.mean(X[:, i]) for i in range(n_features)])
    norm_X = (X - mean)
    scatter_matrix = np.dot(np.transpose(norm_X), norm_X)
    eig_val, eig_vec = np.linalg.eig(scatter_matrix)
    eig_pairs = [(np.abs(eig_val[i]), eig_vec[:, i]) for i in range(n_features)]
    eig_pairs.sort(reverse=True)
    feature = np.array([ele[1] for ele in eig_pairs[:k]])
    data = np.dot(norm_X, np.transpose(feature))
    return data

gtp = get_distance_matrix(r"C:\Users\guozh\Desktop\MD\kras\wt\gtp\traj\traj")

