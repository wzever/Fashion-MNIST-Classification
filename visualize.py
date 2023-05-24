import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from train import *
import warnings
import cv2

warnings.filterwarnings('ignore')

feat = np.load('temp/feat.npy', allow_pickle=True).item()
feat_conv, feat_fc, feat_final = feat['conv'], feat['fc'], feat['final']
labels = np.load('temp/labels.npy')
imgs = np.load('temp/imgs.npy')

colors = ['g', 'b', 'lime', 'y', 'olive', 'r', 'cyan', 'orange', 'chocolate', 'purple']
class_name = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
feat_name = ['convolutional', 'linear', 'final']

class my_PCA:
    def __init__(self, feat, out_dim):
        self.feat = feat
        self.out_dim = out_dim
        self.fit_trans = PCA(2).fit_transform
        self.feat_dim = np.size(feat, 1)
        self.large = self.feat_dim > 30000

    def fit(self):
        feat = self.feat
        feat_mean = np.mean(feat, axis=0)
        feat -= feat_mean

        Covar = np.dot(feat.T, feat)
        lambd, v = np.linalg.eig(Covar)
        new_idx = np.argsort(lambd)[::-1]
        trans_mat = -v[:, new_idx]

        return trans_mat
        
    def trans(self, trans_mat, feat):
        out_feat = np.dot(feat, trans_mat)
        return out_feat[:, : self.out_dim]
    
    def execute(self):
        if self.large:
            return self.fit_trans(self.feat, self.out_dim)
        else:
            return self.trans(self.fit(), self.feat)

class my_TSNE:
    def __init__(self, feat, out_dim, max_iter=2000, perplexity=30., pre_dim=50):
        self.feat = feat
        self.out_dim = out_dim
        self.max_iter = max_iter
        self.perplexity = perplexity
        self.pre_dim = pre_dim
        self.initial_momentum = 0.5
        self.final_momentum = 0.8
        self.eta = 500
        self.min_gain = 0.0001

    def Hbeta(self, D, beta=1.0):
        P_mat = torch.exp(-D.clone() * beta)
        sumP = torch.sum(P_mat)
        H = torch.log(sumP) + beta * torch.sum(D * P_mat) / sumP
        P_mat = P_mat / sumP

        return H, P_mat

    def get_p_mat(self, X, tol=1e-5):
        self.t_sne = TSNE(2).fit_transform
        perplexity = self.perplexity
        n, d = X.shape
        sum_X = torch.sum(X*X, 1)
        D = torch.add(torch.add(-2 * torch.mm(X, X.t()), sum_X).t(), sum_X)

        P_mat = torch.zeros(n, n)
        beta = torch.ones(n, 1)
        logU = torch.log(torch.tensor([perplexity]))
        n_list = [i for i in range(n)]

        for i in range(n):
            betamin = None
            betamax = None
            Di = D[i, n_list[0:i]+n_list[i+1:n]]
            H, thisP = self.Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries = 0
            while torch.abs(Hdiff) > tol and tries < 50:
                if Hdiff > 0:
                    betamin = beta[i].clone()
                    if betamax is None:
                        beta[i] = beta[i] * 2.
                    else:
                        beta[i] = (beta[i] + betamax) / 2.
                else:
                    betamax = beta[i].clone()
                    if betamin is None:
                        beta[i] = beta[i] / 2.
                    else:
                        beta[i] = (beta[i] + betamin) / 2.
                (H, thisP) = self.Hbeta(Di, beta[i])
                Hdiff = H - logU
                tries += 1
            P_mat[i, n_list[0: i]+n_list[i+1: n]] = thisP

        return P_mat

    def call_pca(self, X):
        out_dim = self.pre_dim
        return torch.from_numpy(my_PCA(X.numpy(), out_dim).execute())

    def execute(self):
        X = self.feat
        out_dim = self.out_dim
        pre_dim = self.pre_dim
        large = X.shape[1] > 20000

        X = self.call_pca(X)
        n, d = X.shape

        Y = torch.randn(n, out_dim)
        dY = torch.zeros(n, out_dim)
        iY = torch.zeros(n, out_dim)
        gains = torch.ones(n, out_dim)

        P_mat = self.get_p_mat(X, 1e-5)
        P_mat = P_mat + P_mat.t()
        P_mat = P_mat / torch.sum(P_mat)
        P_mat = P_mat * 4.   
        P_mat = torch.max(P_mat, torch.tensor([1e-21]))

        for iter in range(self.max_iter):
            sum_Y = torch.sum(Y*Y, 1)
            num = -2. * torch.mm(Y, Y.t())
            num = 1. / (1. + torch.add(torch.add(num, sum_Y).t(), sum_Y))
            num[range(n), range(n)] = 0.
            Q = num / torch.sum(num)
            Q = torch.max(Q, torch.tensor([1e-12]))

            PQ = P_mat - Q
            for i in range(n):
                dY[i, :] = torch.sum((PQ[:, i] * num[:, i]).repeat(out_dim, 1).t() * (Y[i, :] - Y), 0)

            if iter < 20:
                momentum = self.initial_momentum
            else:
                momentum = self.final_momentum

            gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)).double() + (gains * 0.8) * ((dY > 0.) == (iY > 0.)).double()
            gains[gains < self.min_gain] = self.min_gain
            iY = momentum * iY - self.eta * (gains * dY)
            Y = Y + iY
            Y = Y - torch.mean(Y, 0)
            if large:
                Y = torch.from_numpy(self.t_sne(X))
                break
            if iter == 100:
                P_mat = P_mat / 4.

        return Y.numpy()

def draw_dots():
    '''
    Plot PCA and t-SNE result with colored scatter diagram.
    '''
    for k, feat in enumerate([feat_conv, feat_fc, feat_final]):
        name = feat_name[k]
        print(f'Computing PCA on {name} feature...')
        pca_idx = my_PCA(feat, out_dim=2).execute()
        print('Done!')
        print(f'Computing t-SNE on {name} feature... (This may be time-consuming)')
        tsne_idx = my_TSNE(torch.from_numpy(feat), out_dim=2).execute()
        print('Done!')
        plt.figure()

        for i in range(10):
            plt.scatter(pca_idx[np.where(labels==i), :1], pca_idx[np.where(labels==i), 1:], c=colors[i], label=class_name[i])
        plt.title(f"PCA feature of {(feat_name[k])} layer")
        ncol = 2 if feat_name[k] == 'final' else 1
        plt.legend(ncol=ncol)
        
        plt.show()

        plt.figure()
        for i in range(10):
            plt.scatter(tsne_idx[np.where(labels==i), :1], tsne_idx[np.where(labels==i), 1:], c=colors[i], label=class_name[i])
        plt.title(f"t-SNE feature of {(feat_name[k])} layer")
        plt.legend()
        plt.show()

def draw_tsne_imgs(imgs):
    '''
    Plot t-SNE results with original image exhibition.
    '''
    print('Drawing t-SNE visualization with real images... (This may be time-consuming)')
    for k, features in enumerate([feat_conv, feat_fc, feat_final]):
        tsne = TSNE(2)
        Y = tsne.fit_transform(features)
        fig, ax = plt.subplots()
        fig.set_size_inches(8, 6)
        plt.axis('off')
        imscatter(Y[:, 0], Y[:, 1], imgs, zoom=0.06, ax=ax)
        plt.title(f"t-SNE feature of {(feat_name[k])} layer")
        plt.show()
    print('Done!')

def imscatter(x, y, images, ax=None, zoom=1):
    if ax is None:
        ax = plt.gca()
    x, y = np.atleast_1d(x, y)
    artists = []
    for x0, y0, im in zip(x, y, images):
        im = im.squeeze(0)
        im = cv2.resize(im, (300, 300))
        im_f = OffsetImage(im, zoom=zoom, cmap='gray')
        ab = AnnotationBbox(im_f, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists

def draw_loss_and_acc():
    print('Drawing loss curves and accuracy curves...')
    file_loss = ['temp/train_loss.npy', 'temp/test_loss.npy']
    file_acc = ['temp/train_accs.npy', 'temp/test_accs.npy']
    for p in file_acc:
        acc = np.load(p)
        plt.plot(np.arange(len(acc)), acc, label=p.split('_')[0])
        plt.title("Training and testing accuracy curve")
    plt.legend()
    plt.grid()
    print('Done!')
    plt.show()
    for p in file_loss:
        loss = np.load(p)
        plt.plot(np.arange(len(acc)), loss, label=p.split('_')[0])
        plt.title("Training and testing loss curve")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == '__main__':
    draw_dots()
    draw_loss_and_acc()
    draw_tsne_imgs(imgs)