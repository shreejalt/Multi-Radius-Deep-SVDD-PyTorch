from base.base_trainer import BaseTrainer
from base.base_dataset import BaseADDataset
from base.base_net import BaseNet
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import logging
import time
import torch
import torch.optim as optim
import numpy as np
from finch import FINCH
from sklearn.manifold import TSNE


class DeepSVDDTrainer(BaseTrainer):

    def __init__(self, objective, R, c, nu: float, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 150,
                 lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
                 n_jobs_dataloader: int = 0, use_multi_radius: bool=False, normal_classes: list = []):
        
        super().__init__(optimizer_name, lr, n_epochs, lr_milestones, batch_size, weight_decay, device,
                         n_jobs_dataloader, normal_classes, use_multi_radius)

        assert objective in ('one-class', 'soft-boundary'), "Objective must be either 'one-class' or 'soft-boundary'."
        self.objective = objective
        
        # Deep SVDD parameters        
        if not self.use_multi_radius:
            self.R = torch.tensor(R, device=self.device)
        else:
            self.R = torch.zeros(size=(len(self.normal_classes), ), device=self.device) # radius R initialized with 0 by default.
        
        self.c = torch.tensor(c, device=self.device) if c is not None else None
        self.nu = nu
       
        # Optimization parameters
        self.warm_up_n_epochs = 10  # number of training epochs for soft-boundary Deep SVDD before radius R gets updated
        
        # Results
        self.train_time = None
        self.test_auc = None
        self.test_time = None
        self.test_scores = None

    def train(self, dataset: BaseADDataset, net: BaseNet):
        logger = logging.getLogger()

        # Set device for network
        net = net.to(self.device)

        # Get train data loader
        train_loader, _ = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                               amsgrad=self.optimizer_name == 'amsgrad')

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)

        # Initialize hypersphere center c (if c not loaded)
        if self.c is None:
            logger.info('Initializing center c...')
            self.c = self.init_center_c(train_loader, net)
            logger.info('Center c initialized.')

        # Training
        logger.info('Starting training...')
        start_time = time.time()
        net.train()
        list_radius = [[] for _ in range(self.c.shape[0])]
        list_loss = []
        for epoch in range(self.n_epochs):

            if epoch in self.lr_milestones:
                logger.info('  LR scheduler: new learning rate is %g' % float(scheduler.get_last_lr()[-1]))

            loss_epoch = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            
            for data in train_loader:
                inputs, _, _, true_targets = data
                inputs = inputs.to(self.device)

                # Zero the network parameter gradients
                optimizer.zero_grad()

                # Update network parameters via backpropagation: forward + backward + optimize
                outputs= net(inputs)
                
                if not self.use_multi_radius:
                    dist = torch.sum((outputs - self.c) ** 2, dim=1)
                    if self.objective == 'soft-boundary':
                      scores = dist - self.R ** 2
                      loss = self.R ** 2 + (1 / self.nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
                    else:
                        loss = torch.mean(dist)
                else:
                    dist = torch.sum((outputs - self.c[true_targets]) ** 2, dim=1)
                    if self.objective == 'soft-boundary':
                        loss = 0
                        for idx in range(self.c.shape[0]):
                            scores = dist[true_targets == idx] - self.R[idx] ** 2
                            loss += torch.sum(self.R[idx] ** 2 + (1 / self.nu) * torch.mean(torch.max(torch.zeros_like(scores), scores)))
                        loss /= self.c.shape[0]
                    else:
                        loss = torch.mean(dist)
                loss.backward()
                optimizer.step()

                # Update hypersphere radius R on mini-batch distances
                if not self.use_multi_radius:
                    if (self.objective == 'soft-boundary') and (epoch >= self.warm_up_n_epochs):
                       self.R.data = torch.tensor(get_radius(dist, self.nu).astype(np.float32), device=self.device)
                else:
                    if (self.objective == 'soft-boundary') and (epoch >= self.warm_up_n_epochs):
                        for idx in range(self.c.shape[0]):
                            quantile = torch.tensor(get_radius(dist[true_targets == idx], self.nu).astype(np.float32), device=self.device)
                            self.R[idx] = quantile
                            list_radius[idx].append(quantile.item())
                list_loss.append(loss.item())
                loss_epoch += loss.item()
                n_batches += 1

            scheduler.step()
            
            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            if self.use_multi_radius:
                radius_str = ' '.join(['R-%d: %.3f' % (idx, self.R[idx].data) for idx in range(self.c.shape[0])])
                logger.info('  Epoch {}/{}\t Time: {:.3f}\t Loss: {:.3f}\t Radius: {:s}'
                        .format(epoch + 1, self.n_epochs, epoch_train_time, loss_epoch / n_batches , radius_str))
            else:
                logger.info('  Epoch {}/{}\t Time: {:.3f}\t Loss: {:.3f}\t Radius: {:.3f}'
                       .format(epoch + 1, self.n_epochs, epoch_train_time, loss_epoch / n_batches , self.R))

        # Plot some graphs
        if self.use_multi_radius:
            plt.clf()
            for idx, ls in enumerate(list_radius):
                plt.plot(ls, label='radius %d' %idx)
            plt.xlabel('No. of epochs')
            plt.ylabel('Radius value')
            plt.legend()
            plt.savefig('radius_plots')
            plt.clf()
            plt.plot(list_loss, c='r')
            plt.xlabel('No.of epochs')
            plt.ylabel('Training loss')
            plt.title('Training loss for Multi-Radius Deep SVDD')
            plt.savefig('loss_plots')
            plt.clf()
            plt.close()
            
        self.train_time = time.time() - start_time
        logger.info('Training time: %.3f' % self.train_time)
        logger.info('Finished training.')

        return net

    def test(self, dataset: BaseADDataset, net: BaseNet):
        logger = logging.getLogger()

        # Set device for network
        net = net.to(self.device)

        # Get test data loader
        _, test_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Testing
        logger.info('Starting testing...')
        start_time = time.time()
        idx_label_score = []
        net.eval()
        with torch.no_grad():
            for data in test_loader:
                inputs, labels, idx, _= data
                inputs = inputs.to(self.device)
                outputs = net(inputs)
                if self.use_multi_radius:
                    dist, idxs = ((outputs - self.c[:, None]) ** 2).sum(dim=2).min(dim=0)
                else:
                    dist = torch.sum((outputs - self.c) ** 2, dim=1)
                    
                if self.objective == 'soft-boundary':
                    if not self.use_multi_radius:
                        scores = dist - self.R ** 2
                    else:
                        scores = dist - self.R[idxs.cpu()] ** 2
                else:
                    scores = dist
                
                # Save triples of (idx, label, score) in a list
                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            labels.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist()))

        self.test_time = time.time() - start_time
        logger.info('Testing time: %.3f' % self.test_time)

        self.test_scores = idx_label_score

        # Compute AUC
        _, labels, scores = zip(*idx_label_score)
        labels = np.array(labels)
        scores = np.array(scores)

        self.test_auc = roc_auc_score(labels, scores)
        logger.info('Test set AUC: {:.2f}%'.format(100. * self.test_auc))

        logger.info('Finished testing.')

    def init_center_c(self, train_loader: DataLoader, net: BaseNet, eps=0.1):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
        
        
        if self.use_multi_radius:
            n_samples = torch.zeros(size=(len(self.normal_classes),))
            c = torch.zeros(size=(len(self.normal_classes), net.rep_dim), device=self.device)
        else:
            n_samples = 0
            c = torch.zeros(size=(net.rep_dim,), device=self.device)

        reps = list()
        labels = list()
        net.eval()
        with torch.no_grad():
            for data in train_loader:
                # get the inputs of the batch
                inputs, _, _, true_targets = data
                inputs = inputs.to(self.device)
                outputs = net(inputs)
                reps.append(outputs.cpu())
                labels.append(true_targets.cpu())
                if self.use_multi_radius:
                    for idx in range(c.shape[0]):
                        c[idx] += torch.sum(outputs[true_targets == idx], dim=0)
                        n_samples[idx] += outputs[true_targets == idx].shape[0]
                else:
                    c += torch.sum(outputs, dim=0)
                    n_samples += outputs.shape[0]
        
        labels = torch.cat(labels, dim=0).numpy()
       
        # Apply finch algorithm
        fin = FINCH()
        labels_cluster = fin.fit_predict(torch.cat(reps, dim=0))
        reps = torch.cat(reps, dim=0).numpy()
        if self.use_multi_radius:
             
            tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(reps)
            for cls in np.unique(labels_cluster):
                x, y = tsne[:, 0][labels_cluster == cls], tsne[:, 1][labels_cluster == cls]
                plt.scatter(x, y, label=cls)
            plt.legend()
            plt.title('Clusters of respective classes by FINCH on MNIST')
            plt.savefig('clusters_finch')

        if self.use_multi_radius:
            for idx in range(c.shape[0]):
                c[idx] /= n_samples[idx]
        else:
            c /= n_samples
        
        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        if self.use_multi_radius:
            for idx in range(c.shape[0]):
                c[idx][(abs(c[idx]) < eps) & (c[idx] < 0)] = -eps
                c[idx][(abs(c[idx]) < eps) & (c[idx] > 0)] = eps
        else:
            c[(abs(c) < eps) & (c < 0)] = -eps
            c[(abs(c) < eps) & (c > 0)] = eps
        
        return c


def get_radius(dist: torch.Tensor, nu: float):
    """Optimally solve for radius R via the (1-nu)-quantile of distances."""
    return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)
