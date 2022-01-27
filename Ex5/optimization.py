import numpy as np
import random


class Regressor:
    def __init__(self) -> None:
        self.X, self.y = self.generate_dataset(n_samples=200, n_features=1)
        n, d = self.X.shape
        self.w = np.zeros((d, 1))
        self.preds = [(0, self.predict(self.X), self.compute_loss(), self.w[0][0])]

    def generate_dataset(self, n_samples, n_features):
        """
        Generates a regression dataset
        Returns:
            X: a numpy.ndarray of shape (100, 2) containing the dataset
            y: a numpy.ndarray of shape (100, 1) containing the labels
        """
        from sklearn.datasets import make_regression

        np.random.seed(42)
        X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=30)
        y = y.reshape(n_samples, 1)
        return X, y

    def linear_regression(self):
        """
        Performs linear regression on a dataset
        Returns:
            y: a numpy.ndarray of shape (n, 1) containing the predictions
        """
        y = np.dot(self.X, self.w)
        return y

    def predict(self, X, w=None):
        """
        Predicts the labels for a given dataset
        X: a numpy.ndarray of shape (n, d) containing the dataset
        Returns:
            y: a numpy.ndarray of shape (n,) containing the predictions
        """
        if w is None: w=self.w
        y = np.dot(X, w)
        return y

    def compute_loss(self, predictions=None):
        """
        Computes the MSE loss of a prediction
        Returns:
            loss: the loss of the prediction
        """
        if predictions is None: predictions = self.linear_regression()
        loss = np.mean((predictions - self.y) ** 2)
        return loss

    def compute_gradient(self):
        """
        Computes the gradient of the MSE loss
        Returns:
            grad: the gradient of the loss with respect to w
        """
        predictions = self.linear_regression()
        dif = (predictions - self.y)
        grad = 2 * np.dot(self.X.T, dif)
        return grad

    def fit(self, optimizer='sgd_optimizer', n_iters=500,
            render_animation=False, gradiant_image=False):
        """
        Trains the model
        optimizer: the optimization algorithm to use
        X: a numpy.ndarray of shape (n, d) containing the dataset
        y: a numpy.ndarray of shape (n, 1) containing the labels
        n_iters: the number of iterations to train for
        """
        figs = []

        for i in range(1, n_iters + 1):
            self.w=eval('self.%s()'%optimizer)

            z=max(int(i<=3)*1, int(3<i<=10)*10, int(10<i<=25)*25, int(25<i)*50)//1
            if i % z == 0:
                self.preds.append((i+1, self.predict(self.X), self.compute_loss(), self.w[0][0]))
            if i % 10 == 0:
                print("Iteration: ", i)
                print("Loss: ", self.compute_loss())

            if render_animation:
                import matplotlib.pyplot as plt
                from moviepy.video.io.bindings import mplfig_to_npimage

                fig = plt.figure()
                plt.scatter(self.X, self.y, color='red')
                plt.plot(self.X, self.predict(self.X), color='blue')
                plt.xlim(self.X.min(), self.X.max())
                plt.ylim(self.y.min(), self.y.max())
                plt.title(f'Optimizer:{optimizer}\nIteration: {i}')
                plt.close()
                figs.append(mplfig_to_npimage(fig))

        if gradiant_image:
            self.plot_gradient(optimizer)

        if render_animation and len(figs) > 0:
            from moviepy.editor import ImageSequenceClip
            clip = ImageSequenceClip(figs, fps=5)
            clip.write_gif(f'{optimizer}_animation.gif', fps=5)

    def gradient_descent(self, alpha=0.001):
        """
        Performs gradient descent to optimize the weights
        alpha: the learning rate
        Returns:
            w: a numpy.ndarray of shape (d, 1) containing the optimized weights
        """
        w = self.w.copy()
        dw = -alpha * self.compute_gradient()

        return w+dw

    def sgd_optimizer(self, alpha=0.01, minibatch=1):
        """
        Performs stochastic gradient descent to optimize the weights
        alpha: the learning rate
        Returns:
            w: a numpy.ndarray of shape (d, 1) containing the optimized weights
        """
        w = self.w.copy()

        idx = np.random.choice(np.arange(len(self.X)), minibatch, replace=False)
        x, y = self.X[idx], self.y[idx]
        predictions = self.predict(x)
        dif = (predictions - y)
        grad = 2 * np.dot(x.T, dif)
        dw = -alpha * grad

        return w+dw

    def sgd_momentum(self, alpha=0.01, beta=0.9, minibatch=1):
        """
        Performs SGD with momentum to optimize the weights
        alpha: the learning rate
        momentum: the momentum
        Returns:
            w: a numpy.ndarray of shape (d, 1) containing the optimized weights
        """
        w = self.w.copy()

        idx = np.random.choice(np.arange(len(self.X)), minibatch, replace=False)
        x, y = self.X[idx], self.y[idx]
        predictions = self.predict(x)
        dif = (predictions - y)
        grad = 2 * np.dot(x.T, dif)
        n_grad = -alpha * grad

        if not hasattr(self, 'v'): self.v=0
        self.v=-beta*self.v+n_grad

        return w+self.v

    def adagrad_optimizer(self, alpha=10, minibatch=10, epsilon=1e-6):
        """
        Performs Adagrad optimization to optimize the weights
        alpha: the learning rate
        epsilon: a small number to avoid division by zero
        Returns:
            w: a numpy.ndarray of shape (d, 1) containing the optimized weights
            ...
        """

        w = self.w.copy()

        idx = np.random.choice(np.arange(len(self.X)), minibatch, replace=False)
        x, y = self.X[idx], self.y[idx]
        predictions = self.predict(x)
        dif = (predictions - y)
        grad = 2 * np.dot(x.T, dif)

        # if not hasattr(self, 'g'): self.g=np.multiply.outer(grad, grad.T)*0
        # self.g += np.multiply.outer(grad, grad.T)
        # evectors, evalues=np.linalg.eig(self.g)
        # sqrt_matrix = evectors @ np.diag(np.sqrt(evalues)) @ np.linalg.inv(evectors)
        # adjusted_alpha = alpha * (sqrt_matrix+epsilon*np.identity(len(self.g))).inverse()

        if not hasattr(self, 'g'): self.g=grad*0
        self.g += grad**2
        adjusted_alpha = alpha / (np.sqrt(self.g)+epsilon)

        dw = -adjusted_alpha * grad

        return w+dw

    def rmsprop_optimizer(self, alpha=16, minibatch=10, beta=0.9, epsilon=1e-6):
        """
        Performs RMSProp optimization to optimize the weights
        g: sum of squared gradients
        alpha: the learning rate
        beta: the momentum
        epsilon: a small number to avoid division by zero
        Returns:
            w: a numpy.ndarray of shape (d, 1) containing the optimized weights
            ...
        """
        w = self.w.copy()

        idx = np.random.choice(np.arange(len(self.X)), minibatch, replace=False)
        x, y = self.X[idx], self.y[idx]
        predictions = self.predict(x)
        dif = (predictions - y)
        grad = 2 * np.dot(x.T, dif)

        if not hasattr(self, 'g'): self.g=grad*0
        self.g += beta*self.g + (1-beta)*grad**2
        adjusted_alpha = alpha / (np.sqrt(self.g)+epsilon)

        dw = -adjusted_alpha * grad

        return w+dw

    def adam_optimizer_fedup(self, alpha=0.5, minibatch=100, beta1=0.36, beta2=0.99, epsilon=1e-6):
        """
        Performs Adam optimization to optimize the weights
        m: the first moment vector
        v: the second moment vector
        alpha: the learning rate
        beta1: the first momentum
        beta2: the second momentum
        epsilon: a small number to avoid division by zero
        Returns:
            w: a numpy.ndarray of shape (d, 1) containing the optimized weights
            ...
        """
        w = self.w.copy()

        idx = np.random.choice(np.arange(len(self.X)), minibatch, replace=False)
        x, y = self.X[idx], self.y[idx]
        predictions = self.predict(x)
        dif = (predictions - y)
        grad = 2 * np.dot(x.T, dif)

        if not hasattr(self, 'v'): self.v=grad*0; self.beta1_=1
        self.v += beta1*self.v + (1-beta1)*grad
        self.beta1_*=beta1
        if not hasattr(self, 'g'): self.g=grad*0; self.beta2_=1
        self.g += beta2*self.g + (1-beta2)*grad**2
        self.beta2_*=beta2

        v_=self.v/(1-self.beta1_)
        g_=self.g/(1-self.beta2_)
        adjusted_alpha = alpha / (np.sqrt(g_)+epsilon)
        dw = -adjusted_alpha * v_

        return w+dw

    def plot_gradient(self, optimizer):
        """
        Plots the gradient descent path for the loss function
        Useful links:
        -   http://www.adeveloperdiary.com/data-science/how-to-visualize-gradient-descent-using-contour-plot-in-python/
        -   https://www.youtube.com/watch?v=zvp8K4iX2Cs&list=LL&index=2
        """
        import matplotlib.pyplot as plt
        x, y=self.X,self.y
        ppp=list(zip(*self.preds))

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20,10))
        ax[0].scatter(x, y, marker='x', s=40, color='k')

        w_grid=[]
        for i in np.linspace(0,100,1000):
            w_grid.append(self.w.copy())
            w_grid[-1][0][0]=i
        loss_grid=[self.compute_loss(self.predict(self.X, w)) for w in w_grid]
        ax[1].plot(np.linspace(0,100,1000), loss_grid, 'k')

        i=0
        colors=['r', 'b', 'g', 'm', 'c', 'orange']
        for j, (k, pred, loss, w) in enumerate(self.preds):
            i%=len(colors); c=colors[i]; i+=1
            ax[0].plot(x, pred, color=c, lw=2, label=r'$\theta_{:.3f}$ = {:.3f}$'.format(k,w))

            if j==0: continue
            ax[1].annotate('', xy=(w, loss), xytext=(self.preds[j-1][3], self.preds[j-1][2]),
                arrowprops={'arrowstyle': '->', 'color': 'r', 'lw': 1}, va='center', ha='center')

        # Labels, titles and a legend.
        ax[1].scatter(ppp[3], ppp[2], s=40, lw=0)
        ax[1].set_xlabel(r'$\theta_1$')
        ax[1].set_ylabel(r'$J(\theta_1)$')
        ax[1].set_title('Cost function')
        ax[0].set_xlabel(r'$x$')
        ax[0].set_ylabel(r'$y$')
        ax[0].set_title('Data and fit')
        ax[0].legend(loc='upper left', fontsize='small')

        plt.tight_layout()
        plt.savefig('%s.png'%optimizer)
        plt.show()


if __name__ == "__main__":
    a=Regressor()
    a.fit()